#!/usr/bin/env python3
"""Prepare V6-C first-continuous Rj brackets from the audited V6-B denominator.

This stage does not label a grid endpoint as a 0.1%-accurate physical
boundary.  It produces the only admissible brackets for the following direct
model log-space bisection stage.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "v6_hts_temperature_tolerance_arc16pancake_nuc600"
INPUT = OUT / "v6b_feasible_designs.csv"
INPUT_AUDIT = OUT / "v6b_status_feasibility_audit.json"
ARCH = OUT / "v6c_s2_architecture_optima_grid.csv"
BRACKETS = OUT / "v6c_s2_joint_tolerance_grid_brackets.csv"
MONOTONICITY = OUT / "v6c_s2_monotonicity_audit.csv"
AUDIT = OUT / "v6c_s2_grid_bracket_audit.json"
CHUNK = 200_000
TEMPERATURES = (4.2, 10.0, 20.0)
N_VALUES = tuple(range(1, 201))
CRITERIA = ("feas", "temp_1pct", "temp_5pct", "temp_10pct", "global_1pct", "global_5pct", "global_10pct")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def trace_first_interval(r: np.ndarray, passed: np.ndarray, criterion: str) -> dict:
    if not bool(passed[0]):
        return {
            f"Rj_max_{criterion}_grid_lower_nOhm": np.nan,
            f"Rj_bracket_{criterion}_upper_nOhm": np.nan,
            f"low_reference_infeasible_{criterion}": True,
            f"right_censored_{criterion}": False,
            f"direct_bisection_required_{criterion}": False,
            f"first_interval_reentry_{criterion}": bool(np.any((~passed[:-1]) & passed[1:])),
            f"binding_condition_{criterion}": "availability",
        }
    failed = np.flatnonzero(~passed)
    reentry = bool(np.any((~passed[:-1]) & passed[1:]))
    if failed.size == 0:
        return {
            f"Rj_max_{criterion}_grid_lower_nOhm": float(r[-1]),
            f"Rj_bracket_{criterion}_upper_nOhm": np.nan,
            f"low_reference_infeasible_{criterion}": False,
            f"right_censored_{criterion}": True,
            f"direct_bisection_required_{criterion}": False,
            f"first_interval_reentry_{criterion}": reentry,
            f"binding_condition_{criterion}": "right_censored_at_100_nOhm",
        }
    ix = int(failed[0])
    if criterion == "feas" or not bool(passed[ix - 1]):
        binding = "availability"
    elif criterion.startswith("temp_"):
        binding = "temperature_relative_LCOE"
    else:
        binding = "global_relative_LCOE"
    return {
        f"Rj_max_{criterion}_grid_lower_nOhm": float(r[ix - 1]),
        f"Rj_bracket_{criterion}_upper_nOhm": float(r[ix]),
        f"low_reference_infeasible_{criterion}": False,
        f"right_censored_{criterion}": False,
        f"direct_bisection_required_{criterion}": True,
        f"first_interval_reentry_{criterion}": reentry,
        f"binding_condition_{criterion}": binding,
    }


def main() -> None:
    if not INPUT.exists() or not INPUT_AUDIT.exists():
        raise FileNotFoundError("V6-B feasible-design table or PASS audit is missing")
    if any(path.exists() for path in (ARCH, BRACKETS, MONOTONICITY, AUDIT)):
        raise FileExistsError("refusing to overwrite V6-C grid-bracket artifacts")
    upstream = json.loads(INPUT_AUDIT.read_text(encoding="utf-8"))
    source_sha = sha256(INPUT)
    if upstream.get("status") != "PASS" or upstream.get("outputs", {}).get("feasible_designs", {}).get("sha256") != source_sha:
        raise AssertionError("V6-B feasible-design table does not match its PASS audit")

    started = datetime.now(timezone.utc)
    best: dict[tuple[float, int, float], dict] = {}
    rows_s2 = 0
    columns = ["scenario", "Top_K", "coolant", "Npw", "rho_turn_uOhm_cm2", "R_joint_nOhm", "AF_ref", "r_cryo_re_fraction", "net_energy_annual_MWh", "lcoe_anchor_USD_per_MWh"]
    for chunk in pd.read_csv(INPUT, usecols=columns, chunksize=CHUNK, low_memory=False):
        data = chunk.loc[
            chunk["scenario"].eq("S2") & chunk["coolant"].eq("He") & chunk["Top_K"].isin(TEMPERATURES)
        ].copy()
        rows_s2 += len(data)
        if data.empty:
            continue
        data["lcoe_anchor_USD_per_MWh"] = pd.to_numeric(data["lcoe_anchor_USD_per_MWh"], errors="raise")
        winners = data.loc[data.groupby(["Top_K", "Npw", "R_joint_nOhm"], sort=False)["lcoe_anchor_USD_per_MWh"].idxmin()]
        for row in winners.to_dict("records"):
            key = (float(row["Top_K"]), int(row["Npw"]), float(row["R_joint_nOhm"]))
            prior = best.get(key)
            if prior is None or float(row["lcoe_anchor_USD_per_MWh"]) < float(prior["lcoe_anchor_USD_per_MWh"]):
                best[key] = row

    best_df = pd.DataFrame(best.values())
    if best_df.empty:
        raise AssertionError("no S2 He feasible architecture optima found")
    r_values = np.sort(best_df["R_joint_nOhm"].unique().astype(float))
    if len(r_values) != 121 or not np.isclose(r_values[0], 1.0) or not np.isclose(r_values[-1], 100.0):
        raise AssertionError(f"unexpected V6 Rj grid: {len(r_values)} points from {r_values[0]} to {r_values[-1]}")
    expected_architectures = len(TEMPERATURES) * len(N_VALUES) * len(r_values)
    full = pd.DataFrame(product(TEMPERATURES, N_VALUES, r_values), columns=["Top_K", "Npw", "R_joint_nOhm"])
    arch = full.merge(best_df, on=["Top_K", "Npw", "R_joint_nOhm"], how="left", validate="one_to_one", indicator=True)
    arch["architecture_feasible"] = arch["_merge"].eq("both")
    arch.drop(columns="_merge", inplace=True)
    references = arch.loc[arch["architecture_feasible"]].groupby("Top_K", sort=True)["lcoe_anchor_USD_per_MWh"].min().to_dict()
    if set(references) != set(TEMPERATURES):
        raise AssertionError(f"temperature reference minima missing: {references}")
    global_reference = float(min(references.values()))
    for top, ref in references.items():
        mask = arch["Top_K"].eq(top) & arch["architecture_feasible"]
        arch.loc[mask, "delta_T_pct"] = 100.0 * (arch.loc[mask, "lcoe_anchor_USD_per_MWh"] - ref) / ref
    mask = arch["architecture_feasible"]
    arch.loc[mask, "delta_global_pct"] = 100.0 * (arch.loc[mask, "lcoe_anchor_USD_per_MWh"] - global_reference) / global_reference
    arch.rename(columns={"rho_turn_uOhm_cm2": "best_rho_turn_uOhm_cm2"}, inplace=True)
    arch.to_csv(ARCH, index=False, encoding="utf-8", float_format="%.17g")

    boundary_rows: list[dict] = []
    monotonicity_rows: list[dict] = []
    for top, n in product(TEMPERATURES, N_VALUES):
        frame = arch.loc[(arch["Top_K"].eq(top)) & (arch["Npw"].eq(n))].sort_values("R_joint_nOhm")
        r = frame["R_joint_nOhm"].to_numpy(dtype=float)
        feasibility = frame["architecture_feasible"].to_numpy(dtype=bool)
        row = {"scenario": "S2", "Top_K": float(top), "coolant": "He", "Npw": int(n), "R_ref_nOhm": 1.0,
               "L_star_temperature_USD_per_MWh": float(references[top]), "L_star_global_USD_per_MWh": global_reference}
        conditions = {"feas": feasibility}
        for pct in (1, 5, 10):
            conditions[f"temp_{pct}pct"] = feasibility & (frame["delta_T_pct"].to_numpy(dtype=float) <= pct)
            conditions[f"global_{pct}pct"] = feasibility & (frame["delta_global_pct"].to_numpy(dtype=float) <= pct)
        for criterion, passed in conditions.items():
            row.update(trace_first_interval(r, passed, criterion))
            transitions_down = int(np.sum(passed[:-1] & ~passed[1:]))
            transitions_up = int(np.sum(~passed[:-1] & passed[1:]))
            monotonicity_rows.append({"scenario": "S2", "Top_K": float(top), "coolant": "He", "Npw": int(n), "criterion": criterion,
                                      "grid_points": len(r), "reference_pass": bool(passed[0]), "pass_count": int(passed.sum()),
                                      "true_to_false_transitions": transitions_down, "false_to_true_transitions": transitions_up,
                                      "reentry_after_first_failure": bool(transitions_up > 0),
                                      "first_interval_only": True})
        boundary_rows.append(row)
    boundaries = pd.DataFrame(boundary_rows)
    boundaries.to_csv(BRACKETS, index=False, encoding="utf-8", float_format="%.17g")
    monotonicity = pd.DataFrame(monotonicity_rows)
    monotonicity.to_csv(MONOTONICITY, index=False, encoding="utf-8")

    if len(arch) != expected_architectures or len(boundaries) != 600 or len(monotonicity) != 4_200:
        raise AssertionError("unexpected V6-C architecture/bracket output sizes")
    audit = {
        "experiment_id": "V6-C-grid-bracket-preparation",
        "status": "PASS_GRID_BRACKETS_NOT_FINAL_BOUNDARIES",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(), "device": "arc_16pancake_nuc600", "scenario": "S2", "coolant": "He",
        "source_dataset": {"path": str(INPUT.relative_to(ROOT)), "sha256": source_sha, "strict_feasible_rows": int(upstream["outputs"]["feasible_designs"]["rows"]), "S2_He_rows_read": rows_s2},
        "definition": {"rho_turn": "minimum frozen-B1-anchor LCOE among V6-B strict-feasible rho samples", "interval": "first continuous true interval from R_ref=1 nOhm only", "temperature_interpolation": "none", "grid": "121-point stored V6 Rj grid", "final_boundary": "requires direct-model log-space bisection to <=0.1% relative error"},
        "lcoe_references_USD_per_MWh": {str(k): float(v) for k, v in references.items()}, "global_reference_USD_per_MWh": global_reference,
        "outputs": {"architecture_optima": {"path": str(ARCH.relative_to(ROOT)), "rows": len(arch), "sha256": sha256(ARCH)}, "grid_brackets": {"path": str(BRACKETS.relative_to(ROOT)), "rows": len(boundaries), "sha256": sha256(BRACKETS)}, "monotonicity": {"path": str(MONOTONICITY.relative_to(ROOT)), "rows": len(monotonicity), "sha256": sha256(MONOTONICITY)}},
        "started_utc": started.isoformat(),
    }
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
