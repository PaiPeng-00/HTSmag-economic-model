#!/usr/bin/env python3
"""Prepare V6.2-C S2-He first-continuous tolerance brackets.

Only the V6.2-B1 strict-feasible table is admissible here.  This script does
not estimate a boundary: each non-censored bracket is passed to the separate
direct-model log-space bisection stage.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds


ROOT = Path(__file__).resolve().parents[3]
V6_BASE = Path(os.environ["V6_RUN_BASE"]) if os.environ.get("V6_RUN_BASE") else ROOT / "v6_2_arc_actual_inductance"
STAGE_B1 = V6_BASE / "stage_B1"
OUT = V6_BASE / "stage_C"
INPUT = STAGE_B1 / "v6_2_b_strict_feasible_designs.csv"
UPSTREAM_AUDIT = STAGE_B1 / "v6_2_b1_and_strict_feasibility_audit.json"
PARQUET_ROOT = V6_BASE / "stage_B1_parquet_v1" / "v6_2_b1_strict_feasible_parquet"
PARQUET_AUDIT = V6_BASE / "stage_B1_parquet_v1" / "v6_2_b1_strict_parquet_audit.json"
ARCH = OUT / "v6_2_c_s2_architecture_optima_grid.csv"
BRACKETS = OUT / "v6_2_c_s2_joint_tolerance_grid_brackets.csv"
MONOTONICITY = OUT / "v6_2_c_s2_monotonicity_audit.csv"
AUDIT = OUT / "v6_2_c_grid_bracket_audit.json"
DEVICE = "arc_16pancake_nuc600_v6_2"
MATRIX_SHA256 = "97242795c8f75319c8f5ea726a1fedd86999e8a66e53ddeb1255e6fb0c659ae5"
TEMPERATURES = (4.2, 10.0, 20.0)
N_VALUES = tuple(range(1, 201))
CRITERIA = ("feas", "temp_1pct", "temp_5pct", "temp_10pct", "global_1pct", "global_5pct", "global_10pct")
CHUNK = 200_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_path(path: Path) -> str:
    """Use repository-relative paths when possible, otherwise preserve the external run root."""
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def trace_first_interval(r: np.ndarray, passed: np.ndarray, criterion: str) -> dict:
    if not bool(passed[0]):
        return {f"Rj_max_{criterion}_grid_lower_nOhm": np.nan, f"Rj_bracket_{criterion}_upper_nOhm": np.nan,
                f"low_reference_infeasible_{criterion}": True, f"right_censored_{criterion}": False,
                f"direct_bisection_required_{criterion}": False, f"first_interval_reentry_{criterion}": bool(np.any((~passed[:-1]) & passed[1:])),
                f"binding_condition_{criterion}": "low_reference_infeasible"}
    failed = np.flatnonzero(~passed)
    reentry = bool(np.any((~passed[:-1]) & passed[1:]))
    if failed.size == 0:
        return {f"Rj_max_{criterion}_grid_lower_nOhm": float(r[-1]), f"Rj_bracket_{criterion}_upper_nOhm": np.nan,
                f"low_reference_infeasible_{criterion}": False, f"right_censored_{criterion}": True,
                f"direct_bisection_required_{criterion}": False, f"first_interval_reentry_{criterion}": reentry,
                f"binding_condition_{criterion}": "right_censored_at_100_nOhm"}
    ix = int(failed[0])
    if criterion == "feas":
        binding = "strict_feasibility"
    elif criterion.startswith("temp_"):
        binding = "temperature_relative_anchor_LCOE"
    else:
        binding = "global_relative_anchor_LCOE"
    return {f"Rj_max_{criterion}_grid_lower_nOhm": float(r[ix - 1]), f"Rj_bracket_{criterion}_upper_nOhm": float(r[ix]),
            f"low_reference_infeasible_{criterion}": False, f"right_censored_{criterion}": False,
            f"direct_bisection_required_{criterion}": True, f"first_interval_reentry_{criterion}": reentry,
            f"binding_condition_{criterion}": binding}


def main() -> None:
    if any(p.exists() for p in (ARCH, BRACKETS, MONOTONICITY, AUDIT)):
        raise FileExistsError("refusing to overwrite existing V6.2-C bracket artifacts")
    use_parquet = PARQUET_ROOT.exists() and PARQUET_AUDIT.exists()
    audit_file = PARQUET_AUDIT if use_parquet else UPSTREAM_AUDIT
    upstream = json.loads(audit_file.read_text(encoding="utf-8"))
    source_sha = sha256(audit_file) if use_parquet else sha256(INPUT)
    if use_parquet:
        if upstream.get("status") != "PASS" or upstream.get("storage", {}).get("strict_root") != PARQUET_ROOT.name:
            raise AssertionError("V6.2 strict-feasible Parquet source is not an audited PASS artifact")
    else:
        strict_meta = upstream.get("outputs", {}).get("strict_feasible", {})
        if upstream.get("status") != "PASS" or strict_meta.get("sha256") != source_sha:
            raise AssertionError("V6.2 strict-feasible source is not the audited PASS artifact")
    if upstream.get("device") != DEVICE or upstream.get("matrix_sha256") != MATRIX_SHA256:
        raise AssertionError("V6.2 device or actual-geometry matrix provenance mismatch")
    OUT.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    best: dict[tuple[float, int, float], dict] = {}
    rows_read = 0
    columns = ["scenario", "Top_K", "coolant", "Npw", "rho_turn_uOhm_cm2", "R_joint_nOhm", "Aplant", "CF_gross", "r_cryo_re_fraction", "E_net_year_MWh", "lcoe_anchor_USD_per_MWh", "TF_system_matrix_sha256"]
    if use_parquet:
        dataset = ds.dataset(str(PARQUET_ROOT), format="parquet", partitioning="hive")
        batches = dataset.scanner(
            columns=columns,
            filter=(ds.field("scenario") == "S2") & (ds.field("coolant") == "He"),
            batch_size=CHUNK,
        ).to_batches()
        chunks = (batch.to_pandas() for batch in batches)
    else:
        chunks = pd.read_csv(INPUT, usecols=columns, chunksize=CHUNK, low_memory=False)
    for chunk in chunks:
        # Hive partition discovery exposes Top_K as text; normalize before
        # applying the temperature predicate so CSV and Parquet paths agree.
        chunk["Top_K"] = pd.to_numeric(chunk["Top_K"], errors="raise")
        data = chunk.loc[chunk["scenario"].eq("S2") & chunk["coolant"].eq("He") & chunk["Top_K"].isin(TEMPERATURES)].copy()
        rows_read += len(data)
        if data.empty:
            continue
        if not data["TF_system_matrix_sha256"].eq(MATRIX_SHA256).all():
            raise AssertionError("strict table contains a non-V6.2 circuit-matrix row")
        winners = data.loc[data.groupby(["Top_K", "Npw", "R_joint_nOhm"], sort=False)["lcoe_anchor_USD_per_MWh"].idxmin()]
        for row in winners.to_dict("records"):
            key = (float(row["Top_K"]), int(row["Npw"]), float(row["R_joint_nOhm"]))
            prior = best.get(key)
            if prior is None or float(row["lcoe_anchor_USD_per_MWh"]) < float(prior["lcoe_anchor_USD_per_MWh"]):
                best[key] = row
    best_df = pd.DataFrame(best.values())
    r_values = np.sort(best_df["R_joint_nOhm"].unique().astype(float))
    if len(r_values) != 121 or not np.isclose(r_values[0], 1.0) or not np.isclose(r_values[-1], 100.0):
        raise AssertionError(f"unexpected V6.2 Rj grid: {len(r_values)} points [{r_values[0]}, {r_values[-1]}]")
    full = pd.DataFrame(product(TEMPERATURES, N_VALUES, r_values), columns=["Top_K", "Npw", "R_joint_nOhm"])
    arch = full.merge(best_df, on=["Top_K", "Npw", "R_joint_nOhm"], how="left", validate="one_to_one", indicator=True)
    arch["architecture_feasible"] = arch.pop("_merge").eq("both")
    references = arch.loc[arch["architecture_feasible"]].groupby("Top_K", sort=True)["lcoe_anchor_USD_per_MWh"].min().to_dict()
    if set(references) != set(TEMPERATURES):
        raise AssertionError(f"temperature reference minima missing: {references}")
    global_reference = float(min(references.values()))
    valid = arch["architecture_feasible"]
    for top, ref in references.items():
        mask = valid & arch["Top_K"].eq(top)
        arch.loc[mask, "delta_T_pct"] = 100.0 * (arch.loc[mask, "lcoe_anchor_USD_per_MWh"] - ref) / ref
    arch.loc[valid, "delta_global_pct"] = 100.0 * (arch.loc[valid, "lcoe_anchor_USD_per_MWh"] - global_reference) / global_reference
    arch.rename(columns={"rho_turn_uOhm_cm2": "best_rho_turn_uOhm_cm2"}, inplace=True)
    arch.to_csv(ARCH, index=False, encoding="utf-8", float_format="%.17g")
    boundary_rows, monotonicity_rows = [], []
    for top, n in product(TEMPERATURES, N_VALUES):
        frame = arch.loc[(arch["Top_K"].eq(top)) & (arch["Npw"].eq(n))].sort_values("R_joint_nOhm")
        r = frame["R_joint_nOhm"].to_numpy(float)
        feasibility = frame["architecture_feasible"].to_numpy(bool)
        row = {"scenario": "S2", "Top_K": float(top), "coolant": "He", "Npw": int(n), "R_ref_nOhm": 1.0,
               "L_star_temperature_USD_per_MWh": float(references[top]), "L_star_global_USD_per_MWh": global_reference}
        conditions = {"feas": feasibility}
        for pct in (1, 5, 10):
            conditions[f"temp_{pct}pct"] = feasibility & (frame["delta_T_pct"].to_numpy(float) <= pct)
            conditions[f"global_{pct}pct"] = feasibility & (frame["delta_global_pct"].to_numpy(float) <= pct)
        for criterion, passed in conditions.items():
            row.update(trace_first_interval(r, passed, criterion))
            monotonicity_rows.append({"scenario": "S2", "Top_K": float(top), "coolant": "He", "Npw": int(n), "criterion": criterion,
                "grid_points": len(r), "reference_pass": bool(passed[0]), "pass_count": int(passed.sum()),
                "true_to_false_transitions": int(np.sum(passed[:-1] & ~passed[1:])), "false_to_true_transitions": int(np.sum(~passed[:-1] & passed[1:])),
                "reentry_after_first_failure": bool(np.any(~passed[:-1] & passed[1:])), "first_interval_only": True})
        boundary_rows.append(row)
    brackets, monotonicity = pd.DataFrame(boundary_rows), pd.DataFrame(monotonicity_rows)
    brackets.to_csv(BRACKETS, index=False, encoding="utf-8", float_format="%.17g")
    monotonicity.to_csv(MONOTONICITY, index=False, encoding="utf-8")
    if len(arch) != 72_600 or len(brackets) != 600 or len(monotonicity) != 4_200:
        raise AssertionError("unexpected V6.2-C output dimensions")
    audit = {"experiment_id": "V6.2-C-grid-bracket-preparation", "status": "PASS_GRID_BRACKETS_NOT_FINAL_BOUNDARIES",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "git_commit": git_commit(), "device": DEVICE, "matrix_sha256": MATRIX_SHA256,
            "scenario": "S2", "coolant": "He", "source_dataset": {"path": audit_path(PARQUET_ROOT if use_parquet else INPUT), "audit_sha256": source_sha, "strict_feasible_rows": int(upstream["strict_feasible_rows"]), "S2_He_rows_read": rows_read},
        "definition": {"rho_turn": "minimum frozen-B1-anchor LCOE among 61 V6.2 rho samples", "interval": "first continuous true interval from R_ref=1 nOhm only", "temperature_interpolation": "none", "grid": "121-point Rj grid 1-100 nOhm", "final_boundary": "requires V6.2 direct-model log-space bisection"},
        "lcoe_references_USD_per_MWh": {str(k): float(v) for k, v in references.items()}, "global_reference_USD_per_MWh": global_reference,
        "outputs": {"architecture_optima": {"path": audit_path(ARCH), "rows": len(arch), "sha256": sha256(ARCH)}, "grid_brackets": {"path": audit_path(BRACKETS), "rows": len(brackets), "sha256": sha256(BRACKETS)}, "monotonicity": {"path": audit_path(MONOTONICITY), "rows": len(monotonicity), "sha256": sha256(MONOTONICITY)}}, "started_utc": started.isoformat()}
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
