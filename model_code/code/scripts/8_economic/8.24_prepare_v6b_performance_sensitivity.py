#!/usr/bin/env python3
"""Prepare V6-B S2-He local log-Rj performance sensitivities for Fig. 6A/B."""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "v6_hts_temperature_tolerance_arc16pancake_nuc600"
INPUT = OUT / "v6b_feasible_designs.csv"
INPUT_AUDIT = OUT / "v6b_status_feasibility_audit.json"
PERFORMANCE = OUT / "v6b_temperature_resolved_performance.csv"
SENSITIVITIES = OUT / "v6b_local_log_rj_sensitivities.csv"
SUMMARY = OUT / "v6b_sensitivity_summary.csv"
MONOTONICITY = OUT / "v6b_monotonicity_audit.csv"
AUDIT = OUT / "v6b_performance_sensitivity_audit.json"
TEMPS = (4.2, 10.0, 20.0)
NVALS = tuple(range(1, 201))
CHUNK = 200_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def local_sensitivity(frame: pd.DataFrame, value_col: str, name: str) -> list[dict]:
    rows = []
    for (top, n), group in frame.groupby(["Top_K", "Npw"], sort=True):
        group = group.sort_values("R_joint_nOhm").reset_index(drop=True)
        r = group["R_joint_nOhm"].to_numpy(float)
        y = pd.to_numeric(group[value_col], errors="coerce").to_numpy(float)
        for i in range(len(group)):
            valid = np.isfinite(y[i]) and y[i] > 0
            left = i > 0 and np.isfinite(y[i - 1]) and y[i - 1] > 0
            right = i + 1 < len(group) and np.isfinite(y[i + 1]) and y[i + 1] > 0
            if valid and left and right:
                slope = abs((math.log(y[i + 1]) - math.log(y[i - 1])) / (math.log(r[i + 1]) - math.log(r[i - 1])))
                method = "centered_log_difference"
            elif valid and right:
                slope = abs((math.log(y[i + 1]) - math.log(y[i])) / (math.log(r[i + 1]) - math.log(r[i])))
                method = "forward_log_difference"
            elif valid and left:
                slope = abs((math.log(y[i]) - math.log(y[i - 1])) / (math.log(r[i]) - math.log(r[i - 1])))
                method = "backward_log_difference"
            else:
                slope = math.nan
                method = "adjacent_feasible_point_missing"
            rows.append({"scenario": "S2", "Top_K": float(top), "coolant": "He", "Npw": int(n),
                         "R_joint_nOhm": float(r[i]), "metric": name, "local_log_elasticity_abs": slope,
                         "difference_method": method})
    return rows


def main() -> None:
    required = (INPUT, INPUT_AUDIT)
    if not all(path.exists() for path in required):
        raise FileNotFoundError("V6-B strict-feasible input/audit missing")
    if any(path.exists() for path in (PERFORMANCE, SENSITIVITIES, SUMMARY, MONOTONICITY, AUDIT)):
        raise FileExistsError("refusing to overwrite V6-B performance-sensitivity outputs")
    input_audit = json.loads(INPUT_AUDIT.read_text(encoding="utf-8"))
    if input_audit.get("status") != "PASS" or sha256(INPUT) != input_audit["outputs"]["feasible_designs"]["sha256"]:
        raise AssertionError("V6-B strict-feasible input does not match PASS audit")
    columns = ["scenario", "Top_K", "coolant", "Npw", "R_joint_nOhm", "rho_turn_uOhm_cm2", "r_cryo_re_fraction", "lcoe_anchor_USD_per_MWh"]
    winners: dict[tuple[float, int, float], dict] = {}
    for chunk in pd.read_csv(INPUT, usecols=columns, chunksize=CHUNK, low_memory=False):
        part = chunk.loc[chunk["scenario"].eq("S2") & chunk["coolant"].eq("He") & chunk["Top_K"].isin(TEMPS) & (pd.to_numeric(chunk["R_joint_nOhm"], errors="coerce") <= 10.0)].copy()
        if part.empty:
            continue
        part["lcoe_anchor_USD_per_MWh"] = pd.to_numeric(part["lcoe_anchor_USD_per_MWh"], errors="raise")
        selected = part.loc[part.groupby(["Top_K", "Npw", "R_joint_nOhm"], sort=False)["lcoe_anchor_USD_per_MWh"].idxmin()]
        for row in selected.to_dict("records"):
            key = (float(row["Top_K"]), int(row["Npw"]), float(row["R_joint_nOhm"]))
            prior = winners.get(key)
            if prior is None or float(row["lcoe_anchor_USD_per_MWh"]) < float(prior["lcoe_anchor_USD_per_MWh"]):
                winners[key] = row
    actual = pd.DataFrame(winners.values())
    rvalues = np.sort(actual["R_joint_nOhm"].unique().astype(float))
    expected = pd.MultiIndex.from_product([TEMPS, NVALS, rvalues], names=["Top_K", "Npw", "R_joint_nOhm"]).to_frame(index=False)
    performance = expected.merge(actual, on=["Top_K", "Npw", "R_joint_nOhm"], how="left", validate="one_to_one", indicator=True)
    performance["architecture_feasible"] = performance["_merge"].eq("both")
    performance.drop(columns="_merge", inplace=True)
    performance.rename(columns={"rho_turn_uOhm_cm2": "best_rho_turn_uOhm_cm2"}, inplace=True)
    performance.to_csv(PERFORMANCE, index=False, encoding="utf-8", float_format="%.17g")
    feasible = performance.loc[performance["architecture_feasible"]].copy()
    sensitivity_rows = local_sensitivity(feasible, "r_cryo_re_fraction", "S_Rj_cryo") + local_sensitivity(feasible, "lcoe_anchor_USD_per_MWh", "S_Rj_LCOE")
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(SENSITIVITIES, index=False, encoding="utf-8", float_format="%.17g")
    summary_rows = []
    for (top, metric), group in sensitivity.groupby(["Top_K", "metric"], sort=True):
        values = group["local_log_elasticity_abs"].dropna().to_numpy(float)
        summary_rows.append({"scenario": "S2", "Top_K": top, "coolant": "He", "metric": metric,
                             "R_domain_nOhm": "1_to_10", "Npw_weighting": "equal_count_over_valid_Npw_Rj_locations",
                             "n_local_sensitivities": len(values), "median": float(np.median(values)),
                             "p25": float(np.quantile(values, .25)), "p75": float(np.quantile(values, .75)),
                             "p90": float(np.quantile(values, .90)), "maximum": float(np.max(values))})
    pd.DataFrame(summary_rows).to_csv(SUMMARY, index=False, encoding="utf-8", float_format="%.17g")
    mono_rows = []
    for (top, n), group in performance.groupby(["Top_K", "Npw"], sort=True):
        present = group.sort_values("R_joint_nOhm")["architecture_feasible"].to_numpy(bool)
        mono_rows.append({"scenario": "S2", "Top_K": top, "coolant": "He", "Npw": n, "grid_points": len(present),
                          "feasible_points": int(present.sum()), "false_to_true_transitions": int(np.sum(~present[:-1] & present[1:])),
                          "true_to_false_transitions": int(np.sum(present[:-1] & ~present[1:]))})
    pd.DataFrame(mono_rows).to_csv(MONOTONICITY, index=False, encoding="utf-8")
    audit = {"experiment_id": "V6-B-temperature-resolved-performance-sensitivity", "status": "PASS", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
             "source": {"path": str(INPUT.relative_to(ROOT)), "sha256": sha256(INPUT)}, "definition": {"scenario": "S2", "coolant": "He", "Npw": "integers 1..200", "R_domain_nOhm": [1.0, 10.0], "rho_optimization": "minimum frozen-B1-anchor LCOE among strict-feasible existing rho samples", "sensitivity": "abs[d ln(y)/d ln(Rj)]", "interior": "centered difference", "endpoints": "one-sided difference", "missing": "no interpolation; requires adjacent feasible point"},
             "outputs": {path.name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for path in (PERFORMANCE, SENSITIVITIES, SUMMARY, MONOTONICITY)}}
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
