#!/usr/bin/env python3
"""Correct V6-E cross-scenario min aggregation for low-reference failures.

For min_s Rj,max(s), any scenario that fails at Rj=1 nOhm makes the cross
architecture low-reference-infeasible.  It must not be omitted as a NaN while
taking the minimum of the remaining finite scenario boundaries.
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "v6_hts_temperature_tolerance_arc16pancake_nuc600"
TOLERANCE = OUT / "v6e_cross_scenario_tolerance.csv"
BINDING = OUT / "v6e_binding_scenario.csv"
COVERAGE = OUT / "v6e_cross_scenario_coverage.csv"
AUDIT = OUT / "v6e_cross_scenario_audit.json"
EVALS = OUT / "v6e_direct_bisection_evaluations.csv"
TAG = "superseded_low_reference_aggregation_bug"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def coverage(boundary: np.ndarray, upper: float, measure: str) -> float:
    clipped = np.clip(np.nan_to_num(boundary, nan=1.0), 1.0, upper)
    if measure == "log_Rj":
        return float(np.log(clipped).mean() / math.log(upper))
    return float(((clipped - 1.0) / (upper - 1.0)).mean())


def backup(path: Path) -> Path:
    destination = path.with_name(f"{path.stem}.{TAG}{path.suffix}")
    if destination.exists():
        raise FileExistsError(f"backup already exists: {destination}")
    shutil.copy2(path, destination)
    return destination


def main() -> None:
    if not all(path.exists() for path in (TOLERANCE, BINDING, COVERAGE, AUDIT, EVALS)):
        raise FileNotFoundError("complete V6-E outputs are required before aggregation correction")
    upstream = json.loads(AUDIT.read_text(encoding="utf-8"))
    if upstream.get("status") != "PASS":
        raise AssertionError("V6-E direct-bisection audit is not PASS")
    tolerance = pd.read_csv(TOLERANCE)
    required = {"scenario", "Top_K", "Npw", "Rj_max_global_5pct_nOhm", "low_reference_infeasible_global_5pct", "right_censored_global_5pct"}
    if not required.issubset(tolerance.columns):
        raise AssertionError("required per-scenario V6-E boundary columns missing")
    source_hash = sha256(TOLERANCE)
    bare = tolerance.drop(columns=[col for col in tolerance.columns if "cross_global_5pct" in col or col == "binding_scenario_global_5pct"])
    cross_rows, binding_rows = [], []
    for (top, coolant, n), group in bare.groupby(["Top_K", "coolant", "Npw"], sort=True):
        low_mask = group["low_reference_infeasible_global_5pct"].astype(bool)
        if low_mask.any():
            low = True; censored = False; boundary = math.nan
            binding_scenarios = ";".join(group.loc[low_mask, "scenario"].astype(str).tolist())
        else:
            values = pd.to_numeric(group["Rj_max_global_5pct_nOhm"], errors="coerce")
            if values.isna().any():
                raise AssertionError("non-low-reference NaN boundary encountered")
            boundary = float(values.min())
            binding_scenarios = ";".join(group.loc[np.isclose(values, boundary, rtol=0, atol=1e-12), "scenario"].astype(str).tolist())
            low = False
            censored = bool(group["right_censored_global_5pct"].astype(bool).all())
        cross_rows.append({"Top_K": float(top), "coolant": coolant, "Npw": int(n), "R_ref_nOhm": 1.0,
                           "Rj_max_cross_global_5pct_nOhm": boundary, "F_R_cross_global_5pct": boundary,
                           "binding_scenario_global_5pct": binding_scenarios,
                           "low_reference_infeasible_cross_global_5pct": low,
                           "right_censored_cross_global_5pct": censored})
        binding_rows.append({"Top_K": float(top), "coolant": coolant, "Npw": int(n), "criterion": "cross_global_5pct",
                             "binding_scenario": binding_scenarios, "Rj_max_cross_global_5pct_nOhm": boundary,
                             "low_reference_infeasible": low, "right_censored": censored})
    cross = pd.DataFrame(cross_rows)
    corrected = bare.merge(cross, on=["Top_K", "coolant", "Npw", "R_ref_nOhm"], how="left", validate="many_to_one")
    coverage_rows = []
    for top, group in cross.groupby("Top_K", sort=True):
        values = group["Rj_max_cross_global_5pct_nOhm"].to_numpy(float)
        for upper, domain in ((10.0, "engineering_1_to_10_nOhm"), (100.0, "stress_1_to_100_nOhm")):
            for measure in ("log_Rj", "linear_Rj"):
                value = coverage(values, upper, measure)
                coverage_rows.append({"experiment_id": "V6-E-cross-scenario-coverage", "model_version": "V6-E-direct-log-bisection", "source_hash": sha256(EVALS),
                                      "Top_K": float(top), "coolant": "He", "criterion": "cross_global_5pct", "measure": measure,
                                      "R_min_nOhm": 1.0, "R_upper_nOhm": upper, "domain": domain, "Npw_weighting": "equal_count_1_to_200",
                                      "coverage": value, "coverage_percent": 100.0 * value,
                                      "N_low_reference_infeasible": int(group["low_reference_infeasible_cross_global_5pct"].sum()),
                                      "N_right_censored": int(group["right_censored_cross_global_5pct"].sum())})
    coverage_df = pd.DataFrame(coverage_rows)
    staged = {TOLERANCE: TOLERANCE.with_suffix(".corrected.tmp.csv"), BINDING: BINDING.with_suffix(".corrected.tmp.csv"), COVERAGE: COVERAGE.with_suffix(".corrected.tmp.csv"), AUDIT: AUDIT.with_suffix(".corrected.tmp.json")}
    corrected.to_csv(staged[TOLERANCE], index=False, encoding="utf-8", float_format="%.17g")
    pd.DataFrame(binding_rows).to_csv(staged[BINDING], index=False, encoding="utf-8")
    coverage_df.to_csv(staged[COVERAGE], index=False, encoding="utf-8", float_format="%.17g")
    backups = {path.name: {"path": str(backup(path).relative_to(ROOT)), "sha256": sha256(path)} for path in (TOLERANCE, BINDING, COVERAGE, AUDIT)}
    audit = {**upstream, "status": "PASS", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
             "postprocessing_correction": {"reason": "cross-scenario min now treats any low-reference-infeasible scenario as low-reference-infeasible cross architecture", "prior_tolerance_sha256": source_hash, "backups": backups},
             "outputs": {TOLERANCE.name: {"path": str(TOLERANCE.relative_to(ROOT)), "sha256": sha256(staged[TOLERANCE])}, BINDING.name: {"path": str(BINDING.relative_to(ROOT)), "sha256": sha256(staged[BINDING])}, COVERAGE.name: {"path": str(COVERAGE.relative_to(ROOT)), "sha256": sha256(staged[COVERAGE])}, EVALS.name: {"path": str(EVALS.relative_to(ROOT)), "sha256": sha256(EVALS)}}}
    staged[AUDIT].write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for original, staged_path in staged.items():
        staged_path.replace(original)
    print(json.dumps({"status": "PASS", "affected_cross_low_reference_rows": int((cross["low_reference_infeasible_cross_global_5pct"].astype(bool)).sum()), "outputs": {path.name: sha256(path) for path in (TOLERANCE, BINDING, COVERAGE, AUDIT)}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
