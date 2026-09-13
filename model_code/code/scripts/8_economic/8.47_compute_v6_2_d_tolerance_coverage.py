#!/usr/bin/env python3
"""V6-D coverage measures derived from V6-C direct log-bisection boundaries.

Coverage is an equal-count measure across the 200 discrete Npw architectures.
It is deliberately reported separately for log-Rj and linear-Rj measures and
for the engineering [1,10] nOhm and stress-test [1,100] nOhm domains.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
V6_BASE = Path(os.environ['V6_RUN_BASE']) if os.environ.get('V6_RUN_BASE') else ROOT / 'v6_2_arc_actual_inductance'
OUT = V6_BASE / 'stage_D'
SOURCE_C = V6_BASE / 'stage_C_fast_r2'
SOURCE_GRID = V6_BASE / 'stage_C'
BOUNDARIES = SOURCE_C / "v6_2_c_joint_tolerance_boundaries.csv"
BRACKETS = SOURCE_GRID / "v6_2_c_s2_joint_tolerance_grid_brackets.csv"
C_AUDIT = SOURCE_C / "v6_2_c_direct_bisection_audit.json"
OUT_COVERAGE = OUT / "v6d_tolerance_coverage.csv"
OUT_SENSITIVITY = OUT / "v6d_measure_sensitivity.csv"
OUT_AUDIT = OUT / "v6d_sampling_audit.json"

CRITERIA = ("feas", "temp_1pct", "temp_5pct", "temp_10pct", "global_1pct", "global_5pct", "global_10pct")
DOMAINS = ((10.0, "engineering_1_to_10_nOhm"), (100.0, "stress_1_to_100_nOhm"))
MEASURES = ("log_Rj", "linear_Rj")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_path(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def contribution(boundary: np.ndarray, upper: float, measure: str) -> np.ndarray:
    clipped = np.clip(np.nan_to_num(boundary, nan=1.0), 1.0, upper)
    if measure == "log_Rj":
        return np.log(clipped) / math.log(upper)
    if measure == "linear_Rj":
        return (clipped - 1.0) / (upper - 1.0)
    raise ValueError(measure)


def main() -> None:
    required = (BOUNDARIES, BRACKETS, C_AUDIT)
    if not all(path.exists() for path in required):
        raise FileNotFoundError("V6-C direct boundary inputs are missing")
    OUT.mkdir(parents=True, exist_ok=True)
    if any(path.exists() for path in (OUT_COVERAGE, OUT_SENSITIVITY, OUT_AUDIT)):
        raise FileExistsError("refusing to overwrite V6-D coverage artifacts")
    direct_audit = json.loads(C_AUDIT.read_text(encoding="utf-8"))
    if direct_audit.get("status") != "PASS":
        raise AssertionError("V6-C direct-bisection audit is not PASS")
    boundaries = pd.read_csv(BOUNDARIES)
    brackets = pd.read_csv(BRACKETS)
    expected = 3 * 200
    if len(boundaries) != expected or len(brackets) != expected:
        raise AssertionError("unexpected V6-C boundary or bracket row count")
    keys = ["scenario", "Top_K", "coolant", "Npw"]
    joined = boundaries.merge(brackets, on=keys, suffixes=("", "_grid"), validate="one_to_one")
    started = datetime.now(timezone.utc)
    coverage_rows = []
    for top in (4.2, 10.0, 20.0):
        for criterion in CRITERIA:
            direct = pd.to_numeric(joined.loc[joined["Top_K"].eq(top), f"Rj_max_{criterion}_nOhm"], errors="coerce").to_numpy(float)
            for upper, domain_name in DOMAINS:
                for measure in MEASURES:
                    values = contribution(direct, upper, measure)
                    coverage_rows.append({
                        "experiment_id": "V6-D-direct-boundary-coverage",
                        "model_version": "V6-C-direct-log-bisection",
                        "source_hash": sha256(BOUNDARIES),
                        "scenario": "S2", "Top_K": top, "coolant": "He", "criterion": criterion,
                        "measure": measure, "R_min_nOhm": 1.0, "R_upper_nOhm": upper, "domain": domain_name,
                        "Npw_weighting": "equal_count_1_to_200",
                        "coverage": float(values.mean()), "coverage_percent": float(100.0 * values.mean()),
                        "N_low_reference_infeasible": int(joined.loc[joined["Top_K"].eq(top), f"low_reference_infeasible_{criterion}"].sum()),
                        "N_right_censored": int(joined.loc[joined["Top_K"].eq(top), f"right_censored_{criterion}"].sum()),
                    })
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(OUT_COVERAGE, index=False, encoding="utf-8", float_format="%.17g")

    sensitivity_rows = []
    for criterion in CRITERIA:
        for upper, domain_name in DOMAINS:
            for top in (4.2, 10.0, 20.0):
                direct = pd.to_numeric(joined.loc[joined["Top_K"].eq(top), f"Rj_max_{criterion}_nOhm"], errors="coerce").to_numpy(float)
                grid = pd.to_numeric(joined.loc[joined["Top_K"].eq(top), f"Rj_max_{criterion}_grid_lower_nOhm"], errors="coerce").to_numpy(float)
                for measure in MEASURES:
                    c_direct = float(contribution(direct, upper, measure).mean())
                    c_grid = float(contribution(grid, upper, measure).mean())
                    sensitivity_rows.append({
                        "experiment_id": "V6-D-measure-and-grid-sensitivity", "model_version": "V6-C-direct-log-bisection",
                        "source_hash": sha256(BOUNDARIES), "scenario": "S2", "Top_K": top, "coolant": "He",
                        "criterion": criterion, "measure": measure, "R_min_nOhm": 1.0, "R_upper_nOhm": upper,
                        "domain": domain_name, "direct_boundary_coverage": c_direct,
                        "stored_grid_lower_endpoint_coverage": c_grid,
                        "stored_grid_minus_direct": c_grid - c_direct,
                        "interpretation": "stored_grid_lower_endpoint_is_conservative_for_finite_first_interval_boundaries",
                    })
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(OUT_SENSITIVITY, index=False, encoding="utf-8", float_format="%.17g")
    audit = {
        "experiment_id": "V6-D-tolerance-coverage", "status": "PASS",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "device": "arc_16pancake_nuc600_v6_2",
        "scenario": "S2", "coolant": "He", "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source": {"v6c_boundaries": {"path": audit_path(BOUNDARIES), "sha256": sha256(BOUNDARIES)},
                "v6c_brackets": {"path": audit_path(BRACKETS), "sha256": sha256(BRACKETS)}},
        "definition": {"Npw": "integer 1..200, equal count", "temperature": "discrete condition; excluded from measure",
                       "R_min_nOhm": 1.0, "measures": list(MEASURES), "domains_nOhm": [list(x) for x in DOMAINS],
                       "right_censoring": "credited only up to stated R_upper; no extrapolation beyond 100 nOhm",
                       "low_reference_infeasible": "zero coverage contribution"},
        "outputs": {path.name: {"path": audit_path(path), "rows": len(pd.read_csv(path)), "sha256": sha256(path)} for path in (OUT_COVERAGE, OUT_SENSITIVITY)},
        "started_utc": started.isoformat(),
    }
    OUT_AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
