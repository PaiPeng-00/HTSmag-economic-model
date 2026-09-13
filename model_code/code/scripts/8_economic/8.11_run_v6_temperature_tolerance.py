#!/usr/bin/env python3
"""V6 baseline gate and resumable direct-evaluation scan launcher.

This driver deliberately delegates every physical/economic evaluation to the
frozen ``scan_full_grid.py`` workflow.  It never interpolates temperature or
joint resistance, and writes only under ``outputs/v6_hts_temperature_tolerance``.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "v6_hts_temperature_tolerance"
B1 = ROOT / "outputs" / "target_price_window" / "tables" / "B1_full_grid_green_eta030_V2_anchor_economics.csv"
SCAN = Path(__file__).with_name("scan_full_grid.py")
SCENARIOS = ("S1", "S2", "S3")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def load_fig6_module():
    path = Path(__file__).with_name("8.10_plot_fig6_near_optimal_commercial.py")
    spec = importlib.util.spec_from_file_location("v6_fig6_audit", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def baseline_audit() -> None:
    """Reproduce immutable B1 hard counts without mutating the B1 input."""
    OUT.mkdir(parents=True, exist_ok=True)
    header = pd.read_csv(B1, nrows=0).columns.tolist()
    required = {
        "scenario": "scenario", "N_pw": "Npw", "R_j_nOhm": "R_joint_nOhm",
        "rho_turn_uohm_cm2": "rho_turn_uOhm_cm2", "operating_temperature_K": "Top_K",
        "coolant": "coolant", "AF_ref": "AF_ref",
        "annual_TF_cryo_fraction": "r_cryo_re_fraction",
        "annual_net_electricity_export": "net_energy_annual_MWh",
        "LCOE_anchor_2025USD_per_MWh": "lcoe_anchor_USD_per_MWh",
    }
    missing = sorted(set(required.values()).difference(header))
    if missing:
        raise RuntimeError(f"B1 column mapping incomplete: {missing}")
    fig6 = load_fig6_module()
    opt, cross = fig6.aggregate_architectures(B1)
    minima = opt.loc[opt.feasible_any.eq(1)].groupby("scenario").scenario_min_LCOE.min().to_dict()
    observed = {
        "common_feasible_architectures": int(cross.common_feasible.sum()),
        "empirical_cross_scenario_architecture_coverage_1pct_count": int((cross.common_feasible.eq(1) & (cross.max_rel_penalty_pct <= 1)).sum()),
        "empirical_cross_scenario_architecture_coverage_5pct_count": int((cross.common_feasible.eq(1) & (cross.max_rel_penalty_pct <= 5)).sum()),
        "empirical_cross_scenario_architecture_coverage_10pct_count": int((cross.common_feasible.eq(1) & (cross.max_rel_penalty_pct <= 10)).sum()),
    }
    expected = {"common_feasible_architectures": 1054, "empirical_cross_scenario_architecture_coverage_1pct_count": 621, "empirical_cross_scenario_architecture_coverage_5pct_count": 886, "empirical_cross_scenario_architecture_coverage_10pct_count": 1020}
    summary = pd.DataFrame([{"scenario": s, "minimum_lcoe_anchor_2025usd_per_mwh": minima[s]} for s in SCENARIOS])
    summary.to_csv(OUT / "v6a_baseline_summary.csv", index=False)
    mapping = {semantic: {"source_column": column} for semantic, column in required.items()}
    mapping["annual_TF_cryo_fraction"]["note"] = "Fraction column; do not use legacy percent column r_cryo_re."
    (OUT / "v6a_column_mapping.json").write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")
    audit = {
        "experiment_id": "V6-A", "status": "PASS" if observed == expected else "FAIL",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "source_dataset": str(B1.relative_to(ROOT)),
        "source_dataset_sha256": sha256(B1), "source_script": str(SCAN.relative_to(ROOT)),
        "source_script_sha256": sha256(SCAN), "git_commit": git_commit(), "model_version": "v3.0.0",
        "cost_boundary": "incremental_anchor_v1_2025usd", "cryoplant_model": "Green rated eta030",
        "rows": 593712, "columns": len(header), "column_mapping": mapping,
        "scenario_minimum_lcoe_anchor_2025usd_per_mwh": minima, "expected": expected, "observed": observed,
        "empirical_coverages_pct": {"1pct": 100 * 621 / 1054, "5pct": 100 * 886 / 1054, "10pct": 100 * 1020 / 1054},
        "input_mutated": False,
    }
    (OUT / "v6a_baseline_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    if audit["status"] != "PASS":
        raise SystemExit("V6-A FAIL; direct V6 scan is blocked")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


def direct_scan() -> None:
    """Run the existing evaluator on the V6 direct grid; its CSV is the cache."""
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ["SCAN_OUTPUT_CSV_NAME"] = "v6_direct_complete_designs.csv"
    os.environ["SCAN_OUTPUT_XLSX_NAME"] = "v6_direct_complete_designs.xlsx"
    os.environ["SCAN_MANIFEST_NAME"] = "v6_direct_complete_designs_manifest.json"
    spec = importlib.util.spec_from_file_location("v6_existing_scan", SCAN)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.TEMP_COOLANT_PAIRS = [(4.2, "He"), (10.0, "He"), (20.0, "He")]
    module.NPW_RANGE = np.arange(1, 201, dtype=int)
    module.R_JOINT_NOHM = np.geomspace(1.0, 100.0, 121)
    module.RHO_TURN_UOHM_CM2 = np.array([10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000,5000,10000], dtype=float)
    module.SCENARIOS = list(SCENARIOS)
    module.OUTPUT_CSV = OUT / "v6_direct_complete_designs.csv"
    module.OUTPUT_XLSX = OUT / "v6_direct_complete_designs.xlsx"
    module.OUTPUT_MANIFEST = OUT / "v6_direct_complete_designs_manifest.json"
    module.main()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("a", "scan"), required=True)
    args = parser.parse_args()
    if args.stage == "a":
        baseline_audit()
    else:
        baseline_audit()
        direct_scan()


if __name__ == "__main__":
    main()
