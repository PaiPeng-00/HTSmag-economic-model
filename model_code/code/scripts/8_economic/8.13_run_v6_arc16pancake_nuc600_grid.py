#!/usr/bin/env python3
"""Run the V6 physical grid with the B1 device identity explicitly frozen.

This is a physical/E0-grid driver only.  The B1 incremental-anchor economic
conversion is a later, separately auditable stage.  It never overwrites the
earlier V6 cache generated with the wrong default device.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "v6_hts_temperature_tolerance_arc16pancake_nuc600"
V6_CHARGE_DIR = OUT / "charging_inputs" / "charging_tf"
SCAN = Path(__file__).with_name("scan_full_grid.py")
B1 = ROOT / "outputs" / "target_price_window" / "tables" / "B1_full_grid_green_eta030_V2_anchor_economics.csv"
DEVICE = "arc_16pancake_nuc600"
SCENARIOS = ("S1", "S2", "S3")
RHO = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 5000, 10000], dtype=float)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def load_scan(*, output_csv: Path, npw: np.ndarray, rjoint: np.ndarray, rho: np.ndarray, temperatures: list[tuple[float, str]]):
    # Must precede importing scan_full_grid.py, which imports fusion_tem.device.
    os.environ["FUSION_DEVICE"] = DEVICE
    os.environ["SCAN_OUTPUT_CSV_NAME"] = output_csv.name
    os.environ["SCAN_OUTPUT_XLSX_NAME"] = output_csv.with_suffix(".xlsx").name
    os.environ["SCAN_MANIFEST_NAME"] = output_csv.with_suffix(".manifest.json").name
    spec = importlib.util.spec_from_file_location("v6_arc16_scan", SCAN)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    if module.cfg.DEVICE != DEVICE:
        raise RuntimeError(f"device identity mismatch: expected {DEVICE}, got {module.cfg.DEVICE}")
    # Use the V6-only complete Npw=1..200 charging-time inputs.
    module.cfg.TF_SYSTEM_CHARGING_SIM_OUTPUT_DIR = V6_CHARGE_DIR
    module.TEMP_COOLANT_PAIRS = temperatures
    module.NPW_RANGE = npw
    module.R_JOINT_NOHM = rjoint
    module.RHO_TURN_UOHM_CM2 = rho
    module.SCENARIOS = list(SCENARIOS)
    module.OUTPUT_CSV = output_csv
    module.OUTPUT_XLSX = output_csv.with_suffix(".xlsx")
    module.OUTPUT_MANIFEST = output_csv.with_suffix(".manifest.json")
    return module


def run(*, output_csv: Path, npw: np.ndarray, rjoint: np.ndarray, rho: np.ndarray, temperatures: list[tuple[float, str]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    module = load_scan(output_csv=output_csv, npw=npw, rjoint=rjoint, rho=rho, temperatures=temperatures)
    module.main()


def preflight() -> None:
    path = OUT / "v6_arc16_b1_overlap_preflight.csv"
    if path.exists():
        raise FileExistsError(f"preflight output already exists: {path}")
    run(output_csv=path, npw=np.array([19]), rjoint=np.array([1.0]), rho=np.array([10000.0]), temperatures=[(10.0, "He")])
    keys = ["scenario", "Top_K", "coolant", "Npw", "rho_turn_uOhm_cm2", "R_joint_nOhm"]
    fields = [
        "Charging_time_999_h", "P_cryo_electric_W", "Q_total_Tc_W", "Q_nuclear_W",
        "Q_radiation_W", "Q_pancake_joint_W", "P_cryo_prod_W", "E_cryo_TF_annual_MWh",
        "AF", "r_cryo_re_fraction", "net_energy_annual_MWh", "E_fusion_th_year_MWh",
    ]
    candidate = pd.read_csv(path, usecols=keys + fields)
    source = pd.read_csv(B1, usecols=keys + fields)
    joined = candidate.merge(source, on=keys, suffixes=("_v6", "_b1"), validate="one_to_one")
    comparisons = {}
    for field in fields:
        delta = (pd.to_numeric(joined[f"{field}_v6"]) - pd.to_numeric(joined[f"{field}_b1"])).abs()
        comparisons[field] = {"max_abs": float(delta.max()), "mismatch_gt_1e-9": int((delta > 1e-9).sum())}
    passed = len(joined) == 3 and all(stat["mismatch_gt_1e-9"] == 0 for stat in comparisons.values())
    audit = {
        "experiment_id": "V6-device-preflight", "status": "PASS" if passed else "FAIL",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "device": DEVICE,
        "source_script": str(SCAN.relative_to(ROOT)), "source_script_sha256": sha256(SCAN),
        "git_commit": git_commit(), "input_b1": str(B1.relative_to(ROOT)), "input_b1_sha256": sha256(B1),
        "coordinates": {"Top_K": 10.0, "coolant": "He", "Npw": 19, "rho_turn_uOhm_cm2": 10000.0, "R_joint_nOhm": 1.0},
        "overlap_rows": int(len(joined)), "comparisons": comparisons,
    }
    (OUT / "v6_arc16_b1_overlap_preflight.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit("B1 overlap preflight failed; full V6 grid is blocked")


def scan() -> None:
    audit_path = OUT / "v6_arc16_b1_overlap_preflight.json"
    if not audit_path.exists() or json.loads(audit_path.read_text(encoding="utf-8")).get("status") != "PASS":
        raise SystemExit("Run --stage preflight and obtain PASS before the full V6 grid")
    output = OUT / "v6_direct_complete_designs_arc16pancake_nuc600.csv"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing V6 grid: {output}")
    run(output_csv=output, npw=np.arange(1, 201, dtype=int), rjoint=np.geomspace(1.0, 100.0, 121), rho=RHO, temperatures=[(4.2, "He"), (10.0, "He"), (20.0, "He")])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("preflight", "scan"), required=True)
    args = parser.parse_args()
    if args.stage == "preflight":
        preflight()
    else:
        scan()


if __name__ == "__main__":
    main()
