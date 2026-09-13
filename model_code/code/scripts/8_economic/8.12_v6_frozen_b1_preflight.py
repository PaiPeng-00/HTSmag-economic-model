#!/usr/bin/env python3
"""Gate V6 on the v3.0.0 frozen source before B1 anchor re-economics.

The current checkout is intentionally not used for physical evaluations here:
it may contain user work-in-progress.  This driver loads the release snapshot,
verifies an exact B1-overlap sample, and can then produce an independently
cached direct grid for subsequent B1 re-economics.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


CURRENT_CODE = Path(__file__).resolve().parents[2]
WORKSPACE = CURRENT_CODE.parents[1]
FROZEN_CODE = WORKSPACE / "12-代码和数据上传" / "HTSmag-economic-model-v3.0.0"
FROZEN_SCAN = FROZEN_CODE / "scripts" / "8_economic" / "scan_full_grid.py"
FROZEN_ANCHOR = FROZEN_CODE / "scripts" / "8_economic" / "8.5_recompute_v2_anchor_economics.py"
OUT = CURRENT_CODE / "outputs" / "v6_hts_temperature_tolerance"
B1 = CURRENT_CODE / "outputs" / "target_price_window" / "tables" / "B1_full_grid_green_eta030_V2_anchor_economics.csv"
PREFLIGHT_RAW = OUT / "v6_frozen_v300_overlap_preflight.csv"
PREFLIGHT_AUDIT = OUT / "v6_frozen_v300_overlap_preflight.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def import_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_frozen_scan():
    if not FROZEN_SCAN.exists() or not FROZEN_ANCHOR.exists():
        raise FileNotFoundError("V3.0.0 frozen source snapshot is incomplete")
    sys.path.insert(0, str(FROZEN_CODE / "src"))
    return import_path("v6_frozen_scan", FROZEN_SCAN)


def run_scan(*, csv: Path, npw: np.ndarray, rjoint: np.ndarray, rho: np.ndarray, temperatures: list[tuple[float, str]]) -> None:
    os.environ["SCAN_OUTPUT_CSV_NAME"] = csv.name
    os.environ["SCAN_OUTPUT_XLSX_NAME"] = csv.with_suffix(".xlsx").name
    os.environ["SCAN_MANIFEST_NAME"] = csv.with_suffix(".manifest.json").name
    scan = load_frozen_scan()
    scan.TEMP_COOLANT_PAIRS = temperatures
    scan.NPW_RANGE = npw
    scan.R_JOINT_NOHM = rjoint
    scan.RHO_TURN_UOHM_CM2 = rho
    scan.SCENARIOS = ["S1", "S2", "S3"]
    scan.OUTPUT_CSV = csv
    scan.OUTPUT_XLSX = csv.with_suffix(".xlsx")
    scan.OUTPUT_MANIFEST = csv.with_suffix(".manifest.json")
    scan.main()


def anchored(frame: pd.DataFrame) -> pd.DataFrame:
    anchor = import_path("v6_frozen_anchor", FROZEN_ANCHOR)
    e0_manifest = json.loads(anchor.E0_MANIFEST.read_text(encoding="utf-8"))
    _, refs = anchor.find_reference_rows(e0_manifest)
    return anchor.transform_chunk(frame, refs, e0_manifest)


def preflight() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for path in (PREFLIGHT_RAW, PREFLIGHT_RAW.with_suffix(".manifest.json")):
        if path.exists():
            path.unlink()
    run_scan(
        csv=PREFLIGHT_RAW,
        npw=np.array([19], dtype=int),
        rjoint=np.array([1.0]),
        rho=np.array([10000.0]),
        temperatures=[(10.0, "He")],
    )
    raw = pd.read_csv(PREFLIGHT_RAW, low_memory=False)
    derived = anchored(raw)
    keys = ["scenario", "Top_K", "coolant", "Npw", "rho_turn_uOhm_cm2", "R_joint_nOhm"]
    fields = [
        "P_cryo_electric_W", "Q_total_Tc_W", "AF", "AF_ref",
        "r_cryo_re_fraction", "net_energy_annual_MWh", "E_fusion_th_year_MWh",
        "lcoe_anchor_USD_per_MWh", "C_plant_anchor_USD", "annual_noncapital_cost_USD",
    ]
    source = pd.read_csv(B1, usecols=keys + fields)
    joined = derived[keys + fields].merge(source, on=keys, suffixes=("_v6", "_b1"), validate="one_to_one")
    stats = {}
    for field in fields:
        delta = (pd.to_numeric(joined[f"{field}_v6"]) - pd.to_numeric(joined[f"{field}_b1"])).abs()
        stats[field] = {"max_abs": float(delta.max()), "mismatch_gt_1e-9": int((delta > 1e-9).sum())}
    passed = len(joined) == 3 and all(v["mismatch_gt_1e-9"] == 0 for v in stats.values())
    audit = {
        "experiment_id": "V6-frozen-source-preflight", "status": "PASS" if passed else "FAIL",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_source_root": str(FROZEN_CODE), "frozen_scan_sha256": sha256(FROZEN_SCAN),
        "frozen_anchor_script_sha256": sha256(FROZEN_ANCHOR), "b1_sha256": sha256(B1),
        "coordinates": {"Top_K": 10.0, "coolant": "He", "Npw": 19, "rho_turn_uOhm_cm2": 10000.0, "R_joint_nOhm": 1.0},
        "overlap_rows": int(len(joined)), "field_comparison": stats,
    }
    PREFLIGHT_AUDIT.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    if not passed:
        raise SystemExit("Frozen-source preflight failed; V6 direct scan remains blocked")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("preflight", "scan"), required=True)
    args = parser.parse_args()
    if args.stage == "preflight":
        preflight()
        return
    audit = json.loads(PREFLIGHT_AUDIT.read_text(encoding="utf-8")) if PREFLIGHT_AUDIT.exists() else {}
    if audit.get("status") != "PASS":
        raise SystemExit("Run --stage preflight and obtain PASS before the frozen V6 direct scan")
    run_scan(
        csv=OUT / "v6_direct_complete_designs_frozen_v300.csv",
        npw=np.arange(1, 201, dtype=int),
        rjoint=np.geomspace(1.0, 100.0, 121),
        rho=np.array([10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000,5000,10000], dtype=float),
        temperatures=[(4.2, "He"), (10.0, "He"), (20.0, "He")],
    )


if __name__ == "__main__":
    main()
