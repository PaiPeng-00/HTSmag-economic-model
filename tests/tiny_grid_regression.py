"""Solve six circuits and 48 economic cases, then compare with frozen V10 records."""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "model/model_code/code"


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/tiny_grid")
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise SystemExit("Select a new --output directory")
    out.mkdir(parents=True)
    os.environ["FUSION_DEVICE"] = "arc_16pancake_nuc600_v6_2"
    os.environ["SCAN_DEFER_FINALIZE"] = "1"
    os.environ["SCAN_PROGRESS_INTERVAL"] = "48"
    os.environ["CIRCUIT_SCALAR_CACHE_CSV"] = str(out / "circuit_cache.csv")
    sys.path.insert(0, str(CODE / "src"))
    cache = load("tiny_circuit_cache", CODE / "scripts/2_charging/2.7_build_circuit_scalar_cache.py")
    sys.argv = ["cache", "--device", "arc_16pancake_nuc600_v6_2", "--scan-config",
                str(ROOT / "tests/tiny_grid.yaml"), "--output", str(out / "circuit_cache.csv")]
    cache.main()
    scan = load("tiny_scan", CODE / "scripts/8_economic/scan_full_grid.py")
    # Isolated regression subset; the production grid file is unchanged.
    scan.NPW_RANGE = np.array([19, 200])
    scan.RHO_TURN_UOHM_CM2 = np.array([10000.])
    scan.R_JOINT_NOHM = np.array([1., 100.])
    scan.OUTPUT_CSV = out / "raw_scan.csv"
    scan.OUTPUT_XLSX = out / "unused.xlsx"
    scan.OUTPUT_MANIFEST = out / "scan_manifest.json"
    scan.main()
    rows = pd.read_csv(scan.OUTPUT_CSV)
    if len(rows) != 48:
        raise ValueError(f"Expected 48 cases, obtained {len(rows)}")
    ledger = load("tiny_ledger", ROOT / "pipeline/build_v10_candidate_ledgers.py")
    refs = ledger.find_references(scan.OUTPUT_CSV)
    rows["lcoe_anchor_USD_per_MWh"] = ledger.anchor_lcoe(rows, refs)
    comparisons = 0
    for (temp, coolant), block in rows.groupby(["Top_K", "coolant"]):
        source = args.data / "full_model_dataset" / f"realizations_T{str(float(temp)).replace('.', 'p')}_{coolant}.parquet"
        archived = pd.read_parquet(source, filters=[("Npw", "in", [19, 200]),
                                                  ("rho_turn_uOhm_cm2", "==", 10000.),
                                                  ("R_joint_nOhm", "in", [1., 100.])])
        for _, row in block.iterrows():
            selected = archived.loc[archived.Npw.eq(row.Npw) & archived.R_joint_nOhm.eq(row.R_joint_nOhm)]
            if len(selected) != 1:
                raise ValueError("Nonunique archived regression key")
            for column in ("Aplant", "E_net_year_MWh", "r_cryo_re_fraction", "lcoe_anchor_USD_per_MWh"):
                np.testing.assert_allclose(row[column], selected.iloc[0][f"{column}_{row.scenario}"],
                                           rtol=2e-9, atol=1e-8, equal_nan=True,
                                           err_msg=f"{temp}/{coolant}/{row.scenario}/{row.Npw}/{row.R_joint_nOhm}/{column}")
                comparisons += 1
    report = {"status": "PASS", "circuits_solved": 6, "scenario_cases_solved": 48,
              "scalar_comparisons": comparisons, "relative_tolerance": 2e-9,
              "full_grid_rerun": "NOT RUN"}
    (out / "TINY_GRID_VALIDATION.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
