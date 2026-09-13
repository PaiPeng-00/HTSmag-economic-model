#!/usr/bin/env python3
"""Directly generate the V6-only charging-time input on its full Npw grid.

The existing 2.5 charging solver is used unchanged.  Results are isolated
from B1 under the V6 output directory, so no V5/B1 upstream table is changed.
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
CHARGE_ROOT = OUT / "charging_inputs"
DEVICE = "arc_16pancake_nuc600"
CHARGE_SCRIPT = ROOT / "scripts" / "2_charging" / "2.5_charge_time999_TF_system_all_Npw=1-200.py"
RHO = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 5000, 10000], dtype=float)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def load_charge_module():
    os.environ["FUSION_DEVICE"] = DEVICE
    spec = importlib.util.spec_from_file_location("v6_charge_time_module", CHARGE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    if module.cfg.DEVICE != DEVICE:
        raise RuntimeError(f"device identity mismatch: expected {DEVICE}, got {module.cfg.DEVICE}")
    module.Npw_values = np.arange(1, 201, dtype=int)
    module.rho_turn_values_uOhm_cm2 = RHO.copy()
    module.rho_turn_values_Ohm_m2 = RHO * 1e-10
    module.fixed_Npw_values = module.Npw_values.copy()
    module.fixed_rho_turn_values_uOhm_cm2 = RHO.copy()
    module.cfg.OUTPUTS_TABLES_DIR = CHARGE_ROOT
    module.cfg.TF_SYSTEM_CHARGING_SIM_OUTPUT_DIR = CHARGE_ROOT / "charging_tf"
    module.cfg.OUTPUTS_FIGURES_DIR = OUT / "charging_figures"
    # V6 needs the auditable table, not this legacy heatmap; avoid making an
    # ancillary plot that assumes the historical plotting subset.
    module.plot_charging_time_heatmap_fixed = lambda *_args, **_kwargs: None
    return module


def expected_paths(module) -> list[Path]:
    paths = []
    for temp in (4.2, 10.0, 20.0):
        ip = module.cfg.TEMPERATURE_CASES[temp]["Ip"]
        nt = module.cfg.TEMPERATURE_CASES[temp]["Ntape_coil"]
        paths.append(module.cfg.TF_SYSTEM_CHARGING_SIM_OUTPUT_DIR / f"Temp_{temp}K_Ip_{ip}A_Ntape_coil_{nt}_charge999" / module.cfg.TF_SYSTEM_CHARGING_SIM_OUTPUT_FILE_Npw1_200)
    return paths


def validate(paths: list[Path]) -> dict:
    checks = []
    for path in paths:
        frame = pd.read_excel(path, index_col=0)
        frame.index = pd.to_numeric(frame.index)
        frame.columns = pd.to_numeric(frame.columns)
        missing = int(frame.reindex(index=RHO, columns=np.arange(1, 201)).isna().sum().sum())
        checks.append({"path": str(path.relative_to(ROOT)), "rows": int(len(frame.index)), "columns": int(len(frame.columns)), "missing_required_cells": missing, "sha256": sha256(path)})
    if any(c["missing_required_cells"] for c in checks):
        raise RuntimeError(f"V6 charging input incomplete: {checks}")
    return {"status": "PASS", "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="permit filling an existing V6-only workbook")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    module = load_charge_module()
    paths = expected_paths(module)
    if any(path.exists() for path in paths) and not args.force:
        raise SystemExit("V6 charging workbook exists; rerun with --force only to resume its own V6 cache")
    for temp in (4.2, 10.0, 20.0):
        module.charge999(temp, module.cfg.TEMPERATURE_CASES[temp]["Ip"], module.cfg.TEMPERATURE_CASES[temp]["Ntape_coil"])
    audit = {
        "experiment_id": "V6-charge-input", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device": DEVICE, "source_script": str(CHARGE_SCRIPT.relative_to(ROOT)), "source_script_sha256": sha256(CHARGE_SCRIPT),
        "git_commit": git_commit(), "temperatures_K": [4.2, 10.0, 20.0], "coolant": "He",
        "Npw": {"min": 1, "max": 200, "count": 200}, "rho_turn_uOhm_cm2": RHO.tolist(),
        **validate(paths),
    }
    (OUT / "v6_charge_input_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
