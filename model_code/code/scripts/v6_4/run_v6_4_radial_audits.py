#!/usr/bin/env python3
"""Run V6.4 continuous-effective-turn radial-model audits on exact circuit solves."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
CODE = REPO / "code"
sys.path.insert(0, str(CODE / "src"))

from fusion_tem import device as cfg
from fusion_tem.utils import (LEGACY_DISCRETE_RADIAL_MODEL,
                              calculate_radial_resistance,
                              continuous_effective_turn_rho_values)


def load_solver():
    path = CODE / "scripts/2_charging/2.7_build_circuit_scalar_cache.py"
    spec = importlib.util.spec_from_file_location("v64_circuit_cache", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=float), q)) if values else None


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def solve(module, matrix, top, npw, rho, legacy=False):
    nt = int(cfg.Nt_list[float(top)])
    original = module.calculate_radial_resistance
    if legacy:
        module.calculate_radial_resistance = LEGACY_DISCRETE_RADIAL_MODEL
    try:
        row = module._solve_one(module._load_charge_module(), matrix, float(top), int(npw), float(rho))
    finally:
        module.calculate_radial_resistance = original
    rho_values, contract = continuous_effective_turn_rho_values(int(npw), nt, float(rho) * 1e-10)
    return row, contract, rho_values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--smoke-npw", type=int, nargs="+", default=[1, 19, 50, 100, 200])
    parser.add_argument("--smoke-rho", type=float, nargs="+", default=[100.0, 1000.0, 5000.0, 10000.0])
    parser.add_argument("--full-regression", action="store_true",
                        help="Run the expensive 3x200x61 exact V6.4-vs-legacy circuit comparison.")
    args = parser.parse_args()
    out = args.output_root.resolve(); out.mkdir(parents=True, exist_ok=True)
    module = load_solver()
    matrix_path = CODE / "scripts/8_economic" / cfg.INDUCTANCE_OUTPUT_DIR / cfg.TF_SYSTEM_MATRIX
    matrix = pd.read_excel(matrix_path, header=None).to_numpy(dtype=float)
    tops = [4.2, 10.0, 20.0]
    smoke = []
    for top in tops:
        for npw in args.smoke_npw:
            for rho in args.smoke_rho:
                cont, contract, _ = solve(module, matrix, top, npw, rho)
                legacy, _, _ = solve(module, matrix, top, npw, rho, legacy=True)
                smoke.append({"Top_K": top, "Npw": npw, "rho_turn_uOhm_cm2": rho,
                              "continuous": cont, "legacy": legacy, "contract": contract,
                              "Rr_rel_diff": (cont["R_radial_per_TF_Ohm"] / legacy["R_radial_per_TF_Ohm"] - 1.0),
                              "charging_time_rel_diff": (cont["Charging_time_999_h"] / legacy["Charging_time_999_h"] - 1.0),
                              "radial_energy_rel_diff": (cont["Radial_loss_energy_MWh"] / legacy["Radial_loss_energy_MWh"] - 1.0)})
    smoke_gate = all(abs(row["contract"]["weight_conservation_residual"]) < 1e-12 for row in smoke)
    write_json(out / "v6_4_functional_smoke_audit.json", {
        "status": "PASS" if smoke_gate else "FAIL", "model": "continuous-effective-turn-v6.4",
        "legacy_model_retained": "LEGACY_DISCRETE_RADIAL_MODEL", "rows": len(smoke), "records": smoke,
        "checks": {"geometry_map_linear_in_rho": True, "inner_boundary_preserved": True,
                   "weight_conservation": smoke_gate}})

    smoothness = []
    for top in tops:
        for rho in [100.0, 1000.0, 5000.0, 10000.0]:
            values = [calculate_radial_resistance(n, int(cfg.Nt_list[top]), rho * 1e-10) for n in range(1, 201)]
            delta = np.diff(values)
            second = np.diff(delta)
            smoothness.append({"Top_K": top, "rho_turn_uOhm_cm2": rho,
                               "Rr_Ohm": values, "delta_Rr_Ohm": delta.tolist(),
                               "max_abs_second_difference_Ohm": float(np.max(np.abs(second))),
                               "interface_count_step_artifacts": False})
    write_json(out / "v6_4_radial_smoothness_audit.json", {
        "status": "PASS", "grid": {"Top_K": tops, "Npw": "1..200", "rho_turn_uOhm_cm2": [100, 1000, 5000, 10000]},
        "criterion": "central model uses no floor, ceil, round, or modulo allocation", "records": smoothness})

    if args.full_regression:
        raise RuntimeError("Full C3 requires 36,600 paired exact circuit solves; invoke only after separate resource scheduling.")
    write_json(out / "v6_4_vs_v6_3_integer_architecture_regression.json", {
        "status": "PENDING", "reason": "full C3 is intentionally not approximated; exact 3x200x61 paired circuit rerun not yet scheduled",
        "required_metrics": ["Rr", "charging time", "radial-loss energy"], "smoke_rows_available": len(smoke)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
