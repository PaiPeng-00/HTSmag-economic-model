#!/usr/bin/env python3
"""Evaluate all S1--S3 economics for one V6.2 temperature/coolant block.

Keeping the three scenarios in one interpreter deliberately reuses immutable
physical/circuit and heat-load caches.  Each temperature owns a distinct CSV,
which makes two-process execution safe and resumable.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEVICE = "arc_16pancake_nuc600_v6_2"
COMPACT_COLUMNS = (
    "Top_K,coolant,scenario,Npw,rho_turn_uOhm_cm2,R_joint_nOhm,status,invalid_reason,"
    "TF_system_matrix_file,TF_system_matrix_sha256,Charging_time_999_h,"
    "Mag_loss_at_charge_W,Radial_loss_at_charge_W,Mag_loss_energy_MWh,Radial_loss_energy_MWh,"
    "P_cryo_electric_W,P_cryo_charge_peak_W,Aplant,pulse_duty_factor,CF_gross,r_cryo_re_fraction,E_gross_year_MWh,E_net_year_MWh,"
    "LCOE_plant_USD_per_MWh,CAPEX_mag_installed_USD,Power_supply_cost_USD,"
    "HTS_price_2025USD_per_kAm,E_cryo_year_MWh,energy_closure_error_MWh,"
    "Q_coil_internal_joint_W,Q_pancake_joint_W,Q_nuclear_W,Q_radiation_W,"
    "Q_current_leads_HTS_W,Q_pipes_coolant_W,Q_pipes_aux_W,Q_quench_W,Q_misc_W,"
    "Q_total_Tc_W,Q_current_leads_Cu_conduction_77K_W,Q_current_leads_Cu_joule_77K_W,"
    "Q_total_77K_W,P_cryo_prod_W,P_cryo_dwell_W,P_cryo_static_W,P_cryo_coolwarm_W,"
    "P_cryo_excdec_W,P_cryo_charge_average_W,P_cryo_charge_base_W"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=float, required=True, choices=(4.2, 10.0, 20.0))
    parser.add_argument("--coolant", choices=("He", "H2"), required=True)
    parser.add_argument("--scenario", choices=("S1", "S2", "S3"), default=None,
                        help="Optional scenario subset for an isolated split run.")
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--npw-max", type=int, default=None,
                        help="Benchmark-only inclusive upper Npw bound.")
    parser.add_argument("--rj-count", type=int, default=121,
                        help="Benchmark-only number of logarithmic Rj points (formal V6.2 uses 121).")
    args = parser.parse_args()
    if args.coolant == "H2" and args.top != 20.0:
        raise ValueError("H2 is defined only at 20 K")
    stage = args.stage.resolve()
    cache = args.cache.resolve()
    if not cache.exists():
        raise FileNotFoundError(cache)
    stage.mkdir(parents=True, exist_ok=True)
    label = f"v6_2_b_{args.top:g}K_{args.coolant}"
    if args.scenario is not None:
        label += f"_{args.scenario}"
    output = stage / f"{label}.csv"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}; use a new stage")
    os.environ.update({
        "FUSION_DEVICE": DEVICE,
        "CIRCUIT_SCALAR_CACHE_CSV": str(cache),
        "SCAN_COMPACT_COLUMNS": COMPACT_COLUMNS,
        "SCAN_FLUSH_INTERVAL": "25000",
        "SCAN_PROGRESS_INTERVAL": "50000",
        "SCAN_DEFER_FINALIZE": "1",
        "WRITE_SCAN_XLSX": "0",
    })
    scan_path = Path(__file__).with_name("scan_full_grid.py")
    spec = importlib.util.spec_from_file_location(f"v62b_{args.top:g}_{args.coolant}", scan_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    if module.cfg.DEVICE != DEVICE:
        raise RuntimeError(f"device mismatch: {module.cfg.DEVICE}")
    module.TEMP_COOLANT_PAIRS = [(args.top, args.coolant)]
    module.SCENARIOS = [args.scenario] if args.scenario is not None else ["S1", "S2", "S3"]
    module.NPW_RANGE = np.arange(1, (args.npw_max or 200) + 1, dtype=int)
    module.RHO_TURN_UOHM_CM2 = np.geomspace(10.0, 10000.0, 61)
    if not 1 <= args.rj_count <= 121:
        raise ValueError("rj-count must be within 1..121")
    module.R_JOINT_NOHM = np.geomspace(1.0, 100.0, args.rj_count)
    module.OUTPUT_CSV = output
    module.OUTPUT_XLSX = stage / f"{label}.xlsx"
    module.OUTPUT_MANIFEST = stage / f"{label}.manifest.json"
    expected = len(module.SCENARIOS) * len(module.NPW_RANGE) * len(module.RHO_TURN_UOHM_CM2) * len(module.R_JOINT_NOHM)
    (stage / f"{label}.start.json").write_text(json.dumps({
        "experiment_id": "V6.2-B",
        "status": "RUNNING",
        "device": DEVICE,
        "expected_rows": expected,
        "compact_column_count": len(COMPACT_COLUMNS.split(",")),
        "cache": str(cache),
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2), encoding="utf-8")
    module.main()


if __name__ == "__main__":
    main()
