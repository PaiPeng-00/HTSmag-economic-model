#!/usr/bin/env python3
"""Recompute V7 plant availability directly from a device-frozen circuit cache.

This deliberately performs only the annual schedule calculation.  It contains
no R_j dimension: for a fixed (Top, Npw, rho_turn), the 99.9% charging time is
shared by S1--S3 and is independent of joint resistance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
from fusion_tem import device as cfg
from fusion_tem.economic.plant_availability import allocate_plant_annual_schedule

SCENARIOS = ("S1", "S2", "S3")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def annual_schedule(charge_time_h: float, scenario: str) -> dict[str, float]:
    """Exact V7 plant-availability accounting used by ``compute_case``."""
    ts = cfg.SCENARIO_DEFINITIONS[scenario]
    schedule = allocate_plant_annual_schedule(
        charge_hours=float(charge_time_h),
        major_days_per_fpy=float(ts["major_scheduled_days_per_fpy"]),
        minor_days_per_fpy=float(ts["minor_scheduled_days_per_fpy"]),
        tf_cycles_per_year=float(ts["tf_cycles_per_year"]),
        cooldown_hours_per_cycle=float(ts["tcool_h"]),
        warmup_hours_per_cycle=float(ts["twarm_h"]),
        discharge_to_charge_ratio=float(ts["kdis"]),
        unplanned_unavailability=float(ts["unplanned_unavailability"]),
        pulse_hours_per_cycle=float(ts["tau_pulse_h"]),
        dwell_hours_per_cycle=float(ts["tau_dwell_h"]),
        hours_per_year=float(cfg.HOURS_PER_YEAR),
    )
    return {
        "Aplant": schedule.plant_availability,
        "pulse_duty_factor": schedule.pulse_duty_factor,
        "CF_gross": schedule.gross_capacity_factor,
        "annual_hours_prod": schedule.pulse_hours,
        "annual_hours_dwell": schedule.dwell_hours,
        "annual_hours_core_scheduled_outage": schedule.core_scheduled_outage_hours,
        "annual_hours_tf_planned_outage": schedule.tf_planned_outage_hours,
        "annual_hours_planned_outage": schedule.planned_outage_hours,
        "annual_hours_unplanned_outage_effective": schedule.effective_unplanned_outage_hours,
        "annual_hours_available": schedule.available_hours,
        "annual_hours_coolwarm": schedule.cooldown_warmup_hours,
        "annual_hours_excdec": schedule.charge_discharge_hours,
        "annual_cycle_count": schedule.equivalent_cycles,
        "annual_cycle_count_definition": "continuous equivalent cycles; no integer flooring",
    }


def derive_boundaries(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scenario, top_k, rho), group in frame.groupby(["scenario", "Top_K", "rho_turn_uOhm_cm2"], sort=True):
        group = group.sort_values("Npw")
        eligible = group["Aplant"] >= 0.80
        first = group.loc[eligible].head(1)
        rows.append({
            "scenario": scenario,
            "Top_K": top_k,
            "rho_turn_uOhm_cm2": rho,
            "Npw_min_Aplant080": int(first["Npw"].iat[0]) if len(first) else np.nan,
            "has_Aplant080": bool(len(first)),
            "Aplant_at_Npw200": float(group.loc[group["Npw"].idxmax(), "Aplant"]),
            "boundary_definition": "smallest_direct_Npw_with_Aplant_ge_0.80; no interpolation",
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--experiment-id", default="V6.1-A-availability")
    args = parser.parse_args()
    if cfg.DEVICE != args.device.lower():
        raise RuntimeError(f"device mismatch: requested {args.device!r}, loaded {cfg.DEVICE!r}")
    cache = args.cache.resolve()
    needed = ["Top_K", "Npw", "rho_turn_uOhm_cm2", "Charging_time_999_h", "circuit_solver_status"]
    raw = pd.read_csv(cache, usecols=needed + (["device"] if "device" in pd.read_csv(cache, nrows=0).columns else []))
    missing = set(needed) - set(raw.columns)
    if missing:
        raise ValueError(f"cache missing required columns: {sorted(missing)}")
    if (raw["circuit_solver_status"] != "success").any():
        raise ValueError("cache includes non-success circuit points")
    if "device" not in raw.columns:
        raise ValueError("cache has no device column; it is not identity-auditable")
    if set(raw["device"].astype(str).str.lower()) != {cfg.DEVICE}:
        raise ValueError("cache device column does not match requested device")
    key = ["Top_K", "Npw", "rho_turn_uOhm_cm2"]
    if raw.duplicated(key).any() or not np.isfinite(raw["Charging_time_999_h"]).all():
        raise ValueError("cache keys or charge times are invalid")

    rows: list[dict] = []
    for item in raw.itertuples(index=False):
        for scenario in SCENARIOS:
            schedule = annual_schedule(float(item.Charging_time_999_h), scenario)
            rows.append({
                "scenario": scenario,
                "Top_K": float(item.Top_K),
                "Npw": int(item.Npw),
                "rho_turn_uOhm_cm2": float(item.rho_turn_uOhm_cm2),
                "Charging_time_999_h": float(item.Charging_time_999_h),
                "device": cfg.DEVICE,
                **schedule,
            })
    result = pd.DataFrame(rows)
    maxima = result.groupby("scenario", as_index=False)["Aplant"].max().rename(columns={"Aplant": "Aplant_max_scenario"})
    boundaries = derive_boundaries(result)

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    stem = args.run_label
    grid_path = out / f"{stem}_availability_grid.csv"
    max_path = out / f"{stem}_Aplant_max_by_scenario.csv"
    boundary_path = out / f"{stem}_Aplant080_boundaries.csv"
    result.sort_values(["scenario", "Top_K", "Npw", "rho_turn_uOhm_cm2"]).to_csv(grid_path, index=False)
    maxima.to_csv(max_path, index=False)
    boundaries.to_csv(boundary_path, index=False)
    audit = {
        "experiment_id": "V7-A-plant-availability" if args.experiment_id == "V6.1-A-availability" else args.experiment_id,
        "run_label": stem,
        "status": "PASS",
        "device": cfg.DEVICE,
        "cache": str(cache),
        "cache_sha256": sha256(cache),
        "cache_rows": int(len(raw)),
        "availability_rows": int(len(result)),
        "Aplant_max_scenario": {x.scenario: float(x.Aplant_max_scenario) for x in maxima.itertuples(index=False)},
        "annual_schedule_source": "fusion_tem.economic.plant_availability.allocate_plant_annual_schedule",
        "annual_schedule_contract": "planned core plus annual-equivalent TF outage; 7% unplanned overlap; pulse and dwell both available",
        "Aplant_requirement": 0.80,
        "Aplant_units": "fraction",
        "outputs": [str(grid_path), str(max_path), str(boundary_path)],
    }
    (out / f"{stem}_availability_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
