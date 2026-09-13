"""S2 steady-state temperature extension for the ARC TF magnet.

Scope fixed by the manuscript decision on 2026-07-16:
  * coolant: He at 10 bar;
  * Top: 4.2, 10, 15, 20, 25, 30, 35 and 40 K;
  * temperature-specific Ip/Nt from the authoritative Nc=16 field model;
  * optimize Npw and pancake-to-pancake joint resistance;
  * use only steady heat loads in the cryogenic model;
  * do not scan turn-to-turn resistivity, T-A charging loss, charge time or
    thermal-stability metrics.

The output delta LCOE is referenced to the global feasible minimum across all
eight temperatures after optimizing Npw and joint resistance.  Feasibility in
this dedicated screen uses only the existing <=50% cryogenic recirculating-
power criterion; charge-time feasibility is intentionally out of scope.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve()
CODE_ROOT = SCRIPT.parents[2]
CODE_MODEL_ROOT = SCRIPT.parents[3]
WORKSPACE_ROOT = SCRIPT.parents[4]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fusion_tem import device as cfg  # noqa: E402
from fusion_tem.economic.lcoe import compute_case, define_parameters  # noqa: E402
from fusion_tem.economic.price_basis import (  # noqa: E402
    HTS_PRICE_2025_USD_PER_KAM_BY_SCENARIO,
)


TEMPERATURES_K = [4.2, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
HE_PRESSURE_BAR = 10.0
EXPECTED_DEVICE = "arc_16pancake_nuc600"
COMPARISON_SCENARIOS = ("S4", "S5", "S6")
EXPECTED_HTS_PRICE_USD_PER_KAM = {
    key: HTS_PRICE_2025_USD_PER_KAM_BY_SCENARIO[key]
    for key in COMPARISON_SCENARIOS
}
FIXED_COMPARISON_NPW = 30
FIXED_COMPARISON_RJ_NOHM = 10.0
HE_DENSITY_KG_M3_AT_10_BAR = {
    4.2: 150.47680,
    10.0: 61.055567,
    15.0: 34.015441,
    20.0: 24.253629,
    25.0: 19.054414,
    30.0: 15.763527,
    35.0: 13.472323,
    40.0: 11.777181,
}
NIST_QUERY_URL = (
    "https://webbook.nist.gov/cgi/fluid.cgi?Action=Data&Wide=on&ID=C7440597"
    "&Type=IsoBar&Digits=8&P=10&THigh=40&TLow=4.2&TInc=0.1&RefState=DEF"
    "&TUnit=K&PUnit=bar&DUnit=kg%2Fm3&HUnit=kJ%2Fkg&WUnit=m%2Fs"
    "&VisUnit=uPa%2As&STUnit=N%2Fm"
)

DEFAULT_OPERATING_POINTS = (
    WORKSPACE_ROOT / "ARC-model" / "results" / "operating_points_Nc16_8T.csv"
)
DEFAULT_OUTPUT_DIR = cfg.OUTPUTS_TABLES_DIR / "s2_steady_temperature_extension_8T"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_operating_points(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Operating-point file not found: {path}\n"
            "Run ARC-model/comsol/1-field-FEM/resolve_operating_points_8T.m first."
        )
    df = pd.read_csv(path)
    required = {
        "Top_K",
        "Bmax_T",
        "minJc_A_per_mm2",
        "Ic_tape_A",
        "Nt_design",
        "Ip_design_A",
        "load_factor",
        "ampturns_MA",
        "dr_TF_m",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Operating-point file is missing columns: {sorted(missing)}")

    df = df.copy()
    df["Top_K"] = df["Top_K"].astype(float)
    df["Nt_design"] = df["Nt_design"].round().astype(int)
    if sorted(df["Top_K"].tolist()) != sorted(TEMPERATURES_K):
        raise ValueError(
            f"Expected temperatures {TEMPERATURES_K}, got {df['Top_K'].tolist()}"
        )

    known = df.set_index("Top_K").loc[[4.2, 10.0, 20.0], "Nt_design"].tolist()
    if known != [700, 850, 1150]:
        raise ValueError(
            f"Nc=16 operating-point regression failed: expected [700, 850, 1150], got {known}"
        )
    if not np.allclose(df["ampturns_MA"], 8.4, rtol=0.0, atol=0.002):
        raise ValueError("Operating points do not preserve 8.4 MA-turn within 0.002 MA")
    if not np.allclose(df["dr_TF_m"], 0.64, rtol=0.0, atol=1e-6):
        raise ValueError("Operating points do not preserve dr_TF=0.64 m")
    return df.sort_values("Top_K").reset_index(drop=True)


def install_temperature_points(op: pd.DataFrame) -> dict:
    """Install the eight operating points in memory without changing core files."""

    ip = {float(row.Top_K): float(row.Ip_design_A) for row in op.itertuples()}
    nt = {float(row.Top_K): int(row.Nt_design) for row in op.itertuples()}

    cfg.Ip_list = ip
    cfg.Nt_list = nt
    cfg.L_HTS_TF_m = {
        top: turns * cfg.L_TAPE_PER_TURN * cfg.NP for top, turns in nt.items()
    }
    cfg.L_HTS_SYSTEM_km = {
        top: length_m * cfg.Ntf * 1e-3
        for top, length_m in cfg.L_HTS_TF_m.items()
    }
    cfg.kAm_HTS_TAPE_LIST = {
        top: cfg.Ic0 * length_km
        for top, length_km in cfg.L_HTS_SYSTEM_km.items()
    }

    params = define_parameters()
    params["tech_scenarios"] = list(COMPARISON_SCENARIOS)
    params["temperatures_K"] = list(TEMPERATURES_K)
    params["coolants"] = ["He"]
    params["current_by_temperature"] = dict(ip)
    params["hts_kAm_by_temperature"] = dict(cfg.kAm_HTS_TAPE_LIST)
    params["coolant_density_kg_m3"]["He"] = dict(HE_DENSITY_KG_M3_AT_10_BAR)
    return params


def validate_price_only_comparison(params: dict) -> None:
    """Require S4/S5/S6 to differ from S2 only in HTS conductor price."""

    s2_years = int(params["tech_scenario_to_years"]["S2"])
    s2_schedule = params["time_scenarios"]["S2"]
    s2_c0 = float(params["C0_nonmagnet_USD_by_scenario"]["S2"])
    s2_coolant_prices = params["coolant_price_per_kg_by_scenario"]["S2"]
    for scenario in COMPARISON_SCENARIOS:
        if int(params["tech_scenario_to_years"][scenario]) != s2_years:
            raise ValueError(f"{scenario} lifetime differs from S2")
        if params["time_scenarios"][scenario] != s2_schedule:
            raise ValueError(f"{scenario} operating schedule differs from S2")
        if not np.isclose(
            float(params["C0_nonmagnet_USD_by_scenario"][scenario]), s2_c0
        ):
            raise ValueError(f"{scenario} C0 differs from S2")
        if params["coolant_price_per_kg_by_scenario"][scenario] != s2_coolant_prices:
            raise ValueError(f"{scenario} coolant prices differ from S2")
        observed_price = float(params["hts_price_per_kAm_by_scenario"][scenario])
        if not np.isclose(observed_price, EXPECTED_HTS_PRICE_USD_PER_KAM[scenario]):
            raise ValueError(
                f"{scenario} HTS price is {observed_price}, expected "
                f"{EXPECTED_HTS_PRICE_USD_PER_KAM[scenario]}"
            )


def scan(op: pd.DataFrame, params: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    op_by_t = op.set_index("Top_K")
    npw_values = np.unique(np.asarray(cfg.NPW_SCAN_VALUES, dtype=int))
    r_joint_nohm_values = np.asarray(cfg.R_JOINT_SCAN_VALUES, dtype=float)
    rows: list[dict] = []
    project_years = int(params["tech_scenario_to_years"]["S2"])

    for top in TEMPERATURES_K:
        point = op_by_t.loc[top]
        for npw in npw_values:
            for r_joint_nohm in r_joint_nohm_values:
                result, heat = compute_case(
                    tech_scenario="S2",
                    year=project_years,
                    temperature_K=top,
                    coolant="He",
                    Npw=int(npw),
                    R_p2p_joint=float(r_joint_nohm) * 1e-9,
                    p=params,
                )
                lcoe = result.get("lcoe_plant_$/MWh", np.nan)
                r_cryo = result.get("r_parasitic_pct", np.nan)
                feasible = bool(
                    np.isfinite(lcoe)
                    and np.isfinite(r_cryo)
                    and r_cryo <= cfg.DELTA_LCOE_MASK_PARASITIC_PCT
                )
                rows.append(
                    {
                        "scenario": "S2",
                        "Top_K": top,
                        "coolant": "He",
                        "pressure_bar": HE_PRESSURE_BAR,
                        "He_density_kg_m3": HE_DENSITY_KG_M3_AT_10_BAR[top],
                        "Bmax_T": float(point["Bmax_T"]),
                        "minJc_A_per_mm2": float(point["minJc_A_per_mm2"]),
                        "Ic_tape_A": float(point["Ic_tape_A"]),
                        "Nt_design": int(point["Nt_design"]),
                        "Ip_A": float(point["Ip_design_A"]),
                        "load_factor": float(point["load_factor"]),
                        "ampturns_MA": float(point["ampturns_MA"]),
                        "dr_TF_m": float(point["dr_TF_m"]),
                        "Npw": int(npw),
                        "R_joint_nOhm": float(r_joint_nohm),
                        "fixed_charge_hours_h": float(cfg.charge_hours_max),
                        "LCOE_plant_USD_per_MWh": lcoe,
                        "LCOE_C0_USD_per_MWh": result.get("lcoe_C0_$/MWh", np.nan),
                        "r_cryo_re_pct": r_cryo,
                        "P_cryo_prod_W": result.get("cryo_power_prod_W", np.nan),
                        "P_cryo_dwell_W": result.get("cryo_power_dwell_W", np.nan),
                        "P_cryo_static_W": result.get("cryo_power_static_W", np.nan),
                        "E_cryo_year_MWh": result.get("cryo_energy_annual_MWh", np.nan),
                        "E_net_year_MWh": result.get("net_energy_annual_MWh", np.nan),
                        "Tape_cost_USD": result.get("tape_cost_$", np.nan),
                        "Coolant_fill_cost_USD": result.get("coolant_fill_cost_$", np.nan),
                        "Power_supply_cost_USD": result.get("power_supply_cost_$", np.nan),
                        "heat_internal_joint_W_per_TF": heat.get("coil_internal_joint", np.nan),
                        "heat_pancake_joint_W_per_TF": heat.get("pancake_joint", np.nan),
                        "heat_nuclear_W_per_TF": heat.get("nuclear", np.nan),
                        "heat_radiation_W_per_TF": heat.get("radiation", np.nan),
                        "heat_current_lead_HTS_W_per_TF": heat.get("current_leads_HTS", np.nan),
                        "heat_coolant_pipe_W_per_TF": heat.get("pipes_coolant", np.nan),
                        "heat_aux_pipe_W_per_TF": heat.get("pipes_aux", np.nan),
                        "feasible_r_cryo_only": feasible,
                    }
                )

    grid = pd.DataFrame(rows)
    feasible = grid[grid["feasible_r_cryo_only"]].copy()
    if feasible.empty:
        raise RuntimeError("No point satisfies the cryogenic recirculating-power criterion")

    global_min = float(feasible["LCOE_plant_USD_per_MWh"].min())
    grid["delta_LCOE_global_USD_per_MWh"] = (
        grid["LCOE_plant_USD_per_MWh"] - global_min
    )

    idx = feasible.groupby("Top_K")["LCOE_plant_USD_per_MWh"].idxmin()
    summary = grid.loc[idx].sort_values("Top_K").reset_index(drop=True)
    summary["within_delta10_USD_per_MWh"] = (
        summary["delta_LCOE_global_USD_per_MWh"] <= 10.0 + 1e-12
    )
    summary["global_reference_LCOE_USD_per_MWh"] = global_min
    return grid, summary


def compare_scenarios_fixed_design(op: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Evaluate S4/S5/S6 at one common design, changing only HTS price."""

    validate_price_only_comparison(params)
    op_by_t = op.set_index("Top_K")
    rows: list[dict] = []
    for scenario in COMPARISON_SCENARIOS:
        project_years = int(params["tech_scenario_to_years"][scenario])
        c0 = float(params["C0_nonmagnet_USD_by_scenario"][scenario])
        hts_price = float(params["hts_price_per_kAm_by_scenario"][scenario])
        he_price = float(params["coolant_price_per_kg_by_scenario"][scenario]["He"])
        schedule = params["time_scenarios"][scenario]
        for top in TEMPERATURES_K:
            point = op_by_t.loc[top]
            result, _ = compute_case(
                tech_scenario=scenario,
                year=project_years,
                temperature_K=top,
                coolant="He",
                Npw=FIXED_COMPARISON_NPW,
                R_p2p_joint=FIXED_COMPARISON_RJ_NOHM * 1e-9,
                p=params,
            )
            lcoe = result.get("lcoe_plant_$/MWh", np.nan)
            r_cryo = result.get("r_parasitic_pct", np.nan)
            feasible = bool(
                np.isfinite(lcoe)
                and np.isfinite(r_cryo)
                and r_cryo <= cfg.DELTA_LCOE_MASK_PARASITIC_PCT
            )
            rows.append(
                {
                    "scenario": scenario,
                    "Top_K": top,
                    "coolant": "He",
                    "pressure_bar": HE_PRESSURE_BAR,
                    "He_density_kg_m3": HE_DENSITY_KG_M3_AT_10_BAR[top],
                    "project_lifetime_years": project_years,
                    "C0_nonmagnet_USD": c0,
                    "HTS_price_USD_per_kAm": hts_price,
                    "He_price_USD_per_kg": he_price,
                    "discount_rate": float(params["discount_rate"]),
                    "tmaint_h": float(schedule["tmaint_h"]),
                    "nmaint": float(schedule["nmaint"]),
                    "tcool_h": float(schedule["tcool_h"]),
                    "twarm_h": float(schedule["twarm_h"]),
                    "kdis": float(schedule["kdis"]),
                    "tau_pulse_h": float(schedule["tau_pulse_h"]),
                    "tau_dwell_h": float(schedule["tau_dwell_h"]),
                    "Npw": FIXED_COMPARISON_NPW,
                    "R_joint_nOhm": FIXED_COMPARISON_RJ_NOHM,
                    "Nt_design": int(point["Nt_design"]),
                    "Ip_A": float(point["Ip_design_A"]),
                    "load_factor": float(point["load_factor"]),
                    "LCOE_plant_USD_per_MWh": lcoe,
                    "LCOE_C0_USD_per_MWh": result.get("lcoe_C0_$/MWh", np.nan),
                    "r_cryo_re_pct": r_cryo,
                    "E_net_year_MWh": result.get("net_energy_annual_MWh", np.nan),
                    "Tape_cost_USD": result.get("tape_cost_$", np.nan),
                    "Coolant_fill_cost_USD": result.get("coolant_fill_cost_$", np.nan),
                    "Power_supply_cost_USD": result.get("power_supply_cost_$", np.nan),
                    "feasible_r_cryo_only": feasible,
                }
            )

    comparison = pd.DataFrame(rows).sort_values(["scenario", "Top_K"]).reset_index(drop=True)
    if len(comparison) != len(COMPARISON_SCENARIOS) * len(TEMPERATURES_K):
        raise RuntimeError("Fixed-design scenario comparison row count is incomplete")
    if not comparison["feasible_r_cryo_only"].all():
        failed = comparison.loc[
            ~comparison["feasible_r_cryo_only"], ["scenario", "Top_K"]
        ].to_dict("records")
        raise RuntimeError(f"Fixed-design scenario comparison contains infeasible points: {failed}")
    comparison["scenario_fixed_design_min_LCOE_USD_per_MWh"] = comparison.groupby(
        "scenario"
    )["LCOE_plant_USD_per_MWh"].transform("min")
    comparison["delta_LCOE_within_scenario_fixed_design_USD_per_MWh"] = (
        comparison["LCOE_plant_USD_per_MWh"]
        - comparison["scenario_fixed_design_min_LCOE_USD_per_MWh"]
    )
    comparison["within_delta10_fixed_design_USD_per_MWh"] = (
        comparison["delta_LCOE_within_scenario_fixed_design_USD_per_MWh"]
        <= 10.0 + 1e-12
    )
    return comparison

def write_outputs(
    operating_points_path: Path,
    op: pd.DataFrame,
    grid: pd.DataFrame,
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    grid_path = output_dir / "s2_steady_temperature_full_grid.csv"
    summary_path = output_dir / "s2_steady_temperature_optimized.csv"
    comparison_path = output_dir / "s4_s5_s6_fixed_design_Npw30_Rj10.csv"
    metadata_path = output_dir / "run_metadata.json"
    grid.to_csv(grid_path, index=False)
    summary.to_csv(summary_path, index=False)
    comparison.to_csv(comparison_path, index=False)

    accepted = summary.loc[summary["within_delta10_USD_per_MWh"], "Top_K"].tolist()
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "S2 steady-state temperature grid plus S4/S5/S6 HTS-price-only fixed-design comparison",
        "device": cfg.DEVICE,
        "temperature_K": TEMPERATURES_K,
        "coolant": "He",
        "pressure_bar": HE_PRESSURE_BAR,
        "density_kg_m3": HE_DENSITY_KG_M3_AT_10_BAR,
        "density_source": "NIST Chemistry WebBook SRD 69 isobar query",
        "density_source_url": NIST_QUERY_URL,
        "operating_points_file": str(operating_points_path),
        "operating_points_sha256": sha256(operating_points_path),
        "known_point_regression": "Nt(4.2,10,20 K)=700,850,1150",
        "scenario_grid": "S2",
        "comparison_scenarios": list(COMPARISON_SCENARIOS),
        "comparison_scenario_parameters": {
            str(row.scenario): {
                "project_lifetime_years": int(row.project_lifetime_years),
                "C0_nonmagnet_USD": float(row.C0_nonmagnet_USD),
                "HTS_price_USD_per_kAm": float(row.HTS_price_USD_per_kAm),
                "He_price_USD_per_kg": float(row.He_price_USD_per_kg),
                "operating_schedule": {
                    "tmaint_h": float(row.tmaint_h),
                    "nmaint": float(row.nmaint),
                    "tcool_h": float(row.tcool_h),
                    "twarm_h": float(row.twarm_h),
                    "kdis": float(row.kdis),
                    "tau_pulse_h": float(row.tau_pulse_h),
                    "tau_dwell_h": float(row.tau_dwell_h),
                },
            }
            for row in comparison.drop_duplicates("scenario").itertuples()
        },
        "comparison_control_definition": "S4/S5/S6 inherit all S2 inputs except HTS conductor price = 100/50/10 constant-2025-US$/(kA m)",
        "fixed_comparison_design": {
            "Npw": FIXED_COMPARISON_NPW,
            "R_joint_nOhm": FIXED_COMPARISON_RJ_NOHM,
        },
        "fixed_comparison_delta_reference": "each HTS-price case's own minimum over the eight temperatures at Npw=30, R_joint=10 nOhm",
        "fixed_charge_hours_h": float(cfg.charge_hours_max),
        "optimization_variables": ["Npw", "R_joint_nOhm"],
        "turn_to_turn_resistivity_scanned": False,
        "TA_charging_loss_included": False,
        "charge_time_feasibility_included": False,
        "thermal_stability_metric_included": False,
        "feasibility_criterion": "r_cryo_re_pct <= 50",
        "delta_LCOE_reference": "global feasible minimum across eight temperatures after Npw/R_joint optimization",
        "delta_LCOE_acceptance_USD_per_MWh": 10.0,
        "accepted_discrete_temperature_points_K": accepted,
        "grid_rows": int(len(grid)),
        "fixed_design_comparison_rows": int(len(comparison)),
        "output_files": {
            "full_grid": str(grid_path),
            "optimized_by_temperature": str(summary_path),
            "fixed_design_scenario_comparison": str(comparison_path),
        },
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(summary[
        [
            "Top_K",
            "Nt_design",
            "Ip_A",
            "load_factor",
            "Npw",
            "R_joint_nOhm",
            "LCOE_plant_USD_per_MWh",
            "delta_LCOE_global_USD_per_MWh",
            "r_cryo_re_pct",
            "within_delta10_USD_per_MWh",
        ]
    ].to_string(index=False))
    print(f"saved: {grid_path}")
    print(f"saved: {summary_path}")
    print(comparison[[
        "scenario", "Top_K", "Npw", "R_joint_nOhm",
        "LCOE_plant_USD_per_MWh",
        "delta_LCOE_within_scenario_fixed_design_USD_per_MWh",
    ]].to_string(index=False))
    print(f"saved: {comparison_path}")
    print(f"saved: {metadata_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--operating-points", type=Path, default=DEFAULT_OPERATING_POINTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    if cfg.DEVICE != EXPECTED_DEVICE:
        raise RuntimeError(
            f"This manuscript scan requires FUSION_DEVICE={EXPECTED_DEVICE}; "
            f"got {cfg.DEVICE}."
        )
    args = parse_args()
    op = load_operating_points(args.operating_points.resolve())
    params = install_temperature_points(op)
    grid, summary = scan(op, params)
    comparison = compare_scenarios_fixed_design(op, params)
    write_outputs(
        args.operating_points.resolve(), op, grid, summary, comparison,
        args.output_dir.resolve(),
    )


if __name__ == "__main__":
    main()
