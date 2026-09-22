#!/usr/bin/env python3
"""Build affected V10 figure inputs from the candidate ledgers and raw scan."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "model" / "model_code" / "code"
sys.path.insert(0, str(CODE / "src"))
os.environ.setdefault("FUSION_DEVICE", "arc_16pancake_nuc600_v6_2")

from fusion_tem import device as cfg  # noqa: E402
from fusion_tem.economic.capital_anchor import (  # noqa: E402
    COOLANT_REPLENISH_FRACTION,
    CORE_VOM_BY_SCENARIO,
    DISCOUNT_RATE_BY_SCENARIO,
    M_COOLANT_FILL,
    M_HTS,
    M_PS,
    PCS_VOM_USD_PER_MWH_E,
    POWER_SUPPLY_PRICE_USD_PER_A,
    PROJECT_YEARS_BY_SCENARIO,
    annual_pcs_fom_usd,
    build_capital_anchor,
    capital_recovery_factor,
)


SCENARIOS = ("S1", "S2", "S3")
KEY3 = ["Npw", "rho_turn_uOhm_cm2", "R_joint_nOhm"]
RAW_SMALL = [
    "Top_K", "coolant", "scenario", *KEY3,
    "Q_coil_internal_joint_W", "Q_pancake_joint_W", "Q_nuclear_W",
    "Q_radiation_W", "Q_pipes_coolant_W", "Q_pipes_aux_W",
    "Q_quench_W", "Q_misc_W", "Q_total_Tc_W",
    "Tape_cost_USD", "HTS_price_2025USD_per_kAm",
    "Power_supply_cost_USD", "Coolant_price_2025USD_per_kg",
    "Coolant_fill_cost_USD", "E_fusion_th_year_MWh",
    "gross_energy_annual_MWh", "net_energy_annual_MWh",
]


def scan_small(raw: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    heat_parts: list[pd.DataFrame] = []
    price_parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(raw, usecols=RAW_SMALL, chunksize=200_000, low_memory=False):
        heat = (
            np.isclose(chunk["Top_K"], 10.0)
            & chunk["coolant"].eq("He")
            & np.isclose(chunk["rho_turn_uOhm_cm2"], 10_000.0)
        )
        if heat.any():
            heat_parts.append(chunk.loc[heat].copy())
        price = (
            chunk["scenario"].eq("S2")
            & chunk["coolant"].eq("He")
            & chunk["Npw"].eq(200)
            & np.isclose(chunk["R_joint_nOhm"], 1.0)
        )
        if price.any():
            price_parts.append(chunk.loc[price].copy())
    heats = pd.concat(heat_parts, ignore_index=True)
    prices = pd.concat(price_parts, ignore_index=True)
    if len(heats) != 200 * 121 * 3 or len(prices) != 3 * 61:
        raise RuntimeError(f"STOP_SMALL_RAW_CONTRACT: heat={len(heats)} price={len(prices)}")
    return heats, prices


def build_heat_panel(raw_heat: pd.DataFrame) -> pd.DataFrame:
    source_cols = [
        "Q_coil_internal_joint_W", "Q_pancake_joint_W", "Q_nuclear_W",
        "Q_radiation_W", "Q_pipes_coolant_W", "Q_pipes_aux_W",
        "Q_quench_W", "Q_misc_W",
    ]
    groups = raw_heat.groupby(["Top_K", "coolant", "rho_turn_uOhm_cm2", "Npw", "R_joint_nOhm"], sort=True)
    if not groups.size().eq(3).all():
        raise RuntimeError("STOP_HEAT_SCENARIO_REPLICA_COUNT")
    if any(not groups[column].nunique(dropna=False).eq(1).all() for column in source_cols):
        raise RuntimeError("STOP_HEAT_SCENARIO_DRIFT")
    panel = groups[source_cols].first().reset_index()
    panel["scenario_replica_count"] = 3
    rename = {column: column.replace("_W", "_W_per_TF") for column in source_cols}
    panel = panel.rename(columns=rename)
    panel["Q_joint_W_per_TF"] = (
        panel["Q_coil_internal_joint_W_per_TF"] + panel["Q_pancake_joint_W_per_TF"]
    )
    panel["Q_background_W_per_TF"] = panel[
        ["Q_radiation_W_per_TF", "Q_pipes_coolant_W_per_TF", "Q_pipes_aux_W_per_TF",
         "Q_quench_W_per_TF", "Q_misc_W_per_TF"]
    ].sum(axis=1)
    panel["Q_total_pulse_W_per_TF"] = (
        panel["Q_joint_W_per_TF"]
        + panel["Q_nuclear_W_per_TF"]
        + panel["Q_background_W_per_TF"]
    )
    order = [
        "Top_K", "coolant", "rho_turn_uOhm_cm2", "Npw", "R_joint_nOhm",
        "Q_coil_internal_joint_W_per_TF", "scenario_replica_count",
        "Q_pancake_joint_W_per_TF", "Q_joint_W_per_TF", "Q_nuclear_W_per_TF",
        "Q_radiation_W_per_TF", "Q_pipes_coolant_W_per_TF", "Q_pipes_aux_W_per_TF",
        "Q_quench_W_per_TF", "Q_misc_W_per_TF", "Q_background_W_per_TF",
        "Q_total_pulse_W_per_TF",
    ]
    panel = panel[order]
    expected_nuclear = float(cfg.V_mag * cfg.NUCLEAR_POWER_DENSITY)
    if not np.allclose(panel["Q_nuclear_W_per_TF"], expected_nuclear):
        raise RuntimeError("STOP_NUCLEAR_PANEL_DRIFT")
    residual = (
        panel["Q_total_pulse_W_per_TF"]
        - panel["Q_joint_W_per_TF"]
        - panel["Q_nuclear_W_per_TF"]
        - panel["Q_background_W_per_TF"]
    ).abs().max()
    if float(residual) > 1e-9:
        raise RuntimeError(f"STOP_HEAT_COMPONENT_CLOSURE: {residual}")
    return panel


def ledger_paths(candidate: Path) -> list[Path]:
    folder = candidate / "full_model_dataset"
    paths = [
        folder / "realizations_T4p2_He.parquet",
        folder / "realizations_T10p0_He.parquet",
        folder / "realizations_T20p0_He.parquet",
        folder / "realizations_T20p0_H2.parquet",
    ]
    if any(not path.is_file() for path in paths):
        raise RuntimeError("STOP_MISSING_LEDGER")
    return paths


def refrigeration_statistics(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_parquet(path) for path in paths]
    rows = []
    for scenario in SCENARIOS:
        for top in (4.2, 10.0, 20.0):
            selected = [frame for frame in frames if np.isclose(float(frame["Top_K"].iloc[0]), top)]
            values = []
            n = 0
            coolants = []
            for frame in selected:
                # Figure 3D uses the same common-availability realization set
                # for all three scenarios, matching the manuscript population.
                mask = frame["availability_pass_all"]
                part = 100.0 * frame.loc[mask, f"r_cryo_re_fraction_{scenario}"].to_numpy(float)
                part = part[np.isfinite(part)]
                values.append(part)
                n += int(mask.sum())
                coolants.append(str(frame["coolant"].iloc[0]))
            pooled = np.concatenate(values)
            q = np.quantile(pooled, [0, .1, .5, .9, 1.0])
            rows.append({
                "group": f"YD-{scenario}-{top:g}", "scenario": scenario, "Top_K": top,
                "coolants": "+".join(coolants), "N": n,
                **{name: float(value) for name, value in zip(
                    ("min_pct", "P10_pct", "median_pct", "P90_pct", "max_pct"), q
                )},
            })
    return pd.DataFrame(rows)


def crossings(r: np.ndarray, delta: np.ndarray, availability: np.ndarray,
              energy: np.ndarray, valid: np.ndarray, eps: float) -> pd.DataFrame:
    passed = availability & energy & valid & (delta <= eps)
    prefix = np.logical_and.accumulate(passed, axis=1)
    n = prefix.sum(axis=1)
    idx = np.arange(len(n)); low = np.maximum(n - 1, 0); high = np.minimum(n, len(r) - 1)
    dlo, dhi = delta[idx, low], delta[idx, high]
    bracket = (n > 0) & (n < len(r))
    interpolable = (
        bracket & availability[idx, high] & energy[idx, high] & valid[idx, high]
        & np.isfinite(dlo) & np.isfinite(dhi) & (dlo <= eps) & (dhi > eps)
    )
    estimate = np.full(len(n), np.nan)
    estimate[n == len(r)] = r[-1]
    estimate[interpolable] = (
        r[low[interpolable]]
        + (eps - dlo[interpolable]) / (dhi[interpolable] - dlo[interpolable])
        * (r[high[interpolable]] - r[low[interpolable]])
    )
    return pd.DataFrame({
        "Rj_tol_nOhm": estimate, "upper_censored": n == len(r),
        "nonmonotonic": np.any(passed & ~prefix, axis=1),
        "r_lo_nOhm": np.where(n > 0, r[low], np.nan),
        "r_hi_nOhm": np.where(n < len(r), r[high], np.nan),
        "delta_lo": np.where(n > 0, dlo, np.nan),
        "delta_hi": np.where(n < len(r), dhi, np.nan),
    })


def fig5_from_ledgers(paths: list[Path], baselines: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    heatmaps, scenario_crossings = [], []
    for path in paths:
        identity = pd.read_parquet(path, columns=["Top_K", "coolant"])
        if str(identity.coolant.iloc[0]) != "He":
            continue
        columns = KEY3 + ["Top_K", "coolant"]
        for scenario in SCENARIOS:
            columns += [f"Aplant_{scenario}", f"E_net_year_MWh_{scenario}",
                        f"lcoe_anchor_USD_per_MWh_{scenario}"]
        frame = pd.read_parquet(path, columns=columns).sort_values(KEY3).reset_index(drop=True)
        top = float(frame.Top_K.iloc[0])
        selected = frame.Npw.isin([50, 200])
        heat = frame.loc[selected, KEY3].copy()
        delta_cols, availability_cols, valid_cols = [], [], []
        for scenario in SCENARIOS:
            lcoe = frame.loc[selected, f"lcoe_anchor_USD_per_MWh_{scenario}"]
            delta_cols.append(lcoe.to_numpy(float) / baselines[scenario] - 1.0)
            availability_cols.append(frame.loc[selected, f"Aplant_{scenario}"].to_numpy(float) >= .8)
            valid_cols.append(
                (frame.loc[selected, f"E_net_year_MWh_{scenario}"].to_numpy(float) > 0)
                & np.isfinite(lcoe.to_numpy(float)) & (lcoe.to_numpy(float) > 0)
            )
        heat["Top_K"] = top
        heat["delta_max_pct"] = 100.0 * np.column_stack(delta_cols).max(axis=1)
        heat["all_availability"] = np.column_stack(availability_cols).all(axis=1)
        heat["all_valid_economics"] = np.column_stack(valid_cols).all(axis=1)
        heatmaps.append(heat)

        r = np.sort(frame.R_joint_nOhm.unique())
        base = frame.drop_duplicates(KEY3[:2])[KEY3[:2]].reset_index(drop=True)
        for scenario in SCENARIOS:
            a = frame[f"Aplant_{scenario}"].to_numpy(float).reshape(-1, len(r))
            energy = frame[f"E_net_year_MWh_{scenario}"].to_numpy(float).reshape(-1, len(r))
            lcoe = frame[f"lcoe_anchor_USD_per_MWh_{scenario}"].to_numpy(float).reshape(-1, len(r))
            delta = lcoe / baselines[scenario] - 1.0
            for eps in (.01, .05, .10, .20):
                part = crossings(r, delta, a >= .8, energy > 0, np.isfinite(lcoe) & (lcoe > 0), eps)
                part = pd.concat([base, part], axis=1)
                part["Top_K"] = top; part["scenario"] = scenario; part["epsilon"] = eps
                scenario_crossings.append(part.loc[part.Npw.ge(20)])
    heat_all = pd.concat(heatmaps, ignore_index=True)
    heat = heat_all.groupby(["Npw", "Top_K", "R_joint_nOhm"], as_index=False).agg(
        delta_max_pct=("delta_max_pct", "max"),
        all_availability=("all_availability", "all"),
        all_valid_economics=("all_valid_economics", "all"),
        N_rho=("rho_turn_uOhm_cm2", "size"),
    )
    if len(heat) != 2 * 3 * 121 or not heat.N_rho.eq(61).all():
        raise RuntimeError("STOP_FIG5_HEATMAP_CONTRACT")
    curves = pd.concat(scenario_crossings, ignore_index=True)
    individual_rows = []
    for key, group in curves.groupby(["Npw", "Top_K", "rho_turn_uOhm_cm2", "epsilon"], sort=True):
        valid = group.Rj_tol_nOhm.notna().all()
        controlling = group.loc[group.Rj_tol_nOhm.idxmin()] if valid else None
        individual_rows.append({
            **dict(zip(["Npw", "Top_K", "rho_turn_uOhm_cm2", "epsilon"], key)),
            "Rj_tol_nOhm": float(controlling.Rj_tol_nOhm) if valid else np.nan,
            "upper_censored": bool(valid and group.upper_censored.all()),
            "controlling_scenario": "" if not valid or group.upper_censored.all() else controlling.scenario,
            "nonmonotonic": bool(group.nonmonotonic.any()),
            **{name: float(controlling[name]) if valid else np.nan for name in
               ("r_lo_nOhm", "r_hi_nOhm", "delta_lo", "delta_hi")},
        })
    individuals = pd.DataFrame(individual_rows)
    summary_rows = []
    for key, group in individuals.groupby(["Npw", "Top_K", "epsilon"], sort=True):
        valid = group.Rj_tol_nOhm.notna().all()
        controlling = group.loc[group.Rj_tol_nOhm.idxmin()] if valid else None
        summary_rows.append({
            **dict(zip(["Npw", "Top_K", "epsilon"], key)), "N_rho": 61,
            "N_valid": int(group.Rj_tol_nOhm.notna().sum()),
            "Rj_tol_nOhm": float(controlling.Rj_tol_nOhm) if valid else np.nan,
            "min_valid_nOhm": float(group.Rj_tol_nOhm.min()) if group.Rj_tol_nOhm.notna().any() else np.nan,
            "max_valid_nOhm": float(group.Rj_tol_nOhm.max()) if group.Rj_tol_nOhm.notna().any() else np.nan,
            "upper_censored": bool(valid and group.upper_censored.all()),
            "N_censored": int(group.upper_censored.sum()),
            "N_nonmonotonic": int(group.nonmonotonic.sum()),
            "controlling_scenario": "" if not valid or group.upper_censored.all() else controlling.controlling_scenario,
            "controlling_rho_uOhm_cm2": float(controlling.rho_turn_uOhm_cm2) if valid and not group.upper_censored.all() else np.nan,
            **{name: float(controlling[name]) if valid else np.nan for name in
               ("r_lo_nOhm", "r_hi_nOhm", "delta_lo", "delta_hi")},
        })
    boundaries = pd.DataFrame(summary_rows)
    return heat, boundaries, individuals


def price_sensitivity(rows: pd.DataFrame, references: dict[str, dict[str, float]]) -> pd.DataFrame:
    scenario = "S2"
    hts_requirement = rows.Tape_cost_USD / rows.HTS_price_2025USD_per_kAm
    ps_current = rows.Power_supply_cost_USD / float(cfg.power_supply_price_perA)
    coolant_mass = rows.Coolant_fill_cost_USD / rows.Coolant_price_2025USD_per_kg
    ps_direct = ps_current * POWER_SUPPLY_PRICE_USD_PER_A
    coolant_fill = coolant_mass * 175.0
    rate = float(DISCOUNT_RATE_BY_SCENARIO[scenario])
    years = float(PROJECT_YEARS_BY_SCENARIO[scenario])
    crf = capital_recovery_factor(rate, years)
    annual_cost = (
        rows.E_fusion_th_year_MWh * CORE_VOM_BY_SCENARIO[scenario]
        + rows.gross_energy_annual_MWh * PCS_VOM_USD_PER_MWH_E
        + annual_pcs_fom_usd() + coolant_fill * COOLANT_REPLENISH_FRACTION
    )
    result = []
    for price in (10.0, 50.0, 100.0):
        delta = (
            M_HTS * (hts_requirement * price - references[scenario]["hts"])
            + M_PS * (ps_direct - references[scenario]["ps"])
            + M_COOLANT_FILL * (coolant_fill - references[scenario]["fill"])
        )
        lcoe = (crf * (build_capital_anchor(scenario).full_anchor_usd + delta) + annual_cost) / rows.net_energy_annual_MWh
        work = rows[["Top_K", "rho_turn_uOhm_cm2"]].copy(); work["lcoe_price"] = lcoe
        best = work.loc[work.groupby("Top_K").lcoe_price.idxmin()].sort_values("Top_K")
        reference = float(best.lcoe_price.min())
        preferred = float(best.loc[best.lcoe_price.idxmin(), "Top_K"])
        for _, row in best.iterrows():
            result.append({
                "HTS_price_USD_per_kAm": price, "Top_K": float(row.Top_K),
                "lcoe_price": float(row.lcoe_price),
                "lcoe_premium_pct": 100.0 * (float(row.lcoe_price) / reference - 1.0),
                "preferred_temperature_K": preferred,
                "selected_rho_turn_uOhm_cm2": float(row.rho_turn_uOhm_cm2),
            })
    return pd.DataFrame(result)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidate, raw, output = args.candidate.resolve(), args.raw.resolve(), args.output.resolve()
    if output.exists():
        raise SystemExit(f"STOP_OUTPUT_EXISTS: {output}")
    audit = json.loads((candidate / "V10_LEDGER_AUDIT.json").read_text(encoding="utf-8"))
    baselines = {key: float(value) for key, value in audit["scenario_minimum_lcoe_USD_per_MWh"].items()}
    paths = ledger_paths(candidate)
    cache = candidate / "figure_input_raw_cache"
    heat_cache = cache / "raw_heat.parquet"
    price_cache = cache / "raw_price.parquet"
    if heat_cache.is_file() and price_cache.is_file():
        raw_heat = pd.read_parquet(heat_cache)
        raw_price = pd.read_parquet(price_cache)
    else:
        raw_heat, raw_price = scan_small(raw)
        cache.mkdir(parents=True, exist_ok=True)
        raw_heat.to_parquet(heat_cache, index=False, compression="zstd")
        raw_price.to_parquet(price_cache, index=False, compression="zstd")
    fig3 = output / "main_figures" / "Fig3"; fig4 = output / "main_figures" / "Fig4"; fig5 = output / "main_figures" / "Fig5"
    for folder in (fig3, fig4, fig5):
        folder.mkdir(parents=True, exist_ok=False)
    build_heat_panel(raw_heat).to_parquet(fig3 / "panel_B_pulse_heat_load.parquet", index=False)
    refrigeration_statistics(paths).to_csv(fig3 / "panel_D_refrigeration_statistics.csv", index=False, float_format="%.17g")
    heat, boundaries, individuals = fig5_from_ledgers(paths, baselines)
    heat.to_csv(fig5 / "panel_A_conservative_heatmaps.csv", index=False, float_format="%.17g")
    boundaries.loc[boundaries.epsilon.eq(.10)].to_csv(
        fig5 / "panel_B_conservative_boundaries.csv", index=False, float_format="%.17g"
    )
    reported_source = output / "reported_values_source"
    reported_source.mkdir(parents=True, exist_ok=False)
    boundaries.to_csv(
        reported_source / "all_margins_conservative_boundaries.csv",
        index=False,
        float_format="%.17g",
    )
    individuals.to_csv(
        reported_source / "individual_rho_interpolated_boundaries.csv",
        index=False,
        float_format="%.17g",
    )
    price_sensitivity(raw_price, audit["reference_costs_USD"]).to_csv(
        fig5 / "panel_C_price_sensitivity.csv", index=False, float_format="%.17g"
    )
    report = {
        "status": "PASS", "physical_model_rerun": "separate upstream step",
        "Q_nuclear_W_per_TF": float(cfg.V_mag * cfg.NUCLEAR_POWER_DENSITY),
        "baselines": baselines,
        "Fig3_heat_rows": 200 * 121, "Fig3_refrigeration_rows": 9,
        "Fig5_heat_rows": len(heat), "Fig5_boundary_rows": int(boundaries.epsilon.eq(.10).sum()),
    }
    (output / "V10_FIGURE_INPUT_AUDIT.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
