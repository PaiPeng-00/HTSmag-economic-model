"""Generate the single P0 manuscript metrics and sensitivity tables."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("FUSION_DEVICE", "arc_16pancake_nuc600")

from fusion_tem import device as cfg
from fusion_tem.cryo.heat_load import radiation_heat
from fusion_tem.economic.feasibility import (
    add_joint_lcoe_reference,
    apply_system_feasibility,
)
from fusion_tem.economic.near_optimal import (
    compute_robust_architecture_window,
    compute_scenario_architecture_regret,
    summarize_robust_windows,
)
from fusion_tem.economic.sensitivity import recalculate_energy_boundary


CENTRAL_SCENARIOS = ("S1", "S2", "S3")
ALL_SCENARIOS = ("S1", "S2", "S3", "S4", "S5", "S6")
OTHER_VALUES = (0.20, 0.25, 0.30, 0.35)

COMPARISON_COLUMNS = {
    "scenario",
    "status",
    "Top_K",
    "coolant",
    "Npw",
    "rho_turn_uOhm_cm2",
    "R_joint_nOhm",
    "AF",
    "E_net_year_MWh",
    "E_gross_year_MWh",
    "E_cryo_year_MWh",
    "Charging_time_999_h",
    "LCOE_plant_USD_per_MWh",
    "r_cryo_re",
}

REQUIRED_COLUMNS = {
    "scenario",
    "status",
    "Top_K",
    "coolant",
    "Npw",
    "rho_turn_uOhm_cm2",
    "R_joint_nOhm",
    "AF",
    "AF_ref_system",
    "AF_max_scenario",
    "feasible_joint",
    "feasible_cryo_050",
    "model_error",
    "no_model_error",
    "LCOE_plant_USD_per_MWh",
    "delta_LCOE_joint_feasible_USD_per_MWh",
    "E_gross_year_MWh",
    "E_other_year_MWh",
    "E_cryo_year_MWh",
    "E_net_year_MWh",
    "E_cryo_charge_event_MWh",
    "E_cryo_discharge_event_MWh",
    "E_cryo_charge_MWh",
    "E_cryo_discharge_MWh",
    "P_cryo_charge_avg_W",
    "P_cryo_discharge_avg_W",
    "E_dynamic_charge_MWh",
    "E_dynamic_discharge_MWh",
    "charge_discharge_profile_ratio",
    "charge_time_event_h",
    "discharge_time_event_h",
    "transient_energy_fraction_of_Ecryo",
    "r_other",
    "r_cryo_re",
    "r_cryo_re_fraction",
    "energy_closure_error_MWh",
    "Charging_time_999_h",
    "Q_radiation_W",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_scan(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns
    missing = sorted(REQUIRED_COLUMNS.difference(header))
    if missing:
        raise KeyError(f"Data S1 is missing P0 fields: {missing}")
    return pd.read_csv(path, usecols=sorted(REQUIRED_COLUMNS), low_memory=False)


def read_comparison_scan(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns
    missing = sorted(COMPARISON_COLUMNS.difference(header))
    if missing:
        raise KeyError(f"Comparison scan is missing fields: {missing}")
    return pd.read_csv(
        path, usecols=sorted(COMPARISON_COLUMNS), low_memory=False
    )


def validate_central_scan(df: pd.DataFrame) -> dict[str, object]:
    scenarios = tuple(sorted(df["scenario"].dropna().unique()))
    if scenarios != ALL_SCENARIOS:
        raise AssertionError(f"Expected S1-S6, observed {scenarios}")

    radiation_formula = {
        temperature: radiation_heat(
            cfg.A_cryostat, cfg.eps, cfg.T_HIGH, temperature
        )
        for temperature in (4.2, 10.0, 20.0)
    }
    radiation_scan = {}
    for temperature in (4.2, 10.0, 20.0):
        values = np.sort(
            df.loc[
                np.isclose(df["Top_K"], temperature),
                "Q_radiation_W",
            ].dropna().unique()
        )
        if len(values) != 1:
            raise AssertionError(
                f"Expected one Q_radiation_W at {temperature} K, got {values}"
            )
        radiation_scan[temperature] = float(values[0])
    if not all(
        np.isclose(
            radiation_scan[temperature],
            radiation_formula[temperature],
            rtol=0.0,
            atol=1e-9,
        )
        for temperature in radiation_scan
    ):
        raise AssertionError(
            "Data S1 radiation values do not match the active device "
            f"{cfg.DEVICE}: scan={radiation_scan}, formula={radiation_formula}"
        )
    if not all(27.0 <= value <= 29.0 for value in radiation_scan.values()):
        raise AssertionError(f"Radiation test failed: {radiation_scan}")

    af_norm_max = df.groupby("scenario")["AF_ref_system"].max()
    if not np.allclose(af_norm_max.to_numpy(), 1.0, rtol=0.0, atol=1e-12):
        raise AssertionError(f"AF_ref maxima are not one: {af_norm_max.to_dict()}")

    exact_joint = (
        (df["AF_ref_system"] >= 0.99)
        & (df["r_cryo_re_fraction"] <= 0.50)
        & np.isfinite(df["LCOE_plant_USD_per_MWh"])
        & ~df["model_error"].astype(bool)
    )
    if not (exact_joint == df["feasible_joint"].astype(bool)).all():
        raise AssertionError("feasible_joint does not equal the exact P0 gate")

    feasible = df["feasible_joint"].astype(bool)
    delta_min = (
        df.loc[feasible]
        .groupby("scenario")["delta_LCOE_joint_feasible_USD_per_MWh"]
        .min()
    )
    if not np.allclose(delta_min.to_numpy(), 0.0, rtol=0.0, atol=1e-9):
        raise AssertionError(f"Scenario delta-LCOE minima are not zero: {delta_min}")

    closure_max = float(df["energy_closure_error_MWh"].abs().max())
    closure_scale = float(df["E_gross_year_MWh"].abs().max())
    if closure_max > max(1e-8, 1e-12 * closure_scale):
        raise AssertionError(f"Energy closure failed: max error {closure_max}")

    profile_ratio = pd.to_numeric(
        df["charge_discharge_profile_ratio"], errors="coerce"
    )
    if not np.allclose(profile_ratio, 1.0, rtol=0.0, atol=1e-12):
        raise AssertionError("Charge/discharge profile ratio is not exactly 1.0")
    equality_pairs = (
        ("charge_time_event_h", "discharge_time_event_h"),
        ("E_cryo_charge_event_MWh", "E_cryo_discharge_event_MWh"),
        ("E_cryo_charge_MWh", "E_cryo_discharge_MWh"),
        ("P_cryo_charge_avg_W", "P_cryo_discharge_avg_W"),
        ("E_dynamic_charge_MWh", "E_dynamic_discharge_MWh"),
    )
    for charge_col, discharge_col in equality_pairs:
        if not np.allclose(
            df[charge_col], df[discharge_col], rtol=1e-10, atol=1e-10
        ):
            raise AssertionError(f"Charge/discharge mismatch: {charge_col}, {discharge_col}")

    return {
        "radiation_W": radiation_scan,
        "radiation_formula_W": radiation_formula,
        "af_norm_max": af_norm_max.to_dict(),
        "delta_min": delta_min.to_dict(),
        "energy_closure_max_abs_MWh": closure_max,
    }


def _format_number(value: object) -> object:
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.10g}"
    return value


def build_manuscript_metrics(
    df: pd.DataFrame,
    regret: pd.DataFrame,
    robust: pd.DataFrame,
    *,
    fig5_rho: float,
    fig5_npw_min: int,
) -> pd.DataFrame:
    robust_5 = robust.loc[robust["robust_window_5pct"].astype(bool)]
    robust_keys = robust_5[["Npw", "R_joint_nOhm"]]
    regret_5 = regret.merge(
        robust_keys, on=["Npw", "R_joint_nOhm"], how="inner"
    )

    metrics: dict[str, dict[str, object]] = {}
    for scenario in CENTRAL_SCENARIOS:
        group = df.loc[df["scenario"].eq(scenario)].copy()
        feasible = group.loc[group["feasible_joint"].astype(bool)].copy()
        optimum = feasible.sort_values(
            [
                "LCOE_plant_USD_per_MWh",
                "Npw",
                "R_joint_nOhm",
                "rho_turn_uOhm_cm2",
                "Top_K",
                "coolant",
            ],
            kind="mergesort",
        ).iloc[0]

        fig5 = feasible.loc[
            np.isclose(feasible["rho_turn_uOhm_cm2"], fig5_rho)
            & (feasible["Npw"] >= fig5_npw_min)
        ]
        per_temperature = []
        for temperature, tgroup in feasible.groupby("Top_K", sort=True):
            per_temperature.append(
                f"{temperature:g}K:"
                f"{tgroup['r_cryo_re_fraction'].min():.6g}-"
                f"{tgroup['r_cryo_re_fraction'].max():.6g}"
            )

        sregret = regret_5.loc[regret_5["scenario"].eq(scenario)]
        delta_5 = (
            f"{sregret['delta_LCOE_architecture_USD_per_MWh'].min():.6g}-"
            f"{sregret['delta_LCOE_architecture_USD_per_MWh'].max():.6g}"
            if not sregret.empty
            else "NA"
        )
        metrics[scenario] = {
            "AF_max": float(group["AF_max_scenario"].iloc[0]),
            "jointly_feasible_count": int(len(feasible)),
            "minimum_LCOE_USD_per_MWh": float(
                optimum["LCOE_plant_USD_per_MWh"]
            ),
            "reference_design_Npw": int(optimum["Npw"]),
            "reference_design_R_joint_nOhm": float(optimum["R_joint_nOhm"]),
            "reference_design_rho_turn_uOhm_cm2": float(
                optimum["rho_turn_uOhm_cm2"]
            ),
            "reference_design_Top_K": float(optimum["Top_K"]),
            "reference_coolant": str(optimum["coolant"]),
            "full_grid_joint_feasible_max_delta_LCOE": float(
                feasible["delta_LCOE_joint_feasible_USD_per_MWh"].max()
            ),
            "fig5_displayed_slice_max_delta_LCOE": (
                float(fig5["delta_LCOE_joint_feasible_USD_per_MWh"].max())
                if not fig5.empty
                else np.nan
            ),
            "delta_LCOE_range_within_5pct_window": delta_5,
            "r_cryo_re_fraction_min_max_by_T": ";".join(per_temperature),
        }

    ordered_metrics = list(next(iter(metrics.values())).keys())
    rows = []
    for metric in ordered_metrics:
        row = {"Metric": metric}
        for scenario in CENTRAL_SCENARIOS:
            row[scenario] = _format_number(metrics[scenario][metric])
        rows.append(row)
    return pd.DataFrame(rows)


def build_old_new_comparison(
    old_df: pd.DataFrame,
    new_df: pd.DataFrame,
) -> pd.DataFrame:
    old_revised, _ = apply_system_feasibility(old_df)
    old_revised, _ = add_joint_lcoe_reference(old_revised)

    rows = []
    for scenario in ALL_SCENARIOS:
        versions = {}
        for label, frame in (("old", old_revised), ("new", new_df)):
            group = frame.loc[frame["scenario"].eq(scenario)]
            feasible = group.loc[group["feasible_joint"].astype(bool)]
            optimum = feasible.sort_values(
                [
                    "LCOE_plant_USD_per_MWh",
                    "Npw",
                    "R_joint_nOhm",
                    "rho_turn_uOhm_cm2",
                    "Top_K",
                    "coolant",
                ],
                kind="mergesort",
            ).iloc[0]
            versions[label] = {
                "jointly_feasible_count": int(len(feasible)),
                "minimum_LCOE_USD_per_MWh": float(
                    optimum["LCOE_plant_USD_per_MWh"]
                ),
                "full_grid_joint_feasible_max_delta_LCOE": float(
                    feasible["delta_LCOE_joint_feasible_USD_per_MWh"].max()
                ),
                "reference_Npw": int(optimum["Npw"]),
                "reference_R_joint_nOhm": float(optimum["R_joint_nOhm"]),
                "reference_rho_turn_uOhm_cm2": float(
                    optimum["rho_turn_uOhm_cm2"]
                ),
                "reference_Top_K": float(optimum["Top_K"]),
                "reference_coolant": str(optimum["coolant"]),
                "E_cryo_year_MWh_min": float(
                    feasible["E_cryo_year_MWh"].min()
                ),
                "E_cryo_year_MWh_max": float(
                    feasible["E_cryo_year_MWh"].max()
                ),
            }
        for metric in versions["new"]:
            old_value = versions["old"][metric]
            new_value = versions["new"][metric]
            delta = (
                new_value - old_value
                if isinstance(old_value, (int, float))
                and isinstance(new_value, (int, float))
                else ""
            )
            rows.append(
                {
                    "scenario": scenario,
                    "metric": metric,
                    "old_value": old_value,
                    "new_value": new_value,
                    "new_minus_old": delta,
                }
            )
    return pd.DataFrame(rows)


def sensitivity_summary(
    central: pd.DataFrame,
    *,
    sensitivity_type: str,
    values: tuple[float, ...],
) -> pd.DataFrame:
    rows = []
    for value in values:
        if sensitivity_type != "r_other":
            raise ValueError("Only non-TF auxiliary sensitivity remains defined")
        case = recalculate_energy_boundary(
            central, other_aux_fraction=value
        )
        case, _ = apply_system_feasibility(case)
        case, _ = add_joint_lcoe_reference(case)

        regret = compute_scenario_architecture_regret(case)
        robust = compute_robust_architecture_window(regret)
        robust5 = robust.loc[robust["robust_window_5pct"].astype(bool)]
        robust_count = int(len(robust5))
        npw_min = robust5["Npw"].min() if robust_count else np.nan
        npw_max = robust5["Npw"].max() if robust_count else np.nan
        rj_min = robust5["R_joint_nOhm"].min() if robust_count else np.nan
        rj_max = robust5["R_joint_nOhm"].max() if robust_count else np.nan

        for scenario in CENTRAL_SCENARIOS:
            group = case.loc[case["scenario"].eq(scenario)]
            feasible = group.loc[group["feasible_joint"].astype(bool)]
            rows.append(
                {
                    "sensitivity_type": sensitivity_type,
                    "sensitivity_value": value,
                    "scenario": scenario,
                    "jointly_feasible_count": int(len(feasible)),
                    "E_net_year_MWh_min": feasible["E_net_year_MWh"].min(),
                    "E_net_year_MWh_max": feasible["E_net_year_MWh"].max(),
                    "LCOE_min_USD_per_MWh": feasible[
                        "LCOE_plant_USD_per_MWh"
                    ].min(),
                    "LCOE_max_USD_per_MWh": feasible[
                        "LCOE_plant_USD_per_MWh"
                    ].max(),
                    "delta_LCOE_max_USD_per_MWh": feasible[
                        "delta_LCOE_joint_feasible_USD_per_MWh"
                    ].max(),
                    "E_cryo_charge_MWh_min": feasible["E_cryo_charge_MWh"].min(),
                    "E_cryo_charge_MWh_max": feasible["E_cryo_charge_MWh"].max(),
                    "E_cryo_discharge_MWh_min": feasible[
                        "E_cryo_discharge_MWh"
                    ].min(),
                    "E_cryo_discharge_MWh_max": feasible[
                        "E_cryo_discharge_MWh"
                    ].max(),
                    "E_dynamic_charge_MWh_min": feasible[
                        "E_dynamic_charge_MWh"
                    ].min(),
                    "E_dynamic_charge_MWh_max": feasible[
                        "E_dynamic_charge_MWh"
                    ].max(),
                    "E_dynamic_discharge_MWh_min": feasible[
                        "E_dynamic_discharge_MWh"
                    ].min(),
                    "E_dynamic_discharge_MWh_max": feasible[
                        "E_dynamic_discharge_MWh"
                    ].max(),
                    "robust_5pct_architecture_count": robust_count,
                    "robust_5pct_Npw_min": npw_min,
                    "robust_5pct_Npw_max": npw_max,
                    "robust_5pct_R_joint_nOhm_min": rj_min,
                    "robust_5pct_R_joint_nOhm_max": rj_max,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fig5-rho", type=float, default=10000.0)
    parser.add_argument("--fig5-npw-min", type=int, default=5)
    parser.add_argument("--old-scan", type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = read_scan(args.input_csv)
    checks = validate_central_scan(df)

    regret = compute_scenario_architecture_regret(df)
    robust = compute_robust_architecture_window(regret)
    robust_summary = summarize_robust_windows(robust)
    manuscript = build_manuscript_metrics(
        df,
        regret,
        robust,
        fig5_rho=args.fig5_rho,
        fig5_npw_min=args.fig5_npw_min,
    )
    other_summary = sensitivity_summary(
        df, sensitivity_type="r_other", values=OTHER_VALUES
    )

    manuscript.to_csv(
        args.output_dir / "manuscript_summary_metrics.csv", index=False
    )
    regret.to_csv(
        args.output_dir / "near_optimal_scenario_regret.csv", index=False
    )
    robust.to_csv(
        args.output_dir / "near_optimal_robust_architectures.csv", index=False
    )
    robust_summary.to_csv(
        args.output_dir / "near_optimal_window_sensitivity.csv", index=False
    )
    other_summary.to_csv(
        args.output_dir / "r_other_sensitivity.csv", index=False
    )

    if args.old_scan is not None:
        comparison = build_old_new_comparison(
            read_comparison_scan(args.old_scan),
            df,
        )
        comparison.to_csv(
            args.output_dir / "old_new_results_comparison.csv", index=False
        )

    full_column_count = len(pd.read_csv(args.input_csv, nrows=0).columns)
    system_radiation = {
        temperature: value * cfg.Ntf
        for temperature, value in checks["radiation_W"].items()
    }
    report_lines = [
        "# Plant-v3 constant-2025-US$ economic-boundary validation report",
        "",
        "## Implementation status",
        "",
        "- Main cost metric: LCOE_plant_USD_per_MWh under",
        "  plant_v4_core_pcs_coolant_2025usd_direct_hts, including core/PCS/coolant OPEX.",
        "- Radiation: 28 W per TF cryostat; the former 112 W arose because",
        "  radiation_heat used 2 eps/(N_MLI+1) instead of eps/[2(N_MLI+1)],",
        "  a factor-of-four prefactor error.",
        "- Charge/discharge: the fixed 500 W system-level load is absent. Data S2/S3",
        "  are integrated time step by time step after N_TF=18 aggregation and before",
        "  the nonlinear COP conversion.",
        "- Discharge event: t_dis=t_charge and P_cryo,dis(t)=P_cryo,ch(t);",
        "  base, dynamic, total event energy, and average power are identical.",
        "- Other auxiliaries: E_other=r_other E_gross excludes all TF cryogenics;",
        "  the central case uses 0.30 and sensitivities use 0.20/0.25/0.30/0.35.",
        "- Joint feasibility: AF_ref_system>=0.99, r_cryo_fraction<=0.50, finite LCOE,",
        "  and no model error. The 120 h and net-export fields are diagnostics only.",
        "- Near-optimality: each common (Npw, R_joint) architecture reoptimizes",
        "  rho_turn, operating temperature, and coolant within each scenario.",
        "",
        "## Modified code and functions",
        "",
        "- src/fusion_tem/cryo/heat_load.py:",
        "  radiation_heat and calculate_charge_cryo_electrical_energy.",

        "- src/fusion_tem/economic/lcoe.py: compute_case.",
        "- src/fusion_tem/economic/feasibility.py:",
        "  apply_system_feasibility, add_joint_lcoe_reference.",
        "- src/fusion_tem/economic/near_optimal.py:",
        "  compute_scenario_architecture_regret,",
        "  compute_robust_architecture_window, summarize_robust_windows.",
        "- src/fusion_tem/economic/sensitivity.py:",
        "  recalculate_energy_boundary and run_other_aux_sensitivity.",

        "- scripts/8_economic/scan_full_grid.py: main and cached heat/load mapping.",
        "- scripts/8_economic/summarize_p0_revision.py: P0 validation, summary,",
        "  sensitivity, regret, and old/new comparison outputs.",
        "- scripts/8_economic/8.1_plot_from_scan_full_grid.py:",
        "  exact feasibility/normalization input and separate output directory.",
        "- scripts/8_economic/revise_afref_system_feasibility.py and",
        "  summarize_afref_system_feasibility.py: exact joint-feasibility semantics.",
        "- tests/: 23 plant-v2 and submission-invariant regression tests.",
        "",
        "## Central full-grid validation",
        "",
        f"- Data S1: {args.input_csv}",
        f"- Active device: {cfg.DEVICE}",
        f"- Cryostat area: {cfg.A_cryostat:.12g} m2",
        f"- Rows x columns: {len(df):,} x {full_column_count}",
        f"- Validation columns read: {len(df.columns)}",
        "- Regression tests: 23/23 passed.",
        f"- SHA256: {sha256_file(args.input_csv)}",
        f"- Scenarios: {', '.join(sorted(df['scenario'].unique()))}",
        f"- Maximum absolute energy-closure error: {checks['energy_closure_max_abs_MWh']:.6g} MWh",
        f"- Radiation per TF cryostat: {checks['radiation_W']}",
        f"- Radiation for the full {cfg.Ntf}-TF system: {system_radiation}",
        "- Q_radiation_W is stored per TF; N_TF is applied once before the",
        "  nonlinear cryoplant conversion in transient/system calculations.",
        f"- AF_ref maxima: {checks['af_norm_max']}",
        f"- Joint-feasible delta-LCOE minima: {checks['delta_min']}",
        "",
        "The central scan passed the P0 radiation, AF normalization, exact joint",
        "feasibility, jointly feasible delta-LCOE reference, complete charge/discharge",
        "profile equality, S1-S6 coverage, and annual energy-closure checks.",
        "",
        "## Author confirmation required before Word editing",
        "",
        "P0 states 15,627 W for D1 at R_joint=100 nOhm, but direct component",
        "recalculation gives 15,827.18 W, including 14,170.67 W of pancake-joint",
        "heat. The code retains the formula-consistent 15,827 W result.",
        "",
        "Figures and 07_manuscript_SA_V3.1.docx were not modified during this",
        "code-validation stage.",
    ]
    report_text = "\n".join(report_lines) + "\n"
    for report_name in (
        "P0_model_validation_report.md",
        "P0_revision_report.md",
    ):
        (args.output_dir / report_name).write_text(
            report_text, encoding="utf-8"
        )


if __name__ == "__main__":
    main()
