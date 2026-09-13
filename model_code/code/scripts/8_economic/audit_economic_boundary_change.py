"""Compare the frozen pre-v2 scan with the plant-OPEX full-grid result."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from fusion_tem.economic.cost_boundary import COST_BOUNDARY_VERSION
from fusion_tem.economic.feasibility import (
    add_joint_lcoe_reference,
    apply_system_feasibility,
)
from fusion_tem.economic.near_optimal import (
    compute_robust_architecture_window,
    compute_scenario_architecture_regret,
    summarize_robust_windows,
)


CENTRAL_SCENARIOS = ("S1", "S2", "S3")
PRICE_SCENARIOS = ("S4", "S5", "S6")
ALL_SCENARIOS = CENTRAL_SCENARIOS + PRICE_SCENARIOS
DESIGN_COLUMNS = (
    "Top_K",
    "coolant",
    "Npw",
    "rho_turn_uOhm_cm2",
    "R_joint_nOhm",
)
BASE_COLUMNS = {
    "scenario",
    "status",
    "AF",
    "r_cryo_re",
    "E_net_year_MWh",
    "E_gross_year_MWh",
    "Charging_time_999_h",
    *DESIGN_COLUMNS,
}
NEW_COST_COLUMNS = {
    "cost_boundary_version",
    "CRF",
    "C0_background_USD",
    "CAPEX_mag_installed_USD",
    "Annualized_CAPEX_total_modeled_USD_per_year",
    "OPEX_core_VOM_USD_per_year",
    "OPEX_PCS_VOM_USD_per_year",
    "OPEX_PCS_FOM_USD_per_year",
    "OPEX_coolant_VOM_USD_per_year",
    "OPEX_total_modeled_USD_per_year",
    "LCOE_plant_USD_per_MWh",
    "LCOE_magnet_only_USD_per_MWh",
}
OLD_LCOE_COLUMN = "LCOE_fullplant_USD_per_MWh"
NEW_LCOE_COLUMN = "LCOE_plant_USD_per_MWh"
OPEX_COLUMNS = (
    "OPEX_core_VOM_USD_per_year",
    "OPEX_PCS_VOM_USD_per_year",
    "OPEX_PCS_FOM_USD_per_year",
    "OPEX_coolant_VOM_USD_per_year",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_selected(path: Path, required: set[str]) -> pd.DataFrame:
    header = set(pd.read_csv(path, nrows=0).columns)
    missing = sorted(required.difference(header))
    if missing:
        raise KeyError(f"{path} is missing audit columns: {missing}")
    return pd.read_csv(path, usecols=sorted(required), low_memory=False)


def _normalize_scan(path: Path, *, model: str) -> pd.DataFrame:
    if model == "old":
        frame = _read_selected(path, BASE_COLUMNS | {OLD_LCOE_COLUMN})
        frame[NEW_LCOE_COLUMN] = pd.to_numeric(
            frame[OLD_LCOE_COLUMN], errors="coerce"
        )
    elif model == "new":
        frame = _read_selected(path, BASE_COLUMNS | NEW_COST_COLUMNS)
        versions = set(frame["cost_boundary_version"].dropna().astype(str))
        if versions != {COST_BOUNDARY_VERSION}:
            raise AssertionError(
                f"Unexpected cost boundary versions in {path}: {sorted(versions)}"
            )
    else:
        raise ValueError(model)
    frame, _ = apply_system_feasibility(frame)
    frame, _ = add_joint_lcoe_reference(frame)
    return frame


def _validate_new_boundary(frame: pd.DataFrame) -> dict[str, float]:
    opex_sum = frame[list(OPEX_COLUMNS)].sum(axis=1)
    opex_error = (
        opex_sum - frame["OPEX_total_modeled_USD_per_year"]
    ).abs()
    if not np.allclose(
        opex_sum,
        frame["OPEX_total_modeled_USD_per_year"],
        rtol=1e-12,
        atol=1e-6,
    ):
        raise AssertionError(f"OPEX identity failed; max error={opex_error.max()}")

    annualized_expected = frame["CRF"] * (
        frame["CAPEX_mag_installed_USD"] + frame["C0_background_USD"]
    )
    annualized_error = (
        annualized_expected
        - frame["Annualized_CAPEX_total_modeled_USD_per_year"]
    ).abs()
    if not np.allclose(
        annualized_expected,
        frame["Annualized_CAPEX_total_modeled_USD_per_year"],
        rtol=1e-12,
        atol=1e-5,
    ):
        raise AssertionError(
            f"Annualized CAPEX identity failed; max error={annualized_error.max()}"
        )

    valid = (
        np.isfinite(frame[NEW_LCOE_COLUMN])
        & np.isfinite(frame["E_net_year_MWh"])
        & (frame["E_net_year_MWh"] > 0.0)
    )
    numerator_observed = (
        frame.loc[valid, NEW_LCOE_COLUMN]
        * frame.loc[valid, "E_net_year_MWh"]
    )
    numerator_expected = (
        frame.loc[valid, "Annualized_CAPEX_total_modeled_USD_per_year"]
        + frame.loc[valid, "OPEX_total_modeled_USD_per_year"]
    )
    numerator_error = (numerator_observed - numerator_expected).abs()
    if not np.allclose(
        numerator_observed,
        numerator_expected,
        rtol=1e-11,
        atol=1e-4,
    ):
        raise AssertionError(
            f"LCOE numerator identity failed; max error={numerator_error.max()}"
        )

    feasible = frame["feasible_joint"].astype(bool)
    if not np.isfinite(frame.loc[feasible, NEW_LCOE_COLUMN]).all():
        raise AssertionError("A jointly feasible row has nonfinite plant LCOE")
    if not (frame.loc[feasible, "E_net_year_MWh"] > 0.0).all():
        raise AssertionError("A jointly feasible row has nonpositive net energy")
    if not (
        frame.loc[valid, NEW_LCOE_COLUMN]
        >= frame.loc[valid, "LCOE_magnet_only_USD_per_MWh"] - 1e-12
    ).all():
        raise AssertionError("Plant LCOE is below the magnet-only diagnostic")

    return {
        "opex_identity_max_abs_USD_per_year": float(opex_error.max()),
        "annualized_capex_identity_max_abs_USD_per_year": float(
            annualized_error.max()
        ),
        "lcoe_numerator_identity_max_abs_USD_per_year": float(
            numerator_error.max()
        ),
    }


def _sorted_optimum(feasible: pd.DataFrame) -> pd.Series:
    return feasible.sort_values(
        [NEW_LCOE_COLUMN, *DESIGN_COLUMNS],
        kind="mergesort",
    ).iloc[0]


def _append_metric(
    rows: list[dict[str, object]],
    *,
    category: str,
    model: str,
    scenario: str,
    metric: str,
    value: object,
) -> None:
    rows.append(
        {
            "category": category,
            "model": model,
            "scenario": scenario,
            "metric": metric,
            "value": value,
        }
    )


def _model_metrics(
    frame: pd.DataFrame,
    *,
    model: str,
    rows: list[dict[str, object]],
    fig5_rho: float,
    fig5_npw_min: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    regret = compute_scenario_architecture_regret(frame)
    robust = compute_robust_architecture_window(regret)
    robust_summary = summarize_robust_windows(robust)
    robust_5 = robust.loc[robust["robust_window_5pct"].astype(bool)]
    regret_5 = regret.merge(
        robust_5[["Npw", "R_joint_nOhm"]],
        on=["Npw", "R_joint_nOhm"],
        how="inner",
    )

    for scenario in CENTRAL_SCENARIOS:
        group = frame.loc[frame["scenario"].eq(scenario)]
        feasible = group.loc[group["feasible_joint"].astype(bool)].copy()
        optimum = _sorted_optimum(feasible)
        _append_metric(
            rows,
            category="scenario_optimum",
            model=model,
            scenario=scenario,
            metric="jointly_feasible_count",
            value=int(len(feasible)),
        )
        _append_metric(
            rows,
            category="scenario_optimum",
            model=model,
            scenario=scenario,
            metric="minimum_LCOE_USD_per_MWh",
            value=float(optimum[NEW_LCOE_COLUMN]),
        )
        for field in DESIGN_COLUMNS:
            _append_metric(
                rows,
                category="scenario_optimum",
                model=model,
                scenario=scenario,
                metric=f"minimum_design_{field}",
                value=optimum[field],
            )
        _append_metric(
            rows,
            category="delta_lcoe",
            model=model,
            scenario=scenario,
            metric="full_grid_joint_feasible_max_delta_USD_per_MWh",
            value=float(
                feasible["delta_LCOE_joint_feasible_USD_per_MWh"].max()
            ),
        )
        fig5 = feasible.loc[
            np.isclose(feasible["rho_turn_uOhm_cm2"], fig5_rho)
            & (feasible["Npw"] >= fig5_npw_min)
        ]
        _append_metric(
            rows,
            category="delta_lcoe",
            model=model,
            scenario=scenario,
            metric="fig5_slice_max_delta_USD_per_MWh",
            value=(
                float(fig5["delta_LCOE_joint_feasible_USD_per_MWh"].max())
                if not fig5.empty
                else np.nan
            ),
        )
        sregret = regret_5.loc[regret_5["scenario"].eq(scenario)]
        for edge, reducer in (("min", "min"), ("max", "max")):
            value = (
                float(
                    getattr(
                        sregret["delta_LCOE_architecture_USD_per_MWh"],
                        reducer,
                    )()
                )
                if not sregret.empty
                else np.nan
            )
            _append_metric(
                rows,
                category="near_optimal_5pct",
                model=model,
                scenario=scenario,
                metric=f"delta_LCOE_{edge}_USD_per_MWh",
                value=value,
            )

    for item in robust_summary.itertuples(index=False):
        _append_metric(
            rows,
            category="robust_architecture",
            model=model,
            scenario="S1-S3",
            metric=f"count_at_{item.regret_threshold_pct:g}pct_regret",
            value=int(item.robust_architecture_count),
        )
    return regret, robust, robust_summary


def _price_and_opex_metrics(
    frame: pd.DataFrame,
    *,
    rows: list[dict[str, object]],
) -> None:
    for scenario in PRICE_SCENARIOS:
        feasible = frame.loc[
            frame["scenario"].eq(scenario)
            & frame["feasible_joint"].astype(bool)
        ].copy()
        optimum = _sorted_optimum(feasible)
        option_min = (
            feasible.groupby(["Top_K", "coolant"], sort=True)[NEW_LCOE_COLUMN]
            .min()
            .sort_index()
        )
        _append_metric(
            rows,
            category="fig6_price_sensitivity",
            model="new",
            scenario=scenario,
            metric="optimum_cryogenic_option",
            value=f"{optimum['Top_K']:g} K {optimum['coolant']}",
        )
        _append_metric(
            rows,
            category="fig6_price_sensitivity",
            model="new",
            scenario=scenario,
            metric="optimum_LCOE_USD_per_MWh",
            value=float(optimum[NEW_LCOE_COLUMN]),
        )
        _append_metric(
            rows,
            category="fig6_price_sensitivity",
            model="new",
            scenario=scenario,
            metric="four_option_LCOE_span_USD_per_MWh",
            value=float(option_min.max() - option_min.min()),
        )
        for (temperature, coolant), value in option_min.items():
            _append_metric(
                rows,
                category="fig6_price_sensitivity",
                model="new",
                scenario=scenario,
                metric=f"option_min_{temperature:g}K_{coolant}_USD_per_MWh",
                value=float(value),
            )

    for scenario in ALL_SCENARIOS:
        feasible = frame.loc[
            frame["scenario"].eq(scenario)
            & frame["feasible_joint"].astype(bool)
        ].copy()
        optimum = _sorted_optimum(feasible)
        total = float(optimum["OPEX_total_modeled_USD_per_year"])
        for field in OPEX_COLUMNS:
            value = float(optimum[field])
            _append_metric(
                rows,
                category="opex_at_scenario_optimum",
                model="new",
                scenario=scenario,
                metric=field,
                value=value,
            )
            _append_metric(
                rows,
                category="opex_at_scenario_optimum",
                model="new",
                scenario=scenario,
                metric=f"{field}_share_of_modeled_OPEX",
                value=value / total,
            )


def _markdown_table(frame: pd.DataFrame) -> list[str]:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-scan", type=Path, required=True)
    parser.add_argument("--new-scan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fig5-rho", type=float, default=10000.0)
    parser.add_argument("--fig5-npw-min", type=int, default=5)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    old = _normalize_scan(args.old_scan, model="old")
    new = _normalize_scan(args.new_scan, model="new")
    checks = _validate_new_boundary(new)

    rows: list[dict[str, object]] = []
    _model_metrics(
        old,
        model="old_pre_v2",
        rows=rows,
        fig5_rho=args.fig5_rho,
        fig5_npw_min=args.fig5_npw_min,
    )
    _, _, robust_new = _model_metrics(
        new,
        model=COST_BOUNDARY_VERSION,
        rows=rows,
        fig5_rho=args.fig5_rho,
        fig5_npw_min=args.fig5_npw_min,
    )
    _price_and_opex_metrics(new, rows=rows)
    for metric, value in checks.items():
        _append_metric(
            rows,
            category="validation",
            model=COST_BOUNDARY_VERSION,
            scenario="all",
            metric=metric,
            value=value,
        )

    audit = pd.DataFrame(rows)
    audit_csv = args.output_dir / "economic_boundary_change_audit.csv"
    audit_md = args.output_dir / "economic_boundary_change_summary.md"
    audit.to_csv(audit_csv, index=False)

    optimum = audit.loc[
        audit["category"].eq("scenario_optimum")
        & audit["metric"].eq("minimum_LCOE_USD_per_MWh")
    ].pivot(index="scenario", columns="model", values="value").reset_index()
    feasible = audit.loc[
        audit["category"].eq("scenario_optimum")
        & audit["metric"].eq("jointly_feasible_count")
    ].pivot(index="scenario", columns="model", values="value").reset_index()
    fig6 = audit.loc[
        audit["category"].eq("fig6_price_sensitivity")
        & audit["metric"].isin(
            (
                "optimum_cryogenic_option",
                "optimum_LCOE_USD_per_MWh",
                "four_option_LCOE_span_USD_per_MWh",
            )
        )
    ].pivot(index="scenario", columns="metric", values="value").reset_index()
    report = [
        "# Economic boundary change audit",
        "",
        f"- Old scan: `{args.old_scan}`",
        f"- Old SHA256: `{_sha256(args.old_scan)}`",
        f"- New scan: `{args.new_scan}`",
        f"- New SHA256: `{_sha256(args.new_scan)}`",
        f"- New cost boundary: `{COST_BOUNDARY_VERSION}`",
        f"- Rows: old={len(old):,}; new={len(new):,}",
        "- Formal feasibility is unchanged: AF_ref>=0.99, r_cryo<=0.50, "
        "finite main LCOE, and no model error; 120 h remains diagnostic.",
        "",
        "## S1-S3 minimum LCOE",
        "",
        *_markdown_table(optimum),
        "",
        "## S1-S3 jointly feasible counts",
        "",
        *_markdown_table(feasible),
        "",
        "## Robust architecture counts",
        "",
        *_markdown_table(robust_new),
        "",
        "## S4-S6 cryogenic-option result",
        "",
        *_markdown_table(fig6),
        "",
        "## Machine-readable detail",
        "",
        f"All design coordinates, delta-LCOE ranges, option minima, and OPEX "
        f"values/shares are in `{audit_csv.name}`.",
    ]
    audit_md.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"[OK] {audit_csv}")
    print(f"[OK] {audit_md}")


if __name__ == "__main__":
    main()
