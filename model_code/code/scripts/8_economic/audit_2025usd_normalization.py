"""Audit the plant-v2 to constant-2025-US$ plant-v3 economic rerun."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from fusion_tem.economic.cost_boundary import COST_BOUNDARY_VERSION
from fusion_tem.economic.feasibility import (
    add_joint_lcoe_reference,
    apply_system_feasibility,
)
from fusion_tem.economic.nonmonetary_invariance import (
    is_monetary_or_dependent_column,
)
from fusion_tem.economic.near_optimal import (
    compute_robust_architecture_window,
    compute_scenario_architecture_regret,
    exact_robust_frontier_at_npw,
    summarize_robust_windows,
)
from fusion_tem.economic.price_basis import (
    PRICE_BASIS_YEAR,
    monetary_conversion_audit_rows,
)


OLD_BOUNDARY = "plant_v2_core_pcs_coolant"
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
KEY_COLUMNS = (*DESIGN_COLUMNS, "scenario")
LCOE_COLUMN = "LCOE_plant_USD_per_MWh"
OPEX_COLUMNS = (
    "OPEX_core_VOM_USD_per_year",
    "OPEX_PCS_VOM_USD_per_year",
    "OPEX_PCS_FOM_USD_per_year",
    "OPEX_coolant_VOM_USD_per_year",
)
METRIC_COLUMNS = {
    *KEY_COLUMNS,
    "status",
    "AF",
    "AF_ref_system",
    "r_cryo_re_fraction",
    "E_net_year_MWh",
    "E_gross_year_MWh",
    "Charging_time_999_h",
    "cost_boundary_version",
    "CRF",
    "C0_background_USD",
    "CAPEX_mag_installed_USD",
    "Annualized_CAPEX_total_modeled_USD_per_year",
    *OPEX_COLUMNS,
    "OPEX_total_modeled_USD_per_year",
    LCOE_COLUMN,
    "LCOE_magnet_only_USD_per_MWh",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_selected(path: Path, required: Iterable[str]) -> pd.DataFrame:
    required = set(required)
    columns = set(pd.read_csv(path, nrows=0).columns)
    missing = sorted(required - columns)
    if missing:
        raise KeyError(f"{path} is missing audit columns: {missing}")
    return pd.read_csv(path, usecols=sorted(required), low_memory=False)


def normalize(path: Path, expected_boundary: str) -> pd.DataFrame:
    frame = read_selected(path, METRIC_COLUMNS)
    versions = set(frame["cost_boundary_version"].dropna().astype(str))
    if versions != {expected_boundary}:
        raise AssertionError(
            f"{path} boundary mismatch: expected {expected_boundary}, observed {sorted(versions)}"
        )
    frame, _ = apply_system_feasibility(frame)
    frame, _ = add_joint_lcoe_reference(frame)
    return frame


def _monetary_or_dependent(column: str) -> bool:
    return is_monetary_or_dependent_column(column)


def compare_nonmonetary_exact(
    old_path: Path,
    new_path: Path,
    *,
    chunksize: int = 5000,
) -> dict[str, object]:
    old_columns = list(pd.read_csv(old_path, nrows=0).columns)
    new_columns = list(pd.read_csv(new_path, nrows=0).columns)
    common = [
        column
        for column in old_columns
        if column in new_columns and not _monetary_or_dependent(column)
    ]
    if not common:
        raise AssertionError("No nonmonetary columns were selected for invariance audit")
    old_iter = pd.read_csv(
        old_path,
        usecols=common,
        dtype=str,
        keep_default_na=False,
        chunksize=chunksize,
    )
    new_iter = pd.read_csv(
        new_path,
        usecols=common,
        dtype=str,
        keep_default_na=False,
        chunksize=chunksize,
    )
    rows_checked = 0
    mismatches = 0
    first_mismatch: dict[str, object] | None = None
    while True:
        try:
            old_chunk = next(old_iter)
        except StopIteration:
            old_chunk = None
        try:
            new_chunk = next(new_iter)
        except StopIteration:
            new_chunk = None
        if old_chunk is None or new_chunk is None:
            if old_chunk is not None or new_chunk is not None:
                raise AssertionError("Old/new scan chunk counts differ")
            break
        if len(old_chunk) != len(new_chunk):
            raise AssertionError("Old/new scan row counts differ within a chunk")
        equal = old_chunk.to_numpy() == new_chunk.to_numpy()
        if not bool(equal.all()):
            mismatch_positions = np.argwhere(~equal)
            mismatches += int(len(mismatch_positions))
            if first_mismatch is None:
                row_index, col_index = mismatch_positions[0]
                column = common[int(col_index)]
                first_mismatch = {
                    "row": int(rows_checked + row_index),
                    "column": column,
                    "old": old_chunk.iloc[int(row_index), int(col_index)],
                    "new": new_chunk.iloc[int(row_index), int(col_index)],
                }
        rows_checked += len(old_chunk)
    if mismatches:
        raise AssertionError(
            f"Nonmonetary bitwise-string audit failed: mismatches={mismatches}; "
            f"first={first_mismatch}"
        )
    required_exact = {
        "AF_ref_system",
        "r_cryo_re_fraction",
        "feasible_joint",
        *KEY_COLUMNS,
    }
    missing_required = sorted(required_exact - set(common))
    if missing_required:
        raise AssertionError(
            f"Required nonmonetary invariant columns were not compared: {missing_required}"
        )
    return {
        "rows_checked": rows_checked,
        "columns_checked": len(common),
        "columns": common,
        "cell_mismatches": mismatches,
        "first_mismatch": first_mismatch,
        "status": "PASS",
    }


def validate_accounting(frame: pd.DataFrame) -> dict[str, float]:
    opex_sum = frame[list(OPEX_COLUMNS)].sum(axis=1)
    opex_error = (opex_sum - frame["OPEX_total_modeled_USD_per_year"]).abs()
    if not np.allclose(
        opex_sum,
        frame["OPEX_total_modeled_USD_per_year"],
        rtol=1e-12,
        atol=1e-6,
    ):
        raise AssertionError(f"OPEX identity failed: max={opex_error.max()}")
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
        raise AssertionError(f"Annualized CAPEX identity failed: max={annualized_error.max()}")
    valid = (
        np.isfinite(frame[LCOE_COLUMN])
        & np.isfinite(frame["E_net_year_MWh"])
        & frame["E_net_year_MWh"].gt(0.0)
    )
    numerator_observed = frame.loc[valid, LCOE_COLUMN] * frame.loc[valid, "E_net_year_MWh"]
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
        raise AssertionError(f"LCOE numerator identity failed: max={numerator_error.max()}")
    return {
        "opex_identity_max_abs_2025USD_per_year": float(opex_error.max()),
        "annualized_capex_identity_max_abs_2025USD_per_year": float(annualized_error.max()),
        "lcoe_numerator_identity_max_abs_2025USD_per_year": float(numerator_error.max()),
    }


def sorted_optimum(feasible: pd.DataFrame) -> pd.Series:
    return feasible.sort_values(
        [LCOE_COLUMN, *DESIGN_COLUMNS], kind="mergesort"
    ).iloc[0]


def collect_metrics(frame: pd.DataFrame, label: str) -> tuple[list[dict[str, object]], pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    regret = compute_scenario_architecture_regret(frame)
    robust = compute_robust_architecture_window(regret)
    robust_summary = summarize_robust_windows(robust)
    for scenario in ALL_SCENARIOS:
        group = frame.loc[frame["scenario"].eq(scenario)]
        feasible = group.loc[group["feasible_joint"].astype(bool)].copy()
        optimum = sorted_optimum(feasible)
        row = {
            "model": label,
            "scenario": scenario,
            "jointly_feasible_count": int(len(feasible)),
            "minimum_LCOE_USD_per_MWh": float(optimum[LCOE_COLUMN]),
            "optimum_Top_K": float(optimum["Top_K"]),
            "optimum_coolant": str(optimum["coolant"]),
            "optimum_Npw": int(optimum["Npw"]),
            "optimum_rho_turn_uOhm_cm2": float(optimum["rho_turn_uOhm_cm2"]),
            "optimum_R_joint_nOhm": float(optimum["R_joint_nOhm"]),
            "full_grid_max_delta_LCOE_USD_per_MWh": float(
                feasible["delta_LCOE_joint_feasible_USD_per_MWh"].max()
            ),
        }
        fig5 = feasible.loc[
            np.isclose(feasible["rho_turn_uOhm_cm2"], 10000.0)
            & feasible["Npw"].ge(5)
        ]
        row["fig5_slice_max_delta_LCOE_USD_per_MWh"] = float(
            fig5["delta_LCOE_joint_feasible_USD_per_MWh"].max()
        )
        option_min = feasible.groupby(["Top_K", "coolant"])[LCOE_COLUMN].min()
        row["four_option_LCOE_span_USD_per_MWh"] = float(
            option_min.max() - option_min.min()
        )
        if label == COST_BOUNDARY_VERSION:
            total_opex = float(optimum["OPEX_total_modeled_USD_per_year"])
            for column in OPEX_COLUMNS:
                value = float(optimum[column])
                row[column] = value
                row[f"{column}_share"] = value / total_opex
        rows.append(row)
    for summary in robust_summary.itertuples(index=False):
        rows.append({
            "model": label,
            "scenario": "S1-S3",
            "robust_threshold_pct": float(summary.regret_threshold_pct),
            "robust_architecture_count": int(summary.robust_architecture_count),
            "robust_Npw_min": summary.Npw_min,
            "robust_Npw_max": summary.Npw_max,
            "robust_R_joint_nOhm_min": summary.R_joint_nOhm_min,
            "robust_R_joint_nOhm_max": summary.R_joint_nOhm_max,
        })
    return rows, regret, robust


def markdown_table(frame: pd.DataFrame) -> list[str]:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.8g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plant-v2", type=Path, required=True)
    parser.add_argument("--plant-v3", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    invariance = compare_nonmonetary_exact(args.plant_v2, args.plant_v3)
    old = normalize(args.plant_v2, OLD_BOUNDARY)
    new = normalize(args.plant_v3, COST_BOUNDARY_VERSION)
    if len(old) != len(new):
        raise AssertionError(f"Row count changed: v2={len(old)}, v3={len(new)}")
    accounting = validate_accounting(new)

    old_rows, _, _ = collect_metrics(old, OLD_BOUNDARY)
    new_rows, _, robust_new = collect_metrics(new, COST_BOUNDARY_VERSION)
    metrics = pd.DataFrame(old_rows + new_rows)
    metrics_path = args.output_dir / "economic_2025usd_change_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    conversion_path = args.output_dir / "economic_2025usd_conversion_audit.csv"
    pd.DataFrame(monetary_conversion_audit_rows()).to_csv(conversion_path, index=False)

    frontier = exact_robust_frontier_at_npw(
        robust_new, npw=200, threshold=0.05
    )
    frontier_path = args.output_dir / "robust_frontier_5pct_2025usd.csv"
    frontier.to_csv(frontier_path, index=False)

    feasible_counts = (
        metrics.loc[
            metrics["scenario"].isin(CENTRAL_SCENARIOS),
            ["model", "scenario", "jointly_feasible_count"],
        ]
        .pivot(index="scenario", columns="model", values="jointly_feasible_count")
        .reset_index()
    )
    if not (
        feasible_counts[OLD_BOUNDARY]
        == feasible_counts[COST_BOUNDARY_VERSION]
    ).all():
        raise AssertionError("Jointly feasible counts changed after monetary normalization")

    robust_rows = metrics.loc[
        metrics["scenario"].eq("S1-S3"),
        [
            "model",
            "robust_threshold_pct",
            "robust_architecture_count",
            "robust_Npw_min",
            "robust_Npw_max",
            "robust_R_joint_nOhm_min",
            "robust_R_joint_nOhm_max",
        ],
    ].dropna(subset=["robust_threshold_pct"])
    robust_5 = robust_rows.loc[
        robust_rows["model"].eq(COST_BOUNDARY_VERSION)
        & np.isclose(robust_rows["robust_threshold_pct"], 5.0)
    ]
    if len(robust_5) != 1:
        raise AssertionError("2025-USD 5% robust count/range was not uniquely reproduced")

    invariance_path = args.output_dir / "economic_2025usd_invariance_audit.json"
    invariance_payload = {
        "price_basis_year": PRICE_BASIS_YEAR,
        "old_scan": str(args.plant_v2),
        "old_sha256": sha256(args.plant_v2),
        "new_scan": str(args.plant_v3),
        "new_sha256": sha256(args.plant_v3),
        "old_rows": len(old),
        "new_rows": len(new),
        "nonmonetary_bitwise_string_comparison": invariance,
        "accounting": accounting,
        "jointly_feasible_counts_identical": True,
        "robust_5pct_reproduced": robust_5.to_dict(orient="records")[0],
    }
    invariance_path.write_text(
        json.dumps(invariance_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    scenario_summary = metrics.loc[
        metrics["scenario"].isin(ALL_SCENARIOS),
        [
            "model",
            "scenario",
            "jointly_feasible_count",
            "minimum_LCOE_USD_per_MWh",
            "optimum_Npw",
            "optimum_R_joint_nOhm",
            "optimum_rho_turn_uOhm_cm2",
            "optimum_Top_K",
            "optimum_coolant",
            "full_grid_max_delta_LCOE_USD_per_MWh",
            "fig5_slice_max_delta_LCOE_USD_per_MWh",
            "four_option_LCOE_span_USD_per_MWh",
        ],
    ]
    summary_path = args.output_dir / "economic_2025usd_change_summary.md"
    report = [
        "# Constant-2025-US$ economic rerun audit",
        "",
        f"- New cost boundary: `{COST_BOUNDARY_VERSION}`",
        f"- Plant-v2 scan SHA256: `{sha256(args.plant_v2)}`",
        f"- 2025-USD scan SHA256: `{sha256(args.plant_v3)}`",
        f"- Rows: {len(new):,}",
        f"- Nonmonetary exact comparison: PASS ({invariance['columns_checked']} columns, {invariance['rows_checked']:,} rows)",
        "- AF_ref, r_cryo, schedules, electromagnetic/cryogenic outputs and joint feasibility are unchanged.",
        "- No optimum coordinate or robust count was assumed; all were recomputed.",
        "",
        "## Scenario results",
        "",
        *markdown_table(scenario_summary),
        "",
        "## Robust architecture counts and ranges",
        "",
        *markdown_table(robust_rows),
        "",
        "## Exact 5% frontier at Npw=200",
        "",
        *markdown_table(frontier),
        "",
        "## Accounting closure",
        "",
        *[f"- {key}: `{value:.10g}`" for key, value in accounting.items()],
        "",
        "The exact conversion ledger is in `economic_2025usd_conversion_audit.csv`; the full machine-readable metrics are in `economic_2025usd_change_metrics.csv`.",
    ]
    summary_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    for path in (
        conversion_path,
        metrics_path,
        frontier_path,
        invariance_path,
        summary_path,
    ):
        print(f"[OK] {path}")


if __name__ == "__main__":
    main()