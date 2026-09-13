#!/usr/bin/env python3
"""Build the canonical V7 publication-data release from a frozen science run.

The scientific run remains immutable.  This script creates a compact publication
layer containing (1) panel-ready figure inputs, (2) renderer-specific inputs that
must not be recomputed inside plotting code, and (3) a machine-readable registry
of the numbers used in the manuscript.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCIENCE = ROOT / "scientific_results_v7_splice_equivalent_20260906"
DEFAULT_RELEASE_ID = "20260906_splice_equivalent"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_file(source: Path, destination: Path, records: list[dict], role: str) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    source_hash = sha256(source)
    destination_hash = sha256(destination)
    if source_hash != destination_hash:
        raise RuntimeError(f"COPY_HASH_MISMATCH:{source}:{destination}")
    records.append(
        {
            "role": role,
            "source": str(source.resolve()),
            "release_path": destination.as_posix(),
            "bytes": destination.stat().st_size,
            "sha256": destination_hash,
        }
    )


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def assert_audits(science: Path) -> list[dict]:
    required = [
        "stage_A/v6_2_a_availability_audit.json",
        "stage_B1/v6_2_b1_and_strict_feasibility_audit.json",
        "stage_C_fast_r2/v6_2_c_direct_bisection_audit.json",
        "stage_E_fast/v6e_cross_scenario_audit.json",
        "stage_H/v6_2_h_audit.json",
        "experiments/capital_anchor_scale_sensitivity_v6_8/CAPITAL_ANCHOR_SCALE_SENSITIVITY_AUDIT_V6_8.json",
        "experiments/coverage_measure_sensitivity_v6_8/COVERAGE_MEASURE_SENSITIVITY_AUDIT_V6_8.json",
        "experiments/fixed_implementation_robustness_v6_8/FIXED_IMPLEMENTATION_ROBUSTNESS_AUDIT_V6_8.json",
    ]
    audits = []
    for relative in required:
        path = science / relative
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "PASS":
            raise RuntimeError(f"SCIENTIFIC_AUDIT_NOT_PASS:{path}")
        audits.append({"path": str(path.resolve()), "sha256": sha256(path), "status": "PASS"})
    return audits


def materialize_fig1(science: Path, approved: Path) -> None:
    main = science / "figure_panel_data" / "main"
    b = pd.read_csv(main / "Fig1B_charging_time_v6_7.csv")
    b.insert(0, "Top_K", 10.0)
    c = pd.read_parquet(main / "Fig1C_annual_availability_v6_7.parquet")
    c.insert(1, "Top_K", 10.0)
    d = pd.read_csv(main / "Fig1D_availability_boundary_v6_7.csv")
    out = approved / "Fig1"
    out.mkdir(parents=True, exist_ok=True)
    b.to_csv(out / "Fig1_B_panel.csv", index=False)
    c.to_csv(out / "Fig1_C_panel.csv", index=False)
    d.to_csv(out / "Fig1_D_panel.csv", index=False)


def materialize_fig2(science: Path, render_root: Path) -> None:
    """Create the actual Fig. 2 renderer inputs from the physical ledger.

    The legacy Fig2B panel file contains a different quantity boundary.  The
    manuscript panel uses cold-end pulse heat load in W per TF, reconstructed
    from the physical component ledger and converted to kW only by the renderer.
    """
    main = science / "figure_panel_data" / "main"
    ledger_path = science / "publication_support" / "B_heatload_physical_ledger.parquet"
    identity = ["Top_K", "coolant", "scenario", "Npw", "rho_turn_uOhm_cm2", "R_joint_nOhm"]
    components = [
        "Q_coil_internal_joint_W",
        "Q_pancake_joint_W",
        "Q_nuclear_W",
        "Q_radiation_W",
        "Q_pipes_coolant_W",
        "Q_pipes_aux_W",
        "Q_quench_W",
        "Q_misc_W",
    ]
    ledger = pd.read_parquet(
        ledger_path,
        columns=[*identity, *components],
        filters=[("Top_K", "==", 10.0), ("coolant", "==", "He"), ("rho_turn_uOhm_cm2", "==", 10000.0)],
    )
    selected = ledger[
        np.isclose(ledger["Top_K"], 10.0)
        & ledger["coolant"].eq("He")
        & np.isclose(ledger["rho_turn_uOhm_cm2"], 10000.0)
    ].copy()
    grouped = selected.groupby(["Npw", "R_joint_nOhm"], sort=True, as_index=False)
    first = grouped.first()
    counts = grouped.size().rename(columns={"size": "scenario_replica_count"})
    panel_b = first.merge(counts, on=["Npw", "R_joint_nOhm"], validate="one_to_one")
    panel_b["Q_joint_W_per_TF"] = panel_b["Q_coil_internal_joint_W"] + panel_b["Q_pancake_joint_W"]
    background = ["Q_radiation_W", "Q_pipes_coolant_W", "Q_pipes_aux_W", "Q_quench_W", "Q_misc_W"]
    panel_b["Q_background_W_per_TF"] = panel_b[background].sum(axis=1)
    panel_b["Q_total_pulse_W_per_TF"] = (
        panel_b["Q_nuclear_W"] + panel_b["Q_joint_W_per_TF"] + panel_b["Q_background_W_per_TF"]
    )
    rename = {column: f"{column}_per_TF" for column in components}
    panel_b = panel_b.rename(columns=rename)
    columns = [
        "Top_K", "coolant", "rho_turn_uOhm_cm2", "Npw", "R_joint_nOhm",
        "Q_coil_internal_joint_W_per_TF", "scenario_replica_count",
        "Q_pancake_joint_W_per_TF", "Q_joint_W_per_TF", "Q_nuclear_W_per_TF",
        "Q_radiation_W_per_TF", "Q_pipes_coolant_W_per_TF", "Q_pipes_aux_W_per_TF",
        "Q_quench_W_per_TF", "Q_misc_W_per_TF", "Q_background_W_per_TF",
        "Q_total_pulse_W_per_TF",
    ]
    panel_b = panel_b[columns].sort_values(["Npw", "R_joint_nOhm"]).reset_index(drop=True)
    if len(panel_b) != 24_200 or panel_b[["Npw", "R_joint_nOhm"]].duplicated().any():
        raise RuntimeError("FIG2B_NATIVE_GRID_CONTRACT_FAIL")
    residual = (
        panel_b["Q_total_pulse_W_per_TF"]
        - panel_b["Q_nuclear_W_per_TF"]
        - panel_b["Q_joint_W_per_TF"]
        - panel_b["Q_background_W_per_TF"]
    ).abs().max()
    if float(residual) > 1e-9:
        raise RuntimeError(f"FIG2B_COMPONENT_SUM_FAIL:{residual}")
    out = render_root / "Fig2"
    out.mkdir(parents=True, exist_ok=True)
    panel_b.to_parquet(out / "panel_B.parquet", index=False)
    pd.read_csv(main / "Fig2C_radial_charging_loss_v6_7.csv").to_parquet(out / "panel_C.parquet", index=False)
    pd.read_csv(main / "Fig2D_annual_refrigeration_burden_summary_v6_7.csv").to_parquet(out / "panel_D.parquet", index=False)


def materialize_fig4(science: Path, approved: Path) -> None:
    source = science / "stage_F2" / "v6_2_f2_price_architecture_grid.parquet"
    minima = science / "figure_panel_data" / "main" / "Fig3B_absolute_headroom_v6_7.csv"
    grid = pd.read_parquet(source)
    minimum = float(pd.read_csv(minima).set_index("scenario").loc["S2", "minimum_lcoe_USD_MWh"])
    panel = grid[np.isclose(grid["HTS_price_USD_per_kAm"], 50.0) & grid["Npw"].isin([50, 200])].copy()
    panel["global_relative_penalty_pct"] = 100.0 * (panel["lcoe_price"] / minimum - 1.0)
    panel = panel.sort_values(["Npw", "Top_K", "R_joint_nOhm"]).reset_index(drop=True)
    expected = {(50, 4.2): 121, (50, 10.0): 121, (50, 20.0): 121,
                (200, 4.2): 109, (200, 10.0): 121, (200, 20.0): 121}
    observed = {(int(n), float(t)): int(k) for (n, t), k in panel.groupby(["Npw", "Top_K"]).size().items()}
    if observed != expected:
        raise RuntimeError(f"FIG4A_SLICE_COUNT_FAIL:{observed}")
    out = approved / "Fig4"
    out.mkdir(parents=True, exist_ok=True)
    target = out / "Fig4A_joint_sensitivity_Npw50_200_v7.csv"
    panel.to_csv(target, index=False)
    write_json(
        target.with_suffix(".audit.json"),
        {
            "status": "PASS",
            "source": str(source.resolve()),
            "source_sha256": sha256(source),
            "minimum_source": str(minima.resolve()),
            "minimum_source_sha256": sha256(minima),
            "S2_global_minimum_lcoe_USD_MWh": minimum,
            "rows": len(panel),
            "output_sha256": sha256(target),
        },
    )


def metric(rows: list[dict], section: str, metric_id: str, condition: str, value: float | int | str,
           unit: str, display: str, source: Path) -> None:
    rows.append(
        {
            "section": section,
            "metric_id": metric_id,
            "condition": condition,
            "value": value,
            "unit": unit,
            "manuscript_display": display,
            "source_file": str(source.resolve()),
            "source_sha256": sha256(source),
        }
    )


def build_key_data(science: Path, render_root: Path, key_root: Path) -> list[dict]:
    rows: list[dict] = []
    main = science / "figure_panel_data" / "main"

    p = science / "stage_A" / "v6_2_a_Aplant_max_by_scenario.csv"
    for row in pd.read_csv(p).itertuples(index=False):
        metric(rows, "Results 1", "plant_availability_max", row.scenario, float(row.Aplant_max_scenario), "fraction", f"{100*row.Aplant_max_scenario:.1f}%", p)
    p = main / "Fig1B_charging_time_v6_7.csv"
    charge = pd.read_csv(p)
    metric(rows, "Results 1", "charging_time_min", "full Npw-rho grid", float(charge.Charging_time_999_h.min()), "h", f"{charge.Charging_time_999_h.min():.1f}", p)
    metric(rows, "Results 1", "charging_time_max", "full Npw-rho grid", float(charge.Charging_time_999_h.max()), "h", f"{charge.Charging_time_999_h.max():.2e}", p)

    p = render_root / "Fig2" / "panel_B.parquet"
    heat = pd.read_parquet(p)
    metric(rows, "Results 2", "pulse_heat_load_min", "10 K; He; rho_turn=10000", float(heat.Q_total_pulse_W_per_TF.min()/1000), "kW/TF", f"{heat.Q_total_pulse_W_per_TF.min()/1000:.2f}", p)
    metric(rows, "Results 2", "pulse_heat_load_max", "10 K; He; rho_turn=10000", float(heat.Q_total_pulse_W_per_TF.max()/1000), "kW/TF", f"{heat.Q_total_pulse_W_per_TF.max()/1000:.1f}", p)
    p = main / "Fig2D_annual_refrigeration_burden_summary_v6_7.csv"
    refrigeration = pd.read_csv(p)
    metric(rows, "Abstract/Results 2", "refrigeration_fraction_min", "availability-qualified designs", float(refrigeration.minimum.min()), "% gross generation", f"{refrigeration.minimum.min():.1f}%", p)
    metric(rows, "Abstract/Results 2", "refrigeration_fraction_max", "availability-qualified designs", float(refrigeration.maximum.max()), "% gross generation", f"{refrigeration.maximum.max():.1f}%", p)
    for row in refrigeration.itertuples(index=False):
        metric(rows, "Results 2", "refrigeration_fraction_median", f"{row.scenario}; {row.Top_K:g} K", float(row.median), "% gross generation", f"{row.median:.1f}%", p)

    p = main / "Fig3B_absolute_headroom_v6_7.csv"
    headroom = pd.read_csv(p)
    for row in headroom.itertuples(index=False):
        metric(rows, "Results 3", "minimum_lcoe", row.scenario, float(row.minimum_lcoe_USD_MWh), "2025 US$/MWh", f"{row.minimum_lcoe_USD_MWh:.0f}", p)
        metric(rows, "Results 3", "lcoe_margin_5pct", row.scenario, float(row.headroom_5pct_USD_MWh), "2025 US$/MWh", f"{row.headroom_5pct_USD_MWh:.1f}", p)
    p = science / "experiments" / "coverage_measure_sensitivity_v6_8" / "coverage_measure_summary_v6_8.csv"
    coverage = pd.read_csv(p)
    primary = coverage[coverage.weighting_method.eq("log_Rj_interval")]
    for threshold in (1, 5, 10):
        row = primary[np.isclose(primary.penalty_threshold_pct, threshold)].iloc[0]
        metric(rows, "Abstract/Results 3", "weighted_design_space_coverage", f"allowed LCOE increase={threshold}%", float(row.C_total_full_denominator_pct), "%", f"{row.C_total_full_denominator_pct:.1f}%", p)
    row = primary[np.isclose(primary.penalty_threshold_pct, 5)].iloc[0]
    metric(rows, "Results 3", "shared_feasible_space", "S1-S3", float(row.C_feasible_pct), "%", f"{row.C_feasible_pct:.1f}%", p)
    metric(rows, "Results 3", "near_optimal_within_shared_feasible", "5% criterion", float(row.C_near_given_feasible_pct), "%", f"{row.C_near_given_feasible_pct:.1f}%", p)

    p = main / "Fig4B_joint_resistance_tolerance_v6_7.csv"
    boundary = pd.read_csv(p)
    for temperature in (4.2, 10.0, 20.0):
        row = boundary[np.isclose(boundary.Top_K, temperature) & boundary.Npw.eq(200)].iloc[0]
        value = float(row.Rj_max_global_5pct_nOhm)
        metric(rows, "Abstract/Results 4", "maximum_allowable_joint_resistance", f"S2; Npw=200; {temperature:g} K; He", value, "nOhm", f"{value:.2f}" if temperature == 4.2 else f"{value:.1f}", p)

    p = science / "experiments" / "fixed_implementation_robustness_v6_8" / "fixed_implementation_summary_v6_8.csv"
    fixed = pd.read_csv(p)
    row = fixed[(fixed.analysis_level == "fixed_full_5D_design_x") & (fixed.weighting_method == "log_Rj_interval") & np.isclose(fixed.penalty_threshold_pct, 5)].iloc[0]
    metric(rows, "Discussion/SI", "fixed_5d_coverage", "5% criterion; log-Rj weighting", float(row.C_total_full_denominator_pct), "%", f"{row.C_total_full_denominator_pct:.1f}%", p)
    p = science / "experiments" / "capital_anchor_scale_sensitivity_v6_8" / "capital_anchor_scale_summary_v6_8.csv"
    capital = pd.read_csv(p)
    for row in capital.itertuples(index=False):
        metric(rows, "Discussion/SI", "capital_anchor_coverage_5pct", f"alpha={row.anchor_scale:g}", float(row.coverage_5pct_log_Rj_full_denominator_pct), "%", f"{row.coverage_5pct_log_Rj_full_denominator_pct:.1f}%", p)
        metric(rows, "Discussion/SI", "capital_anchor_joint_boundary", f"alpha={row.anchor_scale:g}; Npw=200", float(row.Rj_max_global_5pct_nOhm), "nOhm", f"{row.Rj_max_global_5pct_nOhm:.1f}", p)

    key_root.mkdir(parents=True, exist_ok=True)
    csv_path = key_root / "V7_KEY_DATA.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(key_root / "V7_KEY_DATA.json", {"schema": "v7-key-data/1.0", "records": rows})
    lines = [
        "# V7 论文关键数据包",
        "",
        "本目录由冻结科学结果自动生成。`V7_KEY_DATA.csv` 供人工核对，`V7_KEY_DATA.json` 供脚本读取。",
        "数值列保存未四舍五入结果，`manuscript_display` 保存论文显示精度。任何正文数字修改必须先更新模型结果，再重建本包。",
        "",
        "| 章节 | 指标 | 条件 | 数值 | 单位 | 稿件显示 |",
        "|---|---|---|---:|---|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['section']} | {row['metric_id']} | {row['condition']} | {row['value']} | {row['unit']} | {row['manuscript_display']} |")
    (key_root / "README_CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--science-root", type=Path, default=DEFAULT_SCIENCE)
    parser.add_argument("--release-id", default=DEFAULT_RELEASE_ID)
    args = parser.parse_args()
    science = args.science_root.resolve()
    release = ROOT / "publication_data" / "releases" / args.release_id
    figure_data = release / "figure_data"
    key_data = release / "key_data"
    records: list[dict] = []
    audits = assert_audits(science)

    for subset in ("main", "supplementary"):
        source_root = science / "figure_panel_data" / subset
        for source in sorted(source_root.glob("*")):
            if source.is_file():
                copy_file(source, figure_data / "figure_panel_data" / subset / source.name, records, f"scientific panel data: {subset}")

    approved = figure_data / "approved_panel_data"
    render_inputs = figure_data / "render_inputs"
    materialize_fig1(science, approved)
    materialize_fig2(science, render_inputs)
    materialize_fig4(science, approved)
    key_rows = build_key_data(science, render_inputs, key_data)

    generated = []
    for path in sorted(release.rglob("*")):
        if path.is_file() and path.name not in {"MANIFEST.json"}:
            generated.append({"path": path.relative_to(release).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)})
    manifest = {
        "schema": "v7-publication-data-release/1.0",
        "status": "PASS",
        "release_id": args.release_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_root": str(science),
        "scientific_audits": audits,
        "copied_scientific_files": records,
        "key_metric_count": len(key_rows),
        "files": generated,
    }
    write_json(release / "MANIFEST.json", manifest)
    active = {
        "schema": "v7-active-publication-data/1.0",
        "status": "PASS",
        "active_release_id": args.release_id,
        "active_release": str(release.resolve()),
        "scientific_root": str(science),
        "manifest_sha256": sha256(release / "MANIFEST.json"),
    }
    write_json(ROOT / "publication_data" / "ACTIVE_RELEASE.json", active)
    print(json.dumps({"status": "PASS", "release": str(release), "files": len(generated), "key_metrics": len(key_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
