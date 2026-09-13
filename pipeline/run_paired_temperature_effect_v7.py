#!/usr/bin/env python3
"""Paired 4.2 K He versus 20 K He temperature-effect reducer for V7.

The reducer holds (Npw, rho_turn, Rj) fixed, uses the frozen scenario-global
LCOE references, and reads only the already materialized complete-realization
ledgers.  It never calls a physical, cryogenic, circuit, or economic model.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
SCI = ROOT / "scientific_results_v7_splice_equivalent_20260906"
PARENT = SCI / "experiments" / "full_realization_robustness_matrix_v7"
PARENT_AUDIT = PARENT / "FULL_REALIZATION_ROBUSTNESS_MATRIX_V7.json"
LEDGERS = PARENT / "realization_ledgers"
SOURCE_4P2 = LEDGERS / "realizations_T4p2_He.parquet"
SOURCE_20 = LEDGERS / "realizations_T20p0_He.parquet"
OUT = SCI / "experiments" / "paired_temperature_effect_v7"

SCENARIOS = ("S1", "S2", "S3")
BASELINES = {"S1": 1122.1472612313785, "S2": 321.97916120106885, "S3": 78.82074592882216}
EPS = (0.01, 0.05, 0.10, 0.20)
KEYS = ["Npw", "rho_turn_uOhm_cm2", "R_joint_nOhm"]
EXPECTED_PAIRS = 200 * 61 * 121


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def source_gate() -> dict:
    for path in (PARENT_AUDIT, SOURCE_4P2, SOURCE_20):
        if not path.is_file():
            raise RuntimeError(f"STOP_MISSING_SOURCE: {path}")
    audit = json.loads(PARENT_AUDIT.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS":
        raise RuntimeError("STOP_PARENT_AUDIT_NOT_PASS")
    for scenario in SCENARIOS:
        if not np.isclose(float(audit["baselines"][scenario]), BASELINES[scenario], atol=1e-12, rtol=0):
            raise RuntimeError(f"STOP_FROZEN_BASELINE_DRIFT: {scenario}")
    if audit.get("validation", {}).get("merged_realization_rows") != 5_904_800:
        raise RuntimeError("STOP_PARENT_DENOMINATOR_NOT_AUDITED")
    return audit


def load_temperature(path: Path, tag: str) -> pd.DataFrame:
    value_cols = []
    for scenario in SCENARIOS:
        value_cols.extend([
            f"Aplant_{scenario}", f"E_net_year_MWh_{scenario}",
            f"lcoe_anchor_USD_per_MWh_{scenario}", f"r_cryo_re_fraction_{scenario}",
        ])
    stored = ["availability_pass_all", "economic_valid_all", "delta_max"]
    frame = pd.read_parquet(path, columns=KEYS + value_cols + stored)
    if len(frame) != EXPECTED_PAIRS:
        raise RuntimeError(f"STOP_PAIR_SOURCE_ROW_COUNT: {path}={len(frame)}")
    if frame.duplicated(KEYS).any():
        raise RuntimeError(f"STOP_DUPLICATE_PAIR_KEY: {path}")
    axes = tuple(frame[key].nunique() for key in KEYS)
    if axes != (200, 61, 121):
        raise RuntimeError(f"STOP_PAIR_AXIS_CARDINALITY: {path}={axes}")
    rename = {column: f"{column}_{tag}" for column in value_cols}
    rename.update({column: f"stored_{column}_{tag}" for column in stored})
    return frame.rename(columns=rename)


def build_pairs() -> pd.DataFrame:
    low = load_temperature(SOURCE_4P2, "4p2K")
    high = load_temperature(SOURCE_20, "20K")
    pairs = low.merge(high, on=KEYS, how="inner", validate="one_to_one")
    if len(pairs) != EXPECTED_PAIRS or pairs.duplicated(KEYS).any():
        raise RuntimeError("STOP_PAIR_JOIN_CONTRACT")

    for tag in ("4p2K", "20K"):
        availability_cols, valid_cols, delta_cols = [], [], []
        for scenario in SCENARIOS:
            availability = np.isfinite(pairs[f"Aplant_{scenario}_{tag}"]) & pairs[f"Aplant_{scenario}_{tag}"].ge(0.80)
            valid = (
                np.isfinite(pairs[f"E_net_year_MWh_{scenario}_{tag}"])
                & pairs[f"E_net_year_MWh_{scenario}_{tag}"].gt(0)
                & np.isfinite(pairs[f"lcoe_anchor_USD_per_MWh_{scenario}_{tag}"])
                & pairs[f"lcoe_anchor_USD_per_MWh_{scenario}_{tag}"].gt(0)
            )
            pairs[f"availability_pass_{scenario}_{tag}"] = availability
            pairs[f"economic_valid_{scenario}_{tag}"] = valid
            pairs[f"delta_{scenario}_{tag}"] = np.where(
                valid,
                pairs[f"lcoe_anchor_USD_per_MWh_{scenario}_{tag}"] / BASELINES[scenario] - 1.0,
                np.nan,
            )
            availability_cols.append(f"availability_pass_{scenario}_{tag}")
            valid_cols.append(f"economic_valid_{scenario}_{tag}")
            delta_cols.append(f"delta_{scenario}_{tag}")
        pairs[f"availability_pass_all_{tag}"] = pairs[availability_cols].all(axis=1)
        pairs[f"economic_valid_all_{tag}"] = pairs[valid_cols].all(axis=1)
        pairs[f"delta_max_{tag}"] = pairs[delta_cols].max(axis=1)
        pairs.loc[~pairs[f"economic_valid_all_{tag}"], f"delta_max_{tag}"] = np.inf

        if not pairs[f"availability_pass_all_{tag}"].equals(pairs[f"stored_availability_pass_all_{tag}"].astype(bool)):
            raise RuntimeError(f"STOP_PARENT_AVAILABILITY_REGRESSION: {tag}")
        if not pairs[f"economic_valid_all_{tag}"].equals(pairs[f"stored_economic_valid_all_{tag}"].astype(bool)):
            raise RuntimeError(f"STOP_PARENT_VALIDITY_REGRESSION: {tag}")
        if not np.allclose(
            pairs[f"delta_max_{tag}"].to_numpy(float),
            pairs[f"stored_delta_max_{tag}"].to_numpy(float),
            atol=1e-14, rtol=1e-13, equal_nan=True,
        ):
            raise RuntimeError(f"STOP_PARENT_DELTA_REGRESSION: {tag}")

    pairs["YT1_both_temperature_availability"] = (
        pairs["availability_pass_all_4p2K"] & pairs["availability_pass_all_20K"]
    )
    pairs["YT2_both_temperature_economic_valid"] = (
        pairs["YT1_both_temperature_availability"]
        & pairs["economic_valid_all_4p2K"] & pairs["economic_valid_all_20K"]
    )
    for scenario in SCENARIOS:
        denominator = pairs[f"r_cryo_re_fraction_{scenario}_4p2K"]
        numerator = pairs[f"r_cryo_re_fraction_{scenario}_20K"]
        valid_ratio = np.isfinite(denominator) & denominator.gt(0) & np.isfinite(numerator) & numerator.ge(0)
        pairs[f"paired_rcryo_valid_{scenario}"] = valid_ratio
        pairs[f"paired_rcryo_reduction_{scenario}"] = np.where(valid_ratio, 1.0 - numerator / denominator, np.nan)
    return pairs


def population_row(name: str, mask: pd.Series, yt1_n: int, yt2_n: int, epsilon: float | None = None) -> dict:
    n = int(mask.sum())
    fraction_yt1 = None if name == "YT0" else 100 * n / yt1_n
    fraction_yt2 = 100 * n / yt2_n if name == "YT2" or name.startswith("YT-") else None
    return {
        "population": name,
        "epsilon_pct": None if epsilon is None else 100 * epsilon,
        "N_pairs": n,
        "fraction_YT0_pct": 100 * n / EXPECTED_PAIRS,
        "fraction_YT1_pct": fraction_yt1,
        "fraction_YT2_pct": fraction_yt2,
    }


def calculate(pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    yt0 = pd.Series(True, index=pairs.index)
    yt1 = pairs["YT1_both_temperature_availability"]
    yt2 = pairs["YT2_both_temperature_economic_valid"]
    yt1_n, yt2_n = int(yt1.sum()), int(yt2.sum())
    if not (0 < yt2_n <= yt1_n <= EXPECTED_PAIRS):
        raise RuntimeError("STOP_YT_POPULATION_ORDER")

    population_rows = [
        population_row("YT0", yt0, yt1_n, yt2_n),
        population_row("YT1", yt1, yt1_n, yt2_n),
        population_row("YT2", yt2, yt1_n, yt2_n),
    ]
    economic_rows = []
    for epsilon in EPS:
        pass_4p2 = yt2 & pairs["delta_max_4p2K"].le(epsilon)
        pass_20 = yt2 & pairs["delta_max_20K"].le(epsilon)
        rescue = yt2 & ~pairs["delta_max_4p2K"].le(epsilon) & pairs["delta_max_20K"].le(epsilon)
        loss = yt2 & pairs["delta_max_4p2K"].le(epsilon) & ~pairs["delta_max_20K"].le(epsilon)
        both = pass_4p2 & pass_20
        for label, mask in (("YT-4p2", pass_4p2), ("YT-20", pass_20), ("YT-rescue", rescue)):
            population_rows.append(population_row(f"{label}-{100*epsilon:g}pct", mask, yt1_n, yt2_n, epsilon))
        economic_rows.append({
            "epsilon_pct": 100 * epsilon,
            "YT2_denominator_pairs": yt2_n,
            "N_pass_4p2K": int(pass_4p2.sum()),
            "fraction_pass_4p2K_pct": 100 * pass_4p2.sum() / yt2_n,
            "N_pass_20K": int(pass_20.sum()),
            "fraction_pass_20K_pct": 100 * pass_20.sum() / yt2_n,
            "increase_20K_minus_4p2K_percentage_points": 100 * (pass_20.sum() - pass_4p2.sum()) / yt2_n,
            "N_rescue_fail_4p2K_pass_20K": int(rescue.sum()),
            "rescue_fraction_of_YT2_pct": 100 * rescue.sum() / yt2_n,
            "N_loss_pass_4p2K_fail_20K": int(loss.sum()),
            "loss_fraction_of_YT2_pct": 100 * loss.sum() / yt2_n,
            "N_both_pass": int(both.sum()),
        })
        pairs[f"pass_4p2K_{100*epsilon:g}pct"] = pass_4p2
        pairs[f"pass_20K_{100*epsilon:g}pct"] = pass_20
        pairs[f"rescue_{100*epsilon:g}pct"] = rescue

    refrigeration_rows = []
    for scenario in SCENARIOS:
        valid = yt1 & pairs[f"paired_rcryo_valid_{scenario}"]
        if int(valid.sum()) != yt1_n:
            raise RuntimeError(f"STOP_INCOMPLETE_YT1_RCRYO_PAIR: {scenario}")
        r4 = pairs.loc[valid, f"r_cryo_re_fraction_{scenario}_4p2K"].to_numpy(float)
        r20 = pairs.loc[valid, f"r_cryo_re_fraction_{scenario}_20K"].to_numpy(float)
        reduction = pairs.loc[valid, f"paired_rcryo_reduction_{scenario}"].to_numpy(float)
        q = np.quantile(reduction, [0.10, 0.50, 0.90], method="linear")
        refrigeration_rows.append({
            "scenario": scenario,
            "YT1_denominator_pairs": yt1_n,
            "valid_paired_rcryo_pairs": int(len(reduction)),
            "median_rcryo_4p2K_pct": 100 * np.median(r4),
            "median_rcryo_20K_pct": 100 * np.median(r20),
            "paired_reduction_P10_pct": 100 * q[0],
            "paired_reduction_median_pct": 100 * q[1],
            "paired_reduction_P90_pct": 100 * q[2],
            "fraction_rcryo_20K_lt_4p2K_pct": 100 * np.mean(r20 < r4),
            "reduction_from_ratio_of_medians_pct": 100 * (1.0 - np.median(r20) / np.median(r4)),
        })

    economic_summary = {
        "YT2_denominator_pairs": yt2_n,
        "median_delta_max_4p2K_pct": 100 * pairs.loc[yt2, "delta_max_4p2K"].median(),
        "median_delta_max_20K_pct": 100 * pairs.loc[yt2, "delta_max_20K"].median(),
    }
    economic = pd.DataFrame(economic_rows)
    if not economic["N_pass_4p2K"].is_monotonic_increasing or not economic["N_pass_20K"].is_monotonic_increasing:
        raise RuntimeError("STOP_THRESHOLD_MONOTONICITY")
    residual = (
        economic["N_pass_20K"] - economic["N_pass_4p2K"]
        - economic["N_rescue_fail_4p2K_pass_20K"] + economic["N_loss_pass_4p2K_fail_20K"]
    ).abs().max()
    if int(residual) != 0:
        raise RuntimeError("STOP_RESCUE_LOSS_IDENTITY")
    return pd.DataFrame(population_rows), pd.DataFrame(refrigeration_rows), economic, economic_summary


def main() -> int:
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"STOP_NONEMPTY_OUTPUT: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    parent_audit = source_gate()
    print("[1/4] load and pair 4.2 K He with 20 K He", flush=True)
    pairs = build_pairs()
    print("[2/4] calculate YT populations and XT statistics", flush=True)
    populations, refrigeration, economic, economic_summary = calculate(pairs)

    populations.to_csv(OUT / "temperature_population_matrix.csv", index=False, float_format="%.17g")
    refrigeration.to_csv(OUT / "temperature_refrigeration_paired_stats.csv", index=False, float_format="%.17g")
    economic.to_csv(OUT / "temperature_economic_effect_matrix.csv", index=False, float_format="%.17g")
    ledger_cols = KEYS + [
        "YT1_both_temperature_availability", "YT2_both_temperature_economic_valid",
        "delta_max_4p2K", "delta_max_20K",
    ]
    for scenario in SCENARIOS:
        ledger_cols.extend([
            f"r_cryo_re_fraction_{scenario}_4p2K", f"r_cryo_re_fraction_{scenario}_20K",
            f"paired_rcryo_reduction_{scenario}",
        ])
    for epsilon in EPS:
        ledger_cols.extend([
            f"pass_4p2K_{100*epsilon:g}pct", f"pass_20K_{100*epsilon:g}pct", f"rescue_{100*epsilon:g}pct",
        ])
    ledger_path = OUT / "paired_temperature_analysis_ledger.parquet"
    pairs[ledger_cols].sort_values(KEYS).to_parquet(ledger_path, index=False, compression="zstd")
    if pq.ParquetFile(ledger_path).metadata.num_rows != EXPECTED_PAIRS:
        raise RuntimeError("STOP_OUTPUT_LEDGER_ROW_COUNT")

    yt1 = populations.loc[populations["population"].eq("YT1")].iloc[0].to_dict()
    yt2 = populations.loc[populations["population"].eq("YT2")].iloc[0].to_dict()
    row10 = economic.loc[np.isclose(economic["epsilon_pct"], 10.0)].iloc[0].to_dict()
    audit = {
        "status": "PASS",
        "experiment": "V7 paired temperature effect: 4.2 K He versus 20 K He",
        "pair_definition": "same (Npw,rho_turn,Rj); coolant fixed to He; only Top changes",
        "population_definitions": {
            "YT0": "all 200x61x121 paired configurations",
            "YT1": "YT0 with both temperatures satisfying Aplant>=0.80 in S1-S3",
            "YT2": "YT1 with E_net>0 and finite positive LCOE at both temperatures in S1-S3",
            "temperature_retention": "within YT2, delta_max at the stated temperature <= epsilon",
            "rescue": "within YT2, 4.2 K fails and 20 K passes the same epsilon",
        },
        "frozen_global_LCOE_baselines_USD_per_MWh": BASELINES,
        "temperature_specific_reference_recalculation": False,
        "primary_measure": "equal weight per paired sampled configuration",
        "refrigeration_population": "YT1; scenario-specific paired records",
        "economic_population": "YT2; delta_max across S1-S3",
        "results": {
            "YT1": yt1,
            "YT2": yt2,
            "refrigeration": refrigeration.to_dict("records"),
            "economic_summary": economic_summary,
            "economic_10pct": row10,
        },
        "validation": {
            "input_axis_cardinality": {"Npw": 200, "rho_turn": 61, "Rj": 121},
            "one_to_one_pair_count": EXPECTED_PAIRS,
            "parent_availability_validity_delta_regression": "PASS",
            "threshold_monotonicity": "PASS",
            "rescue_loss_identity": "PASS",
            "output_ledger_rows": EXPECTED_PAIRS,
        },
        "source": {
            "parent_audit": str(PARENT_AUDIT.relative_to(ROOT)).replace("\\", "/"),
            "parent_audit_sha256": sha256(PARENT_AUDIT),
            "parent_B1_source_sha256": parent_audit["source"]["sha256"],
            "ledger_4p2K_He_sha256": sha256(SOURCE_4P2),
            "ledger_20K_He_sha256": sha256(SOURCE_20),
        },
        "scientific_model_rerun": False,
        "formal_results_modified": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUT / "PAIRED_TEMPERATURE_EFFECT_V7.json").write_text(
        json.dumps(json_safe(audit), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    report = [
        "# V7 配对温度效应：4.2 K He vs 20 K He\n",
        "状态：**PASS**。固定 `(Npw,rho_turn,Rj)` 和 He，仅改变温度；未重跑科学模型。\n",
        f"- YT0：{EXPECTED_PAIRS:,} 对",
        f"- YT1：{int(yt1['N_pairs']):,} 对，两温度均在 S1--S3 满足 `Aplant>=0.80`",
        f"- YT2：{int(yt2['N_pairs']):,} 对，并且两温度在 S1--S3 均有正净发电和有限正 LCOE",
        f"- YT2 的 median delta_max：4.2 K = {economic_summary['median_delta_max_4p2K_pct']:.8f}%，20 K = {economic_summary['median_delta_max_20K_pct']:.8f}%",
        f"- 10% retention：4.2 K = {row10['fraction_pass_4p2K_pct']:.8f}%，20 K = {row10['fraction_pass_20K_pct']:.8f}%，增加 {row10['increase_20K_minus_4p2K_percentage_points']:.8f} percentage points",
        f"- 10% rescue fraction：{row10['rescue_fraction_of_YT2_pct']:.8f}%（分母 YT2）\n",
        "## Refrigeration（YT1）\n",
        "| Scenario | median rcryo 4.2 K | median rcryo 20 K | paired reduction P10 | median | P90 | 20 K lower fraction |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in refrigeration.to_dict("records"):
        report.append(
            f"| {row['scenario']} | {row['median_rcryo_4p2K_pct']:.8f}% | {row['median_rcryo_20K_pct']:.8f}% | "
            f"{row['paired_reduction_P10_pct']:.8f}% | {row['paired_reduction_median_pct']:.8f}% | "
            f"{row['paired_reduction_P90_pct']:.8f}% | {row['fraction_rcryo_20K_lt_4p2K_pct']:.8f}% |"
        )
    report.extend([
        "\n## Economic retention（YT2）\n",
        "| epsilon | 4.2 K | 20 K | increase | rescue | loss |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    for row in economic.to_dict("records"):
        report.append(
            f"| {row['epsilon_pct']:g}% | {row['fraction_pass_4p2K_pct']:.8f}% | "
            f"{row['fraction_pass_20K_pct']:.8f}% | {row['increase_20K_minus_4p2K_percentage_points']:.8f} pp | "
            f"{row['rescue_fraction_of_YT2_pct']:.8f}% | {row['loss_fraction_of_YT2_pct']:.8f}% |"
        )
    report.extend([
        "\n配对降幅定义为 `1-rcryo(20 K)/rcryo(4.2 K)`，逐 configuration、逐情景计算后取分位数。",
        "经济惩罚继续使用冻结的 scenario-global LCOE 基准，没有计算 temperature-specific minimum。",
    ])
    (OUT / "PAIRED_TEMPERATURE_EFFECT_REPORT_V7.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("[4/4] PASS", flush=True)
    print(json.dumps({
        "YT0": EXPECTED_PAIRS, "YT1": int(yt1["N_pairs"]), "YT2": int(yt2["N_pairs"]),
        "median_reduction_pct": dict(zip(refrigeration["scenario"], refrigeration["paired_reduction_median_pct"])),
        "retention_10pct_4p2K": row10["fraction_pass_4p2K_pct"],
        "retention_10pct_20K": row10["fraction_pass_20K_pct"],
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
