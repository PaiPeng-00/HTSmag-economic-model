#!/usr/bin/env python3
"""Build V10 anchor-economics realization ledgers from the complete raw scan.

The raw full-grid CSV remains read-only.  Outputs are written to an isolated
candidate directory and are promoted only after row, key, and minimum checks.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "model" / "model_code" / "code"
sys.path.insert(0, str(CODE / "src"))
os.environ.setdefault("FUSION_DEVICE", "arc_16pancake_nuc600_v6_2")

from fusion_tem import device as cfg  # noqa: E402
from fusion_tem.economic.capital_anchor import (  # noqa: E402
    CAPITAL_BOUNDARY_VERSION,
    COOLANT_REPLENISH_FRACTION,
    CORE_VOM_BY_SCENARIO,
    DISCOUNT_RATE_BY_SCENARIO,
    HTS_PRICE_BY_SCENARIO,
    M_COOLANT_FILL,
    M_HTS,
    M_PS,
    NON_TF_AUX_FRACTION,
    PCS_VOM_USD_PER_MWH_E,
    POWER_SUPPLY_PRICE_USD_PER_A,
    PROJECT_YEARS_BY_SCENARIO,
    REFERENCE_MAGNET_ID,
    annual_pcs_fom_usd,
    build_capital_anchor,
    capital_recovery_factor,
)


SCENARIOS = ("S1", "S2", "S3")
OPS = ((4.2, "He"), (10.0, "He"), (20.0, "He"), (20.0, "H2"))
KEY3 = ["Npw", "rho_turn_uOhm_cm2", "R_joint_nOhm"]
VALUE_COLS = ["Aplant", "E_net_year_MWh", "lcoe_anchor_USD_per_MWh", "r_cryo_re_fraction"]
EXPECTED_PER_OP_SCENARIO = 1_476_200
EXPECTED_ROWS = EXPECTED_PER_OP_SCENARIO * len(OPS) * len(SCENARIOS)
CHUNK = 200_000

RAW_REQUIRED = [
    "Top_K", "coolant", "scenario", *KEY3,
    "Aplant", "E_net_year_MWh", "r_cryo_re_fraction",
    "Tape_cost_USD", "HTS_price_2025USD_per_kAm",
    "Power_supply_cost_USD", "Coolant_price_2025USD_per_kg",
    "Coolant_fill_cost_USD", "E_fusion_th_year_MWh",
    "gross_energy_annual_MWh", "net_energy_annual_MWh",
]


def op_name(top: float, coolant: str) -> str:
    return f"T{str(top).replace('.', 'p')}_{coolant}"


def clipped_weights(values: np.ndarray, coordinate: str) -> dict[float, float]:
    values = np.sort(np.asarray(values, dtype=float))
    x = np.log(values) if coordinate == "log" else values
    bounds = np.r_[x[0], 0.5 * (x[:-1] + x[1:]), x[-1]]
    weights = np.diff(bounds) / (x[-1] - x[0])
    if not np.isclose(weights.sum(), 1.0, atol=1e-12, rtol=0):
        raise RuntimeError("STOP_WEIGHT_CLOSURE")
    return dict(zip(values, weights))


def find_references(raw: Path) -> dict[str, dict[str, float]]:
    matches: list[pd.DataFrame] = []
    for chunk in pd.read_csv(raw, usecols=RAW_REQUIRED, chunksize=CHUNK, low_memory=False):
        mask = (
            np.isclose(chunk["Top_K"], 10.0)
            & chunk["coolant"].eq("He")
            & chunk["Npw"].eq(19)
            & np.isclose(chunk["rho_turn_uOhm_cm2"], 10_000.0)
            & np.isclose(chunk["R_joint_nOhm"], 1.0)
        )
        if mask.any():
            matches.append(chunk.loc[mask].copy())
        if matches and set(pd.concat(matches, ignore_index=True)["scenario"]) == set(SCENARIOS):
            break
    rows = pd.concat(matches, ignore_index=True)
    counts = rows.groupby("scenario").size().to_dict()
    if counts != {scenario: 1 for scenario in SCENARIOS}:
        raise RuntimeError(f"STOP_REFERENCE_NOT_UNIQUE: {counts}")
    refs: dict[str, dict[str, float]] = {}
    for _, row in rows.iterrows():
        scenario = str(row["scenario"])
        refs[scenario] = {
            "hts": float(row["Tape_cost_USD"]) / float(row["HTS_price_2025USD_per_kAm"])
            * float(HTS_PRICE_BY_SCENARIO[scenario]),
            "ps": float(row["Power_supply_cost_USD"]) / float(cfg.power_supply_price_perA)
            * POWER_SUPPLY_PRICE_USD_PER_A,
            "fill": float(row["Coolant_fill_cost_USD"])
            / float(row["Coolant_price_2025USD_per_kg"]) * 175.0,
        }
    return refs


def anchor_lcoe(chunk: pd.DataFrame, refs: dict[str, dict[str, float]]) -> pd.Series:
    scenario = chunk["scenario"].astype(str)
    hts_requirement = pd.to_numeric(chunk["Tape_cost_USD"], errors="coerce") / pd.to_numeric(
        chunk["HTS_price_2025USD_per_kAm"], errors="coerce"
    )
    ps_current = pd.to_numeric(chunk["Power_supply_cost_USD"], errors="coerce") / float(
        cfg.power_supply_price_perA
    )
    coolant_mass = pd.to_numeric(chunk["Coolant_fill_cost_USD"], errors="coerce") / pd.to_numeric(
        chunk["Coolant_price_2025USD_per_kg"], errors="coerce"
    )
    hts_direct = hts_requirement * scenario.map(HTS_PRICE_BY_SCENARIO).astype(float)
    ps_direct = ps_current * POWER_SUPPLY_PRICE_USD_PER_A
    coolant_unit = chunk["coolant"].map({"He": 175.0, "H2": 10.0}).astype(float)
    coolant_fill = coolant_mass * coolant_unit
    delta_mag = (
        M_HTS * (hts_direct - scenario.map({s: refs[s]["hts"] for s in SCENARIOS}))
        + M_PS * (ps_direct - scenario.map({s: refs[s]["ps"] for s in SCENARIOS}))
        + M_COOLANT_FILL * (coolant_fill - scenario.map({s: refs[s]["fill"] for s in SCENARIOS}))
    )
    plant = scenario.map({s: build_capital_anchor(s).full_anchor_usd for s in SCENARIOS}) + delta_mag
    rates = scenario.map(DISCOUNT_RATE_BY_SCENARIO).astype(float)
    years = scenario.map(PROJECT_YEARS_BY_SCENARIO).astype(float)
    crf = pd.Series(
        [capital_recovery_factor(float(rate), float(year)) for rate, year in zip(rates, years)],
        index=chunk.index,
    )
    annual_cost = (
        pd.to_numeric(chunk["E_fusion_th_year_MWh"], errors="coerce")
        * scenario.map(CORE_VOM_BY_SCENARIO).astype(float)
        + pd.to_numeric(chunk["gross_energy_annual_MWh"], errors="coerce") * PCS_VOM_USD_PER_MWH_E
        + annual_pcs_fom_usd()
        + coolant_fill * COOLANT_REPLENISH_FRACTION
    )
    net = pd.to_numeric(chunk["net_energy_annual_MWh"], errors="coerce")
    numerator = crf * plant + annual_cost
    result = pd.Series(np.nan, index=chunk.index, dtype=float)
    valid = np.isfinite(net) & (net > 0) & np.isfinite(numerator)
    result.loc[valid] = numerator.loc[valid] / net.loc[valid]
    return result


def materialize(raw: Path, compact: Path, refs: dict[str, dict[str, float]]) -> dict[str, int]:
    compact.mkdir(parents=True, exist_ok=False)
    writers: dict[tuple[str, float, str], pq.ParquetWriter] = {}
    counts = {(s, t, c): 0 for s in SCENARIOS for t, c in OPS}
    total = 0
    try:
        for chunk in pd.read_csv(raw, usecols=RAW_REQUIRED, chunksize=CHUNK, low_memory=False):
            chunk["Top_K"] = chunk["Top_K"].astype(float)
            chunk["scenario"] = chunk["scenario"].astype(str)
            chunk["coolant"] = chunk["coolant"].astype(str)
            chunk["lcoe_anchor_USD_per_MWh"] = anchor_lcoe(chunk, refs)
            total += len(chunk)
            for (scenario, top, coolant), part in chunk.groupby(
                ["scenario", "Top_K", "coolant"], sort=False
            ):
                key = (str(scenario), float(top), str(coolant))
                if key not in counts:
                    raise RuntimeError(f"STOP_UNEXPECTED_PARTITION: {key}")
                selected = part[KEY3 + VALUE_COLS].copy()
                selected["Npw"] = selected["Npw"].astype(np.int16)
                table = pa.Table.from_pandas(selected, preserve_index=False)
                if key not in writers:
                    path = compact / f"{scenario}_{op_name(float(top), str(coolant))}.parquet"
                    writers[key] = pq.ParquetWriter(path, table.schema, compression="zstd")
                writers[key].write_table(table)
                counts[key] += len(selected)
    finally:
        for writer in writers.values():
            writer.close()
    if total != EXPECTED_ROWS or any(count != EXPECTED_PER_OP_SCENARIO for count in counts.values()):
        raise RuntimeError(f"STOP_ROW_CONTRACT: total={total}; counts={counts}")
    return {f"{s}/{t:g}/{c}": count for (s, t, c), count in counts.items()}


def build_ledgers(compact: Path, out: Path) -> tuple[list[Path], dict[str, float]]:
    out.mkdir(parents=True, exist_ok=False)
    staged: dict[tuple[float, str], pd.DataFrame] = {}
    minima = {scenario: math.inf for scenario in SCENARIOS}
    for top, coolant in OPS:
        merged: pd.DataFrame | None = None
        for scenario in SCENARIOS:
            path = compact / f"{scenario}_{op_name(top, coolant)}.parquet"
            part = pd.read_parquet(path)
            if len(part) != EXPECTED_PER_OP_SCENARIO or part.duplicated(KEY3).any():
                raise RuntimeError(f"STOP_PARTITION_KEY_CONTRACT: {path}")
            part = part.sort_values(KEY3).reset_index(drop=True)
            rename = {name: f"{name}_{scenario}" for name in VALUE_COLS}
            part = part.rename(columns=rename)
            if merged is None:
                merged = part
            else:
                if not merged[KEY3].equals(part[KEY3]):
                    raise RuntimeError(f"STOP_SCENARIO_KEY_ORDER: {top}/{coolant}/{scenario}")
                merged = pd.concat([merged, part.drop(columns=KEY3)], axis=1)
        assert merged is not None
        for scenario in SCENARIOS:
            valid = (
                merged[f"E_net_year_MWh_{scenario}"].gt(0)
                & np.isfinite(merged[f"lcoe_anchor_USD_per_MWh_{scenario}"])
                & merged[f"lcoe_anchor_USD_per_MWh_{scenario}"].gt(0)
            )
            available = merged[f"Aplant_{scenario}"].ge(0.80)
            if (valid & available).any():
                minima[scenario] = min(
                    minima[scenario],
                    float(merged.loc[valid & available, f"lcoe_anchor_USD_per_MWh_{scenario}"].min()),
                )
        merged["Top_K"] = float(top)
        merged["coolant"] = coolant
        staged[(top, coolant)] = merged

    paths: list[Path] = []
    for (top, coolant), frame in staged.items():
        for scenario in SCENARIOS:
            valid = (
                frame[f"E_net_year_MWh_{scenario}"].gt(0)
                & np.isfinite(frame[f"lcoe_anchor_USD_per_MWh_{scenario}"])
                & frame[f"lcoe_anchor_USD_per_MWh_{scenario}"].gt(0)
            )
            frame[f"economic_valid_{scenario}"] = valid
            frame[f"availability_pass_{scenario}"] = frame[f"Aplant_{scenario}"].ge(0.80)
            frame[f"delta_{scenario}"] = np.where(
                valid, frame[f"lcoe_anchor_USD_per_MWh_{scenario}"] / minima[scenario] - 1.0, np.nan
            )
        frame["economic_valid_all"] = frame[[f"economic_valid_{s}" for s in SCENARIOS]].all(axis=1)
        frame["availability_pass_all"] = frame[[f"availability_pass_{s}" for s in SCENARIOS]].all(axis=1)
        frame["delta_max"] = frame[[f"delta_{s}" for s in SCENARIOS]].max(axis=1)
        frame.loc[~frame["economic_valid_all"], "delta_max"] = np.inf
        rho = np.sort(frame["rho_turn_uOhm_cm2"].unique())
        rj = np.sort(frame["R_joint_nOhm"].unique())
        rho_log, rj_log = clipped_weights(rho, "log"), clipped_weights(rj, "log")
        rho_lin, rj_lin = clipped_weights(rho, "linear"), clipped_weights(rj, "linear")
        op_weight = 1.0 / 3.0 if top in (4.2, 10.0) else 1.0 / 6.0
        frame["weight_W1_equal_realization"] = 1.0
        frame["weight_W2_temperature_balanced"] = op_weight
        frame["weight_W3_log_measure"] = (
            op_weight * frame["rho_turn_uOhm_cm2"].map(rho_log) * frame["R_joint_nOhm"].map(rj_log)
        )
        frame["weight_W4_linear_measure"] = (
            op_weight * frame["rho_turn_uOhm_cm2"].map(rho_lin) * frame["R_joint_nOhm"].map(rj_lin)
        )
        path = out / f"realizations_{op_name(top, coolant)}.parquet"
        frame.to_parquet(path, index=False, compression="zstd")
        paths.append(path)
    return paths, minima


def summarize(paths: list[Path], minima: dict[str, float]) -> dict:
    total = available = economic10 = 0
    cryo: list[np.ndarray] = []
    for path in paths:
        frame = pd.read_parquet(path)
        total += len(frame)
        mask = frame["availability_pass_all"]
        available += int(mask.sum())
        economic10 += int((mask & frame["economic_valid_all"] & frame["delta_max"].le(0.10)).sum())
        for scenario in SCENARIOS:
            values = frame.loc[mask, f"r_cryo_re_fraction_{scenario}"].to_numpy(float)
            cryo.append(values[np.isfinite(values)])
    values = np.concatenate(cryo)
    quantiles = np.quantile(values, [0, 0.1, 0.5, 0.9, 1.0])
    return {
        "status": "PASS",
        "capital_boundary_version": CAPITAL_BOUNDARY_VERSION,
        "reference_magnet_id": REFERENCE_MAGNET_ID,
        "raw_rows": EXPECTED_ROWS,
        "realizations": total,
        "scenario_minimum_lcoe_USD_per_MWh": minima,
        "common_availability_realizations": available,
        "common_availability_fraction": available / total,
        "within_10pct_and_common_availability": economic10,
        "conditional_10pct_retention": economic10 / available,
        "common_availability_r_cryo_fraction_quantiles": {
            name: float(value) for name, value in zip(("min", "p10", "median", "p90", "max"), quantiles)
        },
        "non_tf_aux_fraction": NON_TF_AUX_FRACTION,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw, output = args.raw.resolve(), args.output.resolve()
    if not raw.is_file():
        raise SystemExit(f"STOP_MISSING_RAW: {raw}")
    if output.exists():
        raise SystemExit(f"STOP_OUTPUT_EXISTS: {output}")
    output.mkdir(parents=True)
    compact = output / "compact"
    ledgers = output / "full_model_dataset"
    refs = find_references(raw)
    counts = materialize(raw, compact, refs)
    paths, minima = build_ledgers(compact, ledgers)
    report = summarize(paths, minima)
    report["partition_rows"] = counts
    report["reference_costs_USD"] = refs
    (output / "V10_LEDGER_AUDIT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    shutil.rmtree(compact)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
