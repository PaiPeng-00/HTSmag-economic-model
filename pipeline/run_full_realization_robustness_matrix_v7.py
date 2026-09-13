#!/usr/bin/env python3
"""Build the frozen V7 full-realization robustness matrix offline.

This is a reducer only. It reads the completed B1-anchor table and never calls
the circuit, plant, cryogenic, or economic models.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
SCI = ROOT / "scientific_results_v7_splice_equivalent_20260906"
SOURCE = SCI / "stage_B1" / "v6_2_b1_anchor_economics.csv"
UPSTREAM_AUDIT = SCI / "stage_B1" / "v6_2_b1_and_strict_feasibility_audit.json"
STEP178_AUDIT = SCI / "experiments" / "full_realization_plant_robustness_v7" / "FULL_REALIZATION_PLANT_ROBUSTNESS_V7.json"
OUT = SCI / "experiments" / "full_realization_robustness_matrix_v7"
COMPACT = OUT / "compact_realization_source"
LEDGERS = OUT / "realization_ledgers"

SCENARIOS = ("S1", "S2", "S3")
OPS = ((4.2, "He"), (10.0, "He"), (20.0, "He"), (20.0, "H2"))
BASELINES = {"S1": 1122.1472612313785, "S2": 321.97916120106885, "S3": 78.82074592882216}
EPS = (0.01, 0.02, 0.03, 0.05, 0.10)
KEY3 = ["Npw", "rho_turn_uOhm_cm2", "R_joint_nOhm"]
KEY5 = KEY3 + ["Top_K", "coolant"]
VALUE_COLS = ["Aplant", "E_net_year_MWh", "lcoe_anchor_USD_per_MWh", "r_cryo_re_fraction"]
USECOLS = ["scenario", *KEY5, *VALUE_COLS]
EXPECTED_PER_SCENARIO = 5_904_800
EXPECTED_ROWS = 17_714_400
EXPECTED_PER_OP_SCENARIO = 1_476_200
CHUNK = 250_000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def json_safe(value):
    """Convert pandas/NumPy scalars and non-finite floats to strict JSON."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def op_name(top: float, coolant: str) -> str:
    return f"T{str(top).replace('.', 'p')}_{coolant}"


def clipped_weights(values: np.ndarray, coordinate: str) -> dict[float, float]:
    v = np.sort(np.asarray(values, dtype=float))
    x = np.log(v) if coordinate == "log" else v
    bounds = np.r_[x[0], 0.5 * (x[:-1] + x[1:]), x[-1]]
    w = np.diff(bounds) / (x[-1] - x[0])
    if not np.isclose(w.sum(), 1.0, atol=1e-12, rtol=0):
        raise RuntimeError("STOP_WEIGHT_CLOSURE")
    return dict(zip(v, w))


def source_gate() -> dict:
    for path in (SOURCE, UPSTREAM_AUDIT, STEP178_AUDIT):
        if not path.is_file():
            raise RuntimeError(f"STOP_MISSING_SOURCE: {path}")
    upstream = json.loads(UPSTREAM_AUDIT.read_text(encoding="utf-8"))
    prior = json.loads(STEP178_AUDIT.read_text(encoding="utf-8"))
    expected_hash = upstream["outputs"]["b1_anchor"]["sha256"]
    prior_hash = next(iter(prior["source_files"].values()))
    if upstream.get("status") != "PASS" or prior.get("status") != "PASS" or prior_hash != expected_hash:
        raise RuntimeError("STOP_SOURCE_IDENTITY_NOT_PASS")
    if prior["full_realization_count"] != EXPECTED_PER_SCENARIO:
        raise RuntimeError("STOP_PRIOR_DENOMINATOR_DRIFT")
    for s in SCENARIOS:
        if not np.isclose(prior["scenario_minimum_lcoe_USD_per_MWh"][s], BASELINES[s], atol=1e-12, rtol=0):
            raise RuntimeError(f"STOP_BASELINE_DRIFT: {s}")
    return {"upstream": upstream, "prior": prior, "source_sha256": expected_hash}


def materialize_compact() -> dict:
    COMPACT.mkdir(parents=True, exist_ok=True)
    writers: dict[tuple[str, float, str], pq.ParquetWriter] = {}
    counts = {(s, t, c): 0 for s in SCENARIOS for t, c in OPS}
    total = 0
    try:
        for chunk in pd.read_csv(SOURCE, usecols=USECOLS, chunksize=CHUNK):
            # CSV inference can switch 10/20 K between int64 and float64 across
            # chunks; pin the physical column types before opening writers.
            chunk["Top_K"] = chunk["Top_K"].astype(float)
            chunk["scenario"] = chunk["scenario"].astype(str)
            chunk["coolant"] = chunk["coolant"].astype(str)
            total += len(chunk)
            for (scenario, top, coolant), part in chunk.groupby(["scenario", "Top_K", "coolant"], sort=False):
                key = (str(scenario), float(top), str(coolant))
                if key not in counts:
                    raise RuntimeError(f"STOP_UNEXPECTED_PARTITION: {key}")
                table = pa.Table.from_pandas(part, preserve_index=False)
                if key not in writers:
                    path = COMPACT / f"{scenario}_{op_name(float(top), str(coolant))}.parquet"
                    writers[key] = pq.ParquetWriter(path, table.schema, compression="zstd")
                writers[key].write_table(table)
                counts[key] += len(part)
    finally:
        for writer in writers.values():
            writer.close()
    if total != EXPECTED_ROWS or any(n != EXPECTED_PER_OP_SCENARIO for n in counts.values()):
        raise RuntimeError(f"STOP_COMPACT_ROW_CONTRACT: total={total}; counts={counts}")
    return {f"{s}/{t:g}/{c}": n for (s, t, c), n in counts.items()}


def load_scenario(top: float, coolant: str, scenario: str) -> pd.DataFrame:
    path = COMPACT / f"{scenario}_{op_name(top, coolant)}.parquet"
    frame = pd.read_parquet(path)
    if len(frame) != EXPECTED_PER_OP_SCENARIO or frame.duplicated(KEY3).any():
        raise RuntimeError(f"STOP_PARTITION_KEY_CONTRACT: {path}")
    rename = {col: f"{col}_{scenario}" for col in VALUE_COLS}
    return frame[KEY3 + VALUE_COLS].rename(columns=rename)


def build_ledgers() -> tuple[list[Path], dict]:
    LEDGERS.mkdir(parents=True, exist_ok=True)
    ledger_paths: list[Path] = []
    minima = {s: math.inf for s in SCENARIOS}
    for top, coolant in OPS:
        frames = {s: load_scenario(top, coolant, s) for s in SCENARIOS}
        frame = frames["S1"].merge(frames["S2"], on=KEY3, validate="one_to_one").merge(
            frames["S3"], on=KEY3, validate="one_to_one"
        )
        if len(frame) != EXPECTED_PER_OP_SCENARIO or frame.duplicated(KEY3).any():
            raise RuntimeError(f"STOP_MERGED_LEDGER_KEY_CONTRACT: {top}/{coolant}")
        frame["Top_K"] = top
        frame["coolant"] = coolant
        valid_cols = []
        for s in SCENARIOS:
            valid = (
                frame[f"E_net_year_MWh_{s}"].gt(0)
                & np.isfinite(frame[f"lcoe_anchor_USD_per_MWh_{s}"])
                & frame[f"lcoe_anchor_USD_per_MWh_{s}"].gt(0)
            )
            frame[f"economic_valid_{s}"] = valid
            frame[f"availability_pass_{s}"] = frame[f"Aplant_{s}"].ge(0.80)
            frame[f"delta_{s}"] = np.where(
                valid, frame[f"lcoe_anchor_USD_per_MWh_{s}"] / BASELINES[s] - 1.0, np.nan
            )
            reference_domain = valid & frame[f"availability_pass_{s}"]
            if reference_domain.any():
                minima[s] = min(
                    minima[s],
                    float(frame.loc[reference_domain, f"lcoe_anchor_USD_per_MWh_{s}"].min()),
                )
            valid_cols.append(f"economic_valid_{s}")
        frame["economic_valid_all"] = frame[valid_cols].all(axis=1)
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
        frame["weight_W3_log_measure"] = op_weight * frame["rho_turn_uOhm_cm2"].map(rho_log) * frame["R_joint_nOhm"].map(rj_log)
        frame["weight_W4_linear_measure"] = op_weight * frame["rho_turn_uOhm_cm2"].map(rho_lin) * frame["R_joint_nOhm"].map(rj_lin)
        path = LEDGERS / f"realizations_{op_name(top, coolant)}.parquet"
        frame.sort_values(KEY3).to_parquet(path, index=False, compression="zstd")
        ledger_paths.append(path)
    for s in SCENARIOS:
        if not np.isclose(minima[s], BASELINES[s], atol=1e-12, rtol=0):
            raise RuntimeError(f"STOP_RECONSTRUCTED_BASELINE_DRIFT: {s}={minima[s]}")
    return ledger_paths, minima


def population_mask(frame: pd.DataFrame, population: str, epsilon: float | None) -> pd.Series:
    if population == "Y0":
        return pd.Series(True, index=frame.index)
    if population.startswith("Y1-") and population != "Y1-all":
        return frame[f"availability_pass_{population[-2:]}"].astype(bool)
    if population == "Y1-all":
        return frame["availability_pass_all"].astype(bool)
    if population == "Y2":
        return frame["economic_valid_all"] & frame["delta_max"].le(float(epsilon))
    if population == "Y3":
        return frame["availability_pass_all"] & frame["economic_valid_all"] & frame["delta_max"].le(float(epsilon))
    raise ValueError(population)


def ecdf_quantile_with_invalid(values: np.ndarray, population_count: int, q: float) -> float:
    finite = np.sort(values[np.isfinite(values)])
    rank = int(math.ceil(q * population_count)) - 1
    return float(finite[rank]) if 0 <= rank < len(finite) else math.inf


def summarize_population(paths: list[Path], population: str, epsilon: float | None) -> dict:
    n = 0
    delta_parts, cryo_parts = [], []
    valid_delta_n = 0
    for path in paths:
        frame = pd.read_parquet(path)
        mask = population_mask(frame, population, epsilon)
        n += int(mask.sum())
        delta = frame.loc[mask, "delta_max"].to_numpy(float)
        valid_delta_n += int(np.isfinite(delta).sum())
        delta_parts.append(delta)
        pooled = np.concatenate([frame.loc[mask, f"r_cryo_re_fraction_{s}"].to_numpy(float) for s in SCENARIOS])
        cryo_parts.append(pooled[np.isfinite(pooled)])
    deltas = np.concatenate(delta_parts) if delta_parts else np.empty(0)
    cryo = np.concatenate(cryo_parts) if cryo_parts else np.empty(0)
    qcryo = np.quantile(cryo, [0, 0.1, 0.5, 0.9, 1.0], method="linear") if len(cryo) else np.full(5, np.nan)
    result = {
        "population": population if epsilon is None else f"{population}-{100*epsilon:g}pct",
        "epsilon_pct": None if epsilon is None else 100 * epsilon,
        "N_realizations": n,
        "fraction_all_pct": 100 * n / EXPECTED_PER_SCENARIO,
        "r_cryo_finite_scenario_record_count": int(len(cryo)),
        "r_cryo_min_pct": 100 * qcryo[0], "r_cryo_P10_pct": 100 * qcryo[1],
        "r_cryo_median_pct": 100 * qcryo[2], "r_cryo_P90_pct": 100 * qcryo[3],
        "r_cryo_max_pct": 100 * qcryo[4],
        "delta_max_valid_count": valid_delta_n,
        "delta_max_P50_pct": 100 * ecdf_quantile_with_invalid(deltas, n, 0.50) if n else np.nan,
        "delta_max_P75_pct": 100 * ecdf_quantile_with_invalid(deltas, n, 0.75) if n else np.nan,
        "delta_max_P90_pct": 100 * ecdf_quantile_with_invalid(deltas, n, 0.90) if n else np.nan,
    }
    return result


def build_population_matrix(paths: list[Path]) -> pd.DataFrame:
    specs = [("Y0", None), ("Y1-S1", None), ("Y1-S2", None), ("Y1-S3", None), ("Y1-all", None)]
    specs += [(kind, eps) for kind in ("Y2", "Y3") for eps in EPS]
    rows = [summarize_population(paths, kind, eps) for kind, eps in specs]
    frame = pd.DataFrame(rows)
    n_avail = int(frame.loc[frame["population"].eq("Y1-all"), "N_realizations"].iloc[0])
    frame["f_econ_given_avail_pct"] = np.where(
        frame["population"].str.startswith("Y3-"), 100 * frame["N_realizations"] / n_avail, np.nan
    )
    return frame


def threshold_matrices(paths: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    scenario_rows, weight_rows = [], []
    weight_cols = {
        "W1_equal_realization": "weight_W1_equal_realization",
        "W2_temperature_balanced": "weight_W2_temperature_balanced",
        "W3_log_measure": "weight_W3_log_measure",
        "W4_linear_measure": "weight_W4_linear_measure",
    }
    totals = {w: 0.0 for w in weight_cols}
    avail_weights = {w: 0.0 for w in weight_cols}
    cached = [pd.read_parquet(path) for path in paths]
    for frame in cached:
        for w, col in weight_cols.items():
            totals[w] += float(frame[col].sum())
            avail_weights[w] += float(frame.loc[frame["availability_pass_all"], col].sum())
    for eps in EPS:
        counts = {s: 0 for s in SCENARIOS}
        intersection = 0
        weighted = {w: 0.0 for w in weight_cols}
        for frame in cached:
            for s in SCENARIOS:
                counts[s] += int((frame[f"availability_pass_{s}"] & frame[f"economic_valid_{s}"] & frame[f"delta_{s}"].le(eps)).sum())
            robust = frame["availability_pass_all"] & frame["economic_valid_all"] & frame["delta_max"].le(eps)
            intersection += int(robust.sum())
            for w, col in weight_cols.items():
                weighted[w] += float(frame.loc[robust, col].sum())
        scenario_rows.append({
            "epsilon_pct": 100*eps,
            **{f"{s}_pass_count": counts[s] for s in SCENARIOS},
            **{f"{s}_pass_fraction_pct": 100*counts[s]/EXPECTED_PER_SCENARIO for s in SCENARIOS},
            "intersection_count": intersection,
            "intersection_fraction_pct": 100*intersection/EXPECTED_PER_SCENARIO,
        })
        for w in weight_cols:
            weight_rows.append({
                "epsilon_pct": 100*eps, "weighting": w,
                "unconditional_fraction_pct": 100*weighted[w]/totals[w],
                "conditional_econ_given_avail_pct": 100*weighted[w]/avail_weights[w],
            })
    return pd.DataFrame(scenario_rows), pd.DataFrame(weight_rows)


def rj_boundaries(paths: list[Path]) -> pd.DataFrame:
    rows = []
    frames = [pd.read_parquet(p) for p in paths if "H2" not in p.name]
    for top in (4.2, 10.0, 20.0):
        frame = next(x for x in frames if np.isclose(float(x["Top_K"].iloc[0]), top))
        base = frame["Npw"].eq(200)
        for eps in (0.01, 0.05, 0.10):
            maxima = {}
            for s in SCENARIOS:
                mask = base & frame[f"availability_pass_{s}"] & frame[f"economic_valid_{s}"] & frame[f"delta_{s}"].le(eps)
                maxima[s] = float(frame.loc[mask, "R_joint_nOhm"].max()) if mask.any() else np.nan
            shared = base & frame["availability_pass_all"] & frame["economic_valid_all"] & frame["delta_max"].le(eps)
            rows.append({
                "Top_K": top, "coolant": "He", "Npw": 200, "epsilon_pct": 100*eps,
                **{f"Rj_max_{s}_nOhm": maxima[s] for s in SCENARIOS},
                "Rj_max_min_over_s_nOhm": min(maxima.values()) if all(np.isfinite(list(maxima.values()))) else np.nan,
                "Rj_max_shared_realization_nOhm": float(frame.loc[shared, "R_joint_nOhm"].max()) if shared.any() else np.nan,
                "rho_selection": "exists one sampled rho_turn per scenario; shared column requires one common full realization",
            })
    return pd.DataFrame(rows)


def main() -> int:
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"STOP_NONEMPTY_OUTPUT: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    gate = source_gate()
    print("[1/5] materialize compact frozen source", flush=True)
    compact_counts = materialize_compact()
    print("[2/5] build full-realization ledgers", flush=True)
    paths, minima = build_ledgers()
    print("[3/5] calculate Y0-Y3 population matrix", flush=True)
    populations = build_population_matrix(paths)
    print("[4/5] calculate threshold and weighting matrices", flush=True)
    scenarios, weights = threshold_matrices(paths)
    print("[5/5] calculate Rj boundary diagnostic", flush=True)
    rj = rj_boundaries(paths)

    prior = gate["prior"]
    y35_check = populations.loc[populations["population"].eq("Y3-5pct")].iloc[0]
    s5_check = scenarios.loc[np.isclose(scenarios["epsilon_pct"], 5.0)].iloc[0]
    if len(populations) != 15 or len(scenarios) != 5 or len(weights) != 20 or len(rj) != 9:
        raise RuntimeError("STOP_OUTPUT_MATRIX_SHAPE")
    if int(y35_check["N_realizations"]) != int(prior["robust_realization_count"]):
        raise RuntimeError("STOP_PRIMARY_COUNT_REGRESSION")
    if not np.isclose(float(y35_check["fraction_all_pct"]), float(prior["f_robust_pct"]), atol=1e-12, rtol=0):
        raise RuntimeError("STOP_PRIMARY_FRACTION_REGRESSION")
    for s in SCENARIOS:
        if int(s5_check[f"{s}_pass_count"]) != int(prior["second_pass"]["scenario_pass_counts"][s]):
            raise RuntimeError(f"STOP_SCENARIO_5PCT_REGRESSION: {s}")
    expected_weight_totals = {
        "W1_equal_realization": float(EXPECTED_PER_SCENARIO),
        "W2_temperature_balanced": float(EXPECTED_PER_OP_SCENARIO),
        "W3_log_measure": 200.0,
        "W4_linear_measure": 200.0,
    }
    for weighting, expected_total in expected_weight_totals.items():
        total = 0.0
        column = f"weight_{weighting}"
        for path in paths:
            total += float(pd.read_parquet(path, columns=[column])[column].sum())
        if not np.isclose(total, expected_total, atol=1e-8, rtol=1e-12):
            raise RuntimeError(f"STOP_WEIGHT_DENOMINATOR: {weighting}={total}")

    populations.to_csv(OUT / "Y0_Y3_population_matrix.csv", index=False, float_format="%.17g")
    scenarios.to_csv(OUT / "scenario_threshold_matrix.csv", index=False, float_format="%.17g")
    weights.to_csv(OUT / "weighting_sensitivity_matrix.csv", index=False, float_format="%.17g")
    rj.to_csv(OUT / "Rj_max_temperature_threshold_matrix.csv", index=False, float_format="%.17g")
    y1 = populations.loc[populations["population"].eq("Y1-all")].iloc[0].to_dict()
    y35 = populations.loc[populations["population"].eq("Y3-5pct")].iloc[0].to_dict()
    audit = {
        "status": "PASS", "experiment": "V7 frozen full-realization robustness matrix",
        "definitions": {
            "realization": "(Npw,rho_turn,Rj,Top,coolant), unchanged across S1-S3",
            "primary_measure": "W1, one equal vote per sampled full-factorial realization",
            "r_cryo_distribution": "pooled finite scenario records: three records per retained realization",
            "delta_invalid_handling": "E_net<=0 or nonfinite/nonpositive LCOE fails economic gates and is +infinity in availability-population ECDF quantiles",
            "quantile_method": "r_cryo uses NumPy linear quantiles; epsilon50/75/90 use empirical inverse CDF over the full population including invalid mass at +infinity",
            "legacy_filters_removed": ["AF_ref", "r_cryo_re_fraction<=0.50", "status label", "scenario-specific reoptimization"],
        },
        "baselines": minima, "compact_partition_counts": compact_counts,
        "validation": {
            "matrix_row_counts": {"populations": 15, "scenario_thresholds": 5, "weighting_rows": 20, "Rj_rows": 9},
            "merged_realization_rows": EXPECTED_PER_SCENARIO,
            "step178_primary_5pct_regression": "PASS",
            "step178_scenario_5pct_regression": "PASS",
            "weight_denominator_closure": "PASS",
        },
        "primary": {"Y1_all": y1, "Y3_5pct": y35},
        "source": {"path": str(SOURCE.relative_to(ROOT)).replace("\\", "/"), "sha256": gate["source_sha256"], "identity_inherited_from": str(STEP178_AUDIT.relative_to(ROOT)).replace("\\", "/")},
        "scientific_model_rerun": False, "formal_results_modified": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUT / "FULL_REALIZATION_ROBUSTNESS_MATRIX_V7.json").write_text(
        json.dumps(json_safe(audit), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    y3_rows = populations.loc[populations["population"].str.startswith("Y3-")]
    wwide = weights.pivot(index="epsilon_pct", columns="weighting", values="unconditional_fraction_pct")
    report = [
        "# V7 完整五变量稳健性矩阵\n",
        "状态：**PASS**。只读取冻结的 B1-anchor 全量结果，未重跑科学模型。\n",
        "## 主结果\n",
        f"- Y1-all：{int(y1['N_realizations']):,} / {EXPECTED_PER_SCENARIO:,} = {y1['fraction_all_pct']:.8f}%",
        f"- Y3(5%)：{int(y35['N_realizations']):,} / {EXPECTED_PER_SCENARIO:,} = {y35['fraction_all_pct']:.8f}%",
        f"- 条件保持率 f_econ|avail(5%)：{y35['f_econ_given_avail_pct']:.8f}%",
        f"- Y1-all 的 epsilon50/75/90：{y1['delta_max_P50_pct']:.8f}% / {y1['delta_max_P75_pct']:.8f}% / {y1['delta_max_P90_pct']:.8f}%",
        f"- Y1-all 的 r_cryo [min,P10,median,P90,max]：[{y1['r_cryo_min_pct']:.8f}, {y1['r_cryo_P10_pct']:.8f}, {y1['r_cryo_median_pct']:.8f}, {y1['r_cryo_P90_pct']:.8f}, {y1['r_cryo_max_pct']:.8f}]%",
        f"- Y3(5%) 的 r_cryo [min,P10,median,P90,max]：[{y35['r_cryo_min_pct']:.8f}, {y35['r_cryo_P10_pct']:.8f}, {y35['r_cryo_median_pct']:.8f}, {y35['r_cryo_P90_pct']:.8f}, {y35['r_cryo_max_pct']:.8f}]%\n",
        "## Robustness curve（W1）\n",
        "| epsilon | 全总体保持率 | availability-qualified 条件保持率 |",
        "|---:|---:|---:|",
    ]
    for _, row in y3_rows.iterrows():
        report.append(f"| {row['epsilon_pct']:g}% | {row['fraction_all_pct']:.8f}% | {row['f_econ_given_avail_pct']:.8f}% |")
    report.extend(["\n## W1-W4 无条件保持率\n", "| epsilon | W1 | W2 | W3 | W4 |", "|---:|---:|---:|---:|---:|"])
    for eps, row in wwide.iterrows():
        report.append(
            f"| {eps:g}% | {row['W1_equal_realization']:.8f}% | {row['W2_temperature_balanced']:.8f}% | "
            f"{row['W3_log_measure']:.8f}% | {row['W4_linear_measure']:.8f}% |"
        )
    report.extend([
        "\n## 口径说明\n",
        "- realization 为 `(Npw,rho_turn,Rj,Top,coolant)`，在 S1-S3 中保持不变。",
        "- `E_net>0` 与有限正 LCOE 仅作为经济量有效域；未使用旧的 `r_cryo<=0.50`。",
        "- r_cryo 分位数汇总每个保留 realization 的三个情景记录，因此样本单位是 scenario record。",
        "- Rj 边界诊断固定高电流端 `Npw=200` 与 He；情景列允许从原生 rho_turn 网格中存在一个通过点，共享列要求同一个完整 realization 在三情景均通过。",
    ])
    (OUT / "ROBUSTNESS_MATRIX_REPORT_V7.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "Y1_all_N": int(y1["N_realizations"]), "Y3_5_N": int(y35["N_realizations"]), "f_econ_given_avail_5pct": y35["f_econ_given_avail_pct"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
