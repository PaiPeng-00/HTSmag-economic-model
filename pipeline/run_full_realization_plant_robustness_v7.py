#!/usr/bin/env python3
"""Offline five-variable plant-robustness reducer for the V7 frozen scan.

No plant, circuit, or economic model is evaluated here.  The program only
streams the four completed Stage-B CSVs twice: first to establish the three
scenario reference minima, then to identify full five-variable realizations
that meet the stated cross-scenario plant and LCOE contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCI_DEFAULT = ROOT / "scientific_results_v7_splice_equivalent_20260906"
SCENARIOS = ("S1", "S2", "S3")
OPERATING_BLOCKS = ((4.2, "He"), (10.0, "He"), (20.0, "He"), (20.0, "H2"))
KEY = ["Npw", "rho_turn_uOhm_cm2", "R_joint_nOhm", "Top_K", "coolant"]
USECOLS = [
    "scenario", *KEY, "Aplant", "E_net_year_MWh", "lcoe_anchor_USD_per_MWh",
]
EXPECTED_X_PER_BLOCK = 200 * 61 * 121
EXPECTED_TOTAL_X = EXPECTED_X_PER_BLOCK * len(OPERATING_BLOCKS)
EXPECTED_SCENARIO_ROWS = EXPECTED_TOTAL_X * len(SCENARIOS)
CHUNK_ROWS = 250_000
TOL = 1.0e-12


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def valid_lcoe_domain(frame: pd.DataFrame) -> pd.Series:
    """Necessary numerical domain for the two requested plant-level criteria.

    ``E_net > 0`` is retained only because an LCOE ratio is otherwise not
    physically meaningful.  No legacy AF_ref, cryogenic-burden, status, or
    reoptimization condition is used.
    """
    return (
        frame["Aplant"].ge(0.80)
        & frame["E_net_year_MWh"].gt(0.0)
        & np.isfinite(frame["lcoe_anchor_USD_per_MWh"])
        & frame["lcoe_anchor_USD_per_MWh"].gt(0.0)
    )


def source_path(sci: Path) -> Path:
    """The frozen full B1-anchor table, not the unanchored Stage-B CSVs."""
    path = sci / "stage_B1" / "v6_2_b1_anchor_economics.csv"
    if not path.is_file():
        raise RuntimeError(f"STOP_MISSING_FULL_ANCHORED_SOURCE: {path}")
    return path


def first_pass(path: Path) -> tuple[dict[str, float], dict]:
    minima = {scenario: np.inf for scenario in SCENARIOS}
    audit = {
        "rows_by_scenario": {s: 0 for s in SCENARIOS},
        "numeric_domain_rows_by_scenario": {s: 0 for s in SCENARIOS},
        "Aplant_max_by_scenario": {s: -np.inf for s in SCENARIOS},
    }
    rows = 0
    for frame in pd.read_csv(path, usecols=USECOLS, chunksize=CHUNK_ROWS):
        rows += len(frame)
        for scenario in SCENARIOS:
            part = frame.loc[frame["scenario"].eq(scenario)]
            audit["rows_by_scenario"][scenario] += len(part)
            if not part.empty:
                audit["Aplant_max_by_scenario"][scenario] = max(
                    audit["Aplant_max_by_scenario"][scenario], float(part["Aplant"].max())
                )
                domain = part.loc[valid_lcoe_domain(part)]
                audit["numeric_domain_rows_by_scenario"][scenario] += len(domain)
                if not domain.empty:
                    minima[scenario] = min(minima[scenario], float(domain["lcoe_anchor_USD_per_MWh"].min()))
    if rows != EXPECTED_SCENARIO_ROWS:
        raise RuntimeError(f"STOP_FULL_ANCHORED_ROW_CONTRACT: {rows}")
    if any(not np.isfinite(value) for value in minima.values()):
        raise RuntimeError(f"STOP_EMPTY_REFERENCE_DOMAIN: {minima}")
    if any(rows != EXPECTED_TOTAL_X for rows in audit["rows_by_scenario"].values()):
        raise RuntimeError("STOP_SCENARIO_ROW_CONTRACT")
    return minima, audit


def robust_parts(path: Path, minima: dict[str, float], out: Path) -> tuple[int, dict]:
    threshold = {scenario: 1.05 * minima[scenario] for scenario in SCENARIOS}
    total_robust = 0
    selected = {scenario: [] for scenario in SCENARIOS}
    for frame in pd.read_csv(path, usecols=USECOLS, chunksize=CHUNK_ROWS):
        domain = valid_lcoe_domain(frame)
        for scenario in SCENARIOS:
            keep = domain & frame["scenario"].eq(scenario) & frame["lcoe_anchor_USD_per_MWh"].le(threshold[scenario])
            if keep.any():
                selected[scenario].append(frame.loc[keep, KEY].copy())
    scenario_frames: dict[str, pd.DataFrame] = {}
    for scenario in SCENARIOS:
        frame = pd.concat(selected[scenario], ignore_index=True) if selected[scenario] else pd.DataFrame(columns=KEY)
        if frame.duplicated(KEY).any():
            raise RuntimeError(f"STOP_DUPLICATE_REALIZATION_KEY: {scenario}")
        scenario_frames[scenario] = frame
    robust = scenario_frames["S1"].merge(scenario_frames["S2"], on=KEY, how="inner", validate="one_to_one").merge(
        scenario_frames["S3"], on=KEY, how="inner", validate="one_to_one"
    )
    robust.to_parquet(out / "robust_full_realization_ledger.parquet", index=False, compression="zstd")
    total_robust = len(robust)
    return total_robust, {"scenario_pass_counts": {s: int(len(scenario_frames[s])) for s in SCENARIOS}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sci-root", type=Path, default=SCI_DEFAULT)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    sci = args.sci_root.resolve()
    out = (args.output or sci / "experiments" / "full_realization_plant_robustness_v7").resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"STOP_NONEMPTY_OUTPUT_DIRECTORY: {out}")
    out.mkdir(parents=True, exist_ok=True)

    print("[offline reducer] first pass: reference LCOE minima", flush=True)
    path = source_path(sci)
    minima, first_audit = first_pass(path)
    print("[offline reducer] second pass: robust realization intersection", flush=True)
    robust_count, per_block = robust_parts(path, minima, out)
    result = {
        "status": "PASS",
        "experiment": "V7 full-five-variable realization plant robustness",
        "definition": {
            "realization_i": "(Npw, rho_turn, Rj, Top, coolant), held unchanged across S1-S3",
            "robust_flag": "for every s in S1,S2,S3: Aplant>=0.80; E_net>0; LCOE<=1.05*LCOE_s_star",
            "LCOE_s_star": "minimum finite positive LCOE among rows satisfying Aplant>=0.80 and E_net>0 in scenario s",
            "weighting": "w_i=1: one equal vote per complete sampled five-variable realization",
            "excluded_legacy_conditions": "No AF_ref, r_cryo_re_fraction, status, p/up/u split, or reoptimization condition is applied.",
        },
        "full_realization_count": EXPECTED_TOTAL_X,
        "robust_realization_count": int(robust_count),
        "f_robust": robust_count / EXPECTED_TOTAL_X,
        "f_robust_pct": 100.0 * robust_count / EXPECTED_TOTAL_X,
        "scenario_minimum_lcoe_USD_per_MWh": minima,
        "scenario_lcoe_5pct_limit_USD_per_MWh": {s: 1.05 * minima[s] for s in SCENARIOS},
        "first_pass": first_audit,
        "second_pass": per_block,
        "source_files": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)},
        "scientific_model_rerun": False,
        "formal_results_modified": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out / "FULL_REALIZATION_PLANT_ROBUSTNESS_V7.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out / "FULL_REALIZATION_PLANT_ROBUSTNESS_V7.md").write_text(
        "# V7 完整五变量 realization 的 plant-level robustness\n\n"
        "状态：**PASS**；仅对冻结的全量 B1-anchor CSV 做两遍流式后处理，未重跑任何科学模型。\n\n"
        "- 统计单元：`(Npw, rho_turn, Rj, Top, coolant)`，在 S1--S3 不变。\n"
        "- 权重：`w_i=1`，即每个完整采样 realization 一票；分母为全部 5,904,800 个 realization。\n"
        "- robust 条件：三个情景均满足 `Aplant >= 0.80`、`E_net > 0` 和 `LCOE <= 1.05 LCOE_s*`。\n"
        "- 未使用：`AF_ref`、`r_cryo_re_fraction`、状态标签、p/up/u 拆分或任何 reoptimization。\n\n"
        f"结果：`f_robust = {robust_count:,} / {EXPECTED_TOTAL_X:,} = {100.0 * robust_count / EXPECTED_TOTAL_X:.8f}%`。\n",
        encoding="utf-8",
    )
    print(json.dumps({"f_robust_pct": result["f_robust_pct"], "robust_count": robust_count, "full_count": EXPECTED_TOTAL_X}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
