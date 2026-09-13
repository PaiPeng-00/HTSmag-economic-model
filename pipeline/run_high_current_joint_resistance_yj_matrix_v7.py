#!/usr/bin/env python3
"""YJ-XJ offline matrix for Npw=200, He, 4.2 K versus 20 K.

No scientific model is rerun.  Native-grid tolerances use the continuous
passing prefix beginning at Rj=1 nOhm.  A boundary that reaches 100 nOhm is
right-censored and is never represented as an exact finite tolerance.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCI = ROOT / "scientific_results_v7_splice_equivalent_20260906"
PARENT = SCI / "experiments" / "full_realization_robustness_matrix_v7"
PARENT_AUDIT = PARENT / "FULL_REALIZATION_ROBUSTNESS_MATRIX_V7.json"
LEDGERS = PARENT / "realization_ledgers"
SOURCES = {
    "4.2K": LEDGERS / "realizations_T4p2_He.parquet",
    "20K": LEDGERS / "realizations_T20p0_He.parquet",
}
OUT = SCI / "experiments" / "high_current_joint_resistance_yj_matrix_v7"

SCENARIOS = ("S1", "S2", "S3")
BASELINES = {"S1": 1122.1472612313785, "S2": 321.97916120106885, "S3": 78.82074592882216}
CONDITIONS = (("YJ1", .01), ("YJ5", .05), ("YJ10", .10), ("YJ20", .20))
NPW = 200
RHO_FIXED = 10_000.0
NRHO = 61
NRJ = 121
CAP = 100.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def load_high_current(path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    cols = ["Npw", "rho_turn_uOhm_cm2", "R_joint_nOhm"]
    for scenario in SCENARIOS:
        cols.extend((f"Aplant_{scenario}", f"E_net_year_MWh_{scenario}",
                     f"lcoe_anchor_USD_per_MWh_{scenario}"))
    frame = pd.read_parquet(path, columns=cols)
    frame = frame.loc[frame["Npw"].eq(NPW)].sort_values(
        ["rho_turn_uOhm_cm2", "R_joint_nOhm"]).reset_index(drop=True)
    keys = ["rho_turn_uOhm_cm2", "R_joint_nOhm"]
    if len(frame) != NRHO * NRJ or frame.duplicated(keys).any():
        raise RuntimeError(f"STOP_HIGH_CURRENT_GRID_CONTRACT: {path}")
    rho = np.sort(frame["rho_turn_uOhm_cm2"].unique())
    rj = np.sort(frame["R_joint_nOhm"].unique())
    if len(rho) != NRHO or len(rj) != NRJ or not np.isclose(rj[[0, -1]], [1, CAP]).all():
        raise RuntimeError(f"STOP_AXIS_CONTRACT: {path}")
    if not np.any(np.isclose(rho, RHO_FIXED, atol=0, rtol=0)):
        raise RuntimeError("STOP_FIXED_RHO_NOT_ON_NATIVE_GRID")
    if not np.allclose(frame["R_joint_nOhm"].to_numpy(float), np.tile(rj, NRHO), atol=0, rtol=0):
        raise RuntimeError(f"STOP_RJ_ORDER_CONTRACT: {path}")
    return frame, rho, rj


def trace(frame: pd.DataFrame, epsilon: float, rho: np.ndarray, rj: np.ndarray) -> dict:
    component = {}
    scenario_pass = []
    for scenario in SCENARIOS:
        availability = (np.isfinite(frame[f"Aplant_{scenario}"])
                        & frame[f"Aplant_{scenario}"].ge(.80)).to_numpy(bool).reshape(NRHO, NRJ)
        enet = (np.isfinite(frame[f"E_net_year_MWh_{scenario}"])
                & frame[f"E_net_year_MWh_{scenario}"].gt(0)).to_numpy(bool).reshape(NRHO, NRJ)
        lcoe_values = frame[f"lcoe_anchor_USD_per_MWh_{scenario}"]
        lcoe = (np.isfinite(lcoe_values) & lcoe_values.gt(0)
                & lcoe_values.le((1 + epsilon) * BASELINES[scenario])).to_numpy(bool).reshape(NRHO, NRJ)
        component[scenario] = {"availability": availability, "Enet": enet, "LCOE": lcoe}
        scenario_pass.append(availability & enet & lcoe)
    passed = np.logical_and.reduce(scenario_pass)
    prefix = np.cumprod(passed.astype(np.int8), axis=1).sum(axis=1).astype(int)
    valid = prefix > 0
    censored = prefix == NRJ
    rtol = np.full(NRHO, np.nan)
    rtol[valid] = rj[prefix[valid] - 1]
    reentry = passed[:, 0] & np.any((~passed[:, :-1]) & passed[:, 1:], axis=1)
    causes, controllers = np.full(NRHO, "", object), np.full(NRHO, "", object)
    for row in range(NRHO):
        if censored[row]:
            causes[row] = controllers[row] = "right_censored_unobserved"
            continue
        j = int(prefix[row]) if valid[row] else 0
        failed_causes, failed_scenarios = set(), set()
        for scenario in SCENARIOS:
            for cause in ("availability", "Enet", "LCOE"):
                if not component[scenario][cause][row, j]:
                    failed_causes.add(cause)
                    failed_scenarios.add(scenario)
        causes[row] = "+".join(c for c in ("availability", "Enet", "LCOE") if c in failed_causes)
        controllers[row] = "+".join(s for s in SCENARIOS if s in failed_scenarios)
    return {"passed": passed, "prefix": prefix, "valid": valid, "censored": censored,
            "rtol": rtol, "reentry": reentry, "causes": causes, "controllers": controllers,
            "rho": rho}


def arm(trace_result: dict, variant: str) -> dict:
    if variant == "A":
        indices = np.flatnonzero(np.isclose(trace_result["rho"], RHO_FIXED, atol=0, rtol=0))
        if len(indices) != 1:
            raise RuntimeError("STOP_FIXED_RHO_CARDINALITY")
        selected = indices
    elif variant == "B":
        valid_indices = np.flatnonzero(trace_result["valid"])
        if not len(valid_indices):
            selected = np.array([], dtype=int)
        else:
            maximum = np.nanmax(trace_result["rtol"][valid_indices])
            selected = valid_indices[np.isclose(trace_result["rtol"][valid_indices], maximum, atol=0, rtol=0)]
    else:
        raise ValueError(variant)
    if not len(selected):
        return {"valid": False, "rtol": np.nan, "censored": False, "rho_values": [],
                "cause": "", "controller": "", "reentry_selected": False}
    rtol_values = trace_result["rtol"][selected]
    if not np.allclose(rtol_values, rtol_values[0], atol=0, rtol=0, equal_nan=True):
        raise RuntimeError("STOP_SELECTED_TOLERANCE_TIE_CONTRACT")
    causes = sorted(set(trace_result["causes"][selected]) - {""})
    controllers = sorted(set(trace_result["controllers"][selected]) - {""})
    return {"valid": bool(trace_result["valid"][selected].all()),
            "rtol": float(rtol_values[0]),
            "censored": bool(trace_result["censored"][selected].all()),
            "rho_values": [float(v) for v in trace_result["rho"][selected]],
            "cause": " | ".join(causes), "controller": " | ".join(controllers),
            "reentry_selected": bool(trace_result["reentry"][selected].any())}


def ratio(low: dict, high: dict) -> tuple[float, str, str]:
    if not low["valid"] or not high["valid"]:
        return np.nan, "not_applicable", "N/A"
    observed = high["rtol"] / low["rtol"]
    if low["censored"] and high["censored"]:
        return np.nan, "interval_censored_both", "indeterminate (both >=100 nOhm)"
    if high["censored"]:
        return observed, "lower_bound", f">={observed:.9g}"
    if low["censored"]:
        return observed, "upper_bound", f"<={observed:.9g}"
    return observed, "exact", f"{observed:.9g}"


def main() -> int:
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"STOP_NONEMPTY_OUTPUT: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    parent = json.loads(PARENT_AUDIT.read_text(encoding="utf-8"))
    if parent.get("status") != "PASS" or parent.get("validation", {}).get("merged_realization_rows") != 5_904_800:
        raise RuntimeError("STOP_PARENT_AUDIT")
    for scenario in SCENARIOS:
        if not np.isclose(parent["baselines"][scenario], BASELINES[scenario], atol=1e-12, rtol=0):
            raise RuntimeError(f"STOP_BASELINE_DRIFT: {scenario}")

    frames, rho_ref, rj_ref = {}, None, None
    for temperature, path in SOURCES.items():
        frame, rho, rj = load_high_current(path)
        frames[temperature] = frame
        if rho_ref is None:
            rho_ref, rj_ref = rho, rj
        elif not np.array_equal(rho_ref, rho) or not np.array_equal(rj_ref, rj):
            raise RuntimeError("STOP_TEMPERATURE_AXIS_MISMATCH")

    rows, rho_rows = [], []
    for condition, epsilon in CONDITIONS:
        traces = {temperature: trace(frame, epsilon, rho_ref, rj_ref)
                  for temperature, frame in frames.items()}
        for temperature, traced in traces.items():
            for i, rho_value in enumerate(rho_ref):
                rho_rows.append({"condition": condition, "epsilon": epsilon,
                                 "temperature": temperature, "Npw": NPW,
                                 "rho_turn_uOhm_cm2": rho_value,
                                 "Rtol_grid_nOhm": traced["rtol"][i],
                                 "right_censored": traced["censored"][i],
                                 "valid_boundary": traced["valid"][i],
                                 "first_failure_cause": traced["causes"][i],
                                 "controlling_scenario": traced["controllers"][i],
                                 "pass_fail_pass_reentry": traced["reentry"][i]})
        for variant in ("A", "B"):
            low, high = arm(traces["4.2K"], variant), arm(traces["20K"], variant)
            ratio_value, ratio_type, ratio_display = ratio(low, high)
            rows.append({
                "condition": condition, "epsilon": epsilon, "variant": variant,
                "definition": "fixed rho_turn=10000" if variant == "A" else "rho_turn selectable independently by temperature",
                "Npw": NPW, "coolant": "He",
                "XJ1_Rtol_4p2K_grid_nOhm": low["rtol"],
                "XJ2_Rtol_20K_grid_nOhm": high["rtol"],
                "XJ3_ratio_value_or_bound": ratio_value,
                "XJ3_ratio_type": ratio_type, "XJ3_ratio_display": ratio_display,
                "XJ4_4p2K_right_censored": low["censored"],
                "XJ5_20K_right_censored": high["censored"],
                "XJ6_4p2K_first_failure": low["cause"],
                "XJ6_20K_first_failure": high["cause"],
                "XJ7_4p2K_controlling_scenario": low["controller"],
                "XJ7_20K_controlling_scenario": high["controller"],
                "XJ8_4p2K_pass_fail_pass": low["reentry_selected"],
                "XJ8_20K_pass_fail_pass": high["reentry_selected"],
                "rho_turn_4p2K_selected_values": ";".join(f"{x:.17g}" for x in low["rho_values"]),
                "rho_turn_20K_selected_values": ";".join(f"{x:.17g}" for x in high["rho_values"]),
                "rho_turn_4p2K_tie_N": len(low["rho_values"]),
                "rho_turn_20K_tie_N": len(high["rho_values"]),
                "valid_4p2K": low["valid"], "valid_20K": high["valid"],
                "any_reentry_across_61rho_4p2K": bool(traces["4.2K"]["reentry"].any()),
                "any_reentry_across_61rho_20K": bool(traces["20K"]["reentry"].any()),
            })

    summary = pd.DataFrame(rows)
    rho_ledger = pd.DataFrame(rho_rows)
    if len(summary) != 8 or len(rho_ledger) != 4 * 2 * NRHO:
        raise RuntimeError("STOP_OUTPUT_SHAPE")
    if summary["XJ8_4p2K_pass_fail_pass"].any() or summary["XJ8_20K_pass_fail_pass"].any():
        raise RuntimeError("STOP_UNEXPECTED_SELECTED_REENTRY")
    summary.to_csv(OUT / "YJ_XJ_summary.csv", index=False, float_format="%.17g")
    rho_ledger.to_csv(OUT / "YJ_rho_boundary_ledger.csv", index=False, float_format="%.17g")

    audit = {
        "status": "PASS", "experiment": "V7 high-current YJ-XJ joint-resistance matrix",
        "definition": {"Npw": NPW, "coolant": "He", "temperatures_K": [4.2, 20],
                       "variant_A_rho_turn_uOhm_cm2": RHO_FIXED,
                       "variant_B": "maximize continuous-prefix Rj tolerance over the same native 61-point rho grid independently at each temperature",
                       "boundary": "last native Rj point in continuous passing prefix from 1 nOhm",
                       "right_censoring": "prefix reaches 100 nOhm => Rj_tol>=100 nOhm",
                       "both_censored_ratio": "not identifiable; never forced to 1"},
        "frozen_global_LCOE_baselines_USD_per_MWh": BASELINES,
        "results": summary.to_dict(orient="records"),
        "validation": {"high_current_rows_per_temperature": NRHO * NRJ,
                       "rho_points": NRHO, "Rj_points": NRJ,
                       "fixed_rho_native_grid_match": True,
                       "temperature_axis_identity": "PASS",
                       "summary_rows": len(summary), "rho_ledger_rows": len(rho_ledger),
                       "selected_pass_fail_pass_count": int(summary["XJ8_4p2K_pass_fail_pass"].sum() + summary["XJ8_20K_pass_fail_pass"].sum())},
        "source": {"parent_audit_sha256": sha256(PARENT_AUDIT),
                   "4p2K_ledger_sha256": sha256(SOURCES["4.2K"]),
                   "20K_ledger_sha256": sha256(SOURCES["20K"]),
                   "parent_B1_source_sha256": parent["source"]["sha256"]},
        "scientific_model_rerun": False, "formal_results_modified": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUT / "HIGH_CURRENT_JOINT_RESISTANCE_YJ_MATRIX_V7.json").write_text(
        json.dumps(safe(audit), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    def show_rtol(value, censored):
        if not np.isfinite(value):
            return "N/A"
        return (">=" if censored else "") + f"{value:.6g} nOhm"

    def show_rho(row, temperature):
        tie_count = int(row[f"rho_turn_{temperature}_tie_N"])
        if tie_count == NRHO:
            return f"all {NRHO} native values ({rho_ref[0]:.6g}--{rho_ref[-1]:.6g})"
        return row[f"rho_turn_{temperature}_selected_values"]

    lines = ["# 高电流接头电阻 YJ-XJ 矩阵", "",
             "状态：**PASS**。固定 Npw=200 和 He；仅比较 4.2 K 与 20 K，使用冻结的三情景全局 LCOE 基准。", "",
             "原生连续前缀边界从 1 nOhm 开始；达到 100 nOhm 一律作为右截断。YJ-B 先逐 rho 求边界，再取最大值。", ""]
    order = [("YJ10", "A"), ("YJ10", "B"), ("YJ1", "A"), ("YJ5", "A"), ("YJ20", "A")]
    for condition, variant in order:
        row = summary.loc[summary["condition"].eq(condition) & summary["variant"].eq(variant)].iloc[0]
        lines.extend([
            f"## {condition}-{variant}", "",
            f"- XJ1 = {show_rtol(row['XJ1_Rtol_4p2K_grid_nOhm'], row['XJ4_4p2K_right_censored'])}",
            f"- XJ2 = {show_rtol(row['XJ2_Rtol_20K_grid_nOhm'], row['XJ5_20K_right_censored'])}",
            f"- XJ3 = {row['XJ3_ratio_display']}-fold" if row["XJ3_ratio_type"] != "not_applicable" else "- XJ3 = N/A",
        ])
        if condition == "YJ10":
            lines.extend([
                f"- XJ4/XJ5 = {bool(row['XJ4_4p2K_right_censored'])} / {bool(row['XJ5_20K_right_censored'])}",
                f"- XJ6 (4.2/20 K) = {row['XJ6_4p2K_first_failure']} / {row['XJ6_20K_first_failure']}",
                f"- XJ7 (4.2/20 K) = {row['XJ7_4p2K_controlling_scenario']} / {row['XJ7_20K_controlling_scenario']}",
                f"- XJ8 (4.2/20 K) = {bool(row['XJ8_4p2K_pass_fail_pass'])} / {bool(row['XJ8_20K_pass_fail_pass'])}",
                f"- selected rho_turn (4.2/20 K) = {show_rho(row, '4p2K')} / {show_rho(row, '20K')} uOhm cm2",
            ])
        lines.append("")
    (OUT / "HIGH_CURRENT_JOINT_RESISTANCE_YJ_REPORT_V7.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "PASS", "YJ10": safe(summary.loc[summary["condition"].eq("YJ10")].to_dict(orient="records"))}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
