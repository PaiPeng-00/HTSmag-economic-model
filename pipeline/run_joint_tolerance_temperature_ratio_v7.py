#!/usr/bin/env python3
"""Offline joint-resistance tolerance comparison for 4.2 K He and 20 K He.

For each fixed (Npw, rho_turn), the boundary is the last native Rj grid point
in the continuous passing prefix beginning at 1 nOhm.  A prefix reaching the
100 nOhm scan ceiling is right-censored and is reported as >=100 nOhm.
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
LOW = LEDGERS / "realizations_T4p2_He.parquet"
HIGH = LEDGERS / "realizations_T20p0_He.parquet"
OUT = SCI / "experiments" / "joint_tolerance_temperature_ratio_v7"

SCENARIOS = ("S1", "S2", "S3")
BASELINES = {"S1": 1122.1472612313785, "S2": 321.97916120106885, "S3": 78.82074592882216}
KEY2 = ["Npw", "rho_turn_uOhm_cm2"]
KEY3 = KEY2 + ["R_joint_nOhm"]
NB = 200 * 61
NR = 121
CAP = 100.0
YS = (("YR0", None, "all"), ("YR1", .01, "all"), ("YR5", .05, "all"),
      ("YR10", .10, "all"), ("YR20", .20, "all"),
      ("YR10-S1", .10, "S1"), ("YR10-S2", .10, "S2"), ("YR10-S3", .10, "S3"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def safe(value):
    if isinstance(value, dict):
        return {k: safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def voronoi(values: np.ndarray, coordinate: str) -> dict[float, float]:
    v = np.sort(np.asarray(values, float))
    x = np.log(v) if coordinate == "log" else v
    bounds = np.r_[x[0], (x[:-1] + x[1:]) / 2, x[-1]]
    w = np.diff(bounds) / (x[-1] - x[0])
    if not np.isclose(w.sum(), 1.0, atol=1e-12, rtol=0):
        raise RuntimeError("STOP_WEIGHT_CLOSURE")
    return dict(zip(v, w))


def weighted_quantile(values: np.ndarray, weights: np.ndarray, qs=(.1, .5, .9)) -> np.ndarray:
    good = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values, weights = values[good], weights[good]
    if not len(values):
        return np.full(len(qs), np.nan)
    order = np.argsort(values, kind="mergesort")
    values, weights = values[order], weights[order]
    positions = (np.cumsum(weights) - .5 * weights) / weights.sum()
    return np.interp(qs, positions, values, left=values[0], right=values[-1])


def load(path: Path) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    cols = KEY3[:]
    for s in SCENARIOS:
        cols += [f"Aplant_{s}", f"E_net_year_MWh_{s}", f"lcoe_anchor_USD_per_MWh_{s}"]
    frame = pd.read_parquet(path, columns=cols).sort_values(KEY3).reset_index(drop=True)
    if len(frame) != NB * NR or frame.duplicated(KEY3).any():
        raise RuntimeError(f"STOP_SOURCE_GRID_CONTRACT: {path}")
    rj = np.sort(frame["R_joint_nOhm"].unique())
    bases = frame.loc[frame["R_joint_nOhm"].eq(rj[0]), KEY2].reset_index(drop=True)
    if len(rj) != NR or len(bases) != NB or not np.isclose(rj[[0, -1]], [1, CAP]).all():
        raise RuntimeError(f"STOP_AXIS_CONTRACT: {path}")
    expected = np.tile(rj, NB)
    if not np.allclose(frame["R_joint_nOhm"].to_numpy(float), expected, atol=0, rtol=0):
        raise RuntimeError(f"STOP_RJ_ORDER_CONTRACT: {path}")
    return frame, rj, bases


def components(frame: pd.DataFrame, epsilon: float | None, scope: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = SCENARIOS if scope == "all" else (scope,)
    availability, enet, lcoe = [], [], []
    for s in selected:
        a = np.isfinite(frame[f"Aplant_{s}"]) & frame[f"Aplant_{s}"].ge(.80)
        e = np.isfinite(frame[f"E_net_year_MWh_{s}"]) & frame[f"E_net_year_MWh_{s}"].gt(0)
        lv = np.isfinite(frame[f"lcoe_anchor_USD_per_MWh_{s}"]) & frame[f"lcoe_anchor_USD_per_MWh_{s}"].gt(0)
        if epsilon is not None:
            lv &= frame[f"lcoe_anchor_USD_per_MWh_{s}"].le((1 + epsilon) * BASELINES[s])
        availability.append(a.to_numpy(bool))
        enet.append(e.to_numpy(bool))
        lcoe.append(lv.to_numpy(bool))
    a = np.logical_and.reduce(availability).reshape(NB, NR)
    if epsilon is None:
        return a, np.ones_like(a), np.ones_like(a)
    return a, np.logical_and.reduce(enet).reshape(NB, NR), np.logical_and.reduce(lcoe).reshape(NB, NR)


def boundaries(a: np.ndarray, e: np.ndarray, l: np.ndarray, rj: np.ndarray) -> dict:
    passed = a & e & l
    prefix = np.cumprod(passed.astype(np.int8), axis=1).sum(axis=1).astype(int)
    valid = prefix > 0
    censored = prefix == NR
    rtol = np.full(NB, np.nan)
    rtol[valid] = rj[prefix[valid] - 1]
    next_fail = np.full(NB, np.nan)
    finite = valid & ~censored
    next_fail[finite] = rj[prefix[finite]]
    false_to_true = np.any((~passed[:, :-1]) & passed[:, 1:], axis=1)
    reentry = passed[:, 0] & false_to_true
    binding = np.full(NB, "", dtype=object)
    absent = ~valid
    for mask, indices, label_prefix in ((finite, prefix, ""), (absent, np.zeros(NB, dtype=int), "no_boundary:")):
        rows = np.flatnonzero(mask)
        for row in rows:
            j = int(indices[row])
            causes = []
            if not a[row, j]: causes.append("availability")
            if not e[row, j]: causes.append("Enet")
            if not l[row, j]: causes.append("LCOE")
            binding[row] = label_prefix + "+".join(causes)
    binding[censored] = "right_censored"
    return {"pass": passed, "prefix": prefix, "valid": valid, "censored": censored,
            "rtol": rtol, "next_fail": next_fail, "reentry": reentry,
            "false_to_true": false_to_true, "binding": binding}


def q3(values: np.ndarray, weights: np.ndarray, weighting: str) -> np.ndarray:
    if not len(values):
        return np.full(3, np.nan)
    return np.quantile(values, [.1, .5, .9], method="linear") if weighting == "W1_equal_base" else weighted_quantile(values, weights)


def summarize(name: str, b4: dict, b20: dict, weights: dict[str, np.ndarray], bases: pd.DataFrame) -> tuple[dict, list[dict]]:
    v4, v20 = b4["valid"], b20["valid"]
    both = v4 & v20
    rescue, reverse = ~v4 & v20, v4 & ~v20
    observed_ratio = b20["rtol"][both] / b4["rtol"][both]
    c4, c20 = b4["censored"] & both, b20["censored"] & both
    both_censored = c4 & c20
    lower20 = ~c4 & c20
    upper4 = c4 & ~c20
    exact = ~c4 & ~c20 & both
    conservative_ratio = np.full(NB, np.nan)
    conservative_ratio[exact] = b20["rtol"][exact] / b4["rtol"][exact]
    conservative_ratio[lower20] = CAP / b4["rtol"][lower20]
    conservative_ratio[both_censored] = 0.0
    conservative_ratio[upper4] = 0.0

    sensitivity = []
    for weighting, w in weights.items():
        wb = w[both]
        q4 = q3(b4["rtol"][both], wb, weighting)
        q20 = q3(b20["rtol"][both], wb, weighting)
        qr_obs = q3(observed_ratio, wb, weighting)
        qr_low = q3(conservative_ratio[both], wb, weighting)
        denominator = w[both].sum()
        def paired_pct(mask: np.ndarray) -> float:
            return 100 * w[mask].sum() / denominator if denominator > 0 else np.nan
        row = {"condition": name, "weighting": weighting,
               "paired_weight_mass": denominator,
               **{f"Rtol_4p2K_{q}_nOhm": q4[i] for i, q in enumerate(("P10", "P50", "P90"))},
               **{f"Rtol_20K_{q}_nOhm": q20[i] for i, q in enumerate(("P10", "P50", "P90"))},
               **{f"GR_observed_grid_{q}": qr_obs[i] for i, q in enumerate(("P10", "P50", "P90"))},
               **{f"GR_conservative_lower_{q}": qr_low[i] for i, q in enumerate(("P10", "P50", "P90"))},
               "guaranteed_GR_ge_2_pct": paired_pct(both & (conservative_ratio >= 2)),
               "guaranteed_GR_ge_5_pct": paired_pct(both & (conservative_ratio >= 5)),
               "guaranteed_GR_ge_10_pct": paired_pct(both & (conservative_ratio >= 10)),
               "Rtol_4p2K_right_censored_pct_of_paired": paired_pct(c4),
               "Rtol_20K_right_censored_pct_of_paired": paired_pct(c20)}
        sensitivity.append(row)
    primary = next(row for row in sensitivity if row["weighting"] == "W1_equal_base")
    summary = {
        "condition": name,
        "XR1_valid_4p2K": int(v4.sum()), "XR2_valid_20K": int(v20.sum()), "XR3_both_valid": int(both.sum()),
        "XR9_rescue_N": int(rescue.sum()), "XR9_rescue_pct_all_bases": 100 * rescue.mean(),
        "XR10_reverse_N": int(reverse.sum()), "XR10_reverse_pct_all_bases": 100 * reverse.mean(),
        "XR11_4p2K_censored_N": int((b4["censored"] & v4).sum()),
        "XR11_4p2K_censored_pct_valid": 100 * b4["censored"].sum() / v4.sum() if v4.any() else np.nan,
        "XR11_20K_censored_N": int((b20["censored"] & v20).sum()),
        "XR11_20K_censored_pct_valid": 100 * b20["censored"].sum() / v20.sum() if v20.any() else np.nan,
        "XR12_reentry_4p2K_N": int(b4["reentry"].sum()), "XR12_reentry_4p2K_pct_all": 100 * b4["reentry"].mean(),
        "XR12_reentry_20K_N": int(b20["reentry"].sum()), "XR12_reentry_20K_pct_all": 100 * b20["reentry"].mean(),
        "XR12_reentry_either_temperature_N": int((b4["reentry"] | b20["reentry"]).sum()),
        "XR12_reentry_either_temperature_pct_all": 100 * (b4["reentry"] | b20["reentry"]).mean(),
        "ratio_exact_N": int(exact.sum()), "ratio_lower_bound_20K_censored_N": int(lower20.sum()),
        "ratio_both_censored_N": int(both_censored.sum()), "ratio_upper_bound_4p2K_censored_N": int(upper4.sum()),
        **{k: v for k, v in primary.items() if k not in ("condition", "weighting", "paired_weight_mass")},
    }
    return summary, sensitivity


def trigger_rows(name: str, tag: str, boundary: dict) -> list[dict]:
    finite = boundary["valid"] & ~boundary["censored"]
    denominator = int(finite.sum())
    rows = []
    for cause in ("availability", "Enet", "LCOE"):
        hit = np.array([cause in label.split("+") for label in boundary["binding"]], bool) & finite
        rows.append({"condition": name, "temperature": tag, "cause": cause,
                     "finite_boundary_denominator": denominator, "first_failure_incidence_N": int(hit.sum()),
                     "first_failure_incidence_pct": 100 * hit.sum() / denominator if denominator else np.nan,
                     "note": "incidences can sum above 100% when causes trigger simultaneously"})
    return rows


def main() -> int:
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"STOP_NONEMPTY_OUTPUT: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    parent = json.loads(PARENT_AUDIT.read_text(encoding="utf-8"))
    if parent.get("status") != "PASS" or parent.get("validation", {}).get("merged_realization_rows") != 5_904_800:
        raise RuntimeError("STOP_PARENT_AUDIT")
    for s in SCENARIOS:
        if not np.isclose(parent["baselines"][s], BASELINES[s], atol=1e-12, rtol=0):
            raise RuntimeError(f"STOP_BASELINE_DRIFT: {s}")
    print("[1/4] load complete 4.2 K He and 20 K He grids", flush=True)
    low, rj, bases = load(LOW)
    high, rj20, bases20 = load(HIGH)
    if not np.array_equal(rj, rj20) or not bases.equals(bases20):
        raise RuntimeError("STOP_TEMPERATURE_PAIR_AXIS_MISMATCH")
    rho = np.sort(bases["rho_turn_uOhm_cm2"].unique())
    wlog, wlin = voronoi(rho, "log"), voronoi(rho, "linear")
    weights = {"W1_equal_base": np.ones(NB),
               "W2_log_rho": bases["rho_turn_uOhm_cm2"].map(wlog).to_numpy(float),
               "W3_linear_rho": bases["rho_turn_uOhm_cm2"].map(wlin).to_numpy(float)}
    print("[2/4] trace continuous-prefix boundaries", flush=True)
    summaries, sensitivity, triggers = [], [], []
    ledger = bases.copy()
    for name, epsilon, scope in YS:
        a4, e4, l4 = components(low, epsilon, scope)
        a20, e20, l20 = components(high, epsilon, scope)
        b4, b20 = boundaries(a4, e4, l4, rj), boundaries(a20, e20, l20, rj)
        summary, rows = summarize(name, b4, b20, weights, bases)
        summaries.append(summary); sensitivity.extend(rows)
        triggers.extend(trigger_rows(name, "4.2K_He", b4)); triggers.extend(trigger_rows(name, "20K_He", b20))
        for tag, b in (("4p2K", b4), ("20K", b20)):
            ledger[f"{name}_{tag}_Rtol_grid_lower_nOhm"] = b["rtol"]
            ledger[f"{name}_{tag}_next_fail_nOhm"] = b["next_fail"]
            ledger[f"{name}_{tag}_right_censored"] = b["censored"]
            ledger[f"{name}_{tag}_reentry_after_failure"] = b["reentry"]
            ledger[f"{name}_{tag}_first_failure_cause"] = b["binding"]
        both = b4["valid"] & b20["valid"]
        ledger[f"{name}_GR_observed_grid"] = np.where(both, b20["rtol"] / b4["rtol"], np.nan)
    summary_df, sensitivity_df, trigger_df = pd.DataFrame(summaries), pd.DataFrame(sensitivity), pd.DataFrame(triggers)
    if len(summary_df) != 8 or len(sensitivity_df) != 24 or len(trigger_df) != 48:
        raise RuntimeError("STOP_OUTPUT_SHAPE")
    print("[3/4] write matrices and base-state ledger", flush=True)
    summary_df.to_csv(OUT / "joint_tolerance_condition_summary.csv", index=False, float_format="%.17g")
    sensitivity_df.to_csv(OUT / "joint_tolerance_weighting_sensitivity.csv", index=False, float_format="%.17g")
    trigger_df.to_csv(OUT / "joint_tolerance_first_failure_causes.csv", index=False, float_format="%.17g")
    ledger_path = OUT / "joint_tolerance_base_state_ledger.parquet"
    ledger.to_parquet(ledger_path, index=False, compression="zstd")
    yr10 = summary_df.loc[summary_df["condition"].eq("YR10")].iloc[0].to_dict()
    audit = {"status": "PASS", "experiment": "V7 paired joint-resistance tolerance versus temperature",
             "definition": {"base_state": "(Npw,rho_turn), 12,200 equal-weight states",
                            "comparison": "4.2 K He versus 20 K He; same base state",
                            "boundary": "last native Rj point in continuous passing prefix from 1 nOhm",
                            "right_censoring": "prefix reaches scan ceiling => Rj_tol>=100 nOhm",
                            "ratio_reporting": "observed-grid ratio plus conservative lower quantiles; both-censored ratios have no finite lower ratio and are set to zero only for conservative threshold guarantees",
                            "first_failure_causes": "multi-cause incidence; percentages need not sum to 100"},
             "frozen_global_LCOE_baselines_USD_per_MWh": BASELINES,
             "YR10_primary": yr10,
             "validation": {"base_states": NB, "Rj_points": NR, "pair_axis_identity": "PASS",
                            "continuous_prefix_used": True, "summary_rows": 8, "weighting_rows": 24,
                            "trigger_rows": 48, "ledger_rows": len(ledger)},
             "source": {"parent_audit_sha256": sha256(PARENT_AUDIT), "ledger_4p2K_sha256": sha256(LOW),
                        "ledger_20K_sha256": sha256(HIGH), "parent_B1_source_sha256": parent["source"]["sha256"]},
             "scientific_model_rerun": False, "formal_results_modified": False,
             "created_utc": datetime.now(timezone.utc).isoformat()}
    (OUT / "JOINT_TOLERANCE_TEMPERATURE_RATIO_V7.json").write_text(
        json.dumps(safe(audit), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# V7 配对接头电阻容限：4.2 K He vs 20 K He\n",
             "状态：**PASS**。固定 `(Npw,rho_turn)`，仅沿原生 121 点 Rj 网格追踪从 1 nOhm 开始的连续通过区间。\n",
             "## YR10 主结果\n",
             f"- 有效边界：4.2 K {int(yr10['XR1_valid_4p2K']):,}；20 K {int(yr10['XR2_valid_20K']):,}；两温度均有效 {int(yr10['XR3_both_valid']):,}。",
             f"- 4.2 K Rj_tol [P10,median,P90] = [{yr10['Rtol_4p2K_P10_nOhm']:.6g}, {yr10['Rtol_4p2K_P50_nOhm']:.6g}, {yr10['Rtol_4p2K_P90_nOhm']:.6g}] nOhm。",
             f"- 20 K Rj_tol 原生网格 [P10,median,P90] = [{yr10['Rtol_20K_P10_nOhm']:.6g}, {yr10['Rtol_20K_P50_nOhm']:.6g}, {yr10['Rtol_20K_P90_nOhm']:.6g}] nOhm；100 表示右截断。",
             f"- observed-grid GR [P10,median,P90] = [{yr10['GR_observed_grid_P10']:.6g}, {yr10['GR_observed_grid_P50']:.6g}, {yr10['GR_observed_grid_P90']:.6g}]。",
             f"- 保守保证 P(GR>=2/5/10) = {yr10['guaranteed_GR_ge_2_pct']:.6f}% / {yr10['guaranteed_GR_ge_5_pct']:.6f}% / {yr10['guaranteed_GR_ge_10_pct']:.6f}% 。",
             f"- 100 nOhm 右截断：4.2 K {yr10['XR11_4p2K_censored_pct_valid']:.6f}%；20 K {yr10['XR11_20K_censored_pct_valid']:.6f}%（各自有效边界为分母）。",
             f"- ratio 类型：exact {int(yr10['ratio_exact_N']):,}；20 K lower-bound {int(yr10['ratio_lower_bound_20K_censored_N']):,}；both censored {int(yr10['ratio_both_censored_N']):,}。",
             "\n由于 20 K 右截断超过 50%，20 K median tolerance 和 median GR 不能作为无截断点估计；应采用 `>=` 或 lower-bound 语言。"]
    (OUT / "JOINT_TOLERANCE_TEMPERATURE_RATIO_REPORT_V7.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("[4/4] PASS", flush=True)
    print(json.dumps({"YR10": safe(yr10)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
