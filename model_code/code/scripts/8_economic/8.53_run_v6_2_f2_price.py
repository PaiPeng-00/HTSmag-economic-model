#!/usr/bin/env python3
"""V6.2-F2: HTS-price sensitivity with direct V6.2 Rj refinement.

The grid stage reads only the audited V6.2 strict-feasible Parquet store.  For
finite joint-resistance boundaries, the same V6.2 direct scanner used by C/E
is called at log-space midpoints; no V6/V6.1 output is accepted.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import importlib.util
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BASE = Path(os.environ["V6_RUN_BASE"]) if os.environ.get("V6_RUN_BASE") else ROOT / "v6_2_arc_actual_inductance"
PQ_ROOT = BASE / "stage_B1_parquet_v1" / "v6_2_b1_strict_feasible_parquet"
PQ_AUDIT = BASE / "stage_B1_parquet_v1" / "v6_2_b1_strict_parquet_audit.json"
AVAIL_AUDIT = BASE / "stage_A" / "v6_2_a_availability_audit.json"
PRE = BASE / "stage_F" / "v6_2_f_preflight.json"
F1 = BASE / "stage_F" / "v6_2_f1_h2_input_audit.json"
STAGE = BASE / "stage_F2"
ARCH = STAGE / "v6_2_f2_price_architecture_grid.parquet"
BRACKETS = STAGE / "v6_2_f2_price_grid_brackets.csv"
EVALS = STAGE / "v6_2_f2_direct_bisection_evaluations.parquet"
TOLERANCE = STAGE / "v6_2_f2_price_tolerance.csv"
COVERAGE = STAGE / "v6_2_f2_price_coverage.csv"
AUDIT = STAGE / "v6_2_f2_audit.json"
RAW_DIR = STAGE / "direct_raw"
RUNNING = STAGE / "v6_2_f2_running.json"
FAST = Path(__file__).with_name("8.45_refine_v6_2_c_fast_r2.py")

DEVICE = "arc_16pancake_nuc600_v6_2"
MATRIX = "97242795c8f75319c8f5ea726a1fedd86999e8a66e53ddeb1255e6fb0c659ae5"
PRICES = (10.0, 50.0, 100.0)
TEMPERATURES = (4.2, 10.0, 20.0)
N_VALUES = tuple(range(1, 201))
R_VALUES = np.geomspace(1.0, 100.0, 121)
RHO = np.geomspace(10.0, 10000.0, 61)
WORKERS = 6
MAX_ITER = 50
LOG_TARGET = math.log(1.001)
M_HTS = 2.8436625
_FAST = None


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def read_audited_parquet() -> pd.DataFrame:
    if not PQ_ROOT.exists():
        raise FileNotFoundError(PQ_ROOT)
    parts = []
    for path in sorted(PQ_ROOT.glob("scenario=S2/Top_K=*/coolant=He/*.parquet")):
        top = float(path.parents[1].name.split("=", 1)[1])
        if top not in TEMPERATURES:
            continue
        frame = pd.read_parquet(path)
        frame["scenario"] = "S2"
        frame["Top_K"] = top
        frame["coolant"] = "He"
        parts.append(frame)
    if not parts:
        raise AssertionError("no S2/He strict V6.2 Parquet partitions")
    data = pd.concat(parts, ignore_index=True)
    if data["TF_system_matrix_sha256"].astype(str).ne(MATRIX).any():
        raise AssertionError("V6.2 matrix provenance mismatch in F2 input")
    required = {"Npw", "rho_turn_uOhm_cm2", "R_joint_nOhm", "Aplant", "CF_gross", "r_cryo_re_fraction",
                "E_net_year_MWh", "HTS_requirement_kA_m", "C_plant_anchor_USD",
                "annual_noncapital_cost_USD", "lcoe_anchor_USD_per_MWh", "anchor_economic_valid"}
    missing = required.difference(data.columns)
    if missing:
        raise AssertionError(f"F2 Parquet columns missing: {sorted(missing)}")
    return data


def price_lcoe(data: pd.DataFrame, price: float, q_ref: float, crf: float) -> pd.Series:
    # V6.3 corrected price contract: the anchor is already the full plant
    # capital at the reference HTS price; reprice the complete HTS requirement.
    plant = data["C_plant_anchor_USD"].astype(float) + M_HTS * (price - 50.0) * data["HTS_requirement_kA_m"].astype(float)
    return (crf * plant + data["annual_noncapital_cost_USD"].astype(float)) / data["E_net_year_MWh"].astype(float)


def trace_first(r: np.ndarray, passed: np.ndarray) -> dict:
    if not bool(passed[0]):
        return {"lo": math.nan, "hi": math.nan, "low": True, "censored": False}
    failed = np.flatnonzero(~passed)
    if failed.size == 0:
        return {"lo": float(r[-1]), "hi": math.nan, "low": False, "censored": True}
    i = int(failed[0])
    return {"lo": float(r[i - 1]), "hi": float(r[i]), "low": False, "censored": False}


def direct_worker(job: dict) -> list[dict]:
    global _FAST
    if _FAST is None:
        _FAST = load_module(f"v62_f2_fast_{__name__}", FAST)
    job = dict(job)
    job["raw_root"] = str(RAW_DIR)
    direct = _FAST.worker(job)
    out = []
    q_ref = float(job["q_ref"])
    crf = float(job["crf"])
    for rec in direct:
        raw = ROOT / rec["raw_path"]
        raw_df = pd.read_csv(raw, low_memory=False)
        anchored = _FAST.direct_anchor(raw_df)
        valid = (anchored["status"].astype(str).str.lower().eq("success")
                 & anchored["anchor_economic_valid"].astype(bool)
                 & np.isfinite(anchored["Aplant"]) & (anchored["Aplant"] >= 0.80)
                 & np.isfinite(pd.to_numeric(anchored["r_cryo_re_fraction"], errors="coerce"))
                 & (pd.to_numeric(anchored["r_cryo_re_fraction"], errors="coerce") <= 0.50)
                 & np.isfinite(pd.to_numeric(anchored["E_net_year_MWh"], errors="coerce"))
                 & (pd.to_numeric(anchored["E_net_year_MWh"], errors="coerce") > 0.0))
        for rv, group in anchored.groupby("R_joint_nOhm", sort=True):
            group = group.loc[valid.loc[group.index]].copy()
            for price in PRICES:
                if group.empty:
                    out.append({"HTS_price_USD_per_kAm": price, "Top_K": float(job["Top_K"]), "Npw": int(job["Npw"]), "R_joint_nOhm": float(rv), "feasible": False, "lcoe": math.nan, "best_rho": math.nan})
                    continue
                group["lcoe_price"] = price_lcoe(group, price, q_ref, crf)
                group = group.loc[np.isfinite(group["lcoe_price"]) & (group["lcoe_price"] > 0.0)]
                if group.empty:
                    out.append({"HTS_price_USD_per_kAm": price, "Top_K": float(job["Top_K"]), "Npw": int(job["Npw"]), "R_joint_nOhm": float(rv), "feasible": False, "lcoe": math.nan, "best_rho": math.nan})
                else:
                    row = group.loc[group["lcoe_price"].idxmin()]
                    out.append({"HTS_price_USD_per_kAm": price, "Top_K": float(job["Top_K"]), "Npw": int(job["Npw"]), "R_joint_nOhm": float(rv), "feasible": True, "lcoe": float(row["lcoe_price"]), "best_rho": float(row["rho_turn_uOhm_cm2"])})
    return out


def main() -> None:
    if any(p.exists() for p in (ARCH, BRACKETS, EVALS, TOLERANCE, COVERAGE, AUDIT, RUNNING)):
        raise FileExistsError("refusing existing V6.2-F2 artifacts")
    for p in (PQ_AUDIT, AVAIL_AUDIT, PRE, F1):
        if not p.exists():
            raise FileNotFoundError(p)
    pq_audit = json.loads(PQ_AUDIT.read_text(encoding="utf-8"))
    if pq_audit.get("status") != "PASS" or pq_audit.get("device") != DEVICE or pq_audit.get("matrix_sha256") != MATRIX:
        raise AssertionError("V6.2 B1 Parquet gate failed")
    pre = json.loads(PRE.read_text(encoding="utf-8"))
    if pre.get("status") != "PASS":
        raise AssertionError("V6.2-F preflight failed")
    anchor = load_module("v62_f2_anchor", Path(__file__).with_name("8.5_recompute_v2_anchor_economics.py"))
    crf = float(anchor.capital_recovery_factor(anchor.DISCOUNT_RATE_BY_SCENARIO["S2"], anchor.PROJECT_YEARS_BY_SCENARIO["S2"]))
    data = read_audited_parquet()
    ref = data.loc[(data.Top_K == 10.0) & (data.Npw == 19) & (data.R_joint_nOhm == 1.0) & (data.rho_turn_uOhm_cm2 == 10000.0)]
    if ref.empty:
        raise AssertionError("V6.2 F2 frozen reference row not found")
    q_ref = float(ref.iloc[0]["HTS_requirement_kA_m"])
    data = data.loc[(data.Aplant >= 0.80) & (data.r_cryo_re_fraction <= 0.50) & (data.E_net_year_MWh > 0) & data.anchor_economic_valid.astype(bool)].copy()
    records = []
    for price in PRICES:
        d = data.copy()
        d["lcoe_price"] = price_lcoe(d, price, q_ref, crf)
        d = d.loc[np.isfinite(d.lcoe_price) & (d.lcoe_price > 0)]
        best = d.loc[d.groupby(["Top_K", "Npw", "R_joint_nOhm"], sort=False).lcoe_price.idxmin()].copy()
        best["HTS_price_USD_per_kAm"] = price
        best.rename(columns={"rho_turn_uOhm_cm2": "best_rho_turn_uOhm_cm2"}, inplace=True)
        records.append(best[["HTS_price_USD_per_kAm", "Top_K", "Npw", "R_joint_nOhm", "best_rho_turn_uOhm_cm2", "lcoe_price"]])
    grid = pd.concat(records, ignore_index=True)
    refs_t = grid.groupby(["HTS_price_USD_per_kAm", "Top_K"]).lcoe_price.min().to_dict()
    refs_g = grid.groupby("HTS_price_USD_per_kAm").lcoe_price.min().to_dict()
    rows = []
    for price in PRICES:
        for top in TEMPERATURES:
            for n in N_VALUES:
                f = grid.loc[(grid.HTS_price_USD_per_kAm == price) & (grid.Top_K == top) & (grid.Npw == n)].set_index("R_joint_nOhm").reindex(R_VALUES)
                passed = f.lcoe_price.notna().to_numpy() & (f.lcoe_price.to_numpy(float) <= 1.05 * refs_g[price])
                tr = trace_first(R_VALUES, passed)
                rows.append({"scenario": "S2", "coolant": "He", "HTS_price_USD_per_kAm": price, "Top_K": top, "Npw": n, "L_star_temperature_USD_per_MWh": refs_t[(price, top)], "L_star_global_USD_per_MWh": refs_g[price], **tr})
    brackets = pd.DataFrame(rows)
    STAGE.mkdir(parents=True, exist_ok=True)
    grid.to_parquet(ARCH, index=False, compression="zstd")
    brackets.to_csv(BRACKETS, index=False, float_format="%.17g")
    states = {}
    for r in rows:
        key = (float(r["HTS_price_USD_per_kAm"]), float(r["Top_K"]), int(r["Npw"]))
        states[key] = {"mode": "low" if r["low"] else ("censored" if r["censored"] else "bisect"), "lo": r["lo"], "hi": r["hi"], "it": 0}
    started = datetime.now(timezone.utc)
    RUNNING.write_text(json.dumps({"experiment": "V6.2-F2", "status": "RUNNING", "device": DEVICE, "workers": WORKERS, "started_utc": started.isoformat()}, indent=2), encoding="utf-8")
    evaluations = []
    for iteration in range(1, MAX_ITER + 1):
        groups = {}
        mids = {}
        for key, s in states.items():
            if s["mode"] != "bisect" or 0.5 * math.log(s["hi"] / s["lo"]) <= LOG_TARGET:
                continue
            mid = round(math.sqrt(s["lo"] * s["hi"]), 3)
            if not s["lo"] < mid < s["hi"]:
                raise AssertionError(f"F2 midpoint rounding failure: {key}")
            mids[key] = mid
            groups.setdefault((key[1], key[2]), set()).add(mid)
        if not mids:
            break
        jobs = [{"scenario": "S2", "coolant": "He", "Top_K": top, "Npw": n, "R_joint_nOhm": sorted(rs), "iteration": iteration, "q_ref": q_ref, "crf": crf} for (top, n), rs in sorted(groups.items())]
        found = {}
        with concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS) as pool:
            for batch in pool.map(direct_worker, jobs):
                for rec in batch:
                    found[(rec["HTS_price_USD_per_kAm"], rec["Top_K"], rec["Npw"], rec["R_joint_nOhm"])] = rec
                    evaluations.append({"iteration": iteration, **rec})
        for key, mid in mids.items():
            price, top, n = key
            rec = found[(price, top, n, mid)]
            if rec["feasible"] and rec["lcoe"] <= 1.05 * refs_g[price]:
                states[key]["lo"] = mid
            else:
                states[key]["hi"] = mid
            states[key]["it"] += 1
        print(f"[V6.2-F2] iteration {iteration}: {len(mids)} price-specific midpoints, {len(jobs)} direct batches", flush=True)
    unresolved = [k for k, s in states.items() if s["mode"] == "bisect" and 0.5 * math.log(s["hi"] / s["lo"]) > LOG_TARGET]
    if unresolved:
        raise AssertionError(f"unresolved F2 bisections: {len(unresolved)}")
    pd.DataFrame(evaluations).to_parquet(EVALS, index=False, compression="zstd")
    out = []
    for (price, top, n), s in sorted(states.items()):
        if s["mode"] == "low":
            boundary = err = math.nan
        elif s["mode"] == "censored":
            boundary, err = 100.0, 0.0
        else:
            boundary = math.sqrt(s["lo"] * s["hi"]); err = math.exp(0.5 * math.log(s["hi"] / s["lo"])) - 1.0
        out.append({"scenario": "S2", "coolant": "He", "HTS_price_USD_per_kAm": price, "Top_K": top, "Npw": n, "Rj_max_global_5pct_nOhm": boundary, "relative_boundary_error": err, "low_reference_infeasible": s["mode"] == "low", "right_censored": s["mode"] == "censored", "iterations": s["it"]})
    tol = pd.DataFrame(out)
    tol.to_csv(TOLERANCE, index=False, float_format="%.17g")
    cov = []
    for (price, top), g in tol.groupby(["HTS_price_USD_per_kAm", "Top_K"]):
        v = np.clip(g.Rj_max_global_5pct_nOhm.to_numpy(float), 1.0, 100.0)
        cov.append({"scenario": "S2", "coolant": "He", "HTS_price_USD_per_kAm": price, "Top_K": top, "coverage_log_Rj_1_to_10_pct": float(np.mean(np.clip(np.log10(v), 0, 1)) * 100), "coverage_log_Rj_1_to_100_pct": float(np.mean(np.clip(np.log10(v), 0, 2)) * 50), "N_low_reference_infeasible": int(g.low_reference_infeasible.sum()), "N_right_censored": int(g.right_censored.sum())})
    pd.DataFrame(cov).to_csv(COVERAGE, index=False, float_format="%.17g")
    maxerr = float(np.nanmax(tol.relative_boundary_error.to_numpy(float))) if np.isfinite(tol.relative_boundary_error).any() else 0.0
    audit = {"experiment_id": "V6.2-F2-S2-He-price-direct-log-bisection", "status": "PASS", "timestamp_utc": datetime.now(timezone.utc).isoformat(), "device": DEVICE, "matrix_sha256": MATRIX, "prices_constant_2025_USD_per_kAm": list(PRICES), "grid": {"Npw": "1..200", "Rj_nOhm": "121-point geomspace [1,100]", "rho_points": 61}, "frozen_reference": {"q_ref_HTS_requirement_kA_m": q_ref, "CRF": crf}, "bisection": {"workers": WORKERS, "criterion": "global-relative LCOE <=5%", "target_relative_error": 0.001, "maximum_boundary_error": maxerr}, "upstream": {"parquet_audit_sha256": sha(PQ_AUDIT), "f1_audit_sha256": sha(F1)}, "outputs": {p.name: {"path": str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p), "sha256": sha(p)} for p in (ARCH, BRACKETS, EVALS, TOLERANCE, COVERAGE)}, "started_utc": started.isoformat()}
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    RUNNING.unlink()
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
