#!/usr/bin/env python3
"""V6-F2 direct log-space refinement of price-specific global-5% boundaries."""
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


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "v6_hts_temperature_tolerance_arc16pancake_nuc600"
BRACKETS = OUT / "v6f_f2_s2_he_price_grid_brackets.csv"
BRACKET_AUDIT = OUT / "v6f_f2_price_grid_bracket_audit.json"
ARCH = OUT / "v6f_f2_s2_he_price_architecture_optima_grid.csv"
DRIVER = Path(__file__).with_name("8.13_run_v6_arc16pancake_nuc600_grid.py")
ANCHOR = Path(__file__).with_name("8.5_recompute_v2_anchor_economics.py")
V6E_AUDIT = OUT / "v6e_cross_scenario_audit.json"
RAW_DIR = OUT / "v6f_f2_direct_bisection_raw"
EVALS = OUT / "v6f_f2_price_direct_bisection_evaluations.csv"
TOLERANCE = OUT / "v6f_hts_price_tolerance.csv"
COVERAGE = OUT / "v6f_hts_price_coverage.csv"
AUDIT = OUT / "v6f_f2_price_direct_bisection_audit.json"
REFS = ROOT / "outputs" / "target_price_window" / "tables" / "B0_reference_magnet_rows.csv"
PRICES = (100.0, 50.0, 10.0)
RHO = np.array([10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000,5000,10000], dtype=float)
M_HTS = 2.8436625
MAX_ITERATIONS = 50
LOG_HALF_ERROR_TARGET = math.log(1.001)
WORKERS = 4


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def import_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def worker(job: dict) -> list[dict]:
    top, npw, iteration = float(job["Top_K"]), int(job["Npw"]), int(job["iteration"])
    requested = tuple(float(x) for x in job["R_joint_nOhm"])
    raw = RAW_DIR / f"iter_{iteration:02d}" / f"T{top:g}K_N{npw:03d}.csv"
    raw.parent.mkdir(parents=True, exist_ok=True)
    if raw.exists():
        raise FileExistsError(f"refusing to overwrite {raw}")
    driver = import_path(f"v6f_f2_driver_{os.getpid()}_{iteration}_{npw}", DRIVER)
    scan = driver.load_scan(output_csv=raw, npw=np.array([npw]), rjoint=np.array(requested), rho=RHO, temperatures=[(top, "He")])
    scan.SCENARIOS = ["S2"]
    scan.main()
    anchor = import_path(f"v6f_f2_anchor_{os.getpid()}", ANCHOR)
    manifest = json.loads(anchor.E0_MANIFEST.read_text(encoding="utf-8"))
    transformed = anchor.transform_chunk(pd.read_csv(raw, dtype=str, keep_default_na=False), job["anchor_refs"], manifest)
    transformed["AF_ref"] = pd.to_numeric(transformed["AF"], errors="coerce") / float(job["af_max"])
    rcryo = pd.to_numeric(transformed["r_cryo_re_fraction"], errors="coerce")
    net = pd.to_numeric(transformed["net_energy_annual_MWh"], errors="coerce")
    l50 = pd.to_numeric(transformed["lcoe_anchor_USD_per_MWh"], errors="coerce")
    valid = (transformed["status"].astype(str).str.lower().eq("success") & transformed["anchor_economic_valid"].astype(str).str.lower().isin(("true", "1"))
             & np.isfinite(transformed["AF_ref"]) & np.isfinite(rcryo) & np.isfinite(net) & np.isfinite(l50)
             & (transformed["AF_ref"] >= 0.99) & (rcryo <= 0.50) & (net > 0.0) & (l50 > 0.0))
    q = pd.to_numeric(transformed["HTS_requirement_kA_m"], errors="coerce")
    plant50 = pd.to_numeric(transformed["C_plant_anchor_USD"], errors="coerce")
    crf = pd.to_numeric(transformed["CRF"], errors="coerce")
    noncap = pd.to_numeric(transformed["annual_noncapital_cost_USD"], errors="coerce")
    records: list[dict] = []
    for r_value, group in transformed.groupby("R_joint_nOhm", sort=True):
        for price in PRICES:
            candidates = group.loc[valid.loc[group.index]].copy()
            if candidates.empty:
                records.append({"HTS_price_USD_per_kAm": price, "Top_K": top, "Npw": npw, "R_joint_nOhm": float(r_value), "feasible": False,
                                "lcoe_price_USD_per_MWh": math.nan, "best_rho_turn_uOhm_cm2": math.nan, "raw_path": str(raw.relative_to(ROOT)), "raw_sha256": sha256(raw)})
                continue
            if price == 50.0:
                candidates["lcoe_price_USD_per_MWh"] = l50.loc[candidates.index]
            else:
                plant = plant50.loc[candidates.index] + M_HTS * (price - 50.0) * (q.loc[candidates.index] - float(job["q_ref"]))
                candidates["lcoe_price_USD_per_MWh"] = (crf.loc[candidates.index] * plant + noncap.loc[candidates.index]) / net.loc[candidates.index]
            candidates = candidates.loc[np.isfinite(candidates["lcoe_price_USD_per_MWh"]) & (candidates["lcoe_price_USD_per_MWh"] > 0.0)]
            if candidates.empty:
                records.append({"HTS_price_USD_per_kAm": price, "Top_K": top, "Npw": npw, "R_joint_nOhm": float(r_value), "feasible": False,
                                "lcoe_price_USD_per_MWh": math.nan, "best_rho_turn_uOhm_cm2": math.nan, "raw_path": str(raw.relative_to(ROOT)), "raw_sha256": sha256(raw)})
            else:
                best = candidates.loc[candidates["lcoe_price_USD_per_MWh"].idxmin()]
                records.append({"HTS_price_USD_per_kAm": price, "Top_K": top, "Npw": npw, "R_joint_nOhm": float(r_value), "feasible": True,
                                "lcoe_price_USD_per_MWh": float(best["lcoe_price_USD_per_MWh"]), "best_rho_turn_uOhm_cm2": float(best["rho_turn_uOhm_cm2"]),
                                "raw_path": str(raw.relative_to(ROOT)), "raw_sha256": sha256(raw)})
    return records


def main() -> None:
    required = (BRACKETS, BRACKET_AUDIT, ARCH, V6E_AUDIT, REFS)
    if not all(path.exists() for path in required):
        raise FileNotFoundError("V6-F2 brackets, grid optima, reference rows and V6-E audit are required")
    if any(path.exists() for path in (RAW_DIR, EVALS, TOLERANCE, COVERAGE, AUDIT)):
        raise FileExistsError("refusing to overwrite V6-F2 direct-bisection artifacts")
    bracket_audit = json.loads(BRACKET_AUDIT.read_text(encoding="utf-8"))
    v6e = json.loads(V6E_AUDIT.read_text(encoding="utf-8"))
    if bracket_audit.get("status") != "PASS_GRID_BRACKETS_NOT_FINAL_BOUNDARIES" or v6e.get("status") != "PASS":
        raise AssertionError("V6-F2 upstream gate is not PASS")
    if sha256(BRACKETS) != bracket_audit["outputs"][BRACKETS.name]["sha256"]:
        raise AssertionError("V6-F2 brackets changed after audit")
    qref_rows = pd.read_csv(REFS).query("scenario == 'S2'")
    if len(qref_rows) != 1:
        raise AssertionError("S2 frozen reference row is not unique")
    q_ref = float(qref_rows.iloc[0]["HTS_requirement_kA_m"])
    af_max = float(v6e["fixed_af_max_by_scenario"]["S2"])
    brackets = pd.read_csv(BRACKETS)
    arch = pd.read_csv(ARCH)
    global_refs = {float(k): float(v) for k, v in bracket_audit["global_references_USD_per_MWh"].items()}
    temp_refs = arch.loc[arch["architecture_feasible"]].groupby(["HTS_price_USD_per_kAm", "Top_K"])["lcoe_price_USD_per_MWh"].min().to_dict()
    states: dict[tuple[float, float, int], dict] = {}
    for row in brackets.to_dict("records"):
        key = (float(row["HTS_price_USD_per_kAm"]), float(row["Top_K"]), int(row["Npw"]))
        if bool(row["low_reference_infeasible_global_5pct"]):
            states[key] = {"mode": "low_ref", "lo": math.nan, "hi": math.nan, "iterations": 0}
        elif bool(row["right_censored_global_5pct"]):
            states[key] = {"mode": "right_censored", "lo": math.nan, "hi": math.nan, "iterations": 0}
        else:
            states[key] = {"mode": "bisection", "lo": float(row["Rj_max_global_5pct_grid_lower_nOhm"]), "hi": float(row["Rj_bracket_global_5pct_upper_nOhm"]), "iterations": 0}
    anchor_module = import_path("v6f_f2_parent_anchor", ANCHOR)
    anchor_manifest = json.loads(anchor_module.E0_MANIFEST.read_text(encoding="utf-8"))
    _, anchor_refs = anchor_module.find_reference_rows(anchor_manifest)
    RAW_DIR.mkdir(parents=True)
    evaluations: list[dict] = []
    started = datetime.now(timezone.utc)
    for iteration in range(1, MAX_ITERATIONS + 1):
        candidates: dict[tuple[float, int], set[float]] = {}
        task_mid: dict[tuple[float, float, int], float] = {}
        for key, state in states.items():
            if state["mode"] != "bisection" or 0.5 * math.log(state["hi"] / state["lo"]) <= LOG_HALF_ERROR_TARGET:
                continue
            mid = round(math.sqrt(state["lo"] * state["hi"]), 3)
            if not state["lo"] < mid < state["hi"]:
                raise AssertionError(f"Rj rounding prevents refinement for {key}")
            task_mid[key] = mid
            candidates.setdefault((key[1], key[2]), set()).add(mid)
        if not task_mid:
            break
        jobs = [{"Top_K": top, "Npw": npw, "R_joint_nOhm": sorted(values), "iteration": iteration, "af_max": af_max, "q_ref": q_ref, "anchor_refs": anchor_refs}
                for (top, npw), values in sorted(candidates.items())]
        results: dict[tuple[float, float, int, float], dict] = {}
        with concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS) as pool:
            for result in pool.map(worker, jobs):
                for record in result:
                    results[(record["HTS_price_USD_per_kAm"], record["Top_K"], record["Npw"], record["R_joint_nOhm"])] = record
                    evaluations.append({"iteration": iteration, **record})
        for key, mid in task_mid.items():
            record = results[(key[0], key[1], key[2], mid)]
            state = states[key]
            if bool(record["feasible"]) and float(record["lcoe_price_USD_per_MWh"]) <= global_refs[key[0]] * 1.05:
                state["lo"] = mid
            else:
                state["hi"] = mid
            state["iterations"] += 1
        print(f"[Progress] V6-F2 direct bisection iteration {iteration}: {len(task_mid)} price-specific midpoints across {len(jobs)} model batches", flush=True)
    unresolved = [key for key, state in states.items() if state["mode"] == "bisection" and 0.5 * math.log(state["hi"] / state["lo"]) > LOG_HALF_ERROR_TARGET]
    if unresolved:
        raise AssertionError(f"V6-F2 bisection unresolved: {len(unresolved)}")
    pd.DataFrame(evaluations).to_csv(EVALS, index=False, encoding="utf-8", float_format="%.17g")
    rows = []
    for key, state in sorted(states.items()):
        price, top, npw = key
        if state["mode"] == "low_ref":
            boundary = lower = upper = error = math.nan; low, censored = True, False
        elif state["mode"] == "right_censored":
            boundary, lower, upper, error, low, censored = 100.0, 100.0, math.nan, 0.0, False, True
        else:
            lower, upper = state["lo"], state["hi"]
            boundary = math.sqrt(lower * upper)
            error = math.exp(0.5 * math.log(upper / lower)) - 1.0
            low = censored = False
        rows.append({"scenario": "S2", "coolant": "He", "HTS_price_USD_per_kAm": price, "Top_K": top, "Npw": npw, "R_ref_nOhm": 1.0,
                     "L_star_temperature_USD_per_MWh": float(temp_refs[(price, top)]), "L_star_global_USD_per_MWh": global_refs[price],
                     "Rj_max_global_5pct_nOhm": boundary, "Rj_lower_global_5pct_nOhm": lower, "Rj_upper_global_5pct_nOhm": upper,
                     "relative_boundary_error_global_5pct": error, "low_reference_infeasible_global_5pct": low, "right_censored_global_5pct": censored,
                     "bisection_iterations_global_5pct": state["iterations"], "binding_condition_global_5pct": "global_relative_LCOE_at_reference" if low else ("right_censored_at_100_nOhm" if censored else "global_relative_LCOE")})
    tolerance = pd.DataFrame(rows)
    tolerance.to_csv(TOLERANCE, index=False, encoding="utf-8", float_format="%.17g")
    coverage_rows = []
    for (price, top), group in tolerance.groupby(["HTS_price_USD_per_kAm", "Top_K"], sort=True):
        values = group["Rj_max_global_5pct_nOhm"].to_numpy(float)
        coverage_rows.append({"scenario": "S2", "coolant": "He", "HTS_price_USD_per_kAm": price, "Top_K": top,
                              "coverage_log_Rj_1_to_10_pct": 100.0 * np.mean(np.clip(np.log10(values), 0.0, 1.0)),
                              "coverage_log_Rj_1_to_100_pct": 50.0 * np.mean(np.clip(np.log10(values), 0.0, 2.0)),
                              "N_low_reference_infeasible": int(group["low_reference_infeasible_global_5pct"].sum()), "N_right_censored": int(group["right_censored_global_5pct"].sum())})
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(COVERAGE, index=False, encoding="utf-8", float_format="%.17g")
    finite_errors = [state for state in states.values() if state["mode"] == "bisection"]
    maximum_error = max(math.exp(0.5 * math.log(x["hi"] / x["lo"])) - 1.0 for x in finite_errors) if finite_errors else 0.0
    audit = {"experiment_id": "V6-F2-S2-He-price-direct-log-bisection", "status": "PASS", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
             "device": "arc_16pancake_nuc600", "scenario": "S2", "coolant": "He", "prices_constant_2025_USD_per_kAm": list(PRICES),
             "source": {"brackets": {"path": str(BRACKETS.relative_to(ROOT)), "sha256": sha256(BRACKETS)}, "grid_optima": {"path": str(ARCH.relative_to(ROOT)), "sha256": sha256(ARCH)},
                        "reference_rows": {"path": str(REFS.relative_to(ROOT)), "sha256": sha256(REFS), "Q_HTS_ref_kA_m": q_ref}}, "fixed_af_max_S2": af_max,
             "bisection": {"space": "log_Rj", "target": 0.001, "maximum_observed_midpoint_relative_error": maximum_error, "criteria": "global-relative LCOE <=5%", "rho_optimization": "minimum price-recomputed LCOE among direct-evaluated strict-feasible rho samples"},
             "outputs": {path.name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for path in (EVALS, TOLERANCE, COVERAGE)}, "started_utc": started.isoformat()}
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()



