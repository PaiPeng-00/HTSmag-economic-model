#!/usr/bin/env python3
"""V6-F1 direct H2 bisection and audited 20 K He/H2 comparison table."""
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
BRACKETS = OUT / "v6f_f1_s2_20K_H2_grid_brackets.csv"
BRACKET_AUDIT = OUT / "v6f_f1_h2_grid_bracket_audit.json"
V6C = OUT / "v6c_joint_tolerance_boundaries.csv"
V6C_AUDIT = OUT / "v6c_direct_bisection_audit.json"
V6E_AUDIT = OUT / "v6e_cross_scenario_audit.json"
DRIVER = Path(__file__).with_name("8.13_run_v6_arc16pancake_nuc600_grid.py")
ANCHOR = Path(__file__).with_name("8.5_recompute_v2_anchor_economics.py")
RAW_DIR = OUT / "v6f_f1_h2_direct_bisection_raw"
EVALS = OUT / "v6f_f1_h2_direct_bisection_evaluations.csv"
OUTPUT = OUT / "v6f_coolant_sensitivity.csv"
COVERAGE = OUT / "v6f_coolant_coverage.csv"
AUDIT = OUT / "v6f_f1_h2_direct_bisection_audit.json"
RHO = np.array([10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000,5000,10000], dtype=float)
CRITERIA = ("feas", "condition_5pct")
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
    top, npw, iteration = 20.0, int(job["Npw"]), int(job["iteration"])
    requested = tuple(float(x) for x in job["R_joint_nOhm"])
    raw = RAW_DIR / f"iter_{iteration:02d}" / f"N{npw:03d}.csv"
    raw.parent.mkdir(parents=True, exist_ok=True)
    if raw.exists():
        raise FileExistsError(f"refusing to overwrite {raw}")
    driver = import_path(f"v6f_f1_driver_{os.getpid()}_{iteration}_{npw}", DRIVER)
    scan = driver.load_scan(output_csv=raw, npw=np.array([npw]), rjoint=np.array(requested), rho=RHO, temperatures=[(top, "H2")])
    scan.SCENARIOS = ["S2"]
    scan.main()
    anchor = import_path(f"v6f_f1_anchor_{os.getpid()}", ANCHOR)
    manifest = json.loads(anchor.E0_MANIFEST.read_text(encoding="utf-8"))
    transformed = anchor.transform_chunk(pd.read_csv(raw, dtype=str, keep_default_na=False), job["anchor_refs"], manifest)
    transformed["AF_ref"] = pd.to_numeric(transformed["AF"], errors="coerce") / float(job["af_max"])
    rcryo = pd.to_numeric(transformed["r_cryo_re_fraction"], errors="coerce")
    net = pd.to_numeric(transformed["net_energy_annual_MWh"], errors="coerce")
    lcoe = pd.to_numeric(transformed["lcoe_anchor_USD_per_MWh"], errors="coerce")
    valid = (transformed["status"].astype(str).str.lower().eq("success") & transformed["anchor_economic_valid"].astype(str).str.lower().isin(("true", "1"))
             & np.isfinite(transformed["AF_ref"]) & np.isfinite(rcryo) & np.isfinite(net) & np.isfinite(lcoe)
             & (transformed["AF_ref"] >= 0.99) & (rcryo <= 0.50) & (net > 0.0) & (lcoe > 0.0))
    records: list[dict] = []
    for r_value, group in transformed.groupby("R_joint_nOhm", sort=True):
        candidates = group.loc[valid.loc[group.index]]
        if candidates.empty:
            records.append({"Top_K": top, "Npw": npw, "R_joint_nOhm": float(r_value), "feasible": False, "lcoe_anchor_USD_per_MWh": math.nan,
                            "best_rho_turn_uOhm_cm2": math.nan, "raw_path": str(raw.relative_to(ROOT)), "raw_sha256": sha256(raw)})
        else:
            best = candidates.loc[pd.to_numeric(candidates["lcoe_anchor_USD_per_MWh"], errors="coerce").idxmin()]
            records.append({"Top_K": top, "Npw": npw, "R_joint_nOhm": float(r_value), "feasible": True,
                            "lcoe_anchor_USD_per_MWh": float(best["lcoe_anchor_USD_per_MWh"]), "best_rho_turn_uOhm_cm2": float(best["rho_turn_uOhm_cm2"]),
                            "raw_path": str(raw.relative_to(ROOT)), "raw_sha256": sha256(raw)})
    return records


def main() -> None:
    required = (BRACKETS, BRACKET_AUDIT, V6C, V6C_AUDIT, V6E_AUDIT)
    if not all(path.exists() for path in required):
        raise FileNotFoundError("V6-F1 brackets and V6-C/E audited inputs are required")
    if any(path.exists() for path in (RAW_DIR, EVALS, OUTPUT, COVERAGE, AUDIT)):
        raise FileExistsError("refusing to overwrite V6-F1 direct-bisection artifacts")
    bracket_audit, v6c_audit, v6e = (json.loads(path.read_text(encoding="utf-8")) for path in (BRACKET_AUDIT, V6C_AUDIT, V6E_AUDIT))
    if bracket_audit.get("status") != "PASS_GRID_BRACKETS_NOT_FINAL_BOUNDARIES" or v6c_audit.get("status") != "PASS" or v6e.get("status") != "PASS":
        raise AssertionError("V6-F1 upstream audit gate is not PASS")
    if sha256(BRACKETS) != bracket_audit["outputs"][BRACKETS.name]["sha256"]:
        raise AssertionError("V6-F1 brackets changed after audit")
    af_max = float(v6e["fixed_af_max_by_scenario"]["S2"])
    lstar = float(bracket_audit["definition"]["L_star_condition_USD_per_MWh"])
    brackets = pd.read_csv(BRACKETS)
    states: dict[tuple[int, str], dict] = {}
    for row in brackets.to_dict("records"):
        for criterion in CRITERIA:
            key = (int(row["Npw"]), criterion)
            if bool(row[f"low_reference_infeasible_{criterion}"]):
                states[key] = {"mode": "low_ref", "lo": math.nan, "hi": math.nan, "iterations": 0}
            elif bool(row[f"right_censored_{criterion}"]):
                states[key] = {"mode": "right_censored", "lo": math.nan, "hi": math.nan, "iterations": 0}
            else:
                states[key] = {"mode": "bisection", "lo": float(row[f"Rj_max_{criterion}_grid_lower_nOhm"]), "hi": float(row[f"Rj_bracket_{criterion}_upper_nOhm"]), "iterations": 0}
    anchor_module = import_path("v6f_f1_parent_anchor", ANCHOR)
    anchor_manifest = json.loads(anchor_module.E0_MANIFEST.read_text(encoding="utf-8"))
    _, anchor_refs = anchor_module.find_reference_rows(anchor_manifest)
    RAW_DIR.mkdir(parents=True)
    evaluations: list[dict] = []
    started = datetime.now(timezone.utc)
    for iteration in range(1, MAX_ITERATIONS + 1):
        candidates: dict[int, set[float]] = {}
        task_mid: dict[tuple[int, str], float] = {}
        for key, state in states.items():
            if state["mode"] != "bisection" or 0.5 * math.log(state["hi"] / state["lo"]) <= LOG_HALF_ERROR_TARGET:
                continue
            mid = round(math.sqrt(state["lo"] * state["hi"]), 3)
            if not state["lo"] < mid < state["hi"]:
                raise AssertionError(f"Rj rounding prevents refinement for {key}")
            task_mid[key] = mid
            candidates.setdefault(key[0], set()).add(mid)
        if not task_mid:
            break
        jobs = [{"Npw": npw, "R_joint_nOhm": sorted(values), "iteration": iteration, "af_max": af_max, "anchor_refs": anchor_refs} for npw, values in sorted(candidates.items())]
        results: dict[tuple[int, float], dict] = {}
        with concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS) as pool:
            for result in pool.map(worker, jobs):
                for record in result:
                    results[(record["Npw"], record["R_joint_nOhm"])] = record
                    evaluations.append({"iteration": iteration, **record})
        for key, mid in task_mid.items():
            record = results[(key[0], mid)]
            state = states[key]
            passed = bool(record["feasible"]) and (key[1] == "feas" or float(record["lcoe_anchor_USD_per_MWh"]) <= lstar * 1.05)
            if passed:
                state["lo"] = mid
            else:
                state["hi"] = mid
            state["iterations"] += 1
        print(f"[Progress] V6-F1 H2 direct bisection iteration {iteration}: {len(task_mid)} criterion midpoints across {len(jobs)} model batches", flush=True)
    unresolved = [key for key, state in states.items() if state["mode"] == "bisection" and 0.5 * math.log(state["hi"] / state["lo"]) > LOG_HALF_ERROR_TARGET]
    if unresolved:
        raise AssertionError(f"V6-F1 bisection unresolved: {len(unresolved)}")
    pd.DataFrame(evaluations).to_csv(EVALS, index=False, encoding="utf-8", float_format="%.17g")
    h2_rows = []
    for npw in range(1, 201):
        row = {"scenario": "S2", "Top_K": 20.0, "coolant": "H2", "Npw": npw, "R_ref_nOhm": 1.0, "L_star_condition_USD_per_MWh": lstar}
        for criterion in CRITERIA:
            state = states[(npw, criterion)]
            if state["mode"] == "low_ref":
                boundary = lower = upper = error = math.nan; low, censored = True, False
            elif state["mode"] == "right_censored":
                boundary, lower, upper, error, low, censored = 100.0, 100.0, math.nan, 0.0, False, True
            else:
                lower, upper = state["lo"], state["hi"]; boundary = math.sqrt(lower * upper); error = math.exp(0.5 * math.log(upper / lower)) - 1.0; low = censored = False
            row.update({f"Rj_max_{criterion}_nOhm": boundary, f"Rj_lower_{criterion}_nOhm": lower, f"Rj_upper_{criterion}_nOhm": upper,
                        f"relative_boundary_error_{criterion}": error, f"low_reference_infeasible_{criterion}": low, f"right_censored_{criterion}": censored,
                        f"bisection_iterations_{criterion}": state["iterations"]})
        h2_rows.append(row)
    h2 = pd.DataFrame(h2_rows)
    he_raw = pd.read_csv(V6C)
    he_raw = he_raw.loc[(he_raw["Top_K"].eq(20.0)) & he_raw["coolant"].eq("He")].copy()
    if len(he_raw) != 200:
        raise AssertionError("expected 200 audited V6-C 20 K He rows")
    he = pd.DataFrame({"scenario": "S2", "Top_K": 20.0, "coolant": "He", "Npw": he_raw["Npw"], "R_ref_nOhm": 1.0,
                       "L_star_condition_USD_per_MWh": he_raw["L_star_temperature_USD_per_MWh"],
                       "Rj_max_feas_nOhm": he_raw["Rj_max_feas_nOhm"], "Rj_lower_feas_nOhm": he_raw["Rj_lower_feas_nOhm"], "Rj_upper_feas_nOhm": he_raw["Rj_upper_feas_nOhm"],
                       "relative_boundary_error_feas": he_raw["relative_boundary_error_feas"], "low_reference_infeasible_feas": he_raw["low_reference_infeasible_feas"], "right_censored_feas": he_raw["right_censored_feas"], "bisection_iterations_feas": he_raw["bisection_iterations_feas"],
                       "Rj_max_condition_5pct_nOhm": he_raw["Rj_max_temp_5pct_nOhm"], "Rj_lower_condition_5pct_nOhm": he_raw["Rj_lower_temp_5pct_nOhm"], "Rj_upper_condition_5pct_nOhm": he_raw["Rj_upper_temp_5pct_nOhm"],
                       "relative_boundary_error_condition_5pct": he_raw["relative_boundary_error_temp_5pct"], "low_reference_infeasible_condition_5pct": he_raw["low_reference_infeasible_temp_5pct"], "right_censored_condition_5pct": he_raw["right_censored_temp_5pct"], "bisection_iterations_condition_5pct": he_raw["bisection_iterations_temp_5pct"]})
    combined = pd.concat([he, h2], ignore_index=True).sort_values(["coolant", "Npw"]).reset_index(drop=True)
    combined.to_csv(OUTPUT, index=False, encoding="utf-8", float_format="%.17g")
    coverage_rows = []
    for coolant, group in combined.groupby("coolant", sort=True):
        for criterion in CRITERIA:
            values = group[f"Rj_max_{criterion}_nOhm"].to_numpy(float)
            coverage_rows.append({"scenario": "S2", "Top_K": 20.0, "coolant": coolant, "criterion": criterion,
                                  "coverage_log_Rj_1_to_10_pct": 100.0 * np.mean(np.clip(np.log10(values), 0.0, 1.0)),
                                  "coverage_log_Rj_1_to_100_pct": 50.0 * np.mean(np.clip(np.log10(values), 0.0, 2.0)),
                                  "N_low_reference_infeasible": int(group[f"low_reference_infeasible_{criterion}"].sum()), "N_right_censored": int(group[f"right_censored_{criterion}"].sum())})
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(COVERAGE, index=False, encoding="utf-8", float_format="%.17g")
    maximum_error = max(math.exp(0.5 * math.log(x["hi"] / x["lo"])) - 1.0 for x in states.values() if x["mode"] == "bisection")
    audit = {"experiment_id": "V6-F1-S2-20K-H2-direct-log-bisection", "status": "PASS", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
             "device": "arc_16pancake_nuc600", "scenario": "S2", "comparison": "20 K He vs H2", "source": {"h2_brackets": {"path": str(BRACKETS.relative_to(ROOT)), "sha256": sha256(BRACKETS)}, "v6c_he": {"path": str(V6C.relative_to(ROOT)), "sha256": sha256(V6C)}},
             "definition": {"conditions": ["feasibility", "coolant-condition-relative LCOE <=5%"], "H2_L_star_USD_per_MWh": lstar, "He_L_star_from": "audited V6-C 20K temperature-relative reference"},
             "bisection": {"space": "log_Rj", "target": 0.001, "maximum_observed_midpoint_relative_error": maximum_error, "rho_optimization": "minimum frozen-B1-anchor LCOE among direct-evaluated strict-feasible rho samples"},
             "outputs": {path.name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for path in (EVALS, OUTPUT, COVERAGE)}, "started_utc": started.isoformat()}
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

