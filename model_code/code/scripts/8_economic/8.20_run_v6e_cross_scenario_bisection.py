#!/usr/bin/env python3
"""V6-E cross-scenario direct log-Rj tolerance bisection.

Uses the same strict denominator, first-continuous interval rule, frozen B1
economics, and direct model evaluations as V6-C.  This script intentionally
does not interpolate temperature or join remote re-entry islands.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import importlib.util
import json
import math
import os
import subprocess
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "v6_hts_temperature_tolerance_arc16pancake_nuc600"
INPUT = OUT / "v6b_feasible_designs.csv"
INPUT_AUDIT = OUT / "v6b_status_feasibility_audit.json"
ANCHOR_INPUT = OUT / "v6_b1_anchor_economics_arc16pancake_nuc600.csv"
ANCHOR_AUDIT = OUT / "v6_b1_anchor_economics_audit.json"
DRIVER = Path(__file__).with_name("8.13_run_v6_arc16pancake_nuc600_grid.py")
ANCHOR = Path(__file__).with_name("8.5_recompute_v2_anchor_economics.py")
ARCH = OUT / "v6e_scenario_architecture_optima_grid.csv"
BRACKETS = OUT / "v6e_scenario_joint_tolerance_grid_brackets.csv"
MONOTONICITY = OUT / "v6e_scenario_monotonicity_audit.csv"
GRID_AUDIT = OUT / "v6e_grid_bracket_audit.json"
RAW_DIR = OUT / "v6e_direct_bisection_raw"
EVALS = OUT / "v6e_direct_bisection_evaluations.csv"
TOLERANCE = OUT / "v6e_cross_scenario_tolerance.csv"
BINDING = OUT / "v6e_binding_scenario.csv"
COVERAGE = OUT / "v6e_cross_scenario_coverage.csv"
AUDIT = OUT / "v6e_cross_scenario_audit.json"

SCENARIOS = ("S1", "S2", "S3")
TEMPERATURES = (4.2, 10.0, 20.0)
N_VALUES = tuple(range(1, 201))
CRITERIA = ("feas", "temp_5pct", "global_5pct")
RHO = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 5000, 10000], dtype=float)
MAX_ITERATIONS = 50
LOG_HALF_ERROR_TARGET = math.log(1.001)
WORKERS = 4
CHUNK = 200_000


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


def trace_first_interval(r: np.ndarray, passed: np.ndarray, criterion: str) -> dict:
    if not bool(passed[0]):
        return {"lower": math.nan, "upper": math.nan, "low": True, "censored": False,
                "reentry": bool(np.any((~passed[:-1]) & passed[1:])), "binding": "availability"}
    failed = np.flatnonzero(~passed)
    reentry = bool(np.any((~passed[:-1]) & passed[1:]))
    if not len(failed):
        return {"lower": float(r[-1]), "upper": math.nan, "low": False, "censored": True,
                "reentry": reentry, "binding": "right_censored_at_100_nOhm"}
    ix = int(failed[0])
    binding = "availability" if criterion == "feas" else ("temperature_relative_LCOE" if criterion == "temp_5pct" else "global_relative_LCOE")
    return {"lower": float(r[ix - 1]), "upper": float(r[ix]), "low": False, "censored": False,
            "reentry": reentry, "binding": binding}


def fixed_af_max_by_scenario() -> dict[str, float]:
    values: dict[str, list[np.ndarray]] = {s: [] for s in SCENARIOS}
    for chunk in pd.read_csv(ANCHOR_INPUT, usecols=["scenario", "AF", "AF_ref"], chunksize=CHUNK, low_memory=False):
        for scenario in SCENARIOS:
            part = chunk.loc[chunk["scenario"].eq(scenario)]
            ratio = pd.to_numeric(part["AF"], errors="coerce").to_numpy(float) / pd.to_numeric(part["AF_ref"], errors="coerce").to_numpy(float)
            ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
            if len(ratio):
                values[scenario].append(ratio)
    output = {}
    for scenario, pieces in values.items():
        data = np.concatenate(pieces)
        ref = float(np.median(data))
        if not np.allclose(data, ref, rtol=0, atol=1e-12):
            raise AssertionError(f"fixed AF maximum not unique for {scenario}")
        output[scenario] = ref
    return output


def worker(job: dict) -> list[dict]:
    scenario = str(job["scenario"])
    top = float(job["Top_K"])
    npw = int(job["Npw"])
    iteration = int(job["iteration"])
    requested = tuple(float(x) for x in job["R_joint_nOhm"])
    raw = RAW_DIR / f"iter_{iteration:02d}" / scenario / f"T{top:g}K_N{npw:03d}.csv"
    raw.parent.mkdir(parents=True, exist_ok=True)
    if raw.exists():
        raise FileExistsError(f"refusing to overwrite direct evaluation {raw}")
    driver = import_path(f"v6e_driver_{os.getpid()}_{iteration}_{scenario}_{npw}", DRIVER)
    scan = driver.load_scan(output_csv=raw, npw=np.array([npw]), rjoint=np.array(requested), rho=RHO, temperatures=[(top, "He")])
    scan.SCENARIOS = [scenario]
    scan.main()
    anchor = import_path(f"v6e_anchor_{os.getpid()}_{scenario}", ANCHOR)
    e0_manifest = json.loads(anchor.E0_MANIFEST.read_text(encoding="utf-8"))
    _, refs = anchor.find_reference_rows(e0_manifest)
    transformed = anchor.transform_chunk(pd.read_csv(raw, dtype=str, keep_default_na=False), refs, e0_manifest)
    transformed["AF_ref"] = pd.to_numeric(transformed["AF"], errors="coerce") / float(job["af_max"])
    rcryo = pd.to_numeric(transformed["r_cryo_re_fraction"], errors="coerce")
    net = pd.to_numeric(transformed["net_energy_annual_MWh"], errors="coerce")
    lcoe = pd.to_numeric(transformed["lcoe_anchor_USD_per_MWh"], errors="coerce")
    valid = (transformed["status"].astype(str).str.lower().eq("success")
             & transformed["anchor_economic_valid"].astype(str).str.lower().isin(("true", "1"))
             & np.isfinite(transformed["AF_ref"]) & np.isfinite(rcryo) & np.isfinite(net) & np.isfinite(lcoe)
             & (transformed["AF_ref"] >= .99) & (rcryo <= .50) & (net > 0) & (lcoe > 0))
    raw_hash = sha256(raw)
    records = []
    for r_value, group in transformed.groupby("R_joint_nOhm", sort=True):
        candidates = group.loc[valid.loc[group.index]]
        base = {"scenario": scenario, "Top_K": top, "coolant": "He", "Npw": npw, "R_joint_nOhm": float(r_value),
                "raw_path": str(raw.relative_to(ROOT)), "raw_sha256": raw_hash}
        if candidates.empty:
            records.append({**base, "feasible": False, "best_rho_turn_uOhm_cm2": math.nan, "AF_ref": math.nan,
                            "r_cryo_re_fraction": math.nan, "net_energy_annual_MWh": math.nan, "lcoe_anchor_USD_per_MWh": math.nan})
        else:
            best = candidates.loc[pd.to_numeric(candidates["lcoe_anchor_USD_per_MWh"], errors="coerce").idxmin()]
            records.append({**base, "feasible": True, "best_rho_turn_uOhm_cm2": float(best["rho_turn_uOhm_cm2"]),
                            "AF_ref": float(best["AF_ref"]), "r_cryo_re_fraction": float(best["r_cryo_re_fraction"]),
                            "net_energy_annual_MWh": float(best["net_energy_annual_MWh"]),
                            "lcoe_anchor_USD_per_MWh": float(best["lcoe_anchor_USD_per_MWh"])})
    return records


def passed(record: dict, criterion: str, temp_reference: float, global_reference: float) -> bool:
    if not record["feasible"]:
        return False
    if criterion == "feas":
        return True
    reference = temp_reference if criterion == "temp_5pct" else global_reference
    return float(record["lcoe_anchor_USD_per_MWh"]) <= 1.05 * reference


def coverage(boundary: np.ndarray, upper: float, measure: str) -> float:
    clipped = np.clip(np.nan_to_num(boundary, nan=1.0), 1.0, upper)
    if measure == "log_Rj":
        return float(np.log(clipped).mean() / math.log(upper))
    return float(((clipped - 1.0) / (upper - 1.0)).mean())


def main() -> None:
    required = (INPUT, INPUT_AUDIT, ANCHOR_INPUT, ANCHOR_AUDIT)
    if not all(path.exists() for path in required):
        raise FileNotFoundError("audited V6-B strict input or anchored input missing")
    outputs = (ARCH, BRACKETS, MONOTONICITY, GRID_AUDIT, RAW_DIR, EVALS, TOLERANCE, BINDING, COVERAGE, AUDIT)
    if any(path.exists() for path in outputs):
        raise FileExistsError("refusing to overwrite V6-E artifacts")
    v6b = json.loads(INPUT_AUDIT.read_text(encoding="utf-8"))
    b1 = json.loads(ANCHOR_AUDIT.read_text(encoding="utf-8"))
    if v6b.get("status") != "PASS" or b1.get("status") != "PASS" or sha256(INPUT) != v6b["outputs"]["feasible_designs"]["sha256"] or sha256(ANCHOR_INPUT) != b1["output"]["sha256"]:
        raise AssertionError("upstream V6-B or B1 audit/hash gate failed")
    started = datetime.now(timezone.utc)
    # Grid-stage architecture optima, using only V6-B strict-feasible implementations.
    best: dict[tuple[str, float, int, float], dict] = {}
    columns = ["scenario", "Top_K", "coolant", "Npw", "rho_turn_uOhm_cm2", "R_joint_nOhm", "AF_ref", "r_cryo_re_fraction", "net_energy_annual_MWh", "lcoe_anchor_USD_per_MWh"]
    for chunk in pd.read_csv(INPUT, usecols=columns, chunksize=CHUNK, low_memory=False):
        data = chunk.loc[chunk["scenario"].isin(SCENARIOS) & chunk["coolant"].eq("He") & chunk["Top_K"].isin(TEMPERATURES)].copy()
        if data.empty:
            continue
        data["lcoe_anchor_USD_per_MWh"] = pd.to_numeric(data["lcoe_anchor_USD_per_MWh"], errors="raise")
        winners = data.loc[data.groupby(["scenario", "Top_K", "Npw", "R_joint_nOhm"], sort=False)["lcoe_anchor_USD_per_MWh"].idxmin()]
        for row in winners.to_dict("records"):
            key = (str(row["scenario"]), float(row["Top_K"]), int(row["Npw"]), float(row["R_joint_nOhm"]))
            if key not in best or float(row["lcoe_anchor_USD_per_MWh"]) < float(best[key]["lcoe_anchor_USD_per_MWh"]):
                best[key] = row
    best_df = pd.DataFrame(best.values())
    r_values = np.sort(best_df["R_joint_nOhm"].unique().astype(float))
    if len(r_values) != 121 or not np.isclose(r_values[0], 1.) or not np.isclose(r_values[-1], 100.):
        raise AssertionError("unexpected stored V6 Rj grid")
    full = pd.DataFrame(product(SCENARIOS, TEMPERATURES, N_VALUES, r_values), columns=["scenario", "Top_K", "Npw", "R_joint_nOhm"])
    arch = full.merge(best_df, on=["scenario", "Top_K", "Npw", "R_joint_nOhm"], how="left", validate="one_to_one", indicator=True)
    arch["architecture_feasible"] = arch["_merge"].eq("both")
    arch.drop(columns="_merge", inplace=True)
    refs = arch.loc[arch["architecture_feasible"]].groupby(["scenario", "Top_K"])["lcoe_anchor_USD_per_MWh"].min().to_dict()
    global_refs = arch.loc[arch["architecture_feasible"]].groupby("scenario")["lcoe_anchor_USD_per_MWh"].min().to_dict()
    if len(refs) != 9 or set(global_refs) != set(SCENARIOS):
        raise AssertionError("scenario/temperature LCOE reference minima missing")
    arch.to_csv(ARCH, index=False, encoding="utf-8", float_format="%.17g")
    bracket_rows, mono_rows = [], []
    for scenario, top, n in product(SCENARIOS, TEMPERATURES, N_VALUES):
        frame = arch.loc[(arch["scenario"].eq(scenario)) & (arch["Top_K"].eq(top)) & (arch["Npw"].eq(n))].sort_values("R_joint_nOhm")
        feasibility = frame["architecture_feasible"].to_numpy(bool)
        lcoe = pd.to_numeric(frame["lcoe_anchor_USD_per_MWh"], errors="coerce").to_numpy(float)
        conditions = {"feas": feasibility, "temp_5pct": feasibility & (lcoe <= 1.05 * refs[(scenario, top)]), "global_5pct": feasibility & (lcoe <= 1.05 * global_refs[scenario])}
        out = {"scenario": scenario, "Top_K": top, "coolant": "He", "Npw": n, "R_ref_nOhm": 1., "L_star_temperature_USD_per_MWh": refs[(scenario, top)], "L_star_global_USD_per_MWh": global_refs[scenario]}
        for criterion, values in conditions.items():
            trace = trace_first_interval(r_values, values, criterion)
            out.update({f"Rj_max_{criterion}_grid_lower_nOhm": trace["lower"], f"Rj_bracket_{criterion}_upper_nOhm": trace["upper"], f"low_reference_infeasible_{criterion}": trace["low"], f"right_censored_{criterion}": trace["censored"], f"first_interval_reentry_{criterion}": trace["reentry"], f"binding_condition_{criterion}": trace["binding"]})
            mono_rows.append({"scenario": scenario, "Top_K": top, "coolant": "He", "Npw": n, "criterion": criterion, "reference_pass": bool(values[0]), "pass_count": int(values.sum()), "true_to_false_transitions": int(np.sum(values[:-1] & ~values[1:])), "false_to_true_transitions": int(np.sum(~values[:-1] & values[1:])), "reentry_after_first_failure": trace["reentry"], "first_interval_only": True})
        bracket_rows.append(out)
    brackets = pd.DataFrame(bracket_rows)
    brackets.to_csv(BRACKETS, index=False, encoding="utf-8", float_format="%.17g")
    pd.DataFrame(mono_rows).to_csv(MONOTONICITY, index=False, encoding="utf-8")
    grid_audit = {"experiment_id": "V6-E-grid-brackets", "status": "PASS_GRID_BRACKETS_NOT_FINAL_BOUNDARIES", "timestamp_utc": datetime.now(timezone.utc).isoformat(), "source_hash": sha256(INPUT), "rows": {"architecture_optima": len(arch), "brackets": len(brackets), "monotonicity": len(mono_rows)}, "references": {"temperature": {f"{s}|{t:g}": float(v) for (s, t), v in refs.items()}, "global": {s: float(v) for s, v in global_refs.items()}}, "definition": "strict V6-B candidates; first continuous interval from 1 nOhm; direct bisection required for finite boundaries"}
    GRID_AUDIT.write_text(json.dumps(grid_audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    af_max = fixed_af_max_by_scenario()
    states: dict[tuple[str, float, int, str], dict] = {}
    for row in brackets.to_dict("records"):
        for criterion in CRITERIA:
            key = (str(row["scenario"]), float(row["Top_K"]), int(row["Npw"]), criterion)
            if bool(row[f"low_reference_infeasible_{criterion}"]):
                states[key] = {"mode": "low", "lo": math.nan, "hi": math.nan, "iterations": 0}
            elif bool(row[f"right_censored_{criterion}"]):
                states[key] = {"mode": "censored", "lo": math.nan, "hi": math.nan, "iterations": 0}
            else:
                states[key] = {"mode": "bisection", "lo": float(row[f"Rj_max_{criterion}_grid_lower_nOhm"]), "hi": float(row[f"Rj_bracket_{criterion}_upper_nOhm"]), "iterations": 0}
    RAW_DIR.mkdir(parents=True)
    evaluations = []
    for iteration in range(1, MAX_ITERATIONS + 1):
        requested: dict[tuple[str, float, int], set[float]] = {}
        task_mid: dict[tuple[str, float, int, str], float] = {}
        for key, state in states.items():
            if state["mode"] != "bisection" or .5 * math.log(state["hi"] / state["lo"]) <= LOG_HALF_ERROR_TARGET:
                continue
            mid = round(math.sqrt(state["lo"] * state["hi"]), 3)
            if not state["lo"] < mid < state["hi"]:
                raise AssertionError(f"rounding stopped V6-E bisection: {key}")
            task_mid[key] = mid
            requested.setdefault(key[:3], set()).add(mid)
        if not task_mid:
            break
        jobs = [{"scenario": scenario, "Top_K": top, "Npw": n, "R_joint_nOhm": sorted(r), "iteration": iteration, "af_max": af_max[scenario]} for (scenario, top, n), r in sorted(requested.items())]
        records: dict[tuple[str, float, int, float], dict] = {}
        with concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS) as pool:
            for result in pool.map(worker, jobs):
                for record in result:
                    records[(record["scenario"], record["Top_K"], record["Npw"], record["R_joint_nOhm"])] = record
                    evaluations.append({"iteration": iteration, **record})
        for key, mid in task_mid.items():
            record = records.get((key[0], key[1], key[2], mid))
            if record is None:
                raise AssertionError(f"missing V6-E direct evaluation {key} at {mid}")
            state = states[key]
            if passed(record, key[3], refs[(key[0], key[1])], global_refs[key[0]]):
                state["lo"] = mid
            else:
                state["hi"] = mid
            state["iterations"] += 1
        print(f"[Progress] V6-E direct bisection iteration {iteration}: {len(task_mid)} criterion midpoints across {len(jobs)} model batches", flush=True)
    unresolved = [key for key, state in states.items() if state["mode"] == "bisection" and .5 * math.log(state["hi"] / state["lo"]) > LOG_HALF_ERROR_TARGET]
    if unresolved:
        raise AssertionError(f"V6-E did not reach 0.1% direct-boundary target: {len(unresolved)}")
    pd.DataFrame(evaluations).to_csv(EVALS, index=False, encoding="utf-8", float_format="%.17g")

    rows = []
    for row in brackets.to_dict("records"):
        out = {k: row[k] for k in ("scenario", "Top_K", "coolant", "Npw", "R_ref_nOhm", "L_star_temperature_USD_per_MWh", "L_star_global_USD_per_MWh")}
        for criterion in CRITERIA:
            state = states[(str(row["scenario"]), float(row["Top_K"]), int(row["Npw"]), criterion)]
            if state["mode"] == "low":
                boundary, low, censored, lower, upper, error = math.nan, True, False, math.nan, math.nan, math.nan
            elif state["mode"] == "censored":
                boundary, low, censored, lower, upper, error = 100., False, True, 100., math.nan, 0.
            else:
                lower, upper = state["lo"], state["hi"]
                boundary, low, censored = math.sqrt(lower * upper), False, False
                error = math.exp(.5 * math.log(upper / lower)) - 1.
            out.update({f"Rj_max_{criterion}_nOhm": boundary, f"F_R_{criterion}": boundary, f"Rj_lower_{criterion}_nOhm": lower, f"Rj_upper_{criterion}_nOhm": upper, f"relative_boundary_error_{criterion}": error, f"low_reference_infeasible_{criterion}": low, f"right_censored_{criterion}": censored, f"bisection_iterations_{criterion}": state["iterations"], f"binding_condition_{criterion}": row[f"binding_condition_{criterion}"], f"first_interval_reentry_{criterion}": bool(row[f"first_interval_reentry_{criterion}"])})
        rows.append(out)
    tolerance = pd.DataFrame(rows)
    tolerance.to_csv(TOLERANCE, index=False, encoding="utf-8", float_format="%.17g")
    cross_rows, binding_rows = [], []
    for top, n in product(TEMPERATURES, N_VALUES):
        group = tolerance.loc[(tolerance["Top_K"].eq(top)) & tolerance["Npw"].eq(n)].copy()
        values = pd.to_numeric(group["Rj_max_global_5pct_nOhm"], errors="coerce")
        if values.notna().any():
            min_value = float(values.min())
            binding_scenarios = ";".join(group.loc[np.isclose(values, min_value, rtol=0, atol=1e-12), "scenario"].tolist())
            low = False
        else:
            min_value, binding_scenarios, low = math.nan, ";".join(group.loc[group["low_reference_infeasible_global_5pct"], "scenario"].tolist()), True
        censored = bool(group["right_censored_global_5pct"].all())
        cross_rows.append({"Top_K": top, "coolant": "He", "Npw": n, "R_ref_nOhm": 1., "Rj_max_cross_global_5pct_nOhm": min_value, "F_R_cross_global_5pct": min_value, "binding_scenario_global_5pct": binding_scenarios, "low_reference_infeasible_cross_global_5pct": low, "right_censored_cross_global_5pct": censored})
        binding_rows.append({"Top_K": top, "coolant": "He", "Npw": n, "criterion": "cross_global_5pct", "binding_scenario": binding_scenarios, "Rj_max_cross_global_5pct_nOhm": min_value, "low_reference_infeasible": low, "right_censored": censored})
    cross = pd.DataFrame(cross_rows)
    tolerance = tolerance.merge(cross, on=["Top_K", "coolant", "Npw", "R_ref_nOhm"], how="left", validate="many_to_one")
    tolerance.to_csv(TOLERANCE, index=False, encoding="utf-8", float_format="%.17g")
    pd.DataFrame(binding_rows).to_csv(BINDING, index=False, encoding="utf-8")
    coverage_rows = []
    for top in TEMPERATURES:
        values = cross.loc[cross["Top_K"].eq(top), "Rj_max_cross_global_5pct_nOhm"].to_numpy(float)
        for upper, domain in ((10., "engineering_1_to_10_nOhm"), (100., "stress_1_to_100_nOhm")):
            for measure in ("log_Rj", "linear_Rj"):
                value = coverage(values, upper, measure)
                coverage_rows.append({"experiment_id": "V6-E-cross-scenario-coverage", "model_version": "V6-E-direct-log-bisection", "source_hash": sha256(TOLERANCE), "Top_K": top, "coolant": "He", "criterion": "cross_global_5pct", "measure": measure, "R_min_nOhm": 1., "R_upper_nOhm": upper, "domain": domain, "Npw_weighting": "equal_count_1_to_200", "coverage": value, "coverage_percent": 100. * value, "N_low_reference_infeasible": int(cross.loc[cross["Top_K"].eq(top), "low_reference_infeasible_cross_global_5pct"].sum()), "N_right_censored": int(cross.loc[cross["Top_K"].eq(top), "right_censored_cross_global_5pct"].sum())})
    pd.DataFrame(coverage_rows).to_csv(COVERAGE, index=False, encoding="utf-8", float_format="%.17g")
    max_error = max(math.exp(.5 * math.log(s["hi"] / s["lo"])) - 1. for s in states.values() if s["mode"] == "bisection")
    audit = {"experiment_id": "V6-E-cross-scenario-direct-log-bisection", "status": "PASS", "timestamp_utc": datetime.now(timezone.utc).isoformat(), "device": "arc_16pancake_nuc600", "scenarios": list(SCENARIOS), "coolant": "He", "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "source": {"v6b_feasible": {"sha256": sha256(INPUT)}, "v6_b1_anchor": {"sha256": sha256(ANCHOR_INPUT)}}, "fixed_af_max_by_scenario": af_max, "method": {"criteria": list(CRITERIA), "interval": "first continuous interval from 1 nOhm", "space": "log_Rj", "maximum_observed_midpoint_relative_error": max_error, "target": .001, "rho_optimization": "minimum frozen-B1-anchor LCOE among direct-evaluated strict-feasible rho samples", "temperature_interpolation": "none", "remote_reentry_islands": "excluded"}, "outputs": {path.name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for path in (TOLERANCE, BINDING, COVERAGE, EVALS)}, "started_utc": started.isoformat()}
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
