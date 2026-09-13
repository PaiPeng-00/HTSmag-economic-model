#!/usr/bin/env python3
"""V6.2-E cross-scenario direct log-Rj tolerance bisection using six fast persistent direct workers.

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
import pyarrow.dataset as ds


ROOT = Path(__file__).resolve().parents[3]
V6_BASE = Path(os.environ['V6_RUN_BASE']) if os.environ.get('V6_RUN_BASE') else ROOT / 'v6_2_arc_actual_inductance'
OUT = V6_BASE / 'stage_E_fast'
INPUT = V6_BASE / 'stage_B1' / 'v6_2_b_strict_feasible_designs.csv'
INPUT_AUDIT = V6_BASE / 'stage_B1' / 'v6_2_b1_and_strict_feasibility_audit.json'
PARQUET_ROOT = V6_BASE / 'stage_B1_parquet_v1' / 'v6_2_b1_strict_feasible_parquet'
PARQUET_AUDIT = V6_BASE / 'stage_B1_parquet_v1' / 'v6_2_b1_strict_parquet_audit.json'
AVAIL_AUDIT = V6_BASE / 'stage_A' / 'v6_2_a_availability_audit.json'
FAST_WORKER = Path(__file__).with_name('8.45_refine_v6_2_c_fast_r2.py')
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
RHO = np.geomspace(10.0, 10000.0, 61)
MAX_ITERATIONS = 50
LOG_HALF_ERROR_TARGET = math.log(1.001)
WORKERS = int(os.environ.get("V6_E_WORKERS", "4"))
if WORKERS < 1:
    raise ValueError("V6_E_WORKERS must be a positive integer")
CHUNK = 200_000
_FAST_MODULE = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_path(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


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


def fast_module():
    global _FAST_MODULE
    if _FAST_MODULE is None:
        _FAST_MODULE = import_path('v62_fast_direct_worker', FAST_WORKER)
    return _FAST_MODULE


def verify_plant_availability_contract() -> float:
    audit = json.loads(AVAIL_AUDIT.read_text(encoding='utf-8'))
    if audit.get('status') != 'PASS' or audit.get('device') != 'arc_16pancake_nuc600_v6_2':
        raise AssertionError('V6.2-A availability audit is not PASS')
    requirement = float(audit['Aplant_requirement'])
    if requirement != 0.80:
        raise AssertionError(f'unexpected V7 plant-availability requirement: {requirement}')
    return requirement


def worker(job: dict) -> list[dict]:
    direct = fast_module()
    direct_job = dict(job)
    direct_job['raw_root'] = str(RAW_DIR)
    return direct.worker(direct_job)


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
    use_parquet = PARQUET_ROOT.exists() and PARQUET_AUDIT.exists()
    required = ((PARQUET_ROOT, PARQUET_AUDIT, AVAIL_AUDIT, FAST_WORKER)
                if use_parquet else (INPUT, INPUT_AUDIT, AVAIL_AUDIT, FAST_WORKER))
    if not all(path.exists() for path in required):
        raise FileNotFoundError("audited V6-B strict input or anchored input missing")
    outputs = (ARCH, BRACKETS, MONOTONICITY, GRID_AUDIT, RAW_DIR, EVALS, TOLERANCE, BINDING, COVERAGE, AUDIT)
    if any(path.exists() for path in outputs):
        raise FileExistsError("refusing to overwrite V6-E artifacts")
    audit_file = PARQUET_AUDIT if use_parquet else INPUT_AUDIT
    b1 = json.loads(audit_file.read_text(encoding='utf-8'))
    if b1.get('status') != 'PASS' or b1.get('device') != 'arc_16pancake_nuc600_v6_2':
        raise AssertionError('upstream V6.2-B1 strict table audit gate failed')
    if use_parquet:
        if b1.get('storage', {}).get('strict_root') != PARQUET_ROOT.name:
            raise AssertionError('upstream V6.2-B1 Parquet storage contract mismatch')
        source_sha = sha256(PARQUET_AUDIT)
    else:
        source_sha = sha256(INPUT)
        if source_sha != b1['outputs']['strict_feasible']['sha256']:
            raise AssertionError('upstream V6.2-B1 strict table hash gate failed')
    if b1.get('matrix_sha256') != '97242795c8f75319c8f5ea726a1fedd86999e8a66e53ddeb1255e6fb0c659ae5':
        raise AssertionError('V6.2 actual matrix provenance mismatch')
    OUT.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    # Grid-stage architecture optima, using only V6-B strict-feasible implementations.
    best: dict[tuple[str, float, int, float], dict] = {}
    columns = ["scenario", "Top_K", "coolant", "Npw", "rho_turn_uOhm_cm2", "R_joint_nOhm", "Aplant", "CF_gross", "r_cryo_re_fraction", "E_net_year_MWh", "lcoe_anchor_USD_per_MWh"]
    if use_parquet:
        dataset = ds.dataset(str(PARQUET_ROOT), format='parquet', partitioning='hive')
        batches = dataset.scanner(
            columns=columns,
            filter=ds.field('coolant') == 'He',
            batch_size=CHUNK,
        ).to_batches()
        chunks = (batch.to_pandas() for batch in batches)
    else:
        chunks = pd.read_csv(INPUT, usecols=columns, chunksize=CHUNK, low_memory=False)
    for chunk in chunks:
        # Hive partition discovery exposes Top_K as text; normalize before
        # applying the temperature predicate so CSV and Parquet paths agree.
        chunk["Top_K"] = pd.to_numeric(chunk["Top_K"], errors="raise")
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
    grid_audit = {"experiment_id": "V6-E-grid-brackets", "status": "PASS_GRID_BRACKETS_NOT_FINAL_BOUNDARIES", "timestamp_utc": datetime.now(timezone.utc).isoformat(), "source_audit_hash": source_sha, "source_format": "partitioned_parquet" if use_parquet else "csv", "rows": {"architecture_optima": len(arch), "brackets": len(brackets), "monotonicity": len(mono_rows)}, "references": {"temperature": {f"{s}|{t:g}": float(v) for (s, t), v in refs.items()}, "global": {s: float(v) for s, v in global_refs.items()}}, "definition": "strict V6-B candidates; first continuous interval from 1 nOhm; direct bisection required for finite boundaries"}
    GRID_AUDIT.write_text(json.dumps(grid_audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    aplant_requirement = verify_plant_availability_contract()
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
        jobs = [{"scenario": scenario, "Top_K": top, "Npw": n, "R_joint_nOhm": sorted(r), "iteration": iteration} for (scenario, top, n), r in sorted(requested.items())]
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
    pd.DataFrame(evaluations).to_parquet(OUT / 'v6_2_e_direct_bisection_evaluations.parquet', index=False, compression='zstd')

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
    audit = {"experiment_id": "V7-E-cross-scenario-direct-log-bisection", "status": "PASS", "timestamp_utc": datetime.now(timezone.utc).isoformat(), "device": "arc_16pancake_nuc600_v6_2", "scenarios": list(SCENARIOS), "coolant": "He", "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "source": {"v6_2_b1_strict_feasible": {"format": "partitioned_parquet" if use_parquet else "csv", "audit_sha256": source_sha}, "v6_2_a_availability": {"sha256": sha256(AVAIL_AUDIT)}}, "plant_availability_requirement": aplant_requirement, "method": {"criteria": list(CRITERIA), "interval": "first continuous interval from 1 nOhm", "space": "log_Rj", "maximum_observed_midpoint_relative_error": max_error, "target": .001, "rho_optimization": "minimum frozen-B1-anchor LCOE among direct-evaluated strict-feasible rho samples", "temperature_interpolation": "none", "remote_reentry_islands": "excluded"}, "outputs": {path.name: {"path": audit_path(path), "sha256": sha256(path)} for path in (TOLERANCE, BINDING, COVERAGE, OUT / 'v6_2_e_direct_bisection_evaluations.parquet')}, "started_utc": started.isoformat()}
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
