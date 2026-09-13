#!/usr/bin/env python3
"""V6-C direct-model log-space bisection of first-continuous Rj brackets."""
from __future__ import annotations

import concurrent.futures
import hashlib
import importlib.util
import json
import math
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "v6_hts_temperature_tolerance_arc16pancake_nuc600"
BRACKETS = OUT / "v6c_s2_joint_tolerance_grid_brackets.csv"
BRACKET_AUDIT = OUT / "v6c_s2_grid_bracket_audit.json"
ANCHOR_INPUT = OUT / "v6_b1_anchor_economics_arc16pancake_nuc600.csv"
ANCHOR_AUDIT = OUT / "v6_b1_anchor_economics_audit.json"
DRIVER = Path(__file__).with_name("8.13_run_v6_arc16pancake_nuc600_grid.py")
ANCHOR = Path(__file__).with_name("8.5_recompute_v2_anchor_economics.py")
RAW_DIR = OUT / "v6c_direct_bisection_raw"
EVALS = OUT / "v6c_s2_direct_bisection_evaluations.csv"
BOUNDARIES = OUT / "v6c_joint_tolerance_boundaries.csv"
SUMMARY = OUT / "v6c_tolerance_factor_summary.csv"
BINDING = OUT / "v6c_binding_constraints.csv"
AUDIT = OUT / "v6c_direct_bisection_audit.json"
RHO = np.array([10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000,5000,10000], dtype=float)
CRITERIA = ("feas", "temp_1pct", "temp_5pct", "temp_10pct", "global_1pct", "global_5pct", "global_10pct")
MAX_ITERATIONS = 50
LOG_HALF_ERROR_TARGET = math.log(1.001)  # midpoint relative uncertainty <= 0.1%
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


def fixed_af_max() -> float:
    ratios: list[np.ndarray] = []
    for chunk in pd.read_csv(ANCHOR_INPUT, usecols=["scenario", "AF", "AF_ref"], chunksize=100_000, low_memory=False):
        part = chunk.loc[chunk["scenario"].eq("S2")]
        af = pd.to_numeric(part["AF"], errors="coerce").to_numpy(float)
        ref = pd.to_numeric(part["AF_ref"], errors="coerce").to_numpy(float)
        ratio = af / ref
        ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
        if len(ratio):
            ratios.append(ratio)
    values = np.concatenate(ratios)
    value = float(np.median(values))
    if not np.allclose(values, value, rtol=0, atol=1e-12):
        raise AssertionError("S2 fixed AF maximum is not unique in the anchored V6 grid")
    return value


def worker(job: dict) -> list[dict]:
    """Directly evaluate all requested Rj midpoints for one (T,N) group."""
    top = float(job["Top_K"])
    npw = int(job["Npw"])
    iteration = int(job["iteration"])
    requested = tuple(float(x) for x in job["R_joint_nOhm"])
    af_max = float(job["af_max"])
    raw = RAW_DIR / f"iter_{iteration:02d}" / f"T{top:g}K_N{npw:03d}.csv"
    raw.parent.mkdir(parents=True, exist_ok=True)
    if raw.exists():
        raise FileExistsError(f"refusing to overwrite direct evaluation {raw}")
    driver = import_path(f"v6c_driver_{os.getpid()}_{iteration}_{npw}", DRIVER)
    scan = driver.load_scan(
        output_csv=raw, npw=np.array([npw]), rjoint=np.array(requested), rho=RHO,
        temperatures=[(top, "He")],
    )
    scan.SCENARIOS = ["S2"]
    scan.main()
    anchor = import_path(f"v6c_anchor_{os.getpid()}", ANCHOR)
    e0_manifest = json.loads(anchor.E0_MANIFEST.read_text(encoding="utf-8"))
    _, refs = anchor.find_reference_rows(e0_manifest)
    transformed = anchor.transform_chunk(pd.read_csv(raw, dtype=str, keep_default_na=False), refs, e0_manifest)
    af = pd.to_numeric(transformed["AF"], errors="coerce")
    transformed["AF_ref"] = af / af_max
    rcryo = pd.to_numeric(transformed["r_cryo_re_fraction"], errors="coerce")
    net = pd.to_numeric(transformed["net_energy_annual_MWh"], errors="coerce")
    lcoe = pd.to_numeric(transformed["lcoe_anchor_USD_per_MWh"], errors="coerce")
    valid = (
        transformed["status"].astype(str).str.lower().eq("success")
        & transformed["anchor_economic_valid"].astype(str).str.lower().isin(("true", "1"))
        & np.isfinite(transformed["AF_ref"]) & np.isfinite(rcryo) & np.isfinite(net) & np.isfinite(lcoe)
        & (transformed["AF_ref"] >= 0.99) & (rcryo <= 0.50) & (net > 0.0) & (lcoe > 0.0)
    )
    records: list[dict] = []
    for r_value, group in transformed.groupby("R_joint_nOhm", sort=True):
        candidates = group.loc[valid.loc[group.index]]
        if candidates.empty:
            records.append({"Top_K": top, "Npw": npw, "R_joint_nOhm": float(r_value), "feasible": False,
                            "best_rho_turn_uOhm_cm2": math.nan, "AF_ref": math.nan, "r_cryo_re_fraction": math.nan,
                            "net_energy_annual_MWh": math.nan, "lcoe_anchor_USD_per_MWh": math.nan,
                            "raw_path": str(raw.relative_to(ROOT)), "raw_sha256": sha256(raw)})
        else:
            best = candidates.loc[pd.to_numeric(candidates["lcoe_anchor_USD_per_MWh"], errors="coerce").idxmin()]
            records.append({"Top_K": top, "Npw": npw, "R_joint_nOhm": float(r_value), "feasible": True,
                            "best_rho_turn_uOhm_cm2": float(best["rho_turn_uOhm_cm2"]), "AF_ref": float(best["AF_ref"]),
                            "r_cryo_re_fraction": float(best["r_cryo_re_fraction"]), "net_energy_annual_MWh": float(best["net_energy_annual_MWh"]),
                            "lcoe_anchor_USD_per_MWh": float(best["lcoe_anchor_USD_per_MWh"]),
                            "raw_path": str(raw.relative_to(ROOT)), "raw_sha256": sha256(raw)})
    return records


def condition(record: dict, criterion: str, temp_ref: float, global_ref: float) -> bool:
    if not record["feasible"]:
        return False
    lcoe = float(record["lcoe_anchor_USD_per_MWh"])
    if criterion == "feas":
        return True
    if criterion.startswith("temp_"):
        pct = float(criterion.split("_")[1].replace("pct", ""))
        return lcoe <= temp_ref * (1.0 + pct / 100.0)
    pct = float(criterion.split("_")[1].replace("pct", ""))
    return lcoe <= global_ref * (1.0 + pct / 100.0)


def main() -> None:
    required = (BRACKETS, BRACKET_AUDIT, ANCHOR_INPUT, ANCHOR_AUDIT)
    if not all(path.exists() for path in required):
        raise FileNotFoundError("V6-C brackets or audited anchor input is missing")
    targets = (EVALS, BOUNDARIES, SUMMARY, BINDING, AUDIT, RAW_DIR)
    if any(path.exists() for path in targets):
        raise FileExistsError("refusing to overwrite V6-C direct-bisection artifacts")
    bracket_audit = json.loads(BRACKET_AUDIT.read_text(encoding="utf-8"))
    anchor_audit = json.loads(ANCHOR_AUDIT.read_text(encoding="utf-8"))
    if bracket_audit.get("status") != "PASS_GRID_BRACKETS_NOT_FINAL_BOUNDARIES" or anchor_audit.get("status") != "PASS":
        raise AssertionError("V6-C bracket or anchor gate is not PASS")
    if sha256(ANCHOR_INPUT) != anchor_audit["output"]["sha256"]:
        raise AssertionError("anchored input hash changed after its PASS audit")
    brackets = pd.read_csv(BRACKETS)
    refs = {float(k): float(v) for k, v in bracket_audit["lcoe_references_USD_per_MWh"].items()}
    global_ref = float(bracket_audit["global_reference_USD_per_MWh"])
    af_max = fixed_af_max()
    started = datetime.now(timezone.utc)

    states: dict[tuple[float, int, str], dict] = {}
    for row in brackets.to_dict("records"):
        for criterion in CRITERIA:
            key = (float(row["Top_K"]), int(row["Npw"]), criterion)
            low_ref = bool(row[f"low_reference_infeasible_{criterion}"])
            censored = bool(row[f"right_censored_{criterion}"])
            if low_ref or censored:
                states[key] = {"mode": "low_ref" if low_ref else "right_censored", "lo": math.nan, "hi": math.nan, "iterations": 0}
            else:
                states[key] = {"mode": "bisection", "lo": float(row[f"Rj_max_{criterion}_grid_lower_nOhm"]), "hi": float(row[f"Rj_bracket_{criterion}_upper_nOhm"]), "iterations": 0}

    evaluations: list[dict] = []
    RAW_DIR.mkdir(parents=True)
    for iteration in range(1, MAX_ITERATIONS + 1):
        candidates: dict[tuple[float, int], set[float]] = {}
        task_mid: dict[tuple[float, int, str], float] = {}
        for key, state in states.items():
            if state["mode"] != "bisection" or 0.5 * math.log(state["hi"] / state["lo"]) <= LOG_HALF_ERROR_TARGET:
                continue
            mid = round(math.sqrt(state["lo"] * state["hi"]), 3)
            if not (state["lo"] < mid < state["hi"]):
                raise AssertionError(f"Rj rounding prevents further bisection for {key}: {state}")
            task_mid[key] = mid
            candidates.setdefault((key[0], key[1]), set()).add(mid)
        if not task_mid:
            break
        jobs = [{"Top_K": top, "Npw": n, "R_joint_nOhm": sorted(values), "iteration": iteration, "af_max": af_max}
                for (top, n), values in sorted(candidates.items())]
        records: dict[tuple[float, int, float], dict] = {}
        with concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS) as pool:
            for result in pool.map(worker, jobs):
                for record in result:
                    records[(record["Top_K"], record["Npw"], record["R_joint_nOhm"])] = record
                    evaluations.append({"iteration": iteration, **record})
        for key, mid in task_mid.items():
            record = records.get((key[0], key[1], mid))
            if record is None:
                raise AssertionError(f"missing direct evaluation for {key} at {mid}")
            state = states[key]
            if condition(record, key[2], refs[key[0]], global_ref):
                state["lo"] = mid
            else:
                state["hi"] = mid
            state["iterations"] += 1
        print(f"[Progress] direct bisection iteration {iteration}: {len(task_mid)} criterion midpoints across {len(jobs)} model batches", flush=True)
    unresolved = [key for key, state in states.items() if state["mode"] == "bisection" and 0.5 * math.log(state["hi"] / state["lo"]) > LOG_HALF_ERROR_TARGET]
    if unresolved:
        raise AssertionError(f"V6-C bisection did not reach 0.1% target: {len(unresolved)} unresolved")

    pd.DataFrame(evaluations).to_csv(EVALS, index=False, encoding="utf-8", float_format="%.17g")
    boundary_rows = []
    binding_rows = []
    for row in brackets.to_dict("records"):
        output = {"scenario": "S2", "Top_K": float(row["Top_K"]), "coolant": "He", "Npw": int(row["Npw"]), "R_ref_nOhm": 1.0,
                  "L_star_temperature_USD_per_MWh": refs[float(row["Top_K"])], "L_star_global_USD_per_MWh": global_ref}
        for criterion in CRITERIA:
            state = states[(float(row["Top_K"]), int(row["Npw"]), criterion)]
            if state["mode"] == "low_ref":
                boundary = math.nan; lower = math.nan; upper = math.nan; low = True; censor = False; error = math.nan
            elif state["mode"] == "right_censored":
                boundary = 100.0; lower = 100.0; upper = math.nan; low = False; censor = True; error = 0.0
            else:
                lower, upper = state["lo"], state["hi"]
                boundary = math.sqrt(lower * upper)
                error = math.exp(0.5 * math.log(upper / lower)) - 1.0
                low = False; censor = False
            output.update({f"Rj_max_{criterion}_nOhm": boundary, f"F_R_{criterion}": boundary,
                           f"Rj_lower_{criterion}_nOhm": lower, f"Rj_upper_{criterion}_nOhm": upper,
                           f"relative_boundary_error_{criterion}": error, f"low_reference_infeasible_{criterion}": low,
                           f"right_censored_{criterion}": censor, f"bisection_iterations_{criterion}": state["iterations"],
                           f"binding_condition_{criterion}": row[f"binding_condition_{criterion}"],
                           f"first_interval_reentry_{criterion}": bool(row[f"first_interval_reentry_{criterion}"])})
            binding_rows.append({"scenario": "S2", "Top_K": float(row["Top_K"]), "coolant": "He", "Npw": int(row["Npw"]),
                                 "criterion": criterion, "binding_condition": row[f"binding_condition_{criterion}"],
                                 "low_reference_infeasible": low, "right_censored": censor,
                                 "first_interval_reentry": bool(row[f"first_interval_reentry_{criterion}"])})
        boundary_rows.append(output)
    boundaries = pd.DataFrame(boundary_rows)
    boundaries.to_csv(BOUNDARIES, index=False, encoding="utf-8", float_format="%.17g")
    binding = pd.DataFrame(binding_rows)
    binding.to_csv(BINDING, index=False, encoding="utf-8")
    summary_rows = []
    for (top, criterion), group in pd.DataFrame(binding_rows).groupby(["Top_K", "criterion"], sort=True):
        values = boundaries.loc[boundaries["Top_K"].eq(top), f"F_R_{criterion}"].dropna().to_numpy(float)
        summary_rows.append({"scenario": "S2", "Top_K": top, "coolant": "He", "criterion": criterion,
                             "N_total": 200, "N_low_reference_infeasible": int(group["low_reference_infeasible"].sum()),
                             "N_right_censored": int(group["right_censored"].sum()), "N_finite_boundary": len(values),
                             "F_R_median": float(np.median(values)) if len(values) else math.nan,
                             "F_R_p10": float(np.quantile(values, .10)) if len(values) else math.nan,
                             "F_R_p90": float(np.quantile(values, .90)) if len(values) else math.nan})
    pd.DataFrame(summary_rows).to_csv(SUMMARY, index=False, encoding="utf-8", float_format="%.17g")
    maximum_error = max(float(v) for key, state in states.items() if state["mode"] == "bisection" for v in [math.exp(0.5 * math.log(state["hi"] / state["lo"])) - 1.0])
    audit = {"experiment_id": "V6-C-direct-log-bisection", "status": "PASS", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
             "device": "arc_16pancake_nuc600", "scenario": "S2", "coolant": "He", "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
             "source": {"grid_brackets": {"path": str(BRACKETS.relative_to(ROOT)), "sha256": sha256(BRACKETS)}, "anchored_grid": {"path": str(ANCHOR_INPUT.relative_to(ROOT)), "sha256": sha256(ANCHOR_INPUT)}},
             "fixed_af_max_S2": af_max, "temperature_references_USD_per_MWh": refs, "global_reference_USD_per_MWh": global_ref,
             "bisection": {"space": "log_Rj", "max_iterations": MAX_ITERATIONS, "maximum_observed_midpoint_relative_error": maximum_error, "target": 0.001, "direct_model": "scan_full_grid.py with FUSION_DEVICE=arc_16pancake_nuc600 and V6-only charge inputs", "rho_optimization": "minimum frozen-B1-anchor LCOE among direct-evaluated strict-feasible rho samples"},
             "outputs": {path.name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for path in (EVALS, BOUNDARIES, SUMMARY, BINDING)}, "started_utc": started.isoformat()}
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
