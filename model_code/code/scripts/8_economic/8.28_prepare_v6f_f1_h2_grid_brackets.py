#!/usr/bin/env python3
"""V6-F1: apply frozen B1 economics and bracket 20 K H2 tolerance limits."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "v6_hts_temperature_tolerance_arc16pancake_nuc600"
RAW = OUT / "v6f_f1_s2_20K_H2_physical_grid.csv"
RAW_AUDIT = OUT / "v6f_f1_h2_grid_audit.json"
ANCHOR = Path(__file__).with_name("8.5_recompute_v2_anchor_economics.py")
V6E_AUDIT = OUT / "v6e_cross_scenario_audit.json"
ARCH = OUT / "v6f_f1_s2_20K_H2_architecture_optima_grid.csv"
BRACKETS = OUT / "v6f_f1_s2_20K_H2_grid_brackets.csv"
AUDIT = OUT / "v6f_f1_h2_grid_bracket_audit.json"
N_VALUES = tuple(range(1, 201))
R_VALUES = np.geomspace(1.0, 100.0, 121)
CHUNK = 50_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def import_anchor():
    spec = importlib.util.spec_from_file_location("v6f_f1_anchor", ANCHOR)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def trace(r: np.ndarray, passed: np.ndarray, criterion: str) -> dict:
    if not bool(passed[0]):
        return {f"Rj_max_{criterion}_grid_lower_nOhm": math.nan, f"Rj_bracket_{criterion}_upper_nOhm": math.nan,
                f"low_reference_infeasible_{criterion}": True, f"right_censored_{criterion}": False,
                f"direct_bisection_required_{criterion}": False, f"binding_condition_{criterion}": "availability" if criterion == "feas" else "condition_relative_LCOE_at_reference"}
    failed = np.flatnonzero(~passed)
    if failed.size == 0:
        return {f"Rj_max_{criterion}_grid_lower_nOhm": 100.0, f"Rj_bracket_{criterion}_upper_nOhm": math.nan,
                f"low_reference_infeasible_{criterion}": False, f"right_censored_{criterion}": True,
                f"direct_bisection_required_{criterion}": False, f"binding_condition_{criterion}": "right_censored_at_100_nOhm"}
    ix = int(failed[0])
    return {f"Rj_max_{criterion}_grid_lower_nOhm": float(r[ix - 1]), f"Rj_bracket_{criterion}_upper_nOhm": float(r[ix]),
            f"low_reference_infeasible_{criterion}": False, f"right_censored_{criterion}": False,
            f"direct_bisection_required_{criterion}": True, f"binding_condition_{criterion}": "availability" if criterion == "feas" else "condition_relative_LCOE"}


def main() -> None:
    if not all(path.exists() for path in (RAW, RAW_AUDIT, V6E_AUDIT)):
        raise FileNotFoundError("V6-F1 H2 raw physical grid and PASS gates are required")
    if any(path.exists() for path in (ARCH, BRACKETS, AUDIT)):
        raise FileExistsError("refusing to overwrite V6-F1 H2 bracket artifacts")
    raw_audit = json.loads(RAW_AUDIT.read_text(encoding="utf-8"))
    v6e = json.loads(V6E_AUDIT.read_text(encoding="utf-8"))
    if raw_audit.get("status") != "PASS" or v6e.get("status") != "PASS" or sha256(RAW) != raw_audit["raw"]["sha256"]:
        raise AssertionError("V6-F1 raw H2 grid is incomplete or upstream audit is not PASS")
    af_max = float(v6e["fixed_af_max_by_scenario"]["S2"])
    anchor = import_anchor()
    e0_manifest = json.loads(anchor.E0_MANIFEST.read_text(encoding="utf-8"))
    _, refs = anchor.find_reference_rows(e0_manifest)
    best: dict[tuple[int, float], dict] = {}
    source_rows = 0
    started = datetime.now(timezone.utc)
    for chunk in pd.read_csv(RAW, chunksize=CHUNK, dtype=str, keep_default_na=False):
        source_rows += len(chunk)
        transformed = anchor.transform_chunk(chunk, refs, e0_manifest)
        af = pd.to_numeric(transformed["AF"], errors="coerce")
        transformed["AF_ref"] = af / af_max
        rcryo = pd.to_numeric(transformed["r_cryo_re_fraction"], errors="coerce")
        net = pd.to_numeric(transformed["net_energy_annual_MWh"], errors="coerce")
        lcoe = pd.to_numeric(transformed["lcoe_anchor_USD_per_MWh"], errors="coerce")
        valid = (transformed["status"].astype(str).str.lower().eq("success") & transformed["anchor_economic_valid"].astype(str).str.lower().isin(("true", "1"))
                 & np.isfinite(transformed["AF_ref"]) & np.isfinite(rcryo) & np.isfinite(net) & np.isfinite(lcoe)
                 & (transformed["AF_ref"] >= 0.99) & (rcryo <= 0.50) & (net > 0.0) & (lcoe > 0.0))
        data = transformed.loc[valid].copy()
        if data.empty:
            continue
        data["lcoe_anchor_USD_per_MWh"] = pd.to_numeric(data["lcoe_anchor_USD_per_MWh"], errors="raise")
        winners = data.loc[data.groupby(["Npw", "R_joint_nOhm"], sort=False)["lcoe_anchor_USD_per_MWh"].idxmin()]
        for row in winners.to_dict("records"):
            key = (int(row["Npw"]), float(row["R_joint_nOhm"]))
            prior = best.get(key)
            if prior is None or float(row["lcoe_anchor_USD_per_MWh"]) < float(prior["lcoe_anchor_USD_per_MWh"]):
                best[key] = row
    records = pd.DataFrame(best.values())
    records["Npw"] = pd.to_numeric(records["Npw"], errors="raise").astype(int)
    records["R_joint_nOhm"] = pd.to_numeric(records["R_joint_nOhm"], errors="raise").astype(float)
    full = pd.DataFrame(product(N_VALUES, R_VALUES), columns=["Npw", "R_joint_nOhm"])
    arch = full.merge(records, on=["Npw", "R_joint_nOhm"], how="left", validate="one_to_one", indicator=True)
    arch["architecture_feasible"] = arch.pop("_merge").eq("both")
    lstar = float(arch.loc[arch["architecture_feasible"], "lcoe_anchor_USD_per_MWh"].min())
    mask = arch["architecture_feasible"]
    arch.loc[mask, "delta_condition_pct"] = 100.0 * (arch.loc[mask, "lcoe_anchor_USD_per_MWh"] - lstar) / lstar
    arch.rename(columns={"rho_turn_uOhm_cm2": "best_rho_turn_uOhm_cm2"}, inplace=True)
    arch.to_csv(ARCH, index=False, encoding="utf-8", float_format="%.17g")
    rows = []
    for npw in N_VALUES:
        frame = arch.loc[arch["Npw"].eq(npw)].sort_values("R_joint_nOhm")
        feasible = frame["architecture_feasible"].to_numpy(bool)
        row = {"scenario": "S2", "Top_K": 20.0, "coolant": "H2", "Npw": npw, "R_ref_nOhm": 1.0,
               "L_star_condition_USD_per_MWh": lstar}
        row.update(trace(R_VALUES, feasible, "feas"))
        row.update(trace(R_VALUES, feasible & (frame["delta_condition_pct"].to_numpy(float) <= 5.0), "condition_5pct"))
        rows.append(row)
    brackets = pd.DataFrame(rows)
    brackets.to_csv(BRACKETS, index=False, encoding="utf-8", float_format="%.17g")
    if source_rows != 200 * 121 * 21 or len(arch) != 200 * 121 or len(brackets) != 200:
        raise AssertionError("unexpected V6-F1 H2 output dimensions")
    audit = {"experiment_id": "V6-F1-S2-20K-H2-grid-brackets", "status": "PASS_GRID_BRACKETS_NOT_FINAL_BOUNDARIES",
             "timestamp_utc": datetime.now(timezone.utc).isoformat(), "device": "arc_16pancake_nuc600", "scenario": "S2", "Top_K": 20.0, "coolant": "H2",
             "source": {"raw": {"path": str(RAW.relative_to(ROOT)), "sha256": sha256(RAW), "rows": source_rows},
                        "v6e_audit": {"path": str(V6E_AUDIT.relative_to(ROOT)), "sha256": sha256(V6E_AUDIT)}},
             "fixed_af_max_S2": af_max, "definition": {"rho_optimization": "minimum frozen-B1-anchor LCOE among strict-feasible rho samples",
                            "conditions": ["physical feasibility", "20K-H2-condition-relative LCOE <= 5%"], "L_star_condition_USD_per_MWh": lstar,
                            "final_boundary": "requires direct-model log-space bisection to <=0.1% relative error"},
             "outputs": {path.name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for path in (ARCH, BRACKETS)}, "started_utc": started.isoformat()}
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()


