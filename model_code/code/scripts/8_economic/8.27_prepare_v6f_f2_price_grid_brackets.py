#!/usr/bin/env python3
"""V6-F2 price-only grid screening using the frozen S2 B1 economics.

Price is the sole perturbation: it rescales the direct HTS increment about the
same frozen B1 reference magnet.  Physical fields, all S2 operating terms and
the capital boundary therefore remain unchanged.  This produces brackets only;
the final global-5% limits require direct log-space bisection.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "v6_hts_temperature_tolerance_arc16pancake_nuc600"
INPUT = OUT / "v6_b1_anchor_economics_arc16pancake_nuc600.csv"
INPUT_AUDIT = OUT / "v6_b1_anchor_economics_audit.json"
REFS = ROOT / "outputs" / "target_price_window" / "tables" / "B0_reference_magnet_rows.csv"
ARCH = OUT / "v6f_f2_s2_he_price_architecture_optima_grid.csv"
BRACKETS = OUT / "v6f_f2_s2_he_price_grid_brackets.csv"
AUDIT = OUT / "v6f_f2_price_grid_bracket_audit.json"
PRICES = (100.0, 50.0, 10.0)
TEMPERATURES = (4.2, 10.0, 20.0)
N_VALUES = tuple(range(1, 201))
R_VALUES = np.geomspace(1.0, 100.0, 121)
M_HTS = 2.8436625
CHUNK = 100_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def trace_first_interval(r: np.ndarray, passed: np.ndarray) -> dict:
    if not bool(passed[0]):
        return {"Rj_max_global_5pct_grid_lower_nOhm": math.nan, "Rj_bracket_global_5pct_upper_nOhm": math.nan,
                "low_reference_infeasible_global_5pct": True, "right_censored_global_5pct": False,
                "direct_bisection_required_global_5pct": False, "binding_condition_global_5pct": "global_relative_LCOE_at_reference"}
    failed = np.flatnonzero(~passed)
    if failed.size == 0:
        return {"Rj_max_global_5pct_grid_lower_nOhm": 100.0, "Rj_bracket_global_5pct_upper_nOhm": math.nan,
                "low_reference_infeasible_global_5pct": False, "right_censored_global_5pct": True,
                "direct_bisection_required_global_5pct": False, "binding_condition_global_5pct": "right_censored_at_100_nOhm"}
    ix = int(failed[0])
    return {"Rj_max_global_5pct_grid_lower_nOhm": float(r[ix - 1]), "Rj_bracket_global_5pct_upper_nOhm": float(r[ix]),
            "low_reference_infeasible_global_5pct": False, "right_censored_global_5pct": False,
            "direct_bisection_required_global_5pct": True, "binding_condition_global_5pct": "global_relative_LCOE"}


def main() -> None:
    required = (INPUT, INPUT_AUDIT, REFS)
    if not all(path.exists() for path in required):
        raise FileNotFoundError("V6-F2 requires the audited V6 B1 table and B1 reference rows")
    if any(path.exists() for path in (ARCH, BRACKETS, AUDIT)):
        raise FileExistsError("refusing to overwrite V6-F2 price-grid artifacts")
    upstream = json.loads(INPUT_AUDIT.read_text(encoding="utf-8"))
    if upstream.get("status") != "PASS" or sha256(INPUT) != upstream["output"]["sha256"]:
        raise AssertionError("V6 B1 anchored table does not match its PASS audit")
    reference = pd.read_csv(REFS)
    reference = reference.loc[reference["scenario"].eq("S2")]
    if len(reference) != 1:
        raise AssertionError("frozen B1 S2 reference magnet is not unique")
    q_ref = float(reference.iloc[0]["HTS_requirement_kA_m"])
    started = datetime.now(timezone.utc)
    columns = ["Top_K", "coolant", "scenario", "Npw", "rho_turn_uOhm_cm2", "R_joint_nOhm", "status", "AF_ref",
               "r_cryo_re_fraction", "net_energy_annual_MWh", "lcoe_anchor_USD_per_MWh", "HTS_requirement_kA_m",
               "C_plant_anchor_USD", "annual_noncapital_cost_USD", "CRF", "anchor_economic_valid"]
    best: dict[tuple[float, float, int, float], dict] = {}
    source_rows = 0
    for chunk in pd.read_csv(INPUT, usecols=columns, chunksize=CHUNK, low_memory=False):
        data = chunk.loc[chunk["scenario"].eq("S2") & chunk["coolant"].eq("He") & chunk["Top_K"].isin(TEMPERATURES)].copy()
        source_rows += len(data)
        if data.empty:
            continue
        for name in ("AF_ref", "r_cryo_re_fraction", "net_energy_annual_MWh", "lcoe_anchor_USD_per_MWh", "HTS_requirement_kA_m", "C_plant_anchor_USD", "annual_noncapital_cost_USD", "CRF"):
            data[name] = pd.to_numeric(data[name], errors="coerce")
        strict = (data["status"].astype(str).str.lower().eq("success") & data["anchor_economic_valid"].astype(str).str.lower().isin(("true", "1"))
                  & np.isfinite(data["AF_ref"]) & np.isfinite(data["r_cryo_re_fraction"]) & np.isfinite(data["net_energy_annual_MWh"])
                  & np.isfinite(data["lcoe_anchor_USD_per_MWh"]) & (data["AF_ref"] >= 0.99) & (data["r_cryo_re_fraction"] <= 0.50)
                  & (data["net_energy_annual_MWh"] > 0.0) & (data["lcoe_anchor_USD_per_MWh"] > 0.0))
        data = data.loc[strict]
        for price in PRICES:
            frame = data.copy()
            if price == 50.0:
                frame["lcoe_price_USD_per_MWh"] = frame["lcoe_anchor_USD_per_MWh"]
            else:
                plant = frame["C_plant_anchor_USD"] + M_HTS * (price - 50.0) * (frame["HTS_requirement_kA_m"] - q_ref)
                frame["lcoe_price_USD_per_MWh"] = (frame["CRF"] * plant + frame["annual_noncapital_cost_USD"]) / frame["net_energy_annual_MWh"]
            frame = frame.loc[np.isfinite(frame["lcoe_price_USD_per_MWh"]) & (frame["lcoe_price_USD_per_MWh"] > 0.0)]
            if frame.empty:
                continue
            winners = frame.loc[frame.groupby(["Top_K", "Npw", "R_joint_nOhm"], sort=False)["lcoe_price_USD_per_MWh"].idxmin()]
            for row in winners.to_dict("records"):
                key = (price, float(row["Top_K"]), int(row["Npw"]), float(row["R_joint_nOhm"]))
                prior = best.get(key)
                if prior is None or float(row["lcoe_price_USD_per_MWh"]) < float(prior["lcoe_price_USD_per_MWh"]):
                    best[key] = {"HTS_price_USD_per_kAm": price, **row}
    records = pd.DataFrame(best.values())
    if records.empty:
        raise AssertionError("no V6-F2 strict-feasible S2 He architectures")
    full = pd.DataFrame(product(PRICES, TEMPERATURES, N_VALUES, R_VALUES), columns=["HTS_price_USD_per_kAm", "Top_K", "Npw", "R_joint_nOhm"])
    arch = full.merge(records, on=["HTS_price_USD_per_kAm", "Top_K", "Npw", "R_joint_nOhm"], how="left", validate="one_to_one", indicator=True)
    arch["architecture_feasible"] = arch.pop("_merge").eq("both")
    refs = arch.loc[arch["architecture_feasible"]].groupby(["HTS_price_USD_per_kAm", "Top_K"], sort=True)["lcoe_price_USD_per_MWh"].min().to_dict()
    global_refs = arch.loc[arch["architecture_feasible"]].groupby("HTS_price_USD_per_kAm", sort=True)["lcoe_price_USD_per_MWh"].min().to_dict()
    if len(refs) != 9 or set(global_refs) != set(PRICES):
        raise AssertionError("V6-F2 price/temperature reference minima are incomplete")
    for price in PRICES:
        mask = arch["HTS_price_USD_per_kAm"].eq(price) & arch["architecture_feasible"]
        arch.loc[mask, "delta_global_pct"] = 100.0 * (arch.loc[mask, "lcoe_price_USD_per_MWh"] - global_refs[price]) / global_refs[price]
    arch.rename(columns={"rho_turn_uOhm_cm2": "best_rho_turn_uOhm_cm2"}, inplace=True)
    arch.to_csv(ARCH, index=False, encoding="utf-8", float_format="%.17g")
    rows = []
    for price, top, npw in product(PRICES, TEMPERATURES, N_VALUES):
        frame = arch.loc[(arch["HTS_price_USD_per_kAm"].eq(price)) & (arch["Top_K"].eq(top)) & (arch["Npw"].eq(npw))].sort_values("R_joint_nOhm")
        passed = frame["architecture_feasible"].to_numpy(bool) & (frame["delta_global_pct"].to_numpy(float) <= 5.0)
        rows.append({"scenario": "S2", "coolant": "He", "HTS_price_USD_per_kAm": price, "Top_K": top, "Npw": npw,
                     "R_ref_nOhm": 1.0, "L_star_global_USD_per_MWh": global_refs[price], **trace_first_interval(frame["R_joint_nOhm"].to_numpy(float), passed)})
    brackets = pd.DataFrame(rows)
    brackets.to_csv(BRACKETS, index=False, encoding="utf-8", float_format="%.17g")
    p50 = arch.loc[arch["architecture_feasible"] & arch["HTS_price_USD_per_kAm"].eq(50.0)]
    p50_residual = (p50["lcoe_price_USD_per_MWh"] - p50["lcoe_anchor_USD_per_MWh"]).abs()
    expected = len(PRICES) * len(TEMPERATURES) * len(N_VALUES) * len(R_VALUES)
    if len(arch) != expected or len(brackets) != len(PRICES) * len(TEMPERATURES) * len(N_VALUES) or float(p50_residual.max()) > 1e-12:
        raise AssertionError("V6-F2 price grid dimensions or p=50 identity check failed")
    audit = {"experiment_id": "V6-F2-S2-He-HTS-price-grid-brackets", "status": "PASS_GRID_BRACKETS_NOT_FINAL_BOUNDARIES",
             "timestamp_utc": datetime.now(timezone.utc).isoformat(), "device": "arc_16pancake_nuc600", "scenario": "S2", "coolant": "He",
             "prices_constant_2025_USD_per_kAm": list(PRICES), "temperatures_K": list(TEMPERATURES), "reference_magnet": "arc16_T10K_He_Npw19_rhoturn10000_Rj1nOhm",
             "method": {"price_perturbation": "Cplant(p)=Cplant(p=50)+m_HTS*(p-50)*(Q_HTS-Q_HTS_ref)", "m_HTS": M_HTS,
                        "all_other_S2_terms": "frozen", "rho_optimization": "minimum price-recomputed LCOE among strict-feasible rho samples",
                        "final_boundary": "requires direct-model log-space bisection to <=0.1% relative error"},
             "source": {"v6_b1_anchor": {"path": str(INPUT.relative_to(ROOT)), "sha256": sha256(INPUT), "S2_He_rows_read": source_rows},
                        "reference_rows": {"path": str(REFS.relative_to(ROOT)), "sha256": sha256(REFS), "Q_HTS_ref_kA_m": q_ref}},
             "p50_identity_max_abs_USD_per_MWh": float(p50_residual.max()), "global_references_USD_per_MWh": {str(k): float(v) for k, v in global_refs.items()},
             "outputs": {path.name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for path in (ARCH, BRACKETS)}, "started_utc": started.isoformat()}
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
