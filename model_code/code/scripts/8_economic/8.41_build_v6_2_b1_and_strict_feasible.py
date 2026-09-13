#!/usr/bin/env python3
"""Build V6.2 frozen-B1 economics and the strict V6.2-B denominator.

The V6.2 production grid intentionally retained only 27 columns.  The omitted
B1 inputs are deterministic design constants, reconstructed here and checked
against ``CAPEX_mag_installed_USD`` before the frozen 8.5 transform constants
are applied.  No circuit, heat-load, annual-energy, or plant calculation is
rerun and no V6.1 data are read.
"""
from __future__ import annotations

import argparse
import csv
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
DEVICE = "arc_16pancake_nuc600_v6_2"
RAW_NAMES = ("v6_2_b_4.2K_He.csv", "v6_2_b_10K_He.csv", "v6_2_b_20K_He.csv", "v6_2_b_20K_H2.csv")
KEYS = ("scenario", "Top_K", "coolant", "Npw", "rho_turn_uOhm_cm2", "R_joint_nOhm")
CHUNK = 200_000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def finite(series: pd.Series) -> pd.Series:
    return np.isfinite(pd.to_numeric(series, errors="coerce").to_numpy())


def import_anchor():
    path = Path(__file__).with_name("8.5_recompute_v2_anchor_economics.py")
    spec = importlib.util.spec_from_file_location("v62_frozen_b1", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module, sha256(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-stage", type=Path, required=True)
    parser.add_argument("--availability", type=Path, required=True)
    parser.add_argument("--output-stage", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=None, help="Smoke-test limit only; formal run omits this.")
    args = parser.parse_args()
    raw_stage, out = args.raw_stage.resolve(), args.output_stage.resolve()
    raw_paths = [raw_stage / name for name in RAW_NAMES]
    if any(not path.exists() for path in raw_paths):
        raise FileNotFoundError("missing V6.2-B raw block")
    protected_outputs = ("v6_2_b1_anchor_economics.csv", "v6_2_b_strict_feasible_designs.csv", "v6_2_b1_and_strict_feasibility_audit.json")
    if out.exists() and any((out / name).exists() for name in protected_outputs):
        raise FileExistsError(f"refusing existing V6.2-B1 data output in: {out}")
    out.mkdir(parents=True, exist_ok=True)
    anchor, anchor_sha = import_anchor()
    os.environ["FUSION_DEVICE"] = DEVICE
    from fusion_tem import device as cfg
    from fusion_tem.economic.lcoe import define_parameters
    from fusion_tem.economic.cost_boundary import FUSION_POWER_MWTH
    from fusion_tem.economic.price_basis import POWER_SUPPLY_PRICE_2025_USD_PER_A
    if cfg.DEVICE != DEVICE:
        raise RuntimeError(f"device mismatch: {cfg.DEVICE}")
    p = define_parameters()
    availability_keys = ["scenario", "Top_K", "Npw", "rho_turn_uOhm_cm2"]
    av = pd.read_csv(args.availability, usecols=availability_keys)
    av["Top_K"] = pd.to_numeric(av["Top_K"], errors="raise").round(9)
    av["rho_turn_uOhm_cm2"] = pd.to_numeric(av["rho_turn_uOhm_cm2"], errors="raise").round(9)
    av["Npw"] = pd.to_numeric(av["Npw"], errors="raise").astype(int)
    if len(av) != 109800 or av.duplicated(availability_keys).any():
        raise AssertionError("V6.2 availability grid is not a unique complete 109800-row map")
    out_b1 = out / "v6_2_b1_anchor_economics.csv"
    out_feasible = out / "v6_2_b_strict_feasible_designs.csv"
    rows = strict_rows = 0
    status_counts: dict[str, int] = {}
    capex_max_residual = 0.0
    matrices: set[str] = set()
    reference_residuals: dict[str, float] = {}
    started = datetime.now(timezone.utc).isoformat()
    for raw in raw_paths:
        for chunk in pd.read_csv(raw, chunksize=CHUNK, low_memory=False):
            if args.max_rows is not None:
                remaining = args.max_rows - rows
                if remaining <= 0:
                    break
                chunk = chunk.iloc[:remaining].copy()
            chunk["Top_K"] = pd.to_numeric(chunk["Top_K"], errors="raise").round(9)
            chunk["rho_turn_uOhm_cm2"] = pd.to_numeric(chunk["rho_turn_uOhm_cm2"], errors="raise").round(9)
            chunk["Npw"] = pd.to_numeric(chunk["Npw"], errors="raise").astype(int)
            chunk = chunk.merge(av, on=availability_keys, how="left", validate="many_to_one")
            if chunk["Aplant"].isna().any():
                raise AssertionError("V7 plant availability is missing after the Stage-A key audit")
            matrices.update(chunk["TF_system_matrix_sha256"].dropna().astype(str).unique())
            top = pd.to_numeric(chunk["Top_K"], errors="raise")
            scenario = chunk["scenario"].astype(str)
            hts_req = top.map({float(k): float(v) for k, v in p["hts_kAm_by_temperature"].items()}).astype(float)
            density = pd.Series([p["coolant_density_kg_m3"][c][float(t)] for c, t in zip(chunk["coolant"], top)], index=chunk.index, dtype=float)
            mass = density * float(p["cryo_loop_volume_m3"])
            raw_hts_price = pd.to_numeric(chunk["HTS_price_2025USD_per_kAm"], errors="coerce")
            raw_fill_price = pd.Series([p["coolant_price_per_kg_by_scenario"][s][c] for s, c in zip(scenario, chunk["coolant"])], index=chunk.index, dtype=float)
            reconstructed_capex = hts_req * raw_hts_price + mass * raw_fill_price + pd.to_numeric(chunk["Power_supply_cost_USD"], errors="coerce")
            residual = (pd.to_numeric(chunk["CAPEX_mag_installed_USD"], errors="coerce") - reconstructed_capex).abs()
            valid_residual = residual[np.isfinite(residual)]
            if len(valid_residual): capex_max_residual = max(capex_max_residual, float(valid_residual.max()))
            target_hts = scenario.map(anchor.HTS_PRICE_BY_SCENARIO).astype(float)
            target_fill_price = chunk["coolant"].map({"He": 175.0, "H2": 10.0}).astype(float)
            ps_current = pd.to_numeric(chunk["Power_supply_cost_USD"], errors="coerce") / float(POWER_SUPPLY_PRICE_2025_USD_PER_A)
            c_hts = hts_req * target_hts
            c_ps = ps_current * float(anchor.POWER_SUPPLY_PRICE_USD_PER_A)
            c_fill = mass * target_fill_price
            ref_hts = scenario.map({s: float(p["hts_kAm_by_temperature"][10.0]) * float(anchor.HTS_PRICE_BY_SCENARIO[s]) for s in ("S1", "S2", "S3")}).astype(float)
            ref_ps = scenario.map({s: float(cfg.Ip_list[10.0]) * 19.0 / float(cfg.carrying_factor) * float(anchor.POWER_SUPPLY_PRICE_USD_PER_A) for s in ("S1", "S2", "S3")}).astype(float)
            ref_mass = float(p["cryo_loop_volume_m3"]) * float(p["coolant_density_kg_m3"]["He"][10.0])
            ref_fill = pd.Series(ref_mass * 175.0, index=chunk.index, dtype=float)
            delta = float(anchor.M_HTS) * (c_hts-ref_hts) + float(anchor.M_PS) * (c_ps-ref_ps) + float(anchor.M_COOLANT_FILL) * (c_fill-ref_fill)
            c_anchor = scenario.map({s: float(anchor.build_capital_anchor(s).full_anchor_usd) for s in ("S1", "S2", "S3")}).astype(float)
            rate = scenario.map(anchor.DISCOUNT_RATE_BY_SCENARIO).astype(float)
            years = scenario.map(anchor.PROJECT_YEARS_BY_SCENARIO).astype(int)
            crf = pd.Series([anchor.capital_recovery_factor(r, y) for r, y in zip(rate, years)], index=chunk.index)
            gross = pd.to_numeric(chunk["E_gross_year_MWh"], errors="coerce")
            net = pd.to_numeric(chunk["E_net_year_MWh"], errors="coerce")
            fusion_th = pd.to_numeric(chunk["CF_gross"], errors="coerce") * float(cfg.HOURS_PER_YEAR) * float(FUSION_POWER_MWTH)
            annual_noncapital = fusion_th * scenario.map(anchor.CORE_VOM_BY_SCENARIO).astype(float) + gross * float(anchor.PCS_VOM_USD_PER_MWH_E) + float(anchor.annual_pcs_fom_usd()) + c_fill * float(anchor.COOLANT_REPLENISH_FRACTION)
            plant = c_anchor + delta
            anchor_valid = np.isfinite(net) & (net > 0.0) & np.isfinite(plant) & np.isfinite(annual_noncapital)
            lcoe = pd.Series(np.nan, index=chunk.index, dtype=float)
            lcoe.loc[anchor_valid] = (crf.loc[anchor_valid] * plant.loc[anchor_valid] + annual_noncapital.loc[anchor_valid]) / net.loc[anchor_valid]
            frame = chunk.loc[:, [*KEYS, "status", "invalid_reason", "TF_system_matrix_file", "TF_system_matrix_sha256", "Aplant", "CF_gross", "r_cryo_re_fraction", "E_gross_year_MWh", "E_net_year_MWh"]].copy()
            frame["HTS_requirement_kA_m"] = hts_req; frame["C_anchor_USD"] = c_anchor; frame["C_plant_anchor_USD"] = plant
            frame["annual_noncapital_cost_USD"] = annual_noncapital; frame["lcoe_anchor_USD_per_MWh"] = lcoe; frame["anchor_economic_valid"] = anchor_valid
            frame["capital_boundary_version"] = anchor.CAPITAL_BOUNDARY_VERSION; frame["reference_magnet_id"] = anchor.REFERENCE_MAGNET_ID
            frame.to_csv(out_b1, mode="a", header=not out_b1.exists(), index=False, float_format="%.17g")
            strict = (frame["status"].astype(str).str.lower().eq("success") & finite(frame["Aplant"]) & (pd.to_numeric(frame["Aplant"], errors="coerce") >= .80) & finite(frame["r_cryo_re_fraction"]) & (pd.to_numeric(frame["r_cryo_re_fraction"], errors="coerce") <= .50) & finite(frame["E_net_year_MWh"]) & (pd.to_numeric(frame["E_net_year_MWh"], errors="coerce") > 0.0) & anchor_valid & finite(lcoe) & (lcoe > 0.0))
            frame.loc[strict].to_csv(out_feasible, mode="a", header=not out_feasible.exists(), index=False, float_format="%.17g")
            rows += len(frame); strict_rows += int(strict.sum())
            for key, count in frame["status"].astype(str).value_counts().items(): status_counts[key] = status_counts.get(key, 0) + int(count)
            if rows % 200_000 == 0: print(f"[V6.2-B1] {rows:,} rows; strict={strict_rows:,}", flush=True)
        if args.max_rows is not None and rows >= args.max_rows: break
    if len(matrices) != 1: raise AssertionError(f"nonunique matrix provenance: {matrices}")
    if args.max_rows is None and rows != 17_714_400: raise AssertionError(f"row count {rows} != 17,714,400")
    audit = {"experiment_id":"V6.2-B1-frozen-anchor-and-strict-feasibility","status":"PASS","device":DEVICE,"rows":rows,"strict_feasible_rows":strict_rows,"status_counts":status_counts,"matrix_sha256":next(iter(matrices)),"capex_reconstruction_max_abs_residual_USD":capex_max_residual,"frozen_transform_script_sha256":anchor_sha,"capital_boundary_version":anchor.CAPITAL_BOUNDARY_VERSION,"outputs":{"b1_anchor":{ "path":out_b1.name,"sha256":sha256(out_b1)},"strict_feasible":{ "path":out_feasible.name,"sha256":sha256(out_feasible)}},"started_utc":started,"completed_utc":datetime.now(timezone.utc).isoformat()}
    (out / "v6_2_b1_and_strict_feasibility_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__": main()
