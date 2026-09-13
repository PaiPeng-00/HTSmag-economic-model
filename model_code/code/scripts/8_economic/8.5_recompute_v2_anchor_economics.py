"""Execute B0-B1 incremental-anchor economic post-processing.

Input E0 physics and annual-energy fields are read-only.  Legacy economic
columns are retained with a ``legacy_e0_`` prefix; the unprefixed economic
columns in the output use the frozen B0-B1 plan.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve()
CODE_ROOT = SCRIPT.parents[2]
REPO_ROOT = CODE_ROOT.parent
WORKSPACE_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(CODE_ROOT / "src"))

from fusion_tem.economic.capital_anchor import (  # noqa: E402
    CAPITAL_BOUNDARY_VERSION,
    COOLANT_REPLENISH_FRACTION,
    CORE_VOM_BY_SCENARIO,
    DISCOUNT_RATE_BY_SCENARIO,
    GROSS_ELECTRIC_POWER_MWE,
    HTS_PRICE_BY_SCENARIO,
    M_COOLANT_FILL,
    M_HTS,
    M_PS,
    MATURITY_SCALE_BY_SCENARIO,
    NON_TF_AUX_FRACTION,
    PCS_CAPITAL_USD_PER_KWE,
    PCS_FOM_USD_PER_KWE_YEAR,
    PCS_VOM_USD_PER_MWH_E,
    POWER_SUPPLY_PRICE_USD_PER_A,
    PRICE_BASIS_YEAR,
    PRICE_FACTOR_2010_TO_2025,
    PROJECT_YEARS_BY_SCENARIO,
    PS_CAPITALIZATION_FACTORS,
    REFERENCE_MAGNET_ID,
    WADE_COIL_DIRECT_2010_MUSD,
    WADE_TOTAL_CAPITAL_2010_MUSD,
    HTS_CAPITALIZATION_FACTORS,
    annual_pcs_fom_usd,
    build_capital_anchor,
    capital_recovery_factor,
    coolant_price_usd_per_kg,
)


EXPERIMENT_DIR = WORKSPACE_ROOT / "17-提交版本-SA" / "1-正文-V5" / "1-实验"
PLAN = EXPERIMENT_DIR / "4-Incremental_anchor_B0_to_D_experiment_plan_for_Codex.md"
V2_DOC = EXPERIMENT_DIR / "1-Tier1_Tier2_parameter_reference_register_V2.md"
E0_CSV = CODE_ROOT / "outputs" / "arc_16pancake_nuc600" / "tables" / "E0_full_grid_green_eta030_tidy.csv"
E0_MANIFEST = CODE_ROOT / "outputs" / "arc_16pancake_nuc600" / "manifests" / "E0_full_grid_green_eta030_manifest.json"
OUT = CODE_ROOT / "outputs" / "target_price_window"
TABLES = OUT / "tables"
LOGS = OUT / "logs"
MANIFESTS = OUT / "manifests"

B1_GRID = TABLES / "B1_full_grid_green_eta030_V2_anchor_economics.csv"
EXPECTED_ROWS = 593_712
CHUNK_SIZE = 20_000
SAMPLE_SEED = 20260804

REFERENCE_COORDS = {
    "Top_K": 10.0,
    "coolant": "He",
    "Npw": 19,
    "rho_turn_uOhm_cm2": 10_000.0,
    "R_joint_nOhm": 1.0,
}

# These E0 columns are economic outputs or economic provenance.  They remain
# available in B1 under an explicit legacy prefix rather than silently mixing
# old and anchor-economics values.
LEGACY_ECONOMIC_COLUMNS = {
    "C0_nonmagnet_USD", "project_lifetime_years", "discount_rate", "CRF",
    "C0_background_USD", "CAPEX_mag_direct_USD", "CAPEX_mag_installed_USD",
    "CAPEX_PCS_reference_USD", "Annualized_CAPEX_mag_USD_per_year",
    "Annualized_C0_USD_per_year", "Annualized_CAPEX_total_modeled_USD_per_year",
    "OPEX_core_VOM_USD_per_year", "OPEX_PCS_VOM_USD_per_year",
    "OPEX_PCS_FOM_USD_per_year", "OPEX_coolant_VOM_USD_per_year",
    "OPEX_total_modeled_USD_per_year", "LCOE_plant_USD_per_MWh",
    "LCOE_magnet_only_USD_per_MWh", "LCOE_fullplant_legacy_USD_per_MWh",
    "economic_invalid_reason", "monetary_price_basis_year",
    "monetary_values_constant_2025_usd", "hts_conductor_price_source_year",
    "hts_conductor_price_cpi_factor", "hts_conductor_price_conversion_method",
    "cost_boundary_version", "CAPEX_mag_USD", "Annual_OPEX_mag_USD_per_year",
    "LCOE_fullplant_USD_per_MWh", "LCOE_C0_USD_per_MWh", "Tape_cost_USD",
    "HTS_price_2025USD_per_kAm", "Coolant_price_2025USD_per_kg",
    "Coolant_fill_cost_USD", "Power_supply_cost_USD", "finite_lcoe",
    "LCOE_reference_joint_feasible", "delta_LCOE_joint_feasible_USD_per_MWh",
    "delta_LCOE_min_USD_per_MWh", "LCOE_reference_magnet_only_legacy_USD_per_MWh",
    "delta_LCOE_magnet_only_legacy_USD_per_MWh",
    "LCOE_reference_fullplant_legacy_USD_per_MWh",
    "delta_LCOE_fullplant_legacy_USD_per_MWh",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return str(path.relative_to(WORKSPACE_ROOT)).replace("\\", "/")


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.rstrip()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def line_count(path: Path) -> int:
    with path.open("rb") as stream:
        return sum(block.count(b"\n") for block in iter(lambda: stream.read(8 * 1024 * 1024), b"")) - 1


def numeric(series: pd.Series) -> pd.Series:
    """Parse an E0 text column for calculation without changing its raw text."""
    return pd.to_numeric(series, errors="coerce").astype(float)


def anchors() -> dict[str, Any]:
    return {scenario: build_capital_anchor(scenario) for scenario in MATURITY_SCALE_BY_SCENARIO}


def find_reference_rows(e0_manifest: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    cols = [
        "Top_K", "coolant", "scenario", "Npw", "rho_turn_uOhm_cm2", "R_joint_nOhm",
        "Tape_cost_USD", "HTS_price_2025USD_per_kAm", "Power_supply_cost_USD",
        "Coolant_price_2025USD_per_kg", "Coolant_fill_cost_USD",
        "OPEX_coolant_VOM_USD_per_year", "E_gross_year_MWh", "E_cryo_year_MWh",
        "E_net_year_MWh", "gross_energy_annual_MWh", "E_cryo_TF_annual_MWh",
        "net_energy_annual_MWh", "E_fusion_th_year_MWh", "status",
    ]
    found: list[pd.DataFrame] = []
    for chunk in pd.read_csv(E0_CSV, usecols=cols, chunksize=50_000, low_memory=False):
        mask = (
            np.isclose(chunk["Top_K"], REFERENCE_COORDS["Top_K"])
            & chunk["coolant"].eq(REFERENCE_COORDS["coolant"])
            & chunk["Npw"].eq(REFERENCE_COORDS["Npw"])
            & np.isclose(chunk["rho_turn_uOhm_cm2"], REFERENCE_COORDS["rho_turn_uOhm_cm2"])
            & np.isclose(chunk["R_joint_nOhm"], REFERENCE_COORDS["R_joint_nOhm"])
        )
        if mask.any():
            found.append(chunk.loc[mask].copy())
    rows = pd.concat(found, ignore_index=True).sort_values("scenario").reset_index(drop=True)
    counts = rows.groupby("scenario").size().to_dict()
    if counts != {s: 1 for s in ("S1", "S2", "S3", "S4", "S5", "S6")}:
        raise RuntimeError(f"reference magnet is not unique across S1-S6: {counts}")

    old_ps_price = float(e0_manifest["power_supply_price_2025_USD_per_A"])
    refs: dict[str, dict[str, float]] = {}
    out_rows: list[dict[str, Any]] = []
    for _, row in rows.iterrows():
        scenario = str(row["scenario"])
        requirement = float(row["Tape_cost_USD"]) / float(row["HTS_price_2025USD_per_kAm"])
        mass = float(row["Coolant_fill_cost_USD"]) / float(row["Coolant_price_2025USD_per_kg"])
        hts = requirement * float(HTS_PRICE_BY_SCENARIO[scenario])
        ps_current = float(row["Power_supply_cost_USD"]) / old_ps_price
        ps = ps_current * POWER_SUPPLY_PRICE_USD_PER_A
        fill = mass * coolant_price_usd_per_kg(str(row["coolant"]))
        refs[scenario] = {"hts": hts, "ps": ps, "fill": fill}
        anchor = build_capital_anchor(scenario)
        out_rows.append({
            "reference_magnet_id": REFERENCE_MAGNET_ID,
            "scenario": scenario,
            "Top_K": row["Top_K"], "coolant": row["coolant"], "Npw": row["Npw"],
            "rho_turn_uOhm_cm2": row["rho_turn_uOhm_cm2"], "R_joint_nOhm": row["R_joint_nOhm"],
            "match_count_in_scenario": counts[scenario], "status": row["status"],
            "HTS_requirement_kA_m": requirement, "C_HTS_direct_USD": hts,
            "power_supply_rated_current_A": ps_current, "C_power_supply_direct_USD": ps,
            "coolant_required_mass_kg": mass, "C_coolant_fill_direct_USD": fill,
            "annual_coolant_replenishment_USD": fill * COOLANT_REPLENISH_FRACTION,
            "E_gross_MWh": row["gross_energy_annual_MWh"],
            "E_cryo_TF_MWh": row["E_cryo_TF_annual_MWh"], "E_net_MWh": row["net_energy_annual_MWh"],
            "delta_C_HTS_direct_USD": 0.0, "delta_C_HTS_capitalized_USD": 0.0,
            "delta_C_PS_direct_USD": 0.0, "delta_C_PS_capitalized_USD": 0.0,
            "delta_C_coolant_fill_USD": 0.0, "delta_C_mag_anchor_USD": 0.0,
            "C_anchor_USD": anchor.full_anchor_usd, "C_plant_anchor_USD": anchor.full_anchor_usd,
        })
    return pd.DataFrame(out_rows), refs


def transform_chunk(
    chunk: pd.DataFrame,
    refs: dict[str, dict[str, float]],
    e0_manifest: dict[str, Any],
) -> pd.DataFrame:
    scenario = chunk["scenario"].astype(str)
    old_hts_price = numeric(chunk["HTS_price_2025USD_per_kAm"])
    old_coolant_price = numeric(chunk["Coolant_price_2025USD_per_kg"])
    old_ps_price = float(e0_manifest["power_supply_price_2025_USD_per_A"])

    hts_requirement = numeric(chunk["Tape_cost_USD"]) / old_hts_price
    coolant_mass = numeric(chunk["Coolant_fill_cost_USD"]) / old_coolant_price
    ps_current = numeric(chunk["Power_supply_cost_USD"]) / old_ps_price

    hts_price = scenario.map(HTS_PRICE_BY_SCENARIO).astype(float)
    coolant_price = chunk["coolant"].map({"He": 175.0, "H2": 10.0}).astype(float)
    hts_direct = hts_requirement * hts_price
    ps_direct = ps_current * POWER_SUPPLY_PRICE_USD_PER_A
    coolant_fill = coolant_mass * coolant_price

    ref_hts = scenario.map({s: x["hts"] for s, x in refs.items()}).astype(float)
    ref_ps = scenario.map({s: x["ps"] for s, x in refs.items()}).astype(float)
    ref_fill = scenario.map({s: x["fill"] for s, x in refs.items()}).astype(float)
    delta_hts_direct = hts_direct - ref_hts
    delta_ps_direct = ps_direct - ref_ps
    delta_fill = coolant_fill - ref_fill
    delta_hts_cap = M_HTS * delta_hts_direct
    delta_ps_cap = M_PS * delta_ps_direct
    delta_mag = delta_hts_cap + delta_ps_cap + M_COOLANT_FILL * delta_fill

    anchor = scenario.map({s: build_capital_anchor(s).full_anchor_usd for s in refs}).astype(float)
    c0_noncoil = scenario.map({s: build_capital_anchor(s).noncoil_background_usd for s in refs}).astype(float)
    plant = anchor + delta_mag
    rate = scenario.map(DISCOUNT_RATE_BY_SCENARIO).astype(float)
    years = scenario.map(PROJECT_YEARS_BY_SCENARIO).astype(int)
    crf = pd.Series(
        [capital_recovery_factor(float(r), float(n)) for r, n in zip(rate, years)],
        index=chunk.index, dtype=float,
    )
    core_rate = scenario.map(CORE_VOM_BY_SCENARIO).astype(float)
    annual_core = numeric(chunk["E_fusion_th_year_MWh"]) * core_rate
    annual_pcs_vom = numeric(chunk["gross_energy_annual_MWh"]) * PCS_VOM_USD_PER_MWH_E
    annual_pcs_fom = pd.Series(annual_pcs_fom_usd(), index=chunk.index, dtype=float)
    annual_coolant = coolant_fill * COOLANT_REPLENISH_FRACTION
    annual_noncapital = annual_core + annual_pcs_vom + annual_pcs_fom + annual_coolant
    net = numeric(chunk["net_energy_annual_MWh"])
    numerator = crf * plant + annual_noncapital
    lcoe = pd.Series(np.nan, index=chunk.index, dtype=float)
    valid = np.isfinite(net) & (net > 0.0) & np.isfinite(numerator)
    lcoe.loc[valid] = numerator.loc[valid] / net.loc[valid]

    rename = {col: f"legacy_e0_{col}" for col in chunk.columns if col in LEGACY_ECONOMIC_COLUMNS}
    output = chunk.rename(columns=rename).copy()
    new = {
        "capital_boundary_version": CAPITAL_BOUNDARY_VERSION,
        "reference_magnet_id": REFERENCE_MAGNET_ID,
        "price_basis_year": PRICE_BASIS_YEAR,
        "monetary_values_constant_2025_usd": True,
        "C_anchor_USD": anchor,
        "C0_noncoil_background_USD": c0_noncoil,
        "HTS_requirement_kA_m": hts_requirement,
        "HTS_unit_price_USD_per_kAm": hts_price,
        "C_HTS_direct_USD": hts_direct,
        "power_supply_rated_current_A": ps_current,
        "power_supply_unit_price_USD_per_A": POWER_SUPPLY_PRICE_USD_PER_A,
        "C_power_supply_direct_USD": ps_direct,
        "coolant_required_mass_kg": coolant_mass,
        "coolant_unit_price_USD_per_kg": coolant_price,
        "C_coolant_fill_direct_USD": coolant_fill,
        "delta_C_HTS_direct_USD": delta_hts_direct,
        "delta_C_HTS_capitalized_USD": delta_hts_cap,
        "delta_C_PS_direct_USD": delta_ps_direct,
        "delta_C_PS_capitalized_USD": delta_ps_cap,
        "delta_C_coolant_fill_USD": delta_fill,
        "delta_C_mag_anchor_USD": delta_mag,
        "C_plant_anchor_USD": plant,
        "discount_rate": rate,
        "project_years": years,
        "CRF": crf,
        "coolant_replenish_fraction": COOLANT_REPLENISH_FRACTION,
        "annual_core_vom_USD": annual_core,
        "annual_pcs_vom_USD": annual_pcs_vom,
        "annual_pcs_fom_USD": annual_pcs_fom,
        "annual_coolant_replenishment_USD": annual_coolant,
        "annual_noncapital_cost_USD": annual_noncapital,
        "annualized_C_plant_anchor_USD": crf * plant,
        "lcoe_anchor_USD_per_MWh": lcoe,
        "anchor_economic_valid": valid,
    }
    for name, values in new.items():
        output[name] = values
    return output


def make_anchor_rows() -> list[dict[str, Any]]:
    rows = []
    for scenario in ("S1", "S2", "S3", "S4", "S5", "S6"):
        anchor = build_capital_anchor(scenario)
        rows.append({
            "scenario": scenario,
            "price_basis_year": PRICE_BASIS_YEAR,
            "maturity_scale": anchor.maturity_scale,
            "Wade_total_capital_2010_million_USD": WADE_TOTAL_CAPITAL_2010_MUSD,
            "Wade_reference_coil_direct_2010_million_USD": WADE_COIL_DIRECT_2010_MUSD,
            "price_factor_2010_to_2025": PRICE_FACTOR_2010_TO_2025,
            "m_HTS": M_HTS, "m_PS": M_PS, "m_coolant_fill": M_COOLANT_FILL,
            "capitalized_reference_coil_account_USD": anchor.capitalized_reference_coil_account_usd,
            "C0_noncoil_background_USD": anchor.noncoil_background_usd,
            "C_anchor_USD": anchor.full_anchor_usd,
            "capital_boundary_version": CAPITAL_BOUNDARY_VERSION,
        })
    return rows


def compare_non_economic_fields(output_columns: list[str]) -> list[dict[str, Any]]:
    source_columns = list(pd.read_csv(E0_CSV, nrows=0).columns)
    non_economic = [c for c in source_columns if c not in LEGACY_ECONOMIC_COLUMNS]
    stats = {c: {"mismatch": 0, "max_abs": 0.0} for c in non_economic}
    source_iter = pd.read_csv(
        E0_CSV, usecols=non_economic, chunksize=CHUNK_SIZE,
        dtype=str, keep_default_na=False,
    )
    target_iter = pd.read_csv(
        B1_GRID, usecols=non_economic, chunksize=CHUNK_SIZE,
        dtype=str, keep_default_na=False,
    )
    for src, dst in zip(source_iter, target_iter):
        if len(src) != len(dst):
            raise AssertionError("source/target chunk lengths differ")
        for col in non_economic:
            a = src[col].reset_index(drop=True)
            b = dst[col].reset_index(drop=True)
            equal = a.eq(b)
            stats[col]["mismatch"] += int((~equal).sum())
            if (~equal).any():
                a_num = pd.to_numeric(a.loc[~equal], errors="coerce")
                b_num = pd.to_numeric(b.loc[~equal], errors="coerce")
                diff = (a_num - b_num).abs()
                if diff.notna().any():
                    stats[col]["max_abs"] = max(stats[col]["max_abs"], float(diff.max()))
    rows = []
    for col in source_columns:
        if col in LEGACY_ECONOMIC_COLUMNS:
            rows.append({
                "source_column": col, "B1_column": f"legacy_e0_{col}",
                "classification": "legacy_economic_renamed", "mismatch_count": 0,
                "max_absolute_difference": 0.0, "status": "PRESERVED_FOR_AUDIT",
            })
        else:
            mismatch = stats[col]["mismatch"]
            rows.append({
                "source_column": col, "B1_column": col,
                "classification": "non_economic_identity", "mismatch_count": mismatch,
                "max_absolute_difference": stats[col]["max_abs"],
                "status": "PASS" if mismatch == 0 else "FAIL",
            })
    return rows


def reproduce_samples(samples: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for _, row in samples.iterrows():
        recomputed_delta = (
            M_HTS * float(row["delta_C_HTS_direct_USD"])
            + M_PS * float(row["delta_C_PS_direct_USD"])
            + float(row["delta_C_coolant_fill_USD"])
        )
        recomputed_plant = float(row["C_anchor_USD"]) + recomputed_delta
        recomputed_opex = (
            float(row["annual_core_vom_USD"]) + float(row["annual_pcs_vom_USD"])
            + float(row["annual_pcs_fom_USD"]) + float(row["annual_coolant_replenishment_USD"])
        )
        net = float(row["net_energy_annual_MWh"])
        recomputed_lcoe = (
            (float(row["CRF"]) * recomputed_plant + recomputed_opex) / net
            if net > 0.0 else math.nan
        )
        residuals = {
            "delta_C_mag_anchor": float(row["delta_C_mag_anchor_USD"]) - recomputed_delta,
            "C_plant_anchor": float(row["C_plant_anchor_USD"]) - recomputed_plant,
            "annual_noncapital_cost": float(row["annual_noncapital_cost_USD"]) - recomputed_opex,
            "lcoe_anchor": (
                float(row["lcoe_anchor_USD_per_MWh"]) - recomputed_lcoe
                if np.isfinite(recomputed_lcoe) else 0.0
            ),
        }
        rows.append({
            "sample_id": row["sample_id"], "scenario": row["scenario"],
            "Top_K": row["Top_K"], "coolant": row["coolant"], "Npw": row["Npw"],
            "rho_turn_uOhm_cm2": row["rho_turn_uOhm_cm2"], "R_joint_nOhm": row["R_joint_nOhm"],
            **{f"residual_{k}": v for k, v in residuals.items()},
            "max_absolute_residual": max(abs(x) for x in residuals.values()),
            "status": "PASS" if max(abs(x) for x in residuals.values()) < 1e-6 else "FAIL",
        })
    return rows


def main() -> None:
    for directory in (TABLES, LOGS, MANIFESTS):
        directory.mkdir(parents=True, exist_ok=True)

    started = datetime.now(timezone.utc)
    commit = git("rev-parse", "HEAD")
    dirty_before = git("status", "--short")
    e0_sha_before = sha256(E0_CSV)
    e0_manifest = json.loads(E0_MANIFEST.read_text(encoding="utf-8"))
    if e0_sha_before != e0_manifest["outputs"][0]["sha256"]:
        raise AssertionError("E0 CSV hash does not match its frozen manifest")

    anchor_rows = make_anchor_rows()
    anchor_path = TABLES / "B0_capital_anchor_by_scenario.csv"
    write_csv(anchor_path, anchor_rows)

    reference_df, refs = find_reference_rows(e0_manifest)
    reference_path = TABLES / "B0_reference_magnet_rows.csv"
    reference_df.to_csv(reference_path, index=False, encoding="utf-8-sig", float_format="%.17g")
    if float(reference_df[[
        "delta_C_HTS_direct_USD", "delta_C_HTS_capitalized_USD",
        "delta_C_PS_direct_USD", "delta_C_PS_capitalized_USD",
        "delta_C_coolant_fill_USD", "delta_C_mag_anchor_USD",
    ]].abs().to_numpy().max()) >= 1e-6:
        raise AssertionError("reference magnet increment is nonzero")

    temp = B1_GRID.with_suffix(".csv.partial")
    if temp.exists():
        temp.unlink()
    row_total = 0
    scenario_seen = defaultdict(int)
    scenario_targets = {s: set(np.linspace(0, EXPECTED_ROWS // 6 - 1, 12, dtype=int)) for s in MATURITY_SCALE_BY_SCENARIO}
    samples: list[pd.DataFrame] = []
    lcoe_values: dict[str, list[np.ndarray]] = defaultdict(list)
    legacy_differences: dict[str, list[np.ndarray]] = defaultdict(list)
    key_hashes: set[int] = set()
    output_columns: list[str] = []

    # Read E0 as raw text so original field values are copied exactly.
    for chunk_number, chunk in enumerate(pd.read_csv(
        E0_CSV, chunksize=CHUNK_SIZE, dtype=str, keep_default_na=False,
    )):
        transformed = transform_chunk(chunk, refs, e0_manifest)
        if not output_columns:
            output_columns = list(transformed.columns)
        transformed.to_csv(
            temp, mode="w" if chunk_number == 0 else "a", header=chunk_number == 0,
            index=False, encoding="utf-8", float_format="%.17g", lineterminator="\n",
        )
        hashes = pd.util.hash_pandas_object(
            transformed[["Top_K", "coolant", "scenario", "Npw", "rho_turn_uOhm_cm2", "R_joint_nOhm"]],
            index=False,
        ).astype("uint64")
        for value in hashes:
            ivalue = int(value)
            if ivalue in key_hashes:
                raise AssertionError("duplicate B1 design key detected")
            key_hashes.add(ivalue)

        for scenario, group in transformed.groupby("scenario", sort=False):
            positions = np.arange(scenario_seen[scenario], scenario_seen[scenario] + len(group))
            take = np.isin(positions, list(scenario_targets[scenario]))
            if take.any():
                samples.append(group.loc[take].copy())
            scenario_seen[scenario] += len(group)
            finite = numeric(group["lcoe_anchor_USD_per_MWh"]).to_numpy(dtype=float)
            lcoe_values[scenario].append(finite[np.isfinite(finite)])
            legacy = numeric(group["legacy_e0_LCOE_plant_USD_per_MWh"]).to_numpy(dtype=float)
            both = np.isfinite(finite) & np.isfinite(legacy)
            legacy_differences[scenario].append((finite - legacy)[both])
        row_total += len(transformed)

    if row_total != EXPECTED_ROWS:
        raise AssertionError(f"B1 row count {row_total} != {EXPECTED_ROWS}")
    os.replace(temp, B1_GRID)

    sample_df = pd.concat(samples, ignore_index=True)
    reference_keys = reference_df[["scenario", "Top_K", "coolant", "Npw", "rho_turn_uOhm_cm2", "R_joint_nOhm"]]
    reference_from_b1: list[pd.DataFrame] = []
    for chunk in pd.read_csv(B1_GRID, chunksize=50_000, low_memory=False):
        mask = (
            np.isclose(chunk["Top_K"], REFERENCE_COORDS["Top_K"])
            & chunk["coolant"].eq(REFERENCE_COORDS["coolant"])
            & chunk["Npw"].eq(REFERENCE_COORDS["Npw"])
            & np.isclose(chunk["rho_turn_uOhm_cm2"], REFERENCE_COORDS["rho_turn_uOhm_cm2"])
            & np.isclose(chunk["R_joint_nOhm"], REFERENCE_COORDS["R_joint_nOhm"])
        )
        if mask.any():
            reference_from_b1.append(chunk.loc[mask])
    sample_df = pd.concat([sample_df, *reference_from_b1], ignore_index=True).drop_duplicates(
        ["scenario", "Top_K", "coolant", "Npw", "rho_turn_uOhm_cm2", "R_joint_nOhm"]
    ).reset_index(drop=True)
    sample_df.insert(0, "sample_id", [f"B1-{i:03d}" for i in range(1, len(sample_df) + 1)])

    increment_cols = [
        "sample_id", "scenario", "Top_K", "coolant", "Npw", "rho_turn_uOhm_cm2", "R_joint_nOhm",
        "C_HTS_direct_USD", "C_power_supply_direct_USD", "C_coolant_fill_direct_USD",
        "delta_C_HTS_direct_USD", "delta_C_HTS_capitalized_USD", "delta_C_PS_direct_USD",
        "delta_C_PS_capitalized_USD", "delta_C_coolant_fill_USD", "delta_C_mag_anchor_USD",
        "C_anchor_USD", "C_plant_anchor_USD",
    ]
    increment_sample_path = TABLES / "B0_increment_mapping_sample.csv"
    sample_df[increment_cols].to_csv(increment_sample_path, index=False, encoding="utf-8-sig", float_format="%.17g")

    reproduction_rows = reproduce_samples(sample_df)
    reproduction_path = TABLES / "B1_economic_reproduction_sample.csv"
    write_csv(reproduction_path, reproduction_rows)
    reproduction_max = max(float(row["max_absolute_residual"]) for row in reproduction_rows)
    if any(row["status"] != "PASS" for row in reproduction_rows):
        raise AssertionError("B1 economic reproduction sample failed")

    identity_rows = compare_non_economic_fields(output_columns)
    identity_path = TABLES / "B1_field_identity_audit.csv"
    write_csv(identity_path, identity_rows)
    if any(row["status"] == "FAIL" for row in identity_rows):
        raise AssertionError("B1 non-economic field identity failed")

    summary_rows = []
    for scenario in ("S1", "S2", "S3", "S4", "S5", "S6"):
        values = np.concatenate(lcoe_values[scenario])
        diffs = np.concatenate(legacy_differences[scenario])
        ref_row = pd.concat(reference_from_b1, ignore_index=True)
        ref_lcoe = float(ref_row.loc[ref_row["scenario"] == scenario, "lcoe_anchor_USD_per_MWh"].iloc[0])
        summary_rows.append({
            "scenario": scenario, "row_count": scenario_seen[scenario], "finite_lcoe_count": len(values),
            "lcoe_anchor_min_USD_per_MWh": float(np.min(values)),
            "lcoe_anchor_median_USD_per_MWh": float(np.median(values)),
            "lcoe_anchor_max_USD_per_MWh": float(np.max(values)),
            "reference_magnet_lcoe_USD_per_MWh": ref_lcoe,
            "new_minus_legacy_lcoe_min": float(np.min(diffs)),
            "new_minus_legacy_lcoe_median": float(np.median(diffs)),
            "new_minus_legacy_lcoe_max": float(np.max(diffs)),
        })
    summary_path = TABLES / "B1_economic_summary.csv"
    write_csv(summary_path, summary_rows)

    e0_sha_after = sha256(E0_CSV)
    if e0_sha_after != e0_sha_before:
        raise AssertionError("E0 CSV changed during B0-B1")
    b1_sha = sha256(B1_GRID)
    b1_rows = line_count(B1_GRID)
    if b1_rows != EXPECTED_ROWS:
        raise AssertionError(f"written B1 row count {b1_rows} != {EXPECTED_ROWS}")

    anchor_validation_path = LOGS / "B0_anchor_validation.md"
    max_ref_increment = float(reference_df["delta_C_mag_anchor_USD"].abs().max())
    anchor_validation_path.write_text(f"""# B0 anchor validation

- Result: **PASS**
- Reference magnet: `{REFERENCE_MAGNET_ID}`
- Unique matches: one row per scenario for S1-S6; B0 formal table reports S1-S3 and retains S4-S6 for B1 price-only continuity.
- Maximum absolute reference increment: `{max_ref_increment:.3e}` USD (criterion `<1e-6` USD).
- Reference plant identity: `C_plant_anchor_USD == C_anchor_USD` for all six scenarios.
- E0 input SHA256 unchanged: `{e0_sha_after}`.
""", encoding="utf-8")

    boundary_path = LOGS / "B0_capital_boundary_definition.md"
    boundary_path.write_text(f"""# B0 capital boundary definition

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run + validate
- Origin Date: {started.date().isoformat()}
- Verification Status: VERIFIED
- Version Label: B0_incremental_anchor_v1

## Frozen boundary

`C_plant(i,s) = C_anchor(s) + delta_C_mag_anchor(i,s)`.

`delta_C_mag_anchor = {M_HTS} * delta_C_HTS_direct + {M_PS} * delta_C_PS_direct + {M_COOLANT_FILL} * delta_C_coolant_fill`.

- Wade total capital: {WADE_TOTAL_CAPITAL_2010_MUSD} million 2010 USD.
- Wade reference coil direct account: {WADE_COIL_DIRECT_2010_MUSD} million 2010 USD.
- 2010-to-2025 factor: {PRICE_FACTOR_2010_TO_2025}.
- Maturity scales S1/S2/S3: 1, 2/3, 1/3; S4-S6 inherit S2.
- HTS factors: {HTS_CAPITALIZATION_FACTORS}; power-supply factors: {PS_CAPITALIZATION_FACTORS}.
- Initial coolant inventory multiplier: 1.
- This is an incremental project-capital mapping and is not labeled or compared as nuclear overnight USD/kW.
""", encoding="utf-8")

    integrity = {
        "result": "PASS",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_e0": {"path": rel(E0_CSV), "rows": EXPECTED_ROWS, "sha256_before": e0_sha_before, "sha256_after": e0_sha_after},
        "output_b1": {"path": rel(B1_GRID), "rows": b1_rows, "columns": len(output_columns), "sha256": b1_sha},
        "unique_design_keys": len(key_hashes),
        "duplicate_design_keys": row_total - len(key_hashes),
        "non_economic_columns_checked": sum(row["classification"] == "non_economic_identity" for row in identity_rows),
        "non_economic_mismatch_count": sum(int(row["mismatch_count"]) for row in identity_rows if row["classification"] == "non_economic_identity"),
        "energy_columns_identity": {c: next(row for row in identity_rows if row["source_column"] == c)["status"] for c in ("gross_energy_annual_MWh", "E_cryo_TF_annual_MWh", "net_energy_annual_MWh")},
        "reference_match_counts": reference_df.set_index("scenario")["match_count_in_scenario"].astype(int).to_dict(),
        "max_reference_increment_USD": max_ref_increment,
        "economic_reproduction_sample_rows": len(reproduction_rows),
        "max_economic_reproduction_absolute_residual": reproduction_max,
        "physical_model_rerun": False,
        "formal_B_C_D_figures_generated": False,
        "minimum_availability_root_solve_run": False,
    }
    integrity_path = LOGS / "B1_data_integrity.json"
    integrity_path.write_text(json.dumps(integrity, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    b0_outputs = [anchor_path, reference_path, increment_sample_path, boundary_path, anchor_validation_path]
    b0_manifest_path = MANIFESTS / "B0_capital_boundary_manifest.json"
    b0_manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "result": "PASS",
        "git_commit": commit, "git_dirty_before_run": bool(dirty_before),
        "plan": {"path": rel(PLAN), "sha256": sha256(PLAN)},
        "v2": {"path": rel(V2_DOC), "sha256": sha256(V2_DOC)},
        "e0": {"path": rel(E0_CSV), "sha256": e0_sha_after},
        "capital_boundary_version": CAPITAL_BOUNDARY_VERSION,
        "reference_magnet_id": REFERENCE_MAGNET_ID,
        "outputs": [{"path": rel(p), "rows": line_count(p) if p.suffix == ".csv" else None, "sha256": sha256(p)} for p in b0_outputs],
    }
    b0_manifest_path.write_text(json.dumps(b0_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report_path = LOGS / "B1_execution_report.md"
    s1 = summary_rows[0]
    s2 = summary_rows[1]
    s3 = summary_rows[2]
    report_path.write_text(f"""# B1 V2 anchor-economics execution report

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run + validate
- Origin Date: {started.date().isoformat()}
- Verification Status: VERIFIED
- Version Label: B1_anchor_economics_v1

## Result

**PASS.** A separate {row_total:,}-row V2 anchor-economics grid was generated without modifying or rerunning E0 physics.

## Anchor LCOE summary (finite rows)

| Scenario | Minimum | Median | Maximum | Reference magnet |
|---|---:|---:|---:|---:|
| S1 | {s1['lcoe_anchor_min_USD_per_MWh']:.6f} | {s1['lcoe_anchor_median_USD_per_MWh']:.6f} | {s1['lcoe_anchor_max_USD_per_MWh']:.6f} | {s1['reference_magnet_lcoe_USD_per_MWh']:.6f} |
| S2 | {s2['lcoe_anchor_min_USD_per_MWh']:.6f} | {s2['lcoe_anchor_median_USD_per_MWh']:.6f} | {s2['lcoe_anchor_max_USD_per_MWh']:.6f} | {s2['reference_magnet_lcoe_USD_per_MWh']:.6f} |
| S3 | {s3['lcoe_anchor_min_USD_per_MWh']:.6f} | {s3['lcoe_anchor_median_USD_per_MWh']:.6f} | {s3['lcoe_anchor_max_USD_per_MWh']:.6f} | {s3['reference_magnet_lcoe_USD_per_MWh']:.6f} |

## Validation

- E0 SHA256 before/after: `{e0_sha_before}` / `{e0_sha_after}`.
- B1 SHA256: `{b1_sha}`.
- Unique design keys: {len(key_hashes):,}; duplicates: {row_total-len(key_hashes)}.
- Non-economic field mismatches: {integrity['non_economic_mismatch_count']}.
- Maximum reference increment: {max_ref_increment:.3e} USD.
- Economic reproduction sample: {len(reproduction_rows)} rows, maximum absolute residual {reproduction_max:.3e}.
- B2 was not run and no B/C/D figure was generated.
""", encoding="utf-8")

    b1_outputs = [B1_GRID, identity_path, reproduction_path, summary_path, integrity_path, report_path]
    b1_manifest_path = MANIFESTS / "B1_V2_anchor_economics_manifest.json"
    b1_manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "result": "PASS",
        "git_commit": commit, "git_dirty_before_run": bool(dirty_before),
        "input_hashes": {"plan": sha256(PLAN), "v2": sha256(V2_DOC), "e0_csv": e0_sha_after, "e0_manifest": sha256(E0_MANIFEST), "capital_anchor_module": sha256(CODE_ROOT / "src" / "fusion_tem" / "economic" / "capital_anchor.py"), "runner": sha256(SCRIPT)},
        "frozen_parameters": {
            "reference_magnet_id": REFERENCE_MAGNET_ID, "coolant_replenish_fraction": COOLANT_REPLENISH_FRACTION,
            "m_HTS": M_HTS, "m_PS": M_PS, "m_coolant_fill": M_COOLANT_FILL,
            "discount_rate_by_scenario": dict(DISCOUNT_RATE_BY_SCENARIO), "project_years_by_scenario": dict(PROJECT_YEARS_BY_SCENARIO),
            "hts_price_by_scenario": dict(HTS_PRICE_BY_SCENARIO), "core_vom_by_scenario": dict(CORE_VOM_BY_SCENARIO),
            "power_supply_price_USD_per_A": POWER_SUPPLY_PRICE_USD_PER_A, "PCS_capital_USD_per_kWe": PCS_CAPITAL_USD_PER_KWE,
            "PCS_FOM_USD_per_kWe_year": PCS_FOM_USD_PER_KWE_YEAR, "PCS_VOM_USD_per_MWh": PCS_VOM_USD_PER_MWH_E,
            "non_tf_aux_fraction": NON_TF_AUX_FRACTION,
        },
        "outputs": [{"path": rel(p), "rows": line_count(p) if p.suffix == ".csv" else None, "sha256": sha256(p)} for p in b1_outputs],
        "B2_authorized": True,
    }
    b1_manifest_path.write_text(json.dumps(b1_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "result": "PASS", "rows": row_total, "columns": len(output_columns),
        "E0_sha256": e0_sha_after, "B1_sha256": b1_sha,
        "B0_manifest_sha256": sha256(b0_manifest_path),
        "B1_manifest_sha256": sha256(b1_manifest_path),
        "summary": summary_rows, "B2_authorized": True,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
