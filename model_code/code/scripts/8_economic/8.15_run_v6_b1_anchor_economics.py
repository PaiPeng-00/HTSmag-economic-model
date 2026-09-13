#!/usr/bin/env python3
"""Apply the frozen B1 anchor-economics transform to the correct-device V6 grid.

The source grid is read-only.  This driver deliberately imports the frozen
8.5 transform rather than copying its capital or LCOE equations, while keeping
all V6 artefacts out of the formal B1 output tree.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "v6_hts_temperature_tolerance_arc16pancake_nuc600"
RAW = OUT / "v6_direct_complete_designs_arc16pancake_nuc600.csv"
RAW_MANIFEST = RAW.with_suffix(".manifest.json")
ANCHOR_SCRIPT = Path(__file__).with_name("8.5_recompute_v2_anchor_economics.py")
B1 = ROOT / "outputs" / "target_price_window" / "tables" / "B1_full_grid_green_eta030_V2_anchor_economics.csv"
OUTPUT = OUT / "v6_b1_anchor_economics_arc16pancake_nuc600.csv"
AUDIT = OUT / "v6_b1_anchor_economics_audit.json"
EXPECTED_ROWS = 4_573_800
CHUNK_SIZE = 20_000
REFERENCE = {"Top_K": 10.0, "coolant": "He", "Npw": 19, "rho_turn_uOhm_cm2": 10_000.0, "R_joint_nOhm": 1.0}
KEYS = ["scenario", "Top_K", "coolant", "Npw", "rho_turn_uOhm_cm2", "R_joint_nOhm"]
CHECK_FIELDS = [
    "Charging_time_999_h", "P_cryo_electric_W", "Q_total_Tc_W", "Q_nuclear_W",
    "E_cryo_TF_annual_MWh", "net_energy_annual_MWh", "E_fusion_th_year_MWh", "AF",
    "r_cryo_re_fraction", "C_plant_anchor_USD", "annual_noncapital_cost_USD",
    "lcoe_anchor_USD_per_MWh",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def line_count(path: Path) -> int:
    with path.open("rb") as stream:
        return sum(block.count(b"\n") for block in iter(lambda: stream.read(8 * 1024 * 1024), b"")) - 1


def import_anchor():
    spec = importlib.util.spec_from_file_location("v6_frozen_b1_anchor", ANCHOR_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def reference_mask(frame: pd.DataFrame) -> pd.Series:
    return (
        np.isclose(pd.to_numeric(frame["Top_K"]), REFERENCE["Top_K"])
        & frame["coolant"].eq(REFERENCE["coolant"])
        & frame["Npw"].eq(REFERENCE["Npw"])
        & np.isclose(pd.to_numeric(frame["rho_turn_uOhm_cm2"]), REFERENCE["rho_turn_uOhm_cm2"])
        & np.isclose(pd.to_numeric(frame["R_joint_nOhm"]), REFERENCE["R_joint_nOhm"])
    )


def assert_source(raw_sha: str) -> dict:
    manifest = json.loads(RAW_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("outputs_root") != "outputs\\arc_16pancake_nuc600":
        raise AssertionError("raw V6 manifest does not identify arc_16pancake_nuc600")
    outputs = manifest.get("outputs", [])
    if len(outputs) != 1 or outputs[0].get("sha256") != raw_sha:
        raise AssertionError("raw V6 CSV hash does not match its completion manifest")
    if manifest.get("cost_boundary_version") != "plant_v4_core_pcs_coolant_2025usd_direct_hts":
        raise AssertionError("unexpected raw V6 economic boundary")
    return manifest


def collect_reference_rows(path: Path, columns: list[str]) -> pd.DataFrame:
    found = []
    for chunk in pd.read_csv(path, usecols=columns, chunksize=50_000, low_memory=False):
        mask = reference_mask(chunk)
        if mask.any():
            found.append(chunk.loc[mask].copy())
    rows = pd.concat(found, ignore_index=True)
    rows = rows.loc[rows["scenario"].isin(("S1", "S2", "S3"))].sort_values("scenario").reset_index(drop=True)
    if rows["scenario"].tolist() != ["S1", "S2", "S3"]:
        raise AssertionError(f"expected one S1--S3 reference row, got {rows['scenario'].tolist()}")
    return rows


def run() -> None:
    if not RAW.exists() or not RAW_MANIFEST.exists() or not B1.exists():
        raise FileNotFoundError("required V6 raw grid, manifest, or B1 grid is missing")
    if OUTPUT.with_suffix(".csv.partial").exists() or AUDIT.exists():
        raise FileExistsError("refusing to overwrite a V6 B1-anchor partial result or audit")

    started = datetime.now(timezone.utc)
    raw_sha_before = sha256(RAW)
    raw_manifest = assert_source(raw_sha_before)
    anchor = import_anchor()
    e0_manifest = json.loads(anchor.E0_MANIFEST.read_text(encoding="utf-8"))
    _, refs = anchor.find_reference_rows(e0_manifest)

    columns: list[str] = []
    if OUTPUT.exists():
        # Recovery mode: a downstream audit-only failure occurred after atomic publication.
        # Never rewrite the published V6 CSV.
        rows = line_count(OUTPUT)
        columns = list(pd.read_csv(OUTPUT, nrows=0).columns)
        print(f"[Info] reusing published V6 anchor output ({rows:,} rows) for audit", flush=True)
    else:
        temporary = OUTPUT.with_suffix(".csv.partial")
        rows = 0
        for number, chunk in enumerate(pd.read_csv(RAW, chunksize=CHUNK_SIZE, dtype=str, keep_default_na=False)):
            transformed = anchor.transform_chunk(chunk, refs, e0_manifest)
            if not columns:
                columns = list(transformed.columns)
            transformed.to_csv(
                temporary, mode="w" if number == 0 else "a", header=number == 0,
                index=False, encoding="utf-8", float_format="%.17g", lineterminator="\n",
            )
            rows += len(transformed)
            if rows % 200_000 == 0:
                print(f"[Progress] anchored {rows:,}/{EXPECTED_ROWS:,}", flush=True)
        if rows != EXPECTED_ROWS:
            raise AssertionError(f"V6 raw row count {rows} != {EXPECTED_ROWS}")
        os.replace(temporary, OUTPUT)
    if rows != EXPECTED_ROWS:
        raise AssertionError(f"V6 anchored row count {rows} != {EXPECTED_ROWS}")
    if line_count(OUTPUT) != EXPECTED_ROWS:
        raise AssertionError("written V6 anchored row count is incorrect")
    raw_sha_after = sha256(RAW)
    if raw_sha_after != raw_sha_before:
        raise AssertionError("raw V6 grid changed during frozen B1 economics")

    b1_reference = collect_reference_rows(B1, KEYS + CHECK_FIELDS)
    v6_reference = collect_reference_rows(OUTPUT, KEYS + CHECK_FIELDS)
    joined = v6_reference.merge(b1_reference, on=KEYS, suffixes=("_v6", "_b1"), validate="one_to_one")
    statistics = {}
    for field in CHECK_FIELDS:
        delta = (pd.to_numeric(joined[f"{field}_v6"]) - pd.to_numeric(joined[f"{field}_b1"])).abs()
        statistics[field] = {"max_abs": float(delta.max()), "mismatch_gt_1e-6": int((delta > 1e-6).sum())}
    passed = len(joined) == 3 and all(stat["mismatch_gt_1e-6"] == 0 for stat in statistics.values())
    audit = {
        "experiment_id": "V6-B1-frozen-anchor-economics",
        "status": "PASS" if passed else "FAIL",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device": "arc_16pancake_nuc600",
        "source_raw": {"path": str(RAW.relative_to(ROOT)), "rows": rows, "sha256_before": raw_sha_before, "sha256_after": raw_sha_after},
        "source_raw_manifest_sha256": sha256(RAW_MANIFEST),
        "source_raw_cost_boundary": raw_manifest["cost_boundary_version"],
        "frozen_transform_script": {"path": str(ANCHOR_SCRIPT.relative_to(ROOT)), "sha256": sha256(ANCHOR_SCRIPT)},
        "b1_reference_grid": {"path": str(B1.relative_to(ROOT)), "sha256": sha256(B1)},
        "output": {"path": str(OUTPUT.relative_to(ROOT)), "rows": line_count(OUTPUT), "columns": len(columns), "sha256": sha256(OUTPUT)},
        "reference_coordinates": REFERENCE,
        "overlap_rows": int(len(joined)),
        "field_comparison": statistics,
        "capital_boundary_version": anchor.CAPITAL_BOUNDARY_VERSION,
        "started_utc": started.isoformat(),
    }
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
    if not passed:
        raise SystemExit("V6 B1-anchor overlap audit failed; V6-B--F remains blocked")


if __name__ == "__main__":
    run()
