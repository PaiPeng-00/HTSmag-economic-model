#!/usr/bin/env python3
"""Audit direct V6.1 availability against every old-V6 rho_turn anchor."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

RHO_OLD = np.array([10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000,5000,10000], dtype=float)
KEYS = ["scenario", "Top_K", "Npw", "rho_turn_uOhm_cm2"]
COMPARE = ["Charging_time_999_h", "AF", "AF_max_scenario", "AF_ref"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_old_anchors(path: Path) -> pd.DataFrame:
    cols = KEYS + ["coolant", "R_joint_nOhm"] + COMPARE
    chunks = []
    for chunk in pd.read_csv(path, usecols=cols, chunksize=250_000):
        mask = (
            (chunk["coolant"] == "He")
            & np.isclose(pd.to_numeric(chunk["R_joint_nOhm"]), 1.0, rtol=0.0, atol=1e-12)
            & pd.to_numeric(chunk["rho_turn_uOhm_cm2"]).isin(RHO_OLD)
        )
        selected = chunk.loc[mask, KEYS + COMPARE]
        if len(selected):
            chunks.append(selected)
    old = pd.concat(chunks, ignore_index=True)
    if old.duplicated(KEYS).any():
        raise ValueError("old V6 Rj=1 anchors are not unique")
    expected = 3 * 3 * 200 * len(RHO_OLD)
    if len(old) != expected:
        raise ValueError(f"old V6 anchor count {len(old)} != expected {expected}")
    return old


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--availability", type=Path, required=True)
    parser.add_argument("--old-v6", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-12)
    args = parser.parse_args()
    new = pd.read_csv(args.availability, usecols=KEYS + COMPARE + ["device"])
    new = new.loc[pd.to_numeric(new["rho_turn_uOhm_cm2"]).isin(RHO_OLD)].copy()
    if set(new["device"].astype(str).str.lower()) != {"arc_16pancake_nuc600"}:
        raise ValueError("new availability does not carry the frozen V6 device identity")
    expected = 3 * 3 * 200 * len(RHO_OLD)
    if len(new) != expected or new.duplicated(KEYS).any():
        raise ValueError("new direct-anchor availability grid is incomplete or non-unique")
    old = load_old_anchors(args.old_v6.resolve())
    joined = new.merge(old, on=KEYS, suffixes=("_v61", "_v6"), validate="one_to_one")
    stats = {}
    for field in COMPARE:
        delta = (pd.to_numeric(joined[f"{field}_v61"]) - pd.to_numeric(joined[f"{field}_v6"])) .abs()
        stats[field] = {"max_abs": float(delta.max()), "mismatch_count": int((delta > args.tolerance).sum())}
        joined[f"delta_{field}"] = delta
    passed = len(joined) == expected and all(v["mismatch_count"] == 0 for v in stats.values())
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    detail = out / "v6_1_a_legacy_rho_anchor_regression_detail.csv"
    audit_path = out / "v6_1_a_legacy_rho_anchor_regression_audit.json"
    joined.sort_values(KEYS).to_csv(detail, index=False)
    audit = {
        "experiment_id": "V6.1-A-old-V6-rho-anchor-regression",
        "status": "PASS" if passed else "FAIL",
        "old_v6": str(args.old_v6.resolve()),
        "old_v6_sha256": sha256(args.old_v6.resolve()),
        "new_availability": str(args.availability.resolve()),
        "new_availability_sha256": sha256(args.availability.resolve()),
        "legacy_rho_values_uOhm_cm2": RHO_OLD.tolist(),
        "anchor_count": int(len(joined)),
        "tolerance": args.tolerance,
        "comparisons": stats,
        "detail": str(detail),
    }
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit("legacy rho anchor regression failed")

if __name__ == "__main__":
    main()
