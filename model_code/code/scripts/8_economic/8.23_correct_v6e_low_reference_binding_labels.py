#!/usr/bin/env python3
"""Make V6-E low-reference binding labels criterion-specific and auditable."""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "v6_hts_temperature_tolerance_arc16pancake_nuc600"
TOLERANCE = OUT / "v6e_cross_scenario_tolerance.csv"
AUDIT = OUT / "v6e_cross_scenario_audit.json"
TAG = "superseded_low_reference_binding_label"
CRITERIA = {
    "feas": "availability_at_reference",
    "temp_5pct": "temperature_relative_LCOE_at_reference",
    "global_5pct": "global_relative_LCOE_at_reference",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def backup(path: Path) -> Path:
    target = path.with_name(f"{path.stem}.{TAG}{path.suffix}")
    if target.exists():
        raise FileExistsError(target)
    shutil.copy2(path, target)
    return target


def main() -> None:
    if not TOLERANCE.exists() or not AUDIT.exists():
        raise FileNotFoundError("corrected V6-E output/audit missing")
    data = pd.read_csv(TOLERANCE)
    counts = {}
    for criterion, label in CRITERIA.items():
        mask = data[f"low_reference_infeasible_{criterion}"].astype(bool)
        counts[criterion] = int(mask.sum())
        data.loc[mask, f"binding_condition_{criterion}"] = label
    staged_tolerance = TOLERANCE.with_suffix(".binding_labels.tmp.csv")
    data.to_csv(staged_tolerance, index=False, encoding="utf-8", float_format="%.17g")
    previous = json.loads(AUDIT.read_text(encoding="utf-8"))
    staged_audit = AUDIT.with_suffix(".binding_labels.tmp.json")
    tol_backup, audit_backup = backup(TOLERANCE), backup(AUDIT)
    updated = {**previous, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
               "postprocessing_binding_label_correction": {"reason": "low-reference label is criterion-specific; it is not universally availability", "rows_relabelled": counts, "backups": {"tolerance": {"path": str(tol_backup.relative_to(ROOT)), "sha256": sha256(tol_backup)}, "audit": {"path": str(audit_backup.relative_to(ROOT)), "sha256": sha256(audit_backup)}}},
               "outputs": {**previous["outputs"], TOLERANCE.name: {"path": str(TOLERANCE.relative_to(ROOT)), "sha256": sha256(staged_tolerance)}}}
    staged_audit.write_text(json.dumps(updated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    staged_tolerance.replace(TOLERANCE)
    staged_audit.replace(AUDIT)
    print(json.dumps({"status": "PASS", "rows_relabelled": counts, "tolerance_sha256": sha256(TOLERANCE)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
