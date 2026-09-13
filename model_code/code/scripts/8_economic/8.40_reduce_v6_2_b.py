#!/usr/bin/env python3
"""Stream the V6.2-B compact blocks into an auditable global feasibility summary.

Two passes avoid loading the 17.7-million-row result set into memory.  The
second pass implements the V6-B admission rule exactly: numerical non-error
row, Aplant >= 0.80, r_cryo,re <= 0.50, positive net export, and finite
positive anchored plant LCOE.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

DEVICE = "arc_16pancake_nuc600_v6_2"
FILES = ("v6_2_b_4.2K_He.csv", "v6_2_b_10K_He.csv", "v6_2_b_20K_He.csv", "v6_2_b_20K_H2.csv")
REQUIRED = {"scenario", "status", "Aplant", "r_cryo_re_fraction", "E_net_year_MWh", "LCOE_plant_USD_per_MWh", "TF_system_matrix_sha256"}


def finite(value: str | None) -> float | None:
    try:
        candidate = float(value)  # type: ignore[arg-type]
        return candidate if math.isfinite(candidate) else None
    except (TypeError, ValueError):
        return None


def rows(paths: list[Path]):
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = REQUIRED.difference(reader.fieldnames or [])
            if missing:
                raise KeyError(f"{path.name} missing columns: {sorted(missing)}")
            yield path, reader


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    args = parser.parse_args()
    stage = args.stage.resolve()
    paths = [stage / name for name in FILES]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("missing V6.2-B blocks: " + ", ".join(missing))
    aplant_max: dict[str, float] = defaultdict(lambda: -math.inf)
    matrices: set[str] = set()
    statuses: Counter[str] = Counter()
    row_count = 0
    for _, reader in rows(paths):
        for row in reader:
            row_count += 1
            statuses[str(row.get("status", ""))] += 1
            matrix = str(row.get("TF_system_matrix_sha256", "")).strip()
            if matrix:
                matrices.add(matrix)
            if str(row.get("status", "")).lower() == "error":
                continue
            aplant = finite(row.get("Aplant"))
            if aplant is not None:
                aplant_max[str(row["scenario"])] = max(aplant_max[str(row["scenario"])], aplant)
    if len(matrices) != 1:
        raise RuntimeError(f"matrix provenance is not unique: {sorted(matrices)}")
    if any(not math.isfinite(value) for value in aplant_max.values()) or set(aplant_max) != {"S1", "S2", "S3"}:
        raise RuntimeError(f"incomplete scenario Aplant maxima: {dict(aplant_max)}")
    accepted: Counter[str] = Counter()
    lcoe_min: dict[str, float] = defaultdict(lambda: math.inf)
    for _, reader in rows(paths):
        for row in reader:
            scenario = str(row["scenario"])
            aplant, cryo, net, lcoe = (finite(row.get(key)) for key in ("Aplant", "r_cryo_re_fraction", "E_net_year_MWh", "LCOE_plant_USD_per_MWh"))
            valid = (
                str(row.get("status", "")).lower() != "error" and aplant is not None and cryo is not None and net is not None and lcoe is not None
                and aplant >= 0.80 and cryo <= 0.50 and net > 0.0 and lcoe > 0.0
            )
            if valid:
                accepted[scenario] += 1
                lcoe_min[scenario] = min(lcoe_min[scenario], lcoe)
    if any(not math.isfinite(lcoe_min[scenario]) for scenario in ("S1", "S2", "S3")):
        raise RuntimeError("at least one scenario has no V6-B admitted LCOE point")
    summary = {
        "experiment_id": "V6.2-B", "status": "COMPLETE", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "rows": row_count, "matrix_sha256": next(iter(matrices)), "aplant_max_scenario": dict(aplant_max),
        "admitted_rows_by_scenario": dict(accepted), "anchored_lcoe_min_usd_per_mwh": dict(lcoe_min),
        "status_counts": dict(statuses), "admission_rule": "status_not_error; finite; Aplant>=0.80; r_cryo_re_fraction<=0.50; E_net_year_MWh>0; LCOE_plant>0",
    }
    out = stage / "v6_2_b_global_reducer.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
