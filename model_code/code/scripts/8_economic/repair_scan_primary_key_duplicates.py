"""Audit and remove duplicated design keys from an interrupted/concurrent scan."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from fusion_tem.economic.feasibility import (
    add_joint_lcoe_reference,
    apply_system_feasibility,
)
from tfmag.xlsx import stream_csv_to_xlsx


KEY_COLUMNS = [
    "Top_K",
    "coolant",
    "Npw",
    "rho_turn_uOhm_cm2",
    "R_joint_nOhm",
    "scenario",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--xlsx-path", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    before_sha256 = sha256(args.csv_path)
    frame = pd.read_csv(args.csv_path, low_memory=False)
    missing = sorted(set(KEY_COLUMNS).difference(frame.columns))
    if missing:
        raise KeyError(f"Missing scan primary-key columns: {missing}")

    duplicate_all = frame.duplicated(KEY_COLUMNS, keep=False)
    duplicate_extra = frame.duplicated(KEY_COLUMNS, keep="first")
    duplicate_rows = frame.loc[duplicate_all].copy()
    compare_columns = [
        column for column in frame.columns if column not in KEY_COLUMNS
    ]
    duplicate_rows["_row_hash"] = pd.util.hash_pandas_object(
        duplicate_rows[compare_columns], index=False
    ).to_numpy()
    nonidentical_groups = int(
        (
            duplicate_rows.groupby(KEY_COLUMNS, dropna=False)["_row_hash"]
            .nunique()
            .gt(1)
        ).sum()
    )
    if nonidentical_groups:
        raise AssertionError(
            f"{nonidentical_groups} duplicate key groups have unequal row values"
        )

    repaired = frame.loc[~duplicate_extra].copy()
    if len(repaired) != args.expected_rows:
        raise AssertionError(
            f"Expected {args.expected_rows:,} unique rows, got {len(repaired):,}"
        )
    if repaired.duplicated(KEY_COLUMNS).any():
        raise AssertionError("Primary-key duplicates remain after repair")

    repaired, af_max = apply_system_feasibility(repaired)
    repaired, lcoe_reference = add_joint_lcoe_reference(repaired)
    if not np.isfinite(
        repaired.loc[
            repaired["feasible_joint"].astype(bool),
            "LCOE_plant_USD_per_MWh",
        ]
    ).all():
        raise AssertionError("A repaired jointly feasible row has nonfinite LCOE")

    csv_temp = args.csv_path.with_suffix(".repairing.csv")
    xlsx_temp = args.xlsx_path.with_suffix(".repairing.xlsx")
    repaired.to_csv(csv_temp, index=False)
    xlsx_rows, xlsx_columns = stream_csv_to_xlsx(
        csv_temp, xlsx_temp, sheet_name="scan"
    )
    if (xlsx_rows, xlsx_columns) != (len(repaired) + 1, len(repaired.columns)):
        raise RuntimeError(
            "XLSX dimension mismatch after streaming export: "
            f"{xlsx_rows}x{xlsx_columns}"
        )
    os.replace(csv_temp, args.csv_path)
    os.replace(xlsx_temp, args.xlsx_path)

    report = {
        "input_rows": int(len(frame)),
        "duplicate_rows_including_first": int(duplicate_all.sum()),
        "duplicate_rows_removed": int(duplicate_extra.sum()),
        "duplicate_key_groups": int(
            duplicate_rows.groupby(KEY_COLUMNS, dropna=False).ngroups
        ),
        "nonidentical_duplicate_groups": nonidentical_groups,
        "output_rows": int(len(repaired)),
        "output_columns": int(len(repaired.columns)),
        "primary_key_columns": KEY_COLUMNS,
        "af_max_scenario": {key: float(value) for key, value in af_max.items()},
        "lcoe_reference_joint_feasible": {
            key: float(value) for key, value in lcoe_reference.items()
        },
        "csv_sha256_before": before_sha256,
        "csv_sha256_after": sha256(args.csv_path),
        "xlsx_sha256_after": sha256(args.xlsx_path),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
