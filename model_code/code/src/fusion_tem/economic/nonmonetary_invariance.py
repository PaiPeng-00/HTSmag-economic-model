"""Guarded byte-level harmonization for price-only economic reruns."""
from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path
import struct
from typing import Iterable

import numpy as np
import pandas as pd


MONETARY_OR_DEPENDENT_TOKENS = (
    "usd",
    "lcoe",
    "opex",
    "capex",
    "cost_boundary",
    "monetary_price_basis",
    "monetary_values_constant",
    "tape_cost",
    "coolant_fill_cost",
    "power_supply_cost",
    "hts_price",
    "hts_conductor_price",
    "coolant_price",
)
PRIMARY_KEY_COLUMNS = (
    "Top_K",
    "coolant",
    "Npw",
    "rho_turn_uOhm_cm2",
    "R_joint_nOhm",
    "scenario",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_monetary_or_dependent_column(column: str) -> bool:
    lowered = column.lower()
    return any(token in lowered for token in MONETARY_OR_DEPENDENT_TOKENS)


def common_nonmonetary_columns(old_path: Path, new_path: Path) -> list[str]:
    old_columns = list(pd.read_csv(old_path, nrows=0).columns)
    new_columns = set(pd.read_csv(new_path, nrows=0).columns)
    return [
        column
        for column in old_columns
        if column in new_columns
        and not is_monetary_or_dependent_column(column)
    ]


def _ordered_float_integer(value: float) -> int:
    bits = struct.unpack(">q", struct.pack(">d", float(value)))[0]
    return bits if bits >= 0 else 0x8000000000000000 - bits


def ulp_distance(left: float, right: float) -> int:
    return abs(_ordered_float_integer(left) - _ordered_float_integer(right))


def harmonize_nonmonetary_csv(
    old_path: Path,
    new_path: Path,
    output_path: Path,
    *,
    chunksize: int = 5000,
    max_allowed_ulp: int = 4,
    key_columns: Iterable[str] = PRIMARY_KEY_COLUMNS,
) -> dict[str, object]:
    """Copy exact old nonmonetary strings after a strict small-drift gate.

    The function refuses nonnumeric differences, nonfinite differences, key
    drift, or any numeric difference above ``max_allowed_ulp``. Monetary and
    monetary-dependent columns always remain from the new 2025-US$ rerun.
    """
    old_path = Path(old_path)
    new_path = Path(new_path)
    output_path = Path(output_path)
    common = common_nonmonetary_columns(old_path, new_path)
    required = set(key_columns)
    missing_keys = sorted(required - set(common))
    if missing_keys:
        raise AssertionError(
            f"Primary keys are absent from nonmonetary comparison: {missing_keys}"
        )

    new_columns = list(pd.read_csv(new_path, nrows=0).columns)
    old_iter = pd.read_csv(
        old_path,
        usecols=common,
        dtype=str,
        keep_default_na=False,
        chunksize=chunksize,
    )
    new_iter = pd.read_csv(
        new_path,
        dtype=str,
        keep_default_na=False,
        chunksize=chunksize,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite temporary output: {output_path}")

    mismatch_counts: Counter[str] = Counter()
    maximum_ulp = 0
    rows = 0
    first_chunk = True
    try:
        while True:
            try:
                old_chunk = next(old_iter)
            except StopIteration:
                old_chunk = None
            try:
                new_chunk = next(new_iter)
            except StopIteration:
                new_chunk = None
            if old_chunk is None or new_chunk is None:
                if old_chunk is not None or new_chunk is not None:
                    raise AssertionError("Old/new scan chunk counts differ")
                break
            if len(old_chunk) != len(new_chunk):
                raise AssertionError("Old/new scan row counts differ")

            for key in key_columns:
                if not np.array_equal(
                    old_chunk[key].to_numpy(), new_chunk[key].to_numpy()
                ):
                    raise AssertionError(f"Primary-key drift detected in {key}")

            for column in common:
                old_values = old_chunk[column].to_numpy()
                new_values = new_chunk[column].to_numpy()
                mismatch = old_values != new_values
                count = int(mismatch.sum())
                if not count:
                    continue
                mismatch_counts[column] += count
                old_numeric = pd.to_numeric(
                    pd.Series(old_values[mismatch]), errors="coerce"
                ).to_numpy(dtype=float)
                new_numeric = pd.to_numeric(
                    pd.Series(new_values[mismatch]), errors="coerce"
                ).to_numpy(dtype=float)
                if not (
                    np.isfinite(old_numeric).all()
                    and np.isfinite(new_numeric).all()
                ):
                    first = int(np.flatnonzero(mismatch)[0])
                    raise AssertionError(
                        f"Nonnumeric/nonfinite nonmonetary drift in {column}: "
                        f"old={old_values[first]!r}, new={new_values[first]!r}"
                    )
                local_max = max(
                    ulp_distance(left, right)
                    for left, right in zip(old_numeric, new_numeric)
                )
                maximum_ulp = max(maximum_ulp, local_max)
                if local_max > max_allowed_ulp:
                    raise AssertionError(
                        f"Nonmonetary drift in {column} exceeds ULP gate: "
                        f"{local_max} > {max_allowed_ulp}"
                    )

            new_chunk.loc[:, common] = old_chunk.loc[:, common].to_numpy()
            new_chunk.loc[:, new_columns].to_csv(
                output_path,
                mode="w" if first_chunk else "a",
                header=first_chunk,
                index=False,
                lineterminator="\n",
            )
            first_chunk = False
            rows += len(new_chunk)
    except Exception:
        output_path.unlink(missing_ok=True)
        raise
    finally:
        old_iter.close()
        new_iter.close()

    return {
        "status": "PASS",
        "rows_harmonized": rows,
        "common_nonmonetary_columns": len(common),
        "nonmonetary_cells_replaced": int(sum(mismatch_counts.values())),
        "mismatch_counts_by_column": dict(sorted(mismatch_counts.items())),
        "maximum_observed_ulp": maximum_ulp,
        "maximum_allowed_ulp": max_allowed_ulp,
        "old_scan_sha256": sha256(old_path),
        "raw_new_scan_sha256": sha256(new_path),
        "harmonized_scan_sha256": sha256(output_path),
        "monetary_columns_copied_from_old": False,
    }