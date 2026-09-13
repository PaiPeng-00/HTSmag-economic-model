#!/usr/bin/env python3
"""Build the V6.2 direct-circuit scalar cache in independently resumable blocks.

The only parallel dimension is temperature; use all three independent temperature blocks by default.  A worker never writes another
worker's file; the deterministic merge is performed after every selected block
has completed and passed its matrix/device provenance checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEVICE = "arc_16pancake_nuc600_v6_2"
TOPS = (4.2, 10.0, 20.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def top_label(top: float) -> str:
    return f"{top:g}K"


def worker(top: float, stage: Path, config: Path, npw_max: int | None) -> Path:
    output = stage / f"v6_2_a_{top_label(top)}.csv"
    cmd = [
        sys.executable,
        "-S",
        str(ROOT / "scripts" / "2_charging" / "2.7_build_circuit_scalar_cache.py"),
        "--scan-config", str(config), "--output", str(output),
        "--top", str(top), "--device", DEVICE,
    ]
    if npw_max is not None:
        cmd.extend(["--npw-max", str(npw_max)])
    env = os.environ.copy()
    env["FUSION_DEVICE"] = DEVICE
    subprocess.run(cmd, check=True, cwd=ROOT, env=env)
    return output


def verify_block(path: Path, top: float, expected_rows: int) -> tuple[str, str]:
    data = pd.read_csv(path)
    if len(data) != expected_rows:
        raise RuntimeError(f"{path.name}: expected {expected_rows} rows, found {len(data)}")
    if set(data["device"].dropna().unique()) != {DEVICE}:
        raise RuntimeError(f"{path.name}: device provenance mismatch")
    for column in ("TF_system_matrix_file", "TF_system_matrix_sha256"):
        values = set(data[column].dropna().astype(str).unique())
        if len(values) != 1:
            raise RuntimeError(f"{path.name}: ambiguous {column}: {values}")
    if set(data["Top_K"].astype(float).round(6)) != {round(top, 6)}:
        raise RuntimeError(f"{path.name}: temperature selector leakage")
    return str(data["TF_system_matrix_file"].iloc[0]), str(data["TF_system_matrix_sha256"].iloc[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--scan-config", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--npw-max", type=int, default=None,
                        help="Benchmark-only inclusive upper Npw bound.")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    stage = args.stage.resolve()
    config = args.scan_config.resolve()
    if not config.exists():
        raise FileNotFoundError(config)
    stage.mkdir(parents=True, exist_ok=True)
    npw_count = min(200, args.npw_max) if args.npw_max else 200
    expected_rows = npw_count * 61
    selected: list[tuple[float, Path]] = []
    for top in TOPS:
        block = stage / f"v6_2_a_{top_label(top)}.csv"
        if block.exists() and args.resume:
            selected.append((top, block))
        elif block.exists():
            raise FileExistsError(f"refusing to overwrite {block}; use --resume or a new stage")

    pending = [top for top in TOPS if not (stage / f"v6_2_a_{top_label(top)}.csv").exists()]
    started = datetime.now(timezone.utc).isoformat()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(worker, top, stage, config, args.npw_max): top for top in pending}
        for future in as_completed(futures):
            selected.append((futures[future], future.result()))

    selected.sort(key=lambda item: item[0])
    provenance = [verify_block(path, top, expected_rows) for top, path in selected]
    if len(set(provenance)) != 1:
        raise RuntimeError(f"V6.2 cache blocks use inconsistent matrix provenance: {provenance}")
    final_csv = stage / "v6_2_a_circuit_scalars.csv"
    if final_csv.exists():
        raise FileExistsError(f"refusing to overwrite merged cache {final_csv}")
    with final_csv.open("wb") as target:
        for index, (_, block) in enumerate(selected):
            with block.open("rb") as source:
                if index:
                    source.readline()
                target.write(source.read())
    final = pd.read_csv(final_csv)
    if len(final) != len(TOPS) * expected_rows:
        raise RuntimeError("merged cache row-count mismatch")
    manifest = {
        "experiment_id": "V6.2-A",
        "status": "COMPLETE",
        "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "device": DEVICE,
        "workers": args.workers,
        "npw_max": args.npw_max,
        "expected_rows": len(TOPS) * expected_rows,
        "actual_rows": len(final),
        "matrix_file": provenance[0][0],
        "matrix_sha256": provenance[0][1],
        "block_files": [path.name for _, path in selected],
        "merged_csv": final_csv.name,
        "merged_csv_sha256": sha256(final_csv),
    }
    (stage / "v6_2_a_circuit_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
