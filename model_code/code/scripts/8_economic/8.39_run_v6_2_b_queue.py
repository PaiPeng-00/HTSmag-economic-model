#!/usr/bin/env python3
"""Run independent V6.2-B temperature/coolant blocks with up to four workers."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BLOCKS = ((4.2, "He"), (10.0, "He"), (20.0, "He"), (20.0, "H2"))


def completed_block(stage: Path, label: str) -> bool:
    output = stage / f"{label}.csv"
    log = stage / f"{label}.stdout.log"
    if not output.exists() or not log.exists():
        return False
    tail = log.read_text(encoding="utf-8", errors="replace")[-4096:]
    return "Deferred finalization: V6.1-B reducer will apply full-domain feasibility." in tail


def run_block(top: float, coolant: str, stage: Path, cache: Path, npw_max: int | None) -> str:
    label = f"v6_2_b_{top:g}K_{coolant}"
    command = [sys.executable, "-S", str(Path(__file__).with_name("8.38_run_v6_2_b_temperature_block.py")),
               "--top", str(top), "--coolant", coolant, "--stage", str(stage), "--cache", str(cache)]
    if npw_max is not None:
        command.extend(["--npw-max", str(npw_max)])
    log = stage / f"{label}.stdout.log"
    with log.open("w", encoding="utf-8") as handle:
        subprocess.run(command, cwd=ROOT, env=os.environ.copy(), stdout=handle, stderr=subprocess.STDOUT, check=True)
    return label


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--workers", choices=(1, 2, 3, 4), type=int, default=4)
    parser.add_argument("--npw-max", type=int, default=None)
    parser.add_argument("--resume", action="store_true",
                        help="Skip temperature/coolant blocks whose final CSV already exists.")
    args = parser.parse_args()
    stage, cache = args.stage.resolve(), args.cache.resolve()
    if not cache.exists():
        raise FileNotFoundError(cache)
    stage.mkdir(parents=True, exist_ok=True)
    queued = []
    for top, coolant in BLOCKS:
        label = f"v6_2_b_{top:g}K_{coolant}"
        output = stage / f"{label}.csv"
        if output.exists():
            if args.resume and completed_block(stage, label):
                continue
            if args.resume:
                raise RuntimeError(
                    f"incomplete existing block: {top:g}K {coolant}; "
                    "remove the partial CSV before resuming"
                )
            raise FileExistsError(f"block already exists: {top:g}K {coolant}; use a new stage")
        queued.append((top, coolant))
    started = datetime.now(timezone.utc).isoformat()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_block, top, coolant, stage, cache, args.npw_max): (top, coolant) for top, coolant in queued}
        for future in as_completed(futures):
            print(f"complete: {future.result()}")
    (stage / "v6_2_b_queue_manifest.json").write_text(json.dumps({
        "experiment_id": "V6.2-B", "status": "COMPLETE", "workers": args.workers,
        "started_utc": started, "completed_utc": datetime.now(timezone.utc).isoformat(),
        "blocks": [f"{top:g}K_{coolant}" for top, coolant in BLOCKS],
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
