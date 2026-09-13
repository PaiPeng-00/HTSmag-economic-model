#!/usr/bin/env python3
"""Run one temperature/coolant block by disjoint scenarios and merge it safely."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SCENARIOS = ("S1", "S2", "S3")
COMPLETE_MARKER = "Deferred finalization: V6.1-B reducer will apply full-domain feasibility."


def run_scenario(script: Path, stage: Path, cache: Path, top: float, coolant: str, scenario: str) -> str:
    command = [
        sys.executable,
        "-S",
        str(script),
        "--top",
        str(top),
        "--coolant",
        coolant,
        "--scenario",
        scenario,
        "--stage",
        str(stage),
        "--cache",
        str(cache),
    ]
    log = stage / f"v6_2_b_{top:g}K_{coolant}_{scenario}.stdout.log"
    with log.open("w", encoding="utf-8") as handle:
        subprocess.run(
            command,
            cwd=script.parents[2],
            env=os.environ.copy(),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )
    return scenario


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=float, required=True, choices=(4.2, 10.0, 20.0))
    parser.add_argument("--coolant", choices=("He", "H2"), required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--workers", choices=(1, 2, 3), type=int, default=3)
    args = parser.parse_args()

    script = Path(__file__).with_name("8.38_run_v6_2_b_temperature_block.py")
    stage = args.stage.resolve()
    cache = args.cache.resolve()
    stage.mkdir(parents=True, exist_ok=True)
    base = f"v6_2_b_{args.top:g}K_{args.coolant}"
    final_csv = stage / f"{base}.csv"
    if final_csv.exists():
        raise FileExistsError(f"refusing to overwrite {final_csv}")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_scenario, script, stage, cache, args.top, args.coolant, scenario): scenario
            for scenario in SCENARIOS
        }
        for future in as_completed(futures):
            print(f"complete: {future.result()}", flush=True)

    split_csvs = [stage / f"{base}_{scenario}.csv" for scenario in SCENARIOS]
    split_logs = [stage / f"{base}_{scenario}.stdout.log" for scenario in SCENARIOS]
    split_starts = [stage / f"{base}_{scenario}.start.json" for scenario in SCENARIOS]
    expected_total = 0
    for scenario, csv_path, log_path, start_path in zip(
        SCENARIOS, split_csvs, split_logs, split_starts, strict=True
    ):
        if not csv_path.exists() or not log_path.exists() or not start_path.exists():
            raise RuntimeError(f"incomplete split outputs for {scenario}")
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        if COMPLETE_MARKER not in log_text:
            raise RuntimeError(f"missing completion marker for {scenario}")
        expected_rows = int(json.loads(start_path.read_text(encoding="utf-8"))["expected_rows"])
        if f"Total: {expected_rows} rows" not in log_text:
            raise RuntimeError(f"row-total marker mismatch for {scenario}")
        expected_total += expected_rows

    partial = final_csv.with_suffix(".csv.partial")
    if partial.exists():
        partial.unlink()
    header = None
    with partial.open("wb") as target:
        for index, csv_path in enumerate(split_csvs):
            with csv_path.open("rb") as source:
                current_header = source.readline()
                if header is None:
                    header = current_header
                    target.write(current_header)
                elif current_header != header:
                    raise RuntimeError(f"header mismatch in {csv_path.name}")
                shutil.copyfileobj(source, target, length=16 * 1024 * 1024)
    partial.replace(final_csv)

    combined_log = stage / f"{base}.stdout.log"
    combined_log.write_text(
        "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in split_logs)
        + f"\n[Info] Scenario-split merge complete. Total: {expected_total} rows\n"
        + COMPLETE_MARKER
        + "\n",
        encoding="utf-8",
    )
    (stage / f"{base}.start.json").write_text(
        json.dumps(
            {
                "experiment_id": "V6.2-B",
                "status": "COMPLETE_SCENARIO_SPLIT",
                "device": "arc_16pancake_nuc600_v6_2",
                "expected_rows": expected_total,
                "scenario_parts": list(SCENARIOS),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    for path in split_csvs + split_logs + split_starts:
        path.unlink()
    print(json.dumps({"status": "PASS", "output": str(final_csv), "rows": expected_total}, indent=2))


if __name__ == "__main__":
    main()
