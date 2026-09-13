"""Harmonize price-only rerun nonmonetary strings under a strict ULP gate."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

from fusion_tem.economic.nonmonetary_invariance import (
    harmonize_nonmonetary_csv,
    sha256,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plant-v2", type=Path, required=True)
    parser.add_argument("--plant-v3", type=Path, required=True)
    parser.add_argument("--raw-backup", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-ulp", type=int, default=4)
    args = parser.parse_args()

    if args.raw_backup.exists():
        raise FileExistsError(f"Raw backup already exists: {args.raw_backup}")
    handle, temp_name = tempfile.mkstemp(
        prefix=args.plant_v3.stem + ".harmonized.",
        suffix=".csv",
        dir=args.plant_v3.parent,
    )
    os.close(handle)
    temp_path = Path(temp_name)
    temp_path.unlink()
    try:
        report = harmonize_nonmonetary_csv(
            args.plant_v2,
            args.plant_v3,
            temp_path,
            max_allowed_ulp=args.max_ulp,
        )
        shutil.copy2(args.plant_v3, args.raw_backup)
        os.replace(temp_path, args.plant_v3)
        report["raw_backup_path"] = str(args.raw_backup)
        report["raw_backup_sha256"] = sha256(args.raw_backup)
        report["final_scan_path"] = str(args.plant_v3)
        report["final_scan_sha256"] = sha256(args.plant_v3)
        if report["raw_backup_sha256"] != report["raw_new_scan_sha256"]:
            raise AssertionError("Raw backup SHA256 changed during preservation")
        if report["final_scan_sha256"] != report["harmonized_scan_sha256"]:
            raise AssertionError("Harmonized scan SHA256 changed during promotion")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
    finally:
        temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()