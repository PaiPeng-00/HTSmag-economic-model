"""Export a complete scan CSV to XLSX with bounded memory."""
from __future__ import annotations

import argparse
from pathlib import Path

from tfmag.xlsx import stream_csv_to_xlsx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("xlsx_path", type=Path)
    args = parser.parse_args()
    rows, columns = stream_csv_to_xlsx(
        args.csv_path,
        args.xlsx_path,
        sheet_name="scan",
    )
    print(f"[OK] XLSX dimensions: {rows:,} rows x {columns:,} columns")


if __name__ == "__main__":
    main()
