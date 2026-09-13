"""Memory-bounded export of a wide scan CSV to a complete XLSX workbook."""
from __future__ import annotations

import csv
from pathlib import Path

import xlsxwriter


EXCEL_MAX_ROWS = 1_048_576
EXCEL_MAX_COLUMNS = 16_384


def stream_csv_to_xlsx(
    csv_path: Path,
    xlsx_path: Path,
    *,
    sheet_name: str = "scan",
    progress_interval: int = 50_000,
) -> tuple[int, int]:
    """Write CSV rows sequentially so all cells survive constant-memory mode."""
    csv_path = Path(csv_path)
    xlsx_path = Path(xlsx_path)
    xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    workbook = xlsxwriter.Workbook(
        xlsx_path,
        {
            "constant_memory": True,
            "strings_to_numbers": True,
            "strings_to_formulas": False,
            "strings_to_urls": False,
        },
    )
    # The wide full-grid workbook can trigger the classic ZIP size limit.
    workbook.use_zip64()
    worksheet = workbook.add_worksheet(sheet_name)
    worksheet.freeze_panes(1, 0)

    row_count = 0
    column_count = 0
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream)
            for row_index, row in enumerate(reader):
                if row_index >= EXCEL_MAX_ROWS:
                    raise ValueError(
                        f"CSV exceeds Excel's {EXCEL_MAX_ROWS:,}-row limit"
                    )
                if row_index == 0:
                    column_count = len(row)
                    if column_count > EXCEL_MAX_COLUMNS:
                        raise ValueError(
                            f"CSV exceeds Excel's {EXCEL_MAX_COLUMNS:,}-column limit"
                        )
                elif len(row) != column_count:
                    raise ValueError(
                        f"CSV row {row_index + 1:,} has {len(row)} columns; "
                        f"expected {column_count}"
                    )
                values = [
                    None if value == "" else (
                        True if value == "True" else (
                            False if value == "False" else value
                        )
                    )
                    for value in row
                ]
                worksheet.write_row(row_index, 0, values)
                row_count = row_index + 1
                if (
                    progress_interval > 0
                    and row_count % progress_interval == 0
                ):
                    print(
                        f"[XLSX] Wrote {row_count:,} rows x "
                        f"{column_count:,} columns",
                        flush=True,
                    )
    finally:
        workbook.close()
    return row_count, column_count
