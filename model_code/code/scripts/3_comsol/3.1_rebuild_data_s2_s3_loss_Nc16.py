"""Rebuild the Data S2/S3 loss workbooks from the archived ARC Nc16 COMSOL sweep.

Expected COMSOL CSV files live in:
    ARC-model/results/Nc16_fig5/ARC16_fig5_{config}_T{T}_Npw{Npw}_rho{rho}.csv

Each CSV has columns:
    time_s, magnetisation_int4_W, radial_gev9_W

The output files are written to the heat-load input directory used by the
Python model:
    code-model/code/data/raw/heat_input/{data_s3_magnetization_loss.xlsx, data_s2_radial_loss.xlsx}
"""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve()
CODE_ROOT = SCRIPT.parents[2]
WORKSPACE = SCRIPT.parents[4]
RESULTS = WORKSPACE / "ARC-model" / "results" / "Nc16_fig5"
HEAT_DIR = CODE_ROOT / "data" / "raw" / "heat_input"

T_CASES = [
    (4.2, "4.2K"),
    (10.0, "10.0K"),
    (20.0, "20.0K"),
]

CONFIGS = [
    ("A1", 5, 5000),
    ("B1", 10, 5000),
    ("B2", 10, 1500),
    ("C1", 20, 5000),
    ("C2", 20, 1500),
    ("C3", 20, 100),
    ("C4", 20, 50),
    ("E1", 100, 5000),
    ("E2", 100, 1500),
    ("E3", 100, 100),
    ("E4", 100, 50),
    ("D1", 200, 5000),
    ("D2", 200, 1500),
    ("D3", 200, 100),
    ("D4", 200, 50),
]

def lead_rows(kind: str) -> list[list[str | None]]:
    if kind == "radial":
        model = "ARC_Nc16_radial_loss"
        table = "Data S2 full sweep"
    elif kind == "mag":
        model = "ARC_Nc16_magnetization_loss"
        table = "Data S3 full sweep"
    else:
        raise ValueError(kind)
    return [
        ["Model", model, None, None],
        ["Version", "COMSOL 6.4 (ARC Nc=16, Nt=700/850/1150, Ip=750/617.6/456.5, 2td window)", None, None],
        ["Date", datetime.now().strftime("%Y-%m-%d"), None, None],
        ["Table", table, None, None],
        [None, None, None, None],
    ]


HEADER = ["rho_turn (uΩ*cm^2)", "Npw", "Time (s)", "loss (W)"]


def t_token(T: float) -> str:
    return f"{T:g}"


def read_case(T: float, config: str, npw: int, rho: int) -> np.ndarray | None:
    path = RESULTS / f"ARC16_fig5_{config}_T{t_token(T)}_Npw{npw}_rho{rho}.csv"
    if not path.exists():
        return None
    data = np.loadtxt(path, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 3:
        raise ValueError(f"{path} has {data.shape[1]} columns; expected at least 3")
    return data[:, :3]


def build_rows(kind: str, strict: bool) -> dict[str, list[list[float]]]:
    if kind not in {"mag", "radial"}:
        raise ValueError(kind)
    value_col = 1 if kind == "mag" else 2
    by_sheet: dict[str, list[list[float]]] = {}
    missing: list[str] = []

    for T, sheet in T_CASES:
        rows: list[list[float]] = []
        for config, npw, rho in CONFIGS:
            data = read_case(T, config, npw, rho)
            if data is None:
                missing.append(f"{sheet} {config} Npw={npw} rho={rho}")
                continue
            for row in data:
                rows.append([rho, npw, float(row[0]), float(row[value_col])])
        by_sheet[sheet] = rows

    if missing and strict:
        joined = "\n  ".join(missing)
        raise FileNotFoundError(f"Missing archived Nc16 COMSOL CSVs:\n  {joined}")
    if missing:
        print("[warning] missing cases:")
        for item in missing:
            print(f"  {item}")

    return by_sheet


def sheet_df(rows: list[list[float]], kind: str) -> pd.DataFrame:
    return pd.DataFrame(lead_rows(kind) + [HEADER] + rows)


def backup(path: Path) -> None:
    if not path.exists():
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = path.with_name(f"{path.stem}.bak_data_s23_{stamp}{path.suffix}")
    shutil.copy2(path, dst)
    print(f"backup: {dst}")


def write_book(kind: str, path: Path, strict: bool) -> None:
    by_sheet = build_rows(kind, strict=strict)
    backup(path)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for _T, sheet in T_CASES:
            sheet_df(by_sheet[sheet], kind).to_excel(writer, sheet_name=sheet, header=False, index=False)
    print(f"wrote: {path}")


def verify(path: Path) -> None:
    for _T, sheet in T_CASES:
        df = pd.read_excel(path, sheet_name=sheet, header=0, skiprows=lambda x: x < 5)
        groups = df[[df.columns[0], df.columns[1]]].drop_duplicates()
        peak = df[df.columns[3]].max()
        print(f"  {path.name} [{sheet}]: rows={len(df)}, groups={len(groups)}, peak={peak:.3g} W")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-missing", action="store_true", help="write available rows instead of failing on missing cases")
    args = parser.parse_args()

    HEAT_DIR.mkdir(parents=True, exist_ok=True)
    strict = not args.allow_missing
    outputs = [
        ("mag", HEAT_DIR / "data_s3_magnetization_loss.xlsx"),
        ("radial", HEAT_DIR / "data_s2_radial_loss.xlsx"),
    ]
    for kind, path in outputs:
        write_book(kind, path, strict=strict)
        verify(path)


if __name__ == "__main__":
    main()



