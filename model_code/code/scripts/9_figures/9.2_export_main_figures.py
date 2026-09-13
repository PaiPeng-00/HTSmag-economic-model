"""Export the six labelled main-text figures as paired SVG and PDF files."""

from __future__ import annotations

import shutil
from pathlib import Path

import cairosvg


ROOT = Path(__file__).resolve().parents[4]
MASTER = ROOT / "code-model" / "figures"
FINAL = ROOT / "17-提交版本-SA" / "figures_final"
SOURCES = {
    "Fig1": MASTER / "Fig1_availability.svg",
    "Fig2": MASTER / "Fig2_heatload.svg",
    "Fig3": MASTER / "Fig3_heatload_joint.svg",
    "Fig4": MASTER / "Fig4_recirc_fraction.svg",
    "Fig5": MASTER / "Fig5_deltaLCOE.svg",
    "Fig6": MASTER / "Fig6_HTS_price.svg",
}


def main() -> None:
    FINAL.mkdir(parents=True, exist_ok=True)
    for stem, source in SOURCES.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        svg = FINAL / f"{stem}.svg"
        pdf = FINAL / f"{stem}.pdf"
        shutil.copyfile(source, svg)
        cairosvg.svg2pdf(url=str(svg), write_to=str(pdf))
        print(f"{stem}: {svg.name}, {pdf.name}")


if __name__ == "__main__":
    main()
