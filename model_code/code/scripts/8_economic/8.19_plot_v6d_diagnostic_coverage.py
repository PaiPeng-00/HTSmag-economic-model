#!/usr/bin/env python3
"""Render the V6-D coverage diagnostic from the audited coverage CSV.

Figure contract: quantitative grid; the hero evidence is the temperature
ordering across the two explicitly named Rj domains.  The companion panel
shows the numerical effect of replacing stored-grid endpoints with direct
log-bisection boundaries.  It contains no experimental statistics.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "v6_hts_temperature_tolerance_arc16pancake_nuc600"
SOURCE = OUT / "v6d_tolerance_coverage.csv"
SENSITIVITY = OUT / "v6d_measure_sensitivity.csv"
PDF = OUT / "v6d_diagnostic_coverage.pdf"
SVG = OUT / "v6d_diagnostic_coverage.svg"
TIFF = OUT / "v6d_diagnostic_coverage.tiff"
PREVIEW = OUT / "v6d_diagnostic_coverage_preview.png"

# Mandatory editable-text settings for the selected Python backend.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 8
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["legend.frameon"] = False

COLORS = {4.2: "#0F4D92", 10.0: "#42949E", 20.0: "#B64342"}


def label(ax, text: str) -> None:
    ax.text(-0.13, 1.03, text, transform=ax.transAxes, fontweight="bold", fontsize=9, va="bottom")


def main() -> None:
    if not SOURCE.exists() or not SENSITIVITY.exists():
        raise FileNotFoundError("Run V6-D coverage calculation before its diagnostic figure")
    if any(path.exists() for path in (PDF, SVG, TIFF, PREVIEW)):
        raise FileExistsError("refusing to overwrite V6-D diagnostic exports")
    coverage = pd.read_csv(SOURCE)
    sensitivity = pd.read_csv(SENSITIVITY)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75), constrained_layout=True)
    domains = [("engineering_1_to_10_nOhm", "Engineering [1, 10] nΩ"),
               ("stress_1_to_100_nOhm", "Stress test [1, 100] nΩ")]
    for ax, (domain, title) in zip(axes, domains):
        frame = coverage.loc[(coverage["criterion"].eq("temp_5pct")) & coverage["measure"].eq("log_Rj") & coverage["domain"].eq(domain)]
        x = np.arange(3)
        values = [float(frame.loc[frame["Top_K"].eq(top), "coverage_percent"].iloc[0]) for top in (4.2, 10.0, 20.0)]
        bars = ax.bar(x, values, color=[COLORS[top] for top in (4.2, 10.0, 20.0)], width=0.66)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 1.5, f"{value:.1f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x, ["4.2 K", "10 K", "20 K"])
        ax.set_ylim(0, 108)
        ax.set_ylabel("Log-Rj coverage (%)")
        ax.set_title(title, fontsize=8, pad=5)
        ax.axhline(0, color="#4D4D4D", lw=.5)
    label(axes[0], "a")
    label(axes[1], "b")
    fig.text(.5, -.02, "Temperature-relative 5% LCOE boundary; equal weight for Npw = 1–200", ha="center", fontsize=7)
    for path, kwargs in ((SVG, {}), (PDF, {}), (TIFF, {"dpi": 600}), (PREVIEW, {"dpi": 180})):
        fig.savefig(path, bbox_inches="tight", **kwargs)
    plt.close(fig)
    print(f"Rendered {PDF.name}, {SVG.name}, {TIFF.name}, and {PREVIEW.name}")


if __name__ == "__main__":
    main()
