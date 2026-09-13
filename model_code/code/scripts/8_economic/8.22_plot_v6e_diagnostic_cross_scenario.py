#!/usr/bin/env python3
"""Render V6-E cross-scenario diagnostic using corrected aggregation outputs.

Figure contract: quantitative grid.  Core conclusion: the cross-scenario
global-5-percent economic tolerance is widest at 10 K; 20 K remains broader
than 4.2 K in aggregate but loses S1 eligibility at the high-Npw end.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "v6_hts_temperature_tolerance_arc16pancake_nuc600"
COVERAGE = OUT / "v6e_cross_scenario_coverage.csv"
TOLERANCE = OUT / "v6e_cross_scenario_tolerance.csv"
PDF = OUT / "v6e_diagnostic_cross_scenario.pdf"
SVG = OUT / "v6e_diagnostic_cross_scenario.svg"
TIFF = OUT / "v6e_diagnostic_cross_scenario.tiff"
PREVIEW = OUT / "v6e_diagnostic_cross_scenario_preview.png"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 8
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.linewidth"] = .8
plt.rcParams["legend.frameon"] = False
COLORS = {4.2: "#0F4D92", 10.0: "#42949E", 20.0: "#B64342"}


def panel_label(ax, letter: str) -> None:
    ax.text(-.16, 1.03, letter, transform=ax.transAxes, fontweight="bold", fontsize=9, va="bottom")


def main() -> None:
    if not COVERAGE.exists() or not TOLERANCE.exists():
        raise FileNotFoundError("corrected V6-E cross-scenario outputs are missing")
    if any(path.exists() for path in (PDF, SVG, TIFF, PREVIEW)):
        raise FileExistsError("refusing to overwrite V6-E diagnostic exports")
    cov = pd.read_csv(COVERAGE)
    tol = pd.read_csv(TOLERANCE).drop_duplicates(["Top_K", "Npw"])
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), constrained_layout=True, gridspec_kw={"width_ratios": [1, 1, 1.3]})
    for ax, (domain, title) in zip(axes[:2], (("engineering_1_to_10_nOhm", "Engineering [1, 10] nΩ"), ("stress_1_to_100_nOhm", "Stress test [1, 100] nΩ"))):
        frame = cov.loc[(cov["measure"].eq("log_Rj")) & cov["domain"].eq(domain)].sort_values("Top_K")
        tops = (4.2, 10.0, 20.0)
        values = [float(frame.loc[frame["Top_K"].eq(top), "coverage_percent"].iloc[0]) for top in tops]
        bars = ax.bar(range(3), values, color=[COLORS[top] for top in tops], width=.66)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, value + 1.6, f"{value:.1f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(range(3), ["4.2 K", "10 K", "20 K"])
        ax.set_ylim(0, 108)
        ax.set_ylabel("Cross-scenario log-Rj coverage (%)")
        ax.set_title(title, fontsize=8, pad=5)
    ax = axes[2]
    for top in (4.2, 10.0, 20.0):
        frame = tol.loc[tol["Top_K"].eq(top)].sort_values("Npw")
        y = pd.to_numeric(frame["Rj_max_cross_global_5pct_nOhm"], errors="coerce").to_numpy(float)
        ax.plot(frame["Npw"], y, lw=1.25, color=COLORS[top], label=f"{top:g} K")
    ax.axvspan(181, 200, color="#B64342", alpha=.10, lw=0)
    ax.text(190.5, 75, "20 K: S1\nreference fails", ha="center", va="center", color="#B64342", fontsize=6.4)
    ax.set_yscale("log")
    ax.set_ylim(.9, 120)
    ax.set_xlim(1, 200)
    ax.set_xlabel("Npw")
    ax.set_ylabel("Cross Rj,max (nΩ)")
    ax.legend(loc="lower left", fontsize=6.6)
    for ax, letter in zip(axes, ("a", "b", "c")):
        panel_label(ax, letter)
    fig.text(.5, -.02, "Global-relative 5% LCOE criterion; equal weight for Npw = 1–200; no temperature interpolation", ha="center", fontsize=7)
    for path, kwargs in ((SVG, {}), (PDF, {}), (TIFF, {"dpi": 600}), (PREVIEW, {"dpi": 180})):
        fig.savefig(path, bbox_inches="tight", **kwargs)
    plt.close(fig)
    print(f"Rendered {PDF.name}, {SVG.name}, {TIFF.name}, and {PREVIEW.name}")


if __name__ == "__main__":
    main()
