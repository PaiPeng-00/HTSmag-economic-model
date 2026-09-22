"""Render Supplementary Fig. S7 from exact V10 finite-LCOE survival counts."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
DATA = ROOT / "results/figure_inputs/current/main_figures/Fig4"
PALETTE = json.loads((ROOT / "figures/publication_figures/figures/Fig2/source/raw/PALETTE_B_DEEP.json").read_text(encoding="utf-8"))
COLORS = {4.2: PALETTE["TEMP_4P2K"], 10.0: PALETTE["TEMP_10K"], 20.0: PALETTE["TEMP_20K"]}
CN = os.environ.get("FIG4_CN") == "1"
FONT = "Times New Roman" if CN else "Arial"
mpl.rcParams.update({"font.family": [FONT, "SimSun"] if CN else FONT,
                     "font.sans-serif": [FONT, "SimSun", "Arial"],
                     "font.size": 7, "svg.fonttype": "none", "pdf.fonttype": 42,
                     "axes.linewidth": .48, "xtick.direction": "in", "ytick.direction": "in"})


def label(en: str, zh: str) -> str:
    return zh if CN else en


def render() -> None:
    with (DATA / "supplementary_figS7_full_penalty_survival.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    with (DATA / "panel_B_temperature_penalty_statistics.csv").open(encoding="utf-8", newline="") as stream:
        stats = {float(r["temperature_K"]): r for r in csv.DictReader(stream)}
    fig, ax = plt.subplots(figsize=(166/25.4, 86/25.4))
    fig.subplots_adjust(left=.13, right=.97, bottom=.23, top=.94)
    for temp in (4.2, 10.0, 20.0):
        selected = [r for r in rows if float(r["temperature_K"]) == temp and int(r["valid_realizations_at_or_above_threshold_n"]) > 0]
        x = np.array([float(r["LCOE_increase_threshold_percent"]) for r in selected])
        y = np.array([float(r["fraction_of_valid_realizations_at_or_above_percent"]) for r in selected])
        ax.step(x, y, where="post", color=COLORS[temp], lw=1.55, label=f"{temp:g} K")
        max_x = float(stats[temp]["maximum_percent"])
        min_fraction = 100 / int(stats[temp]["valid_all_scenarios_n"])
        ax.plot(max_x, min_fraction, "o", ms=2.8, color=COLORS[temp], clip_on=False)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(.3, 1.2e5)
    ax.set_ylim(3e-5, 120)
    ax.set_xticks([1, 10, 100, 1000, 10000, 100000], ["1", "10", "100", "1,000", "10,000", "100,000"])
    ax.set_yticks([.0001, .001, .01, .1, 1, 10, 100], ["0.0001", "0.001", "0.01", "0.1", "1", "10", "100"])
    ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=6.7, length=2.5, width=.48)
    ax.set_xlabel(label("LCOE increase threshold (%)", "LCOE 增幅阈值（%）"), fontsize=7.25)
    ax.set_ylabel(label("Valid designs at or above threshold (%)", "达到或超过阈值的有效设计比例（%）"), fontsize=7.1)
    ax.legend(loc="upper right", frameon=False, fontsize=7.6, handlelength=2.4)
    out = ROOT / "manuscript/current" / ("formal_figures_cn" if CN else "formal_figures")
    out.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "svg", "png"):
        fig.savefig(out / f"FigS7.{extension}", dpi=600 if extension == "png" else None)
    plt.close(fig)
    print(out / "FigS7.pdf")


if __name__ == "__main__":
    render()
