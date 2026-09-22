"""Render the V10, LCOE-only three-panel Figure 4 (file asset Fig3)."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[5]
DATA = ROOT / "results/figure_inputs/current/main_figures/Fig4"
PALETTE = json.loads((ROOT / "figures/publication_figures/figures/Fig2/source/raw/PALETTE_B_DEEP.json").read_text(encoding="utf-8"))
COLORS = {4.2: PALETTE["TEMP_4P2K"], 10.0: PALETTE["TEMP_10K"], 20.0: PALETTE["TEMP_20K"]}
CN = os.environ.get("FIG4_CN") == "1"
FONT = "Times New Roman" if CN else "Arial"
MM = 25.4
W, H = 166, 135
mpl.rcParams.update({"font.family": [FONT, "SimSun"] if CN else FONT,
                     "font.sans-serif": [FONT, "SimSun", "Arial"],
                     "font.size": 7, "svg.fonttype": "none", "pdf.fonttype": 42,
                     "axes.linewidth": .48, "xtick.direction": "in", "ytick.direction": "in"})


def label(en: str, zh: str) -> str:
    return zh if CN else en


def axmm(fig, l, t, w, h):
    return fig.add_axes([l/W, (H-t-h)/H, w/W, h/H])


def style(ax):
    for side in ("left", "right", "top", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(.48)
        ax.spines[side].set_color("#303030")
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True,
                   labelsize=6.7, length=2.5, width=.48, pad=2)


def header(fig, y, letter, en, zh, x=11, size=7.7):
    fig.text(x/W, 1-y/H, letter, ha="right", va="top", fontsize=8.6, weight="bold")
    fig.text((x+2.5)/W, 1-y/H, label(en, zh), ha="left", va="top", fontsize=size, weight="bold")


def load_csv(name):
    with (DATA / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def draw_range(ax, y, lo, hi, color, width=.72, alpha=.55):
    if lo < hi:
        ax.hlines(y, lo, hi, color=color, lw=width, alpha=alpha, zorder=2)


def render():
    refs = load_csv("panel_A_scenario_lcoe_reference.csv")
    stats = {float(r["temperature_K"]): r for r in load_csv("panel_B_temperature_penalty_statistics.csv")}
    cdf = load_csv("panel_C_temperature_retention_ecdf.csv")
    fig = plt.figure(figsize=(W/MM, H/MM), facecolor="white")
    header(fig, 4, "A", "Scenario-specific LCOE reference", "各情景 LCOE 参考值")
    header(fig, 4, "B", "Maximum cross-scenario LCOE penalty", "跨情景最大 LCOE 增幅", x=88, size=7.3)
    header(fig, 70, "C", "Designs within LCOE margin", "LCOE 余量内的磁体设计")
    # A: the three fixed scenario-specific references, with 10% and 20%
    # absolute-LCOE margins. Grayscale deliberately avoids temperature coding.
    ax_a = axmm(fig, 22, 17, 56, 40)
    ax_a.set_yscale("log")
    ax_a.set_xlim(.5, 3.5)
    ax_a.set_ylim(65, 1750)
    style(ax_a)
    ax_a.set_xticks([1, 2, 3], ["S1", "S2", "S3"])
    ax_a.set_yticks([80, 100, 200, 500, 1000, 1500], ["80", "100", "200", "500", "1000", "1500"])
    ax_a.minorticks_off()
    ax_a.set_xlabel(label("Scenario", "情景"), fontsize=7.25)
    ax_a.set_ylabel(label(r"LCOE (US\$ MWh$^{-1}$)", "LCOE（美元/MWh）"), fontsize=7.25)
    for x, row in enumerate(refs, 1):
        minimum = float(row["minimum_LCOE_USD_per_MWh"])
        margin10 = float(row["LCOE_at_10pct_margin_USD_per_MWh"])
        margin20 = float(row["LCOE_at_20pct_margin_USD_per_MWh"])
        ax_a.vlines(x, minimum, margin20, color="#B7B7B7", lw=.65, zorder=1)
        ax_a.hlines(margin20, x-.17, x+.17, color="#A9A9A9", lw=1.1, zorder=2)
        ax_a.hlines(margin10, x-.17, x+.17, color="#606060", lw=1.25, zorder=3)
        ax_a.plot(x, minimum, "o", color="#222222", ms=4.25, zorder=4)
    handles = [Line2D([], [], marker="o", ls="", color="#222222", markersize=3.5),
               Line2D([], [], color="#606060", lw=1.3),
               Line2D([], [], color="#A9A9A9", lw=1.3)]
    fig.legend(handles, [label("Minimum", "最低值"), "+10%", "+20%"],
               loc="upper center", bbox_to_anchor=(50/W, 1-10/H),
               ncol=3, frameon=False, fontsize=6, handlelength=1.35,
               columnspacing=1.0, handletextpad=.4, borderaxespad=0)

    # B: match Fig. 3D's point / thick-interval / thin-interval language.
    # The thin line spans the complete finite valid-LCOE range.
    handles_b = [Line2D([], [], marker="o", ls="", color="black", markersize=3.4),
                 Line2D([], [], color="black", lw=2),
                 Line2D([], [], color="black", lw=.65, alpha=.6)]
    fig.legend(handles_b, [label("Median", "中位数"), "P10–P90", label("Full range", "完整范围")],
               loc="upper center", bbox_to_anchor=(125/W, 1-10/H),
               ncol=3, frameon=False, fontsize=6, handlelength=1.6,
               columnspacing=1.0, handletextpad=.4, borderaxespad=0)
    ax_b = axmm(fig, 94, 17, 62, 40)
    ax_b.set_xscale("log")
    ax_b.set_xlim(.3, 100000)
    ax_b.set_ylim(.5, 3.5)
    style(ax_b)
    ax_b.set_yticks([3, 2, 1], ["4.2 K", "10 K", "20 K"])
    ax_b.set_xticks([1, 10, 100, 1000, 10000, 100000],
                   ["1", "10", r"$10^2$", r"$10^3$", r"$10^4$", r"$10^5$"])
    ax_b.minorticks_off()
    for temp, y in ((4.2, 3), (10.0, 2), (20.0, 1)):
        row = stats[temp]
        lo, p10, med, p90, hi = (float(row[k]) for k in (
            "minimum_percent", "P10_percent", "median_percent", "P90_percent", "maximum_percent"))
        assert .3 < lo <= p10 <= med <= p90 <= hi < 100000
        color = COLORS[temp]
        draw_range(ax_b, y, lo, hi, color)
        ax_b.hlines(y, p10, p90, color=color, lw=2/1.115, zorder=3)
        ax_b.plot(med, y, "o", ms=5.4/1.115, color=color, markeredgecolor="white",
                  markeredgewidth=.7/1.115, zorder=4)
    ax_b.set_xlabel(label("Maximum LCOE increase across S1–S3 (%)", "S1–S3 中最大 LCOE 增幅（%）"),
                    fontsize=6.7)

    ax_c = axmm(fig, 27, 83, 129, 40)
    style(ax_c)
    ax_c.set(xlim=(0, 20), ylim=(0, 101))
    ax_c.set_xticks([0, 5, 10, 15, 20])
    ax_c.set_yticks([0, 20, 40, 60, 80, 100])
    ax_c.set_xlabel(label("Allowed LCOE increase (%)", "允许的 LCOE 增幅（%）"), fontsize=7.25)
    ax_c.set_ylabel(label("Fraction of designs within LCOE margin (%)", "LCOE 余量内的设计比例（%）"),
                    fontsize=6.6)
    for temp in (4.2, 10.0, 20.0):
        rows = [r for r in cdf if float(r["temperature_K"]) == temp]
        x = np.array([float(r["allowed_LCOE_increase_percent"]) for r in rows])
        y = np.array([float(r["economic_retention_percent"]) for r in rows])
        ax_c.step(x, y, where="post", color=COLORS[temp], lw=1.8, label=f"{temp:g} K")
    ax_c.axvline(10, color="#9A9A9A", lw=.6, ls=(0, (2, 2)), zorder=1)
    ax_c.legend(loc="lower right", frameon=False, fontsize=7.6, handlelength=2.4,
              borderpad=.2, labelspacing=.35)
    out = ROOT / "manuscript/current" / ("formal_figures_cn" if CN else "formal_figures")
    out.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "svg", "png"):
        fig.savefig(out / f"Fig3.{extension}", dpi=600 if extension == "png" else None)
    plt.close(fig)
    print(out / "Fig3.pdf")


if __name__ == "__main__":
    render()
