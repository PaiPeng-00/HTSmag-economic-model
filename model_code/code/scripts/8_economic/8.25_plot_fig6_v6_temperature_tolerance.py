#!/usr/bin/env python3
"""Build Fig. 6A-F from audited V6-B through V6-E and frozen B1 results."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "v6_hts_temperature_tolerance_arc16pancake_nuc600"
B1 = ROOT / "outputs" / "target_price_window" / "manuscript_b1" / "tables" / "B1_robust_architecture_coverage.csv"
SENS_SUMMARY = OUT / "v6b_sensitivity_summary.csv"
SENS_AUDIT = OUT / "v6b_performance_sensitivity_audit.json"
C_BOUNDARY = OUT / "v6c_joint_tolerance_boundaries.csv"
C_AUDIT = OUT / "v6c_direct_bisection_audit.json"
D_COVERAGE = OUT / "v6d_tolerance_coverage.csv"
D_AUDIT = OUT / "v6d_sampling_audit.json"
E_AUDIT = OUT / "v6e_cross_scenario_audit.json"
FIGURE_DIR = OUT / "fig6_v6_temperature_tolerance"
BASE = FIGURE_DIR / "Fig6_temperature_tolerance"
SOURCE = FIGURE_DIR / "Fig6_temperature_tolerance_source_data.csv"
AUDIT = FIGURE_DIR / "Fig6_temperature_tolerance_audit.json"
COLORS = {4.2: "#0F4D92", 10.0: "#42949E", 20.0: "#B64342"}
TOPS = (4.2, 10.0, 20.0)

# Mandatory Python/vector-text settings.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 7
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.linewidth"] = .7
plt.rcParams["legend.frameon"] = False


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def label(ax, value: str) -> None:
    ax.text(-.16, 1.03, value, transform=ax.transAxes, fontweight="bold", fontsize=9, va="bottom")


def split_plot(ax, x: np.ndarray, y: np.ndarray, mask: np.ndarray, **kwargs) -> None:
    indices = np.flatnonzero(mask)
    if not len(indices):
        return
    breaks = np.flatnonzero(np.diff(indices) > 1) + 1
    for segment in np.split(indices, breaks):
        ax.plot(x[segment], y[segment], **kwargs)


def plot_boundary(ax, data: pd.DataFrame, criterion: str, linestyle: str, label_prefix: str) -> None:
    for top in TOPS:
        g = data.loc[data["Top_K"].eq(top)].sort_values("Npw")
        x = g["Npw"].to_numpy(float)
        y = pd.to_numeric(g[f"Rj_max_{criterion}_nOhm"], errors="coerce").to_numpy(float)
        low = g[f"low_reference_infeasible_{criterion}"].astype(bool).to_numpy()
        censored = g[f"right_censored_{criterion}"].astype(bool).to_numpy()
        finite = np.isfinite(y) & ~low & ~censored
        split_plot(ax, x, y, finite, color=COLORS[top], lw=1.05, ls=linestyle, label=f"{top:g} K {label_prefix}")
        ax.scatter(x[finite], y[finite], s=5, color=COLORS[top], linewidths=0, zorder=3)
        ax.scatter(x[censored], np.full(censored.sum(), 100.0), s=18, marker="^", facecolors="white", edgecolors=COLORS[top], linewidths=.8, zorder=4)


def main() -> None:
    required = (SENS_SUMMARY, SENS_AUDIT, C_BOUNDARY, C_AUDIT, D_COVERAGE, D_AUDIT, E_AUDIT, B1)
    if not all(path.exists() for path in required):
        raise FileNotFoundError("Fig. 6 audited inputs missing")
    if FIGURE_DIR.exists():
        raise FileExistsError("refusing to overwrite an existing Fig. 6 export directory")
    for path in (SENS_AUDIT, C_AUDIT, D_AUDIT, E_AUDIT):
        if json.loads(path.read_text(encoding="utf-8")).get("status") != "PASS":
            raise AssertionError(f"upstream audit is not PASS: {path.name}")
    sensitivity = pd.read_csv(SENS_SUMMARY)
    boundary = pd.read_csv(C_BOUNDARY)
    coverage = pd.read_csv(D_COVERAGE)
    b1 = pd.read_csv(B1)
    FIGURE_DIR.mkdir(parents=True)
    # Source-data package: only data plotted in panels, never raster-derived.
    source_parts = [
        sensitivity.assign(panel=lambda x: np.where(x.metric.eq("S_Rj_cryo"), "6A", "6B")),
        boundary[[c for c in boundary.columns if c in ("scenario", "Top_K", "coolant", "Npw") or "feas" in c or "temp_5pct" in c or "global_5pct" in c]].assign(panel="6C_or_6D"),
        coverage.loc[coverage.criterion.eq("temp_5pct")].assign(panel="6E"),
        b1.assign(panel="6F"),
    ]
    pd.concat(source_parts, ignore_index=True, sort=False).to_csv(SOURCE, index=False, encoding="utf-8", float_format="%.17g")

    fig, axes = plt.subplots(3, 2, figsize=(7.2, 8.0), constrained_layout=True, gridspec_kw={"height_ratios": [1, 1.12, 1]})
    ax_a, ax_b, ax_c, ax_d, ax_e, ax_f = axes.flat
    # A/B: quantile point distributions; n is model locations, not replicate experiments.
    for ax, metric, ylab, letter in ((ax_a, "S_Rj_cryo", r"$S_{R_j}^{\mathrm{cryo}}=|d\ln r_{\mathrm{cryo,re}}/d\ln R_j|$", "a"), (ax_b, "S_Rj_LCOE", r"$S_{R_j}^{\mathrm{LCOE}}=|d\ln\mathrm{LCOE}/d\ln R_j|$", "b")):
        frame = sensitivity.loc[sensitivity.metric.eq(metric)].sort_values("Top_K")
        for i, top in enumerate(TOPS):
            row = frame.loc[frame.Top_K.eq(top)].iloc[0]
            ax.vlines(i, row.p25, row.p75, color=COLORS[top], lw=5.5, alpha=.85)
            ax.vlines(i, row.p25, row.p75, color=COLORS[top], lw=1.1)
            ax.scatter(i, row["median"], s=28, color=COLORS[top], zorder=3)
            ax.scatter(i, row.p90, s=22, marker="D", facecolors="white", edgecolors=COLORS[top], linewidths=.85, zorder=3)
        ax.set_xticks(range(3), ["4.2 K", "10 K", "20 K"])
        ax.set_ylabel(ylab)
        ax.set_title("Joint-resistance sensitivity of refrigeration demand" if metric == "S_Rj_cryo" else "Joint-resistance sensitivity of LCOE", loc="left", fontsize=8)
        label(ax, letter)
        ax.text(.98, .96, "dot: median\nbar: IQR\nopen diamond: P90", transform=ax.transAxes, ha="right", va="top", fontsize=6.2)
    # C: physical feasibility limits.
    plot_boundary(ax_c, boundary, "feas", "-", "")
    ax_c.set_yscale("log"); ax_c.set_ylim(.9, 135); ax_c.set_xlim(1, 200)
    ax_c.set_xlabel(r"$N_{pw}$"); ax_c.set_ylabel(r"$R_{j,\max}^{\mathrm{feas}}$ (nOhm)")
    ax_c.set_title("Feasible joint-resistance limit", loc="left", fontsize=8); label(ax_c, "c")
    ax_c.text(.98, .05, "open triangle: right-censored\nat 100 nOhm", transform=ax_c.transAxes, ha="right", va="bottom", fontsize=6.1)
    ax_c.legend(loc="lower left", fontsize=6.2, ncol=3, handlelength=1.6)
    # D: within-temperature vs global commercial thresholds, both direct boundaries.
    plot_boundary(ax_d, boundary, "temp_5pct", "-", "temperature-relative")
    plot_boundary(ax_d, boundary, "global_5pct", "--", "global-relative")
    ax_d.set_yscale("log"); ax_d.set_ylim(.9, 135); ax_d.set_xlim(1, 200)
    ax_d.set_xlabel(r"$N_{pw}$"); ax_d.set_ylabel(r"$R_{j,\max}^{5\%}$ (nOhm)")
    ax_d.set_title("Near-optimal joint-resistance limit (S2)", loc="left", fontsize=8); label(ax_d, "d")
    ax_d.text(.02, .04, "solid: temperature-relative\ndashed: global-relative", transform=ax_d.transAxes, ha="left", va="bottom", fontsize=6.1)
    # E: named primary coverage and two definition sensitivity checks.
    frame = coverage.loc[coverage.criterion.eq("temp_5pct")]
    x = np.arange(3)
    main = [100 * float(frame.loc[(frame.Top_K.eq(t)) & frame.measure.eq("log_Rj") & frame.domain.eq("engineering_1_to_10_nOhm"), "coverage"].iloc[0]) for t in TOPS]
    linear = [100 * float(frame.loc[(frame.Top_K.eq(t)) & frame.measure.eq("linear_Rj") & frame.domain.eq("engineering_1_to_10_nOhm"), "coverage"].iloc[0]) for t in TOPS]
    stress = [100 * float(frame.loc[(frame.Top_K.eq(t)) & frame.measure.eq("log_Rj") & frame.domain.eq("stress_1_to_100_nOhm"), "coverage"].iloc[0]) for t in TOPS]
    ax_e.bar(x, main, color=[COLORS[t] for t in TOPS], width=.62, alpha=.82, label="log-Rj, [1, 10] nOhm")
    ax_e.plot(x, linear, color="#4D4D4D", lw=.8, marker="o", mfc="white", ms=4.4, label="linear-Rj, [1, 10] nOhm")
    ax_e.plot(x, stress, color="#767676", lw=.8, marker="s", mfc="white", ms=4.0, label="log-Rj, [1, 100] nOhm")
    ax_e.set_xticks(x, ["4.2 K", "10 K", "20 K"]); ax_e.set_ylim(0, 105)
    ax_e.set_ylabel(r"$C_{T,5\%}$ (%)"); ax_e.set_title("Temperature-resolved tolerance coverage (S2)", loc="left", fontsize=8); label(ax_e, "e")
    ax_e.legend(loc="lower left", fontsize=5.7, handlelength=1.2)
    # F: frozen B1 common-feasible commercial coverage, explicitly no regret terminology.
    b1 = b1.sort_values("tolerance_pct")
    ax_f.plot(b1.tolerance_pct, b1.coverage_pct, color="#0F4D92", lw=1.35)
    ax_f.scatter(b1.tolerance_pct, b1.coverage_pct, s=[28, 42, 28], color="#0F4D92", edgecolors="white", linewidths=.7, zorder=3)
    for row in b1.itertuples(index=False):
        ax_f.annotate(f"{row.coverage_pct:.1f}%", (row.tolerance_pct, row.coverage_pct), xytext=(3, 4), textcoords="offset points", fontsize=6.5)
    ax_f.set_xlim(0, 10.5); ax_f.set_ylim(50, 100)
    ax_f.set_xlabel("Allowed maximum relative LCOE penalty across scenarios (%)")
    ax_f.set_ylabel("Coverage of common-feasible architectures (%)")
    ax_f.set_title("Cross-scenario near-optimal coverage", loc="left", fontsize=8); label(ax_f, "f")
    fig.text(.5, .002, "All panels use frozen B1 anchor economics; A/B are local model elasticities, not statistical uncertainty.", ha="center", fontsize=6.3)
    for suffix, kwargs in ((".svg", {}), (".pdf", {}), (".tiff", {"dpi": 600}), ("_preview.png", {"dpi": 180})):
        fig.savefig(BASE.with_name(BASE.name + suffix), bbox_inches="tight", **kwargs)
    plt.close(fig)
    audit = {"figure": "Fig. 6A-F temperature-resolved joint-resistance tolerance", "status": "PASS", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
             "figure_contract": {"core_conclusion": "10 K has the broadest cross-scenario commercial tolerance; 20 K is internally more resistance-tolerant than 4.2 K but its commercial competitiveness is scenario-dependent.", "archetype": "quantitative grid", "backend": "Python", "final_width_mm": 183, "panels": {"A": "cryo local log elasticity quantiles", "B": "LCOE local log elasticity quantiles", "C": "S2 physical feasibility boundary", "D": "S2 temperature/global 5% boundaries", "E": "S2 coverage and measure/domain sensitivity", "F": "frozen B1 cross-scenario coverage"}, "statistics": "none; deterministic grid model and quantiles only", "review_risk": "right-censored points are marked and low-reference-infeasible points are not connected"},
             "inputs": {path.name: sha256(path) for path in required}, "outputs": {path.name: sha256(path) for path in (SOURCE, BASE.with_suffix('.svg'), BASE.with_suffix('.pdf'), BASE.with_suffix('.tiff'), BASE.with_name(BASE.name + '_preview.png'))}}
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
