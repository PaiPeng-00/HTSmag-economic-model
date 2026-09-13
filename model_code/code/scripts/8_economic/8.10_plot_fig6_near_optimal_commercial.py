#!/usr/bin/env python3
"""Build and plot the V5-B1 near-optimal commercial-performance Fig. 6.

The full B1 grid is read in chunks so the script can be rerun without loading the
1.55-GB Data S1 CSV into memory.  The physical screen is intentionally explicit:
AF_ref >= 0.99, r_cryo_re <= 0.50, and finite B1 LCOE.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
import numpy as np
import pandas as pd


SCENARIOS = ("S1", "S2", "S3")
ARCH_KEYS = ["Npw", "R_joint_nOhm"]
LCOE = "lcoe_anchor_USD_per_MWh"
PRIMARY_AF = 0.99
PRIMARY_RCRYO = 0.50
USECOLS = [
    "scenario",
    "Npw",
    "R_joint_nOhm",
    "rho_turn_uOhm_cm2",
    "Top_K",
    "coolant",
    "AF_ref",
    "r_cryo_re_fraction",
    LCOE,
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def aggregate_architectures(input_csv: Path, chunksize: int = 200_000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return arch_opt_by_scenario and arch_cross_scenario from the B1 grid."""

    count_parts: list[pd.DataFrame] = []
    best_parts: list[pd.DataFrame] = []
    all_pairs: set[tuple[int, float]] = set()

    dtype = {
        "scenario": "string",
        "Npw": "int64",
        "R_joint_nOhm": "float64",
        "rho_turn_uOhm_cm2": "float64",
        "Top_K": "float64",
        "coolant": "string",
        "AF_ref": "float64",
        "r_cryo_re_fraction": "float64",
        LCOE: "float64",
    }
    for chunk in pd.read_csv(input_csv, usecols=USECOLS, dtype=dtype, chunksize=chunksize):
        chunk = chunk.loc[chunk["scenario"].isin(SCENARIOS)].copy()
        all_pairs.update(zip(chunk["Npw"].astype(int), chunk["R_joint_nOhm"].astype(float)))
        finite = np.isfinite(chunk[LCOE].to_numpy(dtype=float))
        mask = (
            (chunk["AF_ref"] >= PRIMARY_AF)
            & (chunk["r_cryo_re_fraction"] <= PRIMARY_RCRYO)
            & finite
            & chunk[LCOE].notna()
        )
        feasible = chunk.loc[mask].copy()
        if feasible.empty:
            continue
        counts = (
            feasible.groupby(["scenario", *ARCH_KEYS], sort=False)
            .size()
            .rename("n_feasible_realizations")
            .reset_index()
        )
        count_parts.append(counts)
        best_parts.append(
            feasible.sort_values(["scenario", *ARCH_KEYS, LCOE], kind="mergesort")
            .drop_duplicates(["scenario", *ARCH_KEYS], keep="first")
            [["scenario", *ARCH_KEYS, "rho_turn_uOhm_cm2", "Top_K", "coolant", LCOE]]
        )

    if not all_pairs:
        raise RuntimeError("No architecture keys were found in the input grid")
    pairs = pd.DataFrame(sorted(all_pairs), columns=ARCH_KEYS)
    grid = pd.MultiIndex.from_product([SCENARIOS, pairs["Npw"].unique(), pairs["R_joint_nOhm"].unique()], names=["scenario", *ARCH_KEYS]).to_frame(index=False)
    # The B1 grid is rectangular, but retain this filter if a future release is not.
    grid = grid.merge(pairs.assign(_present=1), on=ARCH_KEYS, how="inner").drop(columns="_present")

    if count_parts:
        counts_all = pd.concat(count_parts, ignore_index=True).groupby(["scenario", *ARCH_KEYS], as_index=False)["n_feasible_realizations"].sum()
    else:
        counts_all = pd.DataFrame(columns=["scenario", *ARCH_KEYS, "n_feasible_realizations"])
    if best_parts:
        best_all = (
            pd.concat(best_parts, ignore_index=True)
            .sort_values(["scenario", *ARCH_KEYS, LCOE], kind="mergesort")
            .drop_duplicates(["scenario", *ARCH_KEYS], keep="first")
        )
    else:
        best_all = pd.DataFrame(columns=["scenario", *ARCH_KEYS, "rho_turn_uOhm_cm2", "Top_K", "coolant", LCOE])

    opt = grid.merge(counts_all, on=["scenario", *ARCH_KEYS], how="left").merge(best_all, on=["scenario", *ARCH_KEYS], how="left")
    opt["n_feasible_realizations"] = opt["n_feasible_realizations"].fillna(0).astype(int)
    opt["feasible_any"] = (opt["n_feasible_realizations"] > 0).astype(int)
    opt = opt.rename(
        columns={
            LCOE: "best_LCOE",
            "rho_turn_uOhm_cm2": "best_rho_turn_uohm_cm2",
            "Top_K": "best_operating_temperature_K",
            "coolant": "best_coolant",
        }
    )
    scenario_min = opt.loc[opt["feasible_any"].eq(1)].groupby("scenario")["best_LCOE"].min().rename("scenario_min_LCOE")
    opt = opt.merge(scenario_min, on="scenario", how="left")
    opt["rel_penalty_pct"] = np.where(
        opt["feasible_any"].eq(1),
        100.0 * (opt["best_LCOE"] - opt["scenario_min_LCOE"]) / opt["scenario_min_LCOE"],
        np.nan,
    )
    opt = opt[[
        "scenario", "Npw", "R_joint_nOhm", "feasible_any", "n_feasible_realizations",
        "best_LCOE", "best_rho_turn_uohm_cm2", "best_operating_temperature_K", "best_coolant",
        "scenario_min_LCOE", "rel_penalty_pct",
    ]].sort_values(["scenario", *ARCH_KEYS], kind="mergesort").reset_index(drop=True)

    cross = pairs.copy()
    for scenario in SCENARIOS:
        sub = opt.loc[opt["scenario"].eq(scenario), ARCH_KEYS + ["feasible_any", "rel_penalty_pct"]].rename(
            columns={"feasible_any": f"feasible_any_{scenario}", "rel_penalty_pct": f"rel_penalty_pct_{scenario}"}
        )
        cross = cross.merge(sub, on=ARCH_KEYS, how="left")
    feasible_cols = [f"feasible_any_{s}" for s in SCENARIOS]
    penalty_cols = [f"rel_penalty_pct_{s}" for s in SCENARIOS]
    cross["n_feasible_scenarios"] = cross[feasible_cols].sum(axis=1).astype(int)
    cross["common_feasible"] = cross["n_feasible_scenarios"].eq(3).astype(int)
    cross["max_rel_penalty_pct"] = np.where(cross["common_feasible"].eq(1), cross[penalty_cols].max(axis=1), np.nan)
    s1, s2, s3 = penalty_cols
    cross["binding_scenario"] = "Not common-feasible"
    common = cross["common_feasible"].eq(1)
    # >= implements the mandated stable tie-break S1 > S2 > S3.
    cross.loc[common & (cross[s1] >= cross[s2]) & (cross[s1] >= cross[s3]), "binding_scenario"] = "S1"
    cross.loc[common & (cross[s1] < cross[s2]) & (cross[s2] >= cross[s3]), "binding_scenario"] = "S2"
    cross.loc[common & (cross[s1] < cross[s2]) & (cross[s2] < cross[s3]), "binding_scenario"] = "S3"
    cross["nearopt_class"] = "Not common-feasible"
    cross.loc[common & (cross["max_rel_penalty_pct"] <= 1), "nearopt_class"] = "≤1%"
    cross.loc[common & (cross["max_rel_penalty_pct"] > 1) & (cross["max_rel_penalty_pct"] <= 5), "nearopt_class"] = "1–5%"
    cross.loc[common & (cross["max_rel_penalty_pct"] > 5) & (cross["max_rel_penalty_pct"] <= 10), "nearopt_class"] = "5–10%"
    cross.loc[common & (cross["max_rel_penalty_pct"] > 10), "nearopt_class"] = ">10%"
    cross = cross.sort_values(ARCH_KEYS, kind="mergesort").reset_index(drop=True)
    expected = {"common": 1054, "le1": 621, "le5": 886, "le10": 1020}
    observed = {
        "common": int(cross["common_feasible"].sum()),
        "le1": int((cross["common_feasible"].eq(1) & (cross["max_rel_penalty_pct"] <= 1)).sum()),
        "le5": int((cross["common_feasible"].eq(1) & (cross["max_rel_penalty_pct"] <= 5)).sum()),
        "le10": int((cross["common_feasible"].eq(1) & (cross["max_rel_penalty_pct"] <= 10)).sum()),
    }
    if observed != expected:
        raise RuntimeError(f"Fig. 6 hard-count check failed: observed={observed}, expected={expected}")
    return opt, cross


def style_axis(ax: plt.Axes, *, xlabel: bool = True, ylabel: bool = True) -> None:
    ax.set_xscale("log")
    ax.set_xlim(0.95, 105)
    ax.set_ylim(0.5, 205)
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100])
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_yticks([1, 5, 10, 20, 50, 100, 200])
    ax.tick_params(axis="both", labelsize=6.5, width=0.55, length=3)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)
    if xlabel:
        ax.set_xlabel(r"Joint resistance, $R_j$ (nΩ)", fontsize=7.2)
    if ylabel:
        ax.set_ylabel(r"Parallel tapes per turn, $N_{pw}$", fontsize=7.2)


def add_letter(ax: plt.Axes, letter: str) -> None:
    ax.text(-0.17, 1.08, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")


def plot_fig6(cross: pd.DataFrame, output: Path) -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    all_points = cross
    common = cross.loc[cross["common_feasible"].eq(1)].copy()
    colors = {
        "S1": "#3E5C9A", "S2": "#2A9D8F", "S3": "#C79A2B", "Not common-feasible": "#D9D9D9",
        "≤1%": "#F7FBFF", "1–5%": "#BDD7E7", "5–10%": "#6BAED6", ">10%": "#2171B5",
    }
    cmap = LinearSegmentedColormap.from_list("fig6_blue", ["#F7FBFF", "#D9EAF7", "#9ECAE1", "#3182BD", "#08519C"])
    norm = Normalize(vmin=0, vmax=10, clip=True)
    fig = plt.figure(figsize=(178 / 25.4, 158 / 25.4), facecolor="white")
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.82], hspace=0.50, wspace=0.30)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[2, :])

    # A: all architecture points first, then common-feasible heatmap.
    ax_a.scatter(all_points.R_joint_nOhm, all_points.Npw, s=14, marker="s", c="#D9D9D9", linewidths=0, zorder=1)
    ax_a.scatter(common.R_joint_nOhm, common.Npw, s=15, marker="s", c=common.max_rel_penalty_pct.clip(upper=10), cmap=cmap, norm=norm, linewidths=0, zorder=2)
    xvals = np.sort(cross.R_joint_nOhm.unique())
    yvals = np.sort(cross.Npw.unique())
    z = common.pivot(index="Npw", columns="R_joint_nOhm", values="max_rel_penalty_pct").reindex(index=yvals, columns=xvals).to_numpy(dtype=float)
    X, Y = np.meshgrid(xvals, yvals)
    contour = ax_a.contour(X, Y, z, levels=[1, 5, 10], colors=["#1f4e79"] * 3, linestyles=["--", "-", "-."], linewidths=[0.8, 1.6, 0.8], zorder=3)
    ax_a.clabel(contour, fmt={1: "1%", 5: "5%", 10: "10%"}, inline=True, fontsize=5.8, inline_spacing=2)
    ax_a.text(0.04, 0.94, "84.1% within 5%", transform=ax_a.transAxes, fontsize=6.4, va="top", color="#17365d", bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5))
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_a, fraction=0.050, pad=0.035, aspect=18)
    cbar.set_ticks([0, 1, 5, 10])
    cbar.set_ticklabels(["0", "1", "5", ">10"])
    cbar.ax.tick_params(labelsize=5.8, width=0.45, length=2)
    cbar.set_label("Maximum relative LCOE penalty (%)", fontsize=6.3, labelpad=4)
    style_axis(ax_a)
    ax_a.set_title("Maximum relative LCOE penalty across scenarios", fontsize=7.4, loc="left", pad=4)
    add_letter(ax_a, "A")

    # B: binding scenario.
    ax_b.scatter(all_points.R_joint_nOhm, all_points.Npw, s=14, marker="s", c="#D9D9D9", linewidths=0)
    for scenario in ("S1", "S2", "S3"):
        sub = common.loc[common.binding_scenario.eq(scenario)]
        ax_b.scatter(sub.R_joint_nOhm, sub.Npw, s=15, marker="s", c=colors[scenario], linewidths=0, label=scenario)
    style_axis(ax_b, ylabel=False)
    ax_b.set_title("Scenario setting the maximum penalty", fontsize=7.4, loc="left", pad=4)
    ax_b.legend(handles=[Line2D([0], [0], marker="s", color="none", markerfacecolor=colors[s], markersize=5, label=s) for s in ("S1", "S2", "S3")] + [Line2D([0], [0], marker="s", color="none", markerfacecolor=colors["Not common-feasible"], markersize=5, label="Not common-feasible")], frameon=False, fontsize=5.8, loc="upper right", borderpad=0.2, handletextpad=0.3, labelspacing=0.25)
    add_letter(ax_b, "B")

    # C: near-optimal classes.
    ax_c.scatter(all_points.R_joint_nOhm, all_points.Npw, s=14, marker="s", c="#D9D9D9", linewidths=0)
    class_order = ["≤1%", "1–5%", "5–10%", ">10%"]
    for category in class_order:
        sub = common.loc[common.nearopt_class.eq(category)]
        ax_c.scatter(sub.R_joint_nOhm, sub.Npw, s=15, marker="s", c=colors[category], linewidths=0, label=category)
    style_axis(ax_c)
    ax_c.set_title("Near-optimality classes", fontsize=7.4, loc="left", pad=4)
    ax_c.legend(handles=[Line2D([0], [0], marker="s", color="none", markerfacecolor=colors[c], markersize=5, label=c) for c in class_order] + [Line2D([0], [0], marker="s", color="none", markerfacecolor=colors["Not common-feasible"], markersize=5, label="Not common-feasible")], frameon=False, fontsize=5.8, loc="upper right", borderpad=0.2, handletextpad=0.3, labelspacing=0.25)
    add_letter(ax_c, "C")

    # D: number of scenarios with a feasible realization.
    d_colors = {0: "#FFFFFF", 1: "#D9D9D9", 2: "#9ECAE1", 3: "#3182BD"}
    for number in (0, 1, 2, 3):
        sub = all_points.loc[all_points.n_feasible_scenarios.eq(number)]
        ax_d.scatter(sub.R_joint_nOhm, sub.Npw, s=15, marker="s", c=d_colors[number], edgecolors="#777777" if number == 0 else "none", linewidths=0.25, label=str(number))
    style_axis(ax_d, ylabel=False)
    ax_d.set_title("Number of scenarios with a feasible realization", fontsize=7.4, loc="left", pad=4)
    ax_d.legend(title="Feasible in n scenarios", handles=[Line2D([0], [0], marker="s", color="none", markerfacecolor=d_colors[n], markeredgecolor="#777777" if n == 0 else "none", markersize=5, label=str(n)) for n in (0, 1, 2, 3)], frameon=False, fontsize=5.8, title_fontsize=5.8, loc="upper right", borderpad=0.2, handletextpad=0.3, labelspacing=0.25)
    add_letter(ax_d, "D")

    # E: cumulative coverage among the 1,054 common-feasible architectures.
    penalties = np.sort(common.max_rel_penalty_pct.to_numpy(dtype=float))
    threshold = np.linspace(0, 10, 1001)
    coverage = np.searchsorted(penalties, threshold, side="right") / len(penalties) * 100
    ax_e.plot(threshold, coverage, color="#08519C", lw=1.5)
    points = {1: 100 * 621 / 1054, 5: 100 * 886 / 1054, 10: 100 * 1020 / 1054}
    for x, y in points.items():
        ax_e.scatter([x], [y], s=28 if x == 5 else 18, color="#08519C" if x == 5 else "#3182BD", edgecolors="white", linewidths=0.6, zorder=3)
        label = f"{y:.1f}%" + ("\n886 / 1,054" if x == 5 else "")
        ax_e.annotate(label, (x, y), xytext=(5, 5), textcoords="offset points", fontsize=6.2, color="#17365d", va="bottom")
    ax_e.set_xlim(0, 10)
    ax_e.set_ylim(0, 100)
    ax_e.set_xticks([0, 1, 2, 5, 7.5, 10])
    ax_e.set_xlabel("Allowed maximum relative LCOE penalty (%)", fontsize=7.2)
    ax_e.set_ylabel("Coverage of common-feasible architectures (%)", fontsize=7.2)
    ax_e.tick_params(axis="both", labelsize=6.5, width=0.55, length=3)
    ax_e.spines["top"].set_visible(False)
    ax_e.spines["right"].set_visible(False)
    ax_e.set_title("Coverage of common-feasible architectures", fontsize=7.4, loc="left", pad=4)
    add_letter(ax_e, "E")

    fig.suptitle("Near-optimal commercial performance across magnet architectures", fontsize=8.2, y=0.995)
    fig.subplots_adjust(left=0.085, right=0.965, bottom=0.075, top=0.955)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".svg"))
    fig.savefig(output.with_suffix(".pdf"))
    fig.savefig(output.with_suffix(".png"), dpi=600)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    opt, cross = aggregate_architectures(args.input)
    table_dir = args.output / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    opt.to_csv(table_dir / "arch_opt_by_scenario.csv", index=False)
    cross.to_csv(table_dir / "arch_cross_scenario.csv", index=False)
    plot_fig6(cross, args.output / "Fig6_near_optimal_commercial")
    audit = {
        "input": str(args.input.resolve()),
        "input_sha256": sha256(args.input),
        "screen": {"AF_ref_min": PRIMARY_AF, "r_cryo_re_max_fraction": PRIMARY_RCRYO, "source_column": "r_cryo_re_fraction (task semantic field: r_cryo_re)", "lcoe": LCOE},
        "architecture_rows": int(len(cross)),
        "common_feasible": int(cross.common_feasible.sum()),
        "within_1pct": int((cross.common_feasible.eq(1) & (cross.max_rel_penalty_pct <= 1)).sum()),
        "within_5pct": int((cross.common_feasible.eq(1) & (cross.max_rel_penalty_pct <= 5)).sum()),
        "within_10pct": int((cross.common_feasible.eq(1) & (cross.max_rel_penalty_pct <= 10)).sum()),
        "output": "Fig6_near_optimal_commercial.{svg,pdf,png}",
    }
    (args.output / "Fig6_near_optimal_commercial_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
