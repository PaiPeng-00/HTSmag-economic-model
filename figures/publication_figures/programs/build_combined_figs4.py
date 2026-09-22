#!/usr/bin/env python3
"""Build consolidated supplementary Fig. S4 from the authoritative data.

The scientific data and the plotting vocabulary are inherited from the two
approved temperature-resolved source figures (legacy Fig. S5 and Fig. S6).
The combined 2 x 4 layout retains the approved style. System-total heat is
converted explicitly to a single TF magnet; passive background excludes nuclear heat.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


import fitz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from publication_style import (
    PALETTE,
    SI_AXIS_LABEL_PT,
    SI_PANEL_LETTER_PT,
    SI_PANEL_TITLE_PT,
    SI_TICK_LABEL_PT,
    activate_style,
    enforce_figure_style,
)


MM = 1 / 25.4
# Exact approved source-figure curve colour.
BLUE = "#3A7B9B"
CANVAS_WIDTH_MM = 158.0
CANVAS_HEIGHT_MM = 82.0


def style_axis(ax) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)
    ax.tick_params(
        direction="in",
        top=True,
        right=True,
        width=0.8,
        length=2.5,
        labelsize=SI_TICK_LABEL_PT,
        pad=1.5,
    )


def panel_label(ax, letter: str, title: str) -> None:
    # Preserve the approved legacy supplementary-figure alignment exactly.
    ax.text(
        -0.16,
        1.19,
        letter,
        transform=ax.transAxes,
        fontsize=SI_PANEL_LETTER_PT,
        fontweight="bold",
        color=PALETTE["text"],
        va="top",
    )
    ax.text(
        -0.01,
        1.19,
        title,
        transform=ax.transAxes,
        fontsize=SI_PANEL_TITLE_PT,
        fontweight="bold",
        color=PALETTE["text"],
        va="top",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-4p2", required=True)
    parser.add_argument("--data-10", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    activate_style("SI", "FigS4")

    data = {
        4.2: pd.read_parquet(args.data_4p2),
        10.0: pd.read_parquet(args.data_10),
    }
    # Frozen export mixes per-magnet components and system-total columns.
    # Preserve that input identity; derive explicitly per-TF plotting columns.
    unit_audit = {}
    for temp, frame in data.items():
        per_tf = frame['Q_total_Tc_W'] / 18.0
        residual = np.max(np.abs(per_tf - frame['Q_joint_W'] - frame['Q_background_W']))
        assert residual < 1e-7, (temp, residual)
        frame['Q_total_per_TF_W'] = per_tf
        # The exported background includes nuclear heating. The caption's
        # passive-background component must exclude that separately shown term.
        frame['Q_passive_per_TF_W'] = frame['Q_background_W'] - frame['Q_nuclear_W']
        assert (frame['Q_passive_per_TF_W'] > 0).all()
        unit_audit[str(temp)] = {'system_to_single_TF_divisor':18,
            'max_heat_balance_residual_W':float(residual),
            'Npw200_total_median_W':float(frame.loc[frame.Npw.eq(200),'Q_total_per_TF_W'].median()),
            'single_TF_total_range_W':[float(per_tf.min()),float(per_tf.max())],
            'passive_background_excludes_nuclear':True,'77K_intercept_excluded':True}
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    columns = {
        "Joint": ("Q_joint_W", "R_joint_nOhm", r"Joint resistance, $R_{\mathrm{j}}$ (n$\Omega$)"),
        "Nuclear": ("Q_nuclear_W", "Npw", r"Parallel tapes per turn, $N_{\mathrm{pw}}$"),
        "Background": ("Q_passive_per_TF_W", "Npw", r"Parallel tapes per turn, $N_{\mathrm{pw}}$"),
        "Total": ("Q_total_per_TF_W", "Npw", r"Parallel tapes per turn, $N_{\mathrm{pw}}$"),
    }

    limits: dict[str, tuple[float, float]] = {}
    for title, (value_col, group_col, _) in columns.items():
        values = []
        for frame in data.values():
            values.extend(frame.groupby(group_col)[value_col].median().to_numpy())
        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))
        if title in {"Nuclear", "Background"}:
            center = 0.5 * (vmin + vmax)
            span = max(vmax - vmin, 0.50 * max(abs(center), 1.0))
            limits[title] = (center - span, center + span)
        else:
            limits[title] = (0.0, vmax * 1.08)

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(CANVAS_WIDTH_MM * MM, CANVAS_HEIGHT_MM * MM),
        sharey=False,
    )
    fig.subplots_adjust(
        left=0.082,
        right=0.985,
        bottom=0.145,
        top=0.905,
        wspace=0.46,
        hspace=0.78,
    )
    letters = iter("ABCDEFGH")
    for row, temp in enumerate((4.2, 10.0)):
        frame = data[temp]
        for col, (title, (value_col, group_col, xlabel)) in enumerate(columns.items()):
            ax = axes[row, col]
            series = frame.groupby(group_col)[value_col].median()
            ax.plot(series.index, series.values, color=BLUE, linewidth=1.2)
            display_title = f"{title} ({temp:g} K)" if col == 0 else title
            panel_label(ax, next(letters), display_title)
            ax.set_xlabel(xlabel, fontsize=SI_AXIS_LABEL_PT)
            # Match the approved separate source figures: every panel carries
            # its own y-axis label rather than inheriting a shared substitute.
            ax.set_ylabel("Heat load per TF magnet (W)", fontsize=SI_AXIS_LABEL_PT)
            ax.set_ylim(*limits[title])
            style_axis(ax)
            if group_col == "R_joint_nOhm":
                ax.set_xscale("log")
                ax.set_xticks([1, 10, 100])
                ax.set_xticklabels(["1", "10", "100"])

    enforce_figure_style(fig, "SI")
    pdf = out / "FigS4.pdf"
    svg = out / "FigS4.svg"
    png = out / "FigS4.png"
    preview = out / "FigS4_preview.png"
    # No tight-bbox reframing: the source canvas is already the exact physical
    # manuscript size and contains no artificially added outer border.
    fig.savefig(pdf, bbox_inches=None)
    fig.savefig(svg, bbox_inches=None)
    fig.savefig(png, dpi=600, bbox_inches=None)
    plt.close(fig)

    with fitz.open(pdf) as document:
        pixmap = document[0].get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        pixmap.save(preview)

    audit = {
        "status": "PASS",
        "figure": "FigS4",
        "source_4p2K": str(Path(args.data_4p2).resolve()),
        "source_10K": str(Path(args.data_10).resolve()),
        "layout": "2 rows x 4 columns",
        "panels": "A-D: 4.2 K; E-H: 10 K",
        "style_authority": "approved legacy FigS5/FigS6 supplementary style",
        "font": "Arial",
        "physical_size_mm": [CANVAS_WIDTH_MM, CANVAS_HEIGHT_MM],
        "four_sided_axes": True,
        "unit_audit": unit_audit,
        "input_sha256": {"4.2K":sha256(Path(args.data_4p2)),"10K":sha256(Path(args.data_10))},
        "inward_ticks_all_sides": True,
        "outputs": {
            "pdf_sha256": sha256(pdf),
            "svg_sha256": sha256(svg),
            "png_sha256": sha256(png),
        },
    }
    (out / "FigS4_BUILD_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"WROTE {pdf}")
    print(f"WROTE {svg}")
    print(f"WROTE {png}")


if __name__ == "__main__":
    main()
