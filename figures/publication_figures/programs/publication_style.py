"""Single authoritative publication style for V6.7 manuscript figures.

This module changes presentation only.  It does not read or transform scientific
data.  Active renderers call :func:`activate_style` before importing their
approved plotting implementation and write review candidates to the isolated
PALETTE_B_DEEP_ALL_BOXED output directory.
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import shutil
from pathlib import Path
from typing import Iterable

import fitz
import matplotlib as mpl
import numpy as np
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import FigureCanvasPdf
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from PIL import Image
from palette_b import PALETTE as MINERAL, TEMPERATURE as MINERAL_TEMPERATURE

MM_PER_INCH = 25.4
MANUSCRIPT_TEXT_WIDTH_PT = 489.10307
TEX_PT_TO_MM = 25.4 / 72.27
MANUSCRIPT_TEXT_WIDTH_MM = MANUSCRIPT_TEXT_WIDTH_PT * TEX_PT_TO_MM
MAIN_FIGURE_WIDTH_MM = MANUSCRIPT_TEXT_WIDTH_MM
OUTER_LEFT_MARGIN_MM = 1.4
OUTER_RIGHT_MARGIN_MM = 1.4
OUTER_TOP_MARGIN_MM = 2.0
OUTER_BOTTOM_MARGIN_MM = 2.0

PANEL_LETTER_PT = 10.0
PANEL_TITLE_PT = 9.2
DEFAULT_TEXT_PT = 7.8
SCHEMATIC_PRIMARY_PT = 8.4
SCHEMATIC_NORMAL_PT = 7.5
SCHEMATIC_SECONDARY_PT = 7.2
AXIS_LABEL_PT = 8.4
TICK_LABEL_PT = 7.8
LEGEND_PT = 8.2
COLORBAR_LABEL_PT = 7.8
COLORBAR_TICK_PT = 7.4
ANNOTATION_PT = DEFAULT_TEXT_PT
SECONDARY_ANNOTATION_PT = DEFAULT_TEXT_PT
MIN_MAIN_ESSENTIAL_TEXT_PT = DEFAULT_TEXT_PT
MINIMUM_ESSENTIAL_TEXT_PT = MIN_MAIN_ESSENTIAL_TEXT_PT

SI_PANEL_LETTER_PT = 9.0
SI_PANEL_TITLE_PT = 8.3
SI_AXIS_LABEL_PT = 7.5
SI_TICK_LABEL_PT = 6.9
SI_LEGEND_PT = 6.9
SI_ANNOTATION_PT = 6.8
SI_MIN_ESSENTIAL_TEXT_PT = 6.8
SI_MINIMUM_ESSENTIAL_TEXT_PT = SI_MIN_ESSENTIAL_TEXT_PT

LW_DATA_PRIMARY = 1.75
LW_DATA_SECONDARY = 1.00
LW_BOUNDARY_PRIMARY = 1.70
LW_CONTOUR_SECONDARY = 0.85
LW_AXIS = 0.80
LW_FRAME = 0.80
LW_ARROW_PRIMARY = 1.30
LW_ARROW_SECONDARY = 1.00
LW_GRID = 0.35
LW_HATCH = 0.80
LW_ERRORBAR_PRIMARY = 2.20
LW_ERRORBAR_SECONDARY = 1.20

MS_PRIMARY = 5.4
MS_SECONDARY = 4.1
MS_SMALL = 3.5
MARKER_EDGE_PRIMARY = 0.90
MARKER_EDGE_SECONDARY = 0.70

TEMPERATURE = {
    value: {"main": color, "light": color}
    for value, color in MINERAL_TEMPERATURE.items()
}
PALETTE = {
    "gray": {"main": MINERAL["guide"], "light": MINERAL["infeasible"]},
    "purple_gray": {"main": MINERAL["scenario_s2"], "light": "#ECE8EB"},
    "blue_green": {"main": MINERAL["cross"], "light": "#D6E5E0"},
    "orange": {"main": MINERAL["scenario_s3"], "light": "#F0E9E3"},
    "text": MINERAL["ink"],
    "axis": MINERAL["spine"],
    "grid": MINERAL["grid"],
}

MAIN_HEIGHT_TARGET_MM = {"Fig1": 114.0, "Fig2": 120.0, "Fig3": 136.0, "Fig4": 112.0}
# Preserve the frozen V6.7 final-PDF page aspect ratios exactly.  The rendered
# content is fitted proportionally inside these boxes, so palette-only changes
# (including infeasible-region hatching) cannot alter manuscript layout.
MAIN_FINAL_HEIGHT_MM = {
    "Fig1": 160.1271386111111,
    "Fig2": 172.6294607374403,
    "Fig3": 197.45377916666666,
    "Fig4": 133.0775360107422,
}
SI_HEIGHT_TARGET_MM = {
    "FigS1": 132.0,
    "FigS3": 190.0,
    "FigS4": 190.0,
    "FigS5": 190.0,
    "FigS6": 190.0,
    "FigS7": 150.0,
}

_ORIGINAL_SAVEFIG = Figure.savefig
_ACTIVE_PROFILE = "MAIN"
_ACTIVE_FIGURE_ID: str | None = None
_PATCHED = False

# The legacy renderers draw on wider internal canvases.  Candidate PDFs are
# reframed vectorially to the real manuscript text width.  These preflight
# factors keep final, manuscript-scale typography at the frozen target sizes.
EXPECTED_POST_SCALE = {
    "Fig1": 1.122,
    "Fig2": 1.115,
    "Fig3": 1.125,
    "Fig4": 1.061,
    "FigS1": 1.0,
    "FigS3": 1.085,
    "FigS4": 1.085,
    "FigS5": 1.055,
    "FigS6": 1.055,
    "FigS7": 1.03,
}

SI_OUTPUT_WIDTH_MM = {
    "FigS1": 160.0,
    "FigS3": 0.94 * MANUSCRIPT_TEXT_WIDTH_MM,
    "FigS4": 0.94 * MANUSCRIPT_TEXT_WIDTH_MM,
    "FigS5": 0.88 * MANUSCRIPT_TEXT_WIDTH_MM,
    "FigS6": 0.88 * MANUSCRIPT_TEXT_WIDTH_MM,
    "FigS7": 0.96 * MANUSCRIPT_TEXT_WIDTH_MM,
}


def mm_to_inch(mm: float) -> float:
    return mm / MM_PER_INCH


def figsize_mm(width_mm: float, height_mm: float) -> tuple[float, float]:
    return mm_to_inch(width_mm), mm_to_inch(height_mm)


def _source_scale() -> float:
    return EXPECTED_POST_SCALE.get(_ACTIVE_FIGURE_ID or "", 1.0)


def _source_pt(final_pt: float) -> float:
    return final_pt / _source_scale()


def source_pt(final_pt: float) -> float:
    """Convert a requested final manuscript point size to renderer points."""
    return _source_pt(final_pt)


def _require_arial() -> str:
    try:
        return font_manager.findfont("Arial", fallback_to_default=False)
    except ValueError as exc:
        raise SystemExit("PUBLICATION_FONT_ARIAL_NOT_AVAILABLE") from exc


def _common_rcparams(base_size: float) -> dict[str, object]:
    _require_arial()
    return {
        "font.family": "Arial",
        "font.sans-serif": ["Arial"],
        "font.size": base_size,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "mathtext.sf": "Arial",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.linewidth": LW_AXIS,
        "axes.edgecolor": PALETTE["axis"],
        "axes.labelcolor": PALETTE["text"],
        "axes.labelsize": AXIS_LABEL_PT if base_size >= MINIMUM_ESSENTIAL_TEXT_PT else SI_AXIS_LABEL_PT,
        "xtick.labelsize": TICK_LABEL_PT if base_size >= MINIMUM_ESSENTIAL_TEXT_PT else SI_TICK_LABEL_PT,
        "ytick.labelsize": TICK_LABEL_PT if base_size >= MINIMUM_ESSENTIAL_TEXT_PT else SI_TICK_LABEL_PT,
        "xtick.major.width": LW_AXIS,
        "ytick.major.width": LW_AXIS,
        "legend.fontsize": LEGEND_PT if base_size >= MINIMUM_ESSENTIAL_TEXT_PT else SI_LEGEND_PT,
        "legend.frameon": False,
        "axes.facecolor": MINERAL["white"],
        "figure.facecolor": MINERAL["white"],
        "hatch.color": MINERAL["hatch"],
        "hatch.linewidth": 0.55,
        "text.color": PALETTE["text"],
        "axes.prop_cycle": mpl.cycler(color=[TEMPERATURE[4.2]["main"], TEMPERATURE[10.0]["main"], TEMPERATURE[20.0]["main"], PALETTE["blue_green"]["main"], PALETTE["orange"]["main"]]),
        "savefig.transparent": False,
    }


def apply_main_rcparams() -> None:
    mpl.rcParams.update(_common_rcparams(MINIMUM_ESSENTIAL_TEXT_PT))


def apply_si_rcparams() -> None:
    mpl.rcParams.update(_common_rcparams(SI_MINIMUM_ESSENTIAL_TEXT_PT))


def style_axis(ax, xlabel=None, ylabel=None, xscale=None, yscale=None) -> None:
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=_source_pt(AXIS_LABEL_PT if _ACTIVE_PROFILE == "MAIN" else SI_AXIS_LABEL_PT))
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=_source_pt(AXIS_LABEL_PT if _ACTIVE_PROFILE == "MAIN" else SI_AXIS_LABEL_PT))
    if xscale is not None:
        ax.set_xscale(xscale)
    if yscale is not None:
        ax.set_yscale(yscale)
    for spine in ax.spines.values():
        spine.set_linewidth(LW_FRAME if spine.get_visible() else LW_AXIS)
        spine.set_color("#000000")
        spine.set_zorder(30)
    ax.tick_params(direction="in", width=LW_AXIS, labelsize=TICK_LABEL_PT if _ACTIVE_PROFILE == "MAIN" else SI_TICK_LABEL_PT)


def style_panel_title(ax, letter: str, title: str) -> None:
    size_letter = _source_pt(PANEL_LETTER_PT if _ACTIVE_PROFILE == "MAIN" else SI_PANEL_LETTER_PT)
    size_title = _source_pt(PANEL_TITLE_PT if _ACTIVE_PROFILE == "MAIN" else SI_PANEL_TITLE_PT)
    ax.text(0.0, 1.02, letter, transform=ax.transAxes, fontsize=size_letter, fontweight="bold", ha="right", va="bottom")
    ax.text(0.02, 1.02, title, transform=ax.transAxes, fontsize=size_title, fontweight="bold", ha="left", va="bottom")


def style_legend(legend) -> None:
    if legend is None:
        return
    for text in legend.get_texts():
        text.set_fontfamily("Arial")
        text.set_fontsize(_source_pt(LEGEND_PT if _ACTIVE_PROFILE == "MAIN" else SI_LEGEND_PT))


def style_colorbar(cbar) -> None:
    cbar.ax.tick_params(direction="in", width=_source_pt(LW_AXIS), labelsize=_source_pt(COLORBAR_TICK_PT if _ACTIVE_PROFILE == "MAIN" else SI_TICK_LABEL_PT))
    cbar.ax.yaxis.label.set_size(_source_pt(COLORBAR_LABEL_PT if _ACTIVE_PROFILE == "MAIN" else SI_AXIS_LABEL_PT))


def style_primary_curve(line: Line2D) -> Line2D:
    line.set_linewidth(_source_pt(LW_DATA_PRIMARY))
    return line


def style_secondary_curve(line: Line2D) -> Line2D:
    line.set_linewidth(_source_pt(LW_DATA_SECONDARY))
    return line


def style_primary_contour(contour) -> None:
    for collection in getattr(contour, "collections", []):
        collection.set_linewidth(_source_pt(LW_BOUNDARY_PRIMARY))


def style_secondary_contour(contour) -> None:
    for collection in getattr(contour, "collections", []):
        collection.set_linewidth(_source_pt(LW_CONTOUR_SECONDARY))


def style_annotation(text, secondary: bool = False) -> None:
    text.set_fontfamily("Arial")
    text.set_fontsize(_source_pt(SECONDARY_ANNOTATION_PT if secondary else ANNOTATION_PT))


def _is_panel_letter(text: str) -> bool:
    return len(text.strip()) == 1 and text.strip() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def enforce_figure_style(fig: Figure, profile: str | None = None) -> None:
    profile = profile or _ACTIVE_PROFILE
    minimum = _source_pt(MINIMUM_ESSENTIAL_TEXT_PT if profile == "MAIN" else SI_MINIMUM_ESSENTIAL_TEXT_PT)
    title_size = _source_pt(PANEL_TITLE_PT if profile == "MAIN" else SI_PANEL_TITLE_PT)
    letter_size = _source_pt(PANEL_LETTER_PT if profile == "MAIN" else SI_PANEL_LETTER_PT)
    tick_size = _source_pt(TICK_LABEL_PT if profile == "MAIN" else SI_TICK_LABEL_PT)
    label_size = _source_pt(AXIS_LABEL_PT if profile == "MAIN" else SI_AXIS_LABEL_PT)
    for text in fig.findobj(match=lambda obj: hasattr(obj, "set_fontfamily") and hasattr(obj, "get_text")):
        with contextlib.suppress(Exception):
            value = text.get_text().strip()
            text.set_fontfamily("Arial")
            if _is_panel_letter(value):
                text.set_fontsize(letter_size)
            elif text.get_fontweight() in ("bold", "semibold", 600, 700):
                text.set_fontsize(max(float(text.get_fontsize()), title_size))
            else:
                text.set_fontsize(max(float(text.get_fontsize()), minimum))
            if _is_panel_letter(value):
                text.set_fontweight("bold")
    for ax in fig.axes:
        for label in (ax.xaxis.label, ax.yaxis.label):
            label.set_fontfamily("Arial")
            label.set_fontsize(label_size)
        for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
            label.set_fontfamily("Arial")
            label.set_fontsize(tick_size)
        if ax.title.get_text():
            ax.title.set_fontfamily("Arial")
            if getattr(ax, "_colorbar", None) is not None:
                # Legacy renderers use colorbar-axis titles as colorbar labels.
                ax.title.set_fontweight("normal")
                ax.title.set_fontsize(_source_pt(COLORBAR_LABEL_PT if profile == "MAIN" else SI_AXIS_LABEL_PT))
            else:
                ax.title.set_fontweight("bold")
                ax.title.set_fontsize(max(float(ax.title.get_fontsize()), title_size))
        no_outer_frame = bool(getattr(ax, "_no_outer_frame", False)) or not ax.axison
        for spine in ax.spines.values():
            if not no_outer_frame:
                spine.set_visible(True)
            if spine.get_visible():
                spine.set_linewidth(_source_pt(LW_AXIS))
                spine.set_color("#000000")
                spine.set_zorder(30)
        ax.tick_params(direction="in", width=_source_pt(LW_AXIS))
        for gridline in [*ax.get_xgridlines(), *ax.get_ygridlines()]:
            gridline.set_linewidth(_source_pt(LW_GRID))
            gridline.set_color(PALETTE["grid"])
        for line in ax.lines:
            if line in ax.get_xgridlines() or line in ax.get_ygridlines():
                continue
            current_width = float(line.get_linewidth())
            if current_width >= 1.55:
                line.set_linewidth(_source_pt(LW_DATA_PRIMARY))
            elif current_width >= 1.05:
                line.set_linewidth(_source_pt(LW_DATA_SECONDARY))
            else:
                line.set_linewidth(max(current_width, _source_pt(LW_HATCH)))
            if line.get_marker() not in (None, "None", "", " "):
                current_marker = float(line.get_markersize())
                target_marker = MS_PRIMARY if current_marker >= 4.5 else MS_SECONDARY if current_marker >= 3.5 else MS_SMALL
                line.set_markersize(_source_pt(target_marker))
                line.set_markeredgewidth(_source_pt(MARKER_EDGE_PRIMARY if target_marker == MS_PRIMARY else MARKER_EDGE_SECONDARY))
        style_legend(ax.get_legend())


def _styled_savefig(self: Figure, fname, *args, **kwargs):
    enforce_figure_style(self)
    suffix = Path(str(fname)).suffix.lower()
    kwargs.pop("bbox_inches", None)
    kwargs["bbox_inches"] = None
    if suffix == ".png":
        kwargs["dpi"] = 600
    return _ORIGINAL_SAVEFIG(self, fname, *args, **kwargs)


def activate_style(profile: str = "MAIN", figure_id: str | None = None) -> None:
    global _ACTIVE_PROFILE, _ACTIVE_FIGURE_ID, _PATCHED
    _ACTIVE_PROFILE = profile.upper()
    _ACTIVE_FIGURE_ID = figure_id
    apply_main_rcparams() if _ACTIVE_PROFILE == "MAIN" else apply_si_rcparams()
    if not _PATCHED:
        Figure.savefig = _styled_savefig
        _PATCHED = True


def style_output_dir(figure_root: Path) -> Path:
    # Keep each regenerated styled candidate inside its own figure package.
    # Formal assets are promoted separately after the selection gates pass.
    target = figure_root / "build"
    target.mkdir(parents=True, exist_ok=True)
    return target


def _raster_content_bbox(page, dpi: int = 180, white_threshold: int = 248) -> fitz.Rect:
    pixmap = page.get_pixmap(matrix=fitz.Matrix(dpi / 72.0, dpi / 72.0), alpha=False, annots=False)
    pixels = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, pixmap.n)
    visible = np.min(pixels[:, :, :3], axis=2) < white_threshold
    ys, xs = np.nonzero(visible)
    if not len(xs):
        return fitz.Rect(page.rect)
    scale = 72.0 / dpi
    return fitz.Rect(xs.min() * scale, ys.min() * scale, (xs.max() + 1) * scale, (ys.max() + 1) * scale)


def _resize_pdf_vector(
    source: Path,
    target: Path,
    width_mm: float,
    height_override_mm: float | None = None,
) -> tuple[float, float, float, dict[str, float]]:
    src = fitz.open(source)
    if len(src) != 1:
        src.close()
        raise ValueError(f"Expected one-page figure PDF: {source}")
    page = src[0]
    visible = _raster_content_bbox(page)
    pad_pt = 0.35 / MM_PER_INCH * 72.0
    clip = fitz.Rect(
        max(page.rect.x0, visible.x0 - pad_pt),
        max(page.rect.y0, visible.y0 - pad_pt),
        min(page.rect.x1, visible.x1 + pad_pt),
        min(page.rect.y1, visible.y1 + pad_pt),
    )
    content_width_mm = width_mm - OUTER_LEFT_MARGIN_MM - OUTER_RIGHT_MARGIN_MM
    source_clip_width_mm = clip.width / 72.0 * MM_PER_INCH
    vector_scale = content_width_mm / source_clip_width_mm
    content_height_mm = clip.height / 72.0 * MM_PER_INCH * vector_scale
    auto_height_mm = content_height_mm + OUTER_TOP_MARGIN_MM + OUTER_BOTTOM_MARGIN_MM
    height_mm = height_override_mm if height_override_mm is not None else auto_height_mm
    out = fitz.open()
    page = out.new_page(width=width_mm / MM_PER_INCH * 72.0, height=height_mm / MM_PER_INCH * 72.0)
    target_rect = fitz.Rect(
        OUTER_LEFT_MARGIN_MM / MM_PER_INCH * 72.0,
        OUTER_TOP_MARGIN_MM / MM_PER_INCH * 72.0,
        (width_mm - OUTER_RIGHT_MARGIN_MM) / MM_PER_INCH * 72.0,
        (height_mm - OUTER_BOTTOM_MARGIN_MM) / MM_PER_INCH * 72.0,
    )
    page.show_pdf_page(target_rect, src, 0, clip=clip, keep_proportion=True)
    out.save(target, garbage=4, deflate=True)
    out.close(); src.close()
    crop = {
        "source_visible_x0_mm": visible.x0 / 72.0 * MM_PER_INCH,
        "source_visible_x1_mm": visible.x1 / 72.0 * MM_PER_INCH,
        "source_clip_width_mm": source_clip_width_mm,
        "left_blank_mm": OUTER_LEFT_MARGIN_MM,
        "right_blank_mm": OUTER_RIGHT_MARGIN_MM,
        "content_width_mm": content_width_mm,
    }
    return width_mm, height_mm, vector_scale, crop


def _resize_svg(source: Path, target: Path, width_mm: float, height_mm: float) -> None:
    text = source.read_text(encoding="utf-8")
    import re
    text = re.sub(r'<svg\s+width="[^"]+"\s+height="[^"]+"', f'<svg width="{width_mm:.6f}mm" height="{height_mm:.6f}mm"', text, count=1)
    target.write_text(text, encoding="utf-8")


def _resize_png(source: Path, target: Path, width_mm: float, height_mm: float, dpi: int = 600) -> None:
    image = Image.open(source).convert("RGB")
    size = (round(width_mm / MM_PER_INCH * dpi), round(height_mm / MM_PER_INCH * dpi))
    image.resize(size, Image.Resampling.LANCZOS).save(target, dpi=(dpi, dpi))


def finalize_style_outputs(root: Path, figure_id: str) -> dict[str, object]:
    branch = style_output_dir(root)
    pdf = branch / f"{figure_id}.pdf"
    svg = branch / f"{figure_id}.svg"
    png = branch / f"{figure_id}.png"
    if not pdf.is_file():
        raise FileNotFoundError(pdf)
    final_pdf = branch / f"{figure_id}_palette_b_bluegray_blackframes_panelheightfix.pdf"
    target_width_mm = MAIN_FIGURE_WIDTH_MM if figure_id.startswith("Fig") and not figure_id.startswith("FigS") else SI_OUTPUT_WIDTH_MM.get(figure_id, MAIN_FIGURE_WIDTH_MM)
    height_override_mm = MAIN_FINAL_HEIGHT_MM.get(figure_id)
    width_mm, height_mm, vector_scale, crop = _resize_pdf_vector(
        pdf,
        final_pdf,
        target_width_mm,
        height_override_mm=height_override_mm,
    )
    final_svg = branch / f"{figure_id}_palette_b_bluegray_blackframes_panelheightfix.svg"
    final_png = branch / f"{figure_id}_palette_b_bluegray_blackframes_panelheightfix.png"
    document = fitz.open(final_pdf)
    final_svg.write_text(document[0].get_svg_image(text_as_path=False), encoding="utf-8")
    pixmap = document[0].get_pixmap(matrix=fitz.Matrix(600 / 72.0, 600 / 72.0), alpha=False, annots=False)
    pixmap.save(final_png)
    document.close()
    record = {
        "status": "CANDIDATE" if figure_id == "Fig3" else "WAITING_AUTHOR_VECTOR" if figure_id == "Fig1" else "PASS",
        "figure_id": figure_id,
        "profile": _ACTIVE_PROFILE,
        "physical_width_mm": width_mm,
        "physical_height_mm": height_mm,
        "content_width_mm": crop["content_width_mm"],
        "left_outer_blank_mm": crop["left_blank_mm"],
        "right_outer_blank_mm": crop["right_blank_mm"],
        "vector_reframe_scale": vector_scale,
        "font": "Arial",
        "minimum_essential_text_pt": MINIMUM_ESSENTIAL_TEXT_PT if _ACTIVE_PROFILE == "MAIN" else SI_MINIMUM_ESSENTIAL_TEXT_PT,
        "source_pdf": str(pdf),
        "style_pdf": str(final_pdf),
        "scientific_data_modified": False,
    }
    (branch / f"{figure_id}_VISUAL_V3_EXPORT.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return record


def create_waiting_author_vector_pdf(
    target: Path,
    width_mm: float,
    height_mm: float,
    asset_name: str,
) -> Path:
    """Create an explicit vector placeholder without a panel letter/title."""
    figure = mpl.figure.Figure(figsize=figsize_mm(width_mm, height_mm), facecolor="white")
    canvas = FigureCanvasPdf(figure)
    axis = figure.add_axes([0.01, 0.03, 0.98, 0.94])
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    rectangle = mpl.patches.Rectangle(
        (0.01, 0.04), 0.98, 0.92, facecolor="#FAFAFA", edgecolor=PALETTE["gray"]["main"],
        linewidth=1.0, linestyle=(0, (5, 3)),
    )
    axis.add_patch(rectangle)
    axis.text(0.5, 0.56, "WAITING AUTHOR VECTOR", ha="center", va="center", fontsize=9.0, fontweight="bold", color=PALETTE["axis"])
    axis.text(0.5, 0.39, asset_name, ha="center", va="center", fontsize=7.4, color=PALETTE["axis"])
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.print_pdf(target)
    return target


def save_publication_figure(fig: Figure, path_pdf: Path, path_svg: Path, path_png: Path, dpi: int = 600) -> None:
    enforce_figure_style(fig)
    fig.savefig(path_pdf, format="pdf", bbox_inches=None)
    fig.savefig(path_svg, format="svg", bbox_inches=None)
    fig.savefig(path_png, format="png", dpi=dpi, bbox_inches=None)
