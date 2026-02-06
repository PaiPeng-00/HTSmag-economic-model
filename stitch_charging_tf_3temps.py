"""
2.6_stitch_charging_tf_3temps.py

Stitch three charging‑time heatmaps at different operating temperatures
from `outputs/figures/charging_tf` into a single horizontal figure, and
add subplot labels: (a) 4.2 K, (b) 10 K, (c) 20 K.

Dependency: svgutils
    pip install svgutils
"""

from pathlib import Path
import re
from typing import Optional

import numpy as np
import config as cfg

try:
    import svgutils.transform as sg
except ImportError as e:
    raise ImportError(
        "svgutils is required for SVG stitching. Please run: pip install svgutils"
    ) from e

# Directory containing charging heatmaps (consistent with 2.5 output)
CHARGING_TF_FIGURES_DIR = Path(cfg.OUTPUTS_FIGURES_DIR) / "charging_tf"
# SVG filename within each temperature subdirectory (same as 2.5 output)
HEATMAP_FILENAME = "charging_time999_heatmap_contour_TF_system_Npw=1-200.svg"
# Three temperatures and their subplot labels
TEMPERATURES = [
    (4.2, "a"),
    (10.0, "b"),
    (20.0, "c"),
]


def _parse_size(size_str: str) -> float:
    """Parse an SVG width/height string into a float."""
    if size_str is None:
        return 0.0
    m = re.findall(r"[0-9.]+", str(size_str))
    return float(m[0]) if m else 0.0


def _find_subdir_for_temp(base_dir: Path, temp: float) -> Optional[Path]:
    """Find the subdirectory that contains the heatmap for a given temperature.

    Expected pattern: `Temp_4.2K_*`, `Temp_10.0K_*`, `Temp_20.0K_*`, etc.
    """
    prefix = f"Temp_{temp}K_"
    for d in base_dir.iterdir():
        if d.is_dir() and d.name.startswith(prefix):
            return d
    return None


def stitch_charging_tf_3temps():
    """Stitch 4.2 K / 10 K / 20 K charging heatmaps into one row and add (a)(b)(c) labels."""
    if not CHARGING_TF_FIGURES_DIR.exists():
        print(f"[error] Directory does not exist: {CHARGING_TF_FIGURES_DIR}")
        print("Please run `2.5_charge_time999_TF_system_all_Npw=1-200.py` first to generate per‑temperature heatmaps.")
        return None

    panel_entries = []  # (root, col_idx, temp, label)
    panel_w = None
    panel_h = None

    for col_idx, (temp, letter) in enumerate(TEMPERATURES):
        subdir = _find_subdir_for_temp(CHARGING_TF_FIGURES_DIR, temp)
        if subdir is None:
            print(f"[warning] No subdirectory found for temperature {temp} K (expected like Temp_{temp}K_*_charge999)")
            continue
        svg_path = subdir / HEATMAP_FILENAME
        if not svg_path.exists():
            print(f"[warning] Missing heatmap SVG file: {svg_path}")
            continue
        fig = sg.fromfile(str(svg_path))
        root = fig.getroot()
        size_w, size_h = fig.get_size()
        w = _parse_size(size_w)
        h = _parse_size(size_h)
        panel_w = w if panel_w is None else max(panel_w, w)
        panel_h = h if panel_h is None else max(panel_h, h)
        panel_entries.append((root, col_idx, temp, letter))

    if not panel_entries or panel_w is None or panel_h is None:
        print("No sufficient charging heatmap subplots found; stitching aborted.")
        return None

    n_cols = len(panel_entries)
    margin_x = panel_w * 0.08
    label_font_size = 22
    top_label_height = label_font_size + 12  # space for (a)(b)(c) labels (and optional temperature labels)

    subplot_area_w = margin_x * (n_cols + 1) + panel_w * n_cols
    subplot_area_h = panel_h
    total_w = subplot_area_w
    total_h = subplot_area_h + top_label_height

    fig_out = sg.SVGFigure(f"{total_w}pt", f"{total_h}pt")
    subplot_start_x = margin_x
    subplot_start_y = top_label_height

    placed_roots = []
    for root, col_idx, temp, letter in panel_entries:
        x = subplot_start_x + col_idx * (panel_w + margin_x)
        y = subplot_start_y
        root.moveto(x, y)
        placed_roots.append(root)
    fig_out.append(placed_roots)

    out_path = CHARGING_TF_FIGURES_DIR / "charging_time999_heatmap_contour_TF_system_3temps_stitched.svg"
    fig_out.save(str(out_path))

    # Add (a)(b)(c) panel labels (and optionally temperature labels, if enabled)
    import xml.etree.ElementTree as ET
    tree = ET.parse(str(out_path))
    root_svg = tree.getroot()
    svg_ns = None
    for key, uri in root_svg.attrib.items():
        if key.startswith("xmlns") and "svg" in uri.lower():
            svg_ns = uri
            break
    if svg_ns is None:
        svg_ns = "http://www.w3.org/2000/svg"
    ET.register_namespace("", svg_ns)

    root_svg.set("width", f"{total_w}pt")
    root_svg.set("height", f"{total_h}pt")
    root_svg.set("viewBox", f"0 0 {total_w} {total_h}")
    labels_group = ET.SubElement(root_svg, f"{{{svg_ns}}}g", {"id": "panel_labels"})

    for root, col_idx, temp, letter in panel_entries:
        # Label position near the top‑left of each subplot, with a small inset margin
        x_left = subplot_start_x + col_idx * (panel_w + margin_x) + 8  # 8 pt inset
        y_top = top_label_height - 10  # slightly below the top edge
        text_elem = ET.SubElement(labels_group, f"{{{svg_ns}}}text", {
            "x": str(x_left),
            "y": str(y_top),
            "font-family": "Arial, sans-serif",
            "font-size": str(label_font_size),
            "font-weight": "bold",
            "text-anchor": "start",
            "dominant-baseline": "hanging",
            "fill": "black",
        })
        text_elem.text = f"{letter}"
        '''
        # Temperature labels: 4.2 K, 10 K, 20 K
        temp_str = str(int(temp)) if temp in (10.0, 20.0) else str(temp)
        temp_elem = ET.SubElement(labels_group, f"{{{svg_ns}}}text", {
            "x": str(x_center),
            "y": str(top_label_height - 4 - label_font_size - 4),
            "font-family": "Arial, sans-serif",
            "font-size": str(int(label_font_size * 0.85)),
            "text-anchor": "middle",
            "dominant-baseline": "bottom",
            "fill": "black",
        })
        temp_elem.text = f"{temp_str} K"'''

    root_svg.remove(labels_group)
    root_svg.append(labels_group)
    tree.write(str(out_path), encoding="utf-8", xml_declaration=True)

    print(f"Stitching completed: {out_path}")
    print("Layout: (a) 4.2 K, (b) 10 K, (c) 20 K")
    return out_path


if __name__ == "__main__":
    stitch_charging_tf_3temps()
