"""Shared SVG finalization for publication figures.

This module operates inside the upstream plotting layer.  It normalizes
Matplotlib path text to a requested *physical* point size after panels have
been positioned, so downstream compensating wrappers are unnecessary.
"""

from __future__ import annotations

import math
import re
import string
from pathlib import Path

from lxml import etree


SVG_NS = "http://www.w3.org/2000/svg"
NS = {"svg": SVG_NS}
NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
SCALE_RE = re.compile(r"scale\(\s*([-+0-9.eE]+)(?:[ ,]+([-+0-9.eE]+))?\s*\)")
TRANSLATE_RE = re.compile(r"translate\(\s*([-+0-9.eE]+)(?:[ ,]+([-+0-9.eE]+))?\s*\)")


def _scale_pair(transform: str | None) -> tuple[float, float]:
    sx = sy = 1.0
    if not transform:
        return sx, sy
    for match in SCALE_RE.finditer(transform):
        x = float(match.group(1))
        y = float(match.group(2)) if match.group(2) is not None else x
        sx *= x
        sy *= y
    return sx, sy


def _ancestor_scale(node: etree._Element, root: etree._Element) -> tuple[float, float]:
    chain: list[etree._Element] = []
    cursor = node.getparent()
    while cursor is not None:
        chain.append(cursor)
        if cursor is root:
            break
        cursor = cursor.getparent()
    sx = sy = 1.0
    for item in reversed(chain):
        x, y = _scale_pair(item.get("transform"))
        sx *= x
        sy *= y
    return sx, sy


def _find_render_transform(
    text_group: etree._Element,
) -> tuple[etree._Element, float] | None:
    """Return the shallow Matplotlib translate-plus-font-scale group."""
    for node in text_group.xpath(".//svg:g[@transform]", namespaces=NS):
        transform = node.get("transform", "")
        scale = SCALE_RE.search(transform)
        if scale is None or TRANSLATE_RE.search(transform) is None:
            continue
        sx = abs(float(scale.group(1)))
        sy = abs(float(scale.group(2))) if scale.group(2) else sx
        value = max(sx, sy)
        if 0.08 <= value <= 0.5:
            return node, value
    return None


def _replace_scale(transform: str, scale_x: float, scale_y: float) -> str:
    match = SCALE_RE.search(transform)
    if match is None:
        raise ValueError(f"No scale transform in {transform!r}")
    old_x = float(match.group(1))
    old_y = float(match.group(2)) if match.group(2) else old_x
    new_x = math.copysign(scale_x, old_x)
    new_y = math.copysign(scale_y, old_y)
    replacement = f"scale({new_x:.12g} {new_y:.12g})"
    return transform[: match.start()] + replacement + transform[match.end() :]


def _replace_translate(transform: str, x: float, y: float) -> str:
    match = TRANSLATE_RE.search(transform)
    replacement = f"translate({x:.9g} {y:.9g})"
    if match is None:
        return f"{replacement} {transform}".strip()
    return transform[: match.start()] + replacement + transform[match.end() :]


def _parse_length(value: str | None) -> float:
    match = NUMBER_RE.search(value or "")
    if match is None:
        raise ValueError(f"Missing numeric SVG length: {value!r}")
    return float(match.group())


def _approximate_path_bbox(
    text_group: etree._Element,
) -> tuple[float, float, float, float] | None:
    xs: list[float] = []
    ys: list[float] = []
    for path in text_group.xpath("./svg:path", namespaces=NS):
        values = [float(value) for value in NUMBER_RE.findall(path.get("d", ""))]
        xs.extend(values[0::2])
        ys.extend(values[1::2])
    if not xs or not ys:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def _is_contour_label(group: etree._Element) -> bool:
    parent = group.getparent()
    parent_id = parent.get("id", "") if parent is not None else ""
    if parent_id.startswith(("QuadContourSet_", "PathCollection_")):
        return True
    return len(group.xpath("./svg:g[@clip-path]", namespaces=NS)) == 1


def normalize_matplotlib_path_text(
    svg_path: str | Path,
    *,
    physical_width_mm: float,
    standard_pt: float = 7.0,
    contour_pt: float = 6.0,
    s4_cell_pt: float | None = None,
) -> dict[str, int]:
    """Set path-based text to exact physical sizes in the finished SVG.

    For Fig. S4, the 13-pt source annotations are identified before their
    scale is replaced and are assigned ``s4_cell_pt``.  Contour labels are
    assigned ``contour_pt``; all remaining path text uses ``standard_pt``.
    """
    svg_path = Path(svg_path)
    tree = etree.parse(
        str(svg_path), etree.XMLParser(remove_blank_text=False, huge_tree=True)
    )
    root = tree.getroot()
    view_box = [float(value) for value in root.get("viewBox", "").split()]
    if len(view_box) != 4 or view_box[2] <= 0:
        raise ValueError(f"Invalid viewBox in {svg_path}")
    root_pt_per_unit = physical_width_mm * 72.0 / 25.4 / view_box[2]

    counts = {"standard": 0, "contour": 0, "s4_cell": 0}
    for group in root.xpath('.//svg:g[starts-with(@id, "text_")]', namespaces=NS):
        render = _find_render_transform(group)
        if render is None:
            bbox = _approximate_path_bbox(group)
            if bbox is None or not _is_contour_label(group):
                continue
            target_pt = contour_pt
            category = "contour"
            if group.get("data-final-font-size-pt") == f"{target_pt:g}":
                counts[category] += 1
                continue
            desired_font_units = target_pt / root_pt_per_unit
            factor = desired_font_units / 18.0
            x0, y0, x1, y1 = bbox
            cx = 0.5 * (x0 + x1)
            cy = 0.5 * (y0 + y1)
            old_transform = group.get("transform", "").strip()
            resize = (
                f"translate({cx:.6g} {cy:.6g}) "
                f"scale({factor:.9g}) "
                f"translate({-cx:.6g} {-cy:.6g})"
            )
            group.set("transform", f"{old_transform} {resize}".strip())
            group.set("data-final-font-size-pt", f"{target_pt:g}")
            group.set("data-contour-font-size-pt", f"{target_pt:g}")
            counts[category] += 1
            continue
        render_node, old_scale = render
        if s4_cell_pt is not None and old_scale * 100.0 <= 13.5:
            target_pt = s4_cell_pt
            category = "s4_cell"
        elif _is_contour_label(group):
            target_pt = contour_pt
            category = "contour"
        else:
            target_pt = standard_pt
            category = "standard"

        ancestor_x, ancestor_y = _ancestor_scale(render_node, root)
        if ancestor_x == 0 or ancestor_y == 0:
            raise ValueError(f"Zero ancestor scale under {group.get('id')}")
        local_x = target_pt / (100.0 * abs(ancestor_x) * root_pt_per_unit)
        local_y = target_pt / (100.0 * abs(ancestor_y) * root_pt_per_unit)
        render_node.set(
            "transform",
            _replace_scale(render_node.get("transform", ""), local_x, local_y),
        )
        group.set("data-final-font-size-pt", f"{target_pt:g}")
        if category == "contour":
            group.set("data-contour-font-size-pt", f"{target_pt:g}")
        counts[category] += 1

    root.set("data-final-width-mm", f"{physical_width_mm:g}")
    root.set("data-font-family", "Arial")
    root.set("data-font-standard-pt", f"{standard_pt:g}")
    if s4_cell_pt is not None:
        root.set("data-font-s4-cell-pt", f"{s4_cell_pt:g}")
    tree.write(
        str(svg_path),
        encoding="utf-8",
        xml_declaration=True,
        pretty_print=False,
    )
    return counts


# 3-by-N heat-map grid geometry, measured from the frozen 160 mm submission set.
# All published grids share viewBox 2200x1600, the row pitch, the panel-letter
# offsets and the typography; only the column count and a few label anchors differ.
# columns=4 -> Fig. 5 / Fig. 6 (temperature x coolant)
# columns=3 -> Fig. 1 / Fig. 4 / fig. S10 (three temperatures)
_GRID_LAYOUTS: dict[int, dict[str, object]] = {
    4: {
        "panel_x": (220.0, 660.0, 1100.0, 1540.0),
        "label_count": 33,
        "y_ticks_x": 205.0,
        "y_title_x": 78.0,
        "row_label_x": 1952.0,
        "bottom_tick_y": 1481.0,
        "x_title_y": 1532.0,
        "expected_headers": ("4.2 K He", "10 K He", "20 K He", "20 K H2"),
    },
    3: {
        "panel_x": (480.0, 920.0, 1360.0),
        "label_count": 29,
        "y_ticks_x": 465.0,
        "y_title_x": 350.0,
        "row_label_x": 1772.0,
        "bottom_tick_y": 1481.0,
        "x_title_y": 1500.0,
        "expected_headers": ("4.2 K", "10 K", "20 K"),
    },
}


def finalize_economic_heatmap_grid(
    svg_path: str | Path,
    *,
    columns: int = 4,
    physical_width_mm: float = 160.0,
    standard_pt: float = 7.0,
    contour_pt: float = 6.0,
    panel_pt: float = 9.0,
    bottom_tick_y: float | None = None,
    y_title_x: float | None = None,
    y_tick_baseline: float = 66.0,
    label_round: int | None = None,
    n_x_ticks: int = 3,
    n_y_ticks: int = 4,
    row_label_x: float | None = None,
    row_label_anchor: str | None = None,
) -> dict[str, object]:
    """Finalize a 3-by-N publication heat-map grid for submission.

    Panel artwork and data coordinates are retained.  The established Fig. 5/6
    geometry, physical typography, and uppercase panel letters are applied in
    the upstream figure-generation layer.

    ``columns=4`` reproduces the Fig. 5 / Fig. 6 layout.  ``columns=3``
    reproduces the Fig. 1 / Fig. 4 / fig. S10 layout, which until 2026-07-26
    existed only as a chain of one-off scratchpad patches; see working-document
    section 14.42.  ``bottom_tick_y`` and ``y_title_x`` override the two anchors
    that legitimately differ between figures of the same column count.
    """
    if columns not in _GRID_LAYOUTS:
        raise ValueError(f"Unsupported grid column count: {columns}")
    layout = _GRID_LAYOUTS[columns]
    panel_x = layout["panel_x"]
    expected_headers = layout["expected_headers"]
    if bottom_tick_y is None:
        bottom_tick_y = layout["bottom_tick_y"]
    if y_title_x is None:
        y_title_x = layout["y_title_x"]
    expected_panels = 3 * columns

    svg_path = Path(svg_path)
    tree = etree.parse(
        str(svg_path), etree.XMLParser(remove_blank_text=False, huge_tree=True)
    )
    root = tree.getroot()
    panels = root.xpath("./svg:g[1]/svg:g", namespaces=NS)
    if len(panels) != expected_panels:
        raise ValueError(
            f"Expected {expected_panels} heat-map panels in {svg_path}, "
            f"found {len(panels)}"
        )

    view_width = 2200.0
    view_height = 1600.0
    final_height_mm = physical_width_mm * view_height / view_width
    target_width_pt = physical_width_mm * 72.0 / 25.4
    user_units_per_pt = view_width / target_width_pt
    standard_units = standard_pt * user_units_per_pt
    panel_units = panel_pt * user_units_per_pt

    panel_y = (100.0, 580.0, 1060.0)
    for index, panel in enumerate(panels):
        row, col = divmod(index, columns)
        panel.set(
            "transform",
            _replace_translate(panel.get("transform", ""), panel_x[col], panel_y[row]),
        )
        panel.set("data-common-grid", f"row-{row + 1}-col-{col + 1}")

    labels_groups = root.xpath("./svg:g[@id='labels']", namespaces=NS)
    if len(labels_groups) != 1:
        raise ValueError(f"Expected one shared labels group in {svg_path}")
    labels = labels_groups[0].xpath("./svg:text", namespaces=NS)
    # 每列的 x 刻度数与每行的 y 刻度数决定共享标签总数。默认 3/4 是原窄幅版式;
    # Fig. 1 扩轴后是 4 个 x 刻度(10/100/1000/10000) 与 5 个 y 刻度(1/5/10/20/50)。
    # 原先这两个数写死在切片索引里(n_bottom = 3*columns、y_ticks[row*4+tick]),
    # 换轴范围就会以 "Expected 29 ... found 35" 报错。
    label_count = columns + n_x_ticks * columns + 1 + n_y_ticks * 3 + 1 + 3
    if len(labels) != label_count:
        raise ValueError(
            f"Expected {label_count} shared labels in {svg_path}, found {len(labels)} "
            f"(columns={columns}, n_x_ticks={n_x_ticks}, n_y_ticks={n_y_ticks})"
        )

    n_bottom = n_x_ticks * columns
    n_y_total = n_y_ticks * 3
    headers = labels[:columns]
    bottom_ticks = labels[columns:columns + n_bottom]
    x_title = labels[columns + n_bottom]
    y_ticks = labels[columns + n_bottom + 1:columns + n_bottom + 1 + n_y_total]
    y_title = labels[columns + n_bottom + 1 + n_y_total]
    row_labels = labels[columns + n_bottom + 2 + n_y_total:columns + n_bottom + 5 + n_y_total]
    if tuple("".join(text.itertext()).strip() for text in headers) != expected_headers:
        raise ValueError(f"Unexpected heat-map header order in {svg_path}")

    for col, text in enumerate(headers):
        text.set("x", f"{panel_x[col] + 180:g}")
        text.set("y", "42")
    _panel_span = 360.0  # 面板在共享标签坐标系里的宽度
    _x_offsets = [_panel_span * i / (n_x_ticks - 1) for i in range(n_x_ticks)]
    for col in range(columns):
        for tick, x_offset in enumerate(_x_offsets):
            text = bottom_ticks[col * n_x_ticks + tick]
            text.set("x", f"{panel_x[col] + x_offset:g}")
            text.set("y", f"{bottom_tick_y:g}")
            text.set("dominant-baseline", "auto")
    x_title.set("x", "1100")
    x_title.set("y", f"{layout['x_title_y']:g}")
    x_title.set("dominant-baseline", "auto")

    # y_tick_baseline 吸收源拼图自身的顶部留白差异: Fig. 4 / fig. S10 为 66,
    # Fig. 1 的拼图多 18 个用户单位, 故为 84。取值以冻结 160 mm 版实测反推。
    first_row_offsets = [
        _parse_length(text.get("y")) - y_tick_baseline for text in y_ticks[:n_y_ticks]
    ]

    def _fmt_y(value: float) -> str:
        # label_round=3 复现冻结 3x3 版的"四舍五入到 3 位小数并去尾零"写法;
        # 不设时保持 3x4 (Fig. 5/6) 既有的 .9g 行为, 避免动到已冻结的候选图。
        if label_round is None:
            return f"{value:.9g}"
        return f"{round(value, label_round):.10g}"

    for row in range(3):
        for tick, offset in enumerate(first_row_offsets):
            text = y_ticks[row * n_y_ticks + tick]
            text.set("x", f"{layout['y_ticks_x']:g}")
            text.set("y", _fmt_y(panel_y[row] + offset))
    y_title.set("x", f"{y_title_x:g}")
    y_title.set("y", "760")
    y_title.set("transform", f"rotate(-90 {y_title_x:g} 760)")
    # 行标签默认左对齐于 layout['row_label_x']。Fig. 5 的 "S1/S2/S3" 很短(约 34 单位)
    # 所以没问题, 但 Fig. 6 的 "100 US$ kA-1 m-1" 宽约 270 单位, 从 1952 起会伸到 2222,
    # 超出 viewBox 宽 2200 -> 右侧被裁。改用右对齐即可: 右边界锁死在画布内,
    # 左边界 1920 仍在末列面板右缘(1900)之外, 不压图。加宽 viewBox 不可行 ——
    # 物理宽固定 160 mm, 加宽会把 7 pt 标准字号压到 6.8 pt。
    _row_x = layout['row_label_x'] if row_label_x is None else row_label_x
    for row, text in enumerate(row_labels):
        text.set("x", f"{_row_x:g}")
        text.set("y", f"{panel_y[row] + 180:g}")
        if row_label_anchor:
            text.set("text-anchor", row_label_anchor)

    for text in labels:
        text.set("font-family", "Arial, Helvetica, sans-serif")
        text.set("font-size", f"{standard_units:.6g}")
        text.set("data-final-font-size-pt", f"{standard_pt:g}")
        for child in text.xpath(".//svg:tspan", namespaces=NS):
            if (child.get("baseline-shift") or "").lower() in {"sub", "super"}:
                child.set("font-size", f"{0.7 * standard_units:.6g}")

    for old in root.xpath("./svg:g[@id='main_panel_labels']", namespaces=NS):
        root.remove(old)
    panel_group = etree.Element(f"{{{SVG_NS}}}g", id="main_panel_labels")
    for index, letter in enumerate(string.ascii_uppercase[:expected_panels]):
        row, col = divmod(index, columns)
        text = etree.SubElement(
            panel_group,
            f"{{{SVG_NS}}}text",
            x=f"{panel_x[col] - 35:g}",
            y=f"{panel_y[row] - 48:g}",
            **{
                "font-family": "Arial, Helvetica, sans-serif",
                "font-size": f"{panel_units:.6g}",
                "font-weight": "bold",
                "fill": "#000000",
                "data-final-font-size-pt": f"{panel_pt:g}",
            },
        )
        text.text = letter
    root.append(panel_group)

    root.set("viewBox", f"0 0 {view_width:g} {view_height:g}")
    root.set("width", f"{physical_width_mm:g}mm")
    root.set("height", f"{final_height_mm:.6g}mm")
    root.set("preserveAspectRatio", "xMidYMid meet")
    root.set("data-common-panel-width", "360")
    root.set("data-common-panel-height", "360")
    if columns == 4:
        root.set("data-common-panel-pitch-x", "440")
        root.set("data-common-panel-pitch-y", "480")
    else:
        # 冻结的 3x3 版(Fig. 1 / Fig. 4 / fig. S10)记的是单一 pitch 属性。
        root.set("data-common-panel-pitch", "480")
    root.set("data-final-width-mm", f"{physical_width_mm:g}")
    root.set("data-font-family", "Arial")
    root.set("data-font-standard-pt", f"{standard_pt:g}")
    root.set("data-font-contour-pt", f"{contour_pt:g}")
    root.set("data-font-panel-pt", f"{panel_pt:g}")
    root.set("data-heatmap-row-gap-user-units", "120")
    root.set("data-panel-letter-top-clearance-user-units", "48")
    root.set("data-shared-y-ticks-synchronised", "12")
    root.set("data-shared-row-labels-synchronised", "3")
    root.set("data-bottom-x-tick-count", f"{n_bottom:g}")
    root.set("data-submission-candidate", "true")
    tree.write(
        str(svg_path),
        encoding="utf-8",
        xml_declaration=True,
        pretty_print=False,
    )

    path_counts = normalize_matplotlib_path_text(
        svg_path,
        physical_width_mm=physical_width_mm,
        standard_pt=standard_pt,
        contour_pt=contour_pt,
    )
    return {
        "final_width_mm": physical_width_mm,
        "final_height_mm": final_height_mm,
        "viewBox": [0.0, 0.0, view_width, view_height],
        "panel_count": len(panels),
        "shared_label_count": len(labels),
        "path_text_groups": path_counts,
    }
