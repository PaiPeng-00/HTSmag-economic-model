"""
2.6_stitch_charging_tf_3temps.py

将 outputs/figures/charging_tf 下三个不同运行温度的充电时间热力图拼接成一行，
并添加子图标签：(a) 4.2 K, (b) 10 K, (c) 20 K。

依赖：svgutils
    pip install svgutils
"""

from pathlib import Path
import re
from typing import Optional

import numpy as np
from fusion_tem import device as cfg
from fusion_tem.publication_svg import normalize_matplotlib_path_text

try:
    import svgutils.transform as sg
except ImportError as e:
    raise ImportError(
        "需要安装 svgutils 才能拼接 SVG，请运行: pip install svgutils"
    ) from e

# 充电热力图所在目录（与 2.5 输出一致）
CHARGING_TF_FIGURES_DIR = Path(cfg.OUTPUTS_FIGURES_DIR) / "charging_tf"
# 每个温度子目录内的 SVG 文件名（与 2.5 输出一致）
HEATMAP_FILENAME = "charging_time999_heatmap_contour_TF_system_Npw=1-200.svg"
# 三个温度及对应子图标签
TEMPERATURES = [
    (4.2, "A"),
    (10.0, "B"),
    (20.0, "C"),
]


def _parse_size(size_str: str) -> float:
    """将 SVG width/height 字符串解析为 float。"""
    if size_str is None:
        return 0.0
    m = re.findall(r"[0-9.]+", str(size_str))
    return float(m[0]) if m else 0.0


def _canonical_subdir_name(temp: float) -> str:
    """按 2.5 的命名模板，用当前器件工作点重建子目录名。"""
    case = cfg.TEMPERATURE_CASES[temp]
    return (f"Temp_{temp}K_Ip_{case['Ip']}A_"
            f"Ntape_coil_{case['Ntape_coil']}_charge999")


def _find_subdir_for_temp(base_dir: Path, temp: float) -> Optional[Path]:
    """定位该温度的充电热力图子目录。

    必须按 cfg.TEMPERATURE_CASES 的 Ip/Ntape_coil 精确匹配。历史上这里用
    `startswith(f"Temp_{temp}K_")` 取 iterdir() 的第一个匹配, 当同一温度下同时
    存在旧工作点目录时会挑错 —— 4.2 K 的 Ip_730.0A/Nt_719(旧 FEM 值)按字典序
    排在 Ip_750.0A/Nt_700(当前口径)之前, 导致 fig. S5 的 4.2 K 面板与 10 K/20 K
    面板不同源(差约 2%)。见工作文档 §14.40.4 缺陷 A。
    """
    exact = base_dir / _canonical_subdir_name(temp)
    if exact.is_dir():
        return exact

    stale = sorted(d.name for d in base_dir.iterdir()
                   if d.is_dir() and d.name.startswith(f"Temp_{temp}K_"))
    if stale:
        raise FileNotFoundError(
            f"未找到 {temp} K 的当前工作点目录 {exact.name}; "
            f"该温度下只有非当前口径的目录 {stale}。"
            "请先用当前器件配置重跑 2.5, 不要回退到旧目录。"
        )
    return None


def stitch_charging_tf_3temps():
    """将 4.2 K / 10 K / 20 K 三张充电热力图拼接为一行，并添加 (a)(b)(c) 标签。"""
    if not CHARGING_TF_FIGURES_DIR.exists():
        print(f"[错误] 目录不存在: {CHARGING_TF_FIGURES_DIR}")
        print("请先运行 2.5_charge_time999_TF_system_all_Npw=1-200.py 生成各温度热力图。")
        return None

    panel_entries = []  # (root, col_idx, temp, label)
    panel_w = None
    panel_h = None

    for col_idx, (temp, letter) in enumerate(TEMPERATURES):
        subdir = _find_subdir_for_temp(CHARGING_TF_FIGURES_DIR, temp)
        if subdir is None:
            print(f"[warning] 未找到温度 {temp} K 的子目录（如 Temp_{temp}K_*_charge999）")
            continue
        svg_path = subdir / HEATMAP_FILENAME
        if not svg_path.exists():
            print(f"[warning] 未找到文件: {svg_path}")
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
        print("没有找到足够的热力图子图，无法拼接。")
        return None

    n_cols = len(panel_entries)
    margin_x = panel_w * 0.08
    label_font_size = 22
    top_label_height = label_font_size + 12  # (a)(b)(c) 及温度标签

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

    # 添加 (a)(b)(c) 和温度标签
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
        # 每个子图左上角（带少量内边距）
        x_left = subplot_start_x + col_idx * (panel_w + margin_x) + 8  # 8pt内边距
        y_top = top_label_height - 10  # 略低于上边缘的内边距
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
        # 温度标签：4.2 K, 10 K, 20 K
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

    print(f"拼接完成: {out_path}")
    print("布局: (a) 4.2 K, (b) 10 K, (c) 20 K")
    return out_path


def stitch_charging_tf_3temps_publication():
    """Build the final vertical Fig. S5 directly from the three upstream SVGs."""
    import cairosvg
    import xml.etree.ElementTree as ET

    if not CHARGING_TF_FIGURES_DIR.exists():
        raise FileNotFoundError(CHARGING_TF_FIGURES_DIR)

    canvas_w = 1100.0
    canvas_h = 1375.0
    panel_x = 262.0
    panel_y = (45.0, 485.0, 925.0)
    label_x = 233.0
    label_y = (48.8281, 488.828, 928.828)
    label_font_size = 21.8281  # 9 pt at the final 160-mm width
    entries = []

    for row, (temp, letter) in enumerate(TEMPERATURES):
        subdir = _find_subdir_for_temp(CHARGING_TF_FIGURES_DIR, temp)
        if subdir is None:
            raise FileNotFoundError(f"No charging-temperature directory for {temp} K")
        source = subdir / HEATMAP_FILENAME
        if not source.is_file():
            raise FileNotFoundError(source)
        figure = sg.fromfile(str(source))
        panel = figure.getroot()
        panel.moveto(panel_x, panel_y[row], 1.0, 1.0)
        entries.append((panel, letter))

    output = CHARGING_TF_FIGURES_DIR / "charging_time999_heatmap_contour_TF_system_3temps_stitched.svg"
    figure_out = sg.SVGFigure("160mm", "200mm")
    figure_out.append([entry[0] for entry in entries])
    figure_out.save(str(output))

    tree = ET.parse(str(output))
    root_svg = tree.getroot()
    svg_ns = "http://www.w3.org/2000/svg"
    ET.register_namespace("", svg_ns)
    root_svg.set("width", "160mm")
    root_svg.set("height", "200mm")
    root_svg.set("viewBox", "0 10 1100 1375")
    root_svg.set("preserveAspectRatio", "xMidYMid meet")
    root_svg.set("data-upstream-publication-figure", "FigS5")
    root_svg.set("data-s5-panel-layout", "three-rows")
    root_svg.set("data-s5-panel-row-gap", "80")
    root_svg.set("data-s5-vertical-crop", "top:10;bottom:65")
    root_svg.set("data-s5-panel-letter-baseline", "alphabetic")
    root_svg.set("data-s5-panel-letter-top-clearance", "20")

    for old in root_svg.findall(f"{{{svg_ns}}}g[@id='panel_labels']"):
        root_svg.remove(old)
    labels = ET.SubElement(root_svg, f"{{{svg_ns}}}g", {"id": "panel_labels"})
    for row, (_panel, letter) in enumerate(entries):
        label = ET.SubElement(
            labels,
            f"{{{svg_ns}}}text",
            {
                "x": f"{label_x:g}",
                "y": f"{label_y[row]:g}",
                "font-family": "Arial, Helvetica, sans-serif",
                "font-size": f"{label_font_size:g}",
                "font-weight": "bold",
                "text-anchor": "start",
                "dominant-baseline": "alphabetic",
                "fill": "#000000",
                "data-final-font-size-pt": "9",
            },
        )
        label.text = letter

    tree.write(str(output), encoding="utf-8", xml_declaration=True)
    font_counts = normalize_matplotlib_path_text(
        output,
        physical_width_mm=160.0,
        standard_pt=7.0,
        contour_pt=6.0,
    )
    cairosvg.svg2pdf(url=str(output), write_to=str(output.with_suffix(".pdf")))
    print(f"Publication Fig. S5 generated upstream: {output}; fonts={font_counts}")
    return output


if __name__ == "__main__":
    stitch_charging_tf_3temps_publication()
