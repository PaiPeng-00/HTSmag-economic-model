"""
4.12_stitch_charging_heatload_timeseries.py

将不同参数组合（Table 1）在充电期间的热负荷随时间堆叠图拼接成一张大图。
运行温度：4.2 K / 10 K / 20 K；接头电阻：10 nΩ / 100 nΩ。
每个 (温度, 接头电阻) 组合生成一张 11 子图的大图，子图对应关系：
  (a) A1, (b) B1, (c) B2, (d) C1, (e) C2, (f) C3, (g) C4, (h) D1, (i) D2, (j) D3, (k) D4

布局：第一行 a b c，第二行 d e f g，第三行 h i j k；子图标签无括号。
下方图例直接拼接 4.10 生成的 heat_load_legend.svg，与 4.10 颜色、线条完全一致。

依赖：svgutils
    pip install svgutils
"""

from pathlib import Path
import re
from typing import Optional, List, Tuple

from fusion_tem import device as cfg

try:
    import svgutils.transform as sg
except ImportError as e:
    raise ImportError(
        "需要安装 svgutils 才能拼接 SVG，请运行: pip install svgutils"
    ) from e

# Table 1 配置与子图标签对应（与 4.10 / 4.11 一致）
# 配置名 -> (Npw, rhot) 见 4.10 configs；此处仅用于文件名
# 面板字母用大写: Science Advances 要求, 且下游 160 mm 定稿的 is_panel_label()
# 判据是 re.fullmatch(r"[A-L]", ...) —— 小写不会被识别为面板字母, 拿不到 9 pt 粗体。
# 2026-07-26 由小写改为大写。
TABLE1_LABEL_TO_CONFIG: List[Tuple[str, str]] = [
    ("A", "A1"), ("B", "B1"), ("C", "B2"), ("D", "C1"), ("E", "C2"),
    ("F", "C3"), ("G", "C4"), ("H", "D1"), ("I", "D2"), ("J", "D3"), ("K", "D4"),
]
# 温度与接头电阻组合（运行温度 20 K / 4.2 K / 10 K，接头电阻 10 / 100 nΩ）
TEMPERATURES_K = [4.2, 10.0, 20.0]
R_JOINT_NOHM_LIST = [10.0, 100.0]

HEATLOAD_BASE = Path(cfg.OUTPUTS_FIGURES_DIR) / "heatload"
OUTPUT_SUBDIR = "charging_timeseries_stitched"
# 布局：第一行 abc(3)，第二行 defg(4)，第三行 hijk(4)
ROWS_PANELS = [3, 4, 4]  # 每行子图个数
N_ROWS = 3
N_COLS = 4
# 图例：直接使用 4.10 生成的 heat_load_legend.svg，与 4.10 完全一致
LEGEND_MARGIN_BOTTOM = 16  # 图例下方留白（pt）
TOP_MARGIN = 28  # 拼接图上方留白（pt）


def _parse_size(size_str: str) -> float:
    if size_str is None:
        return 0.0
    m = re.findall(r"[0-9.]+", str(size_str))
    return float(m[0]) if m else 0.0


def _get_timeseries_path(T_op: float, R_nohm: float, config_name: str) -> Path:
    """与 4.10 输出路径一致。"""
    dir_name = f"Top={T_op}K_Rj={R_nohm}nOhm"
    return HEATLOAD_BASE / dir_name / f"config={config_name}" / f"{config_name}_charging_{T_op}K_timeseries.svg"


def _get_legend_path(T_op: float, R_nohm: float) -> Path:
    """4.10 生成的独立图例路径（与 4.10 save_legend_only 输出一致）。"""
    dir_name = f"Top={T_op}K_Rj={R_nohm}nOhm"
    return HEATLOAD_BASE / dir_name / "config=A1" / cfg.LEGEND_IMAGE_FILE


def stitch_charging_timeseries_one(T_op: float, R_nohm: float) -> Optional[Path]:
    """
    为指定 (T_op, R_joint) 拼接 11 张充电热负荷时间序列堆叠图，子图标签 a–k（无括号）。
    布局：第一行 a b c，第二行 d e f g，第三行 h i j k。
    """
    panel_entries = []  # (root, row_idx, col_idx, label)
    panel_w = None
    panel_h = None

    # 每行起始索引：第一行 0，第二行 3，第三行 7
    row_starts = [0, 3, 7]
    for idx, (label, config_name) in enumerate(TABLE1_LABEL_TO_CONFIG):
        svg_path = _get_timeseries_path(T_op, R_nohm, config_name)
        if not svg_path.exists():
            print(f"[warning] 未找到: {svg_path}")
            continue
        fig = sg.fromfile(str(svg_path))
        root = fig.getroot()
        size_w, size_h = fig.get_size()
        w = _parse_size(size_w)
        h = _parse_size(size_h)
        panel_w = w if panel_w is None else max(panel_w, w)
        panel_h = h if panel_h is None else max(panel_h, h)
        row_idx = 2 if idx >= 7 else (1 if idx >= 3 else 0)
        col_idx = idx - row_starts[row_idx]
        panel_entries.append((root, row_idx, col_idx, label))

    if not panel_entries or panel_w is None or panel_h is None:
        print(f"[warning] T_op={T_op}K Rj={R_nohm}nOhm 无足够子图，跳过。")
        return None

    margin_x = panel_w * 0.06
    margin_y = panel_h * 0.06
    label_font_size = 26
    top_label_height = label_font_size + 8

    # 图例：直接拼接 4.10 的 heat_load_legend.svg
    legend_path = _get_legend_path(T_op, R_nohm)
    legend_h = 0.0
    legend_w = 0.0
    legend_root = None
    if legend_path.exists():
        try:
            fig_legend = sg.fromfile(str(legend_path))
            legend_root = fig_legend.getroot()
            lw, lh = fig_legend.get_size()
            legend_w = _parse_size(lw)
            legend_h = _parse_size(lh)
        except Exception:
            legend_root = None
            legend_w = legend_h = 0.0
    legend_height = legend_h + LEGEND_MARGIN_BOTTOM if legend_root is not None else 0.0

    subplot_area_w = margin_x * (N_COLS + 1) + panel_w * N_COLS
    subplot_area_h = margin_y * (N_ROWS + 1) + panel_h * N_ROWS
    total_w = subplot_area_w
    total_h = TOP_MARGIN + top_label_height + subplot_area_h + legend_height

    fig_out = sg.SVGFigure(f"{total_w}pt", f"{total_h}pt")
    subplot_start_x = margin_x
    subplot_start_y = TOP_MARGIN + top_label_height

    placed_roots = []
    for root, row_idx, col_idx, label in panel_entries:
        x = subplot_start_x + col_idx * (panel_w + margin_x)
        y = subplot_start_y + row_idx * (panel_h + margin_y)
        root.moveto(x, y)
        placed_roots.append((root, label, x, y, row_idx, col_idx))
    fig_out.append([r[0] for r in placed_roots])

    if legend_root is not None:
        legend_x = max(0, (total_w - legend_w) / 2)
        legend_y = total_h - legend_height
        legend_root.moveto(legend_x, legend_y)
        fig_out.append([legend_root])

    out_dir = HEATLOAD_BASE / OUTPUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    T_str = f"{int(T_op)}K" if T_op in (10.0, 20.0) else f"{T_op}K"
    out_name = f"charging_heatload_timeseries_stitched_Top{T_str}_Rj{int(R_nohm)}nOhm.svg"
    out_path = out_dir / out_name
    fig_out.save(str(out_path))

    # 添加 (a)–(k) 标签
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

    for root, label, x, y, row_idx, col_idx in placed_roots:
        # 子图左上角内侧放置 (a), (b), ...
        label_x = x + 14
        label_y = y - label_font_size -4
        text_elem = ET.SubElement(labels_group, f"{{{svg_ns}}}text", {
            "x": str(label_x),
            "y": str(label_y),
            "font-family": "Arial, sans-serif",
            "font-size": str(label_font_size),
            "font-weight": "bold",
            "text-anchor": "start",
            "dominant-baseline": "hanging",
            "fill": "black",
        })
        text_elem.text = label

    root_svg.remove(labels_group)
    root_svg.append(labels_group)
    tree.write(str(out_path), encoding="utf-8", xml_declaration=True)
    print(f"已保存: {out_path}")
    return out_path


def stitch_all_charging_timeseries():
    """为所有 (温度, 接头电阻) 组合生成拼接图。"""
    if not HEATLOAD_BASE.exists():
        print(f"[错误] 目录不存在: {HEATLOAD_BASE}")
        print("请先运行 4.10_heatload_2bars.py 生成充电热负荷时间序列图。")
        return
    count = 0
    for T_op in TEMPERATURES_K:
        for R_nohm in R_JOINT_NOHM_LIST:
            p = stitch_charging_timeseries_one(T_op, R_nohm)
            if p is not None:
                count += 1
    print(f"\n共生成 {count} 张拼接图，目录: {HEATLOAD_BASE / OUTPUT_SUBDIR}")


if __name__ == "__main__":
    stitch_all_charging_timeseries()
