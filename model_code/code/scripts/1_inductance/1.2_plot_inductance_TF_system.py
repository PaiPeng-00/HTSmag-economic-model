# utility_plot_inductance.py
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import re
# 导入全局配置，以便我们知道要处理哪些文件
from fusion_tem import device as cfg
from fusion_tem.publication_svg import normalize_matplotlib_path_text
# 从我们的库中导入绘图函数
from fusion_tem.utils import plot_inductance_heatmap
import xml.etree.ElementTree as ET

try:
    import svgutils.transform as sg
except ImportError as e:
    raise ImportError(
        "需要安装 svgutils 才能拼接 SVG 图像，请先运行：\n\n"
        "    pip install svgutils\n"
    ) from e
# =============================================================================
# 全局绘图配置
# =============================================================================
# 1. 首先设定一个基础样式
plt.style.use('default')
plt.rcParams['font.family'] = 'Arial' 
plt.rcParams['axes.labelsize'] =20
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 20

# [新增] 设置刻度线方向朝内
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['savefig.transparent'] = True
# =============================================================================
NPW_LIST = [1, 20, 100]
# 温度列表：从config中获取，如果没有则使用默认值
TEMPERATURES = list(cfg.Ip_list.keys()) if hasattr(cfg, 'Ip_list') else [4.2, 10.0, 20.0]

def main():
    """
    主执行函数：
    读取已计算好的电感矩阵.xlsx文件，并重新生成对应的热力图。
    支持不同温度和不同Npw的组合。
    """
    print("--- 开始执行电感矩阵可视化脚本 ---")
    
    script_dir = Path(__file__).parent.resolve()
    # 定义电感矩阵数据所在的输入目录
    input_dir = script_dir / cfg.INDUCTANCE_OUTPUT_DIR
    out_dir = Path(cfg.OUTPUTS_FIGURES_DIR) / 'inductance'
    # 如果输出目录不存在，则创建
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
    if not input_dir.exists():
        print(f"错误: 找不到输入目录 '{input_dir}'。")
        print("请先运行 '01_calculate_inductance.py' 来生成电感矩阵数据。")
        return

    # 从全局配置读取要遍历的温度和Npw列表
    for temp in TEMPERATURES:
        # 获取该温度下的带材根数（用于缩放）
        ntape_coil = cfg.Nt_list.get(temp, cfg.N_TOTAL_TAPE) if hasattr(cfg, 'Nt_list') else cfg.N_TOTAL_TAPE
        
        for npw in NPW_LIST:
            # 1. 根据当前的 Npw，调用函数获取动态的 ng 值
            ng_current = cfg.get_ng_for_npw(npw)
            
            # 格式化温度字符串（4.2显示为4.2，10.0和20.0显示为整数）
            if temp == 10.0 or temp == 20.0:
                temp_str = str(int(temp))
            else:
                temp_str = str(temp)
            
            print(f"\n正在处理 Top = {temp_str}K, Npw = {npw} (ng = {ng_current}) 的情况...")

            # 2. 构建输入和输出文件的路径
            # 必须走器件配置: ARC 用 TF_system_L_matrix_ARC.xlsx (R2=0.64 m, CP4 生成),
            # SPARC 才用 TF_system_L_matrix.xlsx (R2=0.30 m)。
            # 2026-07-26 前此处硬编码读 SPARC 版, 见工作文档 §14.39。
            excel_path = input_dir / cfg.TF_SYSTEM_MATRIX
            heatmap_path = out_dir / f"Npw={npw}_Top={temp_str}K_heatmap.svg"

            # 3. 检查Excel文件是否存在
            if not excel_path.exists():
                print(f"  -> 警告: 未找到对应的Excel文件 '{excel_path.name}'，跳过。")
                continue

            # 4. 读取数据并绘图
            try:
                print(f"  -> 正在读取: {excel_path.name}")
                # 从Excel读取数据，并转换为Numpy矩阵
                m_pancakes = pd.read_excel(excel_path, header=None).values

                # 根据并绕根数、带材数缩放（使用该温度下的带材根数）
                m_pancakes = m_pancakes * (ntape_coil * cfg.NP / npw) ** 2
                print(f"  -> 矩阵第一行: {m_pancakes[0]}")
                print(f"  -> 正在生成图像: {heatmap_path.name}")
                plot_inductance_heatmap(
                    m_pancakes,
                    title=f'Inductance Matrix (Np={cfg.NP}, Npw={npw}, Top={temp_str}K)',
                    save_path=str(heatmap_path)
                )
                print("  -> ✅ 处理完成。")

            except Exception as e:
                print(f"  -> ❌ 处理失败: {e}")

    print("\n--- 所有可视化任务完成 ---")


def _parse_size(size_str: str) -> float:
    """
    将 SVG 中的 width/height 字符串（可能带单位，如 '432pt', '800px'）解析为 float 数值。
    仅保留数字和小数点。
    """
    if size_str is None:
        return 0.0
    m = re.findall(r"[0-9.]+", str(size_str))
    return float(m[0]) if m else 0.0


def stitch_inductance_heatmaps():
    """
    拼接电感矩阵热力图：
    - 布局：2列3行（竖着拼接）
    - 左边列：Npw=1的三个温度（从上到下：4.2K, 10K, 20K）- 编号 a, b, c
    - 右边列：Npw=20的三个温度（从上到下：4.2K, 10K, 20K）- 编号 d, e, f
    """
    print("\n--- 开始拼接电感矩阵热力图 ---")
    
    script_dir = Path(__file__).parent.resolve()
    inductance_dir = Path(cfg.OUTPUTS_FIGURES_DIR) / 'inductance'
    
    # 定义要拼接的子图
    # 左边列：Npw=1
    left_panels = [
        (1, 4.2, 'a'),  # (Npw, temp, label)
        (1, 10.0, 'b'),
        (1, 20.0, 'c'),
    ]
    # 右边列：Npw=20
    right_panels = [
        (20, 4.2, 'd'),
        (20, 10.0, 'e'),
        (20, 20.0, 'f'),
    ]
    
    n_rows = 3
    n_cols = 2
    
    # 存储 (root, row_idx, col_idx, label)
    panel_entries = []
    panel_w = None
    panel_h = None
    
    # 读取所有子图
    all_panels = [(left_panels, 0), (right_panels, 1)]  # (panels, col_idx)
    
    for panels, col_idx in all_panels:
        for row_idx, (npw, temp, label) in enumerate(panels):
            # 格式化温度字符串
            if temp == 10.0 or temp == 20.0:
                temp_str = str(int(temp))
            else:
                temp_str = str(temp)
            
            fname = inductance_dir / f"Npw={npw}_Top={temp_str}K_heatmap.svg"
            if not fname.exists():
                print(f"[warning] 未找到子图: {fname}")
                continue
            
            fig = sg.fromfile(str(fname))
            root = fig.getroot()
            
            # 记录所有子图中的最大宽高
            size_w, size_h = fig.get_size()
            w = _parse_size(size_w)
            h = _parse_size(size_h)
            panel_w = w if panel_w is None else max(panel_w, w)
            panel_h = h if panel_h is None else max(panel_h, h)
            
            panel_entries.append((root, row_idx, col_idx, label))
    
    if not panel_entries or panel_w is None or panel_h is None:
        print("没有找到任何可拼接的电感矩阵 SVG 子图，退出。")
        return None
    
    # 设置边距和间隔
    margin_x = panel_w * 0.05
    margin_y = panel_h * 0.05
    
    # 为标签预留空间
    label_font_size = 24
    top_margin = 30  # 顶部边距（用于子图标签）
    left_margin = 20  # 左边边距
    right_margin = 20  # 右边边距
    bottom_margin = 20  # 底部边距
    
    # 计算总尺寸
    subplot_area_w = margin_x * (n_cols + 1) + panel_w * n_cols
    subplot_area_h = margin_y * (n_rows + 1) + panel_h * n_rows
    total_w = subplot_area_w + left_margin + right_margin
    total_h = subplot_area_h + top_margin + bottom_margin
    
    # 构造大画布
    fig_out = sg.SVGFigure(f"{total_w}pt", f"{total_h}pt")
    print(f"total_w: {total_w}pt, total_h: {total_h}pt")
    
    # 按网格布局每个子图
    placed_roots = []
    for root, row_idx, col_idx, label in panel_entries:
        x = left_margin + margin_x + col_idx * (panel_w + margin_x)
        y = top_margin + margin_y + row_idx * (panel_h + margin_y)
        root.moveto(x, y)
        placed_roots.append(root)
    
    fig_out.append(placed_roots)
    
    # 保存拼接后的SVG
    out_path = inductance_dir / "inductance_matrix_grid_Npw1&20.svg"
    fig_out.save(str(out_path))
    
    # 手动修改 SVG 文件，添加标签
    tree = ET.parse(str(out_path))
    root_svg = tree.getroot()
    
    # 获取 SVG 命名空间
    svg_ns = None
    for prefix, uri in root_svg.attrib.items():
        if prefix.startswith('xmlns') and 'svg' in uri.lower():
            svg_ns = uri
            break
    if svg_ns is None:
        svg_ns = 'http://www.w3.org/2000/svg'
    
    ET.register_namespace('', svg_ns)
    
    # 设置根 SVG 元素的尺寸
    root_svg.set('width', f"{total_w}pt")
    root_svg.set('height', f"{total_h}pt")
    root_svg.set('viewBox', f"0 0 {total_w} {total_h}")
    
    # 创建标签 group
    labels_group = ET.SubElement(root_svg, f'{{{svg_ns}}}g', {'id': 'labels'})
    
    # 添加子图标签（a, b, c, d, e, f）
    for root, row_idx, col_idx, label in panel_entries:
        x = left_margin + margin_x + col_idx * (panel_w + margin_x)
        y = top_margin + margin_y + row_idx * (panel_h + margin_y)
        
        # 标签位置：子图左上角
        label_x = x + 20
        label_y = y + 30
        
        text_elem = ET.SubElement(labels_group, f'{{{svg_ns}}}text', {
            'x': str(label_x),
            'y': str(label_y),
            'font-family': 'Arial, sans-serif',
            'font-size': str(label_font_size),
            'font-weight': 'bold',
            'text-anchor': 'start',
            'dominant-baseline': 'top',
            'fill': 'black'
        })
        text_elem.text = f"({label})"
    
    # 确保 labels group 在最上层
    root_svg.remove(labels_group)
    root_svg.append(labels_group)
    
    # 保存文件
    tree.write(str(out_path), encoding='utf-8', xml_declaration=True)
    
    print(f"拼接完成，大图已保存为: {out_path}")
    print(f"布局：左边列（Npw=1）：a=4.2K, b=10K, c=20K；右边列（Npw=20）：d=4.2K, e=10K, f=20K")
    
    return out_path


def stitch_inductance_publication_panels():
    """Build the final two-panel Fig. S4 directly from upstream heat maps.

    The selected panels are the two bounding design cases used in the
    manuscript: 4.2 K with Npw=1 and 20 K with Npw=20.  Both panels are
    scaled uniformly, so square matrix cells and glyph proportions are
    preserved without downstream x/y compensation.
    """
    import cairosvg

    inductance_dir = Path(cfg.OUTPUTS_FIGURES_DIR) / "inductance"
    cases = (
        (1, 4.2, "A", 20.0, "4.2K_Npw1"),
        (20, 20.0, "B", 1215.0, "20K_Npw20"),
    )
    canvas_w = 1800.0
    canvas_h = 2440.0
    panel_scale = 1.17983870968
    panel_font_size = 35.7187  # 9 pt at the final 160-mm width
    output = inductance_dir / "inductance_matrix_grid_Npw1&20.svg"

    entries = []
    for npw, temp, letter, y, case_name in cases:
        temp_str = str(int(temp)) if temp in (10.0, 20.0) else str(temp)
        source = inductance_dir / f"Npw={npw}_Top={temp_str}K_heatmap.svg"
        if not source.is_file():
            raise FileNotFoundError(source)
        figure = sg.fromfile(str(source))
        width_raw, height_raw = figure.get_size()
        panel_w = _parse_size(width_raw)
        panel_h = _parse_size(height_raw)
        panel = figure.getroot()
        x = (canvas_w - panel_w * panel_scale) / 2.0
        panel.moveto(x, y, panel_scale, panel_scale)
        entries.append((panel, x, y, letter, case_name, panel_w, panel_h))

    figure_out = sg.SVGFigure("160mm", "216.889mm")
    figure_out.append([entry[0] for entry in entries])
    figure_out.save(str(output))

    tree = ET.parse(str(output))
    root_svg = tree.getroot()
    svg_ns = "http://www.w3.org/2000/svg"
    ET.register_namespace("", svg_ns)
    root_svg.set("width", "160mm")
    root_svg.set("height", "216.889mm")
    root_svg.set("viewBox", "0 0 1800 2440")
    root_svg.set("preserveAspectRatio", "xMidYMid meet")
    root_svg.set("data-upstream-publication-figure", "FigS4")
    root_svg.set("data-s4-selected-cases", "4.2K_Npw1;20K_Npw20")
    root_svg.set("data-s4-cell-aspect", "1.000000")
    root_svg.set("data-s4-horizontal-compression", "none")
    root_svg.set("data-s4-panel-scale", f"{panel_scale:.12g}")
    root_svg.set("data-s4-text-physical-aspect", "1.000000")
    root_svg.set("data-s4-text-compensation-wrapper-depth", "0")
    root_svg.set("data-s4-panel-letter-top-clearance", "20")

    artwork = root_svg.find(f"{{{svg_ns}}}g")
    if artwork is None or len(list(artwork)) != 2:
        raise RuntimeError("Fig. S4 publication stitch must contain two panels")
    for child, entry in zip(list(artwork), entries, strict=True):
        child.set("data-retained-inductance-case", entry[4])

    for old in root_svg.findall(f"{{{svg_ns}}}g[@id='labels']"):
        root_svg.remove(old)
    labels = ET.SubElement(root_svg, f"{{{svg_ns}}}g", {"id": "labels"})
    for _panel, x, y, letter, _case_name, _panel_w, _panel_h in entries:
        label = ET.SubElement(
            labels,
            f"{{{svg_ns}}}text",
            {
                "x": "217",
                "y": f"{y + panel_font_size:g}",
                "font-family": "Arial, Helvetica, sans-serif",
                "font-size": f"{panel_font_size:g}",
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
        s4_cell_pt=6.0,
    )
    cairosvg.svg2pdf(url=str(output), write_to=str(output.with_suffix(".pdf")))
    print(f"Publication Fig. S4 generated upstream: {output}; fonts={font_counts}")
    return output


if __name__ == "__main__":
    main()
    # 拼接电感矩阵热力图
    stitch_inductance_publication_panels()