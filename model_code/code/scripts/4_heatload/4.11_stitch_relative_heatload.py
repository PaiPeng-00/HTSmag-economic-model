"""
4.11_stitch_relative_heatload.py

将相对热负荷图组合成大图。
生成3个大图，分别对应 4.2K、10K、20K。
每个大图有2行，每行有2个子图区域（左右各一个）：
- 第1行左边（a）：静态，从左到右A1，B1, C1, D1
- 第1行右边（b）：运行，从左到右A1，B1, C1, D1
- 第2行左边（c）：charge状态的峰值热负荷，控制并绕根数一样，从左到右C1, C2, C3, C4
- 第2行右边（d）：charge状态的峰值热负荷，控制匝间电阻率一样，从左到右A1, B1, C1, D1

子图标号（a, b, c, d）放在每个子图区域的左上角。

依赖：svgutils
    pip install svgutils
"""

from pathlib import Path
import re

from fusion_tem import device as cfg

try:
    import svgutils.transform as sg
except ImportError as e:
    raise ImportError(
        "需要安装 svgutils 才能拼接 SVG 图像，请先运行：\n\n"
        "    pip install svgutils\n"
    ) from e


def _parse_size(size_str: str) -> float:
    """
    将 SVG 中的 width/height 字符串（可能带单位，如 '432pt', '800px'）解析为 float 数值。
    仅保留数字和小数点。
    """
    if size_str is None:
        return 0.0
    m = re.findall(r"[0-9.]+", str(size_str))
    return float(m[0]) if m else 0.0


def stitch_relative_heatload_grid(T_op: float, output_dir: Path):
    """
    为指定温度组合相对热负荷图。
    
    布局：
    - 第1行左边（a）：静态，从左到右A1，B1, C1, D1
    - 第1行右边（b）：运行，从左到右A1，B1, C1, D1
    - 第2行左边（c）：charge状态的峰值热负荷，控制并绕根数一样，从左到右C1, C2, C3, C4
    - 第2行右边（d）：charge状态的峰值热负荷，控制匝间电阻率一样，从左到右A1, B1, C1, D1
    
    Args:
        T_op: 运行温度 (4.2, 10.0, 20.0)
        output_dir: 输出目录
    """
    # 定义4个子图区域的配置
    # (a) 第1行左边：静态，A1, B1, C1, D1
    panel_a_configs = ['A1', 'B1', 'C1', 'D1']
    panel_a_mode = 'static'
    
    # (b) 第1行右边：运行，A1, B1, C1, D1
    panel_b_configs = ['A1', 'B1', 'C1', 'D1']
    panel_b_mode = 'operation'
    
    # (c) 第2行左边：charge峰值，控制并绕根数一样，C1, C2, C3, C4
    panel_c_configs = ['C1', 'C2', 'C3', 'C4']
    panel_c_mode = 'charging'
    
    # (d) 第2行右边：charge峰值，控制匝间电阻率一样，A1, B1, C1, D1
    panel_d_configs = ['A1', 'B1', 'C1', 'D1']
    panel_d_mode = 'charging'
    
    # 基础路径（与 4.10_heatload_2bars.py 中的路径格式保持一致）
    # 默认使用 R_joint = 10e-9 (10 nOhm)
    R_p2p_joint = 10e-9
    base_dir = Path(cfg.OUTPUTS_FIGURES_DIR) / "heatload" / f"Top={T_op}K_Rj={R_p2p_joint*1e9}nOhm"
    
    # 存储所有子图：(panel_label, config_name, root, panel_row, panel_col, subplot_col)
    # panel_row: 0=第1行, 1=第2行
    # panel_col: 0=左边, 1=右边
    # subplot_col: 在该面板内的列索引（0-3）
    # 面板字母大写(SA 规定; 下游 is_panel_label 只认 [A-L])。2026-07-26 改。
    all_panels = [
        ('A', panel_a_configs, panel_a_mode, 0, 0),  # 第1行左边
        ('B', panel_b_configs, panel_b_mode, 0, 1),  # 第1行右边
        ('C', panel_c_configs, panel_c_mode, 1, 0),  # 第2行左边
        ('D', panel_d_configs, panel_d_mode, 1, 1),  # 第2行右边
    ]
    
    panel_entries = []
    panel_w = None   # 单个子图的宽度
    panel_h = None   # 单个子图的高度
    
    # 加载所有子图
    for panel_label, configs_list, mode, panel_row, panel_col in all_panels:
        for subplot_col, config_name in enumerate(configs_list):
            file_path = base_dir / f"config={config_name}" / f"{config_name}_{mode}_{T_op}K_relative.svg"
            
            if not file_path.exists():
                print(f"  [警告] 未找到子图: {file_path}")
                continue
            
            try:
                fig = sg.fromfile(str(file_path))
                root = fig.getroot()
                
                # 记录所有子图中的最大宽高
                size_w, size_h = fig.get_size()
                w = _parse_size(size_w)
                h = _parse_size(size_h)
                panel_w = w if panel_w is None else max(panel_w, w)
                panel_h = h if panel_h is None else max(panel_h, h)
                
                panel_entries.append((panel_label, config_name, root, panel_row, panel_col, subplot_col))
            except Exception as e:
                print(f"  [错误] 读取文件失败 {file_path}: {e}")
                continue
    
    if not panel_entries or panel_w is None or panel_h is None:
        print(f"  没有找到任何可拼接的相对热负荷 SVG 子图（T_op={T_op}K），退出。")
        return
    
    # 布局参数
    n_panel_rows = 2  # 2行面板
    n_panel_cols = 2  # 每行2个面板（左右）
    n_subplots_per_panel = 4  # 每个面板内4个子图
    
    # 子图之间的间距（同一面板内）
    subplot_margin_x = panel_w * 0.08  # 增大同一面板内子图之间的间距
    # 面板之间的间距（不同面板之间）
    panel_margin_x = panel_w * 0.3  # 增大左右面板之间的间距
    panel_margin_y = panel_h * 0.1  # 增大上下行之间的间距
    
    # 计算每个面板的宽度（包含4个子图）
    panel_area_w = subplot_margin_x * (n_subplots_per_panel - 1) + panel_w * n_subplots_per_panel
    
    # 边距
    label_font_size = 24
    panel_label_font_size = 32  # 子图标题字号（a, b, c, d）
    left_margin = 40
    top_margin = 50
    bottom_margin = 30
    right_margin = 30
    
    # 查找图例文件（使用任意一个配置的图例文件，它们应该是一样的）
    legend_file = base_dir / "config=A1" / cfg.LEGEND_IMAGE_FILE
    legend_h = 0
    legend_w = 0
    legend_fig = None
    if legend_file.exists():
        try:
            legend_fig = sg.fromfile(str(legend_file))
            legend_size_w, legend_size_h = legend_fig.get_size()
            legend_w = _parse_size(legend_size_w)
            legend_h = _parse_size(legend_size_h)
            print(f"  找到图例文件: {legend_file}, 尺寸: {legend_w}pt x {legend_h}pt")
        except Exception as e:
            print(f"  [警告] 读取图例文件失败: {e}")
            legend_h = 0
            legend_w = 0
            legend_fig = None
    else:
        print(f"  [警告] 未找到图例文件: {legend_file}")
    
    # 图例和内容之间的间距
    legend_caption_spacing = 20
    
    # 计算总尺寸（包含图例和描述文字）
    total_w = left_margin + panel_area_w + panel_margin_x + panel_area_w + right_margin
    # 如果图例宽度超过可用宽度，使用缩放后的高度
    available_width = total_w - left_margin - right_margin
    if legend_w > available_width and legend_h > 0:
        legend_scale = available_width / legend_w
        legend_h_effective = legend_h * legend_scale
        # 缩放会让图例字号偏离 4.10 的 FONT_CONFIG 设定，和面板内字号不一致。
        # 正常情况下 4.10 的 save_legend_only 已按 LEGEND_MAX_WIDTH_PT 自动选好列数，
        # 走不到这里；真走到了说明那个上限该调小。
        print(f"  [警告] 图例宽 {legend_w:.0f}pt > 可用 {available_width:.0f}pt，"
              f"将被缩放 {legend_scale:.3f}× —— 图例字号会与面板不一致，"
              f"请调小 4.10 的 LEGEND_MAX_WIDTH_PT")
    else:
        legend_h_effective = legend_h
    
    total_h = (top_margin + panel_h + panel_margin_y + panel_h + 
               legend_caption_spacing + legend_h_effective + 
               bottom_margin)
    
    # 构造大画布
    fig_out = sg.SVGFigure(f"{total_w}pt", f"{total_h}pt")
    print(f"  T_op={T_op}K: total_w={total_w}pt, total_h={total_h}pt")
    
    # 布局所有子图
    placed_roots = []
    for panel_label, config_name, root, panel_row, panel_col, subplot_col in panel_entries:
        # 计算该子图在整个画布中的位置
        # 面板的起始x位置
        if panel_col == 0:  # 左边面板
            panel_start_x = left_margin
        else:  # 右边面板
            panel_start_x = left_margin + panel_area_w + panel_margin_x
        
        # 面板的起始y位置
        panel_start_y = top_margin + panel_row * (panel_h + panel_margin_y)
        
        # 子图在面板内的x位置
        subplot_x_in_panel = subplot_col * (panel_w + subplot_margin_x)
        
        # 最终位置
        x = panel_start_x + subplot_x_in_panel
        y = panel_start_y
        
        root.moveto(x, y)
        placed_roots.append(root)
    
    # 添加图例（如果存在）
    if legend_fig is not None and legend_h > 0:
        try:
            legend_root = legend_fig.getroot()
            # 如果图例宽度超过画布宽度，需要缩放
            available_width = total_w - left_margin - right_margin
            if legend_w > available_width:
                scale_factor = available_width / legend_w
                legend_w_scaled = legend_w * scale_factor
                legend_h_scaled = legend_h * scale_factor
                # 使用 svgutils 的 scale 方法
                legend_root.scale(scale_factor, scale_factor)
                print(f"  图例宽度超出，缩放因子: {scale_factor:.3f}")
            else:
                legend_w_scaled = legend_w
                legend_h_scaled = legend_h
            
            # 图例位置：底部，居中
            legend_x = (total_w - legend_w_scaled) / 2
            legend_y = top_margin + panel_h + panel_margin_y + panel_h + legend_caption_spacing
            legend_root.moveto(legend_x, legend_y)
            placed_roots.append(legend_root)
        except Exception as e:
            print(f"  [警告] 添加图例失败: {e}")
    
    fig_out.append(placed_roots)
    
    # 保存输出文件
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"relative_heatload_grid_{T_op}K.svg"
    fig_out.save(str(out_path))
    
    # 保存后，手动修改 SVG 文件，添加子图标号（a, b, c, d）
    import xml.etree.ElementTree as ET
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
    
    # 注册命名空间
    ET.register_namespace('', svg_ns)
    
    # 设置根 SVG 元素的尺寸
    root_svg.set('width', f"{total_w}pt")
    root_svg.set('height', f"{total_h}pt")
    root_svg.set('viewBox', f"0 0 {total_w} {total_h}")
    
    # 创建标签 group
    labels_group = ET.SubElement(root_svg, f'{{{svg_ns}}}g', {'id': 'labels'})
    
    # 添加文本样式
    def create_text(x, y, text, font_size=label_font_size, anchor='start', baseline='top', bold=True):
        """创建文本元素"""
        text_elem = ET.SubElement(labels_group, f'{{{svg_ns}}}text', {
            'x': str(x),
            'y': str(y),
            'font-family': 'Arial, sans-serif',
            'font-size': str(font_size),
            'text-anchor': anchor,
            'dominant-baseline': baseline,
            'fill': 'black'
        })
        if bold:
            text_elem.set('font-weight', 'bold')
        text_elem.text = text
        return text_elem
    
    # 为每个面板添加标号（a, b, c, d）在左上角
    panel_positions = {
        'A': (0, 0),  # 第1行左边
        'B': (0, 1),  # 第1行右边
        'C': (1, 0),  # 第2行左边
        'D': (1, 1),  # 第2行右边
    }
    
    for panel_label, (panel_row, panel_col) in panel_positions.items():
        # 计算面板的起始位置
        if panel_col == 0:  # 左边面板
            panel_start_x = left_margin
        else:  # 右边面板
            panel_start_x = left_margin + panel_area_w + panel_margin_x
        
        panel_start_y = top_margin + panel_row * (panel_h + panel_margin_y)
        
        # 标号位置：面板左上角稍微偏移
        label_x = panel_start_x + 5
        label_y = panel_start_y + 5
        
        create_text(label_x, label_y, panel_label, font_size=panel_label_font_size, 
                   anchor='start', baseline='top', bold=True)
    
    # 确保 labels group 在根元素的最后（确保它在最上层显示）
    root_svg.remove(labels_group)
    root_svg.append(labels_group)
    
    # 计算实际内容边界，裁剪空白边
    # 基于已知的元素位置计算边界框
    min_x = float('inf')
    min_y = float('inf')
    max_x = float('-inf')
    max_y = float('-inf')
    
    # 1. 子图区域边界
    # 最左边的子图
    min_x = min(min_x, left_margin)
    # 最右边的子图
    max_x = max(max_x, left_margin + panel_area_w + panel_margin_x + panel_area_w)
    # 最上边的子图
    min_y = min(min_y, top_margin)
    # 最下边的子图
    max_y = max(max_y, top_margin + panel_h + panel_margin_y + panel_h)
    
    # 2. 图例区域边界（如果存在）
    if legend_fig is not None and legend_h > 0:
        available_width = total_w - left_margin - right_margin
        if legend_w > available_width:
            legend_w_effective = available_width
            legend_h_effective = legend_h * (available_width / legend_w)
        else:
            legend_w_effective = legend_w
            legend_h_effective = legend_h
        
        legend_x = (total_w - legend_w_effective) / 2
        legend_y = top_margin + panel_h + panel_margin_y + panel_h + legend_caption_spacing
        
        min_x = min(min_x, legend_x)
        max_x = max(max_x, legend_x + legend_w_effective)
        min_y = min(min_y, legend_y)
        max_y = max(max_y, legend_y + legend_h_effective)
    
    # 3. 子图标题（a, b, c, d）的位置
    for panel_label, (panel_row, panel_col) in panel_positions.items():
        if panel_col == 0:
            panel_start_x = left_margin
        else:
            panel_start_x = left_margin + panel_area_w + panel_margin_x
        panel_start_y = top_margin + panel_row * (panel_h + panel_margin_y)
        label_x = panel_start_x + 5
        label_y = panel_start_y + 5
        # 估算标题文本大小
        label_width = panel_label_font_size * 0.6
        label_height = panel_label_font_size * 1.2
        min_x = min(min_x, label_x)
        max_x = max(max_x, label_x + label_width)
        min_y = min(min_y, label_y)
        max_y = max(max_y, label_y + label_height)
    
    # 添加边距（留出足够的空白边距）
    padding = 40
    min_x = max(0, min_x - padding)
    min_y = max(0, min_y - padding)
    max_x = min(total_w, max_x + padding)
    max_y = min(total_h, max_y + padding)
    
    # 计算裁剪后的尺寸
    cropped_w = max_x - min_x
    cropped_h = max_y - min_y
    
    # 更新viewBox以裁剪空白边
    root_svg.set('viewBox', f"{min_x} {min_y} {cropped_w} {cropped_h}")
    root_svg.set('width', f"{cropped_w}pt")
    root_svg.set('height', f"{cropped_h}pt")
    
    print(f"  裁剪空白边: 原始尺寸 {total_w:.1f}pt x {total_h:.1f}pt -> 裁剪后 {cropped_w:.1f}pt x {cropped_h:.1f}pt")
    
    # 保存文件
    tree.write(str(out_path), encoding='utf-8', xml_declaration=True)
    print(f"  [OK] 组合图已保存: {out_path}")


def stitch_operation_comparison_20K(output_dir: Path):
    """
    生成20K温度下，不同接头电阻的operation模式对比图。
    
    布局：
    - 第1行左边（a）：接头电阻10纳欧，从左到右A1, B1, C1, D1
    - 第1行右边（b）：接头电阻100纳欧，从左到右A1, B1, C1, D1
    - 底部：图例
    
    Args:
        output_dir: 输出目录
    """
    T_op = 20.0
    mode = 'operation'
    configs = ['A1', 'B1', 'C1', 'D1']
    
    # 定义两个面板：10纳欧和100纳欧
    R_joint_10nOhm = 10e-9
    R_joint_100nOhm = 100e-9
    
    # 基础路径（与 4.10_heatload_2bars.py 中的路径格式保持一致）
    # 路径格式：Top={T_op}K_Rj={R_p2p_joint*1e9}nOhm
    base_dir_10nOhm = Path(cfg.OUTPUTS_FIGURES_DIR) / "heatload" / f"Top={T_op}K_Rj={R_joint_10nOhm*1e9}nOhm"
    base_dir_100nOhm = Path(cfg.OUTPUTS_FIGURES_DIR) / "heatload" / f"Top={T_op}K_Rj={R_joint_100nOhm*1e9}nOhm"
    
    panel_entries = []
    panel_w = None
    panel_h = None
    
    # 加载10纳欧的子图（左边，a）
    for subplot_col, config_name in enumerate(configs):
        file_path = base_dir_10nOhm / f"config={config_name}" / f"{config_name}_{mode}_{T_op}K_relative.svg"
        if not file_path.exists():
            print(f"  [警告] 未找到子图: {file_path}")
            continue
        try:
            fig = sg.fromfile(str(file_path))
            root = fig.getroot()
            size_w, size_h = fig.get_size()
            w = _parse_size(size_w)
            h = _parse_size(size_h)
            panel_w = w if panel_w is None else max(panel_w, w)
            panel_h = h if panel_h is None else max(panel_h, h)
            panel_entries.append(('A', config_name, root, 0, 0, subplot_col))
        except Exception as e:
            print(f"  [错误] 读取文件失败 {file_path}: {e}")
    
    # 加载100纳欧的子图（右边，b）
    for subplot_col, config_name in enumerate(configs):
        file_path = base_dir_100nOhm / f"config={config_name}" / f"{config_name}_{mode}_{T_op}K_relative.svg"
        if not file_path.exists():
            print(f"  [警告] 未找到子图: {file_path}")
            continue
        try:
            fig = sg.fromfile(str(file_path))
            root = fig.getroot()
            size_w, size_h = fig.get_size()
            w = _parse_size(size_w)
            h = _parse_size(size_h)
            panel_w = w if panel_w is None else max(panel_w, w)
            panel_h = h if panel_h is None else max(panel_h, h)
            panel_entries.append(('B', config_name, root, 0, 1, subplot_col))
        except Exception as e:
            print(f"  [错误] 读取文件失败 {file_path}: {e}")
    
    if not panel_entries or panel_w is None or panel_h is None:
        print(f"  没有找到任何可拼接的相对热负荷 SVG 子图（20K operation），退出。")
        return
    
    # 布局参数
    n_panel_rows = 1
    n_panel_cols = 2
    n_subplots_per_panel = 4
    
    # 子图之间的间距
    subplot_margin_x = panel_w * 0.08  # 增大同一面板内子图之间的间距
    # 面板之间的间距
    panel_margin_x = panel_w * 0.3  # 增大左右面板之间的间距
    panel_margin_y = panel_h * 0.1  # 增大上下行之间的间距
    
    # 计算每个面板的宽度
    panel_area_w = subplot_margin_x * (n_subplots_per_panel - 1) + panel_w * n_subplots_per_panel
    
    # 边距
    panel_label_font_size = 32
    left_margin = 40
    top_margin = 50
    bottom_margin = 30
    right_margin = 30
    
    # 查找图例文件
    legend_file = base_dir_10nOhm / "config=A1" / cfg.LEGEND_IMAGE_FILE
    legend_h = 0
    legend_w = 0
    legend_fig = None
    if legend_file.exists():
        try:
            legend_fig = sg.fromfile(str(legend_file))
            legend_size_w, legend_size_h = legend_fig.get_size()
            legend_w = _parse_size(legend_size_w)
            legend_h = _parse_size(legend_size_h)
            print(f"  找到图例文件: {legend_file}, 尺寸: {legend_w}pt x {legend_h}pt")
        except Exception as e:
            print(f"  [警告] 读取图例文件失败: {e}")
            legend_fig = None
    
    # 图例和内容之间的间距
    legend_spacing = 50
    
    # 计算总尺寸
    total_w = left_margin + panel_area_w + panel_margin_x + panel_area_w + right_margin
    total_h = (top_margin + panel_h + 
               legend_spacing + legend_h + 
               bottom_margin)
    
    # 构造大画布
    fig_out = sg.SVGFigure(f"{total_w}pt", f"{total_h}pt")
    print(f"  Operation对比图: total_w={total_w}pt, total_h={total_h}pt")
    
    # 布局所有子图
    placed_roots = []
    for panel_label, config_name, root, panel_row, panel_col, subplot_col in panel_entries:
        if panel_col == 0:  # 左边面板
            panel_start_x = left_margin
        else:  # 右边面板
            panel_start_x = left_margin + panel_area_w + panel_margin_x
        
        panel_start_y = top_margin
        subplot_x_in_panel = subplot_col * (panel_w + subplot_margin_x)
        
        x = panel_start_x + subplot_x_in_panel
        y = panel_start_y
        
        root.moveto(x, y)
        placed_roots.append(root)
    
    # 添加图例
    if legend_fig is not None and legend_h > 0:
        try:
            legend_root = legend_fig.getroot()
            available_width = total_w - left_margin - right_margin
            if legend_w > available_width:
                scale_factor = available_width / legend_w
                legend_w_scaled = legend_w * scale_factor
                legend_root.scale(scale_factor, scale_factor)
                print(f"  图例宽度超出，缩放因子: {scale_factor:.3f}")
            else:
                legend_w_scaled = legend_w
            
            legend_x = (total_w - legend_w_scaled) / 2
            legend_y = top_margin + panel_h + legend_spacing
            legend_root.moveto(legend_x, legend_y)
            placed_roots.append(legend_root)
        except Exception as e:
            print(f"  [警告] 添加图例失败: {e}")
    
    fig_out.append(placed_roots)
    
    # 保存输出文件
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "operation_comparison_20K_Rj_10vs100nOhm.svg"
    fig_out.save(str(out_path))
    
    # 添加标签和裁剪
    import xml.etree.ElementTree as ET
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
    
    root_svg.set('width', f"{total_w}pt")
    root_svg.set('height', f"{total_h}pt")
    root_svg.set('viewBox', f"0 0 {total_w} {total_h}")
    
    # 创建标签 group
    labels_group = ET.SubElement(root_svg, f'{{{svg_ns}}}g', {'id': 'labels'})
    
    def create_text(x, y, text, font_size=panel_label_font_size, anchor='start', baseline='top', bold=True):
        text_elem = ET.SubElement(labels_group, f'{{{svg_ns}}}text', {
            'x': str(x),
            'y': str(y),
            'font-family': 'Arial, sans-serif',
            'font-size': str(font_size),
            'text-anchor': anchor,
            'dominant-baseline': baseline,
            'fill': 'black'
        })
        if bold:
            text_elem.set('font-weight', 'bold')
        text_elem.text = text
        return text_elem
    
    # 添加子图标题（a, b）
    panel_positions = {
        'A': (0, 0),  # 左边
        'B': (0, 1),  # 右边
    }
    
    for panel_label, (panel_row, panel_col) in panel_positions.items():
        if panel_col == 0:
            panel_start_x = left_margin
        else:
            panel_start_x = left_margin + panel_area_w + panel_margin_x
        panel_start_y = top_margin
        label_x = panel_start_x + 5
        label_y = panel_start_y + 5
        create_text(label_x, label_y, panel_label, font_size=panel_label_font_size, 
                   anchor='start', baseline='top', bold=True)
    
    # 计算边界并裁剪
    min_x = left_margin
    max_x = left_margin + panel_area_w + panel_margin_x + panel_area_w
    min_y = top_margin
    max_y = top_margin + panel_h + legend_spacing + legend_h
    
    # 子图标题
    for panel_label, (panel_row, panel_col) in panel_positions.items():
        if panel_col == 0:
            panel_start_x = left_margin
        else:
            panel_start_x = left_margin + panel_area_w + panel_margin_x
        label_x = panel_start_x + 5
        label_y = top_margin + 5
        label_width = panel_label_font_size * 0.6
        label_height = panel_label_font_size * 1.2
        min_x = min(min_x, label_x)
        max_x = max(max_x, label_x + label_width)
        min_y = min(min_y, label_y)
        max_y = max(max_y, label_y + label_height)
    
    padding = 40
    min_x = max(0, min_x - padding)
    min_y = max(0, min_y - padding)
    max_x = min(total_w, max_x + padding)
    max_y = min(total_h, max_y + padding)
    
    cropped_w = max_x - min_x
    cropped_h = max_y - min_y
    
    root_svg.set('viewBox', f"{min_x} {min_y} {cropped_w} {cropped_h}")
    root_svg.set('width', f"{cropped_w}pt")
    root_svg.set('height', f"{cropped_h}pt")
    
    print(f"  裁剪空白边: 原始尺寸 {total_w:.1f}pt x {total_h:.1f}pt -> 裁剪后 {cropped_w:.1f}pt x {cropped_h:.1f}pt")
    
    root_svg.remove(labels_group)
    root_svg.append(labels_group)
    
    tree.write(str(out_path), encoding='utf-8', xml_declaration=True)
    print(f"  [OK] Operation对比图已保存: {out_path}")


def generate_readme(output_dir: Path):
    """
    生成README文档，记录所有组合图的生成条件和子图来源。
    
    Args:
        output_dir: 输出目录
    """
    readme_path = output_dir / "README.md"
    
    # 获取当前日期
    from datetime import datetime
    current_date = datetime.now().strftime("%Y年%m月%d日")
    
    readme_content = f"""# 相对热负荷组合图说明文档

本文档记录了 `relative_heatload_grids` 目录下所有组合图的生成条件、参数和子图来源。

**文档生成日期：** {current_date}

## 生成脚本

所有组合图由 `4.11_stitch_relative_heatload.py` 脚本生成。

---

## 输出文件列表

### 1. `relative_heatload_grid_4.2K.svg`
**生成条件：**
- 运行温度：4.2 K
- 接头电阻：10 nΩ (R_p2p_joint = 10e-9 Ω)

**布局说明：**
- **第1行左边（子图 a）**：静态模式（static mode）
  - 配置：A1, B1, C1, D1（从左到右）
- **第1行右边（子图 b）**：运行模式（operation mode）
  - 配置：A1, B1, C1, D1（从左到右）
- **第2行左边（子图 c）**：充电峰值热负荷（charging peak heat load）
  - 控制变量：并绕根数（Npw）相同
  - 配置：C1, C2, C3, C4（从左到右）
- **第2行右边（子图 d）**：充电峰值热负荷（charging peak heat load）
  - 控制变量：匝间电阻率（rhot）相同
  - 配置：A1, B1, C1, D1（从左到右）

**子图文件来源：**
```
{cfg.OUTPUTS_FIGURES_DIR}/heatload/Top=4.2K_Rj=10.0nOhm/config={{配置名}}/{{配置名}}_{{模式}}_4.2K_relative.svg
```
例如：
- `{cfg.OUTPUTS_FIGURES_DIR}/heatload/Top=4.2K_Rj=10.0nOhm/config=A1/A1_static_4.2K_relative.svg`
- `{cfg.OUTPUTS_FIGURES_DIR}/heatload/Top=4.2K_Rj=10.0nOhm/config=A1/A1_operation_4.2K_relative.svg`
- `{cfg.OUTPUTS_FIGURES_DIR}/heatload/Top=4.2K_Rj=10.0nOhm/config=C1/C1_charging_4.2K_relative.svg`

**图例来源：**
```
{cfg.OUTPUTS_FIGURES_DIR}/heatload/Top=4.2K_Rj=10.0nOhm/config=A1/{cfg.LEGEND_IMAGE_FILE}
```

---

### 2. `relative_heatload_grid_10.0K.svg`
**生成条件：**
- 运行温度：10.0 K
- 接头电阻：10 nΩ (R_p2p_joint = 10e-9 Ω)

**布局说明：**
- **第1行左边（子图 a）**：静态模式（static mode）
  - 配置：A1, B1, C1, D1（从左到右）
- **第1行右边（子图 b）**：运行模式（operation mode）
  - 配置：A1, B1, C1, D1（从左到右）
- **第2行左边（子图 c）**：充电峰值热负荷（charging peak heat load）
  - 控制变量：并绕根数（Npw）相同
  - 配置：C1, C2, C3, C4（从左到右）
- **第2行右边（子图 d）**：充电峰值热负荷（charging peak heat load）
  - 控制变量：匝间电阻率（rhot）相同
  - 配置：A1, B1, C1, D1（从左到右）

**子图文件来源：**
```
{cfg.OUTPUTS_FIGURES_DIR}/heatload/Top=10.0K_Rj=10.0nOhm/config={{配置名}}/{{配置名}}_{{模式}}_10.0K_relative.svg
```
例如：
- `{cfg.OUTPUTS_FIGURES_DIR}/heatload/Top=10.0K_Rj=10.0nOhm/config=A1/A1_static_10.0K_relative.svg`
- `{cfg.OUTPUTS_FIGURES_DIR}/heatload/Top=10.0K_Rj=10.0nOhm/config=A1/A1_operation_10.0K_relative.svg`
- `{cfg.OUTPUTS_FIGURES_DIR}/heatload/Top=10.0K_Rj=10.0nOhm/config=C1/C1_charging_10.0K_relative.svg`

**图例来源：**
```
{cfg.OUTPUTS_FIGURES_DIR}/heatload/Top=10.0K_Rj=10.0nOhm/config=A1/{cfg.LEGEND_IMAGE_FILE}
```

---

### 3. `relative_heatload_grid_20.0K.svg`
**生成条件：**
- 运行温度：20.0 K
- 接头电阻：10 nΩ (R_p2p_joint = 10e-9 Ω)

**布局说明：**
- **第1行左边（子图 a）**：静态模式（static mode）
  - 配置：A1, B1, C1, D1（从左到右）
- **第1行右边（子图 b）**：运行模式（operation mode）
  - 配置：A1, B1, C1, D1（从左到右）
- **第2行左边（子图 c）**：充电峰值热负荷（charging peak heat load）
  - 控制变量：并绕根数（Npw）相同
  - 配置：C1, C2, C3, C4（从左到右）
- **第2行右边（子图 d）**：充电峰值热负荷（charging peak heat load）
  - 控制变量：匝间电阻率（rhot）相同
  - 配置：A1, B1, C1, D1（从左到右）

**子图文件来源：**
```
{cfg.OUTPUTS_FIGURES_DIR}/heatload/Top=20.0K_Rj=10.0nOhm/config={{配置名}}/{{配置名}}_{{模式}}_20.0K_relative.svg
```
例如：
- `{cfg.OUTPUTS_FIGURES_DIR}/heatload/Top=20.0K_Rj=10.0nOhm/config=A1/A1_static_20.0K_relative.svg`
- `{cfg.OUTPUTS_FIGURES_DIR}/heatload/Top=20.0K_Rj=10.0nOhm/config=A1/A1_operation_20.0K_relative.svg`
- `{cfg.OUTPUTS_FIGURES_DIR}/heatload/Top=20.0K_Rj=10.0nOhm/config=C1/C1_charging_20.0K_relative.svg`

**图例来源：**
```
{cfg.OUTPUTS_FIGURES_DIR}/heatload/Top=20.0K_Rj=10.0nOhm/config=A1/{cfg.LEGEND_IMAGE_FILE}
```

---

### 4. `operation_comparison_20K_Rj_10vs100nOhm.svg`
**生成条件：**
- 运行温度：20.0 K
- 模式：运行模式（operation mode）
- 接头电阻对比：10 nΩ vs 100 nΩ

**布局说明：**
- **第1行左边（子图 a）**：接头电阻 10 nΩ
  - 配置：A1, B1, C1, D1（从左到右）
- **第1行右边（子图 b）**：接头电阻 100 nΩ
  - 配置：A1, B1, C1, D1（从左到右）

**子图文件来源：**

**10 nΩ 子图：**
```
{cfg.OUTPUTS_FIGURES_DIR}/heatload/Top=20.0K_Rj=10.0nOhm/config={{配置名}}/{{配置名}}_operation_20.0K_relative.svg
```
例如：
- `{cfg.OUTPUTS_FIGURES_DIR}/heatload/Top=20.0K_Rj=10.0nOhm/config=A1/A1_operation_20.0K_relative.svg`

**100 nΩ 子图：**
```
{cfg.OUTPUTS_FIGURES_DIR}/heatload/Top=20.0K_Rj=100.0nOhm/config={{配置名}}/{{配置名}}_operation_20.0K_relative.svg
```
注意：100 nΩ 的目录名格式为 `Top=20.0K_Rj=100.0nOhm`（与 4.10_heatload_2bars.py 中的路径格式一致：`Rj=R_p2p_joint*1e9 nOhm`，其中 R_p2p_joint=100e-9）

例如：
- `{cfg.OUTPUTS_FIGURES_DIR}/heatload/Top=20.0K_Rj=100.0nOhm/config=A1/A1_operation_20.0K_relative.svg`

**图例来源：**
```
{cfg.OUTPUTS_FIGURES_DIR}/heatload/Top=20.0K_Rj=10.0nOhm/config=A1/{cfg.LEGEND_IMAGE_FILE}
```

---

## 磁体配置参数说明

各配置的参数定义（来自 `4.10_heatload_2bars.py`）：

| 配置名 | 并绕根数 (Npw) | 匝间电阻率 (rhot) | 说明 |
|--------|----------------|-------------------|------|
| A1     | 5              | 5000×10⁻¹⁰ Ω·cm² | 基准配置 |
| B1     | 10             | 5000×10⁻¹⁰ Ω·cm² | 增加并绕根数 |
| B2     | 10             | 1500×10⁻¹⁰ Ω·cm² | 降低匝间电阻率 |
| C1     | 20             | 5000×10⁻¹⁰ Ω·cm² | 进一步增加并绕根数 |
| C2     | 20             | 1500×10⁻¹⁰ Ω·cm² | 降低匝间电阻率 |
| C3     | 20             | 100×10⁻¹⁰ Ω·cm²  | 显著降低匝间电阻率 |
| C4     | 20             | 50×10⁻¹⁰ Ω·cm²   | 极低匝间电阻率 |
| D1     | 200            | 5000×10⁻¹⁰ Ω·cm² | 大量并绕根数 |
| D2     | 200            | 1500×10⁻¹⁰ Ω·cm² | 降低匝间电阻率 |
| D3     | 200            | 100×10⁻¹⁰ Ω·cm²  | 显著降低匝间电阻率 |
| D4     | 200            | 50×10⁻¹⁰ Ω·cm²   | 极低匝间电阻率 |

### 控制变量说明

**子图 c（控制并绕根数相同）：**
- C1, C2, C3, C4 的 Npw = 20（相同）
- 变化的是 rhot：5000 → 1500 → 100 → 50 (×10⁻¹⁰ Ω·cm²)

**子图 d（控制匝间电阻率相同）：**
- A1, B1, C1, D1 的 rhot 都接近或等于 5000×10⁻¹⁰ Ω·cm²（相同）
- 变化的是 Npw：5 → 10 → 20 → 200

---

## 子图生成说明

所有子图（`*_relative.svg`）由 `4.10_heatload_2bars.py` 脚本生成。

**子图生成参数：**
- 运行温度（T_op）：4.2 K, 10.0 K, 20.0 K
- 接头电阻（R_p2p_joint）：10 nΩ（默认），100 nΩ（对比图）
- 模式（mode）：
  - `static`：静态模式（无电流）
  - `operation`：运行模式（有电流，有核热）
  - `charging`：充电模式（有电流，无核热，有磁化损耗和径向损耗）

**子图文件命名规则：**
```
{{配置名}}_{{模式}}_{{温度}}K_relative.svg
```

**图例文件：**
- 文件名：`{cfg.LEGEND_IMAGE_FILE}`
- 位置：每个配置目录下
- 内容：所有热源类型的图例（12种热源）

---

## 文件路径结构

```
{cfg.OUTPUTS_FIGURES_DIR}/heatload/
├── relative_heatload_grids/          # 组合图输出目录（本README所在目录）
│   ├── README.md                      # 本文档
│   ├── relative_heatload_grid_4.2K.svg
│   ├── relative_heatload_grid_10.0K.svg
│   ├── relative_heatload_grid_20.0K.svg
│   └── operation_comparison_20K_Rj_10vs100nOhm.svg
│
└── Top={{温度}}K_Rj={{电阻}}nOhm/        # 子图源文件目录（格式：Top=T_op_K_Rj=R_p2p_joint*1e9_nOhm）
    └── config={{配置名}}/
        ├── {{配置名}}_static_{{温度}}K_relative.svg
        ├── {{配置名}}_operation_{{温度}}K_relative.svg
        ├── {{配置名}}_charging_{{温度}}K_relative.svg
        └── {cfg.LEGEND_IMAGE_FILE}
```

---

## 备注

1. 所有路径相对于项目根目录。
2. 子图文件必须存在才能生成组合图，缺失的子图会在控制台输出警告。
3. 图例文件会自动缩放以适应画布宽度（如果超出）。
4. 组合图会自动裁剪空白边缘，保留40pt的边距。
5. 本文档由 `4.11_stitch_relative_heatload.py` 脚本自动生成，每次运行脚本时会自动更新。
"""
    
    # 写入README文件
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"\n[OK] README文档已生成: {readme_path}")


def main():
    """主函数：为所有温度生成组合图"""
    # 输出目录
    output_dir = Path(cfg.OUTPUTS_FIGURES_DIR) / "heatload" / "relative_heatload_grids"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 为每个温度生成组合图
    temperatures = [4.2, 10.0, 20.0]
    
    print("=" * 60)
    print("开始组合相对热负荷图...")
    print("=" * 60)
    
    for T_op in temperatures:
        print(f"\n处理温度: {T_op}K")
        stitch_relative_heatload_grid(T_op, output_dir)
    
    # 生成20K operation模式对比图
    print(f"\n处理20K Operation对比图（10nOhm vs 100nOhm）")
    stitch_operation_comparison_20K(output_dir)
    
    # 生成README文档
    print(f"\n生成README文档...")
    generate_readme(output_dir)
    
    print("\n" + "=" * 60)
    print("所有组合图生成完成！")
    print(f"输出目录: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
