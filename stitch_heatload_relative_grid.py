"""
4.11_stitch_relative_heatload.py

Stitch relative heat-load figures into composite figures.
Produces 3 composites for 4.2K, 10K, 20K. Each has 2 rows x 2 panel regions:
- Row 1 left (a): static, left-to-right A1, B1, C1, D1
- Row 1 right (b): operation, A1, B1, C1, D1
- Row 2 left (c): charging peak heat load, same Npw, C1, C2, C3, C4
- Row 2 right (d): charging peak heat load, same turn resistivity, A1, B1, C1, D1

Panel labels (a, b, c, d) at top-left of each region.

Requires: svgutils (pip install svgutils)
"""

from pathlib import Path
import re

import config as cfg

try:
    import svgutils.transform as sg
except ImportError as e:
    raise ImportError(
        "svgutils is required to stitch SVG figures. Install with:\n\n"
        "    pip install svgutils\n"
    ) from e


def _parse_size(size_str: str) -> float:
    """
    Parse SVG width/height string (e.g. '432pt', '800px') to float.
    Keeps only digits and decimal point.
    """
    if size_str is None:
        return 0.0
    m = re.findall(r"[0-9.]+", str(size_str))
    return float(m[0]) if m else 0.0


def stitch_relative_heatload_grid(T_op: float, output_dir: Path):
    """
    Stitch relative heat-load figures for the given temperature.
    Layout: (a) row1 left static A1-D1, (b) row1 right operation A1-D1,
    (c) row2 left charging peak same Npw C1-C4, (d) row2 right charging peak same rho_turn A1-D1.
    Args:
        T_op: Operating temperature (4.2, 10.0, 20.0)
        output_dir: Output directory
    """
    # Panel (a) row1 left: static A1, B1, C1, D1
    panel_a_configs = ['A1', 'B1', 'C1', 'D1']
    panel_a_mode = 'static'
    
    # (b) row1 right: operation A1, B1, C1, D1
    panel_b_configs = ['A1', 'B1', 'C1', 'D1']
    panel_b_mode = 'operation'
    
    # (c) row2 left: charging peak same Npw C1-C4
    panel_c_configs = ['C1', 'C2', 'C3', 'C4']
    panel_c_mode = 'charging'
    
    # (d) row2 right: charging peak same rho_turn A1, B1, C1, D1
    panel_d_configs = ['A1', 'B1', 'C1', 'D1']
    panel_d_mode = 'charging'
    
    # Base path (same format as 4.10_heatload_2bars.py); R_joint = 10e-9 (10 nOhm)
    R_p2p_joint = 10e-9
    base_dir = Path(cfg.OUTPUTS_FIGURES_DIR) / "heatload" / f"Top={T_op}K_Rj={R_p2p_joint*1e9}nOhm"
    
    # Store (panel_label, config_name, root, panel_row, panel_col, subplot_col)
    # panel_row: 0=row1, 1=row2; panel_col: 0=left, 1=right; subplot_col: 0-3
    all_panels = [
        ('a', panel_a_configs, panel_a_mode, 0, 0),  # row1 left
        ('b', panel_b_configs, panel_b_mode, 0, 1),  # row1 right
        ('c', panel_c_configs, panel_c_mode, 1, 0),  # row2 left
        ('d', panel_d_configs, panel_d_mode, 1, 1),  # row2 right
    ]
    
    panel_entries = []
    panel_w = None   # single subplot width
    panel_h = None   # single subplot height

    # Load all subplots
    for panel_label, configs_list, mode, panel_row, panel_col in all_panels:
        for subplot_col, config_name in enumerate(configs_list):
            file_path = base_dir / f"config={config_name}" / f"{config_name}_{mode}_{T_op}K_relative.svg"
            
            if not file_path.exists():
                print(f"  [Warning] Subplot not found: {file_path}")
                continue
            
            try:
                fig = sg.fromfile(str(file_path))
                root = fig.getroot()
                
                # Track max width/height across subplots
                size_w, size_h = fig.get_size()
                w = _parse_size(size_w)
                h = _parse_size(size_h)
                panel_w = w if panel_w is None else max(panel_w, w)
                panel_h = h if panel_h is None else max(panel_h, h)
                
                panel_entries.append((panel_label, config_name, root, panel_row, panel_col, subplot_col))
            except Exception as e:
                print(f"  [Error] Failed to read {file_path}: {e}")
                continue
    
    if not panel_entries or panel_w is None or panel_h is None:
        print(f"  No relative heat-load SVG subplots found (T_op={T_op}K); exiting.")
        return
    
    # Layout
    n_panel_rows = 2
    n_panel_cols = 2
    n_subplots_per_panel = 4

    subplot_margin_x = panel_w * 0.08   # spacing within panel
    panel_margin_x = panel_w * 0.3      # between left/right panels
    panel_margin_y = panel_h * 0.1      # between rows

    # Panel width (4 subplots)
    panel_area_w = subplot_margin_x * (n_subplots_per_panel - 1) + panel_w * n_subplots_per_panel
    
    # Margins
    label_font_size = 24
    panel_label_font_size = 32  # panel labels (a, b, c, d)
    left_margin = 40
    top_margin = 50
    bottom_margin = 30
    right_margin = 30
    
    # Find legend file (any config's legend; they should be identical)
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
            print(f"  Legend file: {legend_file}, size: {legend_w}pt x {legend_h}pt")
        except Exception as e:
            print(f"  [Warning] Failed to read legend: {e}")
            legend_h = 0
            legend_w = 0
            legend_fig = None
    else:
        print(f"  [Warning] Legend not found: {legend_file}")
    
    # Spacing between legend and content
    legend_caption_spacing = 20
    
    # Total size (legend + content)
    total_w = left_margin + panel_area_w + panel_margin_x + panel_area_w + right_margin
    # If legend wider than content, use scaled height
    available_width = total_w - left_margin - right_margin
    if legend_w > available_width and legend_h > 0:
        legend_scale = available_width / legend_w
        legend_h_effective = legend_h * legend_scale
    else:
        legend_h_effective = legend_h
    
    total_h = (top_margin + panel_h + panel_margin_y + panel_h + 
               legend_caption_spacing + legend_h_effective + 
               bottom_margin)
    
    # Build canvas
    fig_out = sg.SVGFigure(f"{total_w}pt", f"{total_h}pt")
    print(f"  T_op={T_op}K: total_w={total_w}pt, total_h={total_h}pt")
    
    # Place all subplots
    placed_roots = []
    for panel_label, config_name, root, panel_row, panel_col, subplot_col in panel_entries:
        # Subplot position on canvas
        # Panel start x
        if panel_col == 0:  # left panel
            panel_start_x = left_margin
        else:  # right panel
            panel_start_x = left_margin + panel_area_w + panel_margin_x
        
        # Panel start y
        panel_start_y = top_margin + panel_row * (panel_h + panel_margin_y)
        
        # Subplot x within panel
        subplot_x_in_panel = subplot_col * (panel_w + subplot_margin_x)
        
        # Final position
        x = panel_start_x + subplot_x_in_panel
        y = panel_start_y
        
        root.moveto(x, y)
        placed_roots.append(root)
    
    # Add legend if present
    if legend_fig is not None and legend_h > 0:
        try:
            legend_root = legend_fig.getroot()
            # Scale legend if it exceeds canvas width
            available_width = total_w - left_margin - right_margin
            if legend_w > available_width:
                scale_factor = available_width / legend_w
                legend_w_scaled = legend_w * scale_factor
                legend_h_scaled = legend_h * scale_factor
                # Use svgutils scale
                legend_root.scale(scale_factor, scale_factor)
                print(f"  Legend scaled (factor: {scale_factor:.3f})")
            else:
                legend_w_scaled = legend_w
                legend_h_scaled = legend_h
            
            # Legend: bottom, centered
            legend_x = (total_w - legend_w_scaled) / 2
            legend_y = top_margin + panel_h + panel_margin_y + panel_h + legend_caption_spacing
            legend_root.moveto(legend_x, legend_y)
            placed_roots.append(legend_root)
        except Exception as e:
            print(f"  [Warning] Failed to add legend: {e}")
    
    fig_out.append(placed_roots)
    
    # Save output
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"relative_heatload_grid_{T_op}K.svg"
    fig_out.save(str(out_path))
    
    # Post-process SVG: add panel labels (a, b, c, d)
    import xml.etree.ElementTree as ET
    tree = ET.parse(str(out_path))
    root_svg = tree.getroot()
    
    # Get SVG namespace
    svg_ns = None
    for prefix, uri in root_svg.attrib.items():
        if prefix.startswith('xmlns') and 'svg' in uri.lower():
            svg_ns = uri
            break
    if svg_ns is None:
        svg_ns = 'http://www.w3.org/2000/svg'
    
    # Register namespace
    ET.register_namespace('', svg_ns)
    
    # Set root SVG size
    root_svg.set('width', f"{total_w}pt")
    root_svg.set('height', f"{total_h}pt")
    root_svg.set('viewBox', f"0 0 {total_w} {total_h}")
    
    # Create labels group
    labels_group = ET.SubElement(root_svg, f'{{{svg_ns}}}g', {'id': 'labels'})
    
    # Text style
    def create_text(x, y, text, font_size=label_font_size, anchor='start', baseline='top', bold=True):
        """Create text element."""
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
    
    # Panel labels (a, b, c, d) at top-left
    panel_positions = {
        'a': (0, 0),  # row1 left
        'b': (0, 1),  # row1 right
        'c': (1, 0),  # row2 left
        'd': (1, 1),  # row2 right
    }
    
    for panel_label, (panel_row, panel_col) in panel_positions.items():
        # Panel start position
        if panel_col == 0:  # left
            panel_start_x = left_margin
        else:  # right panel
            panel_start_x = left_margin + panel_area_w + panel_margin_x
        
        panel_start_y = top_margin + panel_row * (panel_h + panel_margin_y)
        
        # Label at panel top-left with small offset
        label_x = panel_start_x + 5
        label_y = panel_start_y + 5
        
        create_text(label_x, label_y, panel_label, font_size=panel_label_font_size, 
                   anchor='start', baseline='top', bold=True)
    
    # Keep labels group last so it draws on top
    root_svg.remove(labels_group)
    root_svg.append(labels_group)
    
    # Compute content bounds and trim white space
    min_x = float('inf')
    min_y = float('inf')
    max_x = float('-inf')
    max_y = float('-inf')
    
    # 1. Subplot bounds (leftmost subplot)
    min_x = min(min_x, left_margin)
    # Rightmost subplot
    max_x = max(max_x, left_margin + panel_area_w + panel_margin_x + panel_area_w)
    # Topmost subplot
    min_y = min(min_y, top_margin)
    # Bottommost subplot
    max_y = max(max_y, top_margin + panel_h + panel_margin_y + panel_h)
    
    # 2. Legend bounds (if present)
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
    
    # 3. Panel label (a, b, c, d) positions
    for panel_label, (panel_row, panel_col) in panel_positions.items():
        if panel_col == 0:
            panel_start_x = left_margin
        else:
            panel_start_x = left_margin + panel_area_w + panel_margin_x
        panel_start_y = top_margin + panel_row * (panel_h + panel_margin_y)
        label_x = panel_start_x + 5
        label_y = panel_start_y + 5
        # Estimate label text size
        label_width = panel_label_font_size * 0.6
        label_height = panel_label_font_size * 1.2
        min_x = min(min_x, label_x)
        max_x = max(max_x, label_x + label_width)
        min_y = min(min_y, label_y)
        max_y = max(max_y, label_y + label_height)
    
    # Add margin
    padding = 40
    min_x = max(0, min_x - padding)
    min_y = max(0, min_y - padding)
    max_x = min(total_w, max_x + padding)
    max_y = min(total_h, max_y + padding)
    
    # Cropped size
    cropped_w = max_x - min_x
    cropped_h = max_y - min_y
    
    # Update viewBox to trim white space
    root_svg.set('viewBox', f"{min_x} {min_y} {cropped_w} {cropped_h}")
    root_svg.set('width', f"{cropped_w}pt")
    root_svg.set('height', f"{cropped_h}pt")
    
    print(f"  Cropped: {total_w:.1f}pt x {total_h:.1f}pt -> {cropped_w:.1f}pt x {cropped_h:.1f}pt")
    
    # Save file
    tree.write(str(out_path), encoding='utf-8', xml_declaration=True)
    print(f"  [OK] Composite saved: {out_path}")


def stitch_operation_comparison_20K(output_dir: Path):
    """
    Stitch operation-mode comparison at 20 K for two joint resistances (10 vs 100 nOhm).

    Layout:
    - Row 1 left (a): 10 nOhm, A1, B1, C1, D1
    - Row 1 right (b): 100 nOhm, A1, B1, C1, D1
    - Bottom: legend

    Args:
        output_dir: Output directory.
    """
    T_op = 20.0
    mode = 'operation'
    configs = ['A1', 'B1', 'C1', 'D1']
    
    # Two panels: 10 nOhm and 100 nOhm
    R_joint_10nOhm = 10e-9
    R_joint_100nOhm = 100e-9
    
    # Base path (same format as 4.10_heatload_2bars.py): Top={T_op}K_Rj={R_p2p_joint*1e9}nOhm
    base_dir_10nOhm = Path(cfg.OUTPUTS_FIGURES_DIR) / "heatload" / f"Top={T_op}K_Rj={R_joint_10nOhm*1e9}nOhm"
    base_dir_100nOhm = Path(cfg.OUTPUTS_FIGURES_DIR) / "heatload" / f"Top={T_op}K_Rj={R_joint_100nOhm*1e9}nOhm"
    
    panel_entries = []
    panel_w = None
    panel_h = None
    
    # Load 10 nOhm subplots (left, a)
    for subplot_col, config_name in enumerate(configs):
        file_path = base_dir_10nOhm / f"config={config_name}" / f"{config_name}_{mode}_{T_op}K_relative.svg"
        if not file_path.exists():
            print(f"  [Warning] Subplot not found: {file_path}")
            continue
        try:
            fig = sg.fromfile(str(file_path))
            root = fig.getroot()
            size_w, size_h = fig.get_size()
            w = _parse_size(size_w)
            h = _parse_size(size_h)
            panel_w = w if panel_w is None else max(panel_w, w)
            panel_h = h if panel_h is None else max(panel_h, h)
            panel_entries.append(('a', config_name, root, 0, 0, subplot_col))
        except Exception as e:
            print(f"  [Error] Failed to read {file_path}: {e}")
    
    # Load 100 nOhm subplots (right, b)
    for subplot_col, config_name in enumerate(configs):
        file_path = base_dir_100nOhm / f"config={config_name}" / f"{config_name}_{mode}_{T_op}K_relative.svg"
        if not file_path.exists():
            print(f"  [Warning] Subplot not found: {file_path}")
            continue
        try:
            fig = sg.fromfile(str(file_path))
            root = fig.getroot()
            size_w, size_h = fig.get_size()
            w = _parse_size(size_w)
            h = _parse_size(size_h)
            panel_w = w if panel_w is None else max(panel_w, w)
            panel_h = h if panel_h is None else max(panel_h, h)
            panel_entries.append(('b', config_name, root, 0, 1, subplot_col))
        except Exception as e:
            print(f"  [Error] Failed to read {file_path}: {e}")
    
    if not panel_entries or panel_w is None or panel_h is None:
        print(f"  No relative heatload SVG subplots found (20K operation), exiting.")
        return
    
    # Layout
    n_panel_rows = 1
    n_panel_cols = 2
    n_subplots_per_panel = 4
    
    subplot_margin_x = panel_w * 0.08
    panel_margin_x = panel_w * 0.3
    panel_margin_y = panel_h * 0.1

    # Panel width
    panel_area_w = subplot_margin_x * (n_subplots_per_panel - 1) + panel_w * n_subplots_per_panel
    
    # Margins
    panel_label_font_size = 32
    left_margin = 40
    top_margin = 50
    bottom_margin = 30
    right_margin = 30
    
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
            print(f"  Legend file: {legend_file}, size: {legend_w}pt x {legend_h}pt")
        except Exception as e:
            print(f"  [Warning] Failed to read legend: {e}")
            legend_fig = None
    
    # Spacing between legend and content
    legend_spacing = 50
    
    total_w = left_margin + panel_area_w + panel_margin_x + panel_area_w + right_margin
    total_h = (top_margin + panel_h + 
               legend_spacing + legend_h + 
               bottom_margin)
    
    # Build canvas
    fig_out = sg.SVGFigure(f"{total_w}pt", f"{total_h}pt")
    print(f"  Operation comparison: total_w={total_w}pt, total_h={total_h}pt")
    
    # Place all subplots
    placed_roots = []
    for panel_label, config_name, root, panel_row, panel_col, subplot_col in panel_entries:
        if panel_col == 0:  # left panel
            panel_start_x = left_margin
        else:  # right panel
            panel_start_x = left_margin + panel_area_w + panel_margin_x
        
        panel_start_y = top_margin
        subplot_x_in_panel = subplot_col * (panel_w + subplot_margin_x)
        
        x = panel_start_x + subplot_x_in_panel
        y = panel_start_y
        
        root.moveto(x, y)
        placed_roots.append(root)
    
    if legend_fig is not None and legend_h > 0:
        try:
            legend_root = legend_fig.getroot()
            available_width = total_w - left_margin - right_margin
            if legend_w > available_width:
                scale_factor = available_width / legend_w
                legend_w_scaled = legend_w * scale_factor
                legend_root.scale(scale_factor, scale_factor)
                print(f"  Legend scaled (factor: {scale_factor:.3f})")
            else:
                legend_w_scaled = legend_w
            
            legend_x = (total_w - legend_w_scaled) / 2
            legend_y = top_margin + panel_h + legend_spacing
            legend_root.moveto(legend_x, legend_y)
            placed_roots.append(legend_root)
        except Exception as e:
            print(f"  [Warning] Failed to add legend: {e}")
    
    fig_out.append(placed_roots)
    
    # Save output
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "operation_comparison_20K_Rj_10vs100nOhm.svg"
    fig_out.save(str(out_path))
    
    import xml.etree.ElementTree as ET
    tree = ET.parse(str(out_path))
    root_svg = tree.getroot()
    
    # Get SVG namespace
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
    
    # Create labels group
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
    
    panel_positions = {
        'a': (0, 0),  # left
        'b': (0, 1),  # right
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
    
    # Bounds and crop
    min_x = left_margin
    max_x = left_margin + panel_area_w + panel_margin_x + panel_area_w
    min_y = top_margin
    max_y = top_margin + panel_h + legend_spacing + legend_h
    
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
    
    print(f"  Cropped: {total_w:.1f}pt x {total_h:.1f}pt -> {cropped_w:.1f}pt x {cropped_h:.1f}pt")
    
    root_svg.remove(labels_group)
    root_svg.append(labels_group)
    
    tree.write(str(out_path), encoding='utf-8', xml_declaration=True)
    print(f"  [OK] Operation comparison saved: {out_path}")


def generate_readme(output_dir: Path):
    """
    Generate README documenting how each composite figure was built and where subplots come from.

    Args:
        output_dir: Output directory.
    """
    readme_path = output_dir / "README.md"

    from datetime import datetime
    current_date = datetime.now().strftime("%Y-%m-%d")

    readme_content = f"""# Relative heat load composite figures

This document describes how the composites in `relative_heatload_grids` are built and where subplots come from.

**Generated:** {current_date}

## Script

All composites are produced by `4.11_stitch_relative_heatload.py`.

---

## Output files

### 1. `relative_heatload_grid_4.2K.svg`
**Conditions:** T_op = 4.2 K, R_p2p_joint = 10e-9 Ω.

**Layout:** Row 1 left (a) static, right (b) operation; row 2 left (c) charging (Npw fixed), right (d) charging (rhot fixed). Configs left-to-right: a,b A1,B1,C1,D1; c C1–C4; d A1,B1,C1,D1.

**Subplot path:** `{cfg.OUTPUTS_FIGURES_DIR}/heatload/Top=4.2K_Rj=10.0nOhm/config={{config}}/{{config}}_{{mode}}_4.2K_relative.svg`  
**Legend:** `{cfg.OUTPUTS_FIGURES_DIR}/heatload/Top=4.2K_Rj=10.0nOhm/config=A1/{cfg.LEGEND_IMAGE_FILE}`

---

### 2. `relative_heatload_grid_10.0K.svg`
**Conditions:** T_op = 10.0 K, R_p2p_joint = 10e-9 Ω. Layout as above.

**Subplot path:** `{cfg.OUTPUTS_FIGURES_DIR}/heatload/Top=10.0K_Rj=10.0nOhm/config={{config}}/{{config}}_{{mode}}_10.0K_relative.svg`  
**Legend:** `{cfg.OUTPUTS_FIGURES_DIR}/heatload/Top=10.0K_Rj=10.0nOhm/config=A1/{cfg.LEGEND_IMAGE_FILE}`

---

### 3. `relative_heatload_grid_20.0K.svg`
**Conditions:** T_op = 20.0 K, R_p2p_joint = 10e-9 Ω. Layout as above.

**Subplot path:** `{cfg.OUTPUTS_FIGURES_DIR}/heatload/Top=20.0K_Rj=10.0nOhm/config={{config}}/{{config}}_{{mode}}_20.0K_relative.svg`  
**Legend:** `{cfg.OUTPUTS_FIGURES_DIR}/heatload/Top=20.0K_Rj=10.0nOhm/config=A1/{cfg.LEGEND_IMAGE_FILE}`

---

### 4. `operation_comparison_20K_Rj_10vs100nOhm.svg`
**Conditions:** T_op = 20.0 K, operation mode; 10 nΩ vs 100 nΩ.

**Layout:** Row 1 left (a) 10 nΩ, right (b) 100 nΩ; configs A1, B1, C1, D1.

**10 nΩ:** `{cfg.OUTPUTS_FIGURES_DIR}/heatload/Top=20.0K_Rj=10.0nOhm/config={{config}}/{{config}}_operation_20.0K_relative.svg`  
**100 nΩ:** `{cfg.OUTPUTS_FIGURES_DIR}/heatload/Top=20.0K_Rj=100.0nOhm/config={{config}}/{{config}}_operation_20.0K_relative.svg` (directory format: Top=20.0K_Rj=100.0nOhm)  
**Legend:** `{cfg.OUTPUTS_FIGURES_DIR}/heatload/Top=20.0K_Rj=10.0nOhm/config=A1/{cfg.LEGEND_IMAGE_FILE}`

---

## Magnet config parameters (from 4.10_heatload_2bars.py)

| Config | Npw | rhot (×10⁻¹⁰ Ω·cm²) | Note |
|--------|-----|---------------------|------|
| A1     | 5   | 5000 | baseline |
| B1     | 10  | 5000 | more Npw |
| B2     | 10  | 1500 | lower rhot |
| C1–C4  | 20  | 5000, 1500, 100, 50 | Npw fixed, rhot varies |
| D1–D4  | 200 | 5000, 1500, 100, 50 | same |

Panel c: Npw=20 (C1–C4); panel d: rhot ~5000 (A1, B1, C1, D1), Npw varies.

---

## Subplot generation

Subplots `*_relative.svg` are from `4.10_heatload_2bars.py`. Modes: `static`, `operation`, `charging`. Naming: `{{config}}_{{mode}}_{{T}}K_relative.svg`. Legend: `{cfg.LEGEND_IMAGE_FILE}` in each config dir (12 heat sources).

---

## Path structure

`{cfg.OUTPUTS_FIGURES_DIR}/heatload/` → `relative_heatload_grids/` (this README + composite SVGs) and `Top={{T}}K_Rj={{R}}nOhm/config={{config}}/` (source subplots + legend).

---

## Notes

1. Paths are relative to project root. 2. Missing subplots produce console warnings. 3. Legend is scaled if wider than canvas. 4. Composites are cropped with 40pt margin. 5. This README is auto-generated by `4.11_stitch_relative_heatload.py`.
"""
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print(f"\n[OK] README written: {readme_path}")


def main():
    """Generate composite figures for all temperatures."""
    output_dir = Path(cfg.OUTPUTS_FIGURES_DIR) / "heatload" / "relative_heatload_grids"
    output_dir.mkdir(parents=True, exist_ok=True)
    temperatures = [4.2, 10.0, 20.0]
    print("=" * 60)
    print("Stitching relative heat load figures...")
    print("=" * 60)
    for T_op in temperatures:
        print(f"\nTemperature: {T_op}K")
        stitch_relative_heatload_grid(T_op, output_dir)
    print(f"\n20K operation comparison (10 nOhm vs 100 nOhm)")
    stitch_operation_comparison_20K(output_dir)
    print(f"\nGenerating README...")
    generate_readme(output_dir)
    print("\n" + "=" * 60)
    print("All composites done.")
    print(f"Output: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
