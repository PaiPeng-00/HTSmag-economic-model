"""
9.0_stitch_delta_lcoe_min_svgs.py

Stitch economic heatmap SVG subfigures (e.g. delta_lcoe_min, parasitic ratio,
AF, AF_ref, cryo power) into composite layouts for papers/plots.

- Rows (top to bottom): typically S1, S2, S3 (technology scenarios)
- Columns (left to right): 4.2K He, 10K He, 20K He, 20K H2 (or as configured)

Dependency:
- `svgutils` (install via `pip install svgutils`)
"""

from pathlib import Path
import re
import numpy as np
import config as cfg
import subprocess
INKSCAPE_EXE = r"D:\software\work\inkspace\bin\inkscape.exe"
# PNG export resolution (DPI); default 96, here increased to 300
PNG_EXPORT_DPI = 300

# Axis‑tick configuration imported from config
# Horizontal axis (R_joint): log scale, range 1–100 nOhm (from config.py / plot_library.py)
# Recommended ticks (nΩ): 1, 10, 100 (log10 = 0, 1, 2)
X_AXIS_TICKS_NOHM = [1, 10, 100]  # nOhm, corresponding to log10(1)=0, log10(10)=1, log10(100)=2
X_AXIS_MIN_LOG = 0  # log10(1)
X_AXIS_MAX_LOG = 2  # log10(100)
X_AXIS_MIN_NOHM = 1  # nOhm
X_AXIS_MAX_NOHM = 100  # nOhm

# Vertical axis (Npw): linear scale, range inferred from config.py / plot_library.py
# NPW_SCAN_VALUES = np.concatenate([np.arange(1, 21, 1), np.arange(10, 201, 10)]) → data range 1–200
NPW_VALUES = np.array(cfg.NPW_SCAN_VALUES)
Y_AXIS_MIN = 1    # minimum displayed value (from NPW_SCAN_VALUES)
Y_AXIS_MAX = 200  # maximum displayed value (from NPW_SCAN_VALUES)
# Key ticks chosen from NPW_SCAN_VALUES; should match those recommended in plot_library.py
Y_AXIS_TICKS = [10, 50, 100, 200]

# AF‑heatmap configuration (from 2.9_af_time999_3&3.py).
# These values must remain consistent with FIXED_NPW and FIXED_RHOT used there.
AF_Y_AXIS_MIN = 1
AF_Y_AXIS_MAX = 20
AF_Y_AXIS_TICKS = [1, 5, 10, 20]  # from DESIRED_NPW_TICKS in 2.9_af_time999_3&3.py

# FIXED_RHOT range: 10–1000 μΩ·cm², DESIRED_RHO_TICKS = [10, 100, 1000]
AF_X_AXIS_TICKS_UOHM_CM2 = [10, 100, 1000]  # μΩ·cm², log10(10)=1, log10(100)=2, log10(1000)=3
AF_X_AXIS_MIN_LOG = 1  # log10(10)
AF_X_AXIS_MAX_LOG = 3  # log10(1000)
AF_X_AXIS_MIN_UOHM_CM2 = 10  # μΩ·cm²
AF_X_AXIS_MAX_UOHM_CM2 = 1000  # μΩ·cm²

# Minimum width (pt) reserved for the right‑hand label area
RIGHT_LABEL_WIDTH_MIN = 120

try:
    import svgutils.transform as sg
except ImportError as e:
    raise ImportError(
        "The package `svgutils` is required to stitch SVG figures. "
        "Please install it first with:\n\n"
        "    pip install svgutils\n"
    ) from e


def export_svg_to_pdf(svg_path: Path) -> None:
    pdf_path = svg_path.with_suffix(".pdf")
    try:
        subprocess.run(
            [
                INKSCAPE_EXE,
                str(svg_path),
                "--export-type=pdf",
                f"--export-filename={pdf_path}",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        print(f"[OK] Also exported PDF: {pdf_path}")
    except FileNotFoundError:
        print(f"[info] inkscape.exe not found (incorrect path), skipping PDF export: {svg_path}")
    except subprocess.CalledProcessError as exc:
        print(f"[warning] Inkscape PDF export failed: {exc.stderr.strip()}")


def export_svg_to_png(svg_path: Path) -> None:
    """Export an SVG file to PNG format using Inkscape."""
    png_path = svg_path.with_suffix(".png")
    try:
        subprocess.run(
            [
                INKSCAPE_EXE,
                str(svg_path),
                "--export-type=png",
                f"--export-dpi={PNG_EXPORT_DPI}",
                f"--export-filename={png_path}",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        print(f"[OK] Additional PNG saved: {png_path}")
    except FileNotFoundError:
        print(f"[info] inkscape.exe not found (path error); skipping PNG export for: {svg_path}")
    except subprocess.CalledProcessError as exc:
        print(f"[warning] Inkscape PNG export failed: {exc.stderr.strip()}")

def export_svg_to_pdf_svglib(svg_path: Path) -> None:
    pdf_path = svg_path.with_suffix(".pdf")
    try:
        from svglib.svglib import svg2rlg
        from reportlab.graphics import renderPDF
    except Exception:
        print("[info] svglib/reportlab not installed; skipping SVG→PDF conversion")
        return

    try:
        drawing = svg2rlg(str(svg_path))
        renderPDF.drawToFile(drawing, str(pdf_path))
        print(f"[OK] Additional PDF saved: {pdf_path}")
    except Exception as exc:
        print(f"[warning] svglib export failed: {svg_path} -> {pdf_path}, reason: {exc}")

def _parse_size(size_str: str) -> float:
    """
    Parse an SVG width/height string (e.g. '432pt', '800px') into a float,
    keeping only digits and decimal point.
    """
    if size_str is None:
        return 0.0
    m = re.findall(r"[0-9.]+", str(size_str))
    return float(m[0]) if m else 0.0


def to_title_case(text: str) -> str:
    """
    Convert a label string to sentence case (first letter capitalised,
    remainder lower‑case), preserving units (Ω, μ, cm², etc.) and any
    parenthesised content.
    
    Example:
        "turn-to-turn resistivity" -> "Turn-to-turn resistivity"
        "Coil-to-coil Joint Resistance (nΩ)" -> "Coil-to-coil joint resistance (nΩ)"
        "Turn-to-turn Resistivity (μΩ·cm²)" -> "Turn-to-turn resistivity (μΩ·cm²)"
        "Number of Parallel Tapes" -> "Number of parallel tapes"
    """
    # If the text already contains a unit symbol (in parentheses), handle it specially
    # Pattern: main text + optional parenthesis content (with unit)
    pattern = r'^(.+?)(\s*\([^)]+\))?$'
    match = re.match(pattern, text)
    
    if match:
        main_text = match.group(1).strip()
        unit_part = match.group(2) if match.group(2) else ''
    else:
        main_text = text
        unit_part = ''
    
    # Convert the entire main text to lowercase, then capitalize only the first letter
    if main_text:
        # First convert to all lowercase
        main_text_lower = main_text.lower()
        # Then capitalize only the first letter (if alphabetic)
        if main_text_lower and main_text_lower[0].isalpha():
            main_text_result = main_text_lower[0].upper() + main_text_lower[1:]
        else:
            # If the first character is not a letter, find the first letter and capitalize it
            for i, char in enumerate(main_text_lower):
                if char.isalpha():
                    main_text_result = main_text_lower[:i] + char.upper() + main_text_lower[i+1:]
                    break
            else:
                main_text_result = main_text_lower
    else:
        main_text_result = main_text
    
    # Combine result: main text + unit part (unchanged)
    return main_text_result + unit_part


def stitch_delta_lcoe_min_svgs_single_cmap(cmap_suffix: str = ""):
    """
    Stitch `delta_lcoe_min_heatmap` subplots into a single large SVG (single colormap).
    
    Parameters
    ----------
    cmap_suffix:
        Name of the colormap (e.g. "cividis" or "YlGnBu"). If empty, use the default file names.
    """
    # New path layout: outputs/figures/economic/{cmap_name}/parasitic_ratio/USD
    if cmap_suffix:
        scan_base_dir = Path(cfg.ECONOMIC_FIGURES_DIR) / cmap_suffix / cfg.PARASITIC_RATIO_OUTPUT_DIR
    else:
        # Backward‑compatible fallback: use legacy path when no colormap suffix is given
        scan_base_dir = Path(cfg.ECONOMIC_FIGURES_DIR) / cfg.PARASITIC_RATIO_OUTPUT_DIR
    scan_usd_dir = scan_base_dir / cfg.PARASITIC_RATIO_USD_DIR

    # Rows: technology scenarios
    scenarios = ["S1", "S2", "S3"]
    # Columns: temperature–coolant pairs (consistent with parameter scan)
    temp_coolant_pairs = [
        (4.2, "He"),
        (10.0, "He"),
        (20.0, "He"),
        (20.0, "H2"),
    ]

    n_rows = len(scenarios)
    n_cols = len(temp_coolant_pairs)

    # Store (root, row_idx, col_idx) for later layout once panel size is known
    panel_entries = []
    panel_w = None   # use maximum width over all subplots
    panel_h = None   # use maximum height over all subplots

    for row_idx, scenario in enumerate(scenarios):
        scenario_dir = scan_usd_dir / scenario
        for col_idx, (temp, cool) in enumerate(temp_coolant_pairs):
            label = f"{temp}K_{cool}"
            # Build filename from colormap suffix (format: delta_lcoe_min_heatmap_{scenario}_{label}_{cmap_suffix}.svg)
            if cmap_suffix:
                fname = scenario_dir / f"delta_lcoe_min_heatmap_{scenario}_{label}_{cmap_suffix}.svg"
            else:
                fname = scenario_dir / f"delta_lcoe_min_heatmap_{scenario}_{label}.svg"
                if not fname.exists():
                # If a panel is missing, skip this position (leave blank)
                print(f"[warning] Missing subplot: {fname}")
                continue

            fig = sg.fromfile(str(fname))
            root = fig.getroot()
            
                # Track maximum width/height across all subplots using the SVGFigure size (not group root)
            size_w, size_h = fig.get_size()
            w = _parse_size(size_w)
            h = _parse_size(size_h)
            panel_w = w if panel_w is None else max(panel_w, w)
            panel_h = h if panel_h is None else max(panel_h, h)

            panel_entries.append((root, row_idx, col_idx))

    if not panel_entries or panel_w is None or panel_h is None:
        print(f"No stitchable delta_lcoe_min SVG subplots found (colormap: {cmap_suffix or 'default'}). Exiting.")
        return None

    # Based on maximal panel size, define uniform margins and spacing (slightly enlarged to avoid clipping)
    margin_x = panel_w * 0.1
    margin_y = panel_h * 0.1

    # Reserve space for labels (units: pt)
    label_font_size = 22  # label font size
    left_margin = 20       # spacing between left canvas edge and y‑axis label
    
    # Fixed offsets to keep tick/label distances from subplot borders consistent
    x_axis_subplot_to_tick = 25  # distance from subplot bottom edge to x ticks
    x_axis_tick_to_label = 15    # distance from x ticks to x‑axis label
    y_axis_subplot_to_tick = 15  # distance from subplot left edge to y ticks
    y_axis_tick_to_label = 15    # distance from y ticks to y‑axis label
    
    # Compute label‑area sizes
    top_label_height = label_font_size + 8  # space for temperature–coolant labels above
    bottom_label_height = x_axis_subplot_to_tick + label_font_size + x_axis_tick_to_label + label_font_size  # space for x ticks and label below
    left_label_width = left_margin + y_axis_subplot_to_tick + label_font_size * 2 + y_axis_tick_to_label + label_font_size  # space for y ticks and label at left
    right_label_width = max(label_font_size * 2 + 16, RIGHT_LABEL_WIDTH_MIN)  # space for scenario labels at right

    # Starting position of subplot region (accounting for left labels)
    subplot_start_x = left_label_width
    subplot_start_y = top_label_height

    # Estimate required canvas size including label regions
    subplot_area_w = margin_x * (n_cols+1 ) + panel_w * (n_cols)
    subplot_area_h = margin_y * (n_rows+1) + panel_h * (n_rows )
    total_w = subplot_area_w + left_label_width + right_label_width
    total_h = subplot_area_h + top_label_height + bottom_label_height

    # Construct overall canvas: must include units ("pt") to keep svgutils sizing correct
    fig_out = sg.SVGFigure(f"{total_w}pt", f"{total_h}pt")
    print(f"total_w: {total_w}pt, total_h: {total_h}pt")

    # Layout each subplot on a regular grid (accounting for label offsets)
    placed_roots = []
    for root, row_idx, col_idx in panel_entries:
        x = subplot_start_x + margin_x + col_idx * (panel_w + margin_x)
        y = subplot_start_y + margin_y + row_idx * (panel_h + margin_y)
        root.moveto(x, y)
        placed_roots.append(root)

    fig_out.append(placed_roots)

    # Ensure stitched output folder exists
    stitched_dir = scan_usd_dir / "stitched"
    stitched_dir.mkdir(parents=True, exist_ok=True)

    # Build output filename based on colormap suffix
    if cmap_suffix:
        out_path = stitched_dir / f"delta_lcoe_min_heatmap_grid_S1-S3_{cmap_suffix}.svg"
    else:
        out_path = stitched_dir / "delta_lcoe_min_heatmap_grid_S1-S3.svg"
    fig_out.save(str(out_path))

    # After saving, reopen SVG and add labels / set canvas size precisely
    import xml.etree.ElementTree as ET
    tree = ET.parse(str(out_path))
    root_svg = tree.getroot()
    
    # Get SVG namespace from root element
    svg_ns = None
    for prefix, uri in root_svg.attrib.items():
        if prefix.startswith('xmlns') and 'svg' in uri.lower():
            svg_ns = uri
            break
    if svg_ns is None:
        # Fall back to default namespace if none is found
        svg_ns = 'http://www.w3.org/2000/svg'
    
    # Register namespace
    ET.register_namespace('', svg_ns)
    ns_map = {'svg': svg_ns}
    
    # Set size and viewBox of root SVG element
    root_svg.set('width', f"{total_w}pt")
    root_svg.set('height', f"{total_h}pt")
    root_svg.set('viewBox', f"0 0 {total_w} {total_h}")
    
    # Create label group (with correct namespace)
    labels_group = ET.SubElement(root_svg, f'{{{svg_ns}}}g', {'id': 'labels'})
    
    # Text helper with unified style
    def create_text(x, y, text, font_size=label_font_size, anchor='middle', baseline='middle', bold=False):
        """Create a text element using the correct SVG namespace."""
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
    
    def create_temp_coolant_label(x, y, temp, cool, font_size=label_font_size):
        """Create temperature–coolant labels, supporting subscript for H2."""
        # Format temperature: show 10.0 and 20.0 as integers
        if temp == 10.0 or temp == 20.0:
            temp_str = str(int(temp))
        else:
            temp_str = str(temp)
        
        text_elem = ET.SubElement(labels_group, f'{{{svg_ns}}}text', {
            'x': str(x),
            'y': str(y),
            'font-family': 'Arial, sans-serif',
            'font-size': str(font_size),
            'text-anchor': 'middle',
            'dominant-baseline': 'bottom',
            'fill': 'black'
        })
        
        # Add temperature part (space between number and unit)
        tspan1 = ET.SubElement(text_elem, f'{{{svg_ns}}}tspan')
        tspan1.text = f"{temp_str} K "
        
        # Add coolant part (for H2, place "2" as subscript)
        if cool == "H2":
            tspan2 = ET.SubElement(text_elem, f'{{{svg_ns}}}tspan')
            tspan2.text = "H"
            tspan3 = ET.SubElement(text_elem, f'{{{svg_ns}}}tspan', {
                'baseline-shift': 'sub',
                'font-size': str(font_size * 0.7)
            })
            tspan3.text = "2"
        else:
            tspan2 = ET.SubElement(text_elem, f'{{{svg_ns}}}tspan')
            tspan2.text = cool
        
        return text_elem
    
    # 1. Top: temperature–coolant labels for each column
    for col_idx, (temp, cool) in enumerate(temp_coolant_pairs):
        x_center = subplot_start_x + margin_x + col_idx * (panel_w + margin_x) + panel_w / 2
        y_pos = top_label_height - 8
        create_temp_coolant_label(x_center, y_pos, temp, cool, font_size=label_font_size)
    
    # 2. Bottom: x‑axis ticks (from config) and axis label
    # Tick positions are based on subplot x‑range in log space
    x_ticks = X_AXIS_TICKS_NOHM
    x_tick_log = [np.log10(tick) for tick in x_ticks]  # log10 positions
    
    # Bottom edge of the lowest row of subplots
    bottom_row_subplot_bottom = subplot_start_y + margin_y + (n_rows - 1) * (panel_h + margin_y) + panel_h
    
    for col_idx in range(n_cols):
        subplot_x_start = subplot_start_x + margin_x + col_idx * (panel_w + margin_x)
        subplot_x_width = panel_w
        
        for tick_val, tick_log in zip(x_ticks, x_tick_log):
            # Map log‑space tick to pixel coordinate within subplot
            tick_x_in_subplot = (tick_log - X_AXIS_MIN_LOG) / (X_AXIS_MAX_LOG - X_AXIS_MIN_LOG) * subplot_x_width
            tick_x_global = subplot_x_start + tick_x_in_subplot
            
            # Position for tick labels (subplot bottom + fixed offset)
            y_tick = bottom_row_subplot_bottom + x_axis_subplot_to_tick
            create_text(tick_x_global, y_tick, str(tick_val), font_size=label_font_size, anchor='middle', baseline='top')
    
    # X‑axis label (centered; positioned at fixed distance below ticks)
    x_axis_label_x = subplot_start_x + subplot_area_w / 2
    x_axis_label_y = bottom_row_subplot_bottom + x_axis_subplot_to_tick + label_font_size + x_axis_tick_to_label
    create_text(x_axis_label_x, x_axis_label_y, to_title_case("Coil-to-coil Joint Resistance (nΩ)"), font_size=label_font_size, anchor='middle', baseline='top', bold=False)
    
    # 3. Left: y‑axis ticks (from config) and axis label
    y_ticks = Y_AXIS_TICKS
    # Use NPW scan range as "edge coordinates" so ticks align with cell centers
    y_min = Y_AXIS_MIN
    y_max = Y_AXIS_MAX
    y_range = y_max - y_min
    
    # Left edge of the leftmost column of subplots
    left_col_subplot_left = subplot_start_x + margin_x
    
    for row_idx in range(n_rows):
        subplot_y_start = subplot_start_y + margin_y + row_idx * (panel_h + margin_y)
        subplot_y_height = panel_h
        
        for tick_val in y_ticks:
            # Map data coordinate into subplot‑local pixel coordinate (align with cell centers)
            # Note: SVG y‑axis points downward, so we measure from the bottom.
            # Linear mapping: tick_val in [y_min, y_max] -> [bottom, top] of subplot
            t_norm = (tick_val - y_min) / y_range if y_range > 0 else 0.0
            # Clamp t_norm into [0, 1]
            t_norm = max(0.0, min(1.0, t_norm))
            # Distance measured upward from subplot bottom (subplot coordinates)
            tick_y_from_bottom = t_norm * subplot_y_height
            # Map to global Y (SVG y points downward, so subtract from bottom)
            tick_y_global = subplot_y_start + subplot_y_height - tick_y_from_bottom
            # Ensure tick_y_global stays within subplot vertical range
            tick_y_global = max(subplot_y_start, min(subplot_y_start + subplot_y_height, tick_y_global))
            
            # Choose baseline alignment depending on tick position:
            # topmost tick (max NPW) uses 'bottom'; bottommost uses 'top'; middle ticks use 'middle'.
            if tick_val == y_ticks[-1]:
                baseline_align = 'bottom'
            elif tick_val == y_ticks[0]:
                baseline_align = 'top'
            else:
                baseline_align = 'middle'
            
            # Tick‑label x‑position (fixed distance left of subplot)
            x_tick = left_col_subplot_left - y_axis_subplot_to_tick
            create_text(x_tick, tick_y_global, str(tick_val), font_size=label_font_size, anchor='end', baseline=baseline_align)
    
    # Y‑axis label (centered, rotated 90°, positioned at fixed offset from ticks)
    y_axis_label_x = left_col_subplot_left - y_axis_subplot_to_tick - label_font_size * 2 - y_axis_tick_to_label
    y_axis_label_y = subplot_start_y + subplot_area_h / 2
    y_label_elem = ET.SubElement(labels_group, f'{{{svg_ns}}}text', {
        'x': str(y_axis_label_x),
        'y': str(y_axis_label_y),
        'font-family': 'Arial, sans-serif',
        'font-size': str(label_font_size),
        'text-anchor': 'middle',
        'dominant-baseline': 'middle',
        'fill': 'black',
        'transform': f'rotate(-90 {y_axis_label_x} {y_axis_label_y})'
    })
    y_label_elem.text = to_title_case("Number of Parallel Tapes")
    
    # 4. Right: scenario label for each row
    for row_idx, scenario in enumerate(scenarios):
        label_text = scenario
        x_pos = subplot_start_x + subplot_area_w + 16  # fixed horizontal spacing
        y_center = subplot_start_y + margin_y + row_idx * (panel_h + margin_y) + panel_h / 2
        create_text(x_pos, y_center, label_text, font_size=label_font_size, anchor='start', baseline='middle', bold=False)
    
    # Ensure labels group lives directly under root and uses correct namespace prefix
    # Move labels group to the end so it renders on top of other elements
    root_svg.remove(labels_group)
    root_svg.append(labels_group)
    
    # Save file, keeping namespaces consistent
    tree.write(str(out_path), encoding='utf-8', xml_declaration=True)

    # Optionally export PDF/PNG afterwards (width/height/viewBox are now updated)
    # export_svg_to_pdf(out_path)
    export_svg_to_png(out_path)
    
    print(f"Stitching complete. Combined figure saved to: {out_path}")
    print("Labels added: top temperature–coolant labels; bottom x‑axis ticks (1, 10, 100) and label; "
          "left y‑axis ticks (10, 50, 100, 200) and label; right scenario labels (S1, S2, S3).")
    return out_path


def stitch_delta_lcoe_min_svgs():
    """
    Stitch `delta_lcoe_min_heatmap` subplots into large SVG grids for all configured colormaps.
    """
    for cmap_name in cfg.color_schemes:
        print(f"\nStitching delta_lcoe_min heatmaps (colormap: {cmap_name})...")
        stitch_delta_lcoe_min_svgs_single_cmap(cmap_name)
    
    print("\nStitching of all colormaps completed.")


def stitch_hts_tape_sensitivity_svgs():
    """
    Stitch HTS tape‑price sensitivity `delta_lcoe_min_heatmap` subplots into large SVG grids.
    
    Scenarios: S5, S2, S6 (corresponding to 100, 50, 10 $/kA‑m).
    Supports all configured colormaps.
    """
    for cmap_name in cfg.color_schemes:
        print(f"\nStitching HTS‑tape‑sensitivity heatmaps (colormap: {cmap_name})...")
        # New path layout: outputs/figures/economic/{cmap_name}/parasitic_ratio/USD
        scan_base_dir = Path(cfg.ECONOMIC_FIGURES_DIR) / cmap_name / cfg.PARASITIC_RATIO_OUTPUT_DIR
        scan_usd_dir = scan_base_dir / cfg.PARASITIC_RATIO_USD_DIR

        # Rows: technology scenarios (HTS tape price sensitivity analysis)
        scenarios = ["S5", "S2", "S6"]
        # Right‑side labels: corresponding HTS tape prices
        hts_price_labels = ["100 $/kA-m", "50 $/kA-m", "10 $/kA-m"]
        # Columns: temperature–coolant pairs (consistent with parameter scan)
        temp_coolant_pairs = [
            (4.2, "He"),
            (10.0, "He"),
            (20.0, "He"),
            (20.0, "H2"),
        ]

        n_rows = len(scenarios)
        n_cols = len(temp_coolant_pairs)

        # Store (root, row_idx, col_idx) for later layout once panel size is known
        panel_entries = []
        panel_w = None   # use maximum width over all subplots
        panel_h = None   # use maximum height over all subplots

        for row_idx, scenario in enumerate(scenarios):
            scenario_dir = scan_usd_dir / scenario
            for col_idx, (temp, cool) in enumerate(temp_coolant_pairs):
                label = f"{temp}K_{cool}"
                # File naming pattern: delta_lcoe_min_heatmap_{scenario}_{label}_{cmap_name}.svg
                fname = scenario_dir / f"delta_lcoe_min_heatmap_{scenario}_{label}_{cmap_name}.svg"
                if not fname.exists():
                    # If a panel is missing, skip this position (leave blank)
                    print(f"[warning] Missing subplot: {fname}")
                    continue

                fig = sg.fromfile(str(fname))
                root = fig.getroot()

                # Track maximum width/height using SVGFigure size (not group root)
                size_w, size_h = fig.get_size()
                w = _parse_size(size_w)
                h = _parse_size(size_h)
                panel_w = w if panel_w is None else max(panel_w, w)
                panel_h = h if panel_h is None else max(panel_h, h)

                panel_entries.append((root, row_idx, col_idx))

        if not panel_entries or panel_w is None or panel_h is None:
            print(f"No delta_lcoe_min SVG panels found to stitch (cmap: {cmap_name}); skipping.")
            continue

    # Based on maximal panel size, define uniform margins and spacing (slightly enlarged to avoid clipping)
    margin_x = panel_w * 0.1
    margin_y = panel_h * 0.1

        # Reserve space for labels (units: pt)
        label_font_size = 22  # label font size
        left_margin = 20       # spacing between left canvas edge and y‑axis label
        
        # Fixed offsets to keep tick/label distances from subplot borders consistent
        x_axis_subplot_to_tick = 25  # distance from subplot bottom edge to x ticks
        x_axis_tick_to_label = 15    # distance from x ticks to x‑axis label
        y_axis_subplot_to_tick = 15  # distance from subplot left edge to y ticks
        y_axis_tick_to_label = 15    # distance from y ticks to y‑axis label
        
        # Compute label‑area sizes
        top_label_height = label_font_size + 8  # space for temperature labels above
        bottom_label_height = x_axis_subplot_to_tick + label_font_size + x_axis_tick_to_label + label_font_size  # space for x ticks and label below
        # Right‑side labels need extra room because the text is longer
        right_label_width = max(max(len(label) * label_font_size * 0.5 for label in hts_price_labels) + 20, RIGHT_LABEL_WIDTH_MIN)
        left_label_width = left_margin + y_axis_subplot_to_tick + label_font_size * 2 + y_axis_tick_to_label + label_font_size  # space for y ticks and label at left

        # Starting position of subplot region (accounting for left labels)
        subplot_start_x = left_label_width
        subplot_start_y = top_label_height

        # Estimate required canvas size including label regions
        subplot_area_w = margin_x * (n_cols+1 ) + panel_w * (n_cols)
        subplot_area_h = margin_y * (n_rows+1) + panel_h * (n_rows )
        total_w = subplot_area_w + left_label_width + right_label_width
        total_h = subplot_area_h + top_label_height + bottom_label_height

        # Construct overall canvas: must include units ("pt") so svgutils keeps sizing correct
        fig_out = sg.SVGFigure(f"{total_w}pt", f"{total_h}pt")
        print(f"total_w: {total_w}pt, total_h: {total_h}pt")

        # Layout each subplot on a regular grid (accounting for label offsets)
        placed_roots = []
        for root, row_idx, col_idx in panel_entries:
            x = subplot_start_x + margin_x + col_idx * (panel_w + margin_x)
            y = subplot_start_y + margin_y + row_idx * (panel_h + margin_y)
            root.moveto(x, y)
            placed_roots.append(root)

        fig_out.append(placed_roots)

        # Ensure stitched output folder exists
        stitched_dir = scan_usd_dir / "stitched"
        stitched_dir.mkdir(parents=True, exist_ok=True)

        out_path = stitched_dir / f"delta_lcoe_min_heatmap_grid_HTS_tape_sensitivity_{cmap_name}.svg"
        fig_out.save(str(out_path))

        # After saving, reopen SVG and add labels / set canvas size precisely
        import xml.etree.ElementTree as ET
        tree = ET.parse(str(out_path))
        root_svg = tree.getroot()
        
        # Get SVG namespace from root element
        svg_ns = None
        for prefix, uri in root_svg.attrib.items():
            if prefix.startswith('xmlns') and 'svg' in uri.lower():
                svg_ns = uri
                break
        if svg_ns is None:
            # Fall back to default namespace if none is found
            svg_ns = 'http://www.w3.org/2000/svg'
        
        # Register namespace
        ET.register_namespace('', svg_ns)
        ns_map = {'svg': svg_ns}
        
        # Set size and viewBox of root SVG element
        root_svg.set('width', f"{total_w}pt")
        root_svg.set('height', f"{total_h}pt")
        root_svg.set('viewBox', f"0 0 {total_w} {total_h}")
        
        # Create label group (with correct namespace)
        labels_group = ET.SubElement(root_svg, f'{{{svg_ns}}}g', {'id': 'labels'})
        
        # Text helper with unified style
        def create_text(x, y, text, font_size=label_font_size, anchor='middle', baseline='middle', bold=False):
            """Create a text element using the correct SVG namespace."""
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
        
        def create_temp_coolant_label(x, y, temp, cool, font_size=label_font_size):
            """Create temperature–coolant labels, supporting subscript for H2."""
            # Format temperature: show 10.0 and 20.0 as integers
            if temp == 10.0 or temp == 20.0:
                temp_str = str(int(temp))
            else:
                temp_str = str(temp)
            
            text_elem = ET.SubElement(labels_group, f'{{{svg_ns}}}text', {
                'x': str(x),
                'y': str(y),
                'font-family': 'Arial, sans-serif',
                'font-size': str(font_size),
                'text-anchor': 'middle',
                'dominant-baseline': 'bottom',
                'fill': 'black'
            })
            
            # Add temperature part (space between number and unit)
            tspan1 = ET.SubElement(text_elem, f'{{{svg_ns}}}tspan')
            tspan1.text = f"{temp_str} K "
            
            # Add coolant part (for H2, place "2" as subscript)
            if cool == "H2":
                tspan2 = ET.SubElement(text_elem, f'{{{svg_ns}}}tspan')
                tspan2.text = "H"
                tspan3 = ET.SubElement(text_elem, f'{{{svg_ns}}}tspan', {
                    'baseline-shift': 'sub',
                    'font-size': str(font_size * 0.7)
                })
                tspan3.text = "2"
            else:
                tspan2 = ET.SubElement(text_elem, f'{{{svg_ns}}}tspan')
                tspan2.text = cool
            
            return text_elem
        
        # 1. Top: temperature–coolant labels for each column
        for col_idx, (temp, cool) in enumerate(temp_coolant_pairs):
            x_center = subplot_start_x + margin_x + col_idx * (panel_w + margin_x) + panel_w / 2
            y_pos = top_label_height - 8
            create_temp_coolant_label(x_center, y_pos, temp, cool, font_size=label_font_size)
        
        # 2. Bottom: x‑axis ticks (from config) and axis label
        x_ticks = X_AXIS_TICKS_NOHM
        x_tick_log = [np.log10(tick) for tick in x_ticks]  # log10 positions
        
        # Bottom edge of the lowest row of subplots
        bottom_row_subplot_bottom = subplot_start_y + margin_y + (n_rows - 1) * (panel_h + margin_y) + panel_h
        
        for col_idx in range(n_cols):
            subplot_x_start = subplot_start_x + margin_x + col_idx * (panel_w + margin_x)
            subplot_x_width = panel_w
            
            for tick_val, tick_log in zip(x_ticks, x_tick_log):
                # Map log‑space tick to pixel coordinate within subplot
                tick_x_in_subplot = (tick_log - X_AXIS_MIN_LOG) / (X_AXIS_MAX_LOG - X_AXIS_MIN_LOG) * subplot_x_width
                tick_x_global = subplot_x_start + tick_x_in_subplot
                
                # Position for tick labels (subplot bottom + fixed offset)
                y_tick = bottom_row_subplot_bottom + x_axis_subplot_to_tick
                create_text(tick_x_global, y_tick, str(tick_val), font_size=label_font_size, anchor='middle', baseline='top')
        
        # X‑axis label (centered; positioned at fixed distance below ticks)
        x_axis_label_x = subplot_start_x + subplot_area_w / 2
        x_axis_label_y = bottom_row_subplot_bottom + x_axis_subplot_to_tick + label_font_size + x_axis_tick_to_label
        create_text(x_axis_label_x, x_axis_label_y, to_title_case("Coil-to-coil Joint Resistance (nΩ)"), font_size=label_font_size, anchor='middle', baseline='top', bold=False)
        
        # 3. Left: y‑axis ticks (from config) and axis label
        y_ticks = Y_AXIS_TICKS
        # Use NPW scan range as "edge coordinates" so ticks align with cell centers
        y_min = Y_AXIS_MIN
        y_max = Y_AXIS_MAX
        y_range = y_max - y_min
        
        # Left edge of the leftmost column of subplots
        left_col_subplot_left = subplot_start_x + margin_x
        
        for row_idx in range(n_rows):
            subplot_y_start = subplot_start_y + margin_y + row_idx * (panel_h + margin_y)
            subplot_y_height = panel_h
            
            for tick_val in y_ticks:
                # Map data coordinate into subplot‑local pixel coordinate (align with cell centers)
                # Note: SVG y‑axis points downward, so we measure from the bottom.
                # In matplotlib's pcolormesh, data points correspond to cell centers.
                t_norm = (tick_val - y_min) / y_range if y_range > 0 else 0.0
                # Clamp t_norm into [0, 1]
                t_norm = max(0.0, min(1.0, t_norm))
                # Distance measured upward from subplot bottom (subplot coordinates)
                tick_y_from_bottom = t_norm * subplot_y_height
                # Map to global Y (SVG y points downward, so subtract from bottom)
                tick_y_global = subplot_y_start + subplot_y_height - tick_y_from_bottom
                # Ensure tick_y_global stays within subplot vertical range
                tick_y_global = max(subplot_y_start, min(subplot_y_start + subplot_y_height, tick_y_global))
                
                # Choose baseline alignment depending on tick position:
                # topmost tick (max NPW) uses 'bottom'; bottommost uses 'top'; middle ticks use 'middle'.
                if tick_val == y_ticks[-1]:
                    baseline_align = 'bottom'
                elif tick_val == y_ticks[0]:
                    baseline_align = 'top'
                else:
                    baseline_align = 'middle'
                
                # Tick‑label x‑position (fixed distance left of subplot)
                x_tick = left_col_subplot_left - y_axis_subplot_to_tick
                create_text(x_tick, tick_y_global, str(tick_val), font_size=label_font_size, anchor='end', baseline=baseline_align)
        
        # Y‑axis label (centered, rotated 90°, positioned at fixed offset from ticks)
        y_axis_label_x = left_col_subplot_left - y_axis_subplot_to_tick - label_font_size * 2 - y_axis_tick_to_label
        y_axis_label_y = subplot_start_y + subplot_area_h / 2
        y_label_elem = ET.SubElement(labels_group, f'{{{svg_ns}}}text', {
            'x': str(y_axis_label_x),
            'y': str(y_axis_label_y),
            'font-family': 'Arial, sans-serif',
            'font-size': str(label_font_size),
            'text-anchor': 'middle',
            'dominant-baseline': 'middle',
            'fill': 'black',
            'transform': f'rotate(-90 {y_axis_label_x} {y_axis_label_y})'
        })
        y_label_elem.text = to_title_case("Number of Parallel Tapes")
        
        # 4. Right: HTS tape‑price label for each row
        for row_idx, price_label in enumerate(hts_price_labels):
            x_pos = subplot_start_x + subplot_area_w + 16  # fixed spacing
            y_center = subplot_start_y + margin_y + row_idx * (panel_h + margin_y) + panel_h / 2
            create_text(x_pos, y_center, price_label, font_size=label_font_size, anchor='start', baseline='middle', bold=False)
        
        # Ensure labels group lives directly under root and uses correct namespace prefix
        # Move labels group to the end so it renders above other elements
        root_svg.remove(labels_group)
        root_svg.append(labels_group)
        
        # Save file, keeping namespaces consistent
        tree.write(str(out_path), encoding='utf-8', xml_declaration=True)
        
        # Finally export PNG (width/height/viewBox are already updated)
        # export_svg_to_pdf(out_path)
        export_svg_to_png(out_path)
        
        print(f"Stitching complete. HTS‑tape‑sensitivity figure saved to: {out_path}")
        print("Labels added: top temperature labels; bottom x‑axis ticks (1, 10, 100) and label; "
              "left y‑axis ticks (10, 50, 100, 200) and label; right HTS tape‑price labels (100, 50, 10 $/kA‑m).")


def stitch_parasitic_ratio_svgs():
    """
    Stitch `parasitic_ratio_heatmap` subplots into a 3×3 SVG grid.
    
    Layout:
        - Rows (top to bottom): S1, S2, S3
        - Columns (left to right): 4.2 K, 10 K, 20 K (all He)
    
    Supports all configured colormaps.
    """
    for cmap_name in cfg.color_schemes:
        print(f"\nStitching parasitic‑ratio heatmaps (colormap: {cmap_name})...")
        # New path layout: outputs/figures/economic/{cmap_name}/parasitic_ratio/USD
        scan_base_dir = Path(cfg.ECONOMIC_FIGURES_DIR) / cmap_name / cfg.PARASITIC_RATIO_OUTPUT_DIR
        scan_usd_dir = scan_base_dir / cfg.PARASITIC_RATIO_USD_DIR

        # Rows: technology scenarios
        scenarios = ["S1", "S2", "S3"]
        # Columns: temperatures (all He, so coolant label is unnecessary)
        temperatures = [4.2, 10.0, 20.0]

        n_rows = len(scenarios)
        n_cols = len(temperatures)

        # Store (root, row_idx, col_idx) for later layout once panel size is known
        panel_entries = []
        panel_w = None   # use maximum width over all subplots
        panel_h = None   # use maximum height over all subplots

        for row_idx, scenario in enumerate(scenarios):
            scenario_dir = scan_usd_dir / scenario
            for col_idx, temp in enumerate(temperatures):
                # File naming pattern: parasitic_ratio_heatmap_{scenario}_{temp}K_{cmap_name}.svg
                fname = scenario_dir / f"parasitic_ratio_heatmap_{scenario}_{temp}K_{cmap_name}.svg"
                if not fname.exists():
                    # If a panel is missing, skip this position (leave blank)
                    print(f"[warning] Missing subplot: {fname}")
                    continue

                fig = sg.fromfile(str(fname))
                root = fig.getroot()

                # Track maximum width/height using SVGFigure size (not group root)
                size_w, size_h = fig.get_size()
                w = _parse_size(size_w)
                h = _parse_size(size_h)
                panel_w = w if panel_w is None else max(panel_w, w)
                panel_h = h if panel_h is None else max(panel_h, h)

                panel_entries.append((root, row_idx, col_idx))

        if not panel_entries or panel_w is None or panel_h is None:
            print(f"No parasitic_ratio SVG subplots found to stitch (cmap: {cmap_name}); skipping.")
            continue

        # Based on maximal panel size, define uniform margins and spacing (slightly enlarged to avoid clipping)
        margin_x = panel_w * 0.1
        margin_y = panel_h * 0.1

        # Reserve space for labels (units: pt)
        label_font_size = 22  # label font size
        left_margin = 20       # spacing between left canvas edge and y‑axis label
        
        # Fixed offsets to keep tick/label distances from subplot borders consistent
        x_axis_subplot_to_tick = 25  # distance from subplot bottom edge to x ticks
        x_axis_tick_to_label = 15    # distance from x ticks to x‑axis label
        y_axis_subplot_to_tick = 15  # distance from subplot left edge to y ticks
        y_axis_tick_to_label = 15    # distance from y ticks to y‑axis label
        
        # Compute label‑area sizes
        top_label_height = label_font_size + 8  # space for temperature labels above
        bottom_label_height = x_axis_subplot_to_tick + label_font_size + x_axis_tick_to_label + label_font_size  # space for x ticks and label below
        left_label_width = left_margin + y_axis_subplot_to_tick + label_font_size * 2 + y_axis_tick_to_label + label_font_size  # space for y ticks and label at left
        right_label_width = max(label_font_size * 2 + 16, RIGHT_LABEL_WIDTH_MIN)  # space for scenario labels at right

        # Starting position of subplot region (accounting for left labels)
        subplot_start_x = left_label_width
        subplot_start_y = top_label_height

        # Estimate required canvas size including label regions
        subplot_area_w = margin_x * (n_cols+1 ) + panel_w * (n_cols)
        subplot_area_h = margin_y * (n_rows+1) + panel_h * (n_rows )
        total_w = subplot_area_w + left_label_width + right_label_width
        total_h = subplot_area_h + top_label_height + bottom_label_height

        # Construct overall canvas: must include units ("pt") so svgutils keeps sizing correct
        fig_out = sg.SVGFigure(f"{total_w}pt", f"{total_h}pt")
        print(f"total_w: {total_w}pt, total_h: {total_h}pt")

        # Layout each subplot on a regular grid (accounting for label offsets)
        placed_roots = []
        for root, row_idx, col_idx in panel_entries:
            x = subplot_start_x + margin_x + col_idx * (panel_w + margin_x)
            y = subplot_start_y + margin_y + row_idx * (panel_h + margin_y)
            root.moveto(x, y)
            placed_roots.append(root)

        fig_out.append(placed_roots)

        # Ensure stitched output folder exists
        stitched_dir = scan_usd_dir / "stitched"
        stitched_dir.mkdir(parents=True, exist_ok=True)

        out_path = stitched_dir / "parasitic_ratio_heatmap_grid_S1-S3.svg"
        fig_out.save(str(out_path))

        # After saving, reopen SVG and add labels / set canvas size precisely
        import xml.etree.ElementTree as ET
        tree = ET.parse(str(out_path))
        root_svg = tree.getroot()
        
        # Get SVG namespace from root element
        svg_ns = None
        for prefix, uri in root_svg.attrib.items():
            if prefix.startswith('xmlns') and 'svg' in uri.lower():
                svg_ns = uri
                break
        if svg_ns is None:
            # Fall back to default namespace if none is found
            svg_ns = 'http://www.w3.org/2000/svg'
        
        # Register namespace
        ET.register_namespace('', svg_ns)
        ns_map = {'svg': svg_ns}    
        
        # Set size and viewBox of root SVG element
        root_svg.set('width', f"{total_w}pt")
        root_svg.set('height', f"{total_h}pt")
        root_svg.set('viewBox', f"0 0 {total_w} {total_h}")
        
        # Create label group (with correct namespace)
        labels_group = ET.SubElement(root_svg, f'{{{svg_ns}}}g', {'id': 'labels'})
        
        # Text helper with unified style
        def create_text(x, y, text, font_size=label_font_size, anchor='middle', baseline='middle', bold=False):
            """Create a text element using the correct SVG namespace."""
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
        
        def create_temp_label(x, y, temp, font_size=label_font_size):
            """Create temperature labels (10.0 and 20.0 rendered as integers)."""
            # Format temperature: display 10.0 and 20.0 as integers
            if temp == 10.0 or temp == 20.0:
                temp_str = str(int(temp))
            else:
                temp_str = str(temp)
            
            label_text = f"{temp_str} K"
            create_text(x, y, label_text, font_size=font_size, anchor='middle', baseline='middle')
        
        # 1. Top: temperature labels for each column
        for col_idx, temp in enumerate(temperatures):
            x_center = subplot_start_x + margin_x + col_idx * (panel_w + margin_x) + panel_w / 2
            y_pos = top_label_height - 8
            create_temp_label(x_center, y_pos, temp, font_size=label_font_size)
        
        # 2. Bottom: x‑axis ticks (from config) and axis label
        x_ticks = X_AXIS_TICKS_NOHM
        x_tick_log = [np.log10(tick) for tick in x_ticks]  # log10 positions
        
        # Bottom edge of the lowest row of subplots
        bottom_row_subplot_bottom = subplot_start_y + margin_y + (n_rows - 1) * (panel_h + margin_y) + panel_h
        
        for col_idx in range(n_cols):
            subplot_x_start = subplot_start_x + margin_x + col_idx * (panel_w + margin_x)
            subplot_x_width = panel_w
            
            for tick_val, tick_log in zip(x_ticks, x_tick_log):
                # Map log‑space tick to pixel coordinate within subplot
                tick_x_in_subplot = (tick_log - X_AXIS_MIN_LOG) / (X_AXIS_MAX_LOG - X_AXIS_MIN_LOG) * subplot_x_width
                tick_x_global = subplot_x_start + tick_x_in_subplot
                
                # Position for tick labels (subplot bottom + fixed offset)
                y_tick = bottom_row_subplot_bottom + x_axis_subplot_to_tick
                create_text(tick_x_global, y_tick, str(tick_val), font_size=label_font_size, anchor='middle', baseline='top')
        
        # X‑axis label (centered; positioned at fixed distance below ticks)
        x_axis_label_x = subplot_start_x + subplot_area_w / 2
        x_axis_label_y = bottom_row_subplot_bottom + x_axis_subplot_to_tick + label_font_size + x_axis_tick_to_label
        create_text(x_axis_label_x, x_axis_label_y, to_title_case("Coil-to-coil Joint Resistance (nΩ)"), font_size=label_font_size, anchor='middle', baseline='top', bold=False)
        
        # 3. Left: y‑axis ticks (from config) and axis label
        y_ticks = Y_AXIS_TICKS
        # Use NPW scan range as "edge coordinates" so ticks align with cell centers
        y_min = Y_AXIS_MIN
        y_max = Y_AXIS_MAX
        y_range = y_max - y_min
        
        # Left edge of the leftmost column of subplots
        left_col_subplot_left = subplot_start_x + margin_x
        
        for row_idx in range(n_rows):
            subplot_y_start = subplot_start_y + margin_y + row_idx * (panel_h + margin_y)
            subplot_y_height = panel_h
            
            for tick_val in y_ticks:
                # Map data coordinate into subplot‑local pixel coordinate (align with cell centers)
                # Note: SVG y‑axis points downward, so we measure from the bottom.
                # In matplotlib's pcolormesh, data points correspond to cell centers.
                t_norm = (tick_val - y_min) / y_range if y_range > 0 else 0.0
                # Clamp t_norm into [0, 1]
                t_norm = max(0.0, min(1.0, t_norm))
                # Distance measured upward from subplot bottom (subplot coordinates)
                tick_y_from_bottom = t_norm * subplot_y_height
                # Map to global Y (SVG y points downward, so subtract from bottom)
                tick_y_global = subplot_y_start + subplot_y_height - tick_y_from_bottom
                # Ensure tick_y_global stays within subplot vertical range
                tick_y_global = max(subplot_y_start, min(subplot_y_start + subplot_y_height, tick_y_global))
                
                # Tick‑label x‑position (fixed distance left of subplot)
                x_tick = left_col_subplot_left - y_axis_subplot_to_tick
                create_text(x_tick, tick_y_global, str(tick_val), font_size=label_font_size, anchor='end', baseline='middle')
        
        # Y‑axis label (centered, rotated 90°, positioned at fixed offset from ticks)
        y_axis_label_x = left_col_subplot_left - y_axis_subplot_to_tick - label_font_size * 2 - y_axis_tick_to_label
        y_axis_label_y = subplot_start_y + subplot_area_h / 2
        y_label_elem = ET.SubElement(labels_group, f'{{{svg_ns}}}text', {
            'x': str(y_axis_label_x),
            'y': str(y_axis_label_y),
            'font-family': 'Arial, sans-serif',
            'font-size': str(label_font_size),
            'text-anchor': 'middle',
            'dominant-baseline': 'middle',
            'fill': 'black',
            'transform': f'rotate(-90 {y_axis_label_x} {y_axis_label_y})'
        })
        y_label_elem.text = to_title_case("Number of Parallel Tapes")
        
        # 4. Right: scenario label for each row
        for row_idx, scenario in enumerate(scenarios):
            x_pos = subplot_start_x + subplot_area_w + 16  # fixed spacing
            y_center = subplot_start_y + margin_y + row_idx * (panel_h + margin_y) + panel_h / 2
            create_text(x_pos, y_center, scenario, font_size=label_font_size, anchor='start', baseline='middle', bold=False)
        
        # Ensure labels group lives directly under root and uses correct namespace prefix
        # Move labels group to the end so it renders above other elements
        root_svg.remove(labels_group)
        root_svg.append(labels_group)
        
        # Save file, keeping namespaces consistent
        tree.write(str(out_path), encoding='utf-8', xml_declaration=True)
        
        # Finally export PNG (width/height/viewBox are already updated)
        # export_svg_to_pdf(out_path)
        export_svg_to_png(out_path)
        
        print(f"Stitching complete. Parasitic‑ratio figure saved to: {out_path}")
        print("Labels added: top temperature labels (4.2 K, 10 K, 20 K); bottom x‑axis ticks (1, 10, 100) and label; "
              "left y‑axis ticks (10, 50, 100, 200) and label; right scenario labels (S1, S2, S3).")


def stitch_af_heatmap_svgs():
    """
    Stitch AF heatmap (`AF_heatmap`) subplots into a 3×3 SVG grid.
    
    Layout:
        - Rows (top to bottom): S1, S2, S3
        - Columns (left to right): 4.2 K, 10 K, 20 K
    
    Supports all configured colormaps.
    """
    for cmap_name in cfg.color_schemes:
        print(f"\nStitching AF heatmaps (colormap: {cmap_name})...")
        # New path layout: outputs/figures/economic/{cmap_name}/AF_heatmaps_scenarios/figures
        af_base_dir = Path(cfg.ECONOMIC_FIGURES_DIR) / cmap_name / "AF_heatmaps_scenarios" / "figures"

        # Rows: technology scenarios
        scenarios = ["S1", "S2", "S3"]
        # Columns: temperatures
        temperatures = [4.2, 10.0, 20.0]

        n_rows = len(scenarios)
        n_cols = len(temperatures)

        # Store (root, row_idx, col_idx) for later layout once panel size is known
        panel_entries = []
        panel_w = None   # use maximum width over all subplots
        panel_h = None   # use maximum height over all subplots

        # Important: within each colormap, keep style consistent (do not mix original and *_truncated)
        # If any scenario directory under this colormap contains *_truncated.svg, prefer truncated versions for all.
        prefer_truncated = False
        for _sc in scenarios:
            _dir = af_base_dir / _sc
            if _dir.exists() and any(_dir.glob("AF_heatmap_*_truncated.svg")):
                prefer_truncated = True
                break

        for row_idx, scenario in enumerate(scenarios):
            scenario_dir = af_base_dir / scenario
            if not scenario_dir.exists():
                print(f"[warning] Scenario directory does not exist: {scenario_dir}")
                continue
            for col_idx, temp in enumerate(temperatures):
                # Ensure consistent temperature formatting: always one decimal place
                temp_str = f"{temp:.1f}" if temp % 1 != 0 else f"{int(temp)}.0"
                
                # Choose filename pattern depending on prefer_truncated
                base_name = f"AF_heatmap_{scenario}_Top_{temp_str}K"
                alt_temp_str = f"{int(temp) if temp % 1 == 0 else temp}"
                alt_base_name = f"AF_heatmap_{scenario}_Top_{alt_temp_str}K"

                candidates: list[Path] = []
                if prefer_truncated:
                    candidates.extend([
                        scenario_dir / f"{base_name}_truncated.svg",
                        scenario_dir / f"{alt_base_name}_truncated.svg",
                    ])
                else:
                    candidates.extend([
                        scenario_dir / f"{base_name}.svg",
                        scenario_dir / f"{alt_base_name}.svg",
                    ])

                fname = None
                for cand in candidates:
                    if cand.exists():
                        fname = cand
                        break
                
                if fname is None:
                    existing_files = list(scenario_dir.glob(f"AF_heatmap_{scenario}_Top_*K*.svg"))
                    if existing_files:
                        print(f"[warning] Missing AF_heatmap subplot: {candidates[0]}")
                        print(f"[info] Available files in directory: {[f.name for f in existing_files]}")
                    else:
                        print(f"[warning] Missing AF_heatmap subplot: {candidates[0]}")
                        print(f"[info] No AF_heatmap files found in scenario directory {scenario_dir}")
                    continue
                
                if prefer_truncated:
                    print(f"[info] Using truncated AF_heatmap file: {fname.name}")

                fig = sg.fromfile(str(fname))
                root = fig.getroot()

                # Track maximum width/height using SVGFigure size (not group root)
                size_w, size_h = fig.get_size()
                w = _parse_size(size_w)
                h = _parse_size(size_h)
                panel_w = w if panel_w is None else max(panel_w, w)
                panel_h = h if panel_h is None else max(panel_h, h)

                panel_entries.append((root, row_idx, col_idx))

        if not panel_entries or panel_w is None or panel_h is None:
            print(f"No AF_heatmap SVG subplots found to stitch (cmap: {cmap_name}); skipping.")
            continue

        # Based on maximal panel size, define uniform margins and spacing (slightly enlarged to avoid clipping)
        margin_x = panel_w * 0.15
        margin_y = panel_h * 0.15

        # Reserve space for labels (units: pt)
        label_font_size = 22  # label font size
        left_margin = 20       # spacing between left canvas edge and y‑axis label
        
        # Fixed offsets to keep tick/label distances from subplot borders consistent
        x_axis_subplot_to_tick = 25  # distance from subplot bottom edge to x ticks
        x_axis_tick_to_label = 15    # distance from x ticks to x‑axis label
        y_axis_subplot_to_tick = 15  # distance from subplot left edge to y ticks
        y_axis_tick_to_label = 15    # distance from y ticks to y‑axis label
        
        # Compute label‑area sizes
        top_label_height = label_font_size + 8  # space for temperature labels above
        bottom_label_height = x_axis_subplot_to_tick + label_font_size + x_axis_tick_to_label + label_font_size  # space for x ticks and label below
        left_label_width = left_margin + y_axis_subplot_to_tick + label_font_size * 2 + y_axis_tick_to_label + label_font_size  # space for y ticks and label at left
        right_label_width = max(label_font_size * 2 + 16, RIGHT_LABEL_WIDTH_MIN)  # space for scenario labels at right

        # Starting position of subplot region (accounting for left labels)
        subplot_start_x = left_label_width
        subplot_start_y = top_label_height

        # Estimate required canvas size including label regions
        subplot_area_w = margin_x * (n_cols+1 ) + panel_w * (n_cols)
        subplot_area_h = margin_y * (n_rows+1) + panel_h * (n_rows )
        total_w = subplot_area_w + left_label_width + right_label_width
        total_h = subplot_area_h + top_label_height + bottom_label_height

        # Construct overall canvas: must include units ("pt") so svgutils keeps sizing correct
        fig_out = sg.SVGFigure(f"{total_w}pt", f"{total_h}pt")
        print(f"total_w: {total_w}pt, total_h: {total_h}pt")

        # Layout each subplot on a regular grid (accounting for label offsets)
        placed_roots = []
        for root, row_idx, col_idx in panel_entries:
            x = subplot_start_x + margin_x + col_idx * (panel_w + margin_x)
            y = subplot_start_y + margin_y + row_idx * (panel_h + margin_y)
            root.moveto(x, y)
            placed_roots.append(root)

        fig_out.append(placed_roots)

        # Output naming: keep suffix consistent with chosen variant (truncated vs original)
        output_suffix = "_truncated" if prefer_truncated else ""
        out_path = af_base_dir / f"AF_heatmap_grid_S1-S3{output_suffix}.svg"
        fig_out.save(str(out_path))

        # After saving, reopen SVG and add labels / set canvas size precisely
        import xml.etree.ElementTree as ET
        tree = ET.parse(str(out_path))
        root_svg = tree.getroot()
        
        # Get SVG namespace from root element
        svg_ns = None
        for prefix, uri in root_svg.attrib.items():
            if prefix.startswith('xmlns') and 'svg' in uri.lower():
                svg_ns = uri
                break
        if svg_ns is None:
            # Fall back to default namespace if none is found
            svg_ns = 'http://www.w3.org/2000/svg'
        
        # Register namespace
        ET.register_namespace('', svg_ns)
        ns_map = {'svg': svg_ns}
        
        # Set size and viewBox of root SVG element
        root_svg.set('width', f"{total_w}pt")
        root_svg.set('height', f"{total_h}pt")
        root_svg.set('viewBox', f"0 0 {total_w} {total_h}")
        
        # Create label group (with correct namespace)
        labels_group = ET.SubElement(root_svg, f'{{{svg_ns}}}g', {'id': 'labels'})
        
        # Text helper with unified style
        def create_text(x, y, text, font_size=label_font_size, anchor='middle', baseline='middle', bold=False):
            """Create a text element using the correct SVG namespace."""
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
        
        def create_temp_label(x, y, temp, font_size=label_font_size):
            """Create temperature labels (10.0 and 20.0 rendered as integers)."""
            # Format temperature: display 10.0 and 20.0 as integers
            if temp == 10.0 or temp == 20.0:
                temp_str = str(int(temp))
            else:
                temp_str = str(temp)
            
            label_text = f"{temp_str} K"
            create_text(x, y, label_text, font_size=font_size, anchor='middle', baseline='middle')
        
        # 1. Top: temperature labels for each column
        for col_idx, temp in enumerate(temperatures):
            x_center = subplot_start_x + margin_x + col_idx * (panel_w + margin_x) + panel_w / 2
            y_pos = top_label_height - 8
            create_temp_label(x_center, y_pos, temp, font_size=label_font_size)
        
        # 2. Bottom: x‑axis ticks (from config) and axis label
        # AF heatmap x‑axis is logarithmic
        x_ticks = AF_X_AXIS_TICKS_UOHM_CM2
        x_tick_log = [np.log10(tick) for tick in x_ticks]  # log10 positions
        
        # Bottom edge of the lowest row of subplots
        bottom_row_subplot_bottom = subplot_start_y + margin_y + (n_rows - 1) * (panel_h + margin_y) + panel_h
        
        for col_idx in range(n_cols):
            subplot_x_start = subplot_start_x + margin_x + col_idx * (panel_w + margin_x)
            subplot_x_width = panel_w
            
            for tick_val, tick_log in zip(x_ticks, x_tick_log):
                # Map log‑space tick to pixel coordinate within subplot
                tick_x_in_subplot = (tick_log - AF_X_AXIS_MIN_LOG) / (AF_X_AXIS_MAX_LOG - AF_X_AXIS_MIN_LOG) * subplot_x_width
                tick_x_global = subplot_x_start + tick_x_in_subplot
                
                # Position for tick labels (subplot bottom + fixed offset)
                y_tick = bottom_row_subplot_bottom + x_axis_subplot_to_tick
                create_text(tick_x_global, y_tick, str(tick_val), font_size=label_font_size, anchor='middle', baseline='top')
        
        # X‑axis label (centered; positioned at fixed distance below ticks)
        x_axis_label_x = subplot_start_x + subplot_area_w / 2
        x_axis_label_y = bottom_row_subplot_bottom + x_axis_subplot_to_tick + label_font_size + x_axis_tick_to_label
        create_text(x_axis_label_x, x_axis_label_y, to_title_case("Turn-to-turn Resistivity (μΩ·cm²)"), font_size=label_font_size, anchor='middle', baseline='top', bold=False)
        
        # 3. Left: y‑axis ticks (from config) and axis label
        y_ticks = AF_Y_AXIS_TICKS
        y_min = AF_Y_AXIS_MIN
        y_max = AF_Y_AXIS_MAX
        y_range = y_max - y_min
        
        # Left edge of the leftmost column of subplots
        left_col_subplot_left = subplot_start_x + margin_x
        
        for row_idx in range(n_rows):
            subplot_y_start = subplot_start_y + margin_y + row_idx * (panel_h + margin_y)
            subplot_y_height = panel_h
            
            for tick_val in y_ticks:
                # Map data coordinate into subplot‑local pixel coordinate (align with cell centers)
                # In matplotlib's pcolormesh, data points correspond to cell centers.
                # Note: SVG y‑axis points downward, so we measure from the bottom.
                t_norm = (tick_val - y_min) / y_range if y_range > 0 else 0.0
                # Clamp t_norm into [0, 1]
                t_norm = max(0.0, min(1.0, t_norm))
                # Distance measured upward from subplot bottom (subplot coordinates)
                tick_y_from_bottom = t_norm * subplot_y_height
                # Map to global Y (SVG y points downward, so subtract from bottom)
                tick_y_global = subplot_y_start + subplot_y_height - tick_y_from_bottom
                # Ensure tick_y_global stays within subplot vertical range
                tick_y_global = max(subplot_y_start, min(subplot_y_start + subplot_y_height, tick_y_global))
                
                # Tick‑label x‑position (fixed distance left of subplot)
                x_tick = left_col_subplot_left - y_axis_subplot_to_tick
                create_text(x_tick, tick_y_global, str(tick_val), font_size=label_font_size, anchor='end', baseline='middle')
        
        # Y‑axis label (centered, rotated 90°, positioned at fixed offset from ticks)
        y_axis_label_x = left_col_subplot_left - y_axis_subplot_to_tick - label_font_size * 2 - y_axis_tick_to_label
        y_axis_label_y = subplot_start_y + subplot_area_h / 2
        y_label_elem = ET.SubElement(labels_group, f'{{{svg_ns}}}text', {
            'x': str(y_axis_label_x),
            'y': str(y_axis_label_y),
            'font-family': 'Arial, sans-serif',
            'font-size': str(label_font_size),
            'text-anchor': 'middle',
            'dominant-baseline': 'middle',
            'fill': 'black',
            'transform': f'rotate(-90 {y_axis_label_x} {y_axis_label_y})'
        })
        y_label_elem.text = to_title_case("Number of Parallel Tapes")
        
        # 4. Right: scenario label for each row
        for row_idx, scenario in enumerate(scenarios):
            x_pos = subplot_start_x + subplot_area_w + 16  # fixed spacing
            y_center = subplot_start_y + margin_y + row_idx * (panel_h + margin_y) + panel_h / 2
            create_text(x_pos, y_center, scenario, font_size=label_font_size, anchor='start', baseline='middle', bold=False)
        
        # Ensure labels group lives directly under root and uses correct namespace prefix
        # Move labels group to the end so it renders above other elements
        root_svg.remove(labels_group)
        root_svg.append(labels_group)
        
        # Save file, keeping namespaces consistent
        tree.write(str(out_path), encoding='utf-8', xml_declaration=True)
        
        # Finally export PNG (width/height/viewBox are already updated)
        # export_svg_to_pdf(out_path)
        export_svg_to_png(out_path)
        
        print(f"Stitching complete. AF heatmap figure saved to: {out_path}")
        print("Labels added: top temperature labels (4.2 K, 10 K, 20 K); "
              "bottom x‑axis ticks (10, 100, 1000) and label; "
              "left y‑axis ticks (1, 5, 10, 20) and label; right scenario labels (S1, S2, S3).")



def stitch_af_ref_heatmap_svgs():
    """
    Stitch AF_ref heatmap (`AF_ref_heatmap`) subplots into a 3×3 SVG grid.
    
    Layout:
        - Rows (top to bottom): S1, S2, S3
        - Columns (left to right): 4.2 K, 10 K, 20 K
    
    Supports all configured colormaps.
    """
    for cmap_name in cfg.color_schemes:
        print(f"\nStitching AF_ref heatmaps (colormap: {cmap_name})...")
        # New path layout: outputs/figures/economic/{cmap_name}/AF_ref_heatmaps_scenarios/figures
        af_base_dir = Path(cfg.ECONOMIC_FIGURES_DIR) / cmap_name / "AF_ref_heatmaps_scenarios" / "figures"

        # Rows: technology scenarios
        scenarios = ["S1", "S2", "S3"]
        # Columns: temperatures
        temperatures = [4.2, 10.0, 20.0]

        n_rows = len(scenarios)
        n_cols = len(temperatures)

        # Store (root, row_idx, col_idx) for later layout once panel size is known
        panel_entries = []
        panel_w = None   # use maximum width over all subplots
        panel_h = None   # use maximum height over all subplots

        for row_idx, scenario in enumerate(scenarios):
            scenario_dir = af_base_dir / scenario
            for col_idx, temp in enumerate(temperatures):
                fname = scenario_dir / f"AF_ref_heatmap_{scenario}_Top_{temp}K.svg"
                if not fname.exists():
                    # If a panel is missing, skip this position (leave blank)
                    print(f"[warning] Missing AF_ref_heatmap subplot: {fname}")
                    continue

                fig = sg.fromfile(str(fname))
                root = fig.getroot()

                # Track maximum width/height using SVGFigure size (not group root)
                size_w, size_h = fig.get_size()
                w = _parse_size(size_w)
                h = _parse_size(size_h)
                panel_w = w if panel_w is None else max(panel_w, w)
                panel_h = h if panel_h is None else max(panel_h, h)

                panel_entries.append((root, row_idx, col_idx))

        if not panel_entries or panel_w is None or panel_h is None:
            print(f"No AF_ref_heatmap SVG subplots found to stitch (cmap: {cmap_name}); skipping.")
            continue

        # Based on maximal panel size, define uniform margins and spacing (slightly enlarged to avoid clipping)
        margin_x = panel_w * 0.15
        margin_y = panel_h * 0.15

        # Reserve space for labels (units: pt)
        label_font_size = 22  # label font size
        left_margin = 20       # spacing between left canvas edge and y‑axis label
        
        # Fixed offsets to keep tick/label distances from subplot borders consistent
        x_axis_subplot_to_tick = 25  # distance from subplot bottom edge to x ticks
        x_axis_tick_to_label = 15    # distance from x ticks to x‑axis label
        y_axis_subplot_to_tick = 15  # distance from subplot left edge to y ticks
        y_axis_tick_to_label = 15    # distance from y ticks to y‑axis label
        
        # Compute label‑area sizes
        top_label_height = label_font_size + 8  # space for temperature labels above
        bottom_label_height = x_axis_subplot_to_tick + label_font_size + x_axis_tick_to_label + label_font_size  # space for x ticks and label below
        left_label_width = left_margin + y_axis_subplot_to_tick + label_font_size * 2 + y_axis_tick_to_label + label_font_size  # space for y ticks and label at left
        right_label_width = max(label_font_size * 2 + 16, RIGHT_LABEL_WIDTH_MIN)  # space for scenario labels at right

        # Starting position of subplot region (accounting for left labels)
        subplot_start_x = left_label_width
        subplot_start_y = top_label_height

        # Estimate required canvas size including label regions
        subplot_area_w = margin_x * (n_cols+1 ) + panel_w * (n_cols)
        subplot_area_h = margin_y * (n_rows+1) + panel_h * (n_rows )
        total_w = subplot_area_w + left_label_width + right_label_width
        total_h = subplot_area_h + top_label_height + bottom_label_height

        # Construct overall canvas: must include units ("pt") so svgutils keeps sizing correct
        fig_out = sg.SVGFigure(f"{total_w}pt", f"{total_h}pt")
        print(f"total_w: {total_w}pt, total_h: {total_h}pt")

        # Layout each subplot on a regular grid (accounting for label offsets)
        placed_roots = []
        for root, row_idx, col_idx in panel_entries:
            x = subplot_start_x + margin_x + col_idx * (panel_w + margin_x)
            y = subplot_start_y + margin_y + row_idx * (panel_h + margin_y)
            root.moveto(x, y)
            placed_roots.append(root)

        fig_out.append(placed_roots)

        out_path = af_base_dir / "AF_ref_heatmap_grid_S1-S3.svg"
        fig_out.save(str(out_path))

        # After saving, reopen SVG and add labels / set canvas size precisely
        import xml.etree.ElementTree as ET
        tree = ET.parse(str(out_path))
        root_svg = tree.getroot()
        
        # Get SVG namespace from root element
        svg_ns = None
        for prefix, uri in root_svg.attrib.items():
            if prefix.startswith('xmlns') and 'svg' in uri.lower():
                svg_ns = uri
                break
        if svg_ns is None:
            # Fall back to default namespace if none is found
            svg_ns = 'http://www.w3.org/2000/svg'
        
        # Register namespace
        ET.register_namespace('', svg_ns)
        ns_map = {'svg': svg_ns}
        
        # Set size and viewBox of root SVG element
        root_svg.set('width', f"{total_w}pt")
        root_svg.set('height', f"{total_h}pt")
        root_svg.set('viewBox', f"0 0 {total_w} {total_h}")
        
        # Create label group (with correct namespace)
        labels_group = ET.SubElement(root_svg, f'{{{svg_ns}}}g', {'id': 'labels'})
        
        # Text helper with unified style
        def create_text(x, y, text, font_size=label_font_size, anchor='middle', baseline='middle', bold=False):
            """Create a text element using the correct SVG namespace."""
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
        
        def create_temp_label(x, y, temp, font_size=label_font_size):
            """Create temperature labels (10.0 and 20.0 rendered as integers)."""
            # Format temperature: display 10.0 and 20.0 as integers
            if temp == 10.0 or temp == 20.0:
                temp_str = str(int(temp))
            else:
                temp_str = str(temp)
            
            label_text = f"{temp_str} K"
            create_text(x, y, label_text, font_size=font_size, anchor='middle', baseline='middle')
        
        # 1. Top: temperature labels for each column
        for col_idx, temp in enumerate(temperatures):
            x_center = subplot_start_x + margin_x + col_idx * (panel_w + margin_x) + panel_w / 2
            y_pos = top_label_height - 8
            create_temp_label(x_center, y_pos, temp, font_size=label_font_size)
        
        # 2. Bottom: x‑axis ticks (from config) and axis label
        # AF_ref heatmap x‑axis is logarithmic
        x_ticks = AF_X_AXIS_TICKS_UOHM_CM2
        x_tick_log = [np.log10(tick) for tick in x_ticks]  # log10 positions
        
        # Bottom edge of the lowest row of subplots
        bottom_row_subplot_bottom = subplot_start_y + margin_y + (n_rows - 1) * (panel_h + margin_y) + panel_h
        
        for col_idx in range(n_cols):
            subplot_x_start = subplot_start_x + margin_x + col_idx * (panel_w + margin_x)
            subplot_x_width = panel_w
            
            for tick_val, tick_log in zip(x_ticks, x_tick_log):
                # Map log‑space tick to pixel coordinate within subplot
                tick_x_in_subplot = (tick_log - AF_X_AXIS_MIN_LOG) / (AF_X_AXIS_MAX_LOG - AF_X_AXIS_MIN_LOG) * subplot_x_width
                tick_x_global = subplot_x_start + tick_x_in_subplot
                
                # Position for tick labels (subplot bottom + fixed offset)
                y_tick = bottom_row_subplot_bottom + x_axis_subplot_to_tick
                create_text(tick_x_global, y_tick, str(tick_val), font_size=label_font_size, anchor='middle', baseline='top')
        
        # X-axis label (centered; fixed offset from tick positions)
        x_axis_label_x = subplot_start_x + subplot_area_w / 2
        x_axis_label_y = bottom_row_subplot_bottom + x_axis_subplot_to_tick + label_font_size + x_axis_tick_to_label
        create_text(x_axis_label_x, x_axis_label_y, to_title_case("Turn-to-turn Resistivity (μΩ·cm²)"), font_size=label_font_size, anchor='middle', baseline='top', bold=False)
        
        # 3. Left: y‑axis ticks (from config) and axis label
        y_ticks = AF_Y_AXIS_TICKS
        y_min = AF_Y_AXIS_MIN
        y_max = AF_Y_AXIS_MAX
        y_range = y_max - y_min
        
        # Left edge of the leftmost column of subplots
        left_col_subplot_left = subplot_start_x + margin_x
        
        for row_idx in range(n_rows):
            subplot_y_start = subplot_start_y + margin_y + row_idx * (panel_h + margin_y)
            subplot_y_height = panel_h
            
            for tick_val in y_ticks:
                # Map data coordinate into subplot‑local pixel coordinate (align with cell centers)
                # In matplotlib's pcolormesh, data points correspond to cell centers.
                # Note: SVG y‑axis points downward, so we measure from the bottom.
                t_norm = (tick_val - y_min) / y_range if y_range > 0 else 0.0
                # Clamp t_norm into [0, 1]
                t_norm = max(0.0, min(1.0, t_norm))
                # Distance measured upward from subplot bottom (subplot coordinates)
                tick_y_from_bottom = t_norm * subplot_y_height
                # Map to global Y (SVG y points downward, so subtract from bottom)
                tick_y_global = subplot_y_start + subplot_y_height - tick_y_from_bottom
                # Ensure tick_y_global stays within subplot vertical range
                tick_y_global = max(subplot_y_start, min(subplot_y_start + subplot_y_height, tick_y_global))
                
                # Tick‑label x‑position (fixed distance left of subplot)
                x_tick = left_col_subplot_left - y_axis_subplot_to_tick
                create_text(x_tick, tick_y_global, str(tick_val), font_size=label_font_size, anchor='end', baseline='middle')
        
        # Y‑axis label (centered, rotated 90°, positioned at fixed offset from ticks)
        y_axis_label_x = left_col_subplot_left - y_axis_subplot_to_tick - label_font_size * 2 - y_axis_tick_to_label
        y_axis_label_y = subplot_start_y + subplot_area_h / 2
        y_label_elem = ET.SubElement(labels_group, f'{{{svg_ns}}}text', {
            'x': str(y_axis_label_x),
            'y': str(y_axis_label_y),
            'font-family': 'Arial, sans-serif',
            'font-size': str(label_font_size),
            'text-anchor': 'middle',
            'dominant-baseline': 'middle',
            'fill': 'black',
            'transform': f'rotate(-90 {y_axis_label_x} {y_axis_label_y})'
        })
        y_label_elem.text = to_title_case("Number of Parallel Tapes")
        
        # 4. Right: scenario label for each row
        for row_idx, scenario in enumerate(scenarios):
            x_pos = subplot_start_x + subplot_area_w + 16  # fixed spacing
            y_center = subplot_start_y + margin_y + row_idx * (panel_h + margin_y) + panel_h / 2
            create_text(x_pos, y_center, scenario, font_size=label_font_size, anchor='start', baseline='middle', bold=False)
        
        # Ensure labels group lives directly under root and uses correct namespace prefix
        # Move labels group to the end so it renders above other elements
        root_svg.remove(labels_group)
        root_svg.append(labels_group)
        
        # Save file, keeping namespaces consistent
        tree.write(str(out_path), encoding='utf-8', xml_declaration=True)
        
        # Finally export PNG (width/height/viewBox are already updated)
        #export_svg_to_pdf(out_path)
        export_svg_to_png(out_path)
        
        print(f"Stitching complete. AF_ref heatmap figure saved to: {out_path}")
        print("Labels added: top temperature labels (4.2 K, 10 K, 20 K); "
              "bottom x‑axis ticks (10, 100, 1000) and label; "
              "left y‑axis ticks (1, 5, 10, 20) and label; right scenario labels (S1, S2, S3).")


def stitch_cryo_power_MWe_svgs():
    """
    Stitch cryogenic‑power MWe heatmaps into a 3×3 grid.
    
    For each colormap, subplots are sourced from
    `outputs/figures/economic/{cmap}/parasitic_ratio/cryo_power_MWe`.
    
    Layout:
        - Rows (top to bottom): Pulse, Dwell, Static
        - Columns (left to right): 4.2 K, 10 K, 20 K
    
    X‑axis: joint resistance (nΩ), Y‑axis: number of parallel tapes, values in MWe.
    """
    modes = [("pulse", "Pulse"), ("dwell", "Dwell"), ("static", "Static")]
    temperatures = [4.2, 10.0, 20.0]
    n_rows = len(modes)
    n_cols = len(temperatures)

    for cmap_name in cfg.color_schemes:
        cryo_dir = Path(cfg.ECONOMIC_FIGURES_DIR) / cmap_name / cfg.PARASITIC_RATIO_OUTPUT_DIR / "cryo_power_MWe"
        panel_entries = []
        panel_w = None
        panel_h = None

        for row_idx, (mode_key, _mode_label) in enumerate(modes):
            for col_idx, temp in enumerate(temperatures):
                if temp == 10.0 or temp == 20.0:
                    label = f"{int(temp)}K"
                else:
                    label = f"{temp}K"
                fname = cryo_dir / f"cryo_power_MWe_heatmap_{mode_key}_{label}.svg"
                if not fname.exists():
                    continue
                fig = sg.fromfile(str(fname))
                root = fig.getroot()
                size_w, size_h = fig.get_size()
                w = _parse_size(size_w)
                h = _parse_size(size_h)
                panel_w = w if panel_w is None else max(panel_w, w)
                panel_h = h if panel_h is None else max(panel_h, h)
                panel_entries.append((root, row_idx, col_idx))

        if not panel_entries or panel_w is None or panel_h is None:
            print(f"[warning] Not enough cryo power MWe subplots for colormap {cmap_name}; skipping stitch.")
            continue

        margin_x = panel_w * 0.1
        margin_y = panel_h * 0.1
        label_font_size = 22
        left_margin = 20
        x_axis_subplot_to_tick = 25
        x_axis_tick_to_label = 15
        y_axis_subplot_to_tick = 15
        y_axis_tick_to_label = 15
        top_label_height = label_font_size + 8
        bottom_label_height = x_axis_subplot_to_tick + label_font_size + x_axis_tick_to_label + label_font_size
        left_label_width = left_margin + y_axis_subplot_to_tick + label_font_size * 2 + y_axis_tick_to_label + label_font_size
        right_label_width = max(label_font_size * 2 + 16, RIGHT_LABEL_WIDTH_MIN)
        subplot_start_x = left_label_width
        subplot_start_y = top_label_height
        subplot_area_w = margin_x * (n_cols + 1) + panel_w * n_cols
        subplot_area_h = margin_y * (n_rows + 1) + panel_h * n_rows
        total_w = subplot_area_w + left_label_width + right_label_width
        total_h = subplot_area_h + top_label_height + bottom_label_height

        fig_out = sg.SVGFigure(f"{total_w}pt", f"{total_h}pt")
        placed_roots = []
        for root, row_idx, col_idx in panel_entries:
            x = subplot_start_x + margin_x + col_idx * (panel_w + margin_x)
            y = subplot_start_y + margin_y + row_idx * (panel_h + margin_y)
            root.moveto(x, y)
            placed_roots.append(root)
        fig_out.append(placed_roots)

        stitched_dir = cryo_dir / "stitched"
        stitched_dir.mkdir(parents=True, exist_ok=True)
        out_path = stitched_dir / "cryo_power_MWe_heatmap_grid_3x3.svg"
        fig_out.save(str(out_path))

        import xml.etree.ElementTree as ET
        tree = ET.parse(str(out_path))
        root_svg = tree.getroot()
        svg_ns = None
        for prefix, uri in root_svg.attrib.items():
            if prefix.startswith("xmlns") and "svg" in uri.lower():
                svg_ns = uri
                break
        if svg_ns is None:
            svg_ns = "http://www.w3.org/2000/svg"
        ET.register_namespace("", svg_ns)

        root_svg.set("width", f"{total_w}pt")
        root_svg.set("height", f"{total_h}pt")
        root_svg.set("viewBox", f"0 0 {total_w} {total_h}")
        labels_group = ET.SubElement(root_svg, f"{{{svg_ns}}}g", {"id": "labels"})

        def create_text(x, y, text, font_size=label_font_size, anchor="middle", baseline="middle", bold=False):
            text_elem = ET.SubElement(labels_group, f"{{{svg_ns}}}text", {
                "x": str(x), "y": str(y),
                "font-family": "Arial, sans-serif", "font-size": str(font_size),
                "text-anchor": anchor, "dominant-baseline": baseline, "fill": "black"
            })
            if bold:
                text_elem.set("font-weight", "bold")
            text_elem.text = text
            return text_elem

        # Top: temperature labels
        for col_idx, temp in enumerate(temperatures):
            x_center = subplot_start_x + margin_x + col_idx * (panel_w + margin_x) + panel_w / 2
            y_pos = top_label_height - 8
            temp_str = str(int(temp)) if temp in (10.0, 20.0) else str(temp)
            create_text(x_center, y_pos, f"{temp_str} K", font_size=label_font_size)
        # Bottom: x-axis ticks (log 1, 10, 100 nΩ) and x-axis label
        x_ticks = X_AXIS_TICKS_NOHM
        x_tick_log = [np.log10(t) for t in x_ticks]
        bottom_row_subplot_bottom = subplot_start_y + margin_y + (n_rows - 1) * (panel_h + margin_y) + panel_h
        for col_idx in range(n_cols):
            subplot_x_start = subplot_start_x + margin_x + col_idx * (panel_w + margin_x)
            subplot_x_width = panel_w
            for tick_val, tick_log in zip(x_ticks, x_tick_log):
                t_norm = (tick_log - X_AXIS_MIN_LOG) / (X_AXIS_MAX_LOG - X_AXIS_MIN_LOG) if (X_AXIS_MAX_LOG - X_AXIS_MIN_LOG) != 0 else 0
                t_norm = max(0.0, min(1.0, t_norm))
                tick_x_global = subplot_x_start + t_norm * subplot_x_width
                y_tick = bottom_row_subplot_bottom + x_axis_subplot_to_tick
                create_text(tick_x_global, y_tick, str(tick_val), font_size=label_font_size, anchor="middle", baseline="top")
        x_axis_label_x = subplot_start_x + subplot_area_w / 2
        x_axis_label_y = bottom_row_subplot_bottom + x_axis_subplot_to_tick + label_font_size + x_axis_tick_to_label
        create_text(x_axis_label_x, x_axis_label_y, to_title_case("Coil-to-coil joint resistance (nΩ)"), font_size=label_font_size, anchor="middle", baseline="top")
        # Left: y-axis ticks and label
        y_ticks = Y_AXIS_TICKS
        y_min, y_max = Y_AXIS_MIN, Y_AXIS_MAX
        y_range = y_max - y_min
        left_col_subplot_left = subplot_start_x + margin_x
        for row_idx in range(n_rows):
            subplot_y_start = subplot_start_y + margin_y + row_idx * (panel_h + margin_y)
            subplot_y_height = panel_h
            for tick_val in y_ticks:
                t_norm = (tick_val - y_min) / y_range if y_range > 0 else 0.0
                t_norm = max(0.0, min(1.0, t_norm))
                tick_y_from_bottom = t_norm * subplot_y_height
                tick_y_global = subplot_y_start + subplot_y_height - tick_y_from_bottom
                tick_y_global = max(subplot_y_start, min(subplot_y_start + subplot_y_height, tick_y_global))
                x_tick = left_col_subplot_left - y_axis_subplot_to_tick
                create_text(x_tick, tick_y_global, str(tick_val), font_size=label_font_size, anchor="end", baseline="middle")
        y_axis_label_x = left_col_subplot_left - y_axis_subplot_to_tick - label_font_size * 2 - y_axis_tick_to_label
        y_axis_label_y = subplot_start_y + subplot_area_h / 2
        y_label_elem = ET.SubElement(labels_group, f"{{{svg_ns}}}text", {
            "x": str(y_axis_label_x), "y": str(y_axis_label_y),
            "font-family": "Arial, sans-serif", "font-size": str(label_font_size),
            "text-anchor": "middle", "dominant-baseline": "middle", "fill": "black",
            "transform": f"rotate(-90 {y_axis_label_x} {y_axis_label_y})"
        })
        y_label_elem.text = to_title_case("Number of parallel tapes")
        # Right: mode labels
        for row_idx, (_mode_key, mode_label) in enumerate(modes):
            x_pos = subplot_start_x + subplot_area_w + 16
            y_center = subplot_start_y + margin_y + row_idx * (panel_h + margin_y) + panel_h / 2
            create_text(x_pos, y_center, mode_label, font_size=label_font_size, anchor="start", baseline="middle")

        root_svg.remove(labels_group)
        root_svg.append(labels_group)
        tree.write(str(out_path), encoding="utf-8", xml_declaration=True)
        export_svg_to_png(out_path)
        print(f"Cryo power MWe 3×3 grid stitched ({cmap_name}): {out_path}")


def generate_readme():
    """
    Generate README documenting generation conditions, parameters, and subplot sources for all composite figures.
    """
    from datetime import datetime
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    # Output directory
    scan_base_dir = Path(cfg.ECONOMIC_FIGURES_DIR) 
    af_base_dir = Path(cfg.ECONOMIC_FIGURES_DIR) / "AF_heatmaps_scenarios" / "figures"

    # README path (in main output directory)
    readme_path = scan_base_dir / "README.md"
    
    readme_content = f"""# Economic Analysis Composite Figures

This document describes the generation conditions, parameters, and subplot sources for economic composite figures.

**Generated:** {current_date}

## Generation Script

All composite figures are produced by `9.0_stitch_delta_lcoe_min_svgs.py`.

---

## Output File List

### 1. `delta_lcoe_min_heatmap_grid_S1-S3.svg`
**Description:**
- Delta LCOE min heatmaps for scenarios S1, S2, S3 and temperature–coolant combinations
- X-axis: Coil-to-coil Joint Resistance (log scale), 1–100 nΩ
- Y-axis: Number of Parallel Tapes, 2–200

**Conditions:**
- Scenarios: S1, S2, S3
- Temperature–coolant: 4.2K He, 10K He, 20K He, 20K H₂

**Layout:**
- **Rows (top to bottom):** S1, S2, S3
- **Columns (left to right):** 4.2K He, 10K He, 20K He, 20K H₂
- **Top:** temperature–coolant label per subplot
- **Bottom:** x-axis ticks (1, 10, 100 nΩ) and label "Coil-to-coil Joint Resistance (nΩ)"
- **Left:** y-axis ticks (10, 50, 100, 200) and label "Number of Parallel Tapes" (rotated 90°)
- **Right:** scenario label per subplot (S1, S2, S3)

**Subplot source pattern:**
```
{cfg.ECONOMIC_OUTPUT_DIR}/{{colormap}}/{cfg.PARASITIC_RATIO_OUTPUT_DIR}/{cfg.PARASITIC_RATIO_USD_DIR}/{{scenario}}/delta_lcoe_min_heatmap_{{scenario}}_{{temp}}K_{{coolant}}_{{colormap}}.svg
```
Examples:
- `{cfg.ECONOMIC_OUTPUT_DIR}/viridis/{cfg.PARASITIC_RATIO_OUTPUT_DIR}/{cfg.PARASITIC_RATIO_USD_DIR}/S1/delta_lcoe_min_heatmap_S1_4.2K_He_viridis.svg`
- `{cfg.ECONOMIC_OUTPUT_DIR}/cividis/{cfg.PARASITIC_RATIO_OUTPUT_DIR}/{cfg.PARASITIC_RATIO_USD_DIR}/S1/delta_lcoe_min_heatmap_S1_10.0K_He_cividis.svg`
- `{cfg.ECONOMIC_OUTPUT_DIR}/PuBuGn/{cfg.PARASITIC_RATIO_OUTPUT_DIR}/{cfg.PARASITIC_RATIO_USD_DIR}/S2/delta_lcoe_min_heatmap_S2_20.0K_H2_PuBuGn.svg`

**Output path:**
```
{cfg.ECONOMIC_OUTPUT_DIR}/{{colormap}}/{cfg.PARASITIC_RATIO_OUTPUT_DIR}/{cfg.PARASITIC_RATIO_USD_DIR}/stitched/delta_lcoe_min_heatmap_grid_S1-S3_{{colormap}}.svg
```

---

### 2. `delta_lcoe_min_heatmap_grid_HTS_tape_sensitivity.svg`
**Description:**
- Delta LCOE min heatmaps for HTS tape price sensitivity (100, 50, 10 $/kA-m)
- X-axis: Coil-to-coil Joint Resistance (log), 1–100 nΩ; Y-axis: Number of Parallel Tapes, 2–200

**Conditions:**
- Scenarios: S5, S2, S6 (HTS tape price: 100, 50, 10 $/kA-m)
- Temperature–coolant: 4.2K He, 10K He, 20K He, 20K H₂

**Layout:**
- **Rows:** S5 (100 $/kA-m), S2 (50 $/kA-m), S6 (10 $/kA-m)
- **Columns:** 4.2K He, 10K He, 20K He, 20K H₂
- **Top/Bottom/Left/Right:** same as above

**Subplot source pattern:** same as §1 with scenario S5/S2/S6.

**Output path:**
```
{cfg.ECONOMIC_OUTPUT_DIR}/{{colormap}}/{cfg.PARASITIC_RATIO_OUTPUT_DIR}/{cfg.PARASITIC_RATIO_USD_DIR}/stitched/delta_lcoe_min_heatmap_grid_HTS_tape_sensitivity_{{colormap}}.svg
```

---

### 3. `parasitic_ratio_heatmap_grid_S1-S3.svg`
**Description:**
- Parasitic ratio heatmaps for S1, S2, S3 and temperatures; coolant He for all.
- X-axis: Coil-to-coil Joint Resistance (log), 1–100 nΩ; Y-axis: Number of Parallel Tapes, 2–200

**Conditions:**
- Scenarios: S1, S2, S3; Temperatures: 4.2K, 10K, 20K (He)

**Layout:**
- **Rows:** S1, S2, S3; **Columns:** 4.2K, 10K, 20K
- **Top:** temperature per subplot; **Bottom:** x-axis ticks and label; **Left:** y-axis; **Right:** scenario

**Subplot source pattern:**
```
{cfg.ECONOMIC_OUTPUT_DIR}/{{colormap}}/{cfg.PARASITIC_RATIO_OUTPUT_DIR}/{cfg.PARASITIC_RATIO_USD_DIR}/{{scenario}}/parasitic_ratio_heatmap_{{scenario}}_{{temp}}K_{{colormap}}.svg
```

**Output path:**
```
{cfg.ECONOMIC_OUTPUT_DIR}/{{colormap}}/{cfg.PARASITIC_RATIO_OUTPUT_DIR}/{cfg.PARASITIC_RATIO_USD_DIR}/stitched/parasitic_ratio_heatmap_grid_S1-S3_{{colormap}}.svg
```

---

### 4. `AF_heatmap_grid_S1-S3.svg`
**Description:**
- AF (Availability Factor) heatmaps for S1, S2, S3 and temperatures.
- X-axis: Turn-to-turn Resistivity (log), 10–1000 μΩ·cm²; Y-axis: Number of Parallel Tapes, 1–20

**Conditions:** Scenarios S1, S2, S3; Temperatures 4.2K, 10K, 20K

**Layout:** Rows S1–S3, columns 4.2K / 10K / 20K; top/bottom/left/right labels as above.

**Subplot source pattern:**
```
{cfg.ECONOMIC_OUTPUT_DIR}/{{colormap}}/AF_heatmaps_scenarios/figures/{{scenario}}/AF_heatmap_{{scenario}}_Top_{{temp}}K.svg
```

**Output path:**
```
{cfg.ECONOMIC_OUTPUT_DIR}/{{colormap}}/AF_heatmaps_scenarios/figures/AF_heatmap_grid_S1-S3.svg
```

---

## Scenario Summary

| Scenario | HTS tape price | Note |
|----------|----------------|------|
| S1      | 50 $/kA-m      | Baseline |
| S2      | 50 $/kA-m      | Baseline |
| S3      | 50 $/kA-m      | Baseline |
| S5      | 100 $/kA-m     | High price sensitivity |
| S6      | 10 $/kA-m      | Low price sensitivity |

---

## Subplot Generation

Subplots are produced by the economic analysis script (e.g. `8.0_run_economic_analysis_unified.py`).

**Subplot types:**
1. **delta_lcoe_min_heatmap**: Delta LCOE min heatmap
2. **parasitic_ratio_heatmap**: Parasitic ratio heatmap
3. **AF_heatmap**: Availability Factor heatmap

**Naming:** `delta_lcoe_min_heatmap_{{scenario}}_{{temp}}K_{{coolant}}.svg`, etc.

---

## Path Structure

Outputs are organized by colormap; each colormap has its own folder.

```
{cfg.ECONOMIC_OUTPUT_DIR}/
├── {{colormap}}/   # e.g. viridis, cividis, PuBuGn
│   ├── {cfg.PARASITIC_RATIO_OUTPUT_DIR}/
│   │   └── {cfg.PARASITIC_RATIO_USD_DIR}/
│   │       ├── stitched/
│   │       │   ├── delta_lcoe_min_heatmap_grid_S1-S3_{{colormap}}.svg
│   │       │   ├── delta_lcoe_min_heatmap_grid_HTS_tape_sensitivity_{{colormap}}.svg
│   │       │   └── parasitic_ratio_heatmap_grid_S1-S3_{{colormap}}.svg
│   │       └── {{scenario}}/
│   │           ├── delta_lcoe_min_heatmap_{{scenario}}_{{temp}}K_{{coolant}}_{{colormap}}.svg
│   │           └── parasitic_ratio_heatmap_{{scenario}}_{{temp}}K_{{colormap}}.svg
│   ├── AF_heatmaps_scenarios/
│   │   └── figures/
│   │       ├── AF_heatmap_grid_S1-S3.svg
│   │       └── {{scenario}}/
│   │           └── AF_heatmap_{{scenario}}_Top_{{temp}}K.svg
│   └── AF_ref_heatmaps_scenarios/
│       └── figures/
│           ├── AF_ref_heatmap_grid_S1-S3.svg
│           └── {{scenario}}/
│               └── AF_ref_heatmap_{{scenario}}_Top_{{temp}}K.svg
```

---

## Notes

1. Paths are relative to the project root.
2. Subplot files must exist; missing files produce console warnings.
3. Composite figures add axis labels, tick labels, and scenario labels automatically.
4. This README is auto-generated by `9.0_stitch_delta_lcoe_min_svgs.py` and updated on each run.
"""
    
    
    # Write README file
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"\n[OK] README written: {readme_path}")


if __name__ == "__main__":
    stitch_delta_lcoe_min_svgs()
    stitch_hts_tape_sensitivity_svgs()
    stitch_parasitic_ratio_svgs()
    stitch_cryo_power_MWe_svgs()
    stitch_af_heatmap_svgs()
    stitch_af_ref_heatmap_svgs()
    
    # Generate README
    print("\nGenerating README...")
    generate_readme()

