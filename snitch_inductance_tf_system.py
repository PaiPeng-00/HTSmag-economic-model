# utility_plot_inductance.py
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import re
# Import global config so we know which files to process
import config as cfg
# Import plotting helper from our library
from utils import plot_inductance_heatmap
import xml.etree.ElementTree as ET

try:
    import svgutils.transform as sg
except ImportError as e:
    raise ImportError(
        "The package `svgutils` is required to stitch SVG figures. "
        "Please install it first:\n\n"
        "    pip install svgutils\n"
    ) from e
# =============================================================================
# Global plotting configuration
# =============================================================================
# 1) Set a base style
plt.style.use('default')
plt.rcParams['font.family'] = 'Arial' 
plt.rcParams['axes.labelsize'] =20
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 20

# Axis tick directions pointing inward
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['savefig.transparent'] = True
# =============================================================================
NPW_LIST = [1, 20, 100]
# Temperature list: taken from config if available, otherwise default to [4.2, 10.0, 20.0]
TEMPERATURES = list(cfg.Ip_list.keys()) if hasattr(cfg, 'Ip_list') else [4.2, 10.0, 20.0]

def main():
    """
    Main entry point:
    read the pre‑computed inductance matrix Excel file and regenerate
    the corresponding heatmaps for different temperatures and Npw values.
    """
    print("--- Start inductance‑matrix visualization script ---")
    
    script_dir = Path(__file__).parent.resolve()
    # Define input directory where the inductance matrix Excel file lives
    input_dir = script_dir / cfg.INDUCTANCE_OUTPUT_DIR
    out_dir = Path(cfg.OUTPUTS_FIGURES_DIR) / 'inductance'
    # Create output directory if it does not exist
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
    if not input_dir.exists():
        print(f"Error: input directory '{input_dir}' not found.")
        print("Please run 'analysis_calculate_inductance.py' first to generate the inductance matrix.")
        return

    # Iterate over the temperature and Npw lists from global config
    for temp in TEMPERATURES:
        # Number of tapes per coil at this temperature (for scaling)
        ntape_coil = cfg.Nt_list.get(temp, cfg.N_TOTAL_TAPE) if hasattr(cfg, 'Nt_list') else cfg.N_TOTAL_TAPE
        
        for npw in NPW_LIST:
            # 1) For the current Npw, get the dynamic grouping number ng
            ng_current = cfg.get_ng_for_npw(npw)
            
            # Format temperature string (4.2 stays 4.2, 10.0/20.0 become integers)
            if temp == 10.0 or temp == 20.0:
                temp_str = str(int(temp))
            else:
                temp_str = str(temp)
            
            print(f"\nProcessing Top = {temp_str} K, Npw = {npw} (ng = {ng_current}) ...")

            # 2) Build input and output paths
            excel_path = input_dir / "TF_system_L_matrix.xlsx"
            heatmap_path = out_dir / f"Npw={npw}_Top={temp_str}K_heatmap.svg"

            # 3) Check that the Excel file exists
            if not excel_path.exists():
                print(f"  -> Warning: Excel file '{excel_path.name}' not found, skip.")
                continue

            # 4) Read data and plot
            try:
                print(f"  -> Reading: {excel_path.name}")
                # Read Excel into a NumPy matrix
                m_pancakes = pd.read_excel(excel_path, header=None).values

                # Scale matrix according to Npw and tape count (temperature‑dependent)
                m_pancakes = m_pancakes * (ntape_coil * cfg.NP / npw) ** 2
                print(f"  -> First row of matrix: {m_pancakes[0]}")
                print(f"  -> Generating figure: {heatmap_path.name}")
                plot_inductance_heatmap(
                    m_pancakes,
                    title=f'Inductance Matrix (Np={cfg.NP}, Npw={npw}, Top={temp_str}K)',
                    save_path=str(heatmap_path)
                )
                print("  -> Done.")

            except Exception as e:
                print(f"  -> Failed to process: {e}")

    print("\n--- All inductance‑matrix visualizations finished ---")


def _parse_size(size_str: str) -> float:
    """
    Parse an SVG width/height string (e.g. '432pt', '800px') into a float
    by stripping units and keeping only the numeric value.
    """
    if size_str is None:
        return 0.0
    m = re.findall(r"[0-9.]+", str(size_str))
    return float(m[0]) if m else 0.0


def stitch_inductance_heatmaps():
    """
    Stitch inductance‑matrix heatmaps into a single SVG:
    - Layout: 2 columns × 3 rows (vertical stacking)
    - Left column: Npw = 1 at 4.2K, 10K, 20K (labels a, b, c)
    - Right column: Npw = 20 at 4.2K, 10K, 20K (labels d, e, f)
    """
    print("\n--- Start stitching inductance‑matrix heatmaps ---")
    
    script_dir = Path(__file__).parent.resolve()
    inductance_dir = Path(cfg.OUTPUTS_FIGURES_DIR) / 'inductance'
    
    # Define panels to stitch
    # Left column: Npw = 1
    left_panels = [
        (1, 4.2, 'a'),  # (Npw, temp, label)
        (1, 10.0, 'b'),
        (1, 20.0, 'c'),
    ]
    # Right column: Npw = 20
    right_panels = [
        (20, 4.2, 'd'),
        (20, 10.0, 'e'),
        (20, 20.0, 'f'),
    ]
    
    n_rows = 3
    n_cols = 2
    
    # Store (root, row_idx, col_idx, label)
    panel_entries = []
    panel_w = None
    panel_h = None
    
    # Read all sub‑figures
    all_panels = [(left_panels, 0), (right_panels, 1)]  # (panels, col_idx)
    
    for panels, col_idx in all_panels:
        for row_idx, (npw, temp, label) in enumerate(panels):
            # Format temperature string
            if temp == 10.0 or temp == 20.0:
                temp_str = str(int(temp))
            else:
                temp_str = str(temp)
            
            fname = inductance_dir / f"Npw={npw}_Top={temp_str}K_heatmap.svg"
            if not fname.exists():
                print(f"[warning] Sub‑figure not found: {fname}")
                continue
            
            fig = sg.fromfile(str(fname))
            root = fig.getroot()
            
            # Track maximum width/height among all panels
            size_w, size_h = fig.get_size()
            w = _parse_size(size_w)
            h = _parse_size(size_h)
            panel_w = w if panel_w is None else max(panel_w, w)
            panel_h = h if panel_h is None else max(panel_h, h)
            
            panel_entries.append((root, row_idx, col_idx, label))
    
    if not panel_entries or panel_w is None or panel_h is None:
        print("No inductance‑matrix SVG panels found to stitch, exiting.")
        return None
    
    # Margins and spacing
    margin_x = panel_w * 0.05
    margin_y = panel_h * 0.05
    
    # Reserve space for panel labels
    label_font_size = 24
    top_margin = 30    # top margin (for panel labels)
    left_margin = 20   # left margin
    right_margin = 20  # right margin
    bottom_margin = 20 # bottom margin
    
    # Compute total canvas size
    subplot_area_w = margin_x * (n_cols + 1) + panel_w * n_cols
    subplot_area_h = margin_y * (n_rows + 1) + panel_h * n_rows
    total_w = subplot_area_w + left_margin + right_margin
    total_h = subplot_area_h + top_margin + bottom_margin
    
    # Construct combined SVG canvas
    fig_out = sg.SVGFigure(f"{total_w}pt", f"{total_h}pt")
    print(f"total_w: {total_w}pt, total_h: {total_h}pt")
    
    # Place each panel on the grid
    placed_roots = []
    for root, row_idx, col_idx, label in panel_entries:
        x = left_margin + margin_x + col_idx * (panel_w + margin_x)
        y = top_margin + margin_y + row_idx * (panel_h + margin_y)
        root.moveto(x, y)
        placed_roots.append(root)
    
    fig_out.append(placed_roots)
    
    # Save stitched SVG
    out_path = inductance_dir / "inductance_matrix_grid_Npw1&20.svg"
    fig_out.save(str(out_path))
    
    # Post‑process SVG to add panel labels
    tree = ET.parse(str(out_path))
    root_svg = tree.getroot()
    
    # Detect SVG namespace
    svg_ns = None
    for prefix, uri in root_svg.attrib.items():
        if prefix.startswith('xmlns') and 'svg' in uri.lower():
            svg_ns = uri
            break
    if svg_ns is None:
        svg_ns = 'http://www.w3.org/2000/svg'
    
    ET.register_namespace('', svg_ns)
    
    # Set root SVG element size and viewBox
    root_svg.set('width', f"{total_w}pt")
    root_svg.set('height', f"{total_h}pt")
    root_svg.set('viewBox', f"0 0 {total_w} {total_h}")
    
    # Create label group
    labels_group = ET.SubElement(root_svg, f'{{{svg_ns}}}g', {'id': 'labels'})
    
    # Add panel labels (a, b, c, d, e, f)
    for root, row_idx, col_idx, label in panel_entries:
        x = left_margin + margin_x + col_idx * (panel_w + margin_x)
        y = top_margin + margin_y + row_idx * (panel_h + margin_y)
        
        # Label position: upper‑left corner of each panel
        label_x = x + 20
        label_y = y - 30
        
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
        text_elem.text = f"{label}"
    
    # Ensure labels group is on top
    root_svg.remove(labels_group)
    root_svg.append(labels_group)
    
    # Save updated SVG file
    tree.write(str(out_path), encoding='utf-8', xml_declaration=True)
    
    print(f"Stitching finished. Combined figure saved to: {out_path}")
    print("Layout: left column (Npw=1): a=4.2K, b=10K, c=20K; "
          "right column (Npw=20): d=4.2K, e=10K, f=20K")
    
    return out_path


if __name__ == "__main__":
    main()
    # Stitch inductance heatmaps after individual plots are generated
    stitch_inductance_heatmaps()