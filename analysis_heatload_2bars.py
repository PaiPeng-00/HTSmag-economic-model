# magnet_heat_load_analyzer.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import OrderedDict
import sys
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties
import matplotlib.colors as mcolors

# Import configuration and cryogenic model
try:
    import config as cfg
    import model_cryogenic as hlm
except ImportError as e:
    print(f"[ERROR] Failed to import required modules: {e}")
    print("Please ensure that config.py and model_cryogenic.py are in the same directory as this script.")
    sys.exit(1)

# =============================================================================
# Global font configuration for plotting
# =============================================================================
FONT_CONFIG = {
    'base_font_size': 20,           # base font size
    'axes_label_size': 20,          # axis label size
    'tick_label_size': 20,          # tick label size
    'legend_font_size': 20,         # legend font size (in main plots)
    'legend_standalone_size': 24,   # standalone legend size (for separate legend figures)
    'percent_label_size': 18,       # percentage label size
    'total_label_size': 20,         # total heat‑load label size
    'config_name_size': 20,         # configuration name label size
}
# For time‑series heat‑load plots we use slightly larger fonts
FONT_CONFIG_TIME_SERIES = {
    'base_font_size': 24,           # base font size
    'axes_label_size': 24,          # axis label size
    'tick_label_size': 24,          # tick label size
    'legend_font_size': 24,         # legend font size (in main plots)
    'legend_standalone_size': 28,   # standalone legend size (separate legend figure)
    'percent_label_size': 22,       # percentage label size
    'total_label_size': 24,         # total heat‑load label size
    'config_name_size': 24,         # configuration name label size
}

# Apply global matplotlib settings
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': FONT_CONFIG['base_font_size'],
    'axes.labelsize': FONT_CONFIG['axes_label_size'],
    'xtick.labelsize': FONT_CONFIG['tick_label_size'],
    'ytick.labelsize': FONT_CONFIG['tick_label_size'],
    'legend.fontsize': FONT_CONFIG['legend_font_size'],
    'axes.linewidth': 3,
})
# =============================================================================
# 1. Heat‑load calculation orchestrator (HeatLoadCalculator)
# =============================================================================
class HeatLoadCalculator:
    """
    Orchestrate heat‑load calculations for different operating modes and
    magnet configurations by calling functions from the cryogenic model.
    """
    def _preprocess_comsol_data(self, T_op, Npw, rhot_uOhm_cm2, loss_file_name):
        """
        Read COMSOL data from a single Excel file that contains multiple
        sheets, one per temperature (e.g. "4.2K", "10K", "20K").

        Path: cfg.HEAT_DATA_DIR / loss_file_name

        Note:
            This method uses the unified multi‑sheet layout, consistent with
            `_preprocess_charging_data_by_temp`.
        """
        # New directory layout: use HEAT_DATA_DIR directly (no per‑temperature subfolders)
        file_path = cfg.HEAT_DATA_DIR / loss_file_name
        
        if not file_path.exists():
            print(f"  [!] Warning: COMSOL data file not found: {file_path}")
            return pd.Series(dtype=float)
        
        # Sheet name determined by temperature (e.g. "4.2K", "10K", "20K")
        sheet_name = f"{T_op}K"
        
        try:
            # Try to read the requested sheet; if missing, fall back to the first sheet
            try:
                df = pd.read_excel(file_path, sheet_name=sheet_name, header=0, skiprows=lambda x: x < 5)
            except (ValueError, KeyError):
                # If the requested sheet does not exist, read the first sheet
                df = pd.read_excel(file_path, sheet_name=0, header=0, skiprows=lambda x: x < 5)
                print(f"  [!] Warning: Sheet '{sheet_name}' not found in {file_path}, using first sheet")
            
            # Data format:
            # - col 0: rhot (uOhm·cm²)
            # - col 1: Npw
            # - col 2: time (s)
            # - col 3: loss (W)
            
            # Filter rows matching Npw and rhot
            rhot_col = df.columns[0]
            npw_col = df.columns[1]
            
            # Find the closest rhot value (in case it is not exactly present)
            available_rhots = df[rhot_col].unique()
            closest_rhot = available_rhots[np.argmin(np.abs(available_rhots - rhot_uOhm_cm2))]
            
            # Filter rows with matching Npw and closest rhot
            df_filtered = df[(df[rhot_col] == closest_rhot) & (df[npw_col] == Npw)].copy()
            
            if df_filtered.empty:
                print(f"  [!] Warning: no matching data found (Npw={Npw}, rhot={rhot_uOhm_cm2} uOhm·cm²) in {file_path}")
                return pd.Series(dtype=float)
            
            # Time column is the 3rd column (index 2); convert seconds to hours
            df_filtered['Time (h)'] = df_filtered.iloc[:, 2] / 3600
            # Loss column is the 4th column (index 3)
            value_col_name = df_filtered.columns[3]
            
            return df_filtered.set_index('Time (h)')[value_col_name]

        except Exception as e:
            print(f"  [!] ERROR: failed to read COMSOL file {file_path}: {e}")
            return pd.Series(dtype=float)

    def _preprocess_charging_data_by_temp(self, T_op, Npw, rhot_uOhm_cm2, loss_file_name):
        """
        Read charging‑loss data from a single Excel file with one sheet per
        temperature (e.g. "4.2K", "10K", "20K").

        Path: cfg.HEAT_DATA_DIR / loss_file_name
        Note: file names may contain spaces, e.g. "mag loss.xlsx".

        Data format:
        - col 0: rhot (uOhm·cm²)
        - col 1: Npw
        - col 2: time (s)
        - col 3: loss (W)

        We need to filter data by matching Npw and rhot.
        """
        # New directory layout: use HEAT_DATA_DIR directly (no per‑temperature subfolders)
        file_path = cfg.HEAT_DATA_DIR / loss_file_name
        
        if not file_path.exists():
            print(f"  [!] Warning: charging‑loss data file not found: {file_path}")
            return pd.Series(dtype=float)
        
        # Sheet name determined by temperature (e.g. "4.2K", "10K", "20K")
        sheet_name = f"{T_op}K"
        
        try:
            # Try to read the requested sheet; if missing, fall back to the first sheet
            try:
                df = pd.read_excel(file_path, sheet_name=sheet_name, header=0, skiprows=lambda x: x < 5)
            except (ValueError, KeyError):
                # If the requested sheet is missing, read the first sheet
                df = pd.read_excel(file_path, sheet_name=0, header=0, skiprows=lambda x: x < 5)
                print(f"  [!] Warning: Sheet '{sheet_name}' not found in {file_path}, using first sheet")
            
            # Filter rows matching Npw and rhot
            # Column 0 is rhot, column 1 is Npw
            rhot_col = df.columns[0]
            npw_col = df.columns[1]
            
            # Find the closest rhot value (in case it is not exactly present)
            available_rhots = df[rhot_col].unique()
            closest_rhot = available_rhots[np.argmin(np.abs(available_rhots - rhot_uOhm_cm2))]
            
            # Filter rows for matching Npw and closest rhot; use .copy() to avoid SettingWithCopyWarning
            df_filtered = df[(df[rhot_col] == closest_rhot) & (df[npw_col] == Npw)].copy()
            
            if df_filtered.empty:
                print(f"  [!] Warning: no matching data found (Npw={Npw}, rhot={rhot_uOhm_cm2} uOhm·cm²) in {file_path}")
                return pd.Series(dtype=float)
            
            # Time column is the 3rd column (index 2); convert seconds to hours
            df_filtered['Time (h)'] = df_filtered.iloc[:, 2] / 3600
            # Loss column is the 4th column (index 3)
            value_col_name = df_filtered.columns[3]
            
            return df_filtered.set_index('Time (h)')[value_col_name]

        except Exception as e:
            print(f"  [!] ERROR: failed to read charging‑loss file {file_path}: {e}")
            return pd.Series(dtype=float)

    def _get_power_at_time(self, df_series, t):
        """Simple linear interpolation helper to get loss at a given time."""
        if df_series is None or df_series.empty: return 0
        times = df_series.index.to_numpy()
        idx = np.searchsorted(times, t, side='left')
        if idx == 0: return df_series.iloc[0]
        if idx >= len(times): return df_series.iloc[-1]
        return df_series.iloc[idx - 1]

    def calculate_loads(self, mode: str, Npw, R_p2p_joint: float, 
                        T_op: float, Ip: float, L_tot: float, rhot: float = 0):
        """
        Main function to compute heat loads.
        """
        # --- Heat loads at operating temperature T_op ---
        top_loads = OrderedDict()
        
        is_operating = (mode in ['operation', 'charging'])
        
        # Call cryogenic/heat‑load model functions
        #print(Npw, R_p2p_joint, Ip, L_tot)
        top_loads["Coil-to-coil joint Joule heat"] = hlm.pancake_joint_heat(Npw, R_p2p_joint, Ip) if is_operating else 0
        top_loads["Su-su joint Joule heat"] = hlm.coil_internal_joint_heat(
            Npw, L_total_m=L_tot, L_single_tape=cfg.LEN_PER_SINGEL_REBCO, Ip=Ip, R_ss=cfg.R_SU_JOINT) if is_operating else 0
        top_loads["Nuclear heat"] = hlm.nuclear_heating() if mode == 'operation' else 0
        
        top_loads["Radiative heat"] = hlm.radiation_heat(cfg.A_cryostat,  cfg.eps, cfg.T_HIGH, T_op)
        
        Q_cond_77K, Q_joule_77K, Q_hts_cond = hlm.current_lead_heat(Npw=Npw, Top=T_op, Ip=Ip)
        top_loads["Conductive heat - HTS current leads"] = Q_hts_cond

        # Other static heat loads
        coolant_pipes = hlm.pipe_heat(cfg.N_cool_pipe, cfg.d_in_cool, cfg.d_out_cool, cfg.L_cool, cfg.T_HIGH, T_op)
        other_pipes = hlm.pipe_heat(cfg.N_aux_pipe, cfg.d_in_aux, cfg.d_out_aux, cfg.L_aux, cfg.T_HIGH, T_op)
        other_static = hlm.misc_heat()
        
        top_loads["Conductive heat - coolant transfer lines"] = coolant_pipes
        top_loads["Conductive heat - auxiliary leads"] = other_pipes
        top_loads["Other sources"] = other_static

        # Transient losses (charging mode only)
        if mode == 'charging':
            # For charging, read time‑dependent loss data (magnetisation and radial)
            # from "mag loss.xlsx" and "radial loss.xlsx".
            rhot_uOhm_cm2 = rhot * 1e10  # convert to uOhm·cm²
            mag_loss_file = "mag loss.xlsx"
            radial_loss_file = "radial loss.xlsx"
            mag_loss_series = self._preprocess_charging_data_by_temp(T_op, Npw, rhot_uOhm_cm2, mag_loss_file)
            radial_loss_series = self._preprocess_charging_data_by_temp(T_op, Npw, rhot_uOhm_cm2, radial_loss_file)
            
            snapshot_time = cfg.CHARGE_HOURS
            top_loads["Magnetisation loss"] = self._get_power_at_time(mag_loss_series, snapshot_time)
            top_loads["Radial loss"] = self._get_power_at_time(radial_loss_series, snapshot_time)
        else:
            top_loads["Magnetisation loss"] = 0
            top_loads["Radial loss"] = 0

        # --- Heat loads at 77 K stage ---
        loads_77k = OrderedDict()    
        loads_77k["Joule heat - resistive current leads"] = Q_joule_77K if is_operating else 0
        loads_77k["Conductive heat - resistive current leads"] = Q_cond_77K

        return pd.Series(top_loads), pd.Series(loads_77k)
    # +++ Additional method: time‑series heat‑load calculation during charging +++
    def calculate_transient_timeseries(self, Npw, R_p2p_joint: float, T_op: float, Ip: float, rhot: float, L_tot: float):
        """
        Compute time‑series heat loads over the entire charging process
        (T_op stage only).
        """
        # Define time grid (extend by 20% beyond charge time to see steady state)
        time_points = np.linspace(0, cfg.CHARGE_HOURS * 1.2, 101)
        
        # Preload time‑dependent loss data for charging
        rhot_uOhm_cm2 = rhot * 1e10  # convert to uOhm·cm²
        mag_loss_file = "mag loss.xlsx"
        radial_loss_file = "radial loss.xlsx"
        mag_loss_series = self._preprocess_charging_data_by_temp(T_op, Npw, rhot_uOhm_cm2, mag_loss_file)
        radial_loss_series = self._preprocess_charging_data_by_temp(T_op, Npw, rhot_uOhm_cm2, radial_loss_file)

        all_timesteps_data = []
        for t in time_points:
            loads = OrderedDict()
            current_ratio = min(t / cfg.CHARGE_HOURS, 1.0)
            current = Ip * current_ratio
            # Time‑varying sources
            loads["Coil-to-coil joint Joule heat"] = hlm.pancake_joint_heat(Npw, R_p2p_joint, Ip) * (current_ratio**2)
            loads["Su-su joint Joule heat"] = hlm.coil_internal_joint_heat(Npw, L_total_m=L_tot, Ip=Ip) * (current_ratio**2)


            loads["Magnetisation loss"] = self._get_power_at_time(mag_loss_series, t)
            loads["Radial loss"] = self._get_power_at_time(radial_loss_series, t)
            
            # Static sources (time‑independent)
            loads["Radiative heat"] = hlm.radiation_heat(cfg.A_cryostat,  cfg.eps, cfg.T_HIGH, T_op)
            Q_cond_77K, Q_joule_77K, Q_hts_cond = hlm.current_lead_heat(Npw=Npw, Top=T_op, Ip=Ip)
            loads["Conductive heat - HTS current leads"] = Q_hts_cond
            loads["Conductive heat - coolant transfer lines"] = hlm.pipe_heat(cfg.N_cool_pipe, cfg.d_in_cool, cfg.d_out_cool, cfg.L_cool, cfg.T_HIGH, T_op)
            loads["Conductive heat - auxiliary leads"] = hlm.pipe_heat(cfg.N_aux_pipe, cfg.d_in_aux, cfg.d_out_aux, cfg.L_aux, cfg.T_HIGH, T_op)
            loads["Other sources"] = hlm.misc_heat()

            # No nuclear heating during charging
            loads["Nuclear heat"] = 0
            
            all_timesteps_data.append(loads)
            
        df = pd.DataFrame(all_timesteps_data, index=np.round(time_points, 3))
        df.index.name = 'Time (h)'
        return df

# =============================================================================
# 2. Plotting module (HeatLoadPlotter)
# =============================================================================
class HeatLoadPlotter:
    """
    Handle all plotting tasks, especially dual stacked‑bar plots.
    """
    def __init__(self, output_dir: Path):
        self.output_path = output_dir
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.COLORS = ['#CCE092', '#8DAFDB', '#FABA90', '#FFD865',
                        '#B4C7E7', '#EDEDED', '#2F5597', '#F8B8CC', 
                        '#FBE0EA', '#D3C6F1', '#FFDAC1', '#C4EEDF']
        self.HEAT_SOURCE_COLORS = OrderedDict([
            ("Coil-to-coil joint Joule heat", self.COLORS[0]),
            ("Su-su joint Joule heat", self.COLORS[1]),
            ("Nuclear heat", self.COLORS[2]),
            ("Magnetisation loss", self.COLORS[3]),
            ("Radial loss", self.COLORS[4]),
            ("Radiative heat", self.COLORS[5]),
            ("Conductive heat - HTS current leads", self.COLORS[6]),
            ("Conductive heat - coolant transfer lines", self.COLORS[7]),
            ("Conductive heat - auxiliary leads", self.COLORS[8]),
            ("Other sources", self.COLORS[9]),
            ("Joule heat - resistive current leads", self.COLORS[10]),
            ("Conductive heat - resistive current leads", self.COLORS[11])
        ])

    def _get_text_color_for_background(self, bg_color):
        """
        Return an appropriate text color (white or black) based on background
        colour brightness.

        Args:
            bg_color: background colour (hex string, RGB tuple, etc.)

        Returns:
            'white' or 'black'
        """
        # Convert colour to RGB in [0, 1]
        try:
            rgb = mcolors.to_rgb(bg_color)
        except (ValueError, TypeError):
            # If conversion fails, fall back to black text
            return 'black'
        
        # Compute relative luminance (ITU‑R BT.709)
        # L = 0.299*R + 0.587*G + 0.114*B
        luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        
        # Use white text for dark backgrounds (luminance < 0.5), else black
        return 'white' if luminance < 0.5 else 'black'

    def plot_dual_stacked_bar(self, top_loads: pd.Series, loads_77k: pd.Series, config_name: str, T_op: float, mode: str):
        fig, ax = plt.subplots(figsize=(1.2, 6))
        bar_width, x_pos = 0.3, 0

        bottom_top = 0
        for source, value in top_loads.items():
            if value > 0:
                color = self.HEAT_SOURCE_COLORS.get(source, '#000000')
                ax.bar(x_pos - bar_width/2, value, bar_width, bottom=bottom_top,
                       label=source, color=color,  edgecolor='black')
                bottom_top += value

        bottom_77k = 0
        for source, value in loads_77k.items():
            if value > 0:
                color = self.HEAT_SOURCE_COLORS.get(source, '#000000')
                ax.bar(x_pos + bar_width/2, value, bar_width, bottom=bottom_77k,
                       label=source, color=color, edgecolor='black')
                bottom_77k += value

        total_top, total_77k = top_loads.sum(), loads_77k.sum()
        ax.text(x_pos - bar_width/2, total_top, f'{total_top:.0f} W', 
                ha='center', va='bottom', fontsize=FONT_CONFIG['total_label_size'])
        ax.text(x_pos + bar_width/2, total_77k, f'{total_77k:.0f} W', 
                ha='center', va='bottom', fontsize=FONT_CONFIG['total_label_size'])
        
        ax.set_xticks([x_pos - bar_width/2, x_pos + bar_width/2])
        ax.set_xticklabels([f'T_op = {T_op} K', '77 K Level'])
        ax.set_ylabel('Heat Load (W)')
        
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left',
                  fontsize=FONT_CONFIG['legend_font_size'])

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()

        filename = f"{config_name}_{mode}_{T_op}K.svg"
        save_path = self.output_path / filename
        plt.savefig(save_path, dpi=cfg.PLOT_DPI, bbox_inches='tight',transparent=True)
        print(f"  [✔] Figure saved: {save_path}")
        plt.close(fig)

        # +++ Additional method 1: generate a standalone legend figure +++
    def save_legend_only(self):
        """
        Generate a standalone legend image that contains all heat‑source
        entries.
        """
        handles = [
            Line2D([0], [0], color=color, lw=10, label=label)
            for label, color in self.HEAT_SOURCE_COLORS.items()
        ]
        
        # Dynamically adjust legend figure size to accommodate all entries
        # in 4 columns.
        n_items = len(handles)
        n_cols = 4
        n_rows = (n_items + n_cols - 1) // n_cols  # ceiling division
        
        # Increase figsize so that large‑font legend is clearly readable
        fig_width = 16  # wide enough for 4‑column legend
        fig_height = max(8, n_rows * 1.2)  # at least 1.2 inches per row
        
        fig_legend = plt.figure(figsize=(fig_width, fig_height))
        #ax = fig_legend.add_subplot(111)
        #ax.axis('off')
        
        # Use FontProperties to enforce the desired legend font size
        legend_font = FontProperties(size=FONT_CONFIG['legend_standalone_size'], family='Arial')
        legend = fig_legend.legend(handles=handles, loc='center', frameon=True, edgecolor='white', ncol=n_cols,
                                   prop=legend_font, handlelength=2.0, handletextpad=0.5, columnspacing=1.5)
        
        save_path = self.output_path / cfg.LEGEND_IMAGE_FILE
        # First save; then post‑process SVG to crop extra whitespace
        fig_legend.savefig(save_path, dpi=cfg.PLOT_DPI, bbox_inches='tight', pad_inches=0.1, transparent=True)
        plt.close(fig_legend)
        
        # Crop whitespace from the SVG legend
        try:
            import xml.etree.ElementTree as ET
            import re
            
            def _parse_size(size_str: str) -> float:
                """Parse an SVG size string into a numeric value."""
                if size_str is None:
                    return 0.0
                m = re.findall(r"[0-9.]+", str(size_str))
                return float(m[0]) if m else 0.0
            
            # Read the saved SVG file
            tree = ET.parse(str(save_path))
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
            
            # Get current size and viewBox
            current_width = _parse_size(root_svg.get('width', '0'))
            current_height = _parse_size(root_svg.get('height', '0'))
            viewbox = root_svg.get('viewBox', f"0 0 {current_width} {current_height}")
            viewbox_parts = [float(x) for x in viewbox.split()]
            vb_x, vb_y, vb_w, vb_h = viewbox_parts if len(viewbox_parts) == 4 else (0, 0, current_width, current_height)
            
            # Use legend bbox information if possible. Matplotlib with
            # bbox_inches='tight' already gives a fairly tight viewBox, but we
            # further refine it by analysing all elements.
            
            # Traverse all text and shape elements to infer bounds
            min_x = float('inf')
            min_y = float('inf')
            max_x = float('-inf')
            max_y = float('-inf')
            
            def find_bounds(elem, ns, offset_x=0, offset_y=0):
                """Recursively determine element bounds."""
                nonlocal min_x, min_y, max_x, max_y
                
                # Handle transforms
                tx, ty = offset_x, offset_y
                if 'transform' in elem.attrib:
                    transform_str = elem.attrib['transform']
                    # Extract translate
                    translate_match = re.search(r'translate\(([^)]+)\)', transform_str)
                    if translate_match:
                        coords = [float(x.strip()) for x in translate_match.group(1).replace(',', ' ').split()]
                        if len(coords) >= 2:
                            tx += coords[0]
                            ty += coords[1]
                
                # Check element position
                x, y = None, None
                if 'x' in elem.attrib:
                    try:
                        x = float(elem.attrib['x']) + tx
                    except (ValueError, TypeError):
                        pass
                if 'y' in elem.attrib:
                    try:
                        y = float(elem.attrib['y']) + ty
                    except (ValueError, TypeError):
                        pass
                
                # For text elements
                if elem.tag.endswith('text') and elem.text and elem.text.strip():
                    if x is not None and y is not None:
                        font_size = float(elem.attrib.get('font-size', 12))
                        text = elem.text.strip()
                        text_width = len(text) * font_size * 0.6
                        text_height = font_size * 1.2
                        min_x = min(min_x, x)
                        min_y = min(min_y, y - text_height)
                        max_x = max(max_x, x + text_width)
                        max_y = max(max_y, y)
                
                # For rectangles, paths and other shapes
                if 'width' in elem.attrib and 'height' in elem.attrib:
                    try:
                        w = float(elem.attrib['width'])
                        h = float(elem.attrib['height'])
                        if x is not None and y is not None:
                            min_x = min(min_x, x)
                            min_y = min(min_y, y)
                            max_x = max(max_x, x + w)
                            max_y = max(max_y, y + h)
                    except (ValueError, TypeError):
                        pass
                
                # Recurse into children
                for child in elem:
                    find_bounds(child, ns, tx, ty)
            
            # Walk all elements to get global bounds
            for elem in root_svg:
                find_bounds(elem, svg_ns)
            
            # If valid bounds were found, crop to them
            if min_x != float('inf') and min_y != float('inf'):
                # Add a small padding margin
                padding = 20
                min_x = max(0, min_x - padding)
                min_y = max(0, min_y - padding)
                max_x = min(vb_w, max_x + padding)
                max_y = min(vb_h, max_y + padding)
                
                # Compute cropped size
                cropped_w = max_x - min_x
                cropped_h = max_y - min_y
                
                # Update viewBox and size
                root_svg.set('viewBox', f"{min_x} {min_y} {cropped_w} {cropped_h}")
                root_svg.set('width', f"{cropped_w}pt")
                root_svg.set('height', f"{cropped_h}pt")
                
                # Save cropped SVG
                tree.write(str(save_path), encoding='utf-8', xml_declaration=True)
                # Debug print (kept commented): original vs. cropped size
            else:
                # If bounds could not be inferred, still shrink the viewBox a bit
                padding = 10
                root_svg.set('viewBox', f"{padding} {padding} {vb_w - 2*padding} {vb_h - 2*padding}")
                root_svg.set('width', f"{vb_w - 2*padding}pt")
                root_svg.set('height', f"{vb_h - 2*padding}pt")
                tree.write(str(save_path), encoding='utf-8', xml_declaration=True)
                #print(f"\n[✔] Standalone legend saved to: {save_path} (cropped whitespace)")
        except Exception as e:
            #print(f"\n[✔] Standalone legend saved to: {save_path}")
            print(f"  [Warning] Error while cropping legend whitespace: {e}")

    # +++ Additional method 2: relative stacked‑bar plots +++
    def plot_dual_normalized_stacked_bar(self, top_loads: pd.Series, loads_77k: pd.Series, config_name: str, T_op: float, mode: str):
        
        output_path = self.output_path
        output_path.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(2.5, 8))
        bar_width, x_pos = 0.35, 0
        
        total_top = top_loads.sum()
        if total_top > 0:
            top_loads_pct = (top_loads / total_top) * 100
            bottom_top = 0
            for source, value_pct in top_loads_pct.items():
                if value_pct > 0:
                    color = self.HEAT_SOURCE_COLORS.get(source, '#000000')
                    ax.bar(x_pos - bar_width/2, value_pct, bar_width, bottom=bottom_top,
                           color=color, edgecolor='black')
                    if value_pct > 5:  # only label segments larger than 5%
                        # Choose text colour automatically based on background
                        text_color = self._get_text_color_for_background(color)
                        ax.text(x_pos - bar_width/2, bottom_top + value_pct/2, f'{value_pct:.1f}%', 
                                ha='center', va='center', fontsize=FONT_CONFIG['percent_label_size'], color=text_color)
                    bottom_top += value_pct
            ax.text(x_pos - bar_width*0.7, 101, f'{total_top:.0f} W', 
                    ha='center', va='bottom', fontsize=FONT_CONFIG['total_label_size'])
        
        total_77k = loads_77k.sum()
        if total_77k > 0:
            loads_77k_pct = (loads_77k / total_77k) * 100
            bottom_77k = 0
            for source, value_pct in loads_77k_pct.items():
                if value_pct > 0:
                    color = self.HEAT_SOURCE_COLORS.get(source, '#000000')
                    ax.bar(x_pos + bar_width/2, value_pct, bar_width, bottom=bottom_77k,
                           color=color, edgecolor='black',)
                    if value_pct > 5:
                        # Choose text colour automatically based on background
                        text_color = self._get_text_color_for_background(color)
                        ax.text(x_pos + bar_width/2, bottom_77k + value_pct/2, f'{value_pct:.1f}%', 
                                ha='center', va='center', fontsize=FONT_CONFIG['percent_label_size'], color=text_color)
                    bottom_77k += value_pct
            ax.text(x_pos + bar_width*0.7, 101, f'{total_77k:.0f} W', 
                    ha='center', va='bottom', fontsize=FONT_CONFIG['total_label_size'])

        ax.set_ylim(0, 115)
        ax.axis('off')
        #ax.set_title(f'Heat Load for {config_name} ({mode.capitalize()} Mode) - Relative')
        
        # Manually add label at the bottom
        #ax.text(x_pos - bar_width/2, -5, f'T_op = {T_op} K', ha='center', va='top', fontsize=FONT_CONFIG['config_name_size'])
        ax.text(x_pos , -5, config_name, ha='center', va='top', fontsize=FONT_CONFIG['config_name_size'])

        plt.tight_layout()

        filename = f"{config_name}_{mode}_{T_op}K_relative.svg"
        save_path = output_path / filename
        plt.savefig(save_path, dpi=cfg.PLOT_DPI, bbox_inches='tight',transparent=True)
        #print(f"  [✔] Relative‑value figure saved: {save_path}")
        plt.close(fig)

        # +++ Additional method 3: stacked‑area plots over time +++
    def plot_stacked_area_over_time(self, df: pd.DataFrame, config_name: str,  T_op: float):
        """
        Plot stacked‑area time‑series of heat load at T_op during charging.
        """
        if df.empty:
            print(f"    [!] Warning: no data available for time‑series plot; skipping.")
            return
        
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # Prepare data and colours
        x = df.index
        # Note: transpose DataFrame so that each row is a heat‑source time‑series
        y = df.T.values 
        labels = df.columns
        colors = [self.HEAT_SOURCE_COLORS.get(label, '#000000') for label in labels]

        # Draw stacked‑area plot
        ax.stackplot(x, y, labels=labels, colors=colors, alpha=0.8)

        # Draw total‑heat curve
        total_heat = df.sum(axis=1)
        ax.plot(x, total_heat, color='black', linestyle='--', linewidth=2, label='Total Heat Load')

        # Mark the end of current ramp
        ax.axvline(x=cfg.CHARGE_HOURS, color='gray', linestyle='-.', linewidth=1.5, label='End of Current Ramp')

        ax.set_xlabel('Time (h)')
        ax.set_ylabel('Heat Load (W)')
        #ax.set_title(f'T_op Heat Load vs. Time for {config_name} (Charging Mode, T_op={T_op}K)')
        ax.grid(True, which='both', linestyle=':', linewidth=0.7)
        ax.set_xlim(left=0, right=cfg.CHARGE_HOURS*1.2)
        ax.set_xlabel('Time (h)', fontsize=FONT_CONFIG_TIME_SERIES['axes_label_size'])
        ax.set_ylabel('Heat Load (W)', fontsize=FONT_CONFIG_TIME_SERIES['total_label_size'])
        ax.set_ylim(bottom=0)

        # Get legend entries and reverse order to match stacking order
        handles, labels = ax.get_legend_handles_labels()
        #ax.legend(handles[::-1], labels[::-1], loc='upper left', bbox_to_anchor=(1.02, 1.0))

        plt.tight_layout()  # leave room for legend if enabled
        
        filename = f"{config_name}_charging_{T_op}K_timeseries.svg"
        save_path = self.output_path / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        #print(f"  [✔] Time‑series figure saved: {save_path}")
        plt.close(fig)



class ExcelReporter:
    """
    Collect all computed results and write them into a single Excel file.
    """
    def __init__(self, filename: Path):
        self.filename = filename
        self.records = []  # buffer for per‑case records

    def add_record(self, config_name: str, mode: str, T_op: float, params: dict, 
                     top_loads: pd.Series, loads_77k: pd.Series):
        """
        Append one result record.

        Args:
            config_name (str): configuration name
            mode (str): operating mode
            T_op (float): operating temperature
            params (dict): input‑parameter dict
            top_loads (pd.Series): heat loads at T_op
            loads_77k (pd.Series): heat loads at 77 K
        """
        # Basic input information
        record = OrderedDict()
        record['Config Name'] = config_name
        record['Mode'] = mode
        record['T_op (K)'] = T_op
        record.update(params)  # include Npw, R_p2p_joint, etc.

        total_top = top_loads.sum()
        total_77k = loads_77k.sum()

        # Output heat‑load results; convert Series to dict with suffixes
        for source, value in top_loads.items():
            record[f'{source} (W)'] = value
            pct_value = (value / total_top * 100) if total_top else 0
            record[f'{source} (%)'] = f"{pct_value:.2f}%"
        for source, value in loads_77k.items():
            record[f'{source} (W)'] = value
            pct_value = (value / total_77k * 100) if total_77k else 0
            record[f'{source} (%)'] = f"{pct_value:.2f}%"

        # Add totals
        record['Total T_op Load (W)'] = total_top
        record['Total 77K Load (W)'] = total_77k

        # For Top ≈ 20 K, these correspond to required coolant flow rates
        record['H2 flow rate (g/s)'] = top_loads.sum() / cfg.ENTHALPY_H2_KJ_KG
        record['He flow rate (g/s)'] = top_loads.sum() / cfg.ENTHALPY_HE_KJ_KG
        
        self.records.append(record)

    def save(self):
        """
        Save all collected records into an Excel file.
        """
        if not self.records:
            print("[Info] No records to save to Excel.")
            return

        # Convert records list to DataFrame
        df = pd.DataFrame(self.records)
        df.fillna(0, inplace=True)  # fill missing values with 0 (e.g. absent sources)
        
        try:
            df.to_excel(self.filename, index=False, engine='openpyxl')
            print(f"\n[✔] Summary report saved to: {self.filename}")
        except Exception as e:
            print(f"\n[ERROR] Failed to save Excel file: {e}")
            print("       Please ensure that 'openpyxl' is installed (pip install openpyxl)")

# =============================================================================
# 3. Main execution block
# =============================================================================
if __name__ == "__main__":
    
    # -- Define magnet configurations --
    configs = {
        'A':  {'npw': 5, 'rhot': 5000e-10}, 
        'B':  {'npw': 10,  'rhot': 5000e-10},
        'C':  {'npw': 20, 'rhot': 5000e-10},
        'D':  {'npw': 200, 'rhot': 5000e-10},


        'A1': {'npw': 5, 'rhot': 5000e-10},
        'B1': {'npw': 10, 'rhot': 5000e-10}, 'B2': {'npw': 10, 'rhot': 1500e-10},
        
        'C1': {'npw': 20, 'rhot': 5000e-10}, 'C2': {'npw': 20, 'rhot': 1500e-10},
        'C3': {'npw': 20, 'rhot': 100e-10},  'C4': {'npw': 20, 'rhot': 50e-10},

        'D1': {'npw': 200, 'rhot': 5000e-10}, 'D2': {'npw': 200, 'rhot': 1500e-10},
        'D3': {'npw': 200, 'rhot': 100e-10},  'D4': {'npw': 200, 'rhot': 50e-10},   

        #'E1': {'npw': 100, 'rhot': 5000e-10}, 'E2': {'npw': 100, 'rhot': 1500e-10},
        #'E3': {'npw': 100, 'rhot': 100e-10},  'E4': {'npw': 100, 'rhot': 50e-10},   

    }

    plot_configs = {
        'B2': {'npw': 20, 'rhot': 1500e-10},
    }

    R_p2p_joint_list = [1e-9, 10e-9, 100e-9]
    # Create a single global reporter once at the beginning
    global_reporter = ExcelReporter(Path.cwd() / cfg.HEAT_LOAD_OUTPUT_DIR / "heatload_2bars_results.xlsx")
    # -- Loop over R_joint list (outer loop) --
    for R_p2p_joint in R_p2p_joint_list:
        print(f"\n{'='*60}")
        print(f"Processing R_joint = {R_p2p_joint*1e9} nOhm")
        print(f"{'='*60}")
        # -- Loop over magnet configurations --
        for name, config_params in configs.items():
            Npw = config_params['npw']
            rhot = config_params['rhot']
            params = {
                'Npw': Npw, 'R_p2p_joint': R_p2p_joint, 'rhot': rhot, 
            }
            print(f"\n--- Processing configuration: {name} ---")
            
            # static and operation modes
            for mode in ['static', 'operation']:
                for T_op in [4.2, 10.0, 20.0]:
                    output_dir = Path(cfg.OUTPUTS_FIGURES_DIR) / "heatload" / f"Top={T_op}K_Rj={R_p2p_joint*1e9}nOhm" / f"config={name}"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    calculator = HeatLoadCalculator()
                    plotter = HeatLoadPlotter(output_dir)
                    
                    # -- At the beginning, generate and save a legend once --
                    plotter.save_legend_only()
                    params['Ip'] = cfg.Ip_list[T_op]         # operating current per temperature
                    params['L_tot'] = cfg.L_HTS_TF_m[T_op]   # HTS tape length per TF depends on Top
                    print(f"  Calculating for: {mode} mode at {T_op}K...")
                    # 1. Compute loads
                    top_loads, loads_77k = calculator.calculate_loads(mode=mode, T_op=T_op, **params)
                    # 2. Plot; here we use the normalised stacked‑bar variant
                    # plotter.plot_dual_stacked_bar(top_loads, loads_77k, name, T_op, mode)
                    plotter.plot_dual_normalized_stacked_bar(top_loads, loads_77k, name, T_op, mode)

                    # 3. Add record to global report
                    global_reporter.add_record(name, mode, T_op, params, top_loads, loads_77k)

            # charging mode
            mode = 'charging'
            # Charging cases: currently support 4.2 K, 10 K and 20 K
            for T_op in [4.2, 10.0, 20.0]:
                output_dir = Path(cfg.OUTPUTS_FIGURES_DIR) / "heatload" / f"Top={T_op}K_Rj={R_p2p_joint*1e9}nOhm" / f"config={name}"
                output_dir.mkdir(parents=True, exist_ok=True)
                calculator = HeatLoadCalculator()
                plotter = HeatLoadPlotter(output_dir)
                
                # -- Generate and save legend once per directory --
                plotter.save_legend_only()
                
                print(f"  Calculating for: {mode} mode at {T_op}K...")
                params['Ip'] = cfg.Ip_list[T_op] 
                params['L_tot'] = cfg.L_HTS_TF_m[T_op]
                # 1. Calculate snapshot at end of charging for bar plots and report
                top_loads, loads_77k = calculator.calculate_loads(mode=mode, T_op=T_op, **params)
                plotter.plot_dual_normalized_stacked_bar(top_loads, loads_77k, name, T_op, mode)
                global_reporter.add_record(name, mode, T_op, params, top_loads, loads_77k)
                # 2. Compute full time‑series and plot stacked‑area figure
                print(f"  Calculating time‑series for: {mode} mode at {T_op}K...")
                timeseries_df = calculator.calculate_transient_timeseries(**params, T_op=T_op)
                plotter.plot_stacked_area_over_time(timeseries_df, name, T_op=T_op)

    # --- After all loops, save the Excel report ---
    global_reporter.save()
    print(f"\n[✔] All analysis and plotting completed. Results saved under '{output_dir}'.")
