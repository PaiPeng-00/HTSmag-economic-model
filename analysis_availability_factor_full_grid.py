from __future__ import annotations

"""
Read Availability Factor (AF) data from scan_full_grid.xlsx (produced by scan_full_grid.py),
and output AF tables and heatmaps for three scenarios (S1: HTS prototype/LTS; S2: HTS engineering; S3: HTS mature).
Uses the same data source as 8.1_plot_from_scan_full_grid.py. Runs standalone with cfg;
outputs go to cfg.ECONOMIC_FIGURES_DIR/{cmap_name}/AF_heatmaps_scenarios.

Data: scan_full_grid.xlsx at outputs/tables/scan_full_grid.xlsx.
Columns: Top_K, scenario, coolant, rho_turn_uOhm_cm2, Npw, AF.
Data is interpolated onto a fixed grid (FIXED_RHOT x FIXED_NPW) for plotting.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import patheffects
from scipy.interpolate import RegularGridInterpolator
import config as cfg
# Optional: seaborn style (skip if seaborn not installed)
try:
    import seaborn as sns
    sns.set_context('notebook')
    sns.set_style('whitegrid')
except Exception:
    pass

# Plot parameters
plt.rcParams.update({
    'font.family': 'Arial',
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Arial',
    'mathtext.it': 'Arial:italic',
    'mathtext.bf': 'Arial:bold',
    'font.size': 20,
    'axes.titlesize': 20,
    'axes.labelsize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
    'figure.titlesize': 20,
})
font_size = cfg.HEATMAP_FONT_SIZE_TICK

# -----------------------------
# 0) Fixed plot grid (aligned with patch script)
# -----------------------------

# Npw range: 1-20
FIXED_NPW = np.arange(1,21,1)
DESIRED_NPW_TICKS = [1,  5,10, 20,]
# Tick labels
#DESIRED_NPW_TICKS = [1, 2, 10, 20]
# rho_turn range: 10-1000
FIXED_RHOT =  np.concatenate((np.arange(10, 100, 10), 
                            np.arange(100, 1000, 100), 
                            np.arange(1000, 11000, 1000))) 
FIXED_RHOT =  np.concatenate((np.arange(10, 100, 10), 
                            np.arange(100, 1100, 100)
                            )) 
# Tick labels

DESIRED_RHO_TICKS = [10,  100, 1000]

# -----------------------------
# 1) Load AF data from scan_full_grid.xlsx
# -----------------------------
def load_af_from_scan_full_grid(Top: float, scenario: str, coolant: str = 'He') -> pd.DataFrame:
    """
    Load AF data for the given temperature and scenario from scan_full_grid.
    Prefer CSV (faster); fall back to Excel if CSV is missing.
    
    Args:
        Top: Operating temperature (K)
        scenario: Scenario name (e.g. 'S1', 'S2', 'S3')
        coolant: Coolant type (default 'He')
    
    Returns:
        DataFrame with columns rho_turn_uOhm_cm2, Npw, AF, etc.
    """
    from src.tfmag.paths import ensure_base_dirs
    paths = ensure_base_dirs()
    
    # Prefer CSV (faster)
    csv_path = paths.outputs_tables / "scan_full_grid_tidy.csv"
    excel_path = paths.outputs_tables / "scan_full_grid.xlsx"
    
    af_cache_path = paths.outputs_tables / "scan_full_grid_af_cache.csv"
    
    # Use cache if it exists and is newer than source
    use_cache = False
    if af_cache_path.exists():
        if csv_path.exists():
            cache_time = af_cache_path.stat().st_mtime
            source_time = csv_path.stat().st_mtime
            if cache_time > source_time:
                use_cache = True
        elif excel_path.exists():
            cache_time = af_cache_path.stat().st_mtime
            source_time = excel_path.stat().st_mtime
            if cache_time > source_time:
                use_cache = True
    
    if use_cache:
        try:
            print(f"    Reading from cache: {af_cache_path.name}")
            df = pd.read_csv(af_cache_path)
            print(f"    Read {len(df):,} rows")
        except Exception as e:
            print(f"  [Warning] Cache read failed, will regenerate: {e}")
            use_cache = False
    
    if not use_cache:
        # Read source and build cache
        if csv_path.exists():
            try:
                print(f"    Reading CSV...")
                df_full = pd.read_csv(csv_path)
                print(f"    Read {len(df_full):,} rows")
            except Exception as e:
                print(f"  [Warning] CSV read failed: {e}")
                df_full = None
        else:
            df_full = None
        
        if df_full is None and excel_path.exists():
            try:
                print(f"    Reading Excel...")
                df_full = pd.read_excel(excel_path)
                print(f"    Read {len(df_full):,} rows")
            except Exception as e:
                print(f"  [Warning] Excel read failed: {e}")
                return pd.DataFrame()
        elif df_full is None:
            print(f"  [Warning] Data file missing: {csv_path} or {excel_path}")
            return pd.DataFrame()
        
        # Keep only required columns
        required_cols = ['Top_K', 'scenario', 'coolant', 'rho_turn_uOhm_cm2', 'Npw', 'AF']
        missing_cols = [col for col in required_cols if col not in df_full.columns]
        if missing_cols:
            print(f"  [Warning] Missing required columns: {missing_cols}")
            return pd.DataFrame()
        
        print(f"    Creating cache (required columns only)...")
        df = df_full[required_cols].copy()
        
        # Save cache
        try:
            df.to_csv(af_cache_path, index=False)
            print(f"    Cache saved: {af_cache_path.name}")
        except Exception as e:
            print(f"  [Warning] Cache save failed: {e}")
    
    # Filter by temperature, scenario, coolant (use np.isclose for floats)
    print(f"    Filtering (Top={Top}K, scenario={scenario}, coolant={coolant})...")
    mask = (
        (np.isclose(df['Top_K'], Top, rtol=1e-5, atol=0.1)) &
        (df['scenario'] == scenario) &
        (df['coolant'] == coolant) &
        (df['AF'].notna())
    )
    df_filtered = df[mask].copy()
    
    if df_filtered.empty:
        print(f"  [Warning] No matching data (Top={Top}K, scenario={scenario}, coolant={coolant})")
        return pd.DataFrame()
    
    print(f"    Found {len(df_filtered):,} matching points")
    
    # If AF <= 1, treat as fraction and convert to percentage
    af_max = df_filtered['AF'].max()
    if not df_filtered.empty and af_max <= 1.0:
        print(f"    AF in [0,1] detected; converting to percentage [0,100]")
        df_filtered = df_filtered.copy()
        df_filtered['AF'] = df_filtered['AF'] * 100.0
    
    return df_filtered[['rho_turn_uOhm_cm2', 'Npw', 'AF']].copy()


def interpolate_af_to_grid(df_af: pd.DataFrame, target_rhot: np.ndarray, target_npw: np.ndarray) -> np.ndarray:
    """
    Map AF data from scan_full_grid onto target grid (FIXED_RHOT x FIXED_NPW).
    Same as 2.9_af_time999_3&3.py: nearest-point match, no interpolation.

    Args:
        df_af: DataFrame with rho_turn_uOhm_cm2, Npw, AF
        target_rhot: target rho_turn array (μΩ·cm²)
        target_npw: target Npw array

    Returns:
        2D array of shape (len(target_rhot), len(target_npw))
    """
    if df_af.empty:
        return np.full((len(target_rhot), len(target_npw)), np.nan)
    
    # Unique values for grid (like rhot_grid, npw_grid in 2.9_af_time999_3&3.py)
    available_rhot = df_af['rho_turn_uOhm_cm2'].unique()
    available_npw = df_af['Npw'].unique()
    
    # Build 2D AF grid (like time999_2d); average when multiple values per (rhot, npw)
    rhot_grid = np.sort(available_rhot)
    npw_grid = np.sort(available_npw)
    
    # Lookup: for each (rhot, npw) store all AF values
    af_grid_dict = {}
    for _, row in df_af.iterrows():
        key = (row['rho_turn_uOhm_cm2'], row['Npw'])
        if key not in af_grid_dict:
            af_grid_dict[key] = []
        af_grid_dict[key].append(row['AF'])
    
    # Build 2D grid (like time999_2d)
    af_grid_2d = np.full((len(rhot_grid), len(npw_grid)), np.nan)
    for i, rho_u in enumerate(rhot_grid):
        for j, npw in enumerate(npw_grid):
            # Exact match
            if (rho_u, npw) in af_grid_dict:
                af_grid_2d[i, j] = np.mean(af_grid_dict[(rho_u, npw)])
    
    # For each target point, find nearest grid point (same as 2.9_af_time999_3&3.py)
    af_values = np.zeros((len(target_rhot), len(target_npw)))
    for i, rho_u in enumerate(target_rhot):
        rho_idx = int(np.abs(rhot_grid - rho_u).argmin())
        for j, npw in enumerate(target_npw):
            npw_idx = int(np.abs(npw_grid - npw).argmin())
            af_value = af_grid_2d[rho_idx, npw_idx]
            af_values[i, j] = af_value if not np.isnan(af_value) else np.nan
    
    return af_values


# -----------------------------
# 2) Scenario parameters (from config)
# -----------------------------
SCENARIOS = cfg.SCENARIO_DEFINITIONS

Top_list = [4.2, 10.0, 20.0]

# -----------------------------
# 3) Scenario AF (maintenance, cooldown/warmup, excitation/de-excitation)
# -----------------------------
def compute_AF_with_explicit_breakdown(time_999_h: float,
                                       tmaint_h: float, nmaint: float,
                                       tcool_h: float, twarm_h: float, kdis: float,
                                       tau_pulse_h: float, tau_dwell_h: float,
                                       hours_per_year: float = cfg.HOURS_PER_YEAR):
    """
    Compute AF and stage hours from annual time breakdown:
    - H_maint: total maintenance hours
    - H_coolwarm: cooldown+warmup = nmaint*(tcool + twarm)
    - H_excdec: excitation+de-excitation = nmaint*(tcharge + kdis*tcharge), tcharge = time_999_h
    - H_prod/H_dwell: production/dwell hours per cycle
    """
    tcharge_h = float(max(0.0, time_999_h))
    H_maint = nmaint * tmaint_h
    H_coolwarm = nmaint * (tcool_h + twarm_h)
    H_excdec = nmaint * (tcharge_h + kdis * tcharge_h)

    remaining = hours_per_year - H_maint - H_coolwarm - H_excdec
    if remaining <= 0:
        return 0.0, H_maint, H_coolwarm, H_excdec, 0.0, 0.0

    tcycle = tau_pulse_h + tau_dwell_h
    ncycles = np.floor(remaining / tcycle)
    H_prod = ncycles * tau_pulse_h
    H_dwell = ncycles * tau_dwell_h
    AF = H_prod / hours_per_year *100.0
    return AF, H_maint, H_coolwarm, H_excdec, H_prod, H_dwell


# -----------------------------
# 4) Generate AF tables and heatmaps
# -----------------------------
def make_af_heatmaps_for_scenarios(Top=20.0, out_root: Path | None = None, use_stroke: bool = False):
    """
    Generate side-by-side heatmaps for all scenarios with a shared colorbar.
    One folder per colormap. Data from scan_full_grid.xlsx.
    """
    coolant = 'He'

    # One heatmap set per colormap
    for cmap_name in cfg.color_schemes:
        print(f"\nAF heatmap colormap: {cmap_name}")
        if out_root is None:
            # Path: outputs/figures/economic/{cmap_name}/AF_heatmaps_scenarios
            current_out_root = Path(cfg.ECONOMIC_FIGURES_DIR) / cmap_name / 'AF_heatmaps_scenarios'
        else:
            # If out_root given, create colormap subfolder there
            current_out_root = out_root / cmap_name / 'AF_heatmaps_scenarios'
        current_out_root.mkdir(parents=True, exist_ok=True)

        # Separate data / figures directories
        data_root = current_out_root / "data"
        fig_root = current_out_root / "figures"
        data_root.mkdir(parents=True, exist_ok=True)
        fig_root.mkdir(parents=True, exist_ok=True)

        af_data_all_scenarios = {}
        global_vmin = np.inf
        global_vmax = -np.inf

        # --- Step 1: Load all scenarios from scan_full_grid.xlsx, find global min/max ---
        for key, sc in SCENARIOS.items():
            out_data_dir = data_root / key
            out_data_dir.mkdir(parents=True, exist_ok=True)

            # Load AF from scan_full_grid.xlsx
            print(f"  Loading AF for {key} (Top={Top}K)...")
            df_af = load_af_from_scan_full_grid(Top=Top, scenario=key, coolant=coolant)
            
            if df_af.empty:
                print(f"  [Warning] No data for {key}, skipping")
                AF_values = np.full((len(FIXED_RHOT), len(FIXED_NPW)), np.nan)
            else:
                # Map to target grid
                AF_values = interpolate_af_to_grid(df_af, FIXED_RHOT, FIXED_NPW)
                # Clamp to [0, 100]
                AF_values = np.clip(AF_values, 0.0, 100.0)
            
            # Store data
            af_data_all_scenarios[key] = AF_values
            
            # Update global min/max (ignore NaN)
            valid_values = AF_values[~np.isnan(AF_values)]
            if len(valid_values) > 0:
                global_vmin = min(global_vmin, float(np.nanmin(valid_values)))
                global_vmax = max(global_vmax, float(np.nanmax(valid_values)))

            # Save Excel
            df_AF = pd.DataFrame(AF_values, index=FIXED_RHOT, columns=FIXED_NPW)
            excel_path = out_data_dir / f"AF_data_{key}_full_grid.xlsx"
            df_AF.to_excel(excel_path)

        # --- Step 2: Plot (all subplots) ---
        num_scenarios = len(SCENARIOS)
        # Figsize for 3 subplots, 5x5 each (match plot_library heatmaps)
        fig, axes = plt.subplots(1, num_scenarios, figsize=cfg.HEATMAP_FIGSIZE_GRID, sharex=True, sharey=True)
        
        # Ensure
        if num_scenarios == 1:
            axes = [axes]  # Single subplot: make iterable

        # Support custom/truncated colormap names from config
        cmap = cfg.resolve_cmap(cmap_name) if hasattr(cfg, "resolve_cmap") else cmap_name

        norm = mpl.colors.PowerNorm(gamma=1.4, vmin=global_vmin, vmax=global_vmax)

        im = None  # For colorbar

        # --- Step 3: Draw heatmap and contours per scenario ---
        for ax, (key, sc) in zip(axes, SCENARIOS.items()):
            AF_values = af_data_all_scenarios[key]         # shape (len(FIXED_RHOT), len(FIXED_NPW)) before T
            Z = AF_values.T                                # Z shape (Ny, Nx), Ny=len(FIXED_NPW), Nx=len(FIXED_RHOT)
        
        # ========= 1) Center points (for contours) =========
        rho_c = FIXED_RHOT.astype(float)               # μΩ·cm²
        # Raw coordinates (center); use log scale
        x_c = np.log10(FIXED_RHOT)
        y_c = np.log10(FIXED_NPW)

        # Set axes to log scale
        '''ax.set_xscale('log')
        ax.set_yscale('log')'''

        # Create interpolator (log space)
        interp = RegularGridInterpolator((y_c, x_c), Z, bounds_error=False, fill_value=None)

        # High-res grid (corner grid so contours extend to edges)
        # Center-point grid for interpolation (linear in log space)
        x_center = np.linspace(x_c.min(), x_c.max(), 40)
        y_center = np.linspace(y_c.min(), y_c.max(), 40)
        X_center, Y_center = np.meshgrid(x_center, y_center, indexing='xy')
        Zf = interp((Y_center, X_center))
        
        # Corner-point grid (for pcolormesh/contour, contours to edges)
        # For shading="flat": X,Y corner coords (n+1,m+1), Z center values (n,m)
        dx = (x_center[-1] - x_center[0]) / (len(x_center) - 1) if len(x_center) > 1 else 0
        dy = (y_center[-1] - y_center[0]) / (len(y_center) - 1) if len(y_center) > 1 else 0
        x_edges = np.concatenate([[x_center[0] - dx/2], 
                                  (x_center[:-1] + x_center[1:]) / 2, 
                                  [x_center[-1] + dx/2]])
        y_edges = np.concatenate([[y_center[0] - dy/2], 
                                  (y_center[:-1] + y_center[1:]) / 2, 
                                  [y_center[-1] + dy/2]])
        Xf, Yf = np.meshgrid(x_edges, y_edges, indexing='xy')
        
        # Interpolate Zf (center) to corner grid for contour
        # Corner-grid interpolator
        interp_edges = RegularGridInterpolator((y_center, x_center), Zf, bounds_error=False, fill_value=None)
        Zf_edges = interp_edges((Yf, Xf))

        # Main plot: corner grid Xf,Yf and center Zf, shading="flat"
        # pcolormesh expects X,Y (n+1,m+1), Z (n,m)
        # Xf,Yf are in log space; convert to physical for set_xscale/set_yscale('log')
        '''Xf_phys = 10 ** Xf
        Yf_phys = 10 ** Yf'''
        pcm = ax.pcolormesh(Xf, Yf, Zf, cmap=cmap, norm=norm, shading="flat", alpha=cfg.HEATMAP_ALPHA)
        im = pcm
        
        # Set axis limits so contours reach the edges
        ax.set_xlim(x_edges[0], x_edges[-1])
        ax.set_ylim(y_edges[0], y_edges[-1])

        # Draw contours on smooth grid
        #CS = ax.contour(Xf, Yf, Zf, levels=CONTOUR_LEVELS_SCENARIOS[key],
        #                        colors="white", linewidths=cfg.HEATMAP_CONTOUR_LINEWIDTH_AF, fontsize=font_size)
        # Contour levels 0.1, 0.2, ... up to Vmax; if >4, sample evenly
        max_val = float(np.nanmax(Zf))
        contour_start = 10.0
        contour_step = 10.0

        # Auto-generate contour levels
        # If max_val < contour_start, use smaller step
        if max_val < contour_start:
            # Smaller step from 0
            contour_step = max(1.0, max_val / 10.0)  # At least 10 levels
            levels_raw = np.arange(0, max_val + contour_step/2, contour_step)
        else:
            # From contour_start with contour_step
            levels_raw = np.arange(contour_start, max_val + contour_step/2, contour_step)

        # Ensure max_val is included if not already
        if len(levels_raw) > 0:
            if abs(levels_raw[-1] - max_val) > 1e-3:
                levels = np.append(levels_raw, max_val)
            else:
                levels = levels_raw
        else:
            # If levels_raw empty, use max_val
            levels = np.array([max_val])

        # Add vmax*0.999
        vmax_point_999 = max_val * 0.99
        # Add vmax*0.999 if not already in levels
        insert_999 = True
        for v in levels:
            if abs(v - vmax_point_999) < 1e-5:
                insert_999 = False
                break
        if len(levels) > 0 and insert_999 and vmax_point_999 > levels[0] and vmax_point_999 < levels[-1]:
            levels = np.append(levels, vmax_point_999)
            levels = np.sort(levels)

        # If count <= 4 use as-is; else sample (excluding last max_val)
        if len(levels) <= 5:
            contour_levels = list(levels)
        else:
            # Keep last max; sample 3 main levels from the rest
            n_sample = 5
            idxs = np.round(np.linspace(0, len(levels)-2, n_sample)).astype(int)
            sampled = [levels[i] for i in idxs]
            contour_levels = sampled + [levels[-1]]
            # Add vmax_point_999 if not in contour_levels
            if not any(abs(v - vmax_point_999) < 1e-5 for v in contour_levels):
                contour_levels.append(vmax_point_999)
            # Dedupe and sort
            contour_levels = sorted(set(contour_levels))

        # Clamp to max_val (float-safe); exclude max_val and vmax_point_999 to avoid extra contour at top-right
        contour_levels = [v for v in contour_levels if v < max_val - 1e-6 and abs(v - vmax_point_999) > 1e-6]
        
        if use_stroke:
            # Stroke: path_effects white halo on black contours
            # Draw contours on corner grid so they extend to edges (Xf,Yf log -> physical)
            Xf_phys = 10 ** Xf
            Yf_phys = 10 ** Yf
            CS = ax.contour(Xf_phys, Yf_phys, Zf_edges, levels=contour_levels,
                            colors="black", linewidths=cfg.HEATMAP_CONTOUR_LINEWIDTH_AF)
            # White stroke on contour collection
            for collection in CS.collections:
                collection.set_path_effects([
                    patheffects.withStroke(
                        linewidth=cfg.CONTOUR_STROKE_LINEWIDTH_AF,
                        foreground=cfg.CONTOUR_STROKE_FOREGROUND,
                        alpha=cfg.CONTOUR_STROKE_ALPHA
                    ),
                    patheffects.Normal()
                ])
            # Stroke on contour labels
            texts = CS.clabel(inline=True, colors="black", fmt=cfg.HEATMAP_CONTOUR_FMT_AF, fontsize=font_size)
            if texts:
                for txt in texts:
                    txt.set_path_effects([
                        patheffects.withStroke(
                            linewidth=cfg.LABEL_STROKE_LINEWIDTH,
                            foreground=cfg.LABEL_STROKE_FOREGROUND,
                            alpha=cfg.LABEL_STROKE_ALPHA
                        ),
                        patheffects.Normal()
                    ])
        else:
            # Contours on corner grid to edges (Xf,Yf log -> physical)
            Xf_phys = 10 ** Xf
            Yf_phys = 10 ** Yf
            CS = ax.contour(Xf_phys, Yf_phys, Zf_edges, levels=contour_levels,
                            colors="white", linewidths=cfg.HEATMAP_CONTOUR_LINEWIDTH_AF)
            # Disable contour labels
            CS.clabel(inline=True, colors="white", fmt=cfg.HEATMAP_CONTOUR_FMT_AF, fontsize=font_size)

        # ========= 5) Remove ticks and labels =========
        ax.set_xticks([])
        ax.set_yticks([])
        
        # ========= 6) Frame/title =========
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(2)
            # No subplot title
            ax.set_title("")
             
        # --- Step 4: No shared axis labels or composite colorbar ---
        # Hide all axis labels
        for ax in axes:
            ax.set_xlabel("")
            ax.set_ylabel("")

        # --- Step 5: Save composite (disabled) ---
        # Only save per-scenario heatmaps; tighten spacing to remove white margins
        # fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        # out_svg = fig_root / f"AF_heatmap_all_scenarios_Top_{Top}K.svg"
        # fig.savefig(
        #     out_svg,
        #     transparent=getattr(cfg, "PLOT_TRANSPARENT", True),
        #     bbox_inches="tight",
        #     pad_inches=0.0,
        # )
        
        # --- Step 6: Save colorbar figure ---
        # Same cmap and norm as main plot
        cbar_fig = plt.figure(figsize=cfg.HEATMAP_FIGSIZE_COLORBAR_LARGE)
        cbar_ax2 = cbar_fig.add_axes([0.3, 0.05, 0.1, 0.9])  # Narrow colorbar
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        
        sm.set_array([])
        cbar2 = cbar_fig.colorbar(sm, cax=cbar_ax2)
        # Keep colorbar ticks and title
        cbar2.set_label('Availability Factor (%)')

        # Keep ticks and ticklabels;
        cbar2.outline.set_edgecolor('black')
        cbar2.outline.set_linewidth(2)
        # No tight_layout (would clip colorbar ticks); use subplots_adjust for margin
        cbar_fig.subplots_adjust(left=0.2, right=0.8, top=0.98, bottom=0.02)
        out_cbar_svg = fig_root / f"AF_colorbar_Top_{Top}K.svg"
        cbar_fig.savefig(
            out_cbar_svg,
            transparent=getattr(cfg, "PLOT_TRANSPARENT", True),
            bbox_inches="tight",
            pad_inches=0.0,
        )
        plt.close(cbar_fig)
        plt.close(fig)

        # --- Step 6: Save one AF heatmap per scenario (same logic as composite) ---
        for key, sc in SCENARIOS.items():
            AF_values = af_data_all_scenarios[key]
            Z = AF_values.T

            rho_c = FIXED_RHOT.astype(float)
            # Raw coordinates (center); use log scale
            x_c = np.log10(FIXED_RHOT)
            y_c = np.log10(FIXED_NPW)

            interp = RegularGridInterpolator((y_c, x_c), Z, bounds_error=False, fill_value=None)
            # High-res grid (corner grid so contours extend to edges)
            # Center-point grid for interpolation (linear in log space)
            x_center = np.linspace(x_c.min(), x_c.max(), 40)
            y_center = np.linspace(y_c.min(), y_c.max(), 40)
            X_center, Y_center = np.meshgrid(x_center, y_center, indexing='xy')
            Zf = interp((Y_center, X_center))
            
            # Corner-point grid (for pcolormesh/contour, contours to edges)
            # For shading="flat": X,Y corner coords (n+1,m+1), Z center values (n,m)
            dx = (x_center[-1] - x_center[0]) / (len(x_center) - 1) if len(x_center) > 1 else 0
            dy = (y_center[-1] - y_center[0]) / (len(y_center) - 1) if len(y_center) > 1 else 0
            x_edges = np.concatenate([[x_center[0] - dx/2], 
                                      (x_center[:-1] + x_center[1:]) / 2, 
                                      [x_center[-1] + dx/2]])
            y_edges = np.concatenate([[y_center[0] - dy/2], 
                                      (y_center[:-1] + y_center[1:]) / 2, 
                                      [y_center[-1] + dy/2]])
            Xf, Yf = np.meshgrid(x_edges, y_edges, indexing='xy')
            
            # Interpolate Zf (center) to corner grid for contour
            interp_edges = RegularGridInterpolator((y_center, x_center), Zf, bounds_error=False, fill_value=None)
            Zf_edges = interp_edges((Yf, Xf))

            fig_single, ax_single = plt.subplots(1, 1, figsize=cfg.HEATMAP_FIGSIZE_SINGLE)
            
            # Set axes to log scale
            ax_single.set_xscale('log')
            ax_single.set_yscale('log')
            
            # Corner grid Xf,Yf and center Zf, shading="flat" (Xf,Yf log -> physical for log axes)
            Xf_phys = 10 ** Xf
            Yf_phys = 10 ** Yf
            pcm_single = ax_single.pcolormesh(Xf_phys, Yf_phys, Zf, cmap=cmap, norm=norm, shading="flat", alpha=cfg.HEATMAP_ALPHA)
            
            # Set axis limits so contours reach the edges
            ax_single.set_xlim(10 ** x_edges[0], 10 ** x_edges[-1])
            ax_single.set_ylim(10 ** y_edges[0], 10 ** y_edges[-1])

            max_val = float(np.nanmax(Zf))
            contour_start = 10.0
            contour_step = 10.0
            
            # Auto-generate contour levels
            if max_val < contour_start:
                contour_step = max(1.0, max_val / 10.0)
                levels_raw = np.arange(0, max_val + contour_step/2, contour_step)
            else:
                levels_raw = np.arange(contour_start, max_val + contour_step/2, contour_step)
            
            if len(levels_raw) > 0:
                if abs(levels_raw[-1] - max_val) > 1e-3:
                    levels = np.append(levels_raw, max_val)
                else:
                    levels = levels_raw
            else:
                levels = np.array([max_val])

            vmax_point_999 = max_val * 0.99
            insert_999 = True
            for v in levels:
                if abs(v - vmax_point_999) < 1e-5:
                    insert_999 = False
                    break
            if len(levels) > 0 and insert_999 and vmax_point_999 > levels[0] and vmax_point_999 < levels[-1]:
                levels = np.append(levels, vmax_point_999)
                levels = np.sort(levels)

            if len(levels) <= 5:
                contour_levels = list(levels)
            else:
                n_sample = 5
                idxs = np.round(np.linspace(0, len(levels)-2, n_sample)).astype(int)
                sampled = [levels[i] for i in idxs]
                contour_levels = sampled + [levels[-1]]
                if not any(abs(v - vmax_point_999) < 1e-5 for v in contour_levels):
                    contour_levels.append(vmax_point_999)
                contour_levels = sorted(set(contour_levels))

            # Exclude max_val and vmax_point_999 to avoid extra contour at top-right
            contour_levels = [v for v in contour_levels if v < max_val - 1e-6 and abs(v - vmax_point_999) > 1e-6]
            if use_stroke:
                # Stroke and contours on corner grid (Xf,Yf log -> physical)
                Xf_phys = 10 ** Xf
                Yf_phys = 10 ** Yf
                CS_single = ax_single.contour(Xf_phys, Yf_phys, Zf_edges, levels=contour_levels,
                                              colors="black", linewidths=cfg.HEATMAP_CONTOUR_LINEWIDTH_AF)
            # White stroke on contour collection
            for collection in CS_single.collections:
                collection.set_path_effects([
                    patheffects.withStroke(
                        linewidth=cfg.CONTOUR_STROKE_LINEWIDTH_AF,
                        foreground=cfg.CONTOUR_STROKE_FOREGROUND,
                        alpha=cfg.CONTOUR_STROKE_ALPHA
                    ),
                    patheffects.Normal()
                ])
            texts_single = CS_single.clabel(inline=True, colors="black", fmt=cfg.HEATMAP_CONTOUR_FMT_AF)
            if texts_single:
                for txt in texts_single:
                    txt.set_path_effects([
                        patheffects.withStroke(
                            linewidth=cfg.LABEL_STROKE_LINEWIDTH,
                            foreground=cfg.LABEL_STROKE_FOREGROUND,
                            alpha=cfg.LABEL_STROKE_ALPHA
                        ),
                        patheffects.Normal()
                        ])
            else:
                # Contours on corner grid (Xf,Yf log -> physical)
                Xf_phys = 10 ** Xf
                Yf_phys = 10 ** Yf
                CS_single = ax_single.contour(Xf_phys, Yf_phys, Zf_edges, levels=contour_levels,
                                              colors="black", linewidths=cfg.HEATMAP_CONTOUR_LINEWIDTH_AF)
                CS_single.clabel(inline=True, colors="black", fmt=cfg.HEATMAP_CONTOUR_FMT_AF)
            
            # Match axis limits to corner grid
            ax_single.set_xlim(10 ** x_edges[0], 10 ** x_edges[-1])
            ax_single.set_ylim(10 ** y_edges[0], 10 ** y_edges[-1])

            ax_single.set_xticks([])
            ax_single.set_yticks([])
            for spine in ax_single.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(2)
            ax_single.set_title("")

            single_fig_dir = fig_root / key
            single_fig_dir.mkdir(parents=True, exist_ok=True)
            single_svg = single_fig_dir / f"AF_heatmap_{key}_Top_{Top}K.svg"
            # Remove white margins similarly
            fig_single.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
            fig_single.savefig(
                single_svg,
                transparent=getattr(cfg, "PLOT_TRANSPARENT", True),
                bbox_inches="tight",
                pad_inches=0.0,
            )
            plt.close(fig_single)
        
        # --- Step 7: Write README ---
        info_txt = current_out_root / "README.txt"
        with open(info_txt, "w", encoding="utf-8") as f:
            f.write(
                "This directory is auto-generated by 2.9_af_form_full_grid.py.\n"
                "Each subdir is a scenario with AF data table (xlsx).\n"
                "Data: AF field from scan_full_grid.xlsx. Same source as 8.1_plot_from_scan_full_grid.py.\n"
            )
    
        print(f"Per-scenario AF heatmap figures saved to: {fig_root}")
        print(f"Per-scenario AF data tables saved to: {current_out_root}")




def make_af_ref_heatmaps_for_scenarios(Top=20.0, out_root: Path | None = None, use_stroke: bool = False):
    """
    Relative AF: AF_ref = AF / AFmax(scenario). Plot style matches AF heatmaps.
    One folder per colormap. Data from scan_full_grid.xlsx.
    """
    coolant = 'He'

    for cmap_name in cfg.color_schemes:
        print(f"\nAF_ref heatmap colormap: {cmap_name}")
        if out_root is None:
            # Path: outputs/figures/economic/{cmap_name}/AF_ref_heatmaps_scenarios
            current_out_root = Path(cfg.ECONOMIC_FIGURES_DIR) / cmap_name / 'AF_ref_heatmaps_scenarios'
        else:
            # If out_root given, create colormap subfolder there
            current_out_root = out_root / cmap_name / 'AF_ref_heatmaps_scenarios'
        current_out_root.mkdir(parents=True, exist_ok=True)

        data_root = current_out_root / "data"
        fig_root = current_out_root / "figures"
        data_root.mkdir(parents=True, exist_ok=True)
        fig_root.mkdir(parents=True, exist_ok=True)

        af_ref_data_all_scenarios = {}
        global_vmin = np.inf
        global_vmax = -np.inf

        for key, sc in SCENARIOS.items():
            out_data_dir = data_root / key
            out_data_dir.mkdir(parents=True, exist_ok=True)

            # Load AF from scan_full_grid.xlsx
            print(f"  Loading AF for {key} (Top={Top}K)...")
            df_af = load_af_from_scan_full_grid(Top=Top, scenario=key, coolant=coolant)
            
            if df_af.empty:
                print(f"  [Warning] No data for {key}, skipping")
                AF_values = np.full((len(FIXED_RHOT), len(FIXED_NPW)), np.nan)
            else:
                # Map to target grid
                AF_values = interpolate_af_to_grid(df_af, FIXED_RHOT, FIXED_NPW)
                # Clamp to [0, 100]
                AF_values = np.clip(AF_values, 0.0, 100.0)

            # AF_ref = AF / AFmax
            valid_values = AF_values[~np.isnan(AF_values)]
            if len(valid_values) > 0:
                af_max = float(np.nanmax(valid_values))
                if af_max > 0:
                    AF_ref_values = AF_values / af_max
                else:
                    AF_ref_values = np.zeros_like(AF_values)
            else:
                AF_ref_values = np.full_like(AF_values, np.nan)

            af_ref_data_all_scenarios[key] = AF_ref_values
            
            # Update global min/max (ignore NaN)
            valid_ref_values = AF_ref_values[~np.isnan(AF_ref_values)]
            if len(valid_ref_values) > 0:
                global_vmin = min(global_vmin, float(np.nanmin(valid_ref_values)))
                global_vmax = max(global_vmax, float(np.nanmax(valid_ref_values)))

            df_AF_ref = pd.DataFrame(AF_ref_values, index=FIXED_RHOT, columns=FIXED_NPW)
            excel_path = out_data_dir / f"AF_ref_data_{key}_full_grid.xlsx"
            df_AF_ref.to_excel(excel_path)

        num_scenarios = len(SCENARIOS)
        fig, axes = plt.subplots(1, num_scenarios, figsize=cfg.HEATMAP_FIGSIZE_GRID, sharex=True, sharey=True)
        if num_scenarios == 1:
            axes = [axes]

        # Support custom/truncated colormap names from config
        cmap = cfg.resolve_cmap(cmap_name) if hasattr(cfg, "resolve_cmap") else cmap_name
        norm = mpl.colors.PowerNorm(gamma=1.4, vmin=global_vmin, vmax=global_vmax)
        im = None

        for ax, (key, sc) in zip(axes, SCENARIOS.items()):
            AF_ref_values = af_ref_data_all_scenarios[key]
            Z = AF_ref_values.T


            # Raw coordinates (center); use log scale
            x_c = np.log10(FIXED_RHOT)
            y_c = np.log10(FIXED_NPW)
            
            # Set axes to log scale
            ax.set_xscale('log')
            ax.set_yscale('log')
            
            interp = RegularGridInterpolator((y_c, x_c), Z, bounds_error=False, fill_value=None)
            # High-res grid (corner grid so contours extend to edges)
            # Center-point grid for interpolation (linear in log space)
            x_center = np.linspace(x_c.min(), x_c.max(), 40)
            y_center = np.linspace(y_c.min(), y_c.max(), 40)
            X_center, Y_center = np.meshgrid(x_center, y_center, indexing='xy')
            Zf = interp((Y_center, X_center))
            
            # Corner-point grid (for pcolormesh/contour, contours to edges)
            # For shading="flat": X,Y corner coords (n+1,m+1), Z center values (n,m)
            dx = (x_center[-1] - x_center[0]) / (len(x_center) - 1) if len(x_center) > 1 else 0
            dy = (y_center[-1] - y_center[0]) / (len(y_center) - 1) if len(y_center) > 1 else 0
            x_edges = np.concatenate([[x_center[0] - dx/2], 
                                      (x_center[:-1] + x_center[1:]) / 2, 
                                      [x_center[-1] + dx/2]])
            y_edges = np.concatenate([[y_center[0] - dy/2], 
                                      (y_center[:-1] + y_center[1:]) / 2, 
                                      [y_center[-1] + dy/2]])
            Xf, Yf = np.meshgrid(x_edges, y_edges, indexing='xy')
            
            # Interpolate Zf (center) to corner grid for contour
            interp_edges = RegularGridInterpolator((y_center, x_center), Zf, bounds_error=False, fill_value=None)
            Zf_edges = interp_edges((Yf, Xf))

            # Corner grid Xf,Yf and center Zf, shading="flat" (Xf,Yf log -> physical for log axes)
            Xf_phys = 10 ** Xf
            Yf_phys = 10 ** Yf
            pcm = ax.pcolormesh(Xf_phys, Yf_phys, Zf, cmap=cmap, norm=norm, shading="flat", alpha=cfg.HEATMAP_ALPHA)
            im = pcm
            
            # Set axis limits so contours reach the edges
            ax.set_xlim(10 ** x_edges[0], 10 ** x_edges[-1])
            ax.set_ylim(10 ** y_edges[0], 10 ** y_edges[-1])

            max_val = float(np.nanmax(Zf))
            contour_start = 0.1
            contour_step = 0.1
            
            # Auto-generate contour levels
            if max_val < contour_start:
                contour_step = max(0.01, max_val / 10.0)
                levels_raw = np.arange(0, max_val + contour_step/2, contour_step)
            else:
                levels_raw = np.arange(contour_start, max_val + contour_step/2, contour_step)
            
            if len(levels_raw) > 0:
                if abs(levels_raw[-1] - max_val) > 1e-6:
                    levels = np.append(levels_raw, max_val)
                else:
                    levels = levels_raw
            else:
                levels = np.array([max_val])

            vmax_point_999 = max_val * 0.99
            insert_999 = True
            for v in levels:
                if abs(v - vmax_point_999) < 1e-6:
                    insert_999 = False
                    break
            if len(levels) > 0 and insert_999 and vmax_point_999 > levels[0] and vmax_point_999 < levels[-1]:
                levels = np.append(levels, vmax_point_999)
                levels = np.sort(levels)
            if max_val >= 0.9 and not any(abs(v - 0.9) < 1e-6 for v in levels):
                levels = np.append(levels, 0.9)
                levels = np.sort(levels)
            if max_val >= 0.95 and not any(abs(v - 0.95) < 1e-6 for v in levels):
                levels = np.append(levels, 0.95)
                levels = np.sort(levels)

            if len(levels) <= 5:
                contour_levels = list(levels)
            else:
                n_sample = 5
                idxs = np.round(np.linspace(0, len(levels)-2, n_sample)).astype(int)
                sampled = [levels[i] for i in idxs]
                contour_levels = sampled + [levels[-1]]
                if not any(abs(v - vmax_point_999) < 1e-6 for v in contour_levels):
                    contour_levels.append(vmax_point_999)
                # Include 0.9 contour if max_val >= 0.9
                if max_val >= 0.9 and not any(abs(v - 0.9) < 1e-6 for v in contour_levels):
                    contour_levels.append(0.9)
                # Include 0.95 contour if max_val >= 0.95
                if max_val >= 0.95 and not any(abs(v - 0.95) < 1e-6 for v in contour_levels):
                    contour_levels.append(0.95)
                contour_levels = sorted(set(contour_levels))

            # Exclude max_val and vmax_point_999 to avoid extra contour at top-right
            contour_levels = [v for v in contour_levels if v < max_val - 1e-6 and abs(v - vmax_point_999) > 1e-6]
            if use_stroke:
                # Stroke and contours on corner grid (Xf,Yf log -> physical)
                Xf_phys = 10 ** Xf
                Yf_phys = 10 ** Yf
                CS = ax.contour(Xf_phys, Yf_phys, Zf_edges, levels=contour_levels,
                                colors="black", linewidths=cfg.HEATMAP_CONTOUR_LINEWIDTH_AF, fontsize=font_size)
                # White stroke on contour collection
                for collection in CS.collections:
                    collection.set_path_effects([
                        patheffects.withStroke(
                            linewidth=cfg.CONTOUR_STROKE_LINEWIDTH_AF,
                            foreground=cfg.CONTOUR_STROKE_FOREGROUND,
                            alpha=cfg.CONTOUR_STROKE_ALPHA
                        ),
                        patheffects.Normal()
                    ])
                texts = CS.clabel(inline=True, colors="black", fmt=cfg.HEATMAP_CONTOUR_FMT_AF)
                if texts:
                    for txt in texts:
                        txt.set_path_effects([
                            patheffects.withStroke(
                                linewidth=cfg.LABEL_STROKE_LINEWIDTH,
                                foreground=cfg.LABEL_STROKE_FOREGROUND,
                                alpha=cfg.LABEL_STROKE_ALPHA
                            ),
                            patheffects.Normal()
                        ])
            else:
                # Contours on corner grid (Xf,Yf log -> physical)
                Xf_phys = 10 ** Xf
                Yf_phys = 10 ** Yf
                CS = ax.contour(Xf_phys, Yf_phys, Zf_edges, levels=contour_levels,
                                colors="black", linewidths=cfg.HEATMAP_CONTOUR_LINEWIDTH_AF, fontsize=font_size)
                CS.clabel(inline=True, colors="black", fmt=cfg.HEATMAP_CONTOUR_FMT_AF)
            
            # Match axis limits to corner grid
            ax.set_xlim(10 ** x_edges[0], 10 ** x_edges[-1])
            ax.set_ylim(10 ** y_edges[0], 10 ** y_edges[-1])

            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(2)
            ax.set_title("")

        for ax in axes:
            ax.set_xlabel("")
            ax.set_ylabel("")

        # Only save per-scenario heatmaps, no composite
        # fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        # out_svg = fig_root / f"AF_ref_heatmap_all_scenarios_Top_{Top}K.svg"
        # fig.savefig(
        #     out_svg,
        #     transparent=getattr(cfg, "PLOT_TRANSPARENT", True),
        #     bbox_inches="tight",
        #     pad_inches=0.0,
        # )

        cbar_fig = plt.figure(figsize=cfg.HEATMAP_FIGSIZE_COLORBAR_LARGE)
        cbar_ax2 = cbar_fig.add_axes([0.3, 0.05, 0.1, 0.9])
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar2 = cbar_fig.colorbar(sm, cax=cbar_ax2)
        cbar2.set_label('Relative AF (-)')
        cbar2.outline.set_edgecolor('black')
        cbar2.outline.set_linewidth(2)
        cbar_fig.subplots_adjust(left=0.2, right=0.8, top=0.98, bottom=0.02)
        out_cbar_svg = fig_root / f"AF_ref_colorbar_Top_{Top}K.svg"
        cbar_fig.savefig(
            out_cbar_svg,
            transparent=getattr(cfg, "PLOT_TRANSPARENT", True),
            bbox_inches="tight",
            pad_inches=0.0,
        )
        plt.close(cbar_fig)
        plt.close(fig)

        for key, sc in SCENARIOS.items():
            AF_ref_values = af_ref_data_all_scenarios[key]
            Z = AF_ref_values.T

            # Raw coordinates (center); use log scale
            x_c = np.log10(FIXED_RHOT)
            y_c = np.log10(FIXED_NPW)
            
            interp = RegularGridInterpolator((y_c, x_c), Z, bounds_error=False, fill_value=None)
            # High-res grid (corner grid so contours extend to edges)
            # Center-point grid for interpolation (linear in log space)
            x_center = np.linspace(x_c.min(), x_c.max(), 40)
            y_center = np.linspace(y_c.min(), y_c.max(), 40)
            X_center, Y_center = np.meshgrid(x_center, y_center, indexing='xy')
            Zf = interp((Y_center, X_center))
            
            # Corner-point grid (for pcolormesh/contour, contours to edges)
            # For shading="flat": X,Y corner coords (n+1,m+1), Z center values (n,m)
            dx = (x_center[-1] - x_center[0]) / (len(x_center) - 1) if len(x_center) > 1 else 0
            dy = (y_center[-1] - y_center[0]) / (len(y_center) - 1) if len(y_center) > 1 else 0
            x_edges = np.concatenate([[x_center[0] - dx/2], 
                                      (x_center[:-1] + x_center[1:]) / 2, 
                                      [x_center[-1] + dx/2]])
            y_edges = np.concatenate([[y_center[0] - dy/2], 
                                      (y_center[:-1] + y_center[1:]) / 2, 
                                      [y_center[-1] + dy/2]])
            Xf, Yf = np.meshgrid(x_edges, y_edges, indexing='xy')
            
            # Interpolate Zf (center) to corner grid for contour
            interp_edges = RegularGridInterpolator((y_center, x_center), Zf, bounds_error=False, fill_value=None)
            Zf_edges = interp_edges((Yf, Xf))

            fig_single, ax_single = plt.subplots(1, 1, figsize=cfg.HEATMAP_FIGSIZE_SINGLE)
            
            # Set axes to log scale
            ax_single.set_xscale('log')
            ax_single.set_yscale('log')
            
            # Corner grid Xf,Yf and center Zf, shading="flat" (Xf,Yf log -> physical for log axes)
            Xf_phys = 10 ** Xf
            Yf_phys = 10 ** Yf
            ax_single.pcolormesh(Xf_phys, Yf_phys, Zf, cmap=cmap, norm=norm, shading="flat", alpha=cfg.HEATMAP_ALPHA)
            
            # Set axis limits so contours reach the edges
            ax_single.set_xlim(10 ** x_edges[0], 10 ** x_edges[-1])
            ax_single.set_ylim(10 ** y_edges[0], 10 ** y_edges[-1])

            max_val = float(np.nanmax(Zf))
            contour_start = 0.1
            contour_step = 0.1
            
            # Auto-generate contour levels
            if max_val < contour_start:
                contour_step = max(0.01, max_val / 10.0)
                levels_raw = np.arange(0, max_val + contour_step/2, contour_step)
            else:
                levels_raw = np.arange(contour_start, max_val + contour_step/2, contour_step)
            
            if len(levels_raw) > 0:
                if abs(levels_raw[-1] - max_val) > 1e-6:
                    levels = np.append(levels_raw, max_val)
                else:
                    levels = levels_raw
            else:
                levels = np.array([max_val])

            vmax_point_999 = max_val * 0.99
            insert_999 = True
            for v in levels:
                if abs(v - vmax_point_999) < 1e-6:
                    insert_999 = False
                    break
            if len(levels) > 0 and insert_999 and vmax_point_999 > levels[0] and vmax_point_999 < levels[-1]:
                levels = np.append(levels, vmax_point_999)
                levels = np.sort(levels)
            if max_val >= 0.9 and not any(abs(v - 0.9) < 1e-6 for v in levels):
                levels = np.append(levels, 0.9)
                levels = np.sort(levels)
            if max_val >= 0.95 and not any(abs(v - 0.95) < 1e-6 for v in levels):
                levels = np.append(levels, 0.95)
                levels = np.sort(levels)
            # Include 0.99 contour if max_val >= 0.99
            if max_val >= 0.99 and not any(abs(v - 0.99) < 1e-6 for v in levels):
                levels = np.append(levels, 0.99)
                levels = np.sort(levels)

            if len(levels) <= 5:
                contour_levels = list(levels)
            else:
                n_sample = 5
                idxs = np.round(np.linspace(0, len(levels)-2, n_sample)).astype(int)
                sampled = [levels[i] for i in idxs]
                contour_levels = sampled + [levels[-1]]
                if not any(abs(v - vmax_point_999) < 1e-6 for v in contour_levels):
                    contour_levels.append(vmax_point_999)
                # Include 0.9 contour if max_val >= 0.9
                if max_val >= 0.9 and not any(abs(v - 0.9) < 1e-6 for v in contour_levels):
                    contour_levels.append(0.9)
                # Include 0.95 contour if max_val >= 0.95
                if max_val >= 0.95 and not any(abs(v - 0.95) < 1e-6 for v in contour_levels):
                    contour_levels.append(0.95)
                # Include 0.99 contour if max_val >= 0.99
                if max_val >= 0.99 and not any(abs(v - 0.99) < 1e-6 for v in contour_levels):
                    contour_levels.append(0.99)
                contour_levels = sorted(set(contour_levels))

            # Exclude max_val to avoid extra contour at top-right; keep 0.99
            contour_levels = [v for v in contour_levels if v < max_val - 1e-6]
            if use_stroke:
                # Stroke and contours on corner grid (Xf,Yf log -> physical)
                Xf_phys = 10 ** Xf
                Yf_phys = 10 ** Yf
                CS_single = ax_single.contour(Xf_phys, Yf_phys, Zf_edges, levels=contour_levels,
                                              colors="black", linewidths=cfg.HEATMAP_CONTOUR_LINEWIDTH_AF)
                # White stroke on contour collection
                for collection in CS_single.collections:
                    collection.set_path_effects([
                        patheffects.withStroke(
                            linewidth=cfg.CONTOUR_STROKE_LINEWIDTH_AF,
                            foreground=cfg.CONTOUR_STROKE_FOREGROUND,
                            alpha=cfg.CONTOUR_STROKE_ALPHA
                        ),
                        patheffects.Normal()
                    ])
                texts_single = CS_single.clabel(inline=True, colors="black", fmt=cfg.HEATMAP_CONTOUR_FMT_AF)
                if texts_single:
                    for txt in texts_single:
                        txt.set_path_effects([
                            patheffects.withStroke(
                                linewidth=cfg.LABEL_STROKE_LINEWIDTH,
                                foreground=cfg.LABEL_STROKE_FOREGROUND,
                                alpha=cfg.LABEL_STROKE_ALPHA
                            ),
                            patheffects.Normal()
                        ])
            else:
                # Contours on corner grid (Xf,Yf log -> physical)
                Xf_phys = 10 ** Xf
                Yf_phys = 10 ** Yf
                CS_single = ax_single.contour(Xf_phys, Yf_phys, Zf_edges, levels=contour_levels,
                                              colors="black", linewidths=cfg.HEATMAP_CONTOUR_LINEWIDTH_AF)
                CS_single.clabel(inline=True, colors="black", fmt=cfg.HEATMAP_CONTOUR_FMT_AF)

            ax_single.set_xticks([])
            ax_single.set_yticks([])
            for spine in ax_single.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(2)
            ax_single.set_title("")

            single_fig_dir = fig_root / key
            single_fig_dir.mkdir(parents=True, exist_ok=True)
            single_svg = single_fig_dir / f"AF_ref_heatmap_{key}_Top_{Top}K.svg"
            fig_single.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
            fig_single.savefig(
                single_svg,
                transparent=getattr(cfg, "PLOT_TRANSPARENT", True),
                bbox_inches="tight",
                pad_inches=0.0,
            )
            plt.close(fig_single)

        info_txt = current_out_root / "README.txt"
        with open(info_txt, "w", encoding="utf-8") as f:
            f.write(
                "Auto-generated by 2.9_af_form_full_grid.py. Each subdir = one scenario with AF_ref table (xlsx).\n"
                "Data: AF from scan_full_grid.xlsx. Same source as 8.1_plot_from_scan_full_grid.py.\n"
            )

        print(f"Per-scenario AF_ref heatmaps saved to: {fig_root}")
        print(f"Data tables saved to: {current_out_root}")
if __name__ == "__main__":
    # Stroke (halo) on/off from config
    USE_STROKE = getattr(cfg, "USE_STROKE", False)
    
    for Top in Top_list:  
        make_af_heatmaps_for_scenarios(Top=Top, use_stroke=USE_STROKE)
        make_af_ref_heatmaps_for_scenarios(Top=Top, use_stroke=USE_STROKE)
