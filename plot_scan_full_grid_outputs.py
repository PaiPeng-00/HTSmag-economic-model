"""
8.1_plot_from_scan_full_grid.py

Read `scan_full_grid_tidy.csv` and generate economic heatmaps.

Functions:
1. Read `scan_full_grid_tidy.csv` data.
2. Validate that all required columns exist.
3. Compute `delta_LCOE_min` (difference relative to the per‑scenario minimum).
4. Plot `r_cryo_re` (cryogenic recirculating power fraction) heatmaps.
5. Plot `delta_LCOE_min` heatmaps.

If the data file is missing, print an error suggesting to run `scan_full_grid.py`
with the appropriate scan dimensions enabled.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Set, Tuple

# Import project modules
sys.path.insert(0, str(Path(__file__).parent))
import config as cfg
from src.tfmag.paths import ensure_base_dirs

# Import plotting helpers
from plot_library import (
    plot_parasitic_heatmap_single,
    save_parasitic_heatmap_colorbar,
    plot_delta_lcoe_heatmap_single,
    save_delta_lcoe_heatmap_colorbar,
    plot_cryo_power_heatmap_single,
)
import matplotlib as mpl

# =============================================================================
# Configuration (kept consistent with 8.0)
# =============================================================================

# Plotting config (from the plotting section)
DELTA_LCOE_CMAP = cfg.cmap_parasitic_ratio  # plotting.cmap_parasitic_ratio

# Economic‑analysis config (from the `economic_analysis` section)
DELTA_LCOE_MASK_PARASITIC_PCT = cfg.DELTA_LCOE_MASK_PARASITIC_PCT  # economic_analysis.DELTA_LCOE_MASK_PARASITIC_PCT
DELTA_LCOE_ABS_PCTL = cfg.DELTA_LCOE_ABS_PCTL  # economic_analysis.DELTA_LCOE_ABS_PCTL
USD_TO_CNY_EXCHANGE_RATE = cfg.USD_TO_CNY_EXCHANGE_RATE  # economic_analysis.USD_TO_CNY_EXCHANGE_RATE

# Scan‑parameter config (from `parameter_scan_lists` and `economic_analysis`).
# Use exactly the same configuration as 8.0 to keep grids identical.
# Note: `R_JOINT_SCAN_VALUES` is in nOhm (log‑spaced 1–100 nOhm, 31 points).
# When Ohm is required for computation/interpolation, we explicitly multiply by 1e‑9.
R_JOINT_SCAN_VALUES = cfg.R_JOINT_SCAN_VALUES  # nOhm, log-spaced array
NPW_SCAN_VALUES = cfg.NPW_SCAN_VALUES  # parameter_scan_lists.npw_scan_values
SCAN_TEMP_COOLANT_PAIRS = cfg.SCAN_TEMP_COOLANT_PAIRS  # economic_analysis.scan_temp_coolant_pairs

# Output directories (from `io_paths`, consistent with 8.0)
SCAN_BASE_DIR = Path(cfg.ECONOMIC_FIGURES_DIR) / cfg.PARASITIC_RATIO_OUTPUT_DIR  # io_paths.ECONOMIC_OUTPUT_DIR / io_paths.PARASITIC_RATIO_OUTPUT_DIR
SCAN_USD_DIR = SCAN_BASE_DIR / cfg.PARASITIC_RATIO_USD_DIR  # io_paths.PARASITIC_RATIO_USD_DIR
SCAN_CNY_DIR = SCAN_BASE_DIR / cfg.PARASITIC_RATIO_CNY_DIR  # io_paths.PARASITIC_RATIO_CNY_DIR
# Subdirectory name for single‑panel cryogenic power (MWe) plots
# under each colour scheme: `parasitic_ratio/cryo_power_MWe`.
CRYO_POWER_MWE_SUBDIR = "cryo_power_MWe"

# Enable or disable generating CNY versions of the plots
ENABLE_CNY = True

# Enable or disable halo/stroke (black contour with white outline)
USE_STROKE = cfg.USE_STROKE

# Input data file
paths = ensure_base_dirs()
INPUT_CSV = paths.outputs_tables / "scan_full_grid_tidy.csv"

# =============================================================================
# Helper functions
# =============================================================================

def match_r_joint_by_significant_digits(r1_nohm: float, r2_nohm: float, n_digits: int = 3) -> bool:
    """
    Compare two joint‑resistance values (in nOhm) after rounding to `n_digits`
    decimal places.

    Args:
        r1_nohm: First resistance value (nOhm).
        r2_nohm: Second resistance value (nOhm).
        n_digits: Number of decimal digits used for comparison (default: 3).

    Returns:
        True if the rounded values are equal, otherwise False.
    """
    try:
        # Round both values to `n_digits` decimal places before comparison
        v1 = round(float(r1_nohm), n_digits)
        v2 = round(float(r2_nohm), n_digits)
    except Exception:
        return False
    return v1 == v2


def ensure_complete_data_grid(
    df: pd.DataFrame,
    npw_values: np.ndarray,
    r_joint_values: np.ndarray,
    value_col: str
) -> pd.DataFrame:
    """
    Ensure that the data grid contains all combinations of `Npw` and `R_joint`,
    filling missing combinations with NaN.

    This guarantees that the heatmap grid matches 8.0 exactly.

    Args:
        df: Input DataFrame; must contain `Npw`, `R_joint`, and `value_col`.
        npw_values: Complete array of Npw values (from `NPW_SCAN_VALUES`).
        r_joint_values: Complete array of R_joint values (from
            `R_JOINT_SCAN_VALUES`, in Ohm).
        value_col: Name of the column whose values should populate the grid.

    Returns:
        DataFrame containing a complete grid.
    """
    # Create a complete (Npw, R_joint) grid
    grid_data = []
    for npw in npw_values:
        for r_joint in r_joint_values:
            grid_data.append({
                'Npw': npw,
                'R_joint': r_joint
            })
    
    grid_df = pd.DataFrame(grid_data)
    
    # Merge actual data onto the full grid
    df_work = df.copy()
    
    # If `R_joint_nOhm` exists but `R_joint` does not, convert units
    if 'R_joint_nOhm' in df_work.columns and 'R_joint' not in df_work.columns:
        df_work['R_joint'] = df_work['R_joint_nOhm'] * 1e-9  # nΩ -> Ω
    
    # Check required columns
    if 'Npw' not in df_work.columns or 'R_joint' not in df_work.columns:
        grid_df[value_col] = np.nan
        return grid_df
    
    # Merge using tolerances (to handle floating‑point round‑off)
    merged_df = grid_df.copy()
    merged_df[value_col] = np.nan
    
    if value_col in df_work.columns:
        # For each grid point, find the best‑matching data row(s)
        for idx, row in grid_df.iterrows():
            npw = row['Npw']
            r_joint = row['R_joint']
            
            # Find matching data (with tolerance). Use a relatively loose
            # tolerance for `R_joint` because of floating‑point issues.
            mask = (
                (np.isclose(df_work['Npw'], npw, rtol=0.01)) &
                (np.isclose(df_work['R_joint'], r_joint, rtol=0.01, atol=1e-12))
            )
            if mask.any():
                # If multiple matches exist, take the mean (more robust)
                matched_indices = df_work.index[mask]
                if len(matched_indices) > 0:
                    matched_values = df_work.loc[matched_indices, value_col]
                    # Use the mean of non‑NaN values only
                    valid_values = matched_values.dropna()
                    if len(valid_values) > 0:
                        merged_df.loc[idx, value_col] = valid_values.mean()
                    else:
                        merged_df.loc[idx, value_col] = np.nan
    
    # Ensure there are no duplicate (Npw, R_joint) combinations
    # (should not happen, but guard just in case)
    if merged_df.duplicated(subset=['Npw', 'R_joint']).any():
        merged_df = merged_df.groupby(['Npw', 'R_joint'], as_index=False)[value_col].first()
    
    return merged_df

def check_required_columns(df: pd.DataFrame, required_cols: List[str]) -> Tuple[bool, List[str]]:
    """
    Check whether the DataFrame contains all required columns.

    Returns:
        (all_present, list_of_missing_columns)
    """
    missing_cols = [col for col in required_cols if col not in df.columns]
    return len(missing_cols) == 0, missing_cols


def get_missing_scan_dimensions(df: pd.DataFrame, required_cols: List[str]) -> dict:
    """
    Infer which scan dimensions are missing based on absent columns.

    Returns:
        A dict describing missing scan dimensions.
    """
    missing_info = {}
    
    # Check basic parameter columns
    if 'Top_K' not in df.columns:
        missing_info['Top_K'] = "temperature dimension must be scanned"
    if 'coolant' not in df.columns:
        missing_info['coolant'] = "coolant dimension must be scanned"
    if 'Npw' not in df.columns:
        missing_info['Npw'] = "Npw (number of parallel tapes) dimension must be scanned"
    if 'R_joint_nOhm' not in df.columns:
        missing_info['R_joint_nOhm'] = "joint‑resistance dimension must be scanned"
    if 'scenario' not in df.columns:
        missing_info['scenario'] = "scenario dimension must be scanned"

    # Check data columns
    if 'r_cryo_re' not in df.columns:
        missing_info['r_cryo_re'] = "economic metric `r_cryo_re` must be computed"
    if 'LCOE_magnet_only_USD_per_MWh' not in df.columns:
        missing_info['LCOE_magnet_only_USD_per_MWh'] = "economic metric `LCOE` must be computed"
    
    return missing_info


def calculate_delta_lcoe_min(df: pd.DataFrame, scenarios: List[str]) -> pd.DataFrame:
    """
    Compute the difference to the per‑scenario minimum LCOE.

    Args:
        df: Input DataFrame.
        scenarios: List of scenario names.

    Returns:
        DataFrame with a new column `delta_LCOE_min_USD_per_MWh`.
    """
    df = df.copy()
    
    # If the column already exists, reuse it
    if 'delta_LCOE_min_USD_per_MWh' in df.columns:
        print("Column `delta_LCOE_min_USD_per_MWh` already exists; reusing it.")
        return df
    
    print("\n--- Computing per‑scenario ΔLCOE to the minimum ---")
    
    lcoe_col = 'LCOE_magnet_only_USD_per_MWh'
    r_cryo_col = 'r_cryo_re'
    
    # Initialize new column
    df['delta_LCOE_min_USD_per_MWh'] = np.nan
    
    for scenario in scenarios:
        scenario_mask = df['scenario'] == scenario
        
        if not scenario_mask.any():
            print(f"[{scenario}] Scenario does not exist in the data; skipping.")
            continue
        
        # All LCOE values for this scenario (all temperature–coolant pairs)
        scenario_lcoe = df.loc[scenario_mask, lcoe_col]
        
        # Feasibility mask: parasitic power below the threshold
        if r_cryo_col in df.columns:
            feasible_mask = (
                scenario_lcoe.notna() &
                (df.loc[scenario_mask, r_cryo_col] <= DELTA_LCOE_MASK_PARASITIC_PCT)
            )
        else:
            feasible_mask = scenario_lcoe.notna()
        
        if feasible_mask.any():
            # Minimum LCOE across all temperature–coolant pairs for this scenario
            scenario_min_lcoe = float(scenario_lcoe[feasible_mask].min())
            
            # Log which combination this minimum value comes from
            min_lcoe_row = df.loc[scenario_mask & feasible_mask].loc[
                df.loc[scenario_mask & feasible_mask, lcoe_col] == scenario_min_lcoe
            ]
            if not min_lcoe_row.empty:
                min_temp = min_lcoe_row.iloc[0].get('Top_K', 'N/A')
                min_cool = min_lcoe_row.iloc[0].get('coolant', 'N/A')
                min_npw = min_lcoe_row.iloc[0].get('Npw', 'N/A')
                min_rj = min_lcoe_row.iloc[0].get('R_joint_nOhm', 'N/A')
                print(
                    f"[{scenario}] Scenario minimum LCOE: {scenario_min_lcoe:.3g} ($/MWh) "
                    f"[at: {min_temp} K {min_cool}, Npw={min_npw}, R_joint={min_rj:.4g} nOhm]"
                )
            else:
                print(f"[{scenario}] Scenario minimum LCOE: {scenario_min_lcoe:.3g} ($/MWh)")
            
            # ΔLCOE relative to the scenario minimum for all points
            df.loc[scenario_mask, 'delta_LCOE_min_USD_per_MWh'] = (
                df.loc[scenario_mask, lcoe_col] - scenario_min_lcoe
            )
        else:
            print(f"[{scenario}] WARNING: no feasible points in this scenario; cannot compute minimum.")
    
    return df


def format_array_for_print(arr: np.ndarray) -> str:
    """Format a NumPy array for printing."""
    if len(arr) == 0:
        return "[]"
    
    formatted = np.array2string(
        arr,
        separator=', ',
        suppress_small=True,
        precision=3,
        floatmode='fixed',
        threshold=100
    )
    return formatted


def calculate_lcoe_contour_levels(
    all_finite: np.ndarray, 
    delta_bound: float,
    unit: str = 'USD'  # 'USD' or 'CNY'
) -> np.ndarray:
    """
    Compute a set of contour levels for LCOE.

    Args:
        all_finite: Array of finite data values (already clipped to `delta_bound`).
        delta_bound: Symmetric bound used for clipping.
        unit: Either `'USD'` or `'CNY'`.

    Returns:
        Sorted array of contour levels.
    """
    method = getattr(cfg, 'LCOE_CONTOUR_METHOD', 'adaptive')
    min_spacing = cfg.CONTOUR_MIN_SPACING_USD if unit == 'USD' else cfg.CONTOUR_MIN_SPACING_CNY
    
    if method == 'fixed':
        if unit == 'USD':
            fixed_levels = cfg.LCOE_CONTOUR_FIXED_LEVELS.copy()
        else:
            fixed_levels = cfg.LCOE_CONTOUR_FIXED_LEVELS_CNY.copy()
        
        if 0.0 not in fixed_levels:
            fixed_levels = np.concatenate([fixed_levels, [0.0]])
        
        contour_levels = np.sort(fixed_levels)
    else:
        # Adaptive method
        if len(all_finite) == 0:
            contour_levels = np.array([0.0])
        else:
            global_max_abs = np.nanmax(np.abs(all_finite))
            if unit == 'USD':
                if global_max_abs > 50:
                    total_n_levels = 25
                elif global_max_abs > 20:
                    total_n_levels = 21
                else:
                    total_n_levels = 17
            else:  # CNY
                if global_max_abs > 0.5:
                    total_n_levels = 25
                elif global_max_abs > 0.2:
                    total_n_levels = 21
                else:
                    total_n_levels = 17
            
            # Handle positive and negative values separately
            pos_data = all_finite[all_finite > 0]
            neg_data = all_finite[all_finite < 0]
            
            pos_levels = np.array([])
            neg_levels = np.array([])
            
            n_non_zero = total_n_levels - 1
            
            if len(pos_data) > 0 and len(neg_data) > 0:
                pos_ratio = len(pos_data) / (len(pos_data) + len(neg_data))
                n_pos = max(1, int(round(n_non_zero * pos_ratio)))
                n_neg = max(1, n_non_zero - n_pos)
            elif len(pos_data) > 0:
                n_pos = n_non_zero
                n_neg = 0
            elif len(neg_data) > 0:
                n_pos = 0
                n_neg = n_non_zero
            else:
                n_pos = 0
                n_neg = 0
            
            # For the positive part of the distribution
            if n_pos > 0 and len(pos_data) > 0:
                pos_sorted = np.sort(pos_data)
                n_bins = min(100, len(pos_sorted) // 10)
                if n_bins > 0:
                    hist, bin_edges = np.histogram(pos_sorted, bins=n_bins)
                    hist_density = hist / hist.sum() if hist.sum() > 0 else hist
                    cum_density = np.cumsum(hist_density)
                    cum_density = np.concatenate([[0], cum_density])
                    target_cum_densities = np.linspace(0, 1, n_pos + 1)[1:]
                    pos_levels = np.interp(target_cum_densities, cum_density, bin_edges)
                else:
                    percentiles = np.linspace(0, 100, n_pos + 1)[1:]
                    pos_levels = np.percentile(pos_sorted, percentiles)
            
            # For the negative part of the distribution
            if n_neg > 0 and len(neg_data) > 0:
                neg_abs_sorted = np.sort(np.abs(neg_data))
                n_bins = min(100, len(neg_abs_sorted) // 10)
                if n_bins > 0:
                    hist, bin_edges = np.histogram(neg_abs_sorted, bins=n_bins)
                    hist_density = hist / hist.sum() if hist.sum() > 0 else hist
                    cum_density = np.cumsum(hist_density)
                    cum_density = np.concatenate([[0], cum_density])
                    target_cum_densities = np.linspace(0, 1, n_neg + 1)[1:]
                    neg_abs_levels = np.interp(target_cum_densities, cum_density, bin_edges)
                    neg_levels = -neg_abs_levels
                else:
                    percentiles = np.linspace(0, 100, n_neg + 1)[1:]
                    neg_levels = -np.percentile(neg_abs_sorted, percentiles)
            
            # Combine all contour levels
            if len(neg_levels) > 0 and len(pos_levels) > 0:
                contour_levels = np.concatenate([neg_levels[::-1], [0.0], pos_levels])
            elif len(neg_levels) > 0:
                contour_levels = np.concatenate([neg_levels[::-1], [0.0]])
            elif len(pos_levels) > 0:
                contour_levels = np.concatenate([[0.0], pos_levels])
            else:
                contour_levels = np.array([0.0])
            
            contour_levels = np.sort(contour_levels)
    
    # Enforce minimum spacing between contour levels
    filtered_levels = [contour_levels[0]]
    for i in range(1, len(contour_levels)):
        if abs(contour_levels[i] - filtered_levels[-1]) >= min_spacing:
            filtered_levels.append(contour_levels[i])
    
    return np.array(filtered_levels)


# =============================================================================
# Main API
# =============================================================================

def plot_from_scan_full_grid(
    scenarios: List[str] = None,
    temp_coolant_pairs: List[Tuple[float, str]] = None,
    enable_cny: bool = False
):
    """
    Read `scan_full_grid_tidy.csv` and generate heatmaps.

    Args:
        scenarios: List of scenarios to plot. If None, infer from data.
        temp_coolant_pairs: List of (Top_K, coolant) pairs to plot.
            If None, use configuration `SCAN_TEMP_COOLANT_PAIRS`.
        enable_cny: If True, also generate CNY versions of the heatmaps.
    """
    print("\n" + "=" * 60)
    print("--- Read scan_full_grid_tidy.csv and generate heatmaps ---")
    print("=" * 60)

    # 1. Check input file existence
    if not INPUT_CSV.exists():
        print(f"\n[ERROR] Input file does not exist: {INPUT_CSV}")
        print("Please run `scan_full_grid.py` first to generate this data.")
        return
    
    # 2. Read data
    print(f"\n[Info] Reading data file: {INPUT_CSV}")
    try:
        df = pd.read_csv(INPUT_CSV)
        print(f"[OK] Successfully read {len(df):,} rows.")
    except Exception as e:
        print(f"[ERROR] Failed to read file: {e}")
        return
    
    if df.empty:
        print("[ERROR] Data file is empty.")
        return
    
    # 3. Check required columns
    required_cols = [
        'Top_K', 'coolant', 'Npw', 'R_joint_nOhm', 'scenario',
        'r_cryo_re', 'LCOE_magnet_only_USD_per_MWh'
    ]
    
    all_present, missing_cols = check_required_columns(df, required_cols)
    
    if not all_present:
        print(f"\n[ERROR] Data file is missing required columns:")
        for col in missing_cols:
            print(f"  - {col}")
        
        missing_info = get_missing_scan_dimensions(df, missing_cols)
        print(f"\n[Hint] You need to enable the following scan dimensions in `scan_full_grid.py`:")
        for col, info in missing_info.items():
            print(f"  - {col}: {info}")
        
        print("\nPlease run `scan_full_grid.py` and ensure it includes at least:")
        print("  - TEMP_COOLANT_PAIRS: temperature–coolant combinations")
        print("  - NPW_RANGE: range of Npw values")
        print("  - R_JOINT_NOHM: list of joint‑resistance values")
        print("  - SCENARIOS: list of scenarios")
        return
    
    # 4. Select scenarios and temperature–coolant pairs (same as 8.0)
    if scenarios is None:
        # Derive scenarios from data, but prefer those defined in config
        available_scenarios = sorted(df['scenario'].unique().tolist())
        # If `SCENARIO_DEFINITIONS` exists, prioritise its keys
        if hasattr(cfg, 'SCENARIO_DEFINITIONS'):
            config_scenarios = list(cfg.SCENARIO_DEFINITIONS.keys())
            scenarios = [s for s in config_scenarios if s in available_scenarios]
            if not scenarios:
                scenarios = available_scenarios
        else:
            scenarios = available_scenarios
        print(f"\n[Info] Scenarios to plot: {scenarios}")

    if temp_coolant_pairs is None:
        # Use the same configuration as 8.0
        temp_coolant_pairs = SCAN_TEMP_COOLANT_PAIRS
        print(f"[Info] Temperature–coolant pairs (from config): {temp_coolant_pairs}")

    # 5. Compute `delta_LCOE_min`
    df = calculate_delta_lcoe_min(df, scenarios)

    
    # 6. Global norm for parasitic ratio
    if 'r_cryo_re' in df.columns:
        r_cryo_series = df['r_cryo_re'].replace([np.inf, -np.inf], np.nan).dropna()
        if not r_cryo_series.empty:
            global_vmin = float(r_cryo_series.min())
            global_vmax = float(r_cryo_series.max())
            global_norm_parasitic = mpl.colors.LogNorm(
                vmin=float(f"{global_vmin:.10f}"),
                vmax=float(f"{global_vmax:.10f}")
            )
            print(f"\n[Info] Global colorbar range for r_cryo_re: {global_vmin:.3g}% – {global_vmax:.3g}%")
        else:
            print("[Warning] Column `r_cryo_re` is empty; cannot plot this heatmap.")
            global_norm_parasitic = None
    else:
        global_norm_parasitic = None
    
    # 7. Global norm for ΔLCOE
    if 'delta_LCOE_min_USD_per_MWh' in df.columns:
        # Mask infeasible points
        if 'r_cryo_re' in df.columns:
            bad = df['r_cryo_re'] > DELTA_LCOE_MASK_PARASITIC_PCT
            df.loc[bad, 'delta_LCOE_min_USD_per_MWh'] = np.nan
            print(f"[Info] Masked points with parasitic power > {DELTA_LCOE_MASK_PARASITIC_PCT}% as infeasible "
                  f"({bad.sum()} points).")
        
        delta_series = (
            df['delta_LCOE_min_USD_per_MWh']
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        
        delta_norm = None
        delta_bound = None
        delta_min = None
        delta_max = None
        global_contour_levels = None
        
        if not delta_series.empty:
            # (Previously: use P99.5 percentile of |Δ| as `delta_bound`; now disabled.)
            # delta_bound = float(np.nanpercentile(np.abs(delta_series.values), DELTA_LCOE_ABS_PCTL))
            
            # Do not use percentile clipping; use the actual data range
            delta_min = float(delta_series.min())
            delta_max = float(delta_series.max())
            
            if delta_max > delta_min:  # ensure a valid dynamic range
                linthresh = cfg.DELTA_LCOE_LINTHRESH_USD
                linscale = cfg.DELTA_LCOE_LINSCALE_USD
                
                if delta_min < 0 and delta_max > 0:
                    norm_vmin = delta_min
                    norm_vmax = delta_max
                elif delta_min >= 0:
                    norm_vmin = 0.0
                    norm_vmax = delta_max
                else:
                    norm_vmin = delta_min
                    norm_vmax = 0.0
                
                delta_norm = mpl.colors.SymLogNorm(
                    linthresh=linthresh,
                    linscale=linscale,
                    vmin=norm_vmin,
                    vmax=norm_vmax
                )
                # (Previously: print P99.5‑based bound; now disabled.)
                # print(f"[Info] ΔLCOE_min colourbar range set by |Δ| P{DELTA_LCOE_ABS_PCTL}: ±{delta_bound:.3g} ($/MWh)")
                print(f"[Info] ΔLCOE_min colourbar range: [{delta_min:.3g}, {delta_max:.3g}] ($/MWh)")

                # (Previously: clip with `delta_bound`; now disabled.)
                # all_delta_clipped = df['delta_LCOE_min_USD_per_MWh'].clip(-delta_bound, delta_bound)
                # all_finite = all_delta_clipped.replace([np.inf, -np.inf], np.nan).dropna().values
                # global_contour_levels = calculate_lcoe_contour_levels(all_finite, delta_bound, unit='USD')
                
                # Do not clip; use all finite values directly
                all_finite = delta_series.replace([np.inf, -np.inf], np.nan).dropna().values
                # Use the max absolute value as `delta_bound` for contour levels
                delta_bound = float(np.max(np.abs(all_finite)))
                global_contour_levels = calculate_lcoe_contour_levels(all_finite, delta_bound, unit='USD')
    else:
        delta_norm = None
        delta_bound = None
        delta_min = None
        delta_max = None
        global_contour_levels = None
    
    # 8. Data‑completeness checks (before plotting)
    print("\n--- Checking data completeness ---")
    # Use the scan configuration to check completeness (consistent with 8.0)
    npw_array = np.array(NPW_SCAN_VALUES)
    r_joint_array = np.array(R_JOINT_SCAN_VALUES, dtype=float)  # nOhm, log-spaced array used for checks
    expected_total_points = len(npw_array) * len(r_joint_array)
    
    data_completeness_warnings = []
    for scenario in scenarios:
        scenario_df = df[df['scenario'] == scenario].copy()
        if scenario_df.empty:
            data_completeness_warnings.append(f"  [Warning] {scenario}: scenario data is empty.")
            continue
        
        for temp, cool in temp_coolant_pairs:
            temp_df = scenario_df[
                (scenario_df['Top_K'] == temp) &
                (scenario_df['coolant'] == cool)
            ].copy()
            
            if temp_df.empty:
                data_completeness_warnings.append(
                    f"  [Warning] {scenario} - {temp} K {cool}: no data."
                )
                continue
            
            # Select rows with `rho_turn_uOhm_cm2` ≈ 10000 (for heatmaps)
            RHO_TURN_FOR_HEATMAP = 10000.0
            if 'rho_turn_uOhm_cm2' in temp_df.columns:
                available_rhot = temp_df['rho_turn_uOhm_cm2'].unique()
                if len(available_rhot) > 0:
                    closest_rhot = available_rhot[np.argmin(np.abs(available_rhot - RHO_TURN_FOR_HEATMAP))]
                    temp_df = temp_df[np.isclose(temp_df['rho_turn_uOhm_cm2'], closest_rhot, rtol=0.01)].copy()
            
            if temp_df.empty:
                continue
            
            # Restrict to values defined in the configuration
            if 'Npw' in temp_df.columns:
                temp_df = temp_df[temp_df['Npw'].isin(npw_array)].copy()
            if 'R_joint_nOhm' in temp_df.columns:
                r_joint_nohm_array = r_joint_array  # nOhm
                # Match using the first 4 significant digits
                temp_df = temp_df[
                    temp_df['R_joint_nOhm'].apply(
                        lambda x: any(match_r_joint_by_significant_digits(x, expected_r, n_digits=4)
                                     for expected_r in r_joint_nohm_array)
                    )
                ].copy()
            
            if temp_df.empty:
                data_completeness_warnings.append(
                    f"  [Warning] {scenario} - {temp} K {cool}: no data after filtering."
                )
                continue
            
            # Count data points (deduplicate using 3 decimal places for R_joint_nOhm)
            # Normalise R_joint_nOhm to expected grid values (3‑digit matching)
            r_joint_nohm_array = r_joint_array  # nOhm
            normalized_rows = []
            for _, row in temp_df.iterrows():
                npw = int(row['Npw'])
                r_joint_nohm = float(row['R_joint_nOhm'])
                # Find matching grid value (3‑decimal‑place comparison)
                matched_r = None
                for expected_r in r_joint_nohm_array:
                    if match_r_joint_by_significant_digits(r_joint_nohm, expected_r, n_digits=3):
                        matched_r = expected_r
                        break
                if matched_r is not None:
                    normalized_rows.append((npw, matched_r))
            
            # Deduplicate grid points
            actual_points = len(set(normalized_rows))
            missing_points = expected_total_points - actual_points
            missing_ratio = missing_points / expected_total_points if expected_total_points > 0 else 1.0
            
            if missing_ratio > 0.3:  # more than 30% points are missing
                data_completeness_warnings.append(
                    f"  [Warning] {scenario} - {temp} K {cool}: "
                    f"insufficient data points ({actual_points}/{expected_total_points}, "
                    f"missing {missing_ratio*100:.1f}%)."
                )
    
    if data_completeness_warnings:
        print("\nData‑completeness check:")
        for warning in data_completeness_warnings:
            print(warning)
        # Log missing combinations
        print("\nMissing (Npw, R_joint_nOhm) combinations:")
        for scenario in scenarios:
            scenario_df = df[df['scenario'] == scenario].copy()
            for temp, cool in temp_coolant_pairs:
                temp_df = scenario_df[(scenario_df['Top_K'] == temp) & (scenario_df['coolant'] == cool)].copy()
                if temp_df.empty:
                    continue
                npw_array = np.array(NPW_SCAN_VALUES)
                r_joint_array = np.array(R_JOINT_SCAN_VALUES, dtype=float)  # nOhm, log-spaced (for checking)
                r_joint_nohm_array = r_joint_array  # already in nOhm
                expected_grid = set((n, r) for n in npw_array for r in r_joint_nohm_array)
                
                # Existing data points (matched using 3-decimal-place comparison)
                if 'R_joint_nOhm' in temp_df.columns and 'Npw' in temp_df.columns:
                    existing_grid = set()
                    for _, row in temp_df.drop_duplicates(subset=['Npw', 'R_joint_nOhm']).iterrows():
                        npw = int(row['Npw'])
                        r_joint_nohm = float(row['R_joint_nOhm'])
                        # Find the expected grid value (3-decimal-place comparison)
                        matched_r = None
                        for expected_r in r_joint_nohm_array:
                            if match_r_joint_by_significant_digits(r_joint_nohm, expected_r, n_digits=3):
                                matched_r = expected_r
                                break
                        if matched_r is not None:
                            existing_grid.add((npw, matched_r))
                    
                    missing_grid = expected_grid - existing_grid
                    if missing_grid:
                        print(
                            f"  Missing ({scenario}-{temp} K-{cool}): {len(missing_grid)} points in total."
                        )
                        # Grouped by Npw
                        for npw in npw_array:
                            missing_rj = [r for (n, r) in missing_grid if n==npw]
                            if missing_rj:
                                print(f"    Npw={npw}: missing R_joint_nOhm={missing_rj}")
        print("")
        print("\nHint: Missing data may cause large blank regions in heatmaps.")
        print("      Consider rerunning `scan_full_grid.py` with full scan dimensions enabled.\n")

    # 8b. Cryogenic power (MWe): pre‑compute a global norm shared by all
    # colour schemes (individual plots are generated in step 9).
    norm_cryo = None
    scenario_cryo_df = None
    CRYO_MODES = [
        ("pulse", "P_cryo_prod_W", "Pulse"),
        ("dwell", "P_cryo_dwell_W", "Dwell"),
        ("static", "P_cryo_static_W", "Static"),
    ]
    CRYO_TEMP_COOLANT = [(4.2, "He"), (10.0, "He"), (20.0, "He")]
    cryo_power_cols = ["P_cryo_prod_W", "P_cryo_dwell_W", "P_cryo_static_W"]
    if all(c in df.columns for c in cryo_power_cols):
        npw_array_cryo = np.array(NPW_SCAN_VALUES)
        r_joint_ohm_array_cryo = np.array(R_JOINT_SCAN_VALUES, dtype=float) * 1e-9  # Ω
        scenario_cryo = scenarios[0] if scenarios else "S1"
        scenario_cryo_df = df[df["scenario"] == scenario_cryo].copy()
        all_mwe = []
        for _mode_key, col_w, _ in CRYO_MODES:
            for temp, cool in CRYO_TEMP_COOLANT:
                sub = scenario_cryo_df[
                    (scenario_cryo_df["Top_K"] == temp) & (scenario_cryo_df["coolant"] == cool)
                ].copy()
                if "rho_turn_uOhm_cm2" in sub.columns:
                    RHO_TURN_FOR_HEATMAP = 10000.0
                    available_rhot = sub["rho_turn_uOhm_cm2"].unique()
                    if len(available_rhot) > 0:
                        closest_rhot = available_rhot[np.argmin(np.abs(available_rhot - RHO_TURN_FOR_HEATMAP))]
                        sub = sub[np.isclose(sub["rho_turn_uOhm_cm2"], closest_rhot, rtol=0.01)].copy()
                if sub.empty:
                    continue
                sub = sub[sub["Npw"].isin(npw_array_cryo)].copy()
                if "R_joint_nOhm" in sub.columns:
                    r_joint_nohm_array = np.array(R_JOINT_SCAN_VALUES, dtype=float)
                    sub = sub[
                        sub["R_joint_nOhm"].apply(
                            lambda x: any(
                                match_r_joint_by_significant_digits(x, r, n_digits=3)
                                for r in r_joint_nohm_array
                            )
                        )
                    ].copy()
                if sub.empty:
                    continue
                sub = sub.drop_duplicates(subset=["Npw", "R_joint_nOhm"], keep="first")
                sub["R_joint"] = sub["R_joint_nOhm"] * 1e-9
                sub["cryo_power_MWe"] = sub[col_w] / 1e6
                valid = sub["cryo_power_MWe"].replace([np.inf, -np.inf], np.nan).dropna()
                valid = valid[valid > 0]
                if not valid.empty:
                    all_mwe.extend(valid.tolist())
        if all_mwe:
            vmin_cryo = float(np.min(all_mwe))
            vmax_cryo = float(np.max(all_mwe))
            norm_cryo = mpl.colors.LogNorm(vmin=vmin_cryo, vmax=vmax_cryo)
    else:
        missing_cryo = [c for c in cryo_power_cols if c not in df.columns]
        if missing_cryo:
            print(f"[Info] Skipping cryogenic‑power (MWe) plots (missing columns: {missing_cryo}).")

    # 9. Generate heatmaps for each colour scheme
    for cmap_name in cfg.color_schemes:
        print(f"\nProcessing colour scheme: {cmap_name}")
        # Each colour scheme has its own directory:
        # `outputs/figures/economic/{cmap_name}/parasitic_ratio/USD`
        cmap_base_dir = Path(cfg.ECONOMIC_FIGURES_DIR) / cmap_name
        cmap_scan_base_dir = cmap_base_dir / cfg.PARASITIC_RATIO_OUTPUT_DIR
        cmap_scan_usd_dir = cmap_scan_base_dir / cfg.PARASITIC_RATIO_USD_DIR
        cmap_scan_usd_dir.mkdir(exist_ok=True, parents=True)
        
        # Cryogenic power (MWe) single plots: under
        # `{cmap_name}/parasitic_ratio/cryo_power_MWe`
        if norm_cryo is not None and scenario_cryo_df is not None:
            cmap_cryo_dir = cmap_scan_base_dir / CRYO_POWER_MWE_SUBDIR
            cmap_cryo_dir.mkdir(parents=True, exist_ok=True)
            npw_array_cryo = np.array(NPW_SCAN_VALUES)
            r_joint_ohm_array_cryo = np.array(R_JOINT_SCAN_VALUES, dtype=float) * 1e-9
            for mode_key, col_w, _mode_label in CRYO_MODES:
                for temp, cool in CRYO_TEMP_COOLANT:
                    sub = scenario_cryo_df[
                        (scenario_cryo_df["Top_K"] == temp) & (scenario_cryo_df["coolant"] == cool)
                    ].copy()
                    if "rho_turn_uOhm_cm2" in sub.columns:
                        RHO_TURN_FOR_HEATMAP = 10000.0
                        available_rhot = sub["rho_turn_uOhm_cm2"].unique()
                        if len(available_rhot) > 0:
                            closest_rhot = available_rhot[np.argmin(np.abs(available_rhot - RHO_TURN_FOR_HEATMAP))]
                            sub = sub[np.isclose(sub["rho_turn_uOhm_cm2"], closest_rhot, rtol=0.01)].copy()
                    if sub.empty:
                        continue
                    sub = sub[sub["Npw"].isin(npw_array_cryo)].copy()
                    if "R_joint_nOhm" in sub.columns:
                        r_joint_nohm_array = np.array(R_JOINT_SCAN_VALUES, dtype=float)
                        sub = sub[
                            sub["R_joint_nOhm"].apply(
                                lambda x: any(
                                    match_r_joint_by_significant_digits(x, r, n_digits=3)
                                    for r in r_joint_nohm_array
                                )
                            )
                        ].copy()
                    if sub.empty:
                        continue
                    sub = sub.drop_duplicates(subset=["Npw", "R_joint_nOhm"], keep="first")
                    sub["R_joint"] = sub["R_joint_nOhm"] * 1e-9
                    sub["cryo_power_MWe"] = sub[col_w] / 1e6
                    complete_cryo = ensure_complete_data_grid(
                        sub[["Npw", "R_joint", "cryo_power_MWe"]].copy(),
                        npw_array_cryo,
                        r_joint_ohm_array_cryo,
                        "cryo_power_MWe",
                    )
                    plot_cryo_power_heatmap_single(
                        df=complete_cryo,
                        output_dir=cmap_cryo_dir,
                        cmap=cmap_name,
                        norm=norm_cryo,
                        mode_label=mode_key,
                        temperature_K=temp,
                        value_col="cryo_power_MWe",
                    )
            print(f"[OK] Cryogenic‑power (MWe) plots saved to: {cmap_cryo_dir}")

        # Per‑scenario heatmaps
        for scenario in scenarios:
            scenario_df = df[df['scenario'] == scenario].copy()
            
            if scenario_df.empty:
                continue
            
            output_dir = cmap_scan_usd_dir / scenario
            output_dir.mkdir(exist_ok=True, parents=True)
            
            # Per temperature–coolant combination
            for temp, cool in temp_coolant_pairs:
                temp_df = scenario_df[
                    (scenario_df['Top_K'] == temp) &
                    (scenario_df['coolant'] == cool)
                ].copy()
                
                if temp_df.empty:
                    continue
                
                # Select `rho_turn_uOhm_cm2 ≈ 10000` (for heatmaps)
                RHO_TURN_FOR_HEATMAP = 10000.0  # μΩ·cm²
                if 'rho_turn_uOhm_cm2' in temp_df.columns:
                    # Find the value closest to 10000 (within tolerance)
                    available_rhot = temp_df['rho_turn_uOhm_cm2'].unique()
                    if len(available_rhot) > 0:
                        closest_rhot = available_rhot[np.argmin(np.abs(available_rhot - RHO_TURN_FOR_HEATMAP))]
                        temp_df = temp_df[
                            np.isclose(temp_df['rho_turn_uOhm_cm2'], closest_rhot, rtol=0.01)
                        ].copy()
                
                if temp_df.empty:
                    continue
                
                # Filter data: restrict to the parameter ranges used in 8.0
                # Ensure Npw lies in `NPW_SCAN_VALUES`
                if 'Npw' in temp_df.columns:
                    npw_array = np.array(NPW_SCAN_VALUES)
                    temp_df = temp_df[temp_df['Npw'].isin(npw_array)].copy()
                
                # Ensure `R_joint_nOhm` lies in `R_JOINT_SCAN_VALUES`
                # (same as 8.0, in nOhm).
                if 'R_joint_nOhm' in temp_df.columns:
                    r_joint_nohm_array = np.array(R_JOINT_SCAN_VALUES, dtype=float)  # nOhm
                    # Use 3‑decimal‑place comparison (to mitigate float issues)
                    temp_df = temp_df[
                        temp_df['R_joint_nOhm'].apply(
                            lambda x: any(match_r_joint_by_significant_digits(x, expected_r, n_digits=3)
                                         for expected_r in r_joint_nohm_array)
                        )
                    ].copy()
                
                if temp_df.empty:
                    continue
                
                # Deduplicate (Npw, R_joint_nOhm) combinations
                duplicate_mask = temp_df.duplicated(subset=['Npw', 'R_joint_nOhm'], keep='first')
                if duplicate_mask.any():
                    temp_df = temp_df.drop_duplicates(subset=['Npw', 'R_joint_nOhm'], keep='first')
                
                # Prepare data in the format expected by the plotting helpers
                temp_df_plot = temp_df.copy()
                if 'R_joint_nOhm' in temp_df_plot.columns:
                    temp_df_plot['R_joint'] = temp_df_plot['R_joint_nOhm'] * 1e-9  # nΩ -> Ω (plotting helpers expect Ω)
                
                # Construct the complete arrays used by 8.0.
                # Note: `R_JOINT_SCAN_VALUES` is in nOhm; convert to Ohm here.
                # Plotting helpers will again use these for interpolation & log‑scale.
                npw_array = np.array(NPW_SCAN_VALUES)
                r_joint_array = np.array(R_JOINT_SCAN_VALUES, dtype=float) * 1e-9  # convert to Ω for grid/interpolation
                
                # Plot `r_cryo_re` heatmap (parasitic ratio)
                if global_norm_parasitic is not None and 'r_cryo_re' in temp_df_plot.columns:
                    # Ensure full grid over NPW_SCAN_VALUES × R_JOINT_SCAN_VALUES_PLOT.
                    # The plotting helper will internally use these for interpolation.
                    complete_df_parasitic = ensure_complete_data_grid(
                        temp_df_plot,
                        npw_array,
                        r_joint_array,
                        'r_cryo_re'
                    )
                    
                    # Data‑completeness diagnostic
                    total_points = len(complete_df_parasitic)
                    valid_points = complete_df_parasitic['r_cryo_re'].notna().sum()
                    missing_ratio = 1.0 - (valid_points / total_points) if total_points > 0 else 1.0
                    
                    if missing_ratio > 0.1:  # more than 10% points missing
                        print(f"  [Warning] {scenario} - {temp} K {cool}: `r_cryo_re` data incomplete.")
                        print(f"            Missing points: {total_points - valid_points}/{total_points} "
                              f"({missing_ratio*100:.1f}%).")
                        print("            This may produce large blank regions in the heatmap.")

                    # Rename column to match plotting helper expectations
                    complete_df_parasitic['r_parasitic_pct'] = complete_df_parasitic['r_cryo_re']
                    
                    # Plot with the current colour scheme
                    plot_parasitic_heatmap_single(
                        df=complete_df_parasitic,
                        output_dir=output_dir,
                        cmap=cmap_name,
                        norm=global_norm_parasitic,
                        scenario=scenario,
                        temperature_K=temp,
                        filename_suffix=cmap_name,
                        use_stroke=USE_STROKE
                    )
                
                # Plot `delta_LCOE_min` heatmap
                if delta_norm is not None and 'delta_LCOE_min_USD_per_MWh' in temp_df_plot.columns:
                    # Ensure full grid over NPW_SCAN_VALUES × R_JOINT_SCAN_VALUES_PLOT
                    complete_df_delta = ensure_complete_data_grid(
                        temp_df_plot,
                        npw_array,
                        r_joint_array,
                        'delta_LCOE_min_USD_per_MWh'
                    )
                    
                    # Data‑completeness diagnostic
                    total_points = len(complete_df_delta)
                    valid_points = complete_df_delta['delta_LCOE_min_USD_per_MWh'].notna().sum()
                    missing_ratio = 1.0 - (valid_points / total_points) if total_points > 0 else 1.0
                    
                    if missing_ratio > 0.1:  # more than 10% points missing
                        print(f"  [Warning] {scenario} - {temp} K {cool}: `delta_LCOE_min` data incomplete.")
                        # Log missing (npw, r_joint) pairs
                        missing_rows = complete_df_delta[complete_df_delta['delta_LCOE_min_USD_per_MWh'].isna()]
                        for _, row in missing_rows.iterrows():
                            # Npw as integer; R_joint in nOhm with 4‑sig‑fig formatting
                            npw_i = int(round(float(row["Npw"])))
                            r_nohm = float(row["R_joint"]) * 1e9  # Ω -> nΩ
                            r_nohm_4sig = f"{r_nohm:.4g}"
                            print(f"            Missing combination: npw={npw_i}, R_joint_nOhm≈{r_nohm_4sig}")
                        print(f"            Missing points: {total_points - valid_points}/{total_points} "
                              f"({missing_ratio*100:.1f}%).")
                        print("            This may produce large blank regions in the heatmap.")

                    # Clip data to the plotting bound
                    if delta_bound is not None and np.isfinite(delta_bound):
                        feasible_mask = complete_df_delta['delta_LCOE_min_USD_per_MWh'].notna()
                        complete_df_delta['delta_lcoe_min_plot_$/MWh'] = complete_df_delta['delta_LCOE_min_USD_per_MWh'].copy()
                        complete_df_delta.loc[feasible_mask, 'delta_lcoe_min_plot_$/MWh'] = (
                            complete_df_delta.loc[feasible_mask, 'delta_LCOE_min_USD_per_MWh'].clip(-delta_bound, delta_bound)
                        )
                    else:
                        complete_df_delta['delta_lcoe_min_plot_$/MWh'] = complete_df_delta['delta_LCOE_min_USD_per_MWh']
                    
                    # Keep behaviour identical to 8.0: `auto_generate_levels=True`
                    # lets the helper choose contour levels automatically.
                    plot_delta_lcoe_heatmap_single(
                        df=complete_df_delta,
                        output_dir=output_dir,
                        cmap=cmap_name,
                        norm=delta_norm,
                        scenario=scenario,
                        temperature_K=temp,
                        baseline_point=None,  # do not show an explicit baseline point
                        value_col='delta_lcoe_min_plot_$/MWh',
                        coolant=cool,
                        contour_levels=None,  # keep identical behaviour to 8.0
                        auto_generate_levels=True,  # keep 8.0 behaviour: auto-generate contour levels
                        filename_suffix=cmap_name,
                        use_stroke=USE_STROKE
                    )
                    
                    # Save colourbar (per colour scheme)
                    if delta_norm is not None and delta_min is not None and delta_max is not None:
                        save_delta_lcoe_heatmap_colorbar(
                            output_dir=output_dir,
                            cmap=cmap_name,
                            norm=delta_norm,
                            data_min=delta_min,
                            data_max=delta_max,
                            label=r"$\Delta$LCOE (relative to scenario minimum) ($/\mathrm{MWh}$)",
                            is_global_ref=False,
                            filename_suffix=cmap_name
                        )
        
        # Save shared colourbars (per colour scheme)
        if global_norm_parasitic is not None:
            save_parasitic_heatmap_colorbar(
                output_dir=cmap_scan_usd_dir,
                cmap=cmap_name,
                norm=global_norm_parasitic,
                filename_suffix=cmap_name
            )
        
        if delta_norm is not None:
            save_delta_lcoe_heatmap_colorbar(
                output_dir=cmap_scan_usd_dir,
                cmap=cmap_name,
                norm=delta_norm,
                data_min=delta_min,
                data_max=delta_max,
                filename_suffix=cmap_name
            )
        
        print(f"[OK] All USD heatmaps generated for colour scheme {cmap_name}, saved under: {cmap_scan_usd_dir}")

    print(f"\n[OK] All USD heatmaps successfully generated.")

    # 10. Optionally generate CNY versions
    if enable_cny:
        print("\n" + "=" * 60)
        print("--- Generating CNY versions of the heatmaps ---")
        print("=" * 60)

        # Compute `delta_LCOE_min` in CNY
        df_cny = df.copy()
        if 'delta_LCOE_min_USD_per_MWh' in df_cny.columns:
            # Convert USD/MWh to CNY/kWh (i.e. RMB per kWh).
            # 1 MWh = 1000 kWh, so divide by 1000.
            df_cny['delta_LCOE_min_CNY_per_kWh'] = df_cny['delta_LCOE_min_USD_per_MWh'] * USD_TO_CNY_EXCHANGE_RATE / 1000.0
        
        # Global norm for CNY version
        if 'delta_LCOE_min_CNY_per_kWh' in df_cny.columns:
            # Mask infeasible points
            if 'r_cryo_re' in df_cny.columns:
                bad = df_cny['r_cryo_re'] > DELTA_LCOE_MASK_PARASITIC_PCT
                df_cny.loc[bad, 'delta_LCOE_min_CNY_per_kWh'] = np.nan
            
            delta_series_cny = (
                df_cny['delta_LCOE_min_CNY_per_kWh']
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            
            delta_norm_cny = None
            delta_bound_cny = None
            delta_min_cny = None
            delta_max_cny = None
            
            if not delta_series_cny.empty:
                delta_min_cny = float(delta_series_cny.min())
                delta_max_cny = float(delta_series_cny.max())
                
                if delta_max_cny > delta_min_cny:
                    linthresh = cfg.DELTA_LCOE_LINTHRESH_CNY
                    linscale = cfg.DELTA_LCOE_LINSCALE_CNY
                    
                    if delta_min_cny < 0 and delta_max_cny > 0:
                        norm_vmin = delta_min_cny
                        norm_vmax = delta_max_cny
                    elif delta_min_cny >= 0:
                        norm_vmin = 0.0
                        norm_vmax = delta_max_cny
                    else:
                        norm_vmin = delta_min_cny
                        norm_vmax = 0.0
                    
                    delta_norm_cny = mpl.colors.SymLogNorm(
                        linthresh=linthresh,
                        linscale=linscale,
                        vmin=norm_vmin,
                        vmax=norm_vmax
                    )
                    print(f"[Info] ΔLCOE_min (CNY) colourbar range: "
                          f"[{delta_min_cny:.3g}, {delta_max_cny:.3g}] (CNY/kWh)")
                    
                    all_finite_cny = delta_series_cny.replace([np.inf, -np.inf], np.nan).dropna().values
                    delta_bound_cny = float(np.max(np.abs(all_finite_cny)))
        
        # Generate CNY heatmaps for each colour scheme
        for cmap_name in cfg.color_schemes:
            print(f"\nProcessing CNY colour scheme: {cmap_name}")
            # Directory: `outputs/figures/economic/{cmap_name}/parasitic_ratio/CNY`
            cmap_base_dir = Path(cfg.ECONOMIC_FIGURES_DIR) / cmap_name
            cmap_scan_base_dir = cmap_base_dir / cfg.PARASITIC_RATIO_OUTPUT_DIR
            cmap_scan_cny_dir = cmap_scan_base_dir / cfg.PARASITIC_RATIO_CNY_DIR
            cmap_scan_cny_dir.mkdir(exist_ok=True, parents=True)
            
            # Per‑scenario heatmaps (CNY)
            for scenario in scenarios:
                scenario_df_cny = df_cny[df_cny['scenario'] == scenario].copy()
                
                if scenario_df_cny.empty:
                    continue
                
                output_dir_cny = cmap_scan_cny_dir / scenario
                output_dir_cny.mkdir(exist_ok=True, parents=True)
                
                # Per temperature–coolant combination
                for temp, cool in temp_coolant_pairs:
                    temp_df_cny = scenario_df_cny[
                        (scenario_df_cny['Top_K'] == temp) &
                        (scenario_df_cny['coolant'] == cool)
                    ].copy()
                    
                    if temp_df_cny.empty:
                        continue
                    
                    # Filter `rho_turn_uOhm_cm2 ≈ 10000` (for heatmaps)
                    RHO_TURN_FOR_HEATMAP = 10000.0  # μΩ·cm²
                    if 'rho_turn_uOhm_cm2' in temp_df_cny.columns:
                        available_rhot = temp_df_cny['rho_turn_uOhm_cm2'].unique()
                        if len(available_rhot) > 0:
                            closest_rhot = available_rhot[np.argmin(np.abs(available_rhot - RHO_TURN_FOR_HEATMAP))]
                            temp_df_cny = temp_df_cny[
                                np.isclose(temp_df_cny['rho_turn_uOhm_cm2'], closest_rhot, rtol=0.01)
                            ].copy()
                    
                    if temp_df_cny.empty:
                        continue
                    
                    # Filter parameter ranges
                    if 'Npw' in temp_df_cny.columns:
                        npw_array = np.array(NPW_SCAN_VALUES)
                        temp_df_cny = temp_df_cny[temp_df_cny['Npw'].isin(npw_array)].copy()
                    
                    if 'R_joint_nOhm' in temp_df_cny.columns:
                        r_joint_nohm_array = np.array(R_JOINT_SCAN_VALUES, dtype=float)  # nOhm
                        temp_df_cny = temp_df_cny[
                            temp_df_cny['R_joint_nOhm'].apply(
                                lambda x: any(match_r_joint_by_significant_digits(x, expected_r, n_digits=3)
                                             for expected_r in r_joint_nohm_array)
                            )
                        ].copy()
                    
                    if temp_df_cny.empty:
                        continue
                    
                    # Deduplicate (Npw, R_joint_nOhm)
                    duplicate_mask = temp_df_cny.duplicated(subset=['Npw', 'R_joint_nOhm'], keep='first')
                    if duplicate_mask.any():
                        temp_df_cny = temp_df_cny.drop_duplicates(subset=['Npw', 'R_joint_nOhm'], keep='first')
                    
                    # Prepare data for plotting helper
                    temp_df_plot_cny = temp_df_cny.copy()
                    if 'R_joint_nOhm' in temp_df_plot_cny.columns:
                        temp_df_plot_cny['R_joint'] = temp_df_plot_cny['R_joint_nOhm'] * 1e-9  # nΩ -> Ω
                    
                    # Complete arrays from configuration
                    npw_array = np.array(NPW_SCAN_VALUES)
                    r_joint_array = np.array(R_JOINT_SCAN_VALUES, dtype=float) * 1e-9  # convert to Ω
                    
                    # Plot r_cryo_re heatmap (same as USD version)
                    if global_norm_parasitic is not None and 'r_cryo_re' in temp_df_plot_cny.columns:
                        complete_df_parasitic_cny = ensure_complete_data_grid(
                            temp_df_plot_cny,
                            npw_array,
                            r_joint_array,
                            'r_cryo_re'
                        )
                        
                        complete_df_parasitic_cny['r_parasitic_pct'] = complete_df_parasitic_cny['r_cryo_re']
                        
                        plot_parasitic_heatmap_single(
                            df=complete_df_parasitic_cny,
                            output_dir=output_dir_cny,
                            cmap=cmap_name,
                            norm=global_norm_parasitic,
                            scenario=scenario,
                            temperature_K=temp,
                            filename_suffix=cmap_name,
                            use_stroke=USE_STROKE
                        )
                    
                    # Plot ΔLCOE_min heatmap (CNY version)
                    if delta_norm_cny is not None and 'delta_LCOE_min_CNY_per_kWh' in temp_df_plot_cny.columns:
                        complete_df_delta_cny = ensure_complete_data_grid(
                            temp_df_plot_cny,
                            npw_array,
                            r_joint_array,
                            'delta_LCOE_min_CNY_per_kWh'
                        )
                        
                        # Clip to plotting bound
                        if delta_bound_cny is not None and np.isfinite(delta_bound_cny):
                            feasible_mask = complete_df_delta_cny['delta_LCOE_min_CNY_per_kWh'].notna()
                            complete_df_delta_cny['delta_lcoe_min_plot_CNY_kWh'] = complete_df_delta_cny['delta_LCOE_min_CNY_per_kWh'].copy()
                            complete_df_delta_cny.loc[feasible_mask, 'delta_lcoe_min_plot_CNY_kWh'] = (
                                complete_df_delta_cny.loc[feasible_mask, 'delta_LCOE_min_CNY_per_kWh'].clip(-delta_bound_cny, delta_bound_cny)
                            )
                        else:
                            complete_df_delta_cny['delta_lcoe_min_plot_CNY_kWh'] = complete_df_delta_cny['delta_LCOE_min_CNY_per_kWh']
                        
                        plot_delta_lcoe_heatmap_single(
                            df=complete_df_delta_cny,
                            output_dir=output_dir_cny,
                            cmap=cmap_name,
                            norm=delta_norm_cny,
                            scenario=scenario,
                            temperature_K=temp,
                            baseline_point=None,
                            value_col='delta_lcoe_min_plot_CNY_kWh',
                            coolant=cool,
                            contour_levels=None,
                            auto_generate_levels=True,
                            filename_suffix=cmap_name,
                            use_stroke=USE_STROKE
                        )
                    
            # Save shared colourbars (per colour scheme)
            if global_norm_parasitic is not None:
                save_parasitic_heatmap_colorbar(
                    output_dir=cmap_scan_cny_dir,
                    cmap=cmap_name,
                    norm=global_norm_parasitic,
                    filename_suffix=cmap_name
                )
            
            if delta_norm_cny is not None:
                save_delta_lcoe_heatmap_colorbar(
                    output_dir=cmap_scan_cny_dir,
                    cmap=cmap_name,
                    norm=delta_norm_cny,
                    data_min=delta_min_cny,
                    data_max=delta_max_cny,
                    label=r"$\Delta$LCOE (CNY/kWh)",
                    filename_suffix=cmap_name
                )
            
            print(f"[OK] All CNY heatmaps generated for colour scheme {cmap_name}, "
                  f"saved under: {cmap_scan_cny_dir}")

        print(f"\n[OK] All CNY heatmaps successfully generated.")


# =============================================================================
# Script entry point
# =============================================================================

if __name__ == "__main__":
    print("Starting script to generate heatmaps from scan_full_grid_tidy.csv ...")

    # Optionally restrict the scenarios and temperature–coolant pairs.
    # If None, they are inferred from the data.
    SCENARIOS_TO_PLOT = None  # e.g. ['S1', 'S2', 'S3']
    TEMP_COOLANT_PAIRS_TO_PLOT = None  # e.g. [(4.2, 'He'), (10.0, 'He'), (20.0, 'He'), (20.0, 'H2')]

    # Create output directories
    SCAN_BASE_DIR.mkdir(exist_ok=True, parents=True)
    SCAN_USD_DIR.mkdir(exist_ok=True, parents=True)
    if ENABLE_CNY:
        SCAN_CNY_DIR.mkdir(exist_ok=True, parents=True)
    
    # Run plotting
    plot_from_scan_full_grid(
        scenarios=SCENARIOS_TO_PLOT,
        temp_coolant_pairs=TEMP_COOLANT_PAIRS_TO_PLOT,
        enable_cny=ENABLE_CNY
    )
    
    print("\nScript finished.")
