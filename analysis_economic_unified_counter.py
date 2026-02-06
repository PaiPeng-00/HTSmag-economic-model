# 8.0_run_economic_analysis_unified.py (unified economic‑analysis script)
# Combines the functionality of 8.1, 8.2 and 8.3 into a single script,
# supporting LCOE analysis in both USD and CNY units.
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple

from pandas.core.nanops import F

# --- Import economic model functions ---
from model_economic import define_parameters, compute_case
import config as cfg

# --- Import plotting helpers from the unified plotting library ---
from plot_library import (
    plot_combined_charts,
    plot_parasitic_heatmap_single,
    save_parasitic_heatmap_colorbar,
    plot_parasitic_power_vs_temp,
    save_color_legend,
    save_hatch_legend,
    plot_cryo_mode_grid,
    plot_delta_lcoe_heatmap_single,
    save_delta_lcoe_heatmap_colorbar,
    filter_contour_levels_to_target_count,
)
import matplotlib as mpl

# =============================================================================
# 1. Configuration parameters (from config.py)
# =============================================================================
DELTA_LCOE_CMAP = cfg.cmap_parasitic_ratio
DELTA_LCOE_MASK_PARASITIC_PCT = cfg.DELTA_LCOE_MASK_PARASITIC_PCT
DELTA_LCOE_ABS_PCTL = cfg.DELTA_LCOE_ABS_PCTL
USD_TO_CNY_EXCHANGE_RATE = cfg.USD_TO_CNY_EXCHANGE_RATE

# Parameter‑scan configuration
# Note: R_JOINT_SCAN_VALUES is in nOhm and is only used for scanning/recording;
# it is converted to Ohm (×1e‑9) whenever passed into the economic model.
R_JOINT_SCAN_VALUES = cfg.R_JOINT_SCAN_VALUES  # nOhm
NPW_SCAN_VALUES = cfg.NPW_SCAN_VALUES
SCAN_TEMP_COOLANT_PAIRS = cfg.SCAN_TEMP_COOLANT_PAIRS
ECONOMIC_OPERATING_CONDITIONS = cfg.ECONOMIC_OPERATING_CONDITIONS

# Output‑directory configuration
# Figure directory (figures): SVG outputs, consistent with 9.0_stitch_econimic_svgs.py
SCAN_BASE_DIR_FIG = Path(cfg.ECONOMIC_FIGURES_DIR) / cfg.PARASITIC_RATIO_OUTPUT_DIR
SCAN_USD_DIR = SCAN_BASE_DIR_FIG / cfg.PARASITIC_RATIO_USD_DIR
SCAN_CNY_DIR = SCAN_BASE_DIR_FIG / cfg.PARASITIC_RATIO_CNY_DIR
# Table directory (tables): Excel data files
SCAN_DATA_DIR = Path(cfg.ECONOMIC_OUTPUT_DIR) / cfg.PARASITIC_RATIO_OUTPUT_DIR / cfg.PARASITIC_RATIO_DATA_DIR
ECONOMIC_OUTPUT_DIR = Path(cfg.ECONOMIC_OUTPUT_DIR) / cfg.COST_OUTPUT_DIR
FINAL_ECONOMICS_FILE = ECONOMIC_OUTPUT_DIR / cfg.ECONOMIC_SUMMARY_FILE

# =============================================================================
# 2. Reusable helper functions
# =============================================================================

def record_heatmap_axis_distribution(
    readme_path: Path,
    image_name: str,
    scenario: str,
    temperature_K: float,
    coolant: Optional[str],
    df: pd.DataFrame,
    x_col: str = "R_joint",
    y_col: str = "Npw",
    value_col: Optional[str] = None
):
    """
    Record the axis‑value distributions used in a heatmap into a README.md
    file for documentation.

    Args:
        readme_path: path to the README.md file
        image_name: name of the image (identifier)
        scenario: scenario name
        temperature_K: temperature value (K)
        coolant: coolant type (optional)
        df: data DataFrame
        x_col: x‑axis column name (default "R_joint")
        y_col: y‑axis column name (default "Npw")
        value_col: value column name (optional, to record value ranges)
    """
    # Ensure README.md exists
    readme_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract x‑ and y‑axis data
    if x_col in df.columns:
        x_data = df[x_col].dropna()
        if len(x_data) > 0:
            x_min = float(x_data.min())
            x_max = float(x_data.max())
            x_unique = sorted(x_data.unique())
            x_count = len(x_unique)
        else:
            x_min = x_max = np.nan
            x_unique = []
            x_count = 0
    else:
        x_min = x_max = np.nan
        x_unique = []
        x_count = 0
    
    if y_col in df.columns:
        y_data = df[y_col].dropna()
        if len(y_data) > 0:
            y_min = float(y_data.min())
            y_max = float(y_data.max())
            y_unique = sorted(y_data.unique())
            y_count = len(y_unique)
        else:
            y_min = y_max = np.nan
            y_unique = []
            y_count = 0
    else:
        y_min = y_max = np.nan
        y_unique = []
        y_count = 0
    
    # Extract value range (if value_col is provided)
    value_info = ""
    if value_col and value_col in df.columns:
        value_data = df[value_col].dropna()
        if len(value_data) > 0:
            value_min = float(value_data.min())
            value_max = float(value_data.max())
            value_info = f"\n- **Value range**: [{value_min:.4g}, {value_max:.4g}]"
    
    # Construct label
    if coolant:
        label = f"{temperature_K}K_{coolant}"
    else:
        label = f"{temperature_K}K"
    
    # Format x‑axis data (convert R_joint to nOhm for display)
    if x_col == "R_joint":
        x_min_display = x_min * 1e9 if np.isfinite(x_min) else np.nan  # convert to nOhm
        x_max_display = x_max * 1e9 if np.isfinite(x_max) else np.nan
        x_unit = "nOhm"
        x_name = "Inter‑coil joint resistance (R_joint)"
    else:
        x_min_display = x_min
        x_max_display = x_max
        x_unit = ""
        x_name = x_col
    
    # Format y‑axis label
    if y_col == "Npw":
        y_name = "Number of parallel stacks (Npw)"
    else:
        y_name = y_col
    
    # Format unique‑value lists (with length limits)
    def format_unique_values(values, max_display=20):
        if len(values) == 0:
            return "[]"
        if len(values) <= max_display:
            if len(values) <= 10:
                return str([f"{v:.4g}" if isinstance(v, float) else str(v) for v in values])
            else:
                return f"[{values[0]:.4g}, ..., {values[-1]:.4g}] (total {len(values)})"
        else:
            return f"[{values[0]:.4g}, ..., {values[-1]:.4g}] (total {len(values)}, first {max_display} shown: {[f'{v:.4g}' if isinstance(v, float) else str(v) for v in values[:max_display]]})"
    
    x_unique_str = format_unique_values(x_unique)
    y_unique_str = format_unique_values(y_unique)
    
    # Build record content
    record_content = f"""
### {image_name} - {scenario} - {label}

**X‑axis ({x_name}):**
- **Range**: [{x_min_display:.4g}, {x_max_display:.4g}] {x_unit}
- **Unique count**: {x_count}
- **Unique values**: {x_unique_str}

**Y‑axis ({y_name}):**
- **Range**: [{y_min:.4g}, {y_max:.4g}]
- **Unique count**: {y_count}
- **Unique values**: {y_unique_str}{value_info}

---
"""
    
    # Append to README.md
    with open(readme_path, 'a', encoding='utf-8') as f:
        f.write(record_content)


def format_array_for_print(arr: np.ndarray) -> str:
    """
    Format a NumPy array for printing to avoid scientific notation.
    """
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


def _round_contour_levels(levels: np.ndarray) -> np.ndarray:
    """
    Round contour levels so that they have at most one decimal place (or
    a small number of significant digits depending on magnitude).
    """
    if len(levels) == 0:
        return levels
    
    rounded_levels = []
    for level in levels:
        abs_level = abs(level)
        if abs_level == 0:
            rounded_levels.append(0.0)
        elif abs_level >= 10:
            rounded_levels.append(float(round(level)))
        elif abs_level >= 1:
            rounded_levels.append(round(level, 1))
        elif abs_level >= 0.1:
            rounded_levels.append(round(level, 1))
        else:
            magnitude = 10 ** np.floor(np.log10(abs_level))
            rounded_val = round(level / magnitude) * magnitude
            if abs(rounded_val) < 0.01:
                rounded_levels.append(round(rounded_val, 3))
            else:
                rounded_levels.append(round(rounded_val, 2))
    
    return np.array(rounded_levels)


def calculate_lcoe_contour_levels(
    all_finite: np.ndarray, 
    delta_bound: float,
    unit: str = 'USD'  # 'USD' or 'CNY'
) -> np.ndarray:
    """
    Compute an array of LCOE contour levels.
    
    Args:
        all_finite: array of finite data values (already clipped to delta_bound)
        delta_bound: clipping bound for data
        unit: 'USD' or 'CNY'
    
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
            
            # Treat positive and negative values separately
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
            
            # Positive part
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
            
            # Negative part
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
            
            # Combine positive and negative contour levels
            if len(neg_levels) > 0 and len(pos_levels) > 0:
                contour_levels = np.concatenate([neg_levels[::-1], [0.0], pos_levels])
            elif len(neg_levels) > 0:
                contour_levels = np.concatenate([neg_levels[::-1], [0.0]])
            elif len(pos_levels) > 0:
                contour_levels = np.concatenate([[0.0], pos_levels])
            else:
                contour_levels = np.array([0.0])
            
            contour_levels = np.sort(contour_levels)
    
    # Enforce minimum spacing between neighbouring levels
    filtered_levels = [contour_levels[0]]
    for i in range(1, len(contour_levels)):
        if abs(contour_levels[i] - filtered_levels[-1]) >= min_spacing:
            filtered_levels.append(contour_levels[i])
    
    return np.array(filtered_levels)


def select_subplot_contour_levels(
    z_min: float,
    z_max: float,
    master_levels: np.ndarray,
    target_count: int = 5,
    min_allowed: Optional[float] = None
) -> np.ndarray:
    """
    Select 5–6 suitable contour levels for a single subplot from the
    master contour‑level array.
    """
    if z_min >= z_max or not np.isfinite(z_min) or not np.isfinite(z_max):
        if 0.0 in master_levels:
            return np.array([0.0])
        return np.array([])
    
    if min_allowed is not None and min_allowed >= 0:
        lower_bound = max(z_min, min_allowed)
        upper_bound = z_max
    else:
        margin = (z_max - z_min) * 0.1
        lower_bound = z_min - margin
        upper_bound = z_max + margin
    
    valid_levels = master_levels[
        (master_levels >= lower_bound) &
        (master_levels <= upper_bound)
    ]
    
    if len(valid_levels) == 0:
        if z_min <= 0 <= z_max:
            return np.array([0.0])
        elif z_max < 0:
            closest_idx = np.argmin(np.abs(master_levels - z_max))
            return np.array([master_levels[closest_idx]])
        else:
            closest_idx = np.argmin(np.abs(master_levels - z_min))
            return np.array([master_levels[closest_idx]])
    
    if 5 <= len(valid_levels) <= target_count + 1:
        if min_allowed is not None and min_allowed >= 0:
            valid_levels = valid_levels[(valid_levels >= z_min) & (valid_levels <= z_max)]
        return valid_levels
    
    if len(valid_levels) > target_count:
        has_zero = 0.0 in valid_levels and z_min < 0 < z_max
        selected = []
        
        if has_zero:
            selected.append(0.0)
            neg_levels = valid_levels[valid_levels < 0]
            pos_levels = valid_levels[valid_levels > 0]
            
            remaining = target_count - 1
            n_neg = max(1, remaining // 2)
            n_pos = max(1, remaining - n_neg)
            
            if len(neg_levels) > 0:
                if len(neg_levels) <= n_neg:
                    selected.extend(neg_levels)
                else:
                    indices = np.linspace(0, len(neg_levels) - 1, n_neg, dtype=int)
                    selected.extend(neg_levels[indices])
            
            if len(pos_levels) > 0:
                if len(pos_levels) <= n_pos:
                    selected.extend(pos_levels)
                else:
                    indices = np.linspace(0, len(pos_levels) - 1, n_pos, dtype=int)
                    selected.extend(pos_levels[indices])
        else:
            indices = np.linspace(0, len(valid_levels) - 1, target_count, dtype=int)
            selected = valid_levels[indices].tolist()
        
        selected_array = np.sort(np.array(selected))
        if min_allowed is not None and min_allowed >= 0:
            selected_array = selected_array[(selected_array >= z_min) & (selected_array <= z_max)]
        return selected_array
    
    return valid_levels


def add_delta_lcoe_min_columns(all_df: pd.DataFrame, scenarios: list) -> pd.DataFrame:
    """
    Add columns to all_df giving ΔLCOE relative to the minimum LCOE
    within each scenario.
    """
    print("\n--- Computing per‑scenario ΔLCOE relative to the minimum ---")
    for scenario in scenarios:
        scenario_mask = all_df["scenario"] == scenario
        if "lcoe_magnet_only_$/MWh" not in all_df.columns:
            continue
        
        scenario_lcoe = all_df.loc[scenario_mask, "lcoe_magnet_only_$/MWh"]
        if "r_parasitic_pct" in all_df.columns:
            feasible_mask_scenario = (
                scenario_lcoe.notna()
                & (all_df.loc[scenario_mask, "r_parasitic_pct"] <= DELTA_LCOE_MASK_PARASITIC_PCT)
            )
        else:
            feasible_mask_scenario = scenario_lcoe.notna()
        
        if feasible_mask_scenario.any():
            scenario_min_lcoe = float(scenario_lcoe[feasible_mask_scenario].min())
            print(f"[{scenario}] Scenario minimum LCOE: {scenario_min_lcoe:.3g} ($/MWh)")
            
            all_df.loc[scenario_mask, "delta_lcoe_min_$/MWh"] = (
                all_df.loc[scenario_mask, "lcoe_magnet_only_$/MWh"] - scenario_min_lcoe
            )
            all_df.loc[scenario_mask, "delta_lcoe_min_CNY_per_kWh"] = (
                all_df.loc[scenario_mask, "delta_lcoe_min_$/MWh"] * USD_TO_CNY_EXCHANGE_RATE / 1000
            )
        else:
            print(f"[{scenario}] WARNING: no feasible points in this scenario; cannot compute minimum.")
            all_df.loc[scenario_mask, "delta_lcoe_min_$/MWh"] = np.nan
            all_df.loc[scenario_mask, "delta_lcoe_min_CNY_per_kWh"] = np.nan
    
    return all_df


def convert_usd_to_cny_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert LCOE‑related columns from USD/MWh to CNY/kWh.
    """
    if "lcoe_magnet_only_$/MWh" in df.columns:
        if "lcoe_magnet_only_CNY_per_kWh" not in df.columns:
            df["lcoe_magnet_only_CNY_per_kWh"] = (
                df["lcoe_magnet_only_$/MWh"] * USD_TO_CNY_EXCHANGE_RATE / 1000
            )
    
    if "baseline_lcoe_$/MWh" in df.columns:
        if "baseline_lcoe_CNY_per_kWh" not in df.columns:
            df["baseline_lcoe_CNY_per_kWh"] = (
                df["baseline_lcoe_$/MWh"] * USD_TO_CNY_EXCHANGE_RATE / 1000
            )
    
    if "delta_lcoe_$/MWh" in df.columns:
        if "delta_lcoe_CNY_per_kWh" not in df.columns:
            df["delta_lcoe_CNY_per_kWh"] = (
                df["delta_lcoe_$/MWh"] * USD_TO_CNY_EXCHANGE_RATE / 1000
            )
    
    if "delta_lcoe_global_$/MWh" in df.columns:
        if "delta_lcoe_global_CNY_per_kWh" not in df.columns:
            df["delta_lcoe_global_CNY_per_kWh"] = (
                df["delta_lcoe_global_$/MWh"] * USD_TO_CNY_EXCHANGE_RATE / 1000
            )
    
    return df


# =============================================================================
# 3. Main execution functions
# =============================================================================

def perform_parasitic_power_scan(params: dict, scenarios: list, enable_lcoe: bool = True):
    """
    Perform the parameter scan and generate:
      1) r_parasitic_pct heatmaps (existing),
      2) delta_lcoe_$/MWh heatmaps (optional),
      3) delta_lcoe_global_$/MWh heatmaps (optional),
      4) delta_lcoe_min_$/MWh heatmaps (optional).
    """
    print("\n" + "="*50)
    print("--- Task 1: parameter scan (parasitic + ΔLCOE) ---")
    print("="*50)
    
    # Define baseline point (required in all cases)
    baseline_Npw = cfg.Npw_TARGET
    baseline_Rj = cfg.R_p2p_joint_TARGET
    
    # Check whether Excel data files already exist
    all_excel_exist = True
    all_scenario_dfs = []
    for scenario in scenarios:
        output_dir = SCAN_DATA_DIR / scenario
        excel_path = output_dir / f"scan_data_with_delta_lcoe_{scenario}.xlsx"
        if excel_path.exists():
            print(f"Detected Excel data file for scenario {scenario}; reading: {excel_path}")
            scenario_df = pd.read_excel(excel_path)
            all_scenario_dfs.append(scenario_df)
        else:
            print(f"Excel data file for scenario {scenario} not found: {excel_path}; will perform full computation.")
            all_excel_exist = False
            break
    
    if all_excel_exist and len(all_scenario_dfs) == len(scenarios):
        print("\nAll scenario Excel files found; skipping recomputation, reading data and plotting directly...")
        all_df = pd.concat(all_scenario_dfs, ignore_index=True)
        if "delta_lcoe_min_$/MWh" not in all_df.columns:
            all_df = add_delta_lcoe_min_columns(all_df, scenarios)
        else:
            print("Excel data already contains ΔLCOE_min columns; reusing them directly.")
        skip_cryo_grid = True
    else:
        print("\nPerforming full data computation...")
        skip_cryo_grid = False
        all_rows = []
        cryo_rows = []
        
        # Compute global reference LCOE (20K‑H₂, S3)
        global_reference_scenario = cfg.GLOBAL_REFERENCE_SCENARIO
        global_reference_temp = cfg.GLOBAL_REFERENCE_TEMP
        global_reference_coolant = cfg.GLOBAL_REFERENCE_COOLANT
        global_reference_year = params['tech_scenario_to_years'][global_reference_scenario]
        
        global_ref_res, _ = compute_case(
            tech_scenario=global_reference_scenario,
            year=global_reference_year,
            temperature_K=global_reference_temp,
            coolant=global_reference_coolant,
            Npw=baseline_Npw,
            R_p2p_joint=baseline_Rj,
            p=params
        )
        global_reference_lcoe = np.nan
        if global_ref_res is not None:
            global_reference_lcoe = global_ref_res.get("lcoe_magnet_only_$/MWh", np.nan)
            print(f"Global reference ({global_reference_temp}K-{global_reference_coolant}, {global_reference_scenario}) LCOE: {global_reference_lcoe:.3g} ($/MWh)")
        else:
            print("WARNING: failed to compute global‑reference LCOE")
        
        # Main computation loop
        for scenario in scenarios:
            scenario_year = params['tech_scenario_to_years'][scenario]
            print(f"--- Computing parameter‑scan data for scenario {scenario} (year {scenario_year}) ---")
            for temp, cool in SCAN_TEMP_COOLANT_PAIRS:
                # First compute baseline LCOE
                base_res, _ = compute_case(
                    tech_scenario=scenario,
                    year=scenario_year,
                    temperature_K=temp,
                    coolant=cool,
                    Npw=baseline_Npw,
                    R_p2p_joint=baseline_Rj,
                    p=params
                )
                baseline_lcoe = np.nan
                if base_res is not None:
                    baseline_lcoe = base_res.get("lcoe_magnet_only_$/MWh", np.nan)
                
                for Npw in NPW_SCAN_VALUES:
                    for Rj_nohm in R_JOINT_SCAN_VALUES:  # scan values (nOhm)
                        Rj = float(Rj_nohm) * 1e-9  # convert to Ohm before passing into economic model
                        economic_result, _ = compute_case(
                            tech_scenario=scenario,
                            year=scenario_year,
                            temperature_K=temp,
                            coolant=cool,
                            Npw=Npw,
                            R_p2p_joint=Rj,
                            p=params
                        )
                        
                        if economic_result is None:
                            continue
                        
                        lcoe_B = economic_result.get("lcoe_magnet_only_$/MWh", np.nan)
                        
                        # Compute various ΔLCOE metrics
                        if np.isfinite(lcoe_B) and np.isfinite(baseline_lcoe):
                            delta_lcoe = lcoe_B - baseline_lcoe
                        else:
                            delta_lcoe = np.nan
                        
                        if np.isfinite(lcoe_B) and np.isfinite(global_reference_lcoe):
                            delta_lcoe_global = lcoe_B - global_reference_lcoe
                        else:
                            delta_lcoe_global = np.nan
                        
                        # Compute values in CNY units
                        lcoe_cny_per_kwh = lcoe_B * USD_TO_CNY_EXCHANGE_RATE / 1000 if np.isfinite(lcoe_B) else np.nan
                        baseline_lcoe_cny_per_kwh = baseline_lcoe * USD_TO_CNY_EXCHANGE_RATE / 1000 if np.isfinite(baseline_lcoe) else np.nan
                        delta_lcoe_cny_per_kwh = delta_lcoe * USD_TO_CNY_EXCHANGE_RATE / 1000 if np.isfinite(delta_lcoe) else np.nan
                        delta_lcoe_global_cny_per_kwh = delta_lcoe_global * USD_TO_CNY_EXCHANGE_RATE / 1000 if np.isfinite(delta_lcoe_global) else np.nan
                        
                        all_rows.append({
                            "temperature_K": temp,
                            "coolant": cool,
                            "Npw": Npw,
                            # Record both units for convenient post‑processing/plotting:
                            "R_joint": Rj,            # Ohm
                            "R_joint_nOhm": Rj_nohm,  # nOhm
                            "scenario": scenario,
                            "r_parasitic_pct": economic_result.get("r_parasitic_pct", np.nan),
                            "lcoe_magnet_only_$/MWh": lcoe_B,
                            "baseline_lcoe_$/MWh": baseline_lcoe,
                            "delta_lcoe_$/MWh": delta_lcoe,
                            "delta_lcoe_global_$/MWh": delta_lcoe_global,
                            "lcoe_magnet_only_CNY_per_kWh": lcoe_cny_per_kwh,
                            "baseline_lcoe_CNY_per_kWh": baseline_lcoe_cny_per_kwh,
                            "delta_lcoe_CNY_per_kWh": delta_lcoe_cny_per_kwh,
                            "delta_lcoe_global_CNY_per_kWh": delta_lcoe_global_cny_per_kwh,
                        })
                        
                        # cryo_rows
                        for mode_key in ["static", "dwell", "prod"]:
                            cryo_key = {
                                "static": "cryo_power_static_W",
                                "dwell": "cryo_power_dwell_W",
                                "prod": "cryo_power_prod_W"
                            }[mode_key]
                            cryo_power = economic_result.get(cryo_key, np.nan)
                            cryo_rows.append({
                                "temperature_K": temp,
                                "coolant": cool,
                                "Npw": Npw,
                                "R_joint": Rj,
                                "scenario": scenario,
                                "mode": mode_key,
                                "cryo_power_W": cryo_power
                            })
        
        all_df = pd.DataFrame(all_rows)
        all_df = add_delta_lcoe_min_columns(all_df, scenarios)
    
    # Compute global norms for colourbars
    global_vmin = all_df["r_parasitic_pct"].min()
    global_vmax = all_df["r_parasitic_pct"].max()
    global_norm_parasitic = mpl.colors.LogNorm(
        vmin=float(f"{global_vmin:.10f}"),
        vmax=float(f"{global_vmax:.10f}")
    )
    print(f"Global colourbar range for parasitic fraction: {global_vmin:.3g}% - {global_vmax:.3g}%")
    
    # Global norm for ΔLCOE
    if enable_lcoe:
        bad = (all_df["r_parasitic_pct"] > DELTA_LCOE_MASK_PARASITIC_PCT)
        all_df.loc[bad, "delta_lcoe_$/MWh"] = np.nan
        all_df.loc[bad, "delta_lcoe_global_$/MWh"] = np.nan
        all_df.loc[bad, "delta_lcoe_min_$/MWh"] = np.nan
        print(f"Masked points with parasitic power > {DELTA_LCOE_MASK_PARASITIC_PCT}% as infeasible ({bad.sum()} points).")
        
        delta_series = (
            all_df["delta_lcoe_$/MWh"]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        
        delta_norm = None
        delta_bound = None
        delta_min = None
        delta_max = None
        global_contour_levels = None
        
        if not delta_series.empty:
            delta_bound = float(np.nanpercentile(np.abs(delta_series.values), DELTA_LCOE_ABS_PCTL))
            
            if delta_bound > 0:
                delta_min = float(delta_series.min())
                delta_max = float(delta_series.max())
                
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
                print(f"ΔLCOE colourbar range based on |Δ| P{DELTA_LCOE_ABS_PCTL}: ±{delta_bound:.3g} ($/MWh)")
                
                all_delta_clipped = all_df["delta_lcoe_$/MWh"].clip(-delta_bound, delta_bound)
                all_finite = all_delta_clipped.replace([np.inf, -np.inf], np.nan).dropna().values
                
                global_contour_levels = calculate_lcoe_contour_levels(all_finite, delta_bound, unit='USD')
    
    # Determine README.md path (used to record axis‑value distributions)
    readme_path = Path("outputs/figures/heatload/relative_heatload_grids/README.md")
    readme_path.parent.mkdir(parents=True, exist_ok=True)
    
    # If README.md does not exist, create initial English content
    if not readme_path.exists():
        from datetime import datetime
        current_date = datetime.now().strftime("%Y-%m-%d")
        initial_content = f"""# Economic heatmap data-distribution notes

This document records the axis-value distributions used in economic heatmaps.

**Generated on:** {current_date}

## Recorded distributions

"""
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(initial_content)
    
    # Generate heatmaps for each scenario
    for scenario in scenarios:
        print(f"--- Generating heatmaps for scenario {scenario} ---")
        scenario_df = all_df[all_df["scenario"] == scenario].copy()
        output_dir = SCAN_USD_DIR / scenario
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Save data if it was not already loaded from Excel
        if not (all_excel_exist and len(all_scenario_dfs) == len(scenarios)):
            data_output_dir = SCAN_DATA_DIR / scenario
            data_output_dir.mkdir(exist_ok=True, parents=True)
            excel_path = data_output_dir / f"scan_data_with_delta_lcoe_{scenario}.xlsx"
            scenario_df.to_excel(excel_path, index=False)
        
        # Compute scenario‑specific norms (for global‑reference and relative‑to‑minimum ΔLCOE)
        if enable_lcoe:
            scenario_delta_global_series = (
                scenario_df["delta_lcoe_global_$/MWh"]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            
            scenario_delta_global_norm = None
            scenario_delta_global_bound = None
            scenario_delta_global_min = None
            scenario_delta_global_max = None
            scenario_master_levels_global = None
            
            scenario_delta_min_series = (
                scenario_df["delta_lcoe_min_$/MWh"]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            
            scenario_delta_min_norm = None
            scenario_delta_min_bound = None
            scenario_delta_min_min = None
            scenario_delta_min_max = None
            scenario_master_levels_min = None
            
            if not scenario_delta_global_series.empty:
                scenario_delta_global_bound = float(
                    np.nanpercentile(np.abs(scenario_delta_global_series.values), DELTA_LCOE_ABS_PCTL)
                )
                
                if scenario_delta_global_bound > 0:
                    scenario_delta_global_clipped = scenario_delta_global_series.clip(
                        -scenario_delta_global_bound, scenario_delta_global_bound
                    )
                    scenario_delta_global_min = float(scenario_delta_global_clipped.min())
                    scenario_delta_global_max = float(scenario_delta_global_clipped.max())
                    
                    linthresh_global = cfg.DELTA_LCOE_LINTHRESH_USD
                    linscale_global = cfg.DELTA_LCOE_LINSCALE_USD
                    
                    norm_global_vmin = scenario_delta_global_min
                    norm_global_vmax = scenario_delta_global_max
                    
                    scenario_delta_global_norm = mpl.colors.SymLogNorm(
                        linthresh=linthresh_global,
                        linscale=linscale_global,
                        vmin=norm_global_vmin,
                        vmax=norm_global_vmax
                    )
                    
                    scenario_delta_global_clipped_full = scenario_df["delta_lcoe_global_$/MWh"].clip(
                        -scenario_delta_global_bound, scenario_delta_global_bound
                    )
                    scenario_finite_global = scenario_delta_global_clipped_full.replace(
                        [np.inf, -np.inf], np.nan
                    ).dropna().values
                    
                    scenario_master_levels_global = calculate_lcoe_contour_levels(
                        scenario_finite_global, scenario_delta_global_bound, unit='USD'
                    )
            
            if not scenario_delta_min_series.empty:
                scenario_delta_min_bound = float(
                    np.nanpercentile(np.abs(scenario_delta_min_series.values), DELTA_LCOE_ABS_PCTL)
                )
                
                if scenario_delta_min_bound > 0:
                    scenario_delta_min_clipped = scenario_delta_min_series.clip(
                        -scenario_delta_min_bound, scenario_delta_min_bound
                    )
                    scenario_delta_min_min = float(scenario_delta_min_clipped.min())
                    scenario_delta_min_max = float(scenario_delta_min_clipped.max())
                    
                    linthresh_min = cfg.DELTA_LCOE_LINTHRESH_USD
                    linscale_min = cfg.DELTA_LCOE_LINSCALE_USD
                    
                    norm_min_vmin = scenario_delta_min_min
                    norm_min_vmax = scenario_delta_min_max
                    
                    scenario_delta_min_norm = mpl.colors.SymLogNorm(
                        linthresh=linthresh_min,
                        linscale=linscale_min,
                        vmin=norm_min_vmin,
                        vmax=norm_min_vmax
                    )
                    
                    scenario_delta_min_clipped_full = scenario_df["delta_lcoe_min_$/MWh"].clip(
                        -scenario_delta_min_bound, scenario_delta_min_bound
                    )
                    scenario_finite_min = scenario_delta_min_clipped_full.replace(
                        [np.inf, -np.inf], np.nan
                    ).dropna().values
                    
                    scenario_master_levels_min = calculate_lcoe_contour_levels(
                        scenario_finite_min, scenario_delta_min_bound, unit='USD'
                    )
        
        for temp, cool in SCAN_TEMP_COOLANT_PAIRS:
            temp_df = scenario_df[
                (scenario_df["temperature_K"] == temp) &
                (scenario_df["coolant"] == cool)
            ].copy()
            
            if temp_df.empty:
                continue
            
            # Parasitic‑power heatmap
            plot_parasitic_heatmap_single(
                df=temp_df,
                output_dir=output_dir,
                cmap=cfg.cmap_parasitic_ratio,
                norm=global_norm_parasitic,
                scenario=scenario,
                temperature_K=temp
            )
            
            # Record x/y axis‑value distributions
            record_heatmap_axis_distribution(
                readme_path=readme_path,
                image_name=f"parasitic_ratio_heatmap_{scenario}_{temp}K",
                scenario=scenario,
                temperature_K=temp,
                coolant=cool,
                df=temp_df,
                x_col="R_joint",
                y_col="Npw",
                value_col="r_parasitic_pct"
            )
            
            # ΔLCOE heatmap (USD‑denominated)
            if enable_lcoe and delta_norm is not None:
                if (delta_bound is not None) and np.isfinite(delta_bound):
                    feasible_mask = temp_df["delta_lcoe_$/MWh"].notna()
                    temp_df["delta_lcoe_plot_$/MWh"] = temp_df["delta_lcoe_$/MWh"].copy()
                    temp_df.loc[feasible_mask, "delta_lcoe_plot_$/MWh"] = (
                        temp_df.loc[feasible_mask, "delta_lcoe_$/MWh"].clip(-delta_bound, delta_bound)
                    )
                else:
                    temp_df["delta_lcoe_plot_$/MWh"] = temp_df["delta_lcoe_$/MWh"]
                
                plot_delta_lcoe_heatmap_single(
                    df=temp_df,
                    output_dir=output_dir,
                    cmap=DELTA_LCOE_CMAP,
                    norm=delta_norm,
                    scenario=scenario,
                    temperature_K=temp,
                    baseline_point=(baseline_Npw, baseline_Rj),
                    value_col="delta_lcoe_plot_$/MWh",
                    coolant=cool,
                    contour_levels=global_contour_levels,
                )
                
                # Record x/y axis‑value distributions
                record_heatmap_axis_distribution(
                    readme_path=readme_path,
                    image_name=f"delta_lcoe_heatmap_{scenario}_{temp}K_{cool}",
                    scenario=scenario,
                    temperature_K=temp,
                    coolant=cool,
                    df=temp_df,
                    x_col="R_joint",
                    y_col="Npw",
                    value_col="delta_lcoe_plot_$/MWh"
                )
            
            # Global‑reference ΔLCOE heatmap
            if enable_lcoe and scenario_delta_global_norm is not None:
                if (scenario_delta_global_bound is not None) and np.isfinite(scenario_delta_global_bound):
                    feasible_mask_global = temp_df["delta_lcoe_global_$/MWh"].notna()
                    temp_df["delta_lcoe_global_plot_$/MWh"] = temp_df["delta_lcoe_global_$/MWh"].copy()
                    temp_df.loc[feasible_mask_global, "delta_lcoe_global_plot_$/MWh"] = (
                        temp_df.loc[feasible_mask_global, "delta_lcoe_global_$/MWh"].clip(
                            -scenario_delta_global_bound, scenario_delta_global_bound
                        )
                    )
                else:
                    temp_df["delta_lcoe_global_plot_$/MWh"] = temp_df["delta_lcoe_global_$/MWh"]
                
                plot_delta_lcoe_heatmap_single(
                    df=temp_df,
                    output_dir=output_dir,
                    cmap=DELTA_LCOE_CMAP,
                    norm=scenario_delta_global_norm,
                    scenario=scenario,
                    temperature_K=temp,
                    baseline_point=None,
                    value_col="delta_lcoe_global_plot_$/MWh",
                    coolant=cool,
                    contour_levels=None,
                    auto_generate_levels=True,
                )
                
                # Record x/y axis‑value distributions
                record_heatmap_axis_distribution(
                    readme_path=readme_path,
                    image_name=f"delta_lcoe_global_heatmap_{scenario}_{temp}K_{cool}",
                    scenario=scenario,
                    temperature_K=temp,
                    coolant=cool,
                    df=temp_df,
                    x_col="R_joint",
                    y_col="Npw",
                    value_col="delta_lcoe_global_plot_$/MWh"
                )
            
            # ΔLCOE heatmap relative to scenario minimum
            if enable_lcoe and scenario_delta_min_norm is not None:
                if (scenario_delta_min_bound is not None) and np.isfinite(scenario_delta_min_bound):
                    feasible_mask_min = temp_df["delta_lcoe_min_$/MWh"].notna()
                    temp_df["delta_lcoe_min_plot_$/MWh"] = temp_df["delta_lcoe_min_$/MWh"].copy()
                    temp_df.loc[feasible_mask_min, "delta_lcoe_min_plot_$/MWh"] = (
                        temp_df.loc[feasible_mask_min, "delta_lcoe_min_$/MWh"].clip(
                            -scenario_delta_min_bound, scenario_delta_min_bound
                        )
                    )
                else:
                    temp_df["delta_lcoe_min_plot_$/MWh"] = temp_df["delta_lcoe_min_$/MWh"]
                
                plot_delta_lcoe_heatmap_single(
                    df=temp_df,
                    output_dir=output_dir,
                    cmap=DELTA_LCOE_CMAP,
                    norm=scenario_delta_min_norm,
                    scenario=scenario,
                    temperature_K=temp,
                    baseline_point=None,
                    value_col="delta_lcoe_min_plot_$/MWh",
                    coolant=cool,
                    contour_levels=None,
                    auto_generate_levels=True,
                )
                
                # Record x/y axis‑value distributions
                record_heatmap_axis_distribution(
                    readme_path=readme_path,
                    image_name=f"delta_lcoe_min_heatmap_{scenario}_{temp}K_{cool}",
                    scenario=scenario,
                    temperature_K=temp,
                    coolant=cool,
                    df=temp_df,
                    x_col="R_joint",
                    y_col="Npw",
                    value_col="delta_lcoe_min_plot_$/MWh"
                )
        
        # Save per‑scenario colorbars
        if enable_lcoe and scenario_delta_global_norm is not None:
            save_delta_lcoe_heatmap_colorbar(
                output_dir=output_dir,
                cmap=DELTA_LCOE_CMAP,
                norm=scenario_delta_global_norm,
                data_min=scenario_delta_global_min,
                data_max=scenario_delta_global_max,
                label=r"$\Delta$LCOE ($/\mathrm{MWh}$)",
                is_global_ref=True,
                filename_suffix="cividis"   
            )
        
        if enable_lcoe and scenario_delta_min_norm is not None:
            save_delta_lcoe_heatmap_colorbar(
                output_dir=output_dir,
                cmap=DELTA_LCOE_CMAP,
                norm=scenario_delta_min_norm,
                data_min=scenario_delta_min_min,
                data_max=scenario_delta_min_max,
                label=r"$\Delta$LCOE (relative to scenario minimum) ($/\mathrm{MWh}$)",
                is_global_ref=False,
                filename_suffix="cividis"
            )
        print(f"--- Saved per‑scenario heatmaps and colorbars to {output_dir}")
    
    # Save shared colorbars
    save_parasitic_heatmap_colorbar(
        output_dir=SCAN_USD_DIR,
        cmap=cfg.cmap_parasitic_ratio,
        norm=global_norm_parasitic,
        filename_suffix="cividis"
    )
    
    if enable_lcoe and delta_norm is not None:
        save_delta_lcoe_heatmap_colorbar(
            output_dir=SCAN_USD_DIR,
            cmap=DELTA_LCOE_CMAP,
            norm=delta_norm,
            data_min=delta_min,
            data_max=delta_max,
            filename_suffix="cividis"
        )
        save_delta_lcoe_heatmap_colorbar(
            output_dir=SCAN_USD_DIR,
            cmap=DELTA_LCOE_CMAP,
            norm=delta_norm,
            data_min=delta_min,
            data_max=delta_max,
            filename_suffix="YlGnBu"
        )
    
    # cryo_mode_grid
    if not skip_cryo_grid:
        cryo_df = pd.DataFrame(cryo_rows)
        if not cryo_df.empty:
        print("\n--- Generating cryogenic power heatmaps for each operating mode ---")
            cryo_output_dir = SCAN_USD_DIR / "cryo_power_maps"
            cryo_output_dir.mkdir(exist_ok=True, parents=True)
            temp_sequence = [temp for temp, _ in SCAN_TEMP_COOLANT_PAIRS]
            
            mode_titles = {
                "static": "Cooling Mode (Static)",
                "dwell": "Intermission Mode (Dwell)",
                "prod": "Pulse Mode (Production)"
            }
            for mode_key, title in mode_titles.items():
                mode_df = cryo_df[cryo_df["mode"] == mode_key].copy()
                if mode_df.empty:
                    continue
                mode_df["cryo_power_MW"] = mode_df["cryo_power_W"] / 1e6
                positive_values = mode_df.loc[mode_df["cryo_power_MW"] > 0, "cryo_power_MW"]
                vmin = positive_values.min() if not positive_values.empty else np.nan
                vmax = mode_df["cryo_power_MW"].max()
                if np.isnan(vmin) or vmax <= 0:
                    continue
                norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
                save_path = cryo_output_dir / f"cryo_power_grid_{mode_key}."+cfg.PLOT_FORMAT                
                plot_cryo_mode_grid(
                    df=mode_df,
                    scenarios=scenarios,
                    temperatures=temp_sequence,
                    save_path=save_path,
                    cmap=cfg.cmap_parasitic_ratio,
                    norm=norm,
                    mode_label=title
                )
    else:
        print("\nSkipping cryo_mode_grid plotting (data were read from Excel; cryo data unavailable).")


def perform_global_delta_lcoe_cny_scan(params: dict, scenarios: list):
    """
    Parameter scan: generate global‑reference ΔLCOE heatmaps in CNY/kWh units.
    """
    print("\n" + "="*50)
    print("--- Running global‑reference ΔLCOE parameter scan in CNY/kWh units ---")
    print("="*50)
    
    # Check whether Excel data files already exist
    all_excel_exist = True
    all_scenario_dfs = []
    original_output_dir = SCAN_DATA_DIR
    
    for scenario in scenarios:
        output_dir = original_output_dir / scenario
        excel_path = output_dir / f"scan_data_with_delta_lcoe_{scenario}.xlsx"
        if excel_path.exists():
            print(f"Found Excel data for scenario {scenario}; loading directly from: {excel_path}")
            scenario_df = pd.read_excel(excel_path)
            all_scenario_dfs.append(scenario_df)
        else:
            print(f"Missing Excel data for scenario {scenario}: {excel_path}. Please run the USD version first to generate it.")
            all_excel_exist = False
            break
    
    if not all_excel_exist:
        print("\nERROR: Required Excel data files are missing. Please run the USD‑based scan first.")
        return
    
    print("\nAll scenario Excel files found; loading data and converting to CNY/kWh units...")
    all_df = pd.concat(all_scenario_dfs, ignore_index=True)
    
    # Ensure CNY/kWh columns exist
    all_df = convert_usd_to_cny_columns(all_df)
    
    # Ensure that ΔLCOE relative‑to‑minimum (CNY/kWh) columns exist
    if "delta_lcoe_min_CNY_per_kWh" not in all_df.columns:
        print("CNY dataset lacks ΔLCOE relative‑to‑minimum columns; computing them now...")
        if "lcoe_magnet_only_CNY_per_kWh" not in all_df.columns:
            all_df = convert_usd_to_cny_columns(all_df)
        
        for scenario in scenarios:
            scenario_mask = all_df["scenario"] == scenario
            scenario_lcoe_cny = all_df.loc[scenario_mask, "lcoe_magnet_only_CNY_per_kWh"]
            if "r_parasitic_pct" in all_df.columns:
                feasible_mask = (
                    scenario_lcoe_cny.notna()
                    & (all_df.loc[scenario_mask, "r_parasitic_pct"] <= DELTA_LCOE_MASK_PARASITIC_PCT)
                )
            else:
                feasible_mask = scenario_lcoe_cny.notna()
            
            if feasible_mask.any():
                scenario_min_cny = float(scenario_lcoe_cny[feasible_mask].min())
                print(f"[{scenario}] Scenario minimum LCOE: {scenario_min_cny:.6g} (CNY/kWh)")
                all_df.loc[scenario_mask, "delta_lcoe_min_CNY_per_kWh"] = (
                    all_df.loc[scenario_mask, "lcoe_magnet_only_CNY_per_kWh"] - scenario_min_cny
                )
            else:
                print(f"[{scenario}] WARNING: no feasible points in this scenario; cannot compute minimum (CNY/kWh).")
                all_df.loc[scenario_mask, "delta_lcoe_min_CNY_per_kWh"] = np.nan
    else:
        print("CNY data already contains ΔLCOE_min columns; reusing them directly.")
    
    # Mask infeasible points
    if "r_parasitic_pct" in all_df.columns:
        bad = (all_df["r_parasitic_pct"] > DELTA_LCOE_MASK_PARASITIC_PCT)
        all_df.loc[bad, "delta_lcoe_global_CNY_per_kWh"] = np.nan
        if "delta_lcoe_min_CNY_per_kWh" in all_df.columns:
            all_df.loc[bad, "delta_lcoe_min_CNY_per_kWh"] = np.nan
        print(f"Masked points with parasitic power > {DELTA_LCOE_MASK_PARASITIC_PCT}% as infeasible (total {bad.sum()} points).")
    
    # Generate heatmaps for each scenario
    for scenario in scenarios:
        print(f"--- Generating global‑reference ΔLCOE heatmaps in CNY/kWh for scenario {scenario} ---")
        scenario_df = all_df[all_df["scenario"] == scenario].copy()
        output_dir = SCAN_CNY_DIR / scenario
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Compute scenario‑specific norms in CNY/kWh units
        scenario_delta_global_series = (
            scenario_df["delta_lcoe_global_CNY_per_kWh"]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        
        scenario_delta_global_norm = None
        scenario_delta_global_bound = None
        scenario_delta_global_min = None
        scenario_delta_global_max = None
        scenario_master_levels_global = None
        
        scenario_delta_min_series = None
        scenario_delta_min_norm = None
        scenario_delta_min_bound = None
        scenario_delta_min_min = None
        scenario_delta_min_max = None
        scenario_master_levels_min = None
        
        if "delta_lcoe_min_CNY_per_kWh" in scenario_df.columns:
            scenario_delta_min_series = (
                scenario_df["delta_lcoe_min_CNY_per_kWh"]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            
            if not scenario_delta_min_series.empty:
                scenario_delta_min_bound = float(
                    np.nanpercentile(np.abs(scenario_delta_min_series.values), DELTA_LCOE_ABS_PCTL)
                )
                
                try:
                    max_tick_cny = float(np.nanmax(np.abs(cfg.LCOE_CONTOUR_FIXED_LEVELS_CNY)))
                    if np.isfinite(max_tick_cny) and max_tick_cny > 0:
                        scenario_delta_min_bound = min(scenario_delta_min_bound, max_tick_cny)
                except Exception:
                    pass
                
                if scenario_delta_min_bound > 0:
                    scenario_delta_min_clipped = scenario_delta_min_series.clip(
                        -scenario_delta_min_bound, scenario_delta_min_bound
                    )
                    scenario_delta_min_min = float(scenario_delta_min_clipped.min())
                    scenario_delta_min_max = float(scenario_delta_min_clipped.max())
                    
                    linthresh_min = cfg.DELTA_LCOE_LINTHRESH_CNY
                    linscale_min = cfg.DELTA_LCOE_LINSCALE_CNY
                    
                    norm_min_vmin = scenario_delta_min_min
                    norm_min_vmax = scenario_delta_min_max
                    
                    scenario_delta_min_norm = mpl.colors.SymLogNorm(
                        linthresh=linthresh_min,
                        linscale=linscale_min,
                        vmin=norm_min_vmin,
                        vmax=norm_min_vmax,
                    )
                    
                    scenario_delta_min_clipped_full = scenario_df["delta_lcoe_min_CNY_per_kWh"].clip(
                        -scenario_delta_min_bound, scenario_delta_min_bound
                    )
                    scenario_finite_min = (
                        scenario_delta_min_clipped_full.replace([np.inf, -np.inf], np.nan)
                        .dropna()
                        .values
                    )
                    
                    scenario_master_levels_min = calculate_lcoe_contour_levels(
                        scenario_finite_min, scenario_delta_min_bound, unit='CNY'
                    )
        
        if not scenario_delta_global_series.empty:
            scenario_delta_global_bound = float(
                np.nanpercentile(np.abs(scenario_delta_global_series.values), DELTA_LCOE_ABS_PCTL)
            )
            
            try:
                max_tick_cny = float(np.nanmax(np.abs(cfg.LCOE_CONTOUR_FIXED_LEVELS_CNY)))
                if np.isfinite(max_tick_cny) and max_tick_cny > 0:
                    scenario_delta_global_bound = min(scenario_delta_global_bound, max_tick_cny)
            except Exception:
                pass
            
            if scenario_delta_global_bound > 0:
                scenario_delta_global_clipped = scenario_delta_global_series.clip(
                    -scenario_delta_global_bound, scenario_delta_global_bound
                )
                scenario_delta_global_min = float(scenario_delta_global_clipped.min())
                scenario_delta_global_max = float(scenario_delta_global_clipped.max())
                
                linthresh_global = cfg.DELTA_LCOE_LINTHRESH_CNY
                linscale_global = cfg.DELTA_LCOE_LINSCALE_CNY
                
                norm_global_vmin = scenario_delta_global_min
                norm_global_vmax = scenario_delta_global_max
                
                scenario_delta_global_norm = mpl.colors.SymLogNorm(
                    linthresh=linthresh_global,
                    linscale=linscale_global,
                    vmin=norm_global_vmin,
                    vmax=norm_global_vmax
                )
                
                scenario_delta_global_clipped_full = scenario_df["delta_lcoe_global_CNY_per_kWh"].clip(
                    -scenario_delta_global_bound, scenario_delta_global_bound
                )
                scenario_finite_global = scenario_delta_global_clipped_full.replace(
                    [np.inf, -np.inf], np.nan
                ).dropna().values
                
                scenario_master_levels_global = calculate_lcoe_contour_levels(
                    scenario_finite_global, scenario_delta_global_bound, unit='CNY'
                )
        
        for temp, cool in SCAN_TEMP_COOLANT_PAIRS:
            temp_df = scenario_df[
                (scenario_df["temperature_K"] == temp) &
                (scenario_df["coolant"] == cool)
            ].copy()
            
            if temp_df.empty:
                continue
            
            # Global‑reference ΔLCOE heatmap (CNY/kWh)
            if (scenario_delta_global_bound is not None) and np.isfinite(scenario_delta_global_bound):
                feasible_mask_global = temp_df["delta_lcoe_global_CNY_per_kWh"].notna()
                temp_df["delta_lcoe_global_CNY_per_kWh_plot"] = temp_df["delta_lcoe_global_CNY_per_kWh"].copy()
                temp_df.loc[feasible_mask_global, "delta_lcoe_global_CNY_per_kWh_plot"] = (
                    temp_df.loc[feasible_mask_global, "delta_lcoe_global_CNY_per_kWh"].clip(
                        -scenario_delta_global_bound, scenario_delta_global_bound
                    )
                )
            else:
                temp_df["delta_lcoe_global_CNY_per_kWh_plot"] = temp_df["delta_lcoe_global_CNY_per_kWh"]
            
            if scenario_delta_global_norm is not None and scenario_master_levels_global is not None:
                subplot_data = temp_df["delta_lcoe_global_CNY_per_kWh_plot"].replace([np.inf, -np.inf], np.nan)
                subplot_finite = subplot_data.dropna()
                
                if len(subplot_finite) > 0:
                    z_min = float(subplot_finite.min())
                    z_max = float(subplot_finite.max())
                    subplot_contour_levels = select_subplot_contour_levels(
                        z_min=z_min,
                        z_max=z_max,
                        master_levels=scenario_master_levels_global,
                        target_count=5
                    )
                else:
                    subplot_contour_levels = scenario_master_levels_global[:6] if len(scenario_master_levels_global) >= 6 else scenario_master_levels_global
                
                plot_delta_lcoe_heatmap_single(
                    df=temp_df,
                    output_dir=output_dir,
                    cmap=DELTA_LCOE_CMAP,
                    norm=scenario_delta_global_norm,
                    scenario=scenario,
                    temperature_K=temp,
                    baseline_point=None,
                    value_col="delta_lcoe_global_CNY_per_kWh_plot",
                    coolant=cool,
                    contour_levels=subplot_contour_levels,
                )
            
            # ΔLCOE heatmap relative to scenario minimum (CNY/kWh)
            if (
                scenario_delta_min_norm is not None
                and scenario_master_levels_min is not None
                and "delta_lcoe_min_CNY_per_kWh" in temp_df.columns
            ):
                if (scenario_delta_min_bound is not None) and np.isfinite(scenario_delta_min_bound):
                    feasible_mask_min = temp_df["delta_lcoe_min_CNY_per_kWh"].notna()
                    temp_df["delta_lcoe_min_CNY_per_kWh_plot"] = temp_df["delta_lcoe_min_CNY_per_kWh"].copy()
                    temp_df.loc[feasible_mask_min, "delta_lcoe_min_CNY_per_kWh_plot"] = (
                        temp_df.loc[feasible_mask_min, "delta_lcoe_min_CNY_per_kWh"].clip(
                            -scenario_delta_min_bound, scenario_delta_min_bound
                        )
                    )
                else:
                    temp_df["delta_lcoe_min_CNY_per_kWh_plot"] = temp_df["delta_lcoe_min_CNY_per_kWh"]
                
                subplot_data_min = temp_df["delta_lcoe_min_CNY_per_kWh_plot"].replace([np.inf, -np.inf], np.nan)
                subplot_finite_min = subplot_data_min.dropna()
                
                if len(subplot_finite_min) > 0:
                    z_min_min = float(subplot_finite_min.min())
                    z_max_min = float(subplot_finite_min.max())
                    
                    subplot_contour_levels_min = select_subplot_contour_levels(
                        z_min=z_min_min,
                        z_max=z_max_min,
                        master_levels=scenario_master_levels_min,
                        target_count=5,
                        min_allowed=0.0,
                    )
                    
                    subplot_contour_levels_min = filter_contour_levels_to_target_count(
                        levels=subplot_contour_levels_min,
                        z_min=z_min_min,
                        z_max=z_max_min,
                        master_levels=scenario_master_levels_min,
                        target_min_count=4,
                        target_max_count=5,
                        verbose=False
                    )
                else:
                    subplot_contour_levels_min = (
                        scenario_master_levels_min[:5]
                        if len(scenario_master_levels_min) >= 5
                        else scenario_master_levels_min
                    )
                
                plot_delta_lcoe_heatmap_single(
                    df=temp_df,
                    output_dir=output_dir,
                    cmap=DELTA_LCOE_CMAP,
                    norm=scenario_delta_min_norm,
                    scenario=scenario,
                    temperature_K=temp,
                    baseline_point=None,
                    value_col="delta_lcoe_min_CNY_per_kWh_plot",
                    coolant=cool,
                    contour_levels=subplot_contour_levels_min,
                )
        
        # Save colorbars for this scenario
        if scenario_delta_global_norm is not None:
            save_delta_lcoe_heatmap_colorbar(
                output_dir=output_dir,
                cmap=DELTA_LCOE_CMAP,
                norm=scenario_delta_global_norm,
                data_min=scenario_delta_global_min,
                data_max=scenario_delta_global_max,
                label=r"$\Delta$LCOE ($\mathrm{CNY}/\mathrm{kWh}$)",
                is_global_ref=True,
                filename_suffix="cividis"
            )
            save_delta_lcoe_heatmap_colorbar(
                output_dir=output_dir,
                cmap=DELTA_LCOE_CMAP,
                norm=scenario_delta_global_norm,
                data_min=scenario_delta_global_min,
                data_max=scenario_delta_global_max,
                label=r"$\Delta$LCOE ($\mathrm{CNY}/\mathrm{kWh}$)",
                is_global_ref=True,
                filename_suffix="YlGnBu"
            )
        
        if scenario_delta_min_norm is not None:
            save_delta_lcoe_heatmap_colorbar(
                output_dir=output_dir,
                cmap=DELTA_LCOE_CMAP,
                norm=scenario_delta_min_norm,
                data_min=scenario_delta_min_min,
                data_max=scenario_delta_min_max,
                label=r"$\Delta$LCOE (relative to scenario minimum) ($\mathrm{CNY}/\mathrm{kWh}$)",
                is_global_ref=False,
                filename_suffix="cividis"
            )


def perform_fixed_point_analysis(params: dict):
    """
    Detailed economic analysis at a fixed design point.
    """
    print("\n" + "="*50)
    print("--- Task 2: starting fixed‑design‑point economic analysis ---")
    print("="*50)
    
    TARGET_DESIGN_POINT = {'Npw': cfg.Npw_TARGET, 'R_p2p_joint': cfg.R_p2p_joint_TARGET}
    
    all_economic_results, all_heat_load_results = [], []
    for temp, cool in ECONOMIC_OPERATING_CONDITIONS:
        for tech_scenario in params['tech_scenarios']:
            year = params['tech_scenario_to_years'][tech_scenario]
            economic_result, heat_load_result = compute_case(
                tech_scenario, year, temp, cool,
                TARGET_DESIGN_POINT['Npw'], TARGET_DESIGN_POINT['R_p2p_joint'], params
            )
            if economic_result and heat_load_result:
                all_economic_results.append(economic_result)
                case_identifiers = {
                    'tech_scenario': tech_scenario,
                    'project_years': year,
                    'temperature_K': temp,
                    'coolant': cool,
                    **TARGET_DESIGN_POINT
                }
                all_heat_load_results.append({**case_identifiers, **heat_load_result})
    
    if not all_economic_results:
        print("WARNING: fixed‑point analysis did not produce any results.")
        return
    
    economic_df = pd.DataFrame(all_economic_results)
    heat_load_df = pd.DataFrame(all_heat_load_results)
    
    # Add CNY‑denominated LCOE columns
    if 'lcoe_magnet_only_$/MWh' in economic_df.columns:
        economic_df['lcoe_magnet_only_CNY_per_kWh'] = (
            economic_df['lcoe_magnet_only_$/MWh'] * USD_TO_CNY_EXCHANGE_RATE / 1000
        )
    
    ECONOMIC_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    with pd.ExcelWriter(FINAL_ECONOMICS_FILE, engine='openpyxl') as writer:
        economic_df.to_excel(writer, sheet_name='Economic_Summary', index=False)
        heat_load_df.to_excel(writer, sheet_name='Heat_Load_Breakdown', index=False)
    
    print(f"Economic analysis data saved to: {FINAL_ECONOMICS_FILE}")
    
    # Call all relevant plotting utilities
    print("\n--- Generating economic summary figures ---")
    save_color_legend(ECONOMIC_OUTPUT_DIR / "legend_color_conditions."+cfg.PLOT_FORMAT)
    save_hatch_legend(ECONOMIC_OUTPUT_DIR / "legend_hatch_costs."+cfg.PLOT_FORMAT)
    plot_parasitic_power_vs_temp(str(FINAL_ECONOMICS_FILE), ECONOMIC_OUTPUT_DIR / "parasitic_power_vs_temp."+cfg.PLOT_FORMAT)
    
    for tech_scenario in params['tech_scenarios']:
        year = params['tech_scenario_to_years'][tech_scenario]
        print(f"--- Generating combined summary figure for scenario {tech_scenario} (year {year}) ---")
        combined_fig_path = ECONOMIC_OUTPUT_DIR / f"economic_summary_combined_{tech_scenario}_{year}yr."+cfg.PLOT_FORMAT
        plot_combined_charts(str(FINAL_ECONOMICS_FILE), str(combined_fig_path), project_year_to_plot=year)
    
    print("All economic analysis figures have been generated successfully.")


# =============================================================================
# 4. Main execution entry point
# =============================================================================
if __name__ == "__main__":
    print("Starting unified economic analysis script...")
    
    # Configuration switches
    RUN_PARAMETRIC_SCAN = True       # run parameter scan
    RUN_FIXED_POINT_ANALYSIS = False # run fixed‑point analysis
    ENABLE_LCOE_ANALYSIS = True      # enable LCOE analysis (USD)
    ENABLE_CNY_ANALYSIS = True       # enable CNY‑denominated analysis (requires USD run first)
    
    # Scenarios to analyze
    SCAN_SCENARIOS = list(cfg.SCENARIO_DEFINITIONS.keys())  # or specify a subset, e.g. ['S3']
    #SCAN_SCENARIOS = ['S3']
    
    full_params = define_parameters()
    
    # Create output directories
    SCAN_BASE_DIR_FIG.mkdir(exist_ok=True, parents=True)
    SCAN_USD_DIR.mkdir(exist_ok=True, parents=True)
    SCAN_CNY_DIR.mkdir(exist_ok=True, parents=True)
    SCAN_DATA_DIR.mkdir(exist_ok=True, parents=True)
    
    if RUN_PARAMETRIC_SCAN:
        perform_parasitic_power_scan(full_params, SCAN_SCENARIOS, enable_lcoe=ENABLE_LCOE_ANALYSIS)
        
        if ENABLE_CNY_ANALYSIS:
            perform_global_delta_lcoe_cny_scan(full_params, SCAN_SCENARIOS)
    
    if RUN_FIXED_POINT_ANALYSIS:
        perform_fixed_point_analysis(full_params)
    
    print("\nUnified economic analysis script has completed.")

