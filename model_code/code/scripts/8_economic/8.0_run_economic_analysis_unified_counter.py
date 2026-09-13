# 8.0_run_economic_analysis_unified.py (统一经济分析脚本)
# 合并了8.1、8.2、8.3三个文件的功能，支持USD和CNY单位的LCOE分析
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple

from pandas.core.nanops import F

# --- 导入模型计算函数 ---
from fusion_tem.economic.lcoe import define_parameters, compute_case
from fusion_tem import device as cfg

# --- 从【统一的绘图库】中导入【所有】绘图函数 ---
from fusion_tem.plotting.library import (
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
# 1. 配置参数（从config.py读取）
# =============================================================================
DELTA_LCOE_CMAP = cfg.cmap_parasitic_ratio
DELTA_LCOE_MASK_PARASITIC_PCT = cfg.DELTA_LCOE_MASK_PARASITIC_PCT
DELTA_LCOE_ABS_PCTL = cfg.DELTA_LCOE_ABS_PCTL
USD_TO_CNY_EXCHANGE_RATE = cfg.USD_TO_CNY_EXCHANGE_RATE

# 参数扫描配置
# 注意：R_JOINT_SCAN_VALUES 现在以 nOhm 为单位，这里只作为扫描/记录用；
# 真正传入经济模型时会在用到的地方统一换算为 Ohm（×1e-9）。
R_JOINT_SCAN_VALUES = cfg.R_JOINT_SCAN_VALUES  # nOhm
NPW_SCAN_VALUES = cfg.NPW_SCAN_VALUES
SCAN_TEMP_COOLANT_PAIRS = cfg.SCAN_TEMP_COOLANT_PAIRS
ECONOMIC_OPERATING_CONDITIONS = cfg.ECONOMIC_OPERATING_CONDITIONS

# 输出目录配置
# 图片目录（figures）：用于保存SVG图片，与9.0_stitch_econimic_svgs.py的路径一致
SCAN_BASE_DIR_FIG = Path(cfg.ECONOMIC_FIGURES_DIR) / cfg.PARASITIC_RATIO_OUTPUT_DIR
SCAN_USD_DIR = SCAN_BASE_DIR_FIG / cfg.PARASITIC_RATIO_USD_DIR
SCAN_CNY_DIR = SCAN_BASE_DIR_FIG / cfg.PARASITIC_RATIO_CNY_DIR
# 表格目录（tables）：用于保存Excel数据文件
SCAN_DATA_DIR = Path(cfg.ECONOMIC_OUTPUT_DIR) / cfg.PARASITIC_RATIO_OUTPUT_DIR / cfg.PARASITIC_RATIO_DATA_DIR
ECONOMIC_OUTPUT_DIR = Path(cfg.ECONOMIC_OUTPUT_DIR) / cfg.COST_OUTPUT_DIR
FINAL_ECONOMICS_FILE = ECONOMIC_OUTPUT_DIR / cfg.ECONOMIC_SUMMARY_FILE

# =============================================================================
# 2. 可复用的辅助函数
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
    记录热力图的横轴和纵轴数据分布信息到 README.md 文件。
    
    参数:
        readme_path: README.md 文件路径
        image_name: 图像名称（用于标识）
        scenario: 场景名称
        temperature_K: 温度值
        coolant: 制冷剂类型（可选）
        df: 数据DataFrame
        x_col: 横轴列名（默认 "R_joint"）
        y_col: 纵轴列名（默认 "Npw"）
        value_col: 数值列名（可选，用于记录数值范围）
    """
    # 确保 README.md 文件存在
    readme_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 提取横轴和纵轴数据
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
    
    # 提取数值范围（如果提供了 value_col）
    value_info = ""
    if value_col and value_col in df.columns:
        value_data = df[value_col].dropna()
        if len(value_data) > 0:
            value_min = float(value_data.min())
            value_max = float(value_data.max())
            value_info = f"\n- **数值范围**: [{value_min:.4g}, {value_max:.4g}]"
    
    # 生成标签
    if coolant:
        label = f"{temperature_K}K_{coolant}"
    else:
        label = f"{temperature_K}K"
    
    # 格式化横轴数据（如果是 R_joint，转换为 nOhm 显示）
    if x_col == "R_joint":
        x_min_display = x_min * 1e9 if np.isfinite(x_min) else np.nan  # 转换为 nOhm
        x_max_display = x_max * 1e9 if np.isfinite(x_max) else np.nan
        x_unit = "nOhm"
        x_name = "接头电阻 (R_joint)"
    else:
        x_min_display = x_min
        x_max_display = x_max
        x_unit = ""
        x_name = x_col
    
    # 格式化纵轴数据
    if y_col == "Npw":
        y_name = "并绕根数 (Npw)"
    else:
        y_name = y_col
    
    # 格式化唯一值列表（限制显示长度）
    def format_unique_values(values, max_display=20):
        if len(values) == 0:
            return "[]"
        if len(values) <= max_display:
            if len(values) <= 10:
                return str([f"{v:.4g}" if isinstance(v, float) else str(v) for v in values])
            else:
                return f"[{values[0]:.4g}, ..., {values[-1]:.4g}] (共{len(values)}个)"
        else:
            return f"[{values[0]:.4g}, ..., {values[-1]:.4g}] (共{len(values)}个，仅显示前{max_display}个: {[f'{v:.4g}' if isinstance(v, float) else str(v) for v in values[:max_display]]})"
    
    x_unique_str = format_unique_values(x_unique)
    y_unique_str = format_unique_values(y_unique)
    
    # 生成记录内容
    record_content = f"""
### {image_name} - {scenario} - {label}

**横轴 ({x_name}):**
- **数据范围**: [{x_min_display:.4g}, {x_max_display:.4g}] {x_unit}
- **唯一值数量**: {x_count}
- **唯一值列表**: {x_unique_str}

**纵轴 ({y_name}):**
- **数据范围**: [{y_min:.4g}, {y_max:.4g}]
- **唯一值数量**: {y_count}
- **唯一值列表**: {y_unique_str}{value_info}

---
"""
    
    # 追加到 README.md 文件
    with open(readme_path, 'a', encoding='utf-8') as f:
        f.write(record_content)


def format_array_for_print(arr: np.ndarray) -> str:
    """
    格式化numpy数组用于打印，避免科学计数法。
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
    对等高线值进行舍入，使其最多保留一位小数（或根据数量级决定有效数字）。
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
    unit: str = 'USD'  # 'USD' 或 'CNY'
) -> np.ndarray:
    """
    计算LCOE等高线级别数组。
    
    参数:
        all_finite: 所有有限值的数据数组（已clip到delta_bound范围）
        delta_bound: 数据边界值（用于clip）
        unit: 单位类型，'USD' 或 'CNY'
    
    返回:
        等高线级别数组（已排序）
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
        # 自适应方法
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
            
            # 分别处理正负值数据
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
            
            # 对于正值部分
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
            
            # 对于负值部分
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
            
            # 组合所有等高线级别
            if len(neg_levels) > 0 and len(pos_levels) > 0:
                contour_levels = np.concatenate([neg_levels[::-1], [0.0], pos_levels])
            elif len(neg_levels) > 0:
                contour_levels = np.concatenate([neg_levels[::-1], [0.0]])
            elif len(pos_levels) > 0:
                contour_levels = np.concatenate([[0.0], pos_levels])
            else:
                contour_levels = np.array([0.0])
            
            contour_levels = np.sort(contour_levels)
    
    # 确保等高线之间的最小间隔
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
    从主等高线级别数组中为单个子图选择5-6条合适的等高线。
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
    为每个场景在 all_df 中添加相对于本场景最小 LCOE 的差分列。
    """
    print("\n--- 计算每个场景内相对于最小值的差分LCOE ---")
    for scenario in scenarios:
        scenario_mask = all_df["scenario"] == scenario
        if "lcoe_plant_$/MWh" not in all_df.columns:
            continue
        
        scenario_lcoe = all_df.loc[scenario_mask, "lcoe_plant_$/MWh"]
        if "r_parasitic_pct" in all_df.columns:
            feasible_mask_scenario = (
                scenario_lcoe.notna()
                & (all_df.loc[scenario_mask, "r_parasitic_pct"] <= DELTA_LCOE_MASK_PARASITIC_PCT)
            )
        else:
            feasible_mask_scenario = scenario_lcoe.notna()
        
        if feasible_mask_scenario.any():
            scenario_min_lcoe = float(scenario_lcoe[feasible_mask_scenario].min())
            print(f"[{scenario}] 场景内最小LCOE: {scenario_min_lcoe:.3g} ($/MWh)")
            
            all_df.loc[scenario_mask, "delta_lcoe_min_$/MWh"] = (
                all_df.loc[scenario_mask, "lcoe_plant_$/MWh"] - scenario_min_lcoe
            )
            all_df.loc[scenario_mask, "delta_lcoe_min_CNY_per_kWh"] = (
                all_df.loc[scenario_mask, "delta_lcoe_min_$/MWh"] * USD_TO_CNY_EXCHANGE_RATE / 1000
            )
        else:
            print(f"[{scenario}] 警告：场景内没有可行点，无法计算最小值")
            all_df.loc[scenario_mask, "delta_lcoe_min_$/MWh"] = np.nan
            all_df.loc[scenario_mask, "delta_lcoe_min_CNY_per_kWh"] = np.nan
    
    return all_df


def convert_usd_to_cny_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    将USD/MWh单位的LCOE列转换为CNY/kWh单位。
    """
    if "lcoe_plant_$/MWh" in df.columns:
        if "lcoe_fullplant_CNY_per_kWh" not in df.columns:
            df["lcoe_fullplant_CNY_per_kWh"] = (
                df["lcoe_plant_$/MWh"] * USD_TO_CNY_EXCHANGE_RATE / 1000
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
# 3. 主执行函数
# =============================================================================

def perform_parasitic_power_scan(params: dict, scenarios: list, enable_lcoe: bool = True):
    """
    参数扫描：同时输出
      1) r_parasitic_pct 热力图（原有）
      2) delta_lcoe_$/MWh 热力图（可选）
      3) delta_lcoe_global_$/MWh 热力图（可选）
      4) delta_lcoe_min_$/MWh 热力图（可选）
    """
    print("\n" + "="*50)
    print("--- 任务一：开始执行参数扫描（parasitic + ΔLCOE）---")
    print("="*50)
    
    # 定义baseline点（在所有情况下都需要）
    baseline_Npw = cfg.Npw_TARGET
    baseline_Rj = cfg.R_p2p_joint_TARGET
    
    # 检查是否存在excel数据文件
    all_excel_exist = True
    all_scenario_dfs = []
    for scenario in scenarios:
        output_dir = SCAN_DATA_DIR / scenario
        excel_path = output_dir / f"scan_data_with_delta_lcoe_{scenario}.xlsx"
        if excel_path.exists():
            print(f"检测到 {scenario} 场景的excel数据文件，将直接读取: {excel_path}")
            scenario_df = pd.read_excel(excel_path)
            all_scenario_dfs.append(scenario_df)
        else:
            print(f"未找到 {scenario} 场景的excel数据文件: {excel_path}，将执行计算")
            all_excel_exist = False
            break
    
    if all_excel_exist and len(all_scenario_dfs) == len(scenarios):
        print("\n所有场景的excel数据文件都存在，跳过计算，直接读取数据并绘图...")
        all_df = pd.concat(all_scenario_dfs, ignore_index=True)
        if "delta_lcoe_min_$/MWh" not in all_df.columns:
            all_df = add_delta_lcoe_min_columns(all_df, scenarios)
        else:
            print("Excel数据中已包含相对于最小值的差分LCOE列，直接使用。")
        skip_cryo_grid = True
    else:
        print("\n执行数据计算...")
        skip_cryo_grid = False
        all_rows = []
        cryo_rows = []
        
        # 计算全局基准（20K-H₂，S3）的LCOE值
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
            global_reference_lcoe = global_ref_res.get("lcoe_plant_$/MWh", np.nan)
            print(f"全局基准（{global_reference_temp}K-{global_reference_coolant}，{global_reference_scenario}）LCOE: {global_reference_lcoe:.3g} ($/MWh)")
        else:
            print(f"警告：无法计算全局基准的LCOE")
        
        # 执行计算循环
        for scenario in scenarios:
            scenario_year = params['tech_scenario_to_years'][scenario]
            print(f"--- 正在计算 {scenario} 场景（{scenario_year}年）的参数扫描数据 ---")
            for temp, cool in SCAN_TEMP_COOLANT_PAIRS:
                # 先算 baseline LCOE
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
                    baseline_lcoe = base_res.get("lcoe_plant_$/MWh", np.nan)
                
                for Npw in NPW_SCAN_VALUES:
                    for Rj_nohm in R_JOINT_SCAN_VALUES:  # 扫描值（nOhm）
                        Rj = float(Rj_nohm) * 1e-9  # 换算为 Ohm，传入经济模型
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
                        
                        lcoe_B = economic_result.get("lcoe_plant_$/MWh", np.nan)
                        
                        # 计算各种ΔLCOE
                        if np.isfinite(lcoe_B) and np.isfinite(baseline_lcoe):
                            delta_lcoe = lcoe_B - baseline_lcoe
                        else:
                            delta_lcoe = np.nan
                        
                        if np.isfinite(lcoe_B) and np.isfinite(global_reference_lcoe):
                            delta_lcoe_global = lcoe_B - global_reference_lcoe
                        else:
                            delta_lcoe_global = np.nan
                        
                        # 计算人民币单位
                        lcoe_cny_per_kwh = lcoe_B * USD_TO_CNY_EXCHANGE_RATE / 1000 if np.isfinite(lcoe_B) else np.nan
                        baseline_lcoe_cny_per_kwh = baseline_lcoe * USD_TO_CNY_EXCHANGE_RATE / 1000 if np.isfinite(baseline_lcoe) else np.nan
                        delta_lcoe_cny_per_kwh = delta_lcoe * USD_TO_CNY_EXCHANGE_RATE / 1000 if np.isfinite(delta_lcoe) else np.nan
                        delta_lcoe_global_cny_per_kwh = delta_lcoe_global * USD_TO_CNY_EXCHANGE_RATE / 1000 if np.isfinite(delta_lcoe_global) else np.nan
                        
                        all_rows.append({
                            "temperature_K": temp,
                            "coolant": cool,
                            "Npw": Npw,
                            # 记录两种单位，便于后处理/绘图：
                            "R_joint": Rj,            # Ohm
                            "R_joint_nOhm": Rj_nohm,  # nOhm
                            "scenario": scenario,
                            "r_parasitic_pct": economic_result.get("r_parasitic_pct", np.nan),
                            "lcoe_plant_$/MWh": lcoe_B,
                            "baseline_lcoe_$/MWh": baseline_lcoe,
                            "delta_lcoe_$/MWh": delta_lcoe,
                            "delta_lcoe_global_$/MWh": delta_lcoe_global,
                            "lcoe_fullplant_CNY_per_kWh": lcoe_cny_per_kwh,
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
    
    # 计算全局norm
    global_vmin = all_df["r_parasitic_pct"].min()
    global_vmax = all_df["r_parasitic_pct"].max()
    global_norm_parasitic = mpl.colors.LogNorm(
        vmin=float(f"{global_vmin:.10f}"),
        vmax=float(f"{global_vmax:.10f}")
    )
    print(f"parasitic 全局colorbar范围: {global_vmin:.3g}% - {global_vmax:.3g}%")
    
    # ΔLCOE的global norm
    if enable_lcoe:
        bad = (all_df["r_parasitic_pct"] > DELTA_LCOE_MASK_PARASITIC_PCT)
        all_df.loc[bad, "delta_lcoe_$/MWh"] = np.nan
        all_df.loc[bad, "delta_lcoe_global_$/MWh"] = np.nan
        all_df.loc[bad, "delta_lcoe_min_$/MWh"] = np.nan
        print(f"已mask寄生功率>{DELTA_LCOE_MASK_PARASITIC_PCT}%的点为不可行区域（共{bad.sum()}个点）")
        
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
                print(f"ΔLCOE 色标范围采用 |Δ| 的 P{DELTA_LCOE_ABS_PCTL}: ±{delta_bound:.3g} ($/MWh)")
                
                all_delta_clipped = all_df["delta_lcoe_$/MWh"].clip(-delta_bound, delta_bound)
                all_finite = all_delta_clipped.replace([np.inf, -np.inf], np.nan).dropna().values
                
                global_contour_levels = calculate_lcoe_contour_levels(all_finite, delta_bound, unit='USD')
    
    # 确定 README.md 文件路径（用于记录数据分布信息）
    readme_path = Path("outputs/figures/heatload/relative_heatload_grids/README.md")
    readme_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 如果 README.md 不存在，创建初始内容
    if not readme_path.exists():
        from datetime import datetime
        current_date = datetime.now().strftime("%Y年%m月%d日")
        initial_content = f"""# 经济分析热力图数据分布说明文档

本文档记录了经济分析热力图的横轴和纵轴数据分布信息。

**文档生成日期：** {current_date}

## 数据分布记录

"""
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(initial_content)
    
    # 为每个场景生成热力图
    for scenario in scenarios:
        print(f"--- 正在为 {scenario} 场景生成热力图 ---")
        scenario_df = all_df[all_df["scenario"] == scenario].copy()
        output_dir = SCAN_USD_DIR / scenario
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # 保存数据
        if not (all_excel_exist and len(all_scenario_dfs) == len(scenarios)):
            data_output_dir = SCAN_DATA_DIR / scenario
            data_output_dir.mkdir(exist_ok=True, parents=True)
            excel_path = data_output_dir / f"scan_data_with_delta_lcoe_{scenario}.xlsx"
            scenario_df.to_excel(excel_path, index=False)
        
        # 计算场景特定的norm（用于全局基准和相对于最小值的ΔLCOE）
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
            
            # parasitic 热力图
            plot_parasitic_heatmap_single(
                df=temp_df,
                output_dir=output_dir,
                cmap=cfg.cmap_parasitic_ratio,
                norm=global_norm_parasitic,
                scenario=scenario,
                temperature_K=temp
            )
            
            # 记录横轴和纵轴数据分布
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
            
            # ΔLCOE 热力图
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
                
                # 记录横轴和纵轴数据分布
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
            
            # 全局基准ΔLCOE 热力图
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
                
                # 记录横轴和纵轴数据分布
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
            
            # 相对于最小值的ΔLCOE 热力图
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
                
                # 记录横轴和纵轴数据分布
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
        
        # 保存colorbar
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
        print("---已经保存到", output_dir)
    
    # 保存共享colorbar
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
            print("\n--- 正在生成各模式的制冷功率热力图 ---")
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
        print("\n跳过 cryo_mode_grid 绘制（从excel读取数据，cryo数据不可用）")


def perform_global_delta_lcoe_cny_scan(params: dict, scenarios: list):
    """
    参数扫描：生成CNY/kWh单位的全局基准ΔLCOE热力图
    """
    print("\n" + "="*50)
    print("--- 开始执行CNY/kWh单位的全局基准ΔLCOE参数扫描 ---")
    print("="*50)
    
    # 检查是否存在excel数据文件
    all_excel_exist = True
    all_scenario_dfs = []
    original_output_dir = SCAN_DATA_DIR
    
    for scenario in scenarios:
        output_dir = original_output_dir / scenario
        excel_path = output_dir / f"scan_data_with_delta_lcoe_{scenario}.xlsx"
        if excel_path.exists():
            print(f"检测到 {scenario} 场景的excel数据文件，将直接读取: {excel_path}")
            scenario_df = pd.read_excel(excel_path)
            all_scenario_dfs.append(scenario_df)
        else:
            print(f"未找到 {scenario} 场景的excel数据文件: {excel_path}，需要先运行USD版本生成数据")
            all_excel_exist = False
            break
    
    if not all_excel_exist:
        print("\n错误：缺少excel数据文件，请先运行USD版本生成数据。")
        return
    
    print("\n所有场景的excel数据文件都存在，读取数据并转换为CNY/kWh单位...")
    all_df = pd.concat(all_scenario_dfs, ignore_index=True)
    
    # 确保有CNY/kWh列
    all_df = convert_usd_to_cny_columns(all_df)
    
    # 确保存在相对于最小值的ΔLCOE（CNY/kWh）列
    if "delta_lcoe_min_CNY_per_kWh" not in all_df.columns:
        print("CNY数据中缺少相对于最小值的差分LCOE列，将进行计算...")
        if "lcoe_fullplant_CNY_per_kWh" not in all_df.columns:
            all_df = convert_usd_to_cny_columns(all_df)
        
        for scenario in scenarios:
            scenario_mask = all_df["scenario"] == scenario
            scenario_lcoe_cny = all_df.loc[scenario_mask, "lcoe_fullplant_CNY_per_kWh"]
            if "r_parasitic_pct" in all_df.columns:
                feasible_mask = (
                    scenario_lcoe_cny.notna()
                    & (all_df.loc[scenario_mask, "r_parasitic_pct"] <= DELTA_LCOE_MASK_PARASITIC_PCT)
                )
            else:
                feasible_mask = scenario_lcoe_cny.notna()
            
            if feasible_mask.any():
                scenario_min_cny = float(scenario_lcoe_cny[feasible_mask].min())
                print(f"[{scenario}] 场景内最小LCOE: {scenario_min_cny:.6g} (CNY/kWh)")
                all_df.loc[scenario_mask, "delta_lcoe_min_CNY_per_kWh"] = (
                    all_df.loc[scenario_mask, "lcoe_fullplant_CNY_per_kWh"] - scenario_min_cny
                )
            else:
                print(f"[{scenario}] 警告：场景内没有可行点，无法计算最小值 (CNY/kWh)")
                all_df.loc[scenario_mask, "delta_lcoe_min_CNY_per_kWh"] = np.nan
    else:
        print("CNY数据中已包含相对于最小值的差分LCOE列，直接使用。")
    
    # mask不可行点
    if "r_parasitic_pct" in all_df.columns:
        bad = (all_df["r_parasitic_pct"] > DELTA_LCOE_MASK_PARASITIC_PCT)
        all_df.loc[bad, "delta_lcoe_global_CNY_per_kWh"] = np.nan
        if "delta_lcoe_min_CNY_per_kWh" in all_df.columns:
            all_df.loc[bad, "delta_lcoe_min_CNY_per_kWh"] = np.nan
        print(f"已mask寄生功率>{DELTA_LCOE_MASK_PARASITIC_PCT}%的点为不可行区域（共{bad.sum()}个点）")
    
    # 为每个场景生成热力图
    for scenario in scenarios:
        print(f"--- 正在为 {scenario} 场景生成CNY/kWh单位的全局基准ΔLCOE热力图 ---")
        scenario_df = all_df[all_df["scenario"] == scenario].copy()
        output_dir = SCAN_CNY_DIR / scenario
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # 计算场景特定的norm（CNY/kWh单位）
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
            
            # 全局基准ΔLCOE 热力图（CNY/kWh）
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
            
            # 相对于最小值的ΔLCOE 热力图（CNY/kWh）
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
        
        # 保存colorbar
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
    固定设计点的详细经济分析
    """
    print("\n" + "="*50)
    print("--- 任务二：开始执行固定设计点的经济分析 ---")
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
        print("警告：固定点分析未能生成任何结果。")
        return
    
    economic_df = pd.DataFrame(all_economic_results)
    heat_load_df = pd.DataFrame(all_heat_load_results)
    
    # 添加人民币单位列
    if 'lcoe_plant_$/MWh' in economic_df.columns:
        economic_df['lcoe_fullplant_CNY_per_kWh'] = (
            economic_df['lcoe_plant_$/MWh'] * USD_TO_CNY_EXCHANGE_RATE / 1000
        )
    
    ECONOMIC_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    with pd.ExcelWriter(FINAL_ECONOMICS_FILE, engine='openpyxl') as writer:
        economic_df.to_excel(writer, sheet_name='Economic_Summary', index=False)
        heat_load_df.to_excel(writer, sheet_name='Heat_Load_Breakdown', index=False)
    
    print(f"经济分析数据已保存至: {FINAL_ECONOMICS_FILE}")
    
    # 调用所有相关的独立绘图函数
    print("\n--- 开始生成经济分析总结图表 ---")
    save_color_legend(ECONOMIC_OUTPUT_DIR / "legend_color_conditions."+cfg.PLOT_FORMAT)
    save_hatch_legend(ECONOMIC_OUTPUT_DIR / "legend_hatch_costs."+cfg.PLOT_FORMAT)
    plot_parasitic_power_vs_temp(str(FINAL_ECONOMICS_FILE), ECONOMIC_OUTPUT_DIR / "parasitic_power_vs_temp."+cfg.PLOT_FORMAT)
    
    print("所有经济分析图表已成功生成。")


# =============================================================================
# 4. 主程序执行入口
# =============================================================================
if __name__ == "__main__":
    print("启动统一经济分析脚本...")
    
    # 配置开关
    RUN_PARAMETRIC_SCAN = True      # 运行参数扫描
    RUN_FIXED_POINT_ANALYSIS = False  # 运行固定点分析
    ENABLE_LCOE_ANALYSIS = True      # 启用LCOE分析（USD）
    ENABLE_CNY_ANALYSIS = True       # 启用CNY单位分析（需要先运行USD版本）
    
    # 选择要分析的场景
    SCAN_SCENARIOS = list(cfg.SCENARIO_DEFINITIONS.keys())  # 或指定特定场景，如 ['S3']
    #SCAN_SCENARIOS = ['S3']
    
    full_params = define_parameters()
    
    # 创建输出目录
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
    
    print("\n统一经济分析脚本已执行完毕。")

