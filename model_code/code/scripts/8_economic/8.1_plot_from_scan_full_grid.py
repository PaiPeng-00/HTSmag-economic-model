"""
8.1_plot_from_scan_full_grid.py

从 scan_full_grid_tidy_plant_opex_2025usd_direct_hts.csv 读取数据并绘制经济指标热力图。

功能：
1. 读取 scan_full_grid_tidy_plant_opex_2025usd_direct_hts.csv 数据
2. 验证所需数据列是否存在
3. 计算 delta_LCOE_min（相对于场景内最小值的差分）
4. 绘制 r_cryo_re 再循环功率占比热力图
5. 绘制 delta_LCOE_min 热力图

如果数据不存在，报错提示需要运行 scan_full_grid.py 并设置相应的扫描维度。
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Set, Tuple

# 导入项目模块
from fusion_tem import device as cfg
from tfmag.paths import ensure_base_dirs
from fusion_tem.economic.feasibility import add_joint_lcoe_reference, apply_system_feasibility

# 导入绘图函数
from fusion_tem.plotting.library import (
    plot_parasitic_heatmap_single,
    save_parasitic_heatmap_colorbar,
    plot_delta_lcoe_heatmap_single,
    save_delta_lcoe_heatmap_colorbar,
    plot_cryo_power_heatmap_single,
)
import matplotlib as mpl

# =============================================================================
# 配置参数（与8.0保持一致）
# =============================================================================

# 绘图配置（来自 plotting 节）
DELTA_LCOE_CMAP = cfg.cmap_parasitic_ratio  # plotting.cmap_parasitic_ratio

# 经济学分析配置（来自 economic_analysis 节）
DELTA_LCOE_MASK_PARASITIC_PCT = cfg.DELTA_LCOE_MASK_PARASITIC_PCT  # economic_analysis.DELTA_LCOE_MASK_PARASITIC_PCT
DELTA_LCOE_ABS_PCTL = cfg.DELTA_LCOE_ABS_PCTL  # economic_analysis.DELTA_LCOE_ABS_PCTL
USD_TO_CNY_EXCHANGE_RATE = cfg.USD_TO_CNY_EXCHANGE_RATE  # economic_analysis.USD_TO_CNY_EXCHANGE_RATE

# 参数扫描配置（来自 parameter_scan_lists 和 economic_analysis 节）
# 与8.0使用相同的配置，确保绘制的热力图网格一致
# 注意：R_JOINT_SCAN_VALUES 现在以 nOhm 为单位，是对数数组（1~100 nOhm，共31个点）。
# 在需要以 Ohm 参与计算/插值时，本脚本会在使用前显式乘以 1e-9。
R_JOINT_SCAN_VALUES = cfg.R_JOINT_SCAN_VALUES  # nOhm，对数数组
NPW_SCAN_VALUES = cfg.NPW_SCAN_VALUES  # parameter_scan_lists.npw_scan_values
SCAN_TEMP_COOLANT_PAIRS = cfg.SCAN_TEMP_COOLANT_PAIRS  # economic_analysis.scan_temp_coolant_pairs

# Final manuscript figures use one economic metric only: scenario-specific
# full-plant LCOE written directly by scan_full_grid.py.
import os
from fusion_tem.economic.cost_boundary import COST_BOUNDARY_VERSION

LCOE_COL = "LCOE_plant_USD_per_MWh"
METRIC_TAG = COST_BOUNDARY_VERSION
print(f"[8.1] LCOE 指标源列 = {LCOE_COL}  ->  输出子目录 metric = {METRIC_TAG}")

# ============================================================================
# 逐子图自定义 ΔLCOE 等高线级别 (arc_16pancake_nuc600 手工调参)。
# key = (scenario, temperature_K(float), coolant); value = 等高线级别列表。
# 命中的子图用指定级别; 未命中的仍走自适应 auto_contour_levels。
# ============================================================================
DELTA_CONTOUR_OVERRIDES = {
    ("S1", 4.2, "He"): [10, 20, 40, 100, 200],
    ("S1", 10.0, "He"): [1, 5, 10, 50, 200],
    ("S1", 20.0, "He"): [12, 15, 20, 30, 60],
    ("S1", 20.0, "H2"): [12, 15, 20, 30, 60],
    ("S2", 4.2, "He"): [5, 10, 20, 40, 200],
    ("S2", 10.0, "He"): [0.5, 1, 2, 5, 10],
    ("S2", 20.0, "He"): [2, 3, 5, 10],
    ("S2", 20.0, "H2"): [2, 3, 5, 10],
    ("S3", 4.2, "He"): [5, 10, 20, 40, 100],
    ("S3", 10.0, "He"): [1.5, 2, 5, 10, 20],
    ("S3", 20.0, "He"): [0.2, 0.5, 1, 2, 5],
    ("S3", 20.0, "H2"): [0.2, 0.5, 1, 2, 5],
}

# 开关: True=用上面的逐子图手调覆盖; False=全部交给新版 auto_contour_levels(非均匀热力图法,
# 自动给出 floor避贴底 + cap标签必显 + 视觉等距的整齐等高线, 无需逐面板手调)。
# 字典保留仅作回退/参考。默认 False; 需要用回手调时设环境变量 USE_DELTA_OVERRIDES=1。
USE_DELTA_CONTOUR_OVERRIDES = os.environ.get("USE_DELTA_OVERRIDES", "0") == "1"

# Fig. 5/6 的统一显示下限：仅画 Npw >= NPW_MIN_HEATMAP 的区域。
# 这是固定 rho_turn=10000 μΩ·cm² 切片的保守显示选择，不是 feasibility
# 判据；120 h 充电时间仅保留为诊断量。可用环境变量覆盖。
#
# 2026-07-26: 默认由 5 改为 1。原来的 5 是"保证 Charging_999 < 120 h"的保守裁剪，
# 但它恰好切掉了 LCOE-Npw 曲线被双侧夹逼的左半支：Npw 越小电感越大 -> 充电越慢
# (Npw=1 时 427 h) -> AF_ref 掉到 0.91 而不可行；Npw 越大接头越多 -> r_cryo 升到 1.25。
# 内部最优(S2/20 K He/R_j=1 nΩ 处在 Npw=40)正是这两侧共同夹出来的，
# 裁到 5 就只剩右半支，读者看不出左侧的充电延迟惩罚。
# 另注：拼图器的 Y_AXIS_MIN 一直写死为 1，而裁到 5 时面板 ylim_min 实为 5，
# 二者不一致会让纵轴刻度有约 2% 面板高度的系统性错位；下延到 1 后该假设才成立。
NPW_MIN_HEATMAP = int(os.environ.get("NPW_MIN_HEATMAP", "1"))

# 输出目录可重定向到独立 P0 文件夹，避免覆盖旧图和改变插图状态。
FIGURE_OUTPUT_ROOT = Path(
    os.environ.get("SCAN_FIGURE_OUTPUT_ROOT", cfg.ECONOMIC_FIGURES_DIR)
)
SCAN_BASE_DIR = FIGURE_OUTPUT_ROOT / cfg.PARASITIC_RATIO_OUTPUT_DIR
SCAN_USD_DIR = SCAN_BASE_DIR / cfg.PARASITIC_RATIO_USD_DIR  # io_paths.PARASITIC_RATIO_USD_DIR
SCAN_CNY_DIR = SCAN_BASE_DIR / cfg.PARASITIC_RATIO_CNY_DIR  # io_paths.PARASITIC_RATIO_CNY_DIR
# 制冷功率 MWe 单图输出子目录名（在各配色方案下 parasitic_ratio/cryo_power_MWe）
CRYO_POWER_MWE_SUBDIR = "cryo_power_MWe"

# 快速调试模式 FAST_FIG=1: 跳过 CNY、cryo_power、AF_ref, 配色只画 YlGnBu_trunc_0p8
# (配色由 device.color_schemes 按同一变量控制)。定稿全量出图时不设此变量。
FAST_FIG = os.environ.get("FAST_FIG", "").lower() in ("1", "true", "yes")
# CNY 功能开关
ENABLE_CNY = False if FAST_FIG else True

# 描边（Halo/Stroke）功能开关
USE_STROKE = cfg.USE_STROKE  # 设置为 True 以启用描边技术（黑色等高线+白色描边）

# 输入数据文件
paths = ensure_base_dirs()
INPUT_CSV = Path(
    os.environ.get(
        "SCAN_INPUT_CSV",
        paths.outputs_tables / "scan_full_grid_tidy_plant_opex_2025usd_direct_hts.csv",
    )
)
if not INPUT_CSV.exists():
    raise FileNotFoundError(
        "Plant-v3/2025-USD scan input is missing; refusing to fall back to a legacy schema: "
        f"{INPUT_CSV}"
    )

# =============================================================================
# 辅助函数
# =============================================================================

def match_r_joint_by_significant_digits(r1_nohm: float, r2_nohm: float, n_digits: int = 3) -> bool:
    """
    比较两个接头电阻值（单位：nOhm）在保留 n 位小数后的数值是否匹配。
    
    参数:
        r1_nohm: 第一个电阻值（nOhm）
        r2_nohm: 第二个电阻值（nOhm）
        n_digits: 保留的小数位数（默认3）
    
    返回:
        如果在保留 n 位小数后数值相等，返回True；否则返回False
    """
    try:
        # 统一按 n_digits 位小数进行四舍五入后比较
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
    确保数据网格包含所有 NPW 和 R_joint 的组合，缺失的组合用 NaN 填充。
    这样可以确保绘制的热力图网格与 8.0 完全一致。
    
    Args:
        df: 输入DataFrame，必须包含 'Npw', 'R_joint', 和 value_col 列
        npw_values: 完整的 Npw 值数组（来自 NPW_SCAN_VALUES）
        r_joint_values: 完整的 R_joint 值数组（来自 R_JOINT_SCAN_VALUES，单位：Ω）
        value_col: 要填充的值列名
    
    Returns:
        包含完整网格的DataFrame
    """
    # 创建完整的网格
    grid_data = []
    for npw in npw_values:
        for r_joint in r_joint_values:
            grid_data.append({
                'Npw': npw,
                'R_joint': r_joint
            })
    
    grid_df = pd.DataFrame(grid_data)
    
    # 将实际数据合并到网格中
    df_work = df.copy()
    
    # 如果 df 中有 R_joint_nOhm，需要转换为 R_joint
    if 'R_joint_nOhm' in df_work.columns and 'R_joint' not in df_work.columns:
        df_work['R_joint'] = df_work['R_joint_nOhm'] * 1e-9  # nΩ -> Ω
    
    # 检查必要的列
    if 'Npw' not in df_work.columns or 'R_joint' not in df_work.columns:
        grid_df[value_col] = np.nan
        return grid_df
    
    # 使用容差匹配合并数据（因为可能有浮点数精度问题）
    merged_df = grid_df.copy()
    merged_df[value_col] = np.nan
    
    if value_col in df_work.columns:
        # 对每个网格点，找到最匹配的数据
        for idx, row in grid_df.iterrows():
            npw = row['Npw']
            r_joint = row['R_joint']
            
            # 找到匹配的数据（使用容差）
            # 对于 R_joint，使用更宽松的容差（因为可能有浮点数精度问题）
            mask = (
                (np.isclose(df_work['Npw'], npw, rtol=0.01)) &
                (np.isclose(df_work['R_joint'], r_joint, rtol=0.01, atol=1e-12))
            )
            if mask.any():
                # 如果有多个匹配，取平均值（更合理）
                matched_indices = df_work.index[mask]
                if len(matched_indices) > 0:
                    matched_values = df_work.loc[matched_indices, value_col]
                    # 只取非 NaN 值的平均值
                    valid_values = matched_values.dropna()
                    if len(valid_values) > 0:
                        merged_df.loc[idx, value_col] = valid_values.mean()
                    else:
                        merged_df.loc[idx, value_col] = np.nan
    
    # 确保返回的 DataFrame 没有重复的 (Npw, R_joint) 组合
    # 如果 grid_df 本身有重复（不应该发生），进行去重
    if merged_df.duplicated(subset=['Npw', 'R_joint']).any():
        merged_df = merged_df.groupby(['Npw', 'R_joint'], as_index=False)[value_col].first()
    
    return merged_df

def check_required_columns(df: pd.DataFrame, required_cols: List[str]) -> Tuple[bool, List[str]]:
    """
    检查DataFrame是否包含所需的列。
    
    Returns:
        (是否全部存在, 缺失的列列表)
    """
    missing_cols = [col for col in required_cols if col not in df.columns]
    return len(missing_cols) == 0, missing_cols


def get_missing_scan_dimensions(df: pd.DataFrame, required_cols: List[str]) -> dict:
    """
    根据缺失的列，推断需要扫描的维度。
    
    Returns:
        字典，包含缺失的扫描维度信息
    """
    missing_info = {}
    
    # 检查基础参数列
    if 'Top_K' not in df.columns:
        missing_info['Top_K'] = "需要扫描温度维度"
    if 'coolant' not in df.columns:
        missing_info['coolant'] = "需要扫描制冷剂维度"
    if 'Npw' not in df.columns:
        missing_info['Npw'] = "需要扫描并绕根数维度"
    if 'R_joint_nOhm' not in df.columns:
        missing_info['R_joint_nOhm'] = "需要扫描接头电阻维度"
    if 'scenario' not in df.columns:
        missing_info['scenario'] = "需要扫描场景维度"
    
    # 检查数据列
    if 'r_cryo_re' not in df.columns:
        missing_info['r_cryo_re'] = "需要计算经济学指标（r_cryo_re）"
    if LCOE_COL not in df.columns:
        missing_info[LCOE_COL] = "需要计算经济学指标（LCOE）"
    
    return missing_info


def build_isocharge_slice(df: pd.DataFrame, target_h: float,
                          charge_col: str = 'Charging_time_999_h') -> pd.DataFrame:
    """把数据沿 rho 插值到"固定充电时间 = target_h(小时)"的切片(替代固定-rho 切片)。

    物理: rho(匝间电阻率)是设计自由度; 对每个 Npw 取"恰好达到 target_h 充电时间"的 rho*,
    使全域可用度近似恒定(如 130h ≈ AF_ref 0.99)。可用度不再是混淆项 => ΔLCOE 随 Npw、R_joint
    单调递增、无回环/贴底。充电时间与 R_joint 无关, 故每个 (…,Npw) 只有一个 rho*(Npw);
    达不到 target_h 的 Npw(高 Npw 本就充电更快, 低 Npw 即便最高 rho 也太慢)被自然剔除。

    实现: 对每个 (scenario,Top_K,coolant,Npw,R_joint) 组, 在 log10(rho) 上线性插值所有数值列到
    log10(rho*)。rho_turn_uOhm_cm2 统一标记为 target_h(使下游"固定-rho"过滤全通过);
    真实 rho*(Npw) 存入新列 rho_star_uOhm_cm2 备查。
    """
    if charge_col not in df.columns or 'rho_turn_uOhm_cm2' not in df.columns:
        print(f"[ISO][警告] 缺列 {charge_col}/rho_turn_uOhm_cm2, 跳过 iso 切片")
        return df
    id_cols = ['scenario', 'Top_K', 'coolant', 'Npw', 'R_joint_nOhm']
    num_cols = [c for c in df.columns
                if c not in id_cols + ['rho_turn_uOhm_cm2']
                and pd.api.types.is_numeric_dtype(df[c])]
    out = []
    for (sc, T, cool, N), g in df.groupby(['scenario', 'Top_K', 'coolant', 'Npw'], sort=False):
        cg = g.groupby('rho_turn_uOhm_cm2')[charge_col].mean().sort_index()
        rho = cg.index.values.astype(float)
        ct = cg.values.astype(float)
        m = np.isfinite(ct)
        if m.sum() < 2:
            continue
        rho, ct = rho[m], ct[m]
        if target_h < ct.min() or target_h > ct.max():
            continue  # 该 Npw 达不到该充电时间 -> 剔除
        order = np.argsort(ct)  # charging 随 rho 单调降; 以 ct 升序插 log-rho
        rho_star = float(10.0 ** np.interp(target_h, ct[order], np.log10(rho)[order]))
        xq = np.log10(rho_star)
        for Rj, gr in g.groupby('R_joint_nOhm', sort=False):
            gr = gr.sort_values('rho_turn_uOhm_cm2')
            lrr = np.log10(gr['rho_turn_uOhm_cm2'].values.astype(float))
            row = {'scenario': sc, 'Top_K': T, 'coolant': cool, 'Npw': N,
                   'R_joint_nOhm': Rj, 'rho_turn_uOhm_cm2': float(target_h),
                   'rho_star_uOhm_cm2': rho_star}
            for c in num_cols:
                yv = gr[c].values.astype(float)
                mm = np.isfinite(yv)
                if mm.sum() >= 2:
                    row[c] = float(np.interp(xq, lrr[mm], yv[mm]))
                elif mm.any():
                    row[c] = float(yv[mm][0])
                else:
                    row[c] = np.nan
            out.append(row)
    if not out:
        print("[ISO][警告] iso 切片结果为空, 回退原数据")
        return df
    return pd.DataFrame(out)


def calculate_delta_lcoe_min(df: pd.DataFrame, scenarios: List[str]) -> pd.DataFrame:
    """Use the global scenario reference over jointly feasible designs."""
    out = df.copy()
    required = {
        "AF_ref_system",
        "r_cryo_re_fraction",
        "feasible_cryo_050",
        "model_error",
        "feasible_joint",
    }
    if not required.issubset(out.columns):
        out, _ = apply_system_feasibility(out)

    if "delta_LCOE_joint_feasible_USD_per_MWh" not in out.columns:
        out, references = add_joint_lcoe_reference(out)
        print(f"[Info] Joint-feasible LCOE references: {references.to_dict()}")

    out["delta_LCOE_min_USD_per_MWh"] = out[
        "delta_LCOE_joint_feasible_USD_per_MWh"
    ]
    return out

def format_array_for_print(arr: np.ndarray) -> str:
    """格式化numpy数组用于打印"""
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


# =============================================================================
# 主函数
# =============================================================================

def plot_from_scan_full_grid(
    scenarios: List[str] = None,
    temp_coolant_pairs: List[Tuple[float, str]] = None,
    enable_cny: bool = False
):
    """
    从 scan_full_grid_tidy_plant_opex_2025usd_direct_hts.csv 读取数据并绘制热力图。
    
    Args:
        scenarios: 要绘制的场景列表，如果为None则从数据中自动获取
        temp_coolant_pairs: 要绘制的温度-制冷剂组合列表，如果为None则从数据中自动获取
        enable_cny: 是否启用 CNY 功能，如果为 True 则同时生成 CNY 版本的热力图
    """
    print("\n" + "="*60)
    print("--- 从 scan_full_grid_tidy_plant_opex_2025usd_direct_hts.csv 读取数据并绘制热力图 ---")
    print("="*60)
    
    # 1. 检查输入文件是否存在
    if not INPUT_CSV.exists():
        print(f"\n[错误] 输入文件不存在: {INPUT_CSV}")
        print("请先运行 scan_full_grid.py 生成数据。")
        return
    
    # 2. 读取数据
    print(f"\n[Info] 读取数据文件: {INPUT_CSV}")
    try:
        df = pd.read_csv(INPUT_CSV, low_memory=False)
        versions = set(df["cost_boundary_version"].dropna().astype(str))
        if versions != {COST_BOUNDARY_VERSION}:
            raise ValueError(
                "2025-USD figure input has the wrong cost boundary: "
                f"{sorted(versions)}"
            )
        basis_years = set(
            pd.to_numeric(
                df["monetary_price_basis_year"], errors="coerce"
            ).dropna()
        )
        if basis_years != {2025.0}:
            raise ValueError(
                f"2025-USD figure input has wrong price-basis years: {basis_years}"
            )
        print(f"[OK] 成功读取 {len(df):,} 行数据")
    except Exception as e:
        print(f"[错误] 读取文件失败: {e}")
        return
    
    if df.empty:
        print("[错误] 数据文件为空")
        return

    # 2.4 iso-充电切片(可选, 早于 C0/delta 重算): 固定充电时间=ISO_CHARGE_H(h) 而非固定 rho。
    #     每个 Npw 取达此充电时间的 rho* => 可用度近似恒定(130h≈AF_ref 0.99), ΔLCOE 随 Npw/Rj 单调无回环。
    _iso_h = os.environ.get("ISO_CHARGE_H", "").strip()
    if _iso_h:
        _h = float(_iso_h)
        _n0 = len(df)
        df = build_isocharge_slice(df, _h)
        os.environ["RHO_TURN_HEATMAP"] = str(_h)  # 下游"固定-rho"过滤将全通过(所有行 rho 标记=_h)
        os.environ["NPW_MIN_HEATMAP"] = "1"       # 旧"充电可行性 Npw≥5"下限在 iso 模式已被切片本身取代
        _npws = sorted(df['Npw'].unique()) if 'Npw' in df.columns else []
        # iso 窗口随温度略移(4.2K→2-19, 10/20K→3-20)。为拼图纵轴统一, 用固定公共窗 [3,20]:
        # 保住重要的 10/20K 面板满格; 4.2K 的 Npw=20 本就不可达 130h → 诚实 hatch 一行。
        _iso_nmin = int(os.environ.get("ISO_NPW_MIN", "3"))
        _iso_nmax = int(os.environ.get("ISO_NPW_MAX", "20"))
        globals()['NPW_SCAN_VALUES'] = np.arange(_iso_nmin, _iso_nmax + 1)  # 不再补满 1..200(否则高 Npw 全 hatch)
        print(f"[ISO] iso-充电切片 {_h}h: {_n0:,}→{len(df):,} 行; "
              f"Npw 窗口={min(_npws) if _npws else '?'}..{max(_npws) if _npws else '?'} ({len(_npws)} 个)")

    # Full-plant LCOE is already scenario-specific in the scan CSV.
    # No post-hoc C0 rescaling is allowed in the plotting layer.

    # 3. 检查必需的列
    required_cols = [
        'Top_K', 'coolant', 'Npw', 'R_joint_nOhm', 'scenario',
        'r_cryo_re', 'LCOE_plant_USD_per_MWh'
    ]
    
    all_present, missing_cols = check_required_columns(df, required_cols)
    
    if not all_present:
        print(f"\n[错误] 数据文件缺少必需的列:")
        for col in missing_cols:
            print(f"  - {col}")
        
        missing_info = get_missing_scan_dimensions(df, missing_cols)
        print(f"\n[提示] 需要在 scan_full_grid.py 中设置以下扫描维度:")
        for col, info in missing_info.items():
            print(f"  - {col}: {info}")
        
        print("\n请运行 scan_full_grid.py 并确保包含以下扫描维度:")
        print("  - TEMP_COOLANT_PAIRS: 温度-制冷剂组合")
        print("  - NPW_RANGE: 并绕根数范围")
        print("  - R_JOINT_NOHM: 接头电阻值列表")
        print("  - SCENARIOS: 场景列表")
        return
    
    # 4. 确定要绘制的场景和温度-制冷剂组合（使用与8.0相同的配置）
    if scenarios is None:
        # 从数据中获取场景，但只使用配置中定义的场景
        available_scenarios = sorted(df['scenario'].unique().tolist())
        # 如果配置中有场景定义，优先使用配置中的场景
        if hasattr(cfg, 'SCENARIO_DEFINITIONS'):
            config_scenarios = list(cfg.SCENARIO_DEFINITIONS.keys())
            scenarios = [s for s in config_scenarios if s in available_scenarios]
            if not scenarios:
                scenarios = available_scenarios
        else:
            scenarios = available_scenarios
        print(f"\n[Info] 使用场景: {scenarios}")

    if temp_coolant_pairs is None:
        # 使用与8.0相同的配置
        temp_coolant_pairs = SCAN_TEMP_COOLANT_PAIRS
        print(f"[Info] 使用温度-制冷剂组合（来自配置）: {temp_coolant_pairs}")
    
    # 5. 计算 delta_LCOE_min
    df = calculate_delta_lcoe_min(df, scenarios)

    
    # 6. 计算全局norm（用于parasitic ratio）
    if 'r_cryo_re' in df.columns:
        r_cryo_series = df['r_cryo_re'].replace([np.inf, -np.inf], np.nan).dropna()
        if not r_cryo_series.empty:
            global_vmin = float(r_cryo_series.min())
            global_vmax = float(r_cryo_series.max())
            global_norm_parasitic = mpl.colors.LogNorm(
                vmin=float(f"{global_vmin:.10f}"),
                vmax=float(f"{global_vmax:.10f}")
            )
            print(f"\n[Info] r_cryo_re 全局colorbar范围: {global_vmin:.3g}% - {global_vmax:.3g}%")
        else:
            print("[警告] r_cryo_re 数据为空，无法绘制热力图")
            global_norm_parasitic = None
    else:
        global_norm_parasitic = None
    
    # 7. 计算 delta_LCOE 的全局norm
    if 'delta_LCOE_min_USD_per_MWh' in df.columns:
        # Mask不可行点
        # Joint feasibility is AF_ref_system>=0.99, r_cryo,re<=0.50,
        # finite LCOE, and no model error. In the fixed-rho, Npw>=5 display
        # slice all valid points pass availability, so hatching denotes the
        # explicit TF-cryogenic boundary.
        bad = ~df["feasible_joint"].astype(bool)
        df.loc[bad, "delta_LCOE_min_USD_per_MWh"] = np.nan
        print(f"[Info] Masked {int(bad.sum())} points outside the joint feasibility gate.")
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
            # 注释掉：使用 P99.5 百分位数筛选 delta_bound
            # delta_bound = float(np.nanpercentile(np.abs(delta_series.values), DELTA_LCOE_ABS_PCTL))
            
            # 色标 vmax 用 P90 分位(不是 max): 避免极少数不可行配置的巨大 ΔLCOE(可达数万)
            # 把色标尺度撑到上万, 从而压扁可行区(0~50)、导致等高线全挤在低端/上方无线。
            _dfin = delta_series.replace([np.inf, -np.inf], np.nan).dropna()
            delta_min = float(_dfin.min())
            delta_max = float(np.nanpercentile(_dfin, 90)) if len(_dfin) else float(delta_series.max())
            
            if delta_max > delta_min:  # 确保有有效范围
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
                # 注释掉：P99.5 筛选相关的打印信息
                # print(f"[Info] ΔLCOE_min 色标范围采用 |Δ| 的 P{DELTA_LCOE_ABS_PCTL}: ±{delta_bound:.3g} ($/MWh)")
                print(f"[Info] ΔLCOE_min 色标范围: [{delta_min:.3g}, {delta_max:.3g}] ($/MWh)")
                
                # 注释掉：使用 delta_bound 进行 clip 筛选
                # all_delta_clipped = df['delta_LCOE_min_USD_per_MWh'].clip(-delta_bound, delta_bound)
                # all_finite = all_delta_clipped.replace([np.inf, -np.inf], np.nan).dropna().values
                # global_contour_levels = calculate_lcoe_contour_levels(all_finite, delta_bound, unit='USD')
                
                # 不使用 clip，直接使用所有有限值
                all_finite = delta_series.replace([np.inf, -np.inf], np.nan).dropna().values
                # 使用实际的最大绝对值作为 delta_bound 用于计算 contour levels
                delta_bound = float(np.nanpercentile(np.abs(all_finite), 90)) if len(all_finite) else 1.0
                global_contour_levels = calculate_lcoe_contour_levels(all_finite, delta_bound, unit='USD')
    else:
        delta_norm = None
        delta_bound = None
        delta_min = None
        delta_max = None
        global_contour_levels = None
    
    # 8. 检查数据完整性（在开始绘制前）
    print("\n--- 检查数据完整性 ---")
    # 使用扫描配置检查数据完整性（与8.0保持一致）
    npw_array = np.array(NPW_SCAN_VALUES)
    r_joint_array = np.array(R_JOINT_SCAN_VALUES, dtype=float)  # nOhm，对数数组，用于检查
    expected_total_points = len(npw_array) * len(r_joint_array)
    
    data_completeness_warnings = []
    for scenario in scenarios:
        scenario_df = df[df['scenario'] == scenario].copy()
        if scenario_df.empty:
            data_completeness_warnings.append(f"  [警告] {scenario}: 场景数据为空")
            continue
        
        for temp, cool in temp_coolant_pairs:
            temp_df = scenario_df[
                (scenario_df['Top_K'] == temp) &
                (scenario_df['coolant'] == cool)
            ].copy()
            
            if temp_df.empty:
                data_completeness_warnings.append(f"  [警告] {scenario} - {temp}K {cool}: 无数据")
                continue
            
            # 筛选 rho_turn_uOhm_cm2 = 10000 的数据
            RHO_TURN_FOR_HEATMAP = float(os.environ.get("RHO_TURN_HEATMAP", "10000"))
            if 'rho_turn_uOhm_cm2' in temp_df.columns:
                available_rhot = temp_df['rho_turn_uOhm_cm2'].unique()
                if len(available_rhot) > 0:
                    closest_rhot = available_rhot[np.argmin(np.abs(available_rhot - RHO_TURN_FOR_HEATMAP))]
                    temp_df = temp_df[np.isclose(temp_df['rho_turn_uOhm_cm2'], closest_rhot, rtol=0.01)].copy()
            
            if temp_df.empty:
                continue
            
            # 筛选配置中的值
            if 'Npw' in temp_df.columns:
                temp_df = temp_df[temp_df['Npw'].isin(npw_array)].copy()
            if 'R_joint_nOhm' in temp_df.columns:
                r_joint_nohm_array = r_joint_array  # nOhm
                # 使用前4个有效数字匹配
                temp_df = temp_df[
                    temp_df['R_joint_nOhm'].apply(
                        lambda x: any(match_r_joint_by_significant_digits(x, expected_r, n_digits=4)
                                     for expected_r in r_joint_nohm_array)
                    )
                ].copy()
            
            if temp_df.empty:
                data_completeness_warnings.append(f"  [警告] {scenario} - {temp}K {cool}: 筛选后无数据")
                continue
            
            # 检查数据点数量（使用保留3位小数匹配去重）
            # 将数据中的R_joint_nOhm标准化为期望值（使用保留3位小数匹配）
            r_joint_nohm_array = r_joint_array  # nOhm
            normalized_rows = []
            for _, row in temp_df.iterrows():
                npw = int(row['Npw'])
                r_joint_nohm = float(row['R_joint_nOhm'])
                # 找到匹配的期望值（使用保留3位小数）
                matched_r = None
                for expected_r in r_joint_nohm_array:
                    if match_r_joint_by_significant_digits(r_joint_nohm, expected_r, n_digits=3):
                        matched_r = expected_r
                        break
                if matched_r is not None:
                    normalized_rows.append((npw, matched_r))
            
            # 去重
            actual_points = len(set(normalized_rows))
            missing_points = expected_total_points - actual_points
            missing_ratio = missing_points / expected_total_points if expected_total_points > 0 else 1.0
            
            if missing_ratio > 0.3:  # 如果缺失超过30%
                data_completeness_warnings.append(
                    f"  [警告] {scenario} - {temp}K {cool}: "
                    f"数据点不足 ({actual_points}/{expected_total_points}, 缺失 {missing_ratio*100:.1f}%)"
                )
    
    if data_completeness_warnings:
        print("\n数据完整性检查结果:")
        for warning in data_completeness_warnings:
            print(warning)
        # 输出缺失的组合
        print("\n缺失的 (Npw, R_joint_nOhm) 组合明细：")
        for scenario in scenarios:
            scenario_df = df[df['scenario'] == scenario].copy()
            for temp, cool in temp_coolant_pairs:
                temp_df = scenario_df[(scenario_df['Top_K'] == temp) & (scenario_df['coolant'] == cool)].copy()
                if temp_df.empty:
                    continue
                npw_array = np.array(NPW_SCAN_VALUES)
                r_joint_array = np.array(R_JOINT_SCAN_VALUES, dtype=float)  # nOhm，对数数组检查
                r_joint_nohm_array = r_joint_array  # 已经是 nOhm
                expected_grid = set((n, r) for n in npw_array for r in r_joint_nohm_array)
                
                # 当前已有的数据点（使用保留3位小数匹配）
                if 'R_joint_nOhm' in temp_df.columns and 'Npw' in temp_df.columns:
                    existing_grid = set()
                    for _, row in temp_df.drop_duplicates(subset=['Npw', 'R_joint_nOhm']).iterrows():
                        npw = int(row['Npw'])
                        r_joint_nohm = float(row['R_joint_nOhm'])
                        # 找到匹配的期望值（使用保留3位小数）
                        matched_r = None
                        for expected_r in r_joint_nohm_array:
                            if match_r_joint_by_significant_digits(r_joint_nohm, expected_r, n_digits=3):
                                matched_r = expected_r
                                break
                        if matched_r is not None:
                            existing_grid.add((npw, matched_r))
                    
                    missing_grid = expected_grid - existing_grid
                    if missing_grid:
                        print(f"  缺失 ({scenario}-{temp}K-{cool}): 共{len(missing_grid)}点")
                        # 分组输出
                        for npw in npw_array:
                            missing_rj = [r for (n, r) in missing_grid if n==npw]
                            if missing_rj:
                                print(f"    Npw={npw}: 缺失 R_joint_nOhm={missing_rj}")
        print("")            
        print("\n提示: 如果数据不足，可能导致热力图出现大量空白区域。")
        print("      建议运行 scan_full_grid.py 确保包含所有必要的扫描维度。\n")
    
    # 8b. 制冷功率 MWe：预计算全局 norm，供各配色方案共用（单图在 9 的配色循环内生成）
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
                    RHO_TURN_FOR_HEATMAP = float(os.environ.get("RHO_TURN_HEATMAP", "10000"))
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
            print(f"[Info] 跳过制冷功率 MWe 单图（缺少列: {missing_cryo}）")

    # 9. 为每个配色方案生成热力图
    for cmap_name in cfg.color_schemes:
        print(f"\n处理配色方案: {cmap_name}")
        # 为每个配色方案创建单独的目录：outputs/figures/economic/{cmap_name}/parasitic_ratio/USD
        cmap_base_dir = FIGURE_OUTPUT_ROOT / cmap_name
        cmap_scan_base_dir = cmap_base_dir / cfg.PARASITIC_RATIO_OUTPUT_DIR / METRIC_TAG
        cmap_scan_usd_dir = cmap_scan_base_dir / cfg.PARASITIC_RATIO_USD_DIR
        cmap_scan_usd_dir.mkdir(exist_ok=True, parents=True)
        
        # 制冷功率 MWe 单图：存于同一配色方案下 parasitic_ratio/cryo_power_MWe (FAST_FIG 跳过)
        if (not FAST_FIG) and norm_cryo is not None and scenario_cryo_df is not None:
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
                        RHO_TURN_FOR_HEATMAP = float(os.environ.get("RHO_TURN_HEATMAP", "10000"))
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
            print(f"[OK] 制冷功率 MWe 单图已保存至: {cmap_cryo_dir}")
        
        # 为每个场景生成热力图
        for scenario in scenarios:
            scenario_df = df[df['scenario'] == scenario].copy()
            
            if scenario_df.empty:
                continue
            
            output_dir = cmap_scan_usd_dir / scenario
            output_dir.mkdir(exist_ok=True, parents=True)
            
            # 为每个温度-制冷剂组合绘制热力图
            for temp, cool in temp_coolant_pairs:
                temp_df = scenario_df[
                    (scenario_df['Top_K'] == temp) &
                    (scenario_df['coolant'] == cool)
                ].copy()
                
                if temp_df.empty:
                    continue
                
                # 筛选 rho_turn_uOhm_cm2 = 10000 的数据（用于热力图）
                RHO_TURN_FOR_HEATMAP = float(os.environ.get("RHO_TURN_HEATMAP", "10000"))  # μΩ·cm²
                if 'rho_turn_uOhm_cm2' in temp_df.columns:
                    # 找到最接近 10000 的值（允许一定容差）
                    available_rhot = temp_df['rho_turn_uOhm_cm2'].unique()
                    if len(available_rhot) > 0:
                        closest_rhot = available_rhot[np.argmin(np.abs(available_rhot - RHO_TURN_FOR_HEATMAP))]
                        temp_df = temp_df[
                            np.isclose(temp_df['rho_turn_uOhm_cm2'], closest_rhot, rtol=0.01)
                        ].copy()
                
                if temp_df.empty:
                    continue
                
                # 筛选数据：只使用与8.0相同的配置参数对应的数据点
                # 确保 Npw 在 NPW_SCAN_VALUES 中
                if 'Npw' in temp_df.columns:
                    npw_array = np.array(NPW_SCAN_VALUES)
                    temp_df = temp_df[temp_df['Npw'].isin(npw_array)].copy()
                
                # 确保 R_joint_nOhm 在 R_JOINT_SCAN_VALUES 中（与8.0保持一致，单位：nOhm）
                if 'R_joint_nOhm' in temp_df.columns:
                    r_joint_nohm_array = np.array(R_JOINT_SCAN_VALUES, dtype=float)  # nOhm
                    # 使用保留3位小数匹配（因为可能有浮点数精度问题）
                    temp_df = temp_df[
                        temp_df['R_joint_nOhm'].apply(
                            lambda x: any(match_r_joint_by_significant_digits(x, expected_r, n_digits=3)
                                         for expected_r in r_joint_nohm_array)
                        )
                    ].copy()
                
                if temp_df.empty:
                    continue
                
                # 检查是否有重复的 (Npw, R_joint_nOhm) 组合
                duplicate_mask = temp_df.duplicated(subset=['Npw', 'R_joint_nOhm'], keep='first')
                if duplicate_mask.any():
                    temp_df = temp_df.drop_duplicates(subset=['Npw', 'R_joint_nOhm'], keep='first')
                
                # 准备数据：转换为绘图函数期望的格式
                temp_df_plot = temp_df.copy()
                if 'R_joint_nOhm' in temp_df_plot.columns:
                    temp_df_plot['R_joint'] = temp_df_plot['R_joint_nOhm'] * 1e-9  # nΩ -> Ω (绘图函数期望Ω单位)
                
                # 获取配置中的完整值数组（与8.0保持一致）
                # 注意：R_JOINT_SCAN_VALUES 为 nOhm，这里转换为 Ohm 创建网格；
                # 绘图函数内部会再次根据这些点进行插值和对数刻度绘制。
                npw_array = np.array(NPW_SCAN_VALUES)
                r_joint_array = np.array(R_JOINT_SCAN_VALUES, dtype=float) * 1e-9  # 转换为 Ω，用于网格/插值
                
                # 绘制 r_cryo_re 热力图（parasitic ratio）
                if global_norm_parasitic is not None and 'r_cryo_re' in temp_df_plot.columns:
                    # 确保完整的数据网格（包含所有 NPW_SCAN_VALUES 和 R_JOINT_SCAN_VALUES_PLOT 的组合）
                    # 绘图函数内部会使用 R_JOINT_SCAN_VALUES_PLOT 进行插值
                    complete_df_parasitic = ensure_complete_data_grid(
                        temp_df_plot,
                        npw_array,
                        r_joint_array,
                        'r_cryo_re'
                    )
                    
                    # 检查数据完整性
                    total_points = len(complete_df_parasitic)
                    valid_points = complete_df_parasitic['r_cryo_re'].notna().sum()
                    missing_ratio = 1.0 - (valid_points / total_points) if total_points > 0 else 1.0
                    
                    if missing_ratio > 0.1:  # 如果缺失超过10%
                        print(f"  [警告] {scenario} - {temp}K {cool}: r_cryo_re 数据不完整")
                        print(f"          缺失数据点: {total_points - valid_points}/{total_points} ({missing_ratio*100:.1f}%)")
                        print(f"          可能导致热力图出现大量空白区域")
                    
                    # 重命名列以匹配绘图函数期望的格式
                    complete_df_parasitic['r_parasitic_pct'] = complete_df_parasitic['r_cryo_re']
                    
                    # 使用当前配色方案绘制
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
                
                # 绘制 delta_LCOE_min 热力图
                if delta_norm is not None and 'delta_LCOE_min_USD_per_MWh' in temp_df_plot.columns:
                    # 确保完整的数据网格（包含所有 NPW_SCAN_VALUES 和 R_JOINT_SCAN_VALUES_PLOT 的组合）
                    # 绘图函数内部会使用 R_JOINT_SCAN_VALUES_PLOT 进行插值
                    complete_df_delta = ensure_complete_data_grid(
                        temp_df_plot,
                        npw_array,
                        r_joint_array,
                        'delta_LCOE_min_USD_per_MWh'
                    )
                    
                    # 检查数据完整性
                    total_points = len(complete_df_delta)
                    valid_points = complete_df_delta['delta_LCOE_min_USD_per_MWh'].notna().sum()
                    missing_ratio = 1.0 - (valid_points / total_points) if total_points > 0 else 1.0
                    
                    if missing_ratio > 0.1:  # 如果缺失超过10%
                        print(f"  [警告] {scenario} - {temp}K {cool}: delta_LCOE_min 数据不完整")
                        # 输出缺失的 (npw, rjoint) 组合
                        missing_rows = complete_df_delta[complete_df_delta['delta_LCOE_min_USD_per_MWh'].isna()]
                        for _, row in missing_rows.iterrows():
                            # Npw 按整数输出；R_joint 以 nΩ 输出，并按 4 位有效数字展示/对比
                            npw_i = int(round(float(row["Npw"])))
                            r_nohm = float(row["R_joint"]) * 1e9  # Ω -> nΩ
                            r_nohm_4sig = f"{r_nohm:.4g}"
                            print(f"          缺失组合: npw={npw_i}, R_joint_nOhm≈{r_nohm_4sig}")
                        print(f"          缺失数据点: {total_points - valid_points}/{total_points} ({missing_ratio*100:.1f}%)")
                        print(f"          可能导致热力图出现大量空白区域")
                    
                    # 不再把绘图数据 clip 到 ±delta_bound(P90≈7): 颜色饱和已由 norm(vmax) 处理;
                    # clip 会把可行区数据砍到 vmax, 导致高于 vmax 的等高线(如 S1/4.2K 的 10/20/50)
                    # 没有数据可画(只剩 <=7 的能画)。用 raw 数据 -> 颜色不变(norm饱和), 高端等高线可画。
                    complete_df_delta['delta_lcoe_min_plot_$/MWh'] = complete_df_delta['delta_LCOE_min_USD_per_MWh']

                    # 统一保守绘图下限：固定rho切片上Npw>=5保证Charging_999<120 h，但不与逐点判据等同。
                    complete_df_delta = complete_df_delta[complete_df_delta['Npw'] >= NPW_MIN_HEATMAP].copy()

                    # 开关开则命中手调字典用指定级别; 关(默认)则一律 None -> 走新版自适应
                    _delta_ov = (DELTA_CONTOUR_OVERRIDES.get((scenario, float(temp), cool))
                                 if USE_DELTA_CONTOUR_OVERRIDES else None)
                    plot_delta_lcoe_heatmap_single(
                        df=complete_df_delta,
                        output_dir=output_dir,
                        cmap=cmap_name,
                        norm=delta_norm,
                        scenario=scenario,
                        temperature_K=temp,
                        baseline_point=None,  # 不显示baseline点
                        value_col='delta_lcoe_min_plot_$/MWh',
                        coolant=cool,
                        contour_levels=_delta_ov,
                        auto_generate_levels=(_delta_ov is None),
                        filename_suffix=cmap_name,
                        use_stroke=USE_STROKE
                    )
                    
                    # 保存colorbar（使用当前配色方案）
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
        
        # 保存共享colorbar（使用当前配色方案）
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
        
        print(f"[OK] 配色方案 {cmap_name} 的所有热力图已成功生成，保存目录: {cmap_scan_usd_dir}")
    
    print(f"\n[OK] 所有配色方案的 USD 热力图已成功生成")
    
    # 10. 如果启用 CNY 功能，生成 CNY 版本的热力图
    if enable_cny:
        print("\n" + "="*60)
        print("--- 开始生成 CNY 版本的热力图 ---")
        print("="*60)
        
        # 计算 CNY 版本的 delta_LCOE_min
        df_cny = df.copy()
        if 'delta_LCOE_min_USD_per_MWh' in df_cny.columns:
            # 将 USD/MWh 转换为 CNY/kWh（单位：CNY/kWh，即人民币/度）
            # 1 MWh = 1000 kWh，所以需要除以1000
            df_cny['delta_LCOE_min_CNY_per_kWh'] = df_cny['delta_LCOE_min_USD_per_MWh'] * USD_TO_CNY_EXCHANGE_RATE / 1000.0
        
        # 计算 CNY 版本的全局 norm
        if 'delta_LCOE_min_CNY_per_kWh' in df_cny.columns:
            # Mask不可行点
            bad = ~df_cny["feasible_joint"].astype(bool)
            df_cny.loc[bad, "delta_LCOE_min_CNY_per_kWh"] = np.nan
            print(f"[Info] CNY delta mask uses the joint feasibility gate ({int(bad.sum())} points).")
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
                    print(f"[Info] ΔLCOE_min (CNY) 色标范围: [{delta_min_cny:.3g}, {delta_max_cny:.3g}] (CNY/kWh)")
                    
                    all_finite_cny = delta_series_cny.replace([np.inf, -np.inf], np.nan).dropna().values
                    delta_bound_cny = float(np.max(np.abs(all_finite_cny)))
        
        # 为每个配色方案生成 CNY 版本的热力图
        for cmap_name in cfg.color_schemes:
            print(f"\n处理 CNY 配色方案: {cmap_name}")
            # 为每个配色方案创建单独的目录：outputs/figures/economic/{cmap_name}/parasitic_ratio/CNY
            cmap_base_dir = FIGURE_OUTPUT_ROOT / cmap_name
            cmap_scan_base_dir = cmap_base_dir / cfg.PARASITIC_RATIO_OUTPUT_DIR / METRIC_TAG
            cmap_scan_cny_dir = cmap_scan_base_dir / cfg.PARASITIC_RATIO_CNY_DIR
            cmap_scan_cny_dir.mkdir(exist_ok=True, parents=True)
            
            # 为每个场景生成热力图
            for scenario in scenarios:
                scenario_df_cny = df_cny[df_cny['scenario'] == scenario].copy()
                
                if scenario_df_cny.empty:
                    continue
                
                output_dir_cny = cmap_scan_cny_dir / scenario
                output_dir_cny.mkdir(exist_ok=True, parents=True)
                
                # 为每个温度-制冷剂组合绘制热力图
                for temp, cool in temp_coolant_pairs:
                    temp_df_cny = scenario_df_cny[
                        (scenario_df_cny['Top_K'] == temp) &
                        (scenario_df_cny['coolant'] == cool)
                    ].copy()
                    
                    if temp_df_cny.empty:
                        continue
                    
                    # 筛选 rho_turn_uOhm_cm2 = 10000 的数据（用于热力图）
                    RHO_TURN_FOR_HEATMAP = float(os.environ.get("RHO_TURN_HEATMAP", "10000"))  # μΩ·cm²
                    if 'rho_turn_uOhm_cm2' in temp_df_cny.columns:
                        available_rhot = temp_df_cny['rho_turn_uOhm_cm2'].unique()
                        if len(available_rhot) > 0:
                            closest_rhot = available_rhot[np.argmin(np.abs(available_rhot - RHO_TURN_FOR_HEATMAP))]
                            temp_df_cny = temp_df_cny[
                                np.isclose(temp_df_cny['rho_turn_uOhm_cm2'], closest_rhot, rtol=0.01)
                            ].copy()
                    
                    if temp_df_cny.empty:
                        continue
                    
                    # 筛选数据
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
                    
                    # 检查是否有重复的 (Npw, R_joint_nOhm) 组合
                    duplicate_mask = temp_df_cny.duplicated(subset=['Npw', 'R_joint_nOhm'], keep='first')
                    if duplicate_mask.any():
                        temp_df_cny = temp_df_cny.drop_duplicates(subset=['Npw', 'R_joint_nOhm'], keep='first')
                    
                    # 准备数据：转换为绘图函数期望的格式
                    temp_df_plot_cny = temp_df_cny.copy()
                    if 'R_joint_nOhm' in temp_df_plot_cny.columns:
                        temp_df_plot_cny['R_joint'] = temp_df_plot_cny['R_joint_nOhm'] * 1e-9  # nΩ -> Ω
                    
                    # 获取配置中的完整值数组
                    npw_array = np.array(NPW_SCAN_VALUES)
                    r_joint_array = np.array(R_JOINT_SCAN_VALUES, dtype=float) * 1e-9  # 转换为 Ω
                    
                    # 绘制 r_cryo_re 热力图（parasitic ratio）- CNY 版本与 USD 版本相同
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
                    
                    # 绘制 delta_LCOE_min 热力图（CNY 版本）
                    if delta_norm_cny is not None and 'delta_LCOE_min_CNY_per_kWh' in temp_df_plot_cny.columns:
                        complete_df_delta_cny = ensure_complete_data_grid(
                            temp_df_plot_cny,
                            npw_array,
                            r_joint_array,
                            'delta_LCOE_min_CNY_per_kWh'
                        )
                        
                        # Clip数据到bound范围
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
                    
            # 保存共享colorbar（使用当前配色方案）
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
            
            print(f"[OK] 配色方案 {cmap_name} 的所有 CNY 热力图已成功生成，保存目录: {cmap_scan_cny_dir}")
        
        print(f"\n[OK] 所有配色方案的 CNY 热力图已成功生成")


# =============================================================================
# 主程序执行入口
# =============================================================================

if __name__ == "__main__":
    print("启动从 scan_full_grid_tidy_plant_opex_2025usd_direct_hts.csv 绘制热力图脚本...")
    
    # 可以指定要绘制的场景和温度-制冷剂组合，如果为None则自动从数据中获取
    SCENARIOS_TO_PLOT = None # 例如: ['S1', 'S2', 'S3']
    TEMP_COOLANT_PAIRS_TO_PLOT = None  # 例如: [(4.2, 'He'), (10.0, 'He'), (20.0, 'He'), (20.0, 'H2')]
    
    
    # 创建输出目录
    SCAN_BASE_DIR.mkdir(exist_ok=True, parents=True)
    SCAN_USD_DIR.mkdir(exist_ok=True, parents=True)
    if ENABLE_CNY:
        SCAN_CNY_DIR.mkdir(exist_ok=True, parents=True)
    
    # 执行绘图
    plot_from_scan_full_grid(
        scenarios=SCENARIOS_TO_PLOT,
        temp_coolant_pairs=TEMP_COOLANT_PAIRS_TO_PLOT,
        enable_cny=ENABLE_CNY
    )
    
    print("\n脚本执行完毕。")
