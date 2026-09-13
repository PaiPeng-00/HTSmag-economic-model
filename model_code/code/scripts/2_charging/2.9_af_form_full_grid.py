from __future__ import annotations

"""
用途：
- 从 scan_full_grid_plant_opex_2025usd_direct_hts.xlsx 读取可用因子 AF 数据（由 scan_full_grid.py 生成），
  按你给定的三种场景（场景1：HTS原型/LTS工程堆；场景2：HTS工程化；场景3：HTS成熟）
  输出 AF 数据表与热力图。
- 与 8.1_plot_from_scan_full_grid.py 使用相同的数据源，确保一致性。
- 本脚本独立运行、读 cfg，结果输出至 cfg.ECONOMIC_FIGURES_DIR/{cmap_name}/AF_heatmaps_scenarios 下。

与现有工程的耦合：
- AF 数据来自 scan_full_grid_plant_opex_2025usd_direct_hts.xlsx（由 scan_full_grid.py 生成）：
  路径 = outputs/tables/scan_full_grid_plant_opex_2025usd_direct_hts.xlsx
- 数据字段：Top_K, scenario, coolant, rho_turn_uOhm_cm2, Npw, AF
- 数据会插值到固定网格 (FIXED_RHOT x FIXED_NPW) 用于绘图
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl  # <-- 增加导入
from matplotlib import patheffects
from scipy.interpolate import RegularGridInterpolator
from tfmag.paths import ensure_base_dirs
from fusion_tem import device as cfg  # 全局参数（含路径、默认小时数等）
# === 可选：seaborn 风格（若系统无 seaborn，可忽略）
try:
    import seaborn as sns
    sns.set_context('notebook')
    sns.set_style('whitegrid')
except Exception:
    pass

# plot参数
plt.rcParams.update({
    'font.family': 'Arial',
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Arial',
    'mathtext.it': 'Arial:italic',
    'mathtext.bf': 'Arial:bold',
    'font.size': 20,                # 控制全局字体大小
    'axes.titlesize': 20,           # 子图标题字体
    'axes.labelsize': 20,           # 坐标轴标签字体
    'xtick.labelsize': 20,          # x轴刻度字体
    'ytick.labelsize': 20,          # y轴刻度字体
    'legend.fontsize': 20,          # 图例字体
    'figure.titlesize': 20,        # 整体标题字体
})
font_size = cfg.HEATMAP_FONT_SIZE_TICK

# -----------------------------
# 0) 固定绘制网格（与 patch 程序一致）
# -----------------------------

# Npw 范围：1-200
FIXED_NPW = np.arange(1,21,1)
DESIRED_NPW_TICKS = [1,  5,10, 20,]
# Npw 范围：1-20
#FIXED_NPW = np.arange(1,21,1)
# 标注出来的刻度
#DESIRED_NPW_TICKS = [1, 2, 10, 20]
# ρ_turn 范围：10-10000
FIXED_RHOT =  np.concatenate((np.arange(10, 100, 10), 
                            np.arange(100, 1000, 100), 
                            np.arange(1000, 11000, 1000))) 
FIXED_RHOT =  np.concatenate((np.arange(10, 100, 10), 
                            np.arange(100, 1100, 100)
                            )) 
# 标注出来的刻度

DESIRED_RHO_TICKS = [10,  100, 1000]

# -----------------------------
# 1) 从 scan_full_grid_plant_opex_2025usd_direct_hts.xlsx 读取 AF 数据
# -----------------------------
def load_af_from_scan_full_grid(Top: float, scenario: str, coolant: str = 'He') -> pd.DataFrame:
    """
    从 scan_full_grid 数据文件读取指定温度和场景的 AF 数据。
    优先使用 CSV 文件（更快），如果不存在则使用 Excel 文件。
    
    Args:
        Top: 运行温度 (K)
        scenario: 场景名称 (如 'S1', 'S2', 'S3')
        coolant: 制冷剂类型 (默认 'He')
    
    Returns:
        DataFrame，包含 rho_turn_uOhm_cm2, Npw, AF 等列
    """
    from tfmag.paths import ensure_base_dirs
    paths = ensure_base_dirs()
    
    # 优先使用 CSV 文件（读取更快）
    csv_path = paths.outputs_tables / "scan_full_grid_tidy_plant_opex_2025usd_direct_hts.csv"
    excel_path = paths.outputs_tables / "scan_full_grid_plant_opex_2025usd_direct_hts.xlsx"
    
    # 检查预处理后的精简文件是否存在
    af_cache_path = paths.outputs_tables / "scan_full_grid_af_cache_plant_opex_2025usd.csv"
    
    # 如果缓存文件存在且较新，直接使用
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
            print(f"    从缓存文件读取: {af_cache_path.name}")
            df = pd.read_csv(af_cache_path)
            print(f"    已读取 {len(df):,} 行数据")
        except Exception as e:
            print(f"  [Warning] 读取缓存文件失败，将重新生成: {e}")
            use_cache = False
    
    if not use_cache:
        # 读取源文件并创建缓存
        if csv_path.exists():
            try:
                print(f"    正在读取 CSV 文件...")
                df_full = pd.read_csv(csv_path)
                print(f"    已读取 {len(df_full):,} 行数据")
            except Exception as e:
                print(f"  [Warning] 读取 CSV 文件失败: {e}")
                df_full = None
        else:
            df_full = None
        
        if df_full is None and excel_path.exists():
            try:
                print(f"    正在读取 Excel 文件...")
                df_full = pd.read_excel(excel_path)
                print(f"    已读取 {len(df_full):,} 行数据")
            except Exception as e:
                print(f"  [Warning] 读取 Excel 文件失败: {e}")
                return pd.DataFrame()
        elif df_full is None:
            print(f"  [Warning] 数据文件不存在: {csv_path} 或 {excel_path}")
            return pd.DataFrame()
        
        # 只保留需要的列
        required_cols = ['Top_K', 'scenario', 'coolant', 'rho_turn_uOhm_cm2', 'Npw', 'AF']
        missing_cols = [col for col in required_cols if col not in df_full.columns]
        if missing_cols:
            print(f"  [Warning] 缺少必需的列: {missing_cols}")
            return pd.DataFrame()
        
        print(f"    正在创建缓存文件（仅保留需要的列）...")
        df = df_full[required_cols].copy()
        
        # 保存缓存文件
        try:
            df.to_csv(af_cache_path, index=False)
            print(f"    缓存文件已保存: {af_cache_path.name}")
        except Exception as e:
            print(f"  [Warning] 保存缓存文件失败: {e}")
    
    # 筛选指定温度、场景和制冷剂的数据
    # 注意：使用 np.isclose 进行浮点数比较，避免精度问题
    print(f"    正在筛选数据 (Top={Top}K, scenario={scenario}, coolant={coolant})...")
    mask = (
        (np.isclose(df['Top_K'], Top, rtol=1e-5, atol=0.1)) &
        (df['scenario'] == scenario) &
        (df['coolant'] == coolant) &
        (df['AF'].notna())
    )
    df_filtered = df[mask].copy()
    
    if df_filtered.empty:
        print(f"  [Warning] 未找到匹配的数据 (Top={Top}K, scenario={scenario}, coolant={coolant})")
        return pd.DataFrame()
    
    print(f"    找到 {len(df_filtered):,} 个匹配的数据点")
    
    # 检查 AF 值的范围，如果小于等于1，则认为是小数形式，需要转换为百分比
    af_max = df_filtered['AF'].max()
    if not df_filtered.empty and af_max <= 1.0:
        print(f"    检测到 AF 值为小数形式（0-1），转换为百分比形式（0-100）")
        df_filtered = df_filtered.copy()
        df_filtered['AF'] = df_filtered['AF'] * 100.0
    
    return df_filtered[['rho_turn_uOhm_cm2', 'Npw', 'AF']].copy()


def interpolate_af_to_grid(df_af: pd.DataFrame, target_rhot: np.ndarray, target_npw: np.ndarray) -> np.ndarray:
    """
    将 scan_full_grid 的 AF 数据匹配到目标网格 (FIXED_RHOT x FIXED_NPW)。
    使用与 2.9_af_time999_3&3.py 相同的方式：直接匹配最近点，不进行插值。
    
    Args:
        df_af: DataFrame，包含 rho_turn_uOhm_cm2, Npw, AF 列
        target_rhot: 目标 rho_turn 数组 (μΩ·cm²)
        target_npw: 目标 Npw 数组
    
    Returns:
        2D 数组，形状为 (len(target_rhot), len(target_npw))
    """
    if df_af.empty:
        return np.full((len(target_rhot), len(target_npw)), np.nan)
    
    # 获取唯一值，用于构建网格（类似 2.9_af_time999_3&3.py 中的 rhot_grid 和 npw_grid）
    available_rhot = df_af['rho_turn_uOhm_cm2'].unique()
    available_npw = df_af['Npw'].unique()
    
    # 创建 AF 值的 2D 网格（类似 time999_2d）
    # 首先构建一个完整的网格，对于每个 (rhot, npw) 组合，取平均值（如果有多个值）
    rhot_grid = np.sort(available_rhot)
    npw_grid = np.sort(available_npw)
    
    # 创建查找字典：对于每个 (rhot, npw) 组合，存储所有 AF 值
    af_grid_dict = {}
    for _, row in df_af.iterrows():
        key = (row['rho_turn_uOhm_cm2'], row['Npw'])
        if key not in af_grid_dict:
            af_grid_dict[key] = []
        af_grid_dict[key].append(row['AF'])
    
    # 创建 2D 网格（类似 time999_2d）
    af_grid_2d = np.full((len(rhot_grid), len(npw_grid)), np.nan)
    for i, rho_u in enumerate(rhot_grid):
        for j, npw in enumerate(npw_grid):
            # 查找精确匹配
            if (rho_u, npw) in af_grid_dict:
                af_grid_2d[i, j] = np.mean(af_grid_dict[(rho_u, npw)])
    
    # 对于每个目标点，找到最接近的网格点（与 2.9_af_time999_3&3.py 一致）
    af_values = np.zeros((len(target_rhot), len(target_npw)))
    for i, rho_u in enumerate(target_rhot):
        rho_idx = int(np.abs(rhot_grid - rho_u).argmin())
        for j, npw in enumerate(target_npw):
            npw_idx = int(np.abs(npw_grid - npw).argmin())
            af_value = af_grid_2d[rho_idx, npw_idx]
            af_values[i, j] = af_value if not np.isnan(af_value) else np.nan
    
    return af_values


# -----------------------------
# 2) 三个场景参数（来自 config）
# -----------------------------
SCENARIOS = cfg.SCENARIO_DEFINITIONS

Top_list = [4.2, 10.0, 20.0]

# -----------------------------
# 3) 场景化 AF 计算（显式包含：维护、冷/升温、励/退磁）
# -----------------------------
def compute_AF_with_explicit_breakdown(time_999_h: float,
                                       tmaint_h: float, nmaint: float,
                                       tcool_h: float, twarm_h: float, kdis: float,
                                       tau_pulse_h: float, tau_dwell_h: float,
                                       hours_per_year: float = cfg.HOURS_PER_YEAR):
    """
    根据年度时间分解，计算 AF 与各阶段小时数：
    - H_maint: 维护总时数
    - H_coolwarm: 冷升温总时数 = nmaint*(tcool + twarm)
    - H_excdec: 励/退磁总时数 = 2*nmaint*tcharge，其中 tcharge = time_999_h
    - H_prod/H_dwell: 周期运行的生产/间歇总时数
    """
    tcharge_h = float(max(0.0, time_999_h))
    H_maint = nmaint * tmaint_h
    H_coolwarm = nmaint * (tcool_h + twarm_h)
    if not np.isclose(float(kdis), 1.0):
        raise ValueError("The equal charge/discharge model requires kdis=1.0")
    H_excdec = 2.0 * nmaint * tcharge_h

    remaining = hours_per_year - H_maint - H_coolwarm - H_excdec
    if remaining <= 0:
        return 0.0, H_maint, H_coolwarm, H_excdec, 0.0, 0.0

    tcycle = tau_pulse_h + tau_dwell_h
    ncycles = remaining / tcycle
    H_prod = ncycles * tau_pulse_h
    H_dwell = ncycles * tau_dwell_h
    AF = H_prod / hours_per_year *100.0
    return AF, H_maint, H_coolwarm, H_excdec, H_prod, H_dwell


# -----------------------------
# 4) 生成 AF 表格与热力图 (*** 修改后的函数 ***)
# -----------------------------
def make_af_heatmaps_for_scenarios(Top=20.0, out_root: Path | None = None, use_stroke: bool = False):
    """
    为所有场景生成并排的热力图，并共享一个颜色条。
    支持所有配色方案，每个配色方案保存到单独的文件夹。
    数据来源：从 scan_full_grid_plant_opex_2025usd_direct_hts.xlsx 读取 AF 数据。
    """
    # 确定制冷剂类型（根据温度，默认使用 He）
    coolant = 'He'  # 可以根据需要修改
    
    # 为每个配色方案生成热力图
    for cmap_name in cfg.color_schemes:
        print(f"\n处理 AF 热力图配色方案: {cmap_name}")
        # 每次循环都重新计算 out_root，避免路径递进
        if out_root is None:
            # 新的路径结构：outputs/figures/economic/{cmap_name}/AF_heatmaps_scenarios
            current_out_root = Path(cfg.ECONOMIC_FIGURES_DIR) / cmap_name / 'AF_heatmaps_scenarios'
        else:
            # 如果指定了 out_root，则在其中创建配色方案子文件夹
            current_out_root = out_root / cmap_name / 'AF_heatmaps_scenarios'
        current_out_root.mkdir(parents=True, exist_ok=True)

        # 优化目录结构：按 data / figures 分开保存
        data_root = current_out_root / "data"
        fig_root = current_out_root / "figures"
        data_root.mkdir(parents=True, exist_ok=True)
        fig_root.mkdir(parents=True, exist_ok=True)

        af_data_all_scenarios = {}
        global_vmin = np.inf
        global_vmax = -np.inf

        # --- 步骤 1: 从 scan_full_grid_plant_opex_2025usd_direct_hts.xlsx 读取所有场景的数据并找到全局 min/max ---
        for key, sc in SCENARIOS.items():
            out_data_dir = data_root / key
            out_data_dir.mkdir(parents=True, exist_ok=True)

            # 从 scan_full_grid_plant_opex_2025usd_direct_hts.xlsx 读取 AF 数据
            print(f"  从 scan_full_grid_plant_opex_2025usd_direct_hts.xlsx 读取 {key} 场景的 AF 数据 (Top={Top}K)...")
            df_af = load_af_from_scan_full_grid(Top=Top, scenario=key, coolant=coolant)
            
            if df_af.empty:
                print(f"  [Warning] 未找到 {key} 场景的数据，跳过")
                AF_values = np.full((len(FIXED_RHOT), len(FIXED_NPW)), np.nan)
            else:
                # 插值到目标网格
                AF_values = interpolate_af_to_grid(df_af, FIXED_RHOT, FIXED_NPW)
                # 限制范围在 0-100
                AF_values = np.clip(AF_values, 0.0, 100.0)
            
            # 存储数据
            af_data_all_scenarios[key] = AF_values
            
            # 更新全局 min/max（忽略 NaN）
            valid_values = AF_values[~np.isnan(AF_values)]
            if len(valid_values) > 0:
                global_vmin = min(global_vmin, float(np.nanmin(valid_values)))
                global_vmax = max(global_vmax, float(np.nanmax(valid_values)))

            # 保存 Excel (此逻辑保留)
            df_AF = pd.DataFrame(AF_values, index=FIXED_RHOT, columns=FIXED_NPW)
            excel_path = out_data_dir / f"AF_data_{key}_full_grid.xlsx"
            df_AF.to_excel(excel_path)

        # --- 步骤 2: 绘图 (所有子图) ---
        num_scenarios = len(SCENARIOS)
        # 调整 figsize 以容纳 3 个子图，每个子图 5x5（与 plot_library.py 中的热力图一致）
        fig, axes = plt.subplots(1, num_scenarios, figsize=cfg.HEATMAP_FIGSIZE_GRID, sharex=True, sharey=True)
        
        # 确保
        if num_scenarios == 1:
            axes = [axes] # 如果只有一个子图，使其可迭代

        # 支持 config 中的自定义/截取版配色方案名（如 YlGnBu_trunc_0p8）
        cmap = cfg.resolve_cmap(cmap_name) if hasattr(cfg, "resolve_cmap") else cmap_name

        norm = mpl.colors.PowerNorm(gamma=1.4, vmin=global_vmin, vmax=global_vmax)

        im = None # 用于 colorbar

        # --- 步骤 3: 绘制每个场景的热力图与等高线 ---
        for ax, (key, sc) in zip(axes, SCENARIOS.items()):
            AF_values = af_data_all_scenarios[key]         # 形状: (len(FIXED_RHOT), len(FIXED_NPW)) 的转置前
            Z = AF_values.T                                # Z 需为形状 (Ny, Nx)，Ny=len(FIXED_NPW), Nx=len(FIXED_RHOT)
        
        # ========= 1) 中心点（用于等高线）=========
        rho_c = FIXED_RHOT.astype(float)               # μΩ·cm²
        # 原始坐标（中心）- 使用对数坐标
        x_c = np.log10(FIXED_RHOT)
        y_c = np.log10(FIXED_NPW)

        # 设置坐标轴为对数刻度
        '''ax.set_xscale('log')
        ax.set_yscale('log')'''

        # 创建插值器（使用对数坐标）
        interp = RegularGridInterpolator((y_c, x_c), Z, bounds_error=False, fill_value=None)

        # 生成高分辨率网格（使用边角点网格，使等高线延伸到边缘）
        # 创建中心点网格用于插值（在对数空间中线性分布）
        x_center = np.linspace(x_c.min(), x_c.max(), 40)
        y_center = np.linspace(y_c.min(), y_c.max(), 40)
        X_center, Y_center = np.meshgrid(x_center, y_center, indexing='xy')
        Zf = interp((Y_center, X_center))
        
        # 创建边角点网格（用于pcolormesh和contour，使等高线延伸到边缘）
        # 对于shading="flat"，X和Y应该是边角点坐标（shape为(n+1, m+1)），Z应该是中心点值（shape为(n, m)）
        dx = (x_center[-1] - x_center[0]) / (len(x_center) - 1) if len(x_center) > 1 else 0
        dy = (y_center[-1] - y_center[0]) / (len(y_center) - 1) if len(y_center) > 1 else 0
        x_edges = np.concatenate([[x_center[0] - dx/2], 
                                  (x_center[:-1] + x_center[1:]) / 2, 
                                  [x_center[-1] + dx/2]])
        y_edges = np.concatenate([[y_center[0] - dy/2], 
                                  (y_center[:-1] + y_center[1:]) / 2, 
                                  [y_center[-1] + dy/2]])
        Xf, Yf = np.meshgrid(x_edges, y_edges, indexing='xy')
        
        # 将Zf（中心点值）插值到边角点网格，用于contour
        # 创建边角点插值器
        interp_edges = RegularGridInterpolator((y_center, x_center), Zf, bounds_error=False, fill_value=None)
        Zf_edges = interp_edges((Yf, Xf))

        # 绘制主图（使用边角点网格坐标Xf, Yf和中心点值Zf，shading="flat"）
        # pcolormesh with shading="flat"期望：X, Y shape为(n+1, m+1)，Z shape为(n, m)
        # 注意：Xf, Yf 是对数坐标，但由于设置了 set_xscale('log') 和 set_yscale('log')，需要转换为物理值
        '''Xf_phys = 10 ** Xf
        Yf_phys = 10 ** Yf'''
        pcm = ax.pcolormesh(Xf, Yf, Zf, cmap=cmap, norm=norm, shading="flat", alpha=cfg.HEATMAP_ALPHA)
        im = pcm
        
        # 设置坐标轴范围，确保等高线可以绘制到边缘
        ax.set_xlim(x_edges[0], x_edges[-1])
        ax.set_ylim(y_edges[0], y_edges[-1])

        # 用平滑网格绘制等高线
        #CS = ax.contour(Xf, Yf, Zf, levels=CONTOUR_LEVELS_SCENARIOS[key],
        #                        colors="white", linewidths=cfg.HEATMAP_CONTOUR_LINEWIDTH_AF, fontsize=font_size)
        # 设置等高线为0.1, 0.2, ..., 直到Vmax。如果数量超过4，则从中间隔地采样
        max_val = float(np.nanmax(Zf))
        contour_start = 10.0
        contour_step = 10.0

        # 自动生成等高线级别
        # 如果 max_val 小于 contour_start，使用更小的步长
        if max_val < contour_start:
            # 使用更小的步长，从 0 开始
            contour_step = max(1.0, max_val / 10.0)  # 至少分成10个级别
            levels_raw = np.arange(0, max_val + contour_step/2, contour_step)
        else:
            # 从 contour_start 开始，使用 contour_step 步长
            levels_raw = np.arange(contour_start, max_val + contour_step/2, contour_step)

        # 确保包含max_val（如果未包含且max_val较大明显不等于最后一个整数标签）
        if len(levels_raw) > 0:
            if abs(levels_raw[-1] - max_val) > 1e-3:
                levels = np.append(levels_raw, max_val)
            else:
                levels = levels_raw
        else:
            # 如果 levels_raw 为空，直接使用 max_val
            levels = np.array([max_val])

        # 增加一个 vmax*0.999
        vmax_point_999 = max_val * 0.99
        # 若 vmax*0.999 不在 levels 内且与现有最大值不同，则加入
        insert_999 = True
        for v in levels:
            if abs(v - vmax_point_999) < 1e-5:
                insert_999 = False
                break
        if len(levels) > 0 and insert_999 and vmax_point_999 > levels[0] and vmax_point_999 < levels[-1]:
            levels = np.append(levels, vmax_point_999)
            levels = np.sort(levels)

        # 如果数量小于等于4，直接用，否则采样（除了最后一个max_val）
        if len(levels) <= 5:
            contour_levels = list(levels)
        else:
            # 取末尾最大值，其余均匀采样3个“主”等高线
            n_sample = 5
            idxs = np.round(np.linspace(0, len(levels)-2, n_sample)).astype(int)
            sampled = [levels[i] for i in idxs]
            contour_levels = sampled + [levels[-1]]
            # 若 vmax_point_999 不在 contour_levels，则添加
            if not any(abs(v - vmax_point_999) < 1e-5 for v in contour_levels):
                contour_levels.append(vmax_point_999)
            # 去重并排序保证顺序
            contour_levels = sorted(set(contour_levels))

        # 最后保证不超过max_val（浮点安全），并排除max_val和vmax_point_999以避免右上角多余等高线
        contour_levels = [v for v in contour_levels if v < max_val - 1e-6 and abs(v - vmax_point_999) > 1e-6]
        
        if use_stroke:
            # 使用描边技术：使用path_effects为黑色等高线添加白色描边
            # 使用边角点网格绘制等高线，使等高线延伸到边缘
            # 注意：Xf, Yf 是对数坐标，需要转换为物理值
            Xf_phys = 10 ** Xf
            Yf_phys = 10 ** Yf
            CS = ax.contour(Xf_phys, Yf_phys, Zf_edges, levels=contour_levels,
                            colors="black", linewidths=cfg.HEATMAP_CONTOUR_LINEWIDTH_AF)
            # 为等高线集合添加白色描边效果
            for collection in CS.collections:
                collection.set_path_effects([
                    patheffects.withStroke(
                        linewidth=cfg.CONTOUR_STROKE_LINEWIDTH_AF,
                        foreground=cfg.CONTOUR_STROKE_FOREGROUND,
                        alpha=cfg.CONTOUR_STROKE_ALPHA
                    ),
                    patheffects.Normal()
                ])
            # 为等高线文字标注添加描边效果
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
            # 使用边角点网格绘制等高线，使等高线延伸到边缘
            # 注意：Xf, Yf 是对数坐标，需要转换为物理值
            Xf_phys = 10 ** Xf
            Yf_phys = 10 ** Yf
            CS = ax.contour(Xf_phys, Yf_phys, Zf_edges, levels=contour_levels,
                            colors="white", linewidths=cfg.HEATMAP_CONTOUR_LINEWIDTH_AF)
            # 取消等高线文字标注
            CS.clabel(inline=True, colors="white", fmt=cfg.HEATMAP_CONTOUR_FMT_AF, fontsize=font_size)

        # ========= 5) 移除刻度与标签 =========
        ax.set_xticks([])
        ax.set_yticks([])
        
        # ========= 6) 边框/标题 =========
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(2)
            # 取消子图标题
            ax.set_title("")
             
        # --- 步骤 4: 取消共享标签与组合图 colorbar ---
        # 不显示任何坐标轴标签
        for ax in axes:
            ax.set_xlabel("")
            ax.set_ylabel("")

        # --- 步骤 5: 保存组合图（已禁用）---
        # 不再生成组合热力图，只保存单个场景的热力图
        # 去除多余白边：收紧子图间距与边距
        # fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        # out_svg = fig_root / f"AF_heatmap_all_scenarios_Top_{Top}K.svg"
        # fig.savefig(
        #     out_svg,
        #     transparent=getattr(cfg, "PLOT_TRANSPARENT", True),
        #     bbox_inches="tight",
        #     pad_inches=0.0,
        # )
        
        # --- 步骤 6: 单独保存一张 colorbar ---
        # 使用与主图一致的 cmap 与 norm
        cbar_fig = plt.figure(figsize=cfg.HEATMAP_FIGSIZE_COLORBAR_LARGE)
        cbar_ax2 = cbar_fig.add_axes([0.3, 0.05, 0.1, 0.9])  # 调整宽度使 colorbar 更窄
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        
        sm.set_array([])
        cbar2 = cbar_fig.colorbar(sm, cax=cbar_ax2)
        # 保留colorbar刻度，colorbar标签标题
        cbar2.set_label('Availability Factor (%)')

        # 保留ticks和ticklabels，不
        cbar2.outline.set_edgecolor('black')
        cbar2.outline.set_linewidth(2)
        # 不用 tight_layout，否则 colorbar 刻度会被剪裁；可以用 cbar_fig.subplots_adjust 加一点边距
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

        # --- 步骤 6: 为每个场景单独保存一张 AF 热力图（代码逻辑与组合图一致） ---
        for key, sc in SCENARIOS.items():
            AF_values = af_data_all_scenarios[key]
            Z = AF_values.T

            rho_c = FIXED_RHOT.astype(float)
            # 原始坐标（中心）- 使用对数坐标
            x_c = np.log10(FIXED_RHOT)
            y_c = np.log10(FIXED_NPW)

            interp = RegularGridInterpolator((y_c, x_c), Z, bounds_error=False, fill_value=None)
            # 生成高分辨率网格（使用边角点网格，使等高线延伸到边缘）
            # 创建中心点网格用于插值（在对数空间中线性分布）
            x_center = np.linspace(x_c.min(), x_c.max(), 40)
            y_center = np.linspace(y_c.min(), y_c.max(), 40)
            X_center, Y_center = np.meshgrid(x_center, y_center, indexing='xy')
            Zf = interp((Y_center, X_center))
            
            # 创建边角点网格（用于pcolormesh和contour，使等高线延伸到边缘）
            # 对于shading="flat"，X和Y应该是边角点坐标（shape为(n+1, m+1)），Z应该是中心点值（shape为(n, m)）
            dx = (x_center[-1] - x_center[0]) / (len(x_center) - 1) if len(x_center) > 1 else 0
            dy = (y_center[-1] - y_center[0]) / (len(y_center) - 1) if len(y_center) > 1 else 0
            x_edges = np.concatenate([[x_center[0] - dx/2], 
                                      (x_center[:-1] + x_center[1:]) / 2, 
                                      [x_center[-1] + dx/2]])
            y_edges = np.concatenate([[y_center[0] - dy/2], 
                                      (y_center[:-1] + y_center[1:]) / 2, 
                                      [y_center[-1] + dy/2]])
            Xf, Yf = np.meshgrid(x_edges, y_edges, indexing='xy')
            
            # 将Zf（中心点值）插值到边角点网格，用于contour
            interp_edges = RegularGridInterpolator((y_center, x_center), Zf, bounds_error=False, fill_value=None)
            Zf_edges = interp_edges((Yf, Xf))

            fig_single, ax_single = plt.subplots(1, 1, figsize=cfg.HEATMAP_FIGSIZE_SINGLE)
            
            # 设置坐标轴为对数刻度
            ax_single.set_xscale('log')
            ax_single.set_yscale('log')
            
            # 使用边角点网格坐标Xf, Yf和中心点值Zf，shading="flat"
            # 注意：Xf, Yf 是对数坐标，但由于设置了 set_xscale('log') 和 set_yscale('log')，需要转换为物理值
            Xf_phys = 10 ** Xf
            Yf_phys = 10 ** Yf
            pcm_single = ax_single.pcolormesh(Xf_phys, Yf_phys, Zf, cmap=cmap, norm=norm, shading="flat", alpha=cfg.HEATMAP_ALPHA)
            
            # 设置坐标轴范围，确保等高线可以绘制到边缘
            ax_single.set_xlim(10 ** x_edges[0], 10 ** x_edges[-1])
            ax_single.set_ylim(10 ** y_edges[0], 10 ** y_edges[-1])

            max_val = float(np.nanmax(Zf))
            contour_start = 10.0
            contour_step = 10.0
            
            # 自动生成等高线级别
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

            # 排除max_val和vmax_point_999以避免右上角多余等高线
            contour_levels = [v for v in contour_levels if v < max_val - 1e-6 and abs(v - vmax_point_999) > 1e-6]
            if use_stroke:
                # 使用描边技术：使用path_effects为黑色等高线添加白色描边
                # 使用边角点网格绘制等高线，使等高线延伸到边缘
                # 注意：Xf, Yf 是对数坐标，需要转换为物理值
                Xf_phys = 10 ** Xf
                Yf_phys = 10 ** Yf
                CS_single = ax_single.contour(Xf_phys, Yf_phys, Zf_edges, levels=contour_levels,
                                              colors="black", linewidths=cfg.HEATMAP_CONTOUR_LINEWIDTH_AF)
            # 为等高线集合添加白色描边效果
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
                # 使用边角点网格绘制等高线，使等高线延伸到边缘
                # 注意：Xf, Yf 是对数坐标，需要转换为物理值
                Xf_phys = 10 ** Xf
                Yf_phys = 10 ** Yf
                CS_single = ax_single.contour(Xf_phys, Yf_phys, Zf_edges, levels=contour_levels,
                                              colors="black", linewidths=cfg.HEATMAP_CONTOUR_LINEWIDTH_AF)
                CS_single.clabel(inline=True, colors="black", fmt=cfg.HEATMAP_CONTOUR_FMT_AF)
            
            # 确保坐标轴范围与边角点网格一致
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
            # 同样去除白边
            fig_single.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
            fig_single.savefig(
                single_svg,
                transparent=getattr(cfg, "PLOT_TRANSPARENT", True),
                bbox_inches="tight",
                pad_inches=0.0,
            )
            plt.close(fig_single)
        
        # --- 步骤 7: 写入 README ---
        info_txt = current_out_root / "README.txt"
        with open(info_txt, "w", encoding="utf-8") as f:
            f.write(
                "本目录由 2.9_af_form_full_grid.py 自动生成；\n"
                "每个子目录对应一个场景，含 AF 数据表 (xlsx)。\n"
                "数据来源：scan_full_grid_plant_opex_2025usd_direct_hts.xlsx 中的 AF 字段。\n"
                "与 8.1_plot_from_scan_full_grid.py 使用相同的数据源。\n"
            )
    
        print(f"单独场景AF热力图图像保存到: {fig_root}")
        print(f"单独场景AF热力图数据表保存到: {current_out_root}")




def load_scenario_af_max_from_scan() -> dict[str, float]:
    default = ensure_base_dirs().outputs_tables / "scan_full_grid_tidy_AFref099.csv"
    scan_path = Path(os.environ.get("SCAN_INPUT_CSV", default))
    frame = pd.read_csv(scan_path, usecols=["scenario", "AF_max_scenario"])
    maxima = frame.groupby("scenario")["AF_max_scenario"].first().to_dict()
    missing = sorted(set(SCENARIOS) - set(maxima))
    if missing:
        raise ValueError(f"Missing AF maxima for scenarios: {missing}")
    return {key: float(maxima[key]) for key in SCENARIOS}


def make_af_ref_heatmaps_for_scenarios(Top=20.0, out_root: Path | None = None, use_stroke: bool = False, af_max_by_scenario: dict[str, float] | None = None):
    """
    生成相对 AF：AF_ref = AF / AFmax(场景)，绘图方式参考 AF 热力图。
    支持所有配色方案，每个配色方案保存到单独的文件夹。
    数据来源：从 scan_full_grid_plant_opex_2025usd_direct_hts.xlsx 读取 AF 数据。
    """
    # 确定制冷剂类型（根据温度，默认使用 He）
    coolant = 'He'  # 可以根据需要修改
    if af_max_by_scenario is None:
        af_max_by_scenario = load_scenario_af_max_from_scan()
    
    # 为每个配色方案生成热力图
    for cmap_name in cfg.color_schemes:
        print(f"\n处理 AF_ref 热力图配色方案: {cmap_name}")
        # 每次循环都重新计算 out_root，避免路径递进
        if out_root is None:
            # 新的路径结构：outputs/figures/economic/{cmap_name}/AF_ref_heatmaps_scenarios
            current_out_root = Path(cfg.ECONOMIC_FIGURES_DIR) / cmap_name / 'AF_ref_heatmaps_scenarios'
        else:
            # 如果指定了 out_root，则在其中创建配色方案子文件夹
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

            # 从 scan_full_grid_plant_opex_2025usd_direct_hts.xlsx 读取 AF 数据
            print(f"  从 scan_full_grid_plant_opex_2025usd_direct_hts.xlsx 读取 {key} 场景的 AF 数据 (Top={Top}K)...")
            df_af = load_af_from_scan_full_grid(Top=Top, scenario=key, coolant=coolant)
            
            if df_af.empty:
                print(f"  [Warning] 未找到 {key} 场景的数据，跳过")
                AF_values = np.full((len(FIXED_RHOT), len(FIXED_NPW)), np.nan)
            else:
                # 插值到目标网格
                AF_values = interpolate_af_to_grid(df_af, FIXED_RHOT, FIXED_NPW)
                # 限制范围在 0-100
                AF_values = np.clip(AF_values, 0.0, 100.0)

            # 计算 AF_ref = AF / AFmax
            valid_values = AF_values[~np.isnan(AF_values)]
            if len(valid_values) > 0:
                af_max = float(af_max_by_scenario[key])
                if af_max > 0:
                    AF_ref_values = AF_values / af_max
                else:
                    AF_ref_values = np.zeros_like(AF_values)
            else:
                AF_ref_values = np.full_like(AF_values, np.nan)

            af_ref_data_all_scenarios[key] = AF_ref_values
            
            # 更新全局 min/max（忽略 NaN）
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

        # 支持 config 中的自定义/截取版配色方案名（如 YlGnBu_trunc_0p8）
        cmap = cfg.resolve_cmap(cmap_name) if hasattr(cfg, "resolve_cmap") else cmap_name
        norm = mpl.colors.PowerNorm(gamma=1.4, vmin=global_vmin, vmax=global_vmax)
        im = None

        for ax, (key, sc) in zip(axes, SCENARIOS.items()):
            AF_ref_values = af_ref_data_all_scenarios[key]
            Z = AF_ref_values.T


            # 原始坐标（中心）- 使用对数坐标
            x_c = np.log10(FIXED_RHOT)
            y_c = np.log10(FIXED_NPW)
            
            # 设置坐标轴为对数刻度
            ax.set_xscale('log')
            ax.set_yscale('log')
            
            interp = RegularGridInterpolator((y_c, x_c), Z, bounds_error=False, fill_value=None)
            # 生成高分辨率网格（使用边角点网格，使等高线延伸到边缘）
            # 创建中心点网格用于插值（在对数空间中线性分布）
            x_center = np.linspace(x_c.min(), x_c.max(), 40)
            y_center = np.linspace(y_c.min(), y_c.max(), 40)
            X_center, Y_center = np.meshgrid(x_center, y_center, indexing='xy')
            Zf = interp((Y_center, X_center))
            
            # 创建边角点网格（用于pcolormesh和contour，使等高线延伸到边缘）
            # 对于shading="flat"，X和Y应该是边角点坐标（shape为(n+1, m+1)），Z应该是中心点值（shape为(n, m)）
            dx = (x_center[-1] - x_center[0]) / (len(x_center) - 1) if len(x_center) > 1 else 0
            dy = (y_center[-1] - y_center[0]) / (len(y_center) - 1) if len(y_center) > 1 else 0
            x_edges = np.concatenate([[x_center[0] - dx/2], 
                                      (x_center[:-1] + x_center[1:]) / 2, 
                                      [x_center[-1] + dx/2]])
            y_edges = np.concatenate([[y_center[0] - dy/2], 
                                      (y_center[:-1] + y_center[1:]) / 2, 
                                      [y_center[-1] + dy/2]])
            Xf, Yf = np.meshgrid(x_edges, y_edges, indexing='xy')
            
            # 将Zf（中心点值）插值到边角点网格，用于contour
            interp_edges = RegularGridInterpolator((y_center, x_center), Zf, bounds_error=False, fill_value=None)
            Zf_edges = interp_edges((Yf, Xf))

            # 使用边角点网格坐标Xf, Yf和中心点值Zf，shading="flat"
            # 注意：Xf, Yf 是对数坐标，但由于设置了 set_xscale('log') 和 set_yscale('log')，需要转换为物理值
            Xf_phys = 10 ** Xf
            Yf_phys = 10 ** Yf
            pcm = ax.pcolormesh(Xf_phys, Yf_phys, Zf, cmap=cmap, norm=norm, shading="flat", alpha=cfg.HEATMAP_ALPHA)
            im = pcm
            
            # 设置坐标轴范围，确保等高线可以绘制到边缘
            ax.set_xlim(10 ** x_edges[0], 10 ** x_edges[-1])
            ax.set_ylim(10 ** y_edges[0], 10 ** y_edges[-1])

            max_val = float(np.nanmax(Zf))
            contour_start = 0.1
            contour_step = 0.1
            
            # 自动生成等高线级别
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
                # 确保 0.9 等高线被包含（如果 max_val >= 0.9）
                if max_val >= 0.9 and not any(abs(v - 0.9) < 1e-6 for v in contour_levels):
                    contour_levels.append(0.9)
                # 确保 0.95 等高线被包含（如果 max_val >= 0.95）
                if max_val >= 0.95 and not any(abs(v - 0.95) < 1e-6 for v in contour_levels):
                    contour_levels.append(0.95)
                contour_levels = sorted(set(contour_levels))

            # 排除max_val和vmax_point_999以避免右上角多余等高线
            contour_levels = [v for v in contour_levels if v < max_val - 1e-6 and abs(v - vmax_point_999) > 1e-6]
            if use_stroke:
                # 使用描边技术：使用path_effects为黑色等高线添加白色描边
                # 使用边角点网格绘制等高线，使等高线延伸到边缘
                # 注意：Xf, Yf 是对数坐标，需要转换为物理值
                Xf_phys = 10 ** Xf
                Yf_phys = 10 ** Yf
                CS = ax.contour(Xf_phys, Yf_phys, Zf_edges, levels=contour_levels,
                                colors="black", linewidths=cfg.HEATMAP_CONTOUR_LINEWIDTH_AF, fontsize=font_size)
                # 为等高线集合添加白色描边效果
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
                # 使用边角点网格绘制等高线，使等高线延伸到边缘
                # 注意：Xf, Yf 是对数坐标，需要转换为物理值
                Xf_phys = 10 ** Xf
                Yf_phys = 10 ** Yf
                CS = ax.contour(Xf_phys, Yf_phys, Zf_edges, levels=contour_levels,
                                colors="black", linewidths=cfg.HEATMAP_CONTOUR_LINEWIDTH_AF, fontsize=font_size)
                CS.clabel(inline=True, colors="black", fmt=cfg.HEATMAP_CONTOUR_FMT_AF)
            
            # 确保坐标轴范围与边角点网格一致
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

        # 不再生成组合热力图，只保存单个场景的热力图
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

            # 原始坐标（中心）- 使用对数坐标
            x_c = np.log10(FIXED_RHOT)
            y_c = np.log10(FIXED_NPW)
            
            interp = RegularGridInterpolator((y_c, x_c), Z, bounds_error=False, fill_value=None)
            # 生成高分辨率网格（使用边角点网格，使等高线延伸到边缘）
            # 创建中心点网格用于插值（在对数空间中线性分布）
            x_center = np.linspace(x_c.min(), x_c.max(), 40)
            y_center = np.linspace(y_c.min(), y_c.max(), 40)
            X_center, Y_center = np.meshgrid(x_center, y_center, indexing='xy')
            Zf = interp((Y_center, X_center))
            
            # 创建边角点网格（用于pcolormesh和contour，使等高线延伸到边缘）
            # 对于shading="flat"，X和Y应该是边角点坐标（shape为(n+1, m+1)），Z应该是中心点值（shape为(n, m)）
            dx = (x_center[-1] - x_center[0]) / (len(x_center) - 1) if len(x_center) > 1 else 0
            dy = (y_center[-1] - y_center[0]) / (len(y_center) - 1) if len(y_center) > 1 else 0
            x_edges = np.concatenate([[x_center[0] - dx/2], 
                                      (x_center[:-1] + x_center[1:]) / 2, 
                                      [x_center[-1] + dx/2]])
            y_edges = np.concatenate([[y_center[0] - dy/2], 
                                      (y_center[:-1] + y_center[1:]) / 2, 
                                      [y_center[-1] + dy/2]])
            Xf, Yf = np.meshgrid(x_edges, y_edges, indexing='xy')
            
            # 将Zf（中心点值）插值到边角点网格，用于contour
            interp_edges = RegularGridInterpolator((y_center, x_center), Zf, bounds_error=False, fill_value=None)
            Zf_edges = interp_edges((Yf, Xf))

            fig_single, ax_single = plt.subplots(1, 1, figsize=cfg.HEATMAP_FIGSIZE_SINGLE)
            
            # 设置坐标轴为对数刻度
            ax_single.set_xscale('log')
            ax_single.set_yscale('log')
            
            # 使用边角点网格坐标Xf, Yf和中心点值Zf，shading="flat"
            # 注意：Xf, Yf 是对数坐标，但由于设置了 set_xscale('log') 和 set_yscale('log')，需要转换为物理值
            Xf_phys = 10 ** Xf
            Yf_phys = 10 ** Yf
            ax_single.pcolormesh(Xf_phys, Yf_phys, Zf, cmap=cmap, norm=norm, shading="flat", alpha=cfg.HEATMAP_ALPHA)
            
            # 设置坐标轴范围，确保等高线可以绘制到边缘
            ax_single.set_xlim(10 ** x_edges[0], 10 ** x_edges[-1])
            ax_single.set_ylim(10 ** y_edges[0], 10 ** y_edges[-1])

            max_val = float(np.nanmax(Zf))
            contour_start = 0.1
            contour_step = 0.1
            
            # 自动生成等高线级别
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
            # 确保 0.99 等高线被包含（如果 max_val >= 0.99）
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
                # 确保 0.9 等高线被包含（如果 max_val >= 0.9）
                if max_val >= 0.9 and not any(abs(v - 0.9) < 1e-6 for v in contour_levels):
                    contour_levels.append(0.9)
                # 确保 0.95 等高线被包含（如果 max_val >= 0.95）
                if max_val >= 0.95 and not any(abs(v - 0.95) < 1e-6 for v in contour_levels):
                    contour_levels.append(0.95)
                # 确保 0.99 等高线被包含（如果 max_val >= 0.99）
                if max_val >= 0.99 and not any(abs(v - 0.99) < 1e-6 for v in contour_levels):
                    contour_levels.append(0.99)
                contour_levels = sorted(set(contour_levels))

            # 排除max_val以避免右上角多余等高线，但保留0.99等高线
            contour_levels = [v for v in contour_levels if v < max_val - 1e-6]
            if use_stroke:
                # 使用描边技术：使用path_effects为黑色等高线添加白色描边
                # 使用边角点网格绘制等高线，使等高线延伸到边缘
                # 注意：Xf, Yf 是对数坐标，需要转换为物理值
                Xf_phys = 10 ** Xf
                Yf_phys = 10 ** Yf
                CS_single = ax_single.contour(Xf_phys, Yf_phys, Zf_edges, levels=contour_levels,
                                              colors="black", linewidths=cfg.HEATMAP_CONTOUR_LINEWIDTH_AF)
                # 为等高线集合添加白色描边效果
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
                # 使用边角点网格绘制等高线，使等高线延伸到边缘
                # 注意：Xf, Yf 是对数坐标，需要转换为物理值
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
                "本目录由 2.9_af_form_full_grid.py 自动生成；\n"
                "每个子目录对应一个场景，并保存 AF_ref 数据表(xlsx)。\n"
                "数据来源：scan_full_grid_plant_opex_2025usd_direct_hts.xlsx 中的 AF 字段。\n"
                "与 8.1_plot_from_scan_full_grid.py 使用相同的数据源。\n"
            )

        print(f"单独场景AF_ref热力图图像保存到: {fig_root}")
        print(f"数据表保存到: {current_out_root}")
if __name__ == "__main__":
    # 描边（Halo/Stroke）功能开关（从config读取）
    USE_STROKE = getattr(cfg, "USE_STROKE", False)  # 从config读取，默认为False
    
    af_max_by_scenario = load_scenario_af_max_from_scan()
    for Top in Top_list:  
        make_af_heatmaps_for_scenarios(Top=Top, use_stroke=USE_STROKE)
        make_af_ref_heatmaps_for_scenarios(Top=Top, use_stroke=USE_STROKE, af_max_by_scenario=af_max_by_scenario)
