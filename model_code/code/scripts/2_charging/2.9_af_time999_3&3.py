from __future__ import annotations

"""
用途：
- 读取 "TF 系统 charge999 结果" Excel（来自你已有的充放电仿真流程），
  其表格给出在 (rho_turn, Npw) 网格上的 `time_999_h`（达到 99.9% 目标电流的励磁时间，单位小时）。
- 在"年度运行模型"下，按你给定的三种场景（场景1：HTS原型/LTS工程堆；场景2：HTS工程化；场景3：HTS成熟）
  计算可用因子 AF，并输出 AF 数据表与热力图。
- 不改变现有 economic_model.py / economic_model_af_patch.py 的接口；本脚本独立运行、读 cfg，结果输出至 cfg.ECONOMIC_OUTPUT_DIR/AF_heatmaps_scenarios 下。

与现有工程的耦合：
- `time_999_h` 来自你的 `2.5_charge_time999_TF_system_all_Npw=1-200.py` 生成的 Excel：
  路径 = cwd / cfg.TF_SYSTEM_CHARGING_SIM_OUTPUT_DIR / f"Temp_{Top}K_Ip_{Ip}A_Ntape_coil_{Ntape_coil}_charge999" / cfg.TF_SYSTEM_CHARGING_SIM_OUTPUT_FILE_Npw1_200
- 经济模型 (economic_model.py) 仍使用 cfg 中的年度参数；若需将"场景参数"进一步用于年能量/经济性计算，可在调用 economic_model 前短暂覆盖 cfg.pulse_hours / cfg.dwell_hours / cfg.maintenance_hours_per_year，或将这些参数作为入参扩展（保留给后续）。
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl  # <-- 增加导入
from matplotlib import patheffects
from scipy.interpolate import RegularGridInterpolator
from fusion_tem import device as cfg
from tfmag.paths import ensure_base_dirs  # 全局参数（含路径、默认小时数等）
from fusion_tem.plotting.library import plot_availability_heatmap_single  # Fig5/6 同款渲染
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

# ---------------------------------------------------------------------------
# 画幅范围。**默认全幅**；置环境变量 FIG1_FULL_RANGE=0 可退回旧的窄幅版式。
#
# 为什么改用全幅: AF_ref = AF / AF_max_scenario, 而 AF_max_scenario 取自**全网格**
# (Npw 至 200、ρ_turn 至 10 000)。旧窄幅只画到 Npw=20 / ρ=1000, 电感修复后
# (充电最多慢 2.36 倍) 窄幅内 AF_ref 最大值仅 0.979–0.995 ——
# **S1/S2 三个温度全部够不到 0.99**, 0.99 等高线只在 S3 的右上角出现,
# 而正文把 0.99 称作 primary availability-feasibility boundary。见工作文档 §14.48。
#
# 注意: 9.0 的轴刻度是硬编码的, 有同名开关联动, 两边必须一致。
# ---------------------------------------------------------------------------
FIG1_FULL_RANGE = os.environ.get("FIG1_FULL_RANGE", "1") == "1"

if FIG1_FULL_RANGE:
    # 全幅: 与全网格/Fig5/6 同口径。charging_tf 覆盖 ρ 10–10 000(37点) × Npw 1–200(110点)。
    # 上限取 40: 0.99 线最高到 Npw~41(S1 @rho=10)。拉到 200 会让 85% 画面变成无特征平台;
    # 取 50 则 S3 面板上方留白过多(S3 的 0.99 线只到 ~27)。
    _NPW_MAX = int(os.environ.get("FIG1_NPW_MAX", "40"))
    _RHO_MAX = float(os.environ.get("FIG1_RHO_MAX", "10000"))
    # 显示网格必须与数据表原生网格**逐点重合**。下方 AF 循环是最近邻取点(argmin),
    # 显示点若落在两个数据点之间就会与邻点塌缩到同一列 -> 等高线出台阶。
    # 故这里直接复用 2.5 在 CHARGING_GRID=logspace_full 下写出的同一网格:
    #   rho = logspace(1,4,37) 均匀 log; Npw = 1..50 全整数(数据每个整数都有, 不做 log 取样)。
    FIXED_NPW = np.arange(1, _NPW_MAX + 1)
    DESIRED_NPW_TICKS = sorted(set(
        [t for t in (1, 5, 10, 20, 50, 100, 200) if t <= _NPW_MAX] + [_NPW_MAX]))
    FIXED_RHOT = np.round(np.logspace(1, np.log10(_RHO_MAX), 37))
    DESIRED_RHO_TICKS = [t for t in (10, 100, 1000, 10000) if t <= _RHO_MAX]
    FIG1_CHARGING_DIR = ensure_base_dirs().outputs_tables / "charging_tf_logspace_full"
else:
    # 窄幅(默认, 稿件现用)
    FIXED_NPW = np.arange(1,21,1)
    DESIRED_NPW_TICKS = [1,  5,10, 20,]
    # ρ_turn: 10–1000, 均匀 log 间距(logspace) —— "切到 logspace"(见 FIG1_CHARGING_DIR)。
    # 旧(线性/十进制, 在 log 轴上不均匀 -> 低值等高线残余小弯):
    #   np.concatenate((np.arange(10,100,10), np.arange(100,1100,100)))
    FIXED_RHOT = np.round(np.logspace(1, 3, 25))   # [10,12,15,...,825,1000] 共25点(均匀log)
    DESIRED_RHO_TICKS = [10,  100, 1000]
    # Fig1 专用: 读"均匀 log 网格"的充电数据(charging_tf_logspace)。
    # 只在 2.9 本地生效, **不改全局 cfg.TF_SYSTEM_CHARGING_SIM_OUTPUT_DIR** ->
    # 扫描(scan_full_grid)/Fig5/6 仍用原 charging_tf, 完全不受影响。
    FIG1_CHARGING_DIR = ensure_base_dirs().outputs_tables / "charging_tf_logspace"
Top_list = [4.2, 10.0, 20.0]                       # Fig1 三温(列)

# Fig1(AF_ref) 固定语义等高线级别(与稿件一致): 相对可用度 10%/30%/60%/90%/95%/99%
# 2026-07-26: 去掉 0.6 —— 扩轴后低值区被压缩, 0.3/0.6/0.9 三条挤在一起。
# 0.6 无判据含义(判据是 0.99, 0.90/0.95 为 sensitivity guides), 删之最不损失信息。
AF_REF_SEMANTIC_LEVELS = [0.3, 0.9, 0.95, 0.99]
def load_scenario_af_max_from_scan() -> dict[str, float]:
    """Load scenario-only AF maxima from the versioned full-grid Data S1."""
    default = ensure_base_dirs().outputs_tables / "scan_full_grid_tidy_AFref099.csv"
    scan_path = Path(os.environ.get("SCAN_INPUT_CSV", default))
    if not scan_path.exists():
        raise FileNotFoundError(
            f"Scenario-level AF reference data not found: {scan_path}. "
            "Run revise_afref_system_feasibility.py first."
        )
    frame = pd.read_csv(scan_path, usecols=["scenario", "AF_max_scenario"])
    maxima = frame.groupby("scenario")["AF_max_scenario"].first().to_dict()
    missing = sorted(set(SCENARIOS) - set(maxima))
    if missing:
        raise ValueError(f"Missing AF maxima for Fig. 1 scenarios: {missing}")
    return {key: float(maxima[key]) for key in SCENARIOS}
# -----------------------------
# 1) 读取 charge999 Excel
# -----------------------------
def load_time999_from_cfg(Top=20.0):
    """
    返回：rhot_grid (μΩ·cm²), npw_grid, time999_2d (小时)
    """
    case = cfg.TEMPERATURE_CASES[Top]
    Ntape_coil = case['Ntape_coil']
    Ip = case['Ip']
    base_dir = Path.cwd() / FIG1_CHARGING_DIR / f"Temp_{Top}K_Ip_{Ip}A_Ntape_coil_{Ntape_coil}_charge999"
    file_path = base_dir / cfg.TF_SYSTEM_CHARGING_SIM_OUTPUT_FILE_Npw1_200
    df = pd.read_excel(file_path, index_col=0)
    rhot_grid = df.index.to_numpy(dtype=float)
    npw_grid = df.columns.to_numpy(dtype=float)
    time999_2d = df.to_numpy(dtype=float)  # 小时
    return rhot_grid, npw_grid, time999_2d, file_path


# -----------------------------
# 2) 三个场景参数（来自 config）
# -----------------------------
SCENARIOS = {k: cfg.SCENARIO_DEFINITIONS[k] for k in ("S1", "S2", "S3")}  # Fig1 三场景(行)


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
    """
    # 读取 time_999 表
    rhot_grid, npw_grid, time999_2d, source_path = load_time999_from_cfg(Top)
    
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

        # --- 步骤 1: 计算所有场景的数据并找到全局 min/max ---
        for key, sc in SCENARIOS.items():
            out_data_dir = data_root / key
            out_data_dir.mkdir(parents=True, exist_ok=True)

            AF_values = np.zeros((len(FIXED_RHOT), len(FIXED_NPW)))
            for i, rho_u in enumerate(FIXED_RHOT):
                rho_idx = int(np.abs(rhot_grid - rho_u).argmin())
                for j, Npw in enumerate(FIXED_NPW):
                    col_idx = int(np.abs(npw_grid - Npw).argmin())
                    time_999_h = float(time999_2d[rho_idx, col_idx])

                    AF, _, _, _, _, _ = compute_AF_with_explicit_breakdown(
                        time_999_h=time_999_h,
                        tmaint_h=sc["tmaint_h"], nmaint=sc["nmaint"],
                        tcool_h=sc["tcool_h"], twarm_h=sc["twarm_h"], kdis=sc["kdis"],
                        tau_pulse_h=sc["tau_pulse_h"], tau_dwell_h=sc["tau_dwell_h"],
                        hours_per_year=cfg.HOURS_PER_YEAR
                    )
                    AF_values[i, j] = min(max(AF, 0.0), 100.0)
            
            # 存储数据
            af_data_all_scenarios[key] = AF_values
            
            # 更新全局 min/max（忽略 NaN）
            valid_values = AF_values[~np.isnan(AF_values)]
            if len(valid_values) > 0:
                global_vmin = min(global_vmin, float(np.nanmin(valid_values)))
                global_vmax = max(global_vmax, float(np.nanmax(valid_values)))

            # 保存 Excel (此逻辑保留)
            df_AF = pd.DataFrame(AF_values, index=FIXED_RHOT, columns=FIXED_NPW)
            excel_path = out_data_dir / f"AF_data_{key}.xlsx"
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
        # 原始坐标（中心）

        x_c = np.log10(FIXED_RHOT)
        y_c = np.log10(FIXED_NPW)

        # 创建插值器
        interp = RegularGridInterpolator((y_c, x_c), Z, bounds_error=False, fill_value=None)

        # 生成高分辨率网格（使用边角点网格，使等高线延伸到边缘）
        # 创建边角点网格：每个数据点对应网格的角，而不是中心
        x_range = x_c.max() - x_c.min()
        y_range = y_c.max() - y_c.min()
        # 创建中心点网格用于插值
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
            CS = ax.contour(Xf, Yf, Zf_edges, levels=contour_levels,
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
            CS = ax.contour(Xf, Yf, Zf_edges, levels=contour_levels,
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
        x_c = np.log10(FIXED_RHOT)
        y_c = np.log10(FIXED_NPW)

        interp = RegularGridInterpolator((y_c, x_c), Z, bounds_error=False, fill_value=None)
        # 生成高分辨率网格（使用边角点网格，使等高线延伸到边缘）
        # 创建中心点网格用于插值
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
        # 使用边角点网格坐标Xf, Yf和中心点值Zf，shading="flat"
        pcm_single = ax_single.pcolormesh(Xf, Yf, Zf, cmap=cmap, norm=norm, shading="flat", alpha=cfg.HEATMAP_ALPHA)
        
        # 设置坐标轴范围，确保等高线可以绘制到边缘
        ax_single.set_xlim(x_edges[0], x_edges[-1])
        ax_single.set_ylim(y_edges[0], y_edges[-1])

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
            CS_single = ax_single.contour(Xf, Yf, Zf_edges, levels=contour_levels,
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
            CS_single = ax_single.contour(Xf, Yf, Zf_edges, levels=contour_levels,
                                          colors="black", linewidths=cfg.HEATMAP_CONTOUR_LINEWIDTH_AF)
            CS_single.clabel(inline=True, colors="black", fmt=cfg.HEATMAP_CONTOUR_FMT_AF)
        
        # 确保坐标轴范围与边角点网格一致
        ax_single.set_xlim(x_edges[0], x_edges[-1])
        ax_single.set_ylim(y_edges[0], y_edges[-1])

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
                "本目录由 2.9_af_time999_3&3.py 自动生成；\n"
                "每个子目录对应一个场景，含 AF 数据表 (xlsx)。\n"
                "来源的 time_999_h Excel 路径：{}\n".format(source_path)
            )
    
        print(f"单独场景AF热力图图像保存到: {fig_root}")
        print(f"单独场景AF热力图数据表保存到: {current_out_root}")




def make_af_ref_heatmaps_for_scenarios(Top=20.0, out_root: Path | None = None, use_stroke: bool = False, af_max_by_scenario: dict[str, float] | None = None):
    """
    生成相对 AF：AF_ref = AF / AFmax(场景)，绘图方式参考 AF 热力图。
    支持所有配色方案，每个配色方案保存到单独的文件夹。
    """
    rhot_grid, npw_grid, time999_2d, source_path = load_time999_from_cfg(Top)
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

            AF_values = np.zeros((len(FIXED_RHOT), len(FIXED_NPW)))
            for i, rho_u in enumerate(FIXED_RHOT):
                rho_idx = int(np.abs(rhot_grid - rho_u).argmin())
                for j, Npw in enumerate(FIXED_NPW):
                    col_idx = int(np.abs(npw_grid - Npw).argmin())
                    time_999_h = float(time999_2d[rho_idx, col_idx])

                    AF, _, _, _, _, _ = compute_AF_with_explicit_breakdown(
                        time_999_h=time_999_h,
                        tmaint_h=sc["tmaint_h"], nmaint=sc["nmaint"],
                        tcool_h=sc["tcool_h"], twarm_h=sc["twarm_h"], kdis=sc["kdis"],
                        tau_pulse_h=sc["tau_pulse_h"], tau_dwell_h=sc["tau_dwell_h"],
                        hours_per_year=cfg.HOURS_PER_YEAR,
                    )
                    AF_values[i, j] = min(max(AF, 0.0), 100.0)

            af_max = float(af_max_by_scenario[key])
            if af_max > 0:
                # compute_AF_with_explicit_breakdown returns AF in percent,
                # whereas AF_max_scenario is stored as a fraction in Data S1.
                AF_ref_values = (AF_values / 100.0) / af_max
                if np.nanmax(AF_ref_values) > 1.0 + 1e-6:
                    raise ValueError(
                        f"AF_ref exceeds unity for {key} at {Top} K; check AF units and scenario normalization."
                    )
            else:
                AF_ref_values = np.zeros_like(AF_values)

            af_ref_data_all_scenarios[key] = AF_ref_values
            
            # 更新全局 min/max（忽略 NaN）
            valid_ref_values = AF_ref_values[~np.isnan(AF_ref_values)]
            if len(valid_ref_values) > 0:
                global_vmin = min(global_vmin, float(np.nanmin(valid_ref_values)))
                global_vmax = max(global_vmax, float(np.nanmax(valid_ref_values)))

            df_AF_ref = pd.DataFrame(AF_ref_values, index=FIXED_RHOT, columns=FIXED_NPW)
            excel_path = out_data_dir / f"AF_ref_data_{key}.xlsx"
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

            x_c = np.log10(FIXED_RHOT)
            y_c = np.log10(FIXED_NPW)
            interp = RegularGridInterpolator((y_c, x_c), Z, bounds_error=False, fill_value=None)
            # 生成高分辨率网格（使用边角点网格，使等高线延伸到边缘）
            # 创建中心点网格用于插值
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
            pcm = ax.pcolormesh(Xf, Yf, Zf, cmap=cmap, norm=norm, shading="flat", alpha=cfg.HEATMAP_ALPHA)
            im = pcm
            
            # 设置坐标轴范围，确保等高线可以绘制到边缘
            ax.set_xlim(x_edges[0], x_edges[-1])
            ax.set_ylim(y_edges[0], y_edges[-1])

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
                CS = ax.contour(Xf, Yf, Zf_edges, levels=contour_levels,
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
                CS = ax.contour(Xf, Yf, Zf_edges, levels=contour_levels,
                                colors="black", linewidths=cfg.HEATMAP_CONTOUR_LINEWIDTH_AF, fontsize=font_size)
                CS.clabel(inline=True, colors="black", fmt=cfg.HEATMAP_CONTOUR_FMT_AF)
            
            # 确保坐标轴范围与边角点网格一致
            ax.set_xlim(x_edges[0], x_edges[-1])
            ax.set_ylim(y_edges[0], y_edges[-1])

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
            # Fig5/6(library) 同款渲染: 细网格平滑 + 白描边 + 固定语义等高线, 只隐藏贴边标签不删线
            Z_af = af_ref_data_all_scenarios[key].T   # (npw, rho)
            single_fig_dir = fig_root / key
            _levels = [v for v in AF_REF_SEMANTIC_LEVELS if v <= float(np.nanmax(Z_af)) + 1e-9]
            plot_availability_heatmap_single(
                rho_turn_vals=FIXED_RHOT, npw_vals=FIXED_NPW, Z_af_ref=Z_af,
                output_dir=single_fig_dir, cmap=cmap_name, norm=norm,
                scenario=key, temperature_K=Top,
                contour_levels=_levels, use_stroke=use_stroke,
            )

        info_txt = current_out_root / "README.txt"
        with open(info_txt, "w", encoding="utf-8") as f:
            f.write(
                "本目录由 2.9_af_time999_3&3.py 自动生成；\n"
                "每个子目录对应一个场景，并保存 AF_ref 数据表(xlsx)。\n"
                "time_999_h Excel 路径：{}\n".format(source_path)
            )

        print(f"单独场景AF_ref热力图图像保存到: {fig_root}")
        print(f"数据表保存到: {current_out_root}")
if __name__ == "__main__":
    # 描边（Halo/Stroke）功能开关（从config读取）
    USE_STROKE = getattr(cfg, "USE_STROKE", False)  # 从config读取，默认为False
    
    af_max_by_scenario = load_scenario_af_max_from_scan()
    print(f"Scenario-level AF maxima: {af_max_by_scenario}")
    for Top in Top_list:  
        make_af_heatmaps_for_scenarios(Top=Top, use_stroke=USE_STROKE)
        make_af_ref_heatmaps_for_scenarios(Top=Top, use_stroke=USE_STROKE, af_max_by_scenario=af_max_by_scenario)
