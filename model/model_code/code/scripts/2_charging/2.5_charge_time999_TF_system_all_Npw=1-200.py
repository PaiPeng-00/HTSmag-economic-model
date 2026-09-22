import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import solve_ivp
from pathlib import Path
from matplotlib.colors import LogNorm, PowerNorm
# 直接从 config.py 导入配置
from fusion_tem import device as cfg
# 直接从 utils.py 导入 calculate_radial_resistance 函数
from fusion_tem.utils import calculate_radial_resistance

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'  # 普通字体
plt.rcParams['mathtext.it'] = 'Arial:italic'  # 斜体
plt.rcParams['mathtext.bf'] = 'Arial:bold'  # 粗体
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['savefig.transparent'] = True
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'


# 1. 定义 Npw 和 rho_turn 的取值范围
#Npw_values = np.concatenate((np.arange(1, 10, 1), np.arange(10, 60, 10), np.arange(60, 110, 20)))
Npw_values = np.concatenate((np.arange(1, 100, 1),
                            np.arange(100, 201, 10)))

rho_turn_values_uOhm_cm2 = np.concatenate((np.arange(10, 110, 10),
                                            np.arange(100, 2000, 100),
                                            np.arange(2000, 11000, 1000))) # 匝间电阻率，从 10 到 10^4 微欧姆·厘米² (取值: 10, ~46, ~215, ~1000, ~4641, 10000)


def _scan_grid_requirement():
    """读 configs/scan_full_grid.yaml, 取扫描侧要求的 Npw / rho_turn 网格。

    configs/scan_full_grid.yaml 是扫描网格的唯一权威来源。本脚本自己的网格更密
    (fig. S5 的等高线需要), 故取**并集**: 既保证覆盖扫描要求的每一个点(scan 侧
    已改为精确命中, 不覆盖会直接报错), 又不降低 fig. S5 的分辨率。见工作文档 §14.45。
    """
    path = Path(cfg.CONFIGS_DIR) / "scan_full_grid.yaml"
    if not path.is_file():
        return None, None
    try:
        import yaml
    except ImportError:
        return None, None
    c = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    npw = c.get("npw_values")
    rho = c.get("rho_turn_uohm_cm2_values")
    npw = np.asarray(npw, dtype=float) if isinstance(npw, list) else None
    rho = np.asarray(rho, dtype=float) if isinstance(rho, list) else None
    return npw, rho


_req_npw, _req_rho = _scan_grid_requirement()
if _req_npw is not None:
    _before = len(Npw_values)
    Npw_values = np.unique(np.concatenate([np.asarray(Npw_values, dtype=float), _req_npw])).astype(int)
    _added = len(Npw_values) - _before
    if _added:
        print(f"[2.5] 并入 scan_full_grid.yaml 的 Npw 网格: 新增 {_added} 点 -> 共 {len(Npw_values)}")
if _req_rho is not None:
    _before = len(rho_turn_values_uOhm_cm2)
    rho_turn_values_uOhm_cm2 = np.unique(
        np.concatenate([np.asarray(rho_turn_values_uOhm_cm2, dtype=float), _req_rho])
    )
    _added = len(rho_turn_values_uOhm_cm2) - _before
    if _added:
        print(f"[2.5] 并入 scan_full_grid.yaml 的 rho 网格: 新增 {_added} 点 -> 共 {len(rho_turn_values_uOhm_cm2)}")

rho_turn_values_Ohm_m2 = rho_turn_values_uOhm_cm2 * 1e-10 # 转换为 欧姆·米² (1 uOhm·cm² = 1e-10 Ohm·m²)

# 固定绘制的 Npw 与 rho_turn 列表
fixed_Npw_values = np.concatenate((np.arange(1, 10, 1),
                                    np.arange(10, 21, 2),
                                    np.arange(30, 61, 10),
                                    np.arange(60, 210, 20)))
fixed_rho_turn_values_uOhm_cm2 = np.concatenate((np.arange(10, 110, 10),
                                                np.arange(500, 2000, 500),
                                                np.arange(2000, 11000, 1000)))

# --- 充电网格模式 (可复现选项) ----------------------------------------------
# 默认: 原网格(线性/十进制, rho 10-10000, Npw 1-200) -> 输出 charging_tf/ (Fig5/6 的可行性判据用)。
# 环境变量 CHARGING_GRID=logspace: 用于 Fig1 —— 均匀 log 的 rho(10-1000, 25点) + Npw 1-20,
#   消除线性网格在 log 轴上的不均匀(低值等高线更顺); 输出到独立目录 charging_tf_logspace/,
#   **不动原 charging_tf**, 故 Fig5/6/扫描完全不受影响。
import os as _os
_CHARGING_GRID = _os.environ.get("CHARGING_GRID", "default").lower()
_CHARGING_SUBDIR = "charging_tf"
if _CHARGING_GRID == "logspace":
    rho_turn_values_uOhm_cm2 = np.logspace(1, 3, 25)          # 10..1000 均匀log(excel索引按四舍五入存储)
    rho_turn_values_Ohm_m2 = rho_turn_values_uOhm_cm2 * 1e-10
    Npw_values = np.arange(1, 21)                             # Fig1 只需 1-20
    fixed_Npw_values = np.arange(1, 21)
    fixed_rho_turn_values_uOhm_cm2 = np.round(rho_turn_values_uOhm_cm2)
    _CHARGING_SUBDIR = "charging_tf_logspace"
    print(f"[2.5] CHARGING_GRID=logspace: rho={len(rho_turn_values_uOhm_cm2)}点(均匀log,10-1000), "
          f"Npw=1-20 -> {_CHARGING_SUBDIR}")
elif _CHARGING_GRID == "logspace_full":
    # Fig1 扩轴版(见工作文档 §14.48/§14.49)。相对上面的 logspace:
    #   - rho 扩到 10-10000, 仍是均匀 log(37点)。原 charging_tf 是十进制网格
    #     (10,20,..,100,200,..,1000,1100,..,2000,3000,..), log10 间距在 0.0223-0.3010
    #     之间差 13.5 倍, 画在 log 轴上低 rho 端有大空档、高 rho 端过密 -> 等高线折点多。
    #   - Npw 取 1-50 **全部整数**: 数据每个整数都有意义, 用 log 取样会跳过 13/16/19/21...,
    #     白白丢分辨率并在 Npw 方向造成折角。
    rho_turn_values_uOhm_cm2 = np.logspace(1, 4, 37)
    rho_turn_values_Ohm_m2 = rho_turn_values_uOhm_cm2 * 1e-10
    Npw_values = np.arange(1, 51)
    fixed_Npw_values = np.arange(1, 51)
    fixed_rho_turn_values_uOhm_cm2 = np.round(rho_turn_values_uOhm_cm2)
    _CHARGING_SUBDIR = "charging_tf_logspace_full"
    print(f"[2.5] CHARGING_GRID=logspace_full: rho={len(rho_turn_values_uOhm_cm2)}点(均匀log,10-10000), "
          f"Npw=1-50(全整数) -> {_CHARGING_SUBDIR}")

# ✅ 自定义刻度
desired_npw_ticks = [1, 5, 10,  20, 100, 200]
desired_rho_ticks = [20, 50, 100,   1500, 5000, 10000]
# 从 2.0_simulate_charging.py 复制的核心函数
def simulate_charging_system(L_matrix: np.ndarray, R_values: list, npw,
                             I_target_local=None, Ntape_coil_local=None, steady_hours=None):
    """
    新增参数:
      - I_target_local: (float) 当前温度下每根导体目标电流 (A)，若为 None 则使用 cfg.I_TARGET
      - Ntape_coil_local: (float) 每个线圈使用的带材总根数，若为 None 则使用 cfg.N_TOTAL_TAPE
    其他逻辑尽量保持不变。
    """
    n = L_matrix.shape[0]
    # 如果外部提供，则使用；否则回退到全局配置
    I_target_used = I_target_local if I_target_local is not None else cfg.I_TARGET
    total_tape_used = Ntape_coil_local if Ntape_coil_local is not None else cfg.N_TOTAL_TAPE

    # I_steady_total 之前用 cfg.I_TARGET * ntape
    # 现在用 I_target_used * npw （并绕根数 npw）
    I_steady_total = I_target_used * npw

    n = L_matrix.shape[0] 
    # 调整充电和稳态时间以观察更明显的变化
    charge_duration_s = cfg.CHARGE_HOURS * 3600
    ramp_rate = I_steady_total / charge_duration_s if charge_duration_s > 0 else 0
    # A fixed observation window can end before the 99.9% crossing at slow V6.2 points.
    # The cache builder therefore supplies a conservative time-constant-derived horizon.
    steady_hours_used = cfg.STEADY_HOURS if steady_hours is None else float(steady_hours)
    if not np.isfinite(steady_hours_used) or steady_hours_used <= 0.0:
        raise ValueError("steady_hours must be finite and positive")
    total_duration_s = charge_duration_s + steady_hours_used * 3600

    # 处理奇异电感矩阵
    try:
        L_inv = np.linalg.inv(L_matrix)
    except np.linalg.LinAlgError:
        epsilon = 1e-9
        L_regularized = L_matrix + np.eye(n) * epsilon
        try:
            L_inv = np.linalg.inv(L_regularized)
        except np.linalg.LinAlgError:
            print("错误: 正则化后矩阵仍然奇异，无法求解。")
            return None, None, None, None, None

    R_diag = np.diag(R_values)

    # 定义微分方程系统
    def ode_system(t, I_L):
        I_total_t = min(ramp_rate * t, I_steady_total)
        dI_L_dt = L_inv @ (R_diag @ (np.full(n, I_total_t) - I_L))
        return dI_L_dt

    # 阶段一：充电过程
    t_eval_charge = np.linspace(0, charge_duration_s, int(cfg.CHARGE_HOURS * 60) + 1)
    initial_I_L = np.zeros(n)
    
    solution_charge = solve_ivp(
        fun=ode_system,
        t_span=[0, charge_duration_s],
        y0=initial_I_L,
        t_eval=t_eval_charge,
        method='Radau', # 推荐用于刚性系统的求解器
        rtol=1e-4, # 相对容差
        atol=1e-4  # 绝对容差
    )
    if not solution_charge.success:
        print(f"错误: 充电阶段求解失败 - {solution_charge.message}")
        return None, None, None, None, None

    # 阶段二：稳态过程
    # 输出点前密后疏: 前10000h每小时(精确捕捉99.9%穿越), 之后每1000h(确认稳态)
    # 原 STEADY_HOURS*60 (每分钟,192000h) = 1150万点/每解1.6GB, 改后~万点, 提速~1000x, 精度不变
    _t_dense_end = min(charge_duration_s + 10000 * 3600, total_duration_s)
    t_eval_steady = np.unique(np.concatenate([
        np.arange(charge_duration_s, _t_dense_end, 3600.0),
        np.arange(_t_dense_end, total_duration_s, 1000.0 * 3600),
        [total_duration_s],
    ]))
    initial_I_L_steady = solution_charge.y[:, -1] # 稳态阶段的初始条件是充电阶段的最终状态

    solution_steady = solve_ivp(
        fun=ode_system,
        t_span=[charge_duration_s, total_duration_s],
        y0=initial_I_L_steady,
        t_eval=t_eval_steady,
        method='Radau',
        rtol=1e-4,
        atol=1e-4
    )
    if not solution_steady.success:
        print(f"错误: 稳态阶段求解失败 - {solution_steady.message}")
        return None, None, None, None, None
        
    # 合并两阶段结果
    t_sol = np.concatenate([solution_charge.t, solution_steady.t[1:]])
    I_L_sol = np.hstack([solution_charge.y, solution_steady.y[:, 1:]]).T

    # 后处理计算总电流、径向电流和功率
    I_total_over_time = np.minimum(ramp_rate * t_sol, I_steady_total)
    I_R_sol = I_total_over_time[:, np.newaxis] - I_L_sol
    P_R_sol = I_R_sol**2 * R_values

    return t_sol, I_L_sol, I_R_sol, P_R_sol, I_total_over_time

CHARGING_TIME_OUTPUT_DECIMALS = 2


def calculate_time_to_999(t_seconds, I_L_solution, npw, I_target_per_conductor=None):
    """
    计算总环向电流达到总目标电流99.9%所需的时间。

    参数:
    - t_seconds (np.ndarray): 时间数组 (单位: 秒).
    - I_L_solution (np.ndarray): 环向电流随时间变化的解，形状为 (n_times, n_pancakes).
    - npw (int): 并绕根数.
    - I_target_per_conductor (float): 每根导体目标电流 (A). 若为 None 则使用 cfg.I_TARGET.

    返回:
    - float: 达到99.9%充电状态的时间 (单位: 小时). 如果未达到，则返回 None.
    """
    if t_seconds is None or I_L_solution is None:
        return None

    if I_target_per_conductor is None:
        I_target_per_conductor = cfg.I_TARGET

    n_pancakes = I_L_solution.shape[1]
    total_target_current = I_target_per_conductor * npw * n_pancakes
    target_999_current = 0.999 * total_target_current

    total_azimuthal_current_vs_time = np.sum(I_L_solution, axis=1)

    indices = np.where(total_azimuthal_current_vs_time >= target_999_current)

    if len(indices[0]) > 0:
        first_index = indices[0][0]
        if first_index == 0:
            time_to_999_seconds = float(t_seconds[first_index])
        else:
            previous_index = first_index - 1
            t0 = float(t_seconds[previous_index])
            t1 = float(t_seconds[first_index])
            i0 = float(total_azimuthal_current_vs_time[previous_index])
            i1 = float(total_azimuthal_current_vs_time[first_index])
            if not (np.isfinite(t0) and np.isfinite(t1) and np.isfinite(i0) and np.isfinite(i1) and t1 > t0 and i1 > i0):
                raise RuntimeError("invalid 99.9% current-crossing bracket")
            fraction = (target_999_current - i0) / (i1 - i0)
            if not 0.0 <= fraction <= 1.0:
                raise RuntimeError("99.9% current-crossing interpolation fraction outside [0, 1]")
            time_to_999_seconds = t0 + fraction * (t1 - t0)
        return round(time_to_999_seconds / 3600.0, CHARGING_TIME_OUTPUT_DECIMALS)
    else:
        return None
        
def plot_charging_time_heatmap_fixed(df_full, fixed_Npw_values, fixed_rho_values, output_path):
    """绘制只包含固定 Npw 和 rho_turn 值的热力图"""
    # 读取 Excel 后处理
    df_from_excel = df_full
    # 保证 index 是整数
    df_from_excel.index = np.round(df_from_excel.index.values, 0).astype(int)
    df_from_excel.columns = df_from_excel.columns.astype(int)

    # 固定的 rho 也转成 int
    fixed_rho_values = np.round(fixed_rho_values, 0).astype(int)

    # 筛选
    df_filtered = df_from_excel.loc[
        df_from_excel.index.isin(fixed_rho_values),
        fixed_Npw_values]

    matrix = df_filtered.values
    npw_values = df_filtered.columns.values
    rho_values = df_filtered.index.values

    # Keep the publication panel at the historical 8 x 6 in canvas, but use
    # explicit axes positions so long logarithmic tick labels and the colorbar
    # remain separated after the three panels are stacked at 160 mm width.
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_axes([0.18, 0.16, 0.57, 0.80])
    cbar_ax = fig.add_axes([0.80, 0.16, 0.025, 0.80])
    norm = PowerNorm(
        gamma=0.2,
        vmin=np.nanmin(matrix[matrix > 0]),
        vmax=np.nanmax(matrix)
    )

    ax = sns.heatmap(
        matrix,
        norm=norm,
        annot=False,
        fmt=".2f",
        cmap="cividis",
        ax=ax,
        cbar_ax=cbar_ax,
        cbar_kws={'label': '99.9% charging time (h)'},
        xticklabels=[str(int(x)) for x in npw_values],
        yticklabels=[f"{int(y)}" for y in rho_values]
    )

    ax.invert_yaxis()

    # 添加等值线
    X_plot, Y_plot = np.meshgrid(np.arange(len(npw_values)),
                                 np.arange(len(rho_values)))
    # 删除 10000(与 5000 在陡峭角落重叠), 改用 50000 代表极慢充电角(最大约 97000 h)
    levels = [96,  97,100,  120,
               200, 500, 1000, 5000, 50000]
    if cfg.CHARGE_HOURS == 24:
        levels = [24, 24.5, 25,  30, 50, 100, 
                150, 200, 300, 500, 1000, 5000, 10000]
    # Colorbar 设置
    colorbar = ax.collections[0].colorbar
    # The 96, 97 and 100 h levels are intentionally retained as contours, but
    # labelling all of them on the narrow colorbar creates an unreadable stack.
    # A sparse set spans the full range without changing the data mapping.
    tick_values = [96, 120, 500, 5000, 50000]
    if cfg.CHARGE_HOURS == 24:
        tick_values = [24, 30, 100, 500, 5000, 10000]
    colorbar.set_ticks(tick_values)
    colorbar.set_ticklabels([f"{v:.0f}" for v in tick_values])
    colorbar.ax.tick_params(axis="y", pad=5)
    colorbar.set_label("99.9% charging time (h)", labelpad=18)

    contour_lines = ax.contour(
        X_plot + 0.5, Y_plot + 0.5, matrix,
        levels=levels, colors='white', linestyles='dashed', linewidths=1.5
    )
    ax.clabel(contour_lines, inline=True, fontsize=14, fmt='%.2f')



    # 找出这些刻度在当前矩阵索引中的位置
    xtick_positions = [i + 0.5 for i, val in enumerate(npw_values) if val in desired_npw_ticks]
    ytick_positions = [i + 0.5 for i, val in enumerate(rho_values) if val in desired_rho_ticks]

    # 设置自定义刻度与标签
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels([str(v) for v in desired_npw_ticks if v in npw_values], rotation=0)
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels([str(v) for v in desired_rho_ticks if v in rho_values], rotation=0)

    ax.tick_params(axis="x", pad=5)
    ax.tick_params(axis="y", pad=7)
    ax.set_xlabel(r'Number of parallel tapes per turn, $N_{\mathrm{pw}}$', labelpad=8)
    ax.set_ylabel(
        r'Turn-to-turn contact resistivity, $\rho_{\mathrm{turn}}$ (μΩ·cm²)',
        labelpad=12,
    )
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def charge999(T, Ip_case, Ntape_coil_case):
    # --- 主脚本部分，用于生成热力图数据 ---

    # 路径设置
    script_dir = Path(__file__).parent.resolve()
    output_dir = script_dir / cfg.OUTPUTS_TABLES_DIR / _CHARGING_SUBDIR / f"Temp_{T}K_Ip_{Ip_case}A_Ntape_coil_{Ntape_coil_case}_charge999"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_excel_path = output_dir / cfg.TF_SYSTEM_CHARGING_SIM_OUTPUT_FILE_Npw1_200
    figure_dir = Path(cfg.OUTPUTS_FIGURES_DIR) / _CHARGING_SUBDIR / f"Temp_{T}K_Ip_{Ip_case}A_Ntape_coil_{Ntape_coil_case}_charge999"
    figure_dir.mkdir(parents=True, exist_ok=True)
    output_plot_path = figure_dir / cfg.TF_SYSTEM_CHARGING_SIM_OUTPUT_PLOT_FILE_Npw1_200
    inductance_dir = script_dir / cfg.INDUCTANCE_OUTPUT_DIR

    # 2. 尝试从 Excel 文件读取现有数据
    df_existing = None
    if output_excel_path.exists():
        print(f"检测到现有数据文件: {output_excel_path}，尝试读取...")
        try:
            df_existing = pd.read_excel(output_excel_path, index_col=0)
            # 确保列名和索引名与目标DataFrame的类型一致
            df_existing.columns = df_existing.columns.astype(int)
            df_existing.index = df_existing.index.astype(float) 
            print("✅ 成功读取现有数据。")
        except Exception as e:
            print(f"❌ 读取现有数据失败: {e}。将从头开始计算。")
            df_existing = None # 读取失败则重置为None

    # Load the device-selected physical 18-by-18 single-turn matrix.
    # V6.2 uses the independently calculated ARC geometry artifact.
    L_single_wound_file = inductance_dir / cfg.TF_SYSTEM_MATRIX

    if not L_single_wound_file.exists():
        raise FileNotFoundError(
            f"TF-system inductance matrix is required; no synthetic fallback is permitted: {L_single_wound_file}"
        )
    L_single_wound_matrix = pd.read_excel(L_single_wound_file, header=None).values
    if L_single_wound_matrix.shape != (cfg.Ntf, cfg.Ntf):
        raise ValueError(
            f"invalid TF-system inductance matrix shape {L_single_wound_matrix.shape}; expected {(cfg.Ntf, cfg.Ntf)}"
        )

    # 7. 循环计算每种组合下的充电时间
    print("开始计算充电时间矩阵...")
    for i, rho_turn in enumerate(rho_turn_values_Ohm_m2):
        for j, npw in enumerate(Npw_values):
            # 获取当前单元格的值
            # 获取当前单元格的值 (使用 .loc 确保基于标签的访问)
            # 使用 .at 以获得更快的单个元素访问
            current_rho_label = (rho_turn*1e10).round(0)
            current_npw_label = npw
            current_cell_value = charging_time_df.at[current_rho_label, current_npw_label]
        
            # 如果当前单元格已经有数据（不是NaN），则跳过
            if pd.notna(current_cell_value):
                print(f"  跳过 Npw={npw}, rho_turn={rho_turn/1e-10:.0f} uOhm·cm² (已存在数据: {current_cell_value:.2f} 小时)")
                continue
            print(f"  计算 Npw={npw}, rho_turn={rho_turn/1e-10:.0f} uOhm·cm²...")
            
            # 根据 Npw 缩放电感矩阵
            # LaTeX: $L_{并绕} = L_{单匝} / N_{pw}^2$
            L_matrix_current = L_single_wound_matrix *(Ntape_coil_case*cfg.NP/npw)**2
            #print(f"电感矩阵 ：{L_matrix_current}")
            # 计算径向电阻, 一个TF的径向电阻是所有线圈的径向电阻之和
            Rr = calculate_radial_resistance(npw, Ntape_coil_case, rho_turn) *cfg.NP
            R_values = [Rr] * cfg.Ntf # 假设所有饼状线圈的径向电阻相同
            
            #print(f"  计算得到每个TF磁体的径向电阻: {Rr*1e6:.2f} μΩ")

            # 模拟充电过程
            t, I_L, _, _, _ = simulate_charging_system(
                L_matrix_current, R_values, npw=npw,
                I_target_local=Ip_case, Ntape_coil_local=Ntape_coil_case)
            
            # 计算 99.9% 充电时间
            time_999_h = calculate_time_to_999(t, I_L, npw=npw, I_target_per_conductor=Ip_case)
            
            # 将结果存储到DataFrame
            charging_time_df.loc[rho_turn_values_uOhm_cm2[i].round(0), npw] = time_999_h
            
            if time_999_h is None:
                print("    未能达到 99.9% 充电目标。")
            else:
                print(f"    充电时间: {time_999_h:.2f} 小时")

    print("\n充电时间矩阵计算完成。")

    # 5. 保存数据到 Excel 文件
    charging_time_df.to_excel(output_excel_path)
    print(f"✅ 充电时间数据已保存至: {output_excel_path}")

    # 6. 从 Excel 文件读取数据（用于演示“后续读取”功能）
    # 从 Excel 读取完整数据
    df_from_excel = pd.read_excel(output_excel_path, index_col=0)
    df_from_excel.columns = df_from_excel.columns.astype(int)
    df_from_excel.index = df_from_excel.index.astype(float)

    print(fixed_rho_turn_values_uOhm_cm2)
    # 绘制热力图
    plot_charging_time_heatmap_fixed(
        df_from_excel,
        fixed_Npw_values,
        fixed_rho_turn_values_uOhm_cm2,
        output_plot_path
    )

    print(f"\n✅ 热力图已保存为 {output_plot_path}")

if __name__ == "__main__":
    for temp in cfg.TEMPERATURE_CASES:
        Ip_case = cfg.TEMPERATURE_CASES[temp]['Ip']
        Ntape_coil_case = cfg.TEMPERATURE_CASES[temp]['Ntape_coil']
        charge999(temp, Ip_case, Ntape_coil_case)
