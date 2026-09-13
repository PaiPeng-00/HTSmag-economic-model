import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import solve_ivp
from pathlib import Path
from matplotlib.colors import LogNorm
# 直接从 config.py 导入配置
from fusion_tem import device as cfg
# 直接从 utils.py 导入 calculate_radial_resistance 函数
from fusion_tem.utils import calculate_radial_resistance
plt.rcParams['font.family'] = 'Times New Roman'
# 从 2.0_simulate_charging.py 复制的核心函数
def simulate_charging_system(L_matrix: np.ndarray, R_values: list, ntape):
    """
    模拟线圈系统的充电和稳态过程。
    参数:
    - L_matrix (np.ndarray): 电感矩阵。
    - R_values (list): 每个线圈的径向电阻值列表。
    - ntape (int): 并绕根数。
    返回:
    - tuple: (时间数组, 环向电流解, 径向电流解, 径向功率解, 总输入电流)
             如果模拟失败则返回 None。
    """
    n = L_matrix.shape[0]
    I_steady_total = cfg.I_TARGET * ntape
    
    # 调整充电和稳态时间以观察更明显的变化
    charge_duration_s = cfg.CHARGE_HOURS * 3600
    ramp_rate = I_steady_total / charge_duration_s if charge_duration_s > 0 else 0
    total_duration_s = charge_duration_s + cfg.STEADY_HOURS * 3600

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
    t_eval_steady = np.linspace(charge_duration_s, total_duration_s, int(cfg.STEADY_HOURS * 60) + 1)
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

def calculate_time_to_999(t_seconds, I_L_solution, npw):
    """
    计算总环向电流达到总目标电流99.9%所需的时间。
    参数:
    - t_seconds (np.ndarray): 时间数组 (单位: 秒).
    - I_L_solution (np.ndarray): 环向电流随时间变化的解，形状为 (n_times, n_pancakes).
    - npw (int): 并绕根数.
    返回:
    - float: 达到99.9%充电状态的时间 (单位: 小时). 如果未达到，则返回 None.
    """
    if t_seconds is None or I_L_solution is None:
        return None

    n_pancakes = I_L_solution.shape[1]
    total_target_current = cfg.I_TARGET * npw * n_pancakes
    target_999_current = 0.999 * total_target_current

    total_azimuthal_current_vs_time = np.sum(I_L_solution, axis=1)

    # 找到第一次达到或超过目标电流的索引
    indices = np.where(total_azimuthal_current_vs_time >= target_999_current)
    
    if len(indices[0]) > 0:
        first_index = indices[0][0]
        time_to_999_seconds = t_seconds[first_index]
        return time_to_999_seconds / 3600.0
    else:
        return None

# --- 主脚本部分，用于生成热力图数据 ---
# 1. 定义 Npw 和 rho_turn 的取值范围
#Npw_values = np.concatenate((np.arange(1, 10, 1), np.arange(10, 60, 10), np.arange(60, 110, 20)))
Npw_values = np.arange(1, 101, 1)
rho_turn_values_uOhm_cm2 = np.concatenate((np.arange(10, 100, 10), np.arange(100, 1000, 200), np.arange(1000, 10000, 2000))) # 匝间电阻率，从 10 到 10^4 微欧姆·厘米² (取值: 10, ~46, ~215, ~1000, ~4641, 10000)
#rho_turn_values_uOhm_cm2 = np.arange(900, 10100, 100)
rho_turn_values_Ohm_m2 = rho_turn_values_uOhm_cm2 * 1e-10 # 转换为 欧姆·米² (1 uOhm·cm² = 1e-10 Ohm·m²)

# 路径设置
script_dir = Path(__file__).parent.resolve()
output_dir = script_dir / cfg.CHARGING_SIM_OUTPUT_DIR 
output_excel_path = output_dir / cfg.CHARGING_SIM_OUTPUT_FILE
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

# 3. 确定最终用于构建 DataFrame 的所有 Npw 和 rho_turn 的并集
all_npw_values = set(Npw_values)
all_rho_turn_values_uOhm_cm2 = set(rho_turn_values_uOhm_cm2.round(0))

if df_existing is not None:
    all_npw_values.update(df_existing.columns.values)
    all_rho_turn_values_uOhm_cm2.update(df_existing.index.values)

# 将集合转换回排序的列表或数组，以便DataFrame的有序性
all_npw_values_sorted = sorted(list(all_npw_values))
all_rho_turn_values_uOhm_cm2_sorted = sorted(list(all_rho_turn_values_uOhm_cm2))

# 4. 使用这个并集来初始化 charging_time_df
charging_time_df = pd.DataFrame(index=all_rho_turn_values_uOhm_cm2_sorted, 
                                columns=all_npw_values_sorted)
charging_time_df.index.name = '匝间电阻率 (μΩ·cm²)'
charging_time_df.columns.name = '并绕根数 (Npw)'

# 5. 将读取到的现有数据填充到这个更全面的 DataFrame 中
if df_existing is not None:
    charging_time_df.update(df_existing)
    print("✅ 现有数据已合并到新的DataFrame结构中。")
else:
    print("未检测到现有数据文件，将从头开始计算。")

# 3. 读取 L_single_wound_matrix
# 根据用户提供的相对路径和文件名构建完整路径
L_single_wound_file = inductance_dir / f"inductance_matrix_Np{cfg.NP}_Npw1_ng{cfg.get_ng_for_npw(1)}.xlsx"

L_single_wound_matrix = None
if L_single_wound_file.exists():
    L_single_wound_matrix = pd.read_excel(L_single_wound_file, header=None).values
    print(f"✅ 成功读取 L_single_wound_matrix 从: {L_single_wound_file}")
else:
    print(f"⚠️ 警告: 未找到 L_single_wound_matrix 文件: {L_single_wound_file}")
    print("将使用默认的示例矩阵进行演示。请确保文件存在以便进行准确计算。")
    # 如果文件不存在，则使用示例矩阵，以确保代码能够运行
    L_single_wound_matrix = np.diag(np.full(cfg.NP, 5e-3)) + \
                            np.diag(np.full(cfg.NP - 1, 0.5e-3), k=1) + \
                            np.diag(np.full(cfg.NP - 1, 0.5e-3), k=-1)

# 确保 L_single_wound_matrix 的维度与 cfg.NP 匹配
if L_single_wound_matrix.shape[0] != cfg.NP or L_single_wound_matrix.shape[1] != cfg.NP:
    print(f"⚠️ 警告: 读取的电感矩阵维度 {L_single_wound_matrix.shape} 与 cfg.NP={cfg.NP} 不匹配。")
    print("将使用默认的示例矩阵进行演示。请检查电感文件或 cfg.NP 设置。")
    L_single_wound_matrix = np.diag(np.full(cfg.NP, 5e-3)) + \
                            np.diag(np.full(cfg.NP - 1, 0.5e-3), k=1) + \
                            np.diag(np.full(cfg.NP - 1, 0.5e-3), k=-1)


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
        L_matrix_current = L_single_wound_matrix / (npw**2)
        
        # 计算径向电阻
        Rr = calculate_radial_resistance(npw, rho_turn)
        R_values = [Rr] * cfg.NP # 假设所有饼状线圈的径向电阻相同

        # 模拟充电过程
        t, I_L, _, _, _ = simulate_charging_system(L_matrix_current, R_values, ntape=npw)
        
        # 计算 99.9% 充电时间
        time_999_h = calculate_time_to_999(t, I_L, npw=npw)
        
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
print(f"\n正在从 Excel 文件读取数据以绘制热力图: {output_excel_path}")
try:
    # 读取Excel文件，并确保索引和列名正确
    df_from_excel = pd.read_excel(output_excel_path, index_col=0)
    # 将列名转换为整数类型，因为Npw_values是整数
    df_from_excel.columns = df_from_excel.columns.astype(int)
    # 将DataFrame转换为NumPy数组用于绘图
    charging_time_matrix_for_plot = df_from_excel.values
    # 确保Npw_values和rho_turn_values_uOhm_cm2与读取的DataFrame的索引/列匹配
    # 如果DataFrame的索引/列与Npw_values/rho_turn_values_uOhm_cm2不完全一致，这里可能需要调整
    # 但由于我们是先用这些值创建DataFrame再保存，所以通常是匹配的
    plot_Npw_values = df_from_excel.columns.values
    plot_rho_turn_values_uOhm_cm2 = df_from_excel.index.values
    print("✅ 数据成功从 Excel 读取。")
except Exception as e:
    print(f"❌ 从 Excel 读取数据失败: {e}")
    print("将使用内存中的 charging_time_df 进行绘图。")
    charging_time_matrix_for_plot = charging_time_df.values
    plot_Npw_values = Npw_values
    plot_rho_turn_values_uOhm_cm2 = rho_turn_values_uOhm_cm2

# 7. 绘制热力图
plt.figure(figsize=(12, 10)) # 调整图像大小以获得更好的可读性

xticks = [str(int(x)) if x in [1, 10, 100] else "" for i, x in enumerate(Npw_values)]
yticks = [f"{x:.0f}" if x in [10, 100, 1000, 10000] else "" for i, x in enumerate(rho_turn_values_uOhm_cm2)]

# 绘制热力图，移除单元格分隔
ax = sns.heatmap(
    charging_time_matrix_for_plot,
    norm=LogNorm(vmin=np.nanmin(charging_time_matrix_for_plot[charging_time_matrix_for_plot > 0]),
                 vmax=np.nanmax(charging_time_matrix_for_plot)),
    annot=False, # 不在单元格内显示数值，等值线会显示
    fmt=".2f",
    cmap="viridis",
    cbar_kws={'label': '99.9% charging time (hours)'},
    xticklabels=xticks,
    yticklabels=yticks
)

# 添加等值线
X, Y = np.meshgrid(np.arange(len(Npw_values)), np.arange(len(rho_turn_values_Ohm_m2)))
# 定义等值线级别，可以根据实际数据范围调整
#levels = np.logspace(np.nanmin(charging_time_matrix), np.nanmax(charging_time_matrix), 10)
levels = [96.5,97,98,99,100, 110, 120, 130, 140, 150,  200, 300,500,1000,5000,10000]  # 自定义等值线值

# 绘制等值线，并添加标签
contour_lines = ax.contour(X + 0.5, Y + 0.5, charging_time_matrix_for_plot, levels=levels, colors='white', linestyles='dashed', linewidths=1.5)
ax.clabel(contour_lines, inline=True, fontsize=10, fmt='%.2f')


plt.xlabel('Number of pw ($N_{pw}$)', fontsize=14)
plt.ylabel('Turn resistivity ($ρ_{turn}$, μΩ·cm²)', fontsize=14)
plt.title('99.9% charging time vs. Number of pw and turn resistivity (with contour lines)', fontsize=16)

plt.tight_layout() # 自动调整子图参数，使之填充整个图像区域
heatmap_path = Path(cfg.OUTPUTS_FIGURES_DIR) / "charging" / "charging_time_heatmap_contour.svg"
heatmap_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(heatmap_path, dpi=300) # 保存为高分辨率图片
plt.close()

print(f"\n✅ 热力图已保存为 {heatmap_path}")
