# 02_simulate_charging.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pathlib import Path

# 导入全局配置和工具函数
from fusion_tem import device as cfg
from fusion_tem.utils import calculate_radial_resistance
# =============================================================================
# 全局绘图配置
# =============================================================================
# 1. 首先设定一个基础样式
plt.style.use('default')
plt.rcParams['font.family'] = 'Arial' 
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

# [新增] 设置刻度线方向朝内
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
# =============================================================================
# 1. 核心计算与模拟函数
# =============================================================================

def simulate_charging_system(L_matrix: np.ndarray, R_values: list, ntape):
    """
    [MODIFIED] 使用两阶段求解法，模拟线圈系统的充电和稳态过程。
    """
    n = L_matrix.shape[0]
    I_steady_total = cfg.I_TARGET * ntape
    
    charge_duration_s = cfg.CHARGE_HOURS * 3600
    ramp_rate = I_steady_total / charge_duration_s if charge_duration_s > 0 else 0
    total_duration_s = charge_duration_s + cfg.STEADY_HOURS * 3600

    # --- 正则化处理，与之前保持一致 ---
    try:
        L_inv = np.linalg.inv(L_matrix)
    except np.linalg.LinAlgError:
        print("警告: 电感矩阵是奇异的。正在尝试正则化处理...")
        epsilon = 1e-9
        L_regularized = L_matrix + np.eye(n) * epsilon
        try:
            L_inv = np.linalg.inv(L_regularized)
            print("正则化成功，计算继续。")
        except np.linalg.LinAlgError:
            print("错误: 正则化后矩阵仍然奇异，无法求解。请检查模型。")
            return None, None, None, None, None

    R_diag = np.diag(R_values)

    # --- 统一的微分方程模型 ---
    def ode_system(t, I_L):
        # I_total(t) 是一个分段函数：线性上升，然后保持平稳
        I_total_t = min(ramp_rate * t, I_steady_total)
        dI_L_dt = L_inv @ (R_diag @ (np.full(n, I_total_t) - I_L))
        return dI_L_dt

    # --- [MODIFIED] 阶段一：充电过程 (0 到 T_charging) ---
    print("  -- 阶段一：正在计算充电过程...")
    # 定义充电阶段的时间点
    t_eval_charge = np.linspace(0, charge_duration_s, int(cfg.CHARGE_HOURS * 60) + 1)
    initial_I_L = np.zeros(n)
    
    solution_charge = solve_ivp(
        fun=ode_system,
        t_span=[0, charge_duration_s],
        y0=initial_I_L,
        t_eval=t_eval_charge,
        method='Radau',
        rtol=1e-4,
        atol=1e-4
    )
    if not solution_charge.success:
        print(f"错误: 充电阶段求解失败 - {solution_charge.message}")
        return None, None, None, None, None

    # --- [MODIFIED] 阶段二：稳态过程 (T_charging 到 T_total) ---
    print("  -- 阶段二：正在计算稳态过程...")
    # 定义稳态阶段的时间点
    t_eval_steady = np.linspace(charge_duration_s, total_duration_s, int(cfg.STEADY_HOURS * 60) + 1)
    # 稳态阶段的初始条件是充电阶段的最终状态
    initial_I_L_steady = solution_charge.y[:, -1]

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
        
    # --- [MODIFIED] 合并两阶段的结果 ---
    # 合并时间点，并移除重复的 T_charging 时刻
    t_sol = np.concatenate([solution_charge.t, solution_steady.t[1:]])
    # 垂直堆叠两个解矩阵
    I_L_sol = np.hstack([solution_charge.y, solution_steady.y[:, 1:]]).T

    # --- 后处理，与之前保持一致 ---
    I_total_over_time = np.minimum(ramp_rate * t_sol, I_steady_total)
    I_R_sol = I_total_over_time[:, np.newaxis] - I_L_sol
    P_R_sol = I_R_sol**2 * R_values

    return t_sol, I_L_sol, I_R_sol, P_R_sol, I_total_over_time

# =============================================================================
# 2. 新增功能: 计算充电时间
# =============================================================================
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

    indices = np.where(total_azimuthal_current_vs_time >= target_999_current)
    
    if len(indices[0]) > 0:
        first_index = indices[0][0]
        time_to_999_seconds = t_seconds[first_index]
        return time_to_999_seconds / 3600.0
    else:
        return None
# =============================================================================
# 3. 绘图函数
# =============================================================================

def plot_simulation_results(t_h, I_L, I_R, P_R, title_prefix, output_path):
    """绘制电流、功率的图像"""
    n = I_L.shape[1]
    
    # --- 绘制周向电流和径向电流 ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for i in range(n):
        ax1.plot(t_h, I_L[:, i], label=f'Coil {i+1}')
        ax2.plot(t_h, I_R[:, i], '--', label=f'Coil {i+1}')
    ax1.axvline(x=cfg.CHARGE_END_TIME_H, ls='--', c='r', alpha=0.7)
    ax2.axvline(x=cfg.CHARGE_END_TIME_H, ls='--', c='r', alpha=0.7)
    ax1.set_ylabel('Azimuthal Current (A)'), ax1.set_title(f'{title_prefix} - Inductor Currents'), ax1.grid(True), ax1.legend()
    ax2.set_ylabel('Radial Current (A)'), ax2.set_title(f'{title_prefix} - Resistor Currents'), ax2.grid(True), ax2.legend()
    plt.xlabel('Time (h)'), plt.tight_layout()
    plt.savefig(output_path / "currents.svg", dpi=300)
    plt.close()

def plot_radial_current_ratio(results_dict, title, output_path):
    """比较不同配置下的"总径向电流 / 总输入电流"占比。"""
    plt.figure(figsize=(12, 7))
    for label, (t_h, I_L, I_R, I_total) in results_dict.items():
        # [MODIFIED] 忽略 t=0 的数据点，从第二个点开始切片
        t_plot = t_h[1:]
        I_R_plot = I_R[1:]
        I_total_plot = I_total[1:]*cfg.NP
        
        # 如果切片后没有数据，则跳过
        if t_plot.size == 0:
            continue
        total_I_R = np.sum(I_R_plot, axis=1)
        ratio = np.divide(total_I_R, I_total_plot, out=np.zeros_like(I_total_plot, dtype=float), where=I_total_plot!=0)
        
        plt.plot(t_plot, ratio * 100, label=label) # 使用切片后的数据绘图    
    plt.axvline(x=cfg.CHARGE_END_TIME_H, ls='--', c='r', alpha=0.7, label='End of Charge')
    plt.xlabel('Time (h)')
    plt.ylabel('Total Radial Current / Total Input Current (%)')
    #plt.title(title)
    plt.legend(), plt.grid(True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ 径向电流比图已保存至: {output_path.name}")

def plot_azimuthal_current_ratio(results_dict, title, output_path):
    """
    [新增] 比较不同配置下的"总环向电流 / 总输入电流"占比。
    """
    plt.figure(figsize=(12, 7))
    for label, (t_h, I_L, I_R, I_total) in results_dict.items():
        # [MODIFIED] 忽略 t=0 的数据点，从第二个点开始切片
        t_plot = t_h[1:]
        I_L_plot = I_L[1:]
        I_total_plot = I_total[1:]*cfg.NP

        if t_plot.size == 0:
            continue

        total_I_L = np.sum(I_L_plot, axis=1)
        ratio = np.divide(total_I_L, I_total_plot, out=np.zeros_like(I_total_plot, dtype=float), where=I_total_plot!=0)
        
        plt.plot(t_plot, ratio * 100, label=label) # 使用切片后的数据绘图
    plt.axvline(x=cfg.CHARGE_END_TIME_H, ls='--', c='r', alpha=0.7, label='End of Charge')
    plt.xlabel('Time (h)')
    plt.ylabel('Total Azimuthal Current / Total Input Current (%)') # 更新Y轴标签
    #plt.title(title)
    plt.legend(), plt.grid(True), plt.ylim(bottom=50, top=100.5) # Y轴范围设为80-100.5%更利于观察
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ 环向电流比图已保存至: {output_path.name}")
# *** 新增绘图函数 ***
def plot_charging_time(df, x_col, y_col, title, output_path, log_x=False):
    """
    绘制充电时间随某个变量变化的图。
    """
    plt.figure(figsize=(10, 6))
    plt.plot(df[x_col], df[y_col], 'o-', label='Time to 99.9%')
    if log_x:
        plt.xscale('log')
    plt.xlabel(x_col)
    plt.ylabel(y_col.replace('_', ' ') + ' (hours)')
    plt.title(title)
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ {title} 图已保存至: {output_path.name}")

# =============================================================================
# 3. 主执行流程
# =============================================================================

def main():
    """主函数入口，协调执行两个实验"""
    script_dir = Path(__file__).parent.resolve()
    inductance_dir = script_dir / cfg.INDUCTANCE_OUTPUT_DIR
    output_dir_base = script_dir / cfg.CHARGING_SIM_OUTPUT_DIR
    
    # 确保所有必需的目录都存在
    output_dir_base.mkdir(exist_ok=True)
    if not inductance_dir.exists():
        print(f"错误：找不到电感数据目录: {inductance_dir}")
        print("请先运行 '01_calculate_inductance.py'。")
        return

    # --- 实验一：改变 Npw，固定电阻率 ---
    print("\n" + "="*50)
    print("实验一: 改变并绕根数 (Npw)，固定电阻率")
    print("="*50)
    results_exp1 = {}
    exp1_summary_data = [] # <-- 新增
    output_dir_exp1 = output_dir_base / f"Exp1_Vary_Npw_fix_rhot={cfg.RHO_TURN_EXP1_FIXED*1e10}"
    output_dir_exp1.mkdir(parents=True, exist_ok=True)
    for npw in cfg.NPW_LIST_EXP1:
        
        print(f"\n--- 处理 Npw = {npw} ---")        
        output_dir_exp = output_dir_exp1 / f"Npw_{npw}"
        output_dir_exp.mkdir(parents=True, exist_ok=True)
        ng_current = cfg.get_ng_for_npw(npw)
        print(f"\n开始计算 Npw = {npw} 的情况 (动态 ng = {ng_current})...")
 
        matrix_file = inductance_dir / f"inductance_matrix_Np{cfg.NP}_Npw{npw}_ng{ng_current}.xlsx"
        if not matrix_file.exists():
            print(f"警告: 找不到电感文件 {matrix_file}，跳过 Npw={npw}")
            continue
        L_matrix = pd.read_excel(matrix_file, header=None).values
        print(L_matrix)
        
        Rr = calculate_radial_resistance( npw, cfg.RHO_TURN_EXP1_FIXED)
        R_values = [Rr] * cfg.NP
        print(f"计算得到径向电阻: {Rr*1e6:.2f} μΩ")

        t, I_L, I_R, P_R, I_total = simulate_charging_system(L_matrix, R_values, ntape=npw)
        if t is None: continue # 如果模拟失败，跳到下一个循环
        
        # *** 调用新增功能 ***
        time_999_h = calculate_time_to_999(t, I_L, npw)
        if time_999_h is not None:
            print(f"✅ 计算得到充电至99.9%的时间为: {time_999_h:.2f} 小时")
        else:
            print("⚠️ 在模拟时间内未能达到99.9%的充电目标。")
        # <-- 新增: 保存汇总数据 -->
        exp1_summary_data.append({'Npw': npw, 'Radial_Resistance_uOhm': Rr * 1e6, 'Time_to_99.9%_h': time_999_h})
        
        t_h = t / 3600
        results_exp1[f"Npw = {npw}"] = (t_h, I_L, I_R, I_total)
        plot_simulation_results(t_h, I_L, I_R, P_R, f"Npw={npw}", output_dir_exp)

    # [修改] 调用两个绘图函数
    plot_radial_current_ratio(
        results_exp1,
        'Radial Current Ratio vs. Time (Fixed Resistivity, Varying Npw)',
        output_dir_exp1 /  "Exp1_Vary_Npw_Radial_Ratio.svg"
    )
    plot_azimuthal_current_ratio( # [新增]
        results_exp1,
        'Azimuthal Current Ratio vs. Time (Fixed Resistivity, Varying Npw)',
        output_dir_exp1 /  "Exp1_Vary_Npw_Azimuthal_Ratio.svg"
    )

    # <-- 新增: 保存Excel文件和汇总图 -->
    df_exp1 = pd.DataFrame(exp1_summary_data)
    excel_path_exp1 = output_dir_exp1 / "Exp1_charging_time_summary.xlsx"
    df_exp1.to_excel(excel_path_exp1, index=False)
    print(f"\n✅ 实验一汇总结果已保存至: {excel_path_exp1.name}")
    plot_charging_time(df_exp1.dropna(), 'Npw', 'Time_to_99.9%_h', 'Time to 99.9% Charge vs. Npw', output_dir_exp1 / "Exp1_charging_time_vs_Npw.svg")

    # --- 实验二：改变 rho_turn，固定 Npw ---
    print("\n" + "="*50)
    print(f"实验二: 改变匝间电阻率 (rho_turn)，固定 Npw={cfg.NPW_EXP2_FIXED}")
    print("="*50)
    exp2_summary_data = [] # <-- 新增
    npw_fixed = cfg.NPW_EXP2_FIXED
    output_dir_exp2 = output_dir_base / f"Exp2_Vary_Rho_fix_Npw={npw_fixed}"
    output_dir_exp2.mkdir(parents=True, exist_ok=True)
    results_exp2 = {}
    ng_current = cfg.get_ng_for_npw(npw_fixed)
    matrix_file_exp2 = inductance_dir / f"inductance_matrix_Np{cfg.NP}_Npw{npw_fixed}_ng{ng_current}.xlsx"
    if not matrix_file_exp2.exists():
        print(f"错误: 找不到 Npw={npw_fixed} 的电感文件，无法进行实验二。")
        return
    L_matrix_exp2 = pd.read_excel(matrix_file_exp2, header=None).values
    Nt_physical_exp2 = cfg.N_TOTAL_TAPE // npw_fixed
    
    for rho_turn in cfg.RHO_TURN_LIST_EXP2:
        rho_turn_uOhm_cm2 = rho_turn * 1e10
        print(f"\n--- 处理 rho_turn = {rho_turn_uOhm_cm2:.0f} μΩ·cm² ---")

        output_dir_exp = output_dir_exp2 / f"Rho_{rho_turn_uOhm_cm2:.0f}"
        output_dir_exp.mkdir(parents=True, exist_ok=True)
        
        Rr = calculate_radial_resistance(npw_fixed, rho_turn)
        R_values = [Rr] * cfg.NP
        print(f"计算得到径向电阻: {Rr*1e6:.2f} μΩ")

        t, I_L, I_R, P_R, I_total = simulate_charging_system(L_matrix_exp2, R_values, ntape=npw_fixed)
        if t is None: continue

        # *** 调用新增功能 ***
        time_999_h = calculate_time_to_999(t, I_L, npw_fixed)
        if time_999_h is not None:
            print(f"✅ 计算得到充电至99.9%的时间为: {time_999_h:.2f} 小时")
        else:
            print("⚠️ 在模拟时间内未能达到99.9%的充电目标。")
        # <-- 新增: 保存汇总数据 -->
        exp2_summary_data.append({'rho_turn_uOhm_cm2': rho_turn_uOhm_cm2, 'Radial_Resistance_uOhm': Rr * 1e6, 'Time_to_99.9%_h': time_999_h})

        t_h = t / 3600
        label = f"ρ_turn = {rho_turn_uOhm_cm2:.0f} μΩ·cm²"
        results_exp2[label] = (t_h, I_L, I_R, I_total)
        plot_simulation_results(t_h, I_L, I_R, P_R, f"Rho={rho_turn_uOhm_cm2:.0f}", output_dir_exp)
    # <-- 新增: 保存Excel文件和汇总图 -->
    df_exp2 = pd.DataFrame(exp2_summary_data)
    excel_path_exp2 = output_dir_exp2 / "Exp2_charging_time_summary.xlsx"
    df_exp2.to_excel(excel_path_exp2, index=False)
    print(f"\n✅ 实验二汇总结果已保存至: {excel_path_exp2.name}")
    plot_charging_time(df_exp2.dropna(), 'rho_turn_uOhm_cm2', 'Time_to_99.9%_h', 'Time to 99.9% Charge vs. Turn Resistivity', output_dir_exp2 / "Exp2_charging_time_vs_Rho.svg", log_x=True)

        # [修改] 调用两个绘图函数
    plot_radial_current_ratio(
        results_exp2,
        'Radial Current Ratio vs. Time (Fixed Npw, Varying Resistivity)',
        output_dir_exp2 / "Exp2_Vary_Rho_Radial_Ratio.svg"
    )
    plot_azimuthal_current_ratio( # [新增]
        results_exp2,
        'Azimuthal Current Ratio vs. Time (Fixed Npw, Varying Resistivity)',
        output_dir_exp2 / "Exp2_Vary_Rho_Azimuthal_Ratio.svg"
    )

if __name__ == '__main__':
    main()