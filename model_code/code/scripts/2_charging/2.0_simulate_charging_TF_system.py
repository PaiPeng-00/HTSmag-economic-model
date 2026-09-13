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
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 18

# [新增] 设置刻度线方向朝内
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

color_charge_hour = 'black'
# =============================================================================
# 1. 核心计算与模拟函数
# =============================================================================
def simulate_charging_system(L_matrix: np.ndarray, R_values: list, npw,
                             I_target_local=None, Ntape_coil_local=None):
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

# =============================================================================
# 2. 新增功能: 计算充电时间
# =============================================================================
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
        time_to_999_seconds = t_seconds[first_index]
        return time_to_999_seconds / 3600.0
    else:
        return None
# =============================================================================
# 3. 绘图函数
# =============================================================================
def plot_current_ratios(results_dict, title, output_path_prefix, charge_end_time=None):
    """
    同时绘制径向与环向电流占比 (%)。
    环向 + 径向 = 100%，并在图中标出充电结束时间。
    """
    if not results_dict:
        print("⚠️ 无数据可绘制。")
        return

    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(results_dict)))

    for idx, (label, (t_h, I_L, I_R, I_total)) in enumerate(results_dict.items()):
        total_L = np.sum(I_L, axis=1)
        total_R = np.sum(I_R, axis=1)
        total_all = total_L + total_R
        total_all[total_all == 0] = np.nan

        ratio_R = total_R / total_all * 100
        ratio_L = total_L / total_all * 100

        plt.plot(t_h, ratio_R, lw=2, color=colors[idx], label=f"{label} (Radial)")
        plt.plot(t_h, ratio_L, "--", lw=2, color=colors[idx], label=f"{label} (Azimuthal)")

    if charge_end_time is not None:
        plt.axvline(x=charge_end_time, ls="--", c=color_charge_hour, alpha=0.6, label="End of charge")

    plt.xlabel("Time (h)")
    plt.ylabel("Current Ratio (%)")
    #plt.title(title)
    plt.gcf().subplots_adjust(top=0.88)  # 题目和图像之间设置间隔
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 105)
    plt.tight_layout()
    plt.savefig(f"{output_path_prefix}_Current_Ratio.svg", dpi=300)
    plt.close()

def plot_simulation_results(t_h, I_L, I_R, P_R, title_prefix, output_path):
    """绘制电流、功率的图像"""
    n = I_L.shape[1]
    
    # --- 绘制周向电流和径向电流 ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for i in range(n):
        ax1.plot(t_h, I_L[:, i], label=f'Coil {i+1}')
        ax2.plot(t_h, I_R[:, i], '--', label=f'Coil {i+1}')
    ax1.axvline(x=cfg.CHARGE_HOURS, ls='--', c=color_charge_hour, alpha=0.7)
    ax2.axvline(x=cfg.CHARGE_HOURS, ls='--', c=color_charge_hour, alpha=0.7)
    ax1.set_ylabel('Azimuthal Current (A)'), ax1.set_title(f'{title_prefix} - Inductor Currents'), ax1.grid(True), ax1.legend()
    ax2.set_ylabel('Radial Current (A)'), ax2.set_title(f'{title_prefix} - Resistor Currents'), ax2.grid(True), ax2.legend()
    plt.xlabel('Time (h)'), plt.tight_layout()
    plt.savefig(output_path / "currents.svg", dpi=300)
    #print(f"✅ 电流图像已保存至: {output_path / "currents.svg"}")
    plt.close()

def plot_radial_current_ratio_multi(results_dict, title, output_path_prefix, charge_end_time=None):
    """
    绘制径向电流占比 (%) 随时间变化曲线（单独图）。
    每组 (Npw, rho_turn) 一条曲线，并标出充电结束线。
    """
    if not results_dict:
        print("⚠️ 无数据可绘制径向占比。")
        return

    plt.figure(figsize=(8, 6))
    colors = plt.cm.plasma(np.linspace(0, 1, len(results_dict)))

    for idx, (label, (t_h, I_L, I_R, I_total)) in enumerate(results_dict.items()):
        total_L = np.sum(I_L, axis=1)
        total_R = np.sum(I_R, axis=1)
        total_all = total_L + total_R
        total_all[total_all == 0] = np.nan

        ratio_R = total_R / total_all * 100
        plt.plot(t_h, ratio_R, lw=2, color=colors[idx], label=f"{label}")

    if charge_end_time is not None:
        plt.axvline(x=charge_end_time, ls="--", c=color_charge_hour, alpha=0.6, label="End of charge")

    plt.xlabel("Time (h)")
    plt.ylabel("Radial current ratio (%)")
    #plt.title(title + " — Radial Component")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(0, 105)
    plt.tight_layout()
    plt.savefig(f"{output_path_prefix}_Radial_Current_Ratio.svg", dpi=300)
    plt.close()

def plot_azimuthal_current_ratio_multi(results_dict, title, output_path_prefix, charge_end_time=None):
    """
    绘制环向电流占比 (%) 随时间变化曲线（单独图）。
    每组 (Npw, rho_turn) 一条曲线，并标出充电结束线。
    """
    if not results_dict:
        print("⚠️ 无数据可绘制环向占比。")
        return

    plt.figure(figsize=(8, 6))
    colors = plt.cm.cividis(np.linspace(0, 1, len(results_dict)))

    for idx, (label, (t_h, I_L, I_R, I_total)) in enumerate(results_dict.items()):
        total_L = np.sum(I_L, axis=1)
        total_R = np.sum(I_R, axis=1)
        total_all = total_L + total_R
        total_all[total_all == 0] = np.nan

        ratio_L = total_L / total_all * 100
        plt.plot(t_h, ratio_L, lw=2, color=colors[idx], label=f"{label}")

    if charge_end_time is not None:
        plt.axvline(x=charge_end_time, ls="--", c=color_charge_hour, alpha=0.6, label="End of charge")

    plt.xlabel("Time (h)")
    plt.ylabel("Azimuthal current ratio (%)")
    #plt.title(title + " — Azimuthal Component")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(0, 105)
    plt.tight_layout()
    plt.savefig(f"{output_path_prefix}_Azimuthal_Current_Ratio.svg", dpi=300)
    plt.close()

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
    #plt.title(title)
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ {title} 图已保存至: {str(output_path)}")

# =============================================================================
# 3. 主执行流程
# =============================================================================

def main():
    """主函数入口，协调执行两个实验"""
    script_dir = Path(__file__).parent.resolve()
    inductance_dir = script_dir / cfg.INDUCTANCE_OUTPUT_DIR
    base_output = script_dir / cfg.TF_SYSTEM_CHARGING_SIM_OUTPUT_DIR
    base_output.mkdir(exist_ok=True)
    # 确保所有必需的目录都存在

    if not inductance_dir.exists():
        print(f"错误：找不到电感数据目录: {inductance_dir}")
        print("请先运行 '01_calculate_inductance.py'。")
        return
    
    if not base_output.exists():
        print(f"错误：找不到输出目录: {base_output}")
        print("请先运行 '01_calculate_inductance.py'。")
        return

    # 对每个温区创建子目录并运行完整的实验流程
    for T, case in cfg.TEMPERATURE_CASES.items():
        label = case['label']
        Ntape_coil_case = case['Ntape_coil']
        Ip_case = case['Ip']

        print("\n" + "-"*60)
        print(f"Running temperature case: {label} (T={T} K)  Ip={Ip_case}, Ntape_coil={Ntape_coil_case}")
        print("-"*60)

        # 每个温区单独输出目录
        output_for_T = base_output / f"T_{str(T).replace('.','p')}"
        output_for_T.mkdir(parents=True, exist_ok=True)
        # --- 实验一：改变 Npw，固定电阻率 ---
        print("\n" + "="*50)
        print("实验一: 改变并绕根数 (Npw)，固定电阻率")
        print("="*50)
        results_exp1 = {}
        exp1_summary_data = [] # <-- 新增
        output_dir_exp1 = output_for_T / f"Exp1_Vary_Npw_fix_rhot={cfg.RHO_TURN_EXP1_FIXED*1e10}"
        output_dir_exp1.mkdir(parents=True, exist_ok=True)
        for npw in cfg.NPW_LIST_EXP1:
            
            print(f"\n--- 处理 Npw = {npw} ---")        
            output_dir_exp = output_dir_exp1 / f"Npw_{npw}"
            output_dir_exp.mkdir(parents=True, exist_ok=True)
            ng_current = cfg.get_ng_for_npw(npw)
            print(f"\n开始计算 Npw = {npw} 的情况 (动态 ng = {ng_current})...")
    
            matrix_file = inductance_dir / f"TF_system_L_matrix.xlsx"
            if not matrix_file.exists():
                print(f"警告: 找不到电感文件 {matrix_file}，跳过 Npw={npw}")
                continue
            #这个电感矩阵是mfco计算的，只有一匝
            L_matrix = pd.read_excel(matrix_file, header=None).values
            # 根据Npw缩放电感矩阵

            # LaTeX: $L_{并绕} = L_{单匝} * (匝数 / Npw) ^2$
            Nt_local = Ntape_coil_case / npw #每个线圈的匝数=每个线圈的带材根数/并绕根数
            Nturn_magnet = Nt_local * cfg.NP #每个TF磁体的匝数
            L_matrix = L_matrix * Nturn_magnet**2
            print(L_matrix)
            
            # 径向电阻为每个线圈的径向电阻的串联总和
            Rr = calculate_radial_resistance(
                npw, Ntape_coil= Ntape_coil_case, rho_turn= cfg.RHO_TURN_EXP1_FIXED) * cfg.NP # 一个TF的径向电阻是所有线圈的径向电阻之和
            R_values = [Rr] * cfg.Ntf
            print(f"计算得到每个TF磁体的径向电阻: {Rr*1e6:.2f} μΩ")
            print(Ip_case, Ntape_coil_case)
            t, I_L, I_R, P_R, I_total = simulate_charging_system(
                L_matrix, R_values, npw=npw,
                I_target_local=Ip_case, Ntape_coil_local=Ntape_coil_case)
            if t is None: continue # 如果模拟失败，跳到下一个循环
            
            # *** 调用新增功能 ***
            time_999_h = time_999_h = calculate_time_to_999(
                t, I_L, npw, I_target_per_conductor=Ip_case)

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
        plot_current_ratios(results_exp1, "Exp1 - Vary Npw", output_dir_exp1 / "Exp1_Vary_Npw", charge_end_time=cfg.CHARGE_HOURS)
        plot_radial_current_ratio_multi(results_exp1, "Exp1 - Vary Npw", output_dir_exp1 / "Exp1_Vary_Npw", charge_end_time=cfg.CHARGE_HOURS)
        plot_azimuthal_current_ratio_multi(results_exp1, "Exp1 - Vary Npw", output_dir_exp1 / "Exp1_Vary_Npw", charge_end_time=cfg.CHARGE_HOURS)



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
        output_dir_exp2 = output_for_T / f"Exp2_Vary_Rho_fix_Npw={npw_fixed}"
        output_dir_exp2.mkdir(parents=True, exist_ok=True)
        results_exp2 = {}
        matrix_file_exp2 = inductance_dir / f"TF_system_L_matrix.xlsx"
        if not matrix_file_exp2.exists():
            print(f"错误: 找不到 Npw={npw_fixed} 的电感文件，无法进行实验二。")
            return
        L_matrix_exp2 = pd.read_excel(matrix_file_exp2, header=None).values

        Nt_local = Ntape_coil_case / npw_fixed #每个线圈的匝数=每个线圈的带材根数/并绕根数
        Nturn_magnet = Nt_local * cfg.NP #每个TF磁体的匝数
        L_matrix_exp2 = L_matrix * Nturn_magnet**2
        
        
        for rho_turn in cfg.RHO_TURN_LIST_EXP2:
            rho_turn_uOhm_cm2 = rho_turn * 1e10
            print(f"\n--- 处理 rho_turn = {rho_turn_uOhm_cm2:.0f} μΩ·cm² ---")

            output_dir_exp = output_dir_exp2 / f"Rho_{rho_turn_uOhm_cm2:.0f}"
            output_dir_exp.mkdir(parents=True, exist_ok=True)
            # 一个TF磁体的径向电阻为每个线圈的径向电阻的串联总和
            Rr = calculate_radial_resistance(
                npw_fixed, Ntape_coil= Ntape_coil_case, rho_turn=rho_turn) * cfg.NP
            R_values = [Rr] * cfg.Ntf
            print(f"计算得到径向电阻: {Rr*1e6:.2f} μΩ")

            t, I_L, I_R, P_R, I_total = simulate_charging_system(L_matrix_exp2, 
                        R_values, npw=npw_fixed,
                        I_target_local=Ip_case, Ntape_coil_local=Ntape_coil_case)
            if t is None: continue

            # *** 调用新增功能 ***
            time_999_h = time_999_h = calculate_time_to_999(
                t, I_L, npw, I_target_per_conductor=Ip_case)

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

            #绘图函数
        plot_current_ratios(results_exp2, "Exp2 - Vary Rho_turn", output_dir_exp2 / "Exp2_Vary_Rho", charge_end_time=cfg.CHARGE_HOURS)
        plot_radial_current_ratio_multi(results_exp2, "Exp2 - Vary Rho_turn", output_dir_exp2 / "Exp2_Vary_Rho", charge_end_time=cfg.CHARGE_HOURS)
        plot_azimuthal_current_ratio_multi(results_exp2, "Exp2 - Vary Rho_turn", output_dir_exp2 / "Exp2_Vary_Rho", charge_end_time=cfg.CHARGE_HOURS)



if __name__ == '__main__':
    main()