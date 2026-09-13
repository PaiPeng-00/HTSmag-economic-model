# utility_plot_resistance_vs_rhot.py
import numpy as np
import matplotlib.pyplot as plt
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
plt.rcParams['legend.fontsize'] = 20

# [新增] 设置刻度线方向朝内
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['savefig.transparent'] = True
# =============================================================================
def Rradial_vs_rhot(T):
    """
    主执行函数：
    为实验二的配置（固定Npw，改变rho_turn），计算并绘制径向电阻的变化曲线。
    """
    Ip_case = cfg.TEMPERATURE_CASES[T]['Ip']
    Nt_coil_case = cfg.TEMPERATURE_CASES[T]['Ntape_coil']
    print(f"--- 开始生成径向电阻 vs. 匝间电阻率曲线图, Temp={T}K, Ip={Ip_case}A, Nt_coil={Nt_coil_case} ---")

    # --- 1. 从config文件获取实验二的参数 ---
    npw_fixed = cfg.NPW_EXP2_FIXED
    rho_turn_list_ohm_m2 = cfg.RHO_TURN_LIST_EXP2
    print(f"固定并绕根数 (Npw) = {npw_fixed}")

    # --- 2. 循环计算每个rho_turn对应的径向电阻 ---
    resistance_values_uOhm = []
    for rho_turn in rho_turn_list_ohm_m2:
        Rr_ohm = calculate_radial_resistance(npw_fixed, Nt_coil_case, rho_turn)
        Rr_uOhm = Rr_ohm * 1e6
        resistance_values_uOhm.append(Rr_uOhm)
        rho_turn_uOhm_cm2 = rho_turn * 1e10
        print(f"  -> 当 rho_turn = {rho_turn_uOhm_cm2:.0f} [μΩ·cm²] 时, Rr = {Rr_uOhm:.2f} [μΩ]")

    # --- 3. 准备绘图数据 ---
    rho_turn_plot_units = [r * 1e10 for r in rho_turn_list_ohm_m2]

    # --- 4. 开始绘图 ---
    #plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(rho_turn_plot_units, resistance_values_uOhm, color='black', zorder=1)
    ax.scatter(rho_turn_plot_units, resistance_values_uOhm, color='#87CEEB', s=60, zorder=2, edgecolors='white')

    ax.set_xlabel('Turn-to-turn Resistivity (μΩ·cm²)')
    ax.set_ylabel('Coil Equivalent Resistance (μΩ)')
    ax.set_xscale('log')
    

    ax.tick_params(axis='both', which='major')
    
    from matplotlib.ticker import ScalarFormatter
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xticks(rho_turn_plot_units) # 确保主要刻度点在数据点上


    # 2. 循环遍历每个数据点，手动绘制垂直线和底部文字标签
    for x_val, y_val in zip(rho_turn_plot_units, resistance_values_uOhm):
        # 绘制从x轴到数据点的灰色虚线
        ax.plot([x_val, x_val], [0, y_val], color='lightgray', linestyle='--', linewidth=1.5, zorder=0)
        # 在x轴下方添加数值标签
        ax.text(x_val, -0.04, f'{x_val:.0f}', ha='center', va='top', fontsize=20,
                transform=ax.get_xaxis_transform())

    # 隐藏原始的x轴刻度标签，避免与我们手动添加的文字重叠
    ax.tick_params(axis='x', labelbottom=False)
    # 设置x轴label离x轴更远一些
    ax.xaxis.set_label_coords(0.5, -0.095)
    # -------------------
     # 手动设置X轴和Y轴的显示范围
    # 1. 将X轴的左边界设置为一个比最小数据点50更小的值，例如30，以在左侧留出空间
    # 2. 将Y轴的下边界设置为一个比0稍小的值，以给底部的文字标签留出空间
    ax.set_xlim(left=30, right=max(rho_turn_plot_units) * 1.2)
    ax.set_ylim(bottom=0, top=max(resistance_values_uOhm) * 1.1)

    # --- 5. 保存图像 ---
    output_dir = Path(__file__).parent.resolve() / cfg.CHARGING_SIM_OUTPUT_DIR / "Exp2_Vary_Rho" / f"Temp_{T}K_Ip_{Ip_case}A_Ntape_coil_{Nt_coil_case}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # [核心修改] 使用 cfg 中的全局设定来构建文件名和参数
    output_filename = f"resistance_vs_rhot_curve.svg"
    output_path = output_dir / output_filename
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"\n✅ 曲线图已成功保存至: {output_path}")
    plt.show()

if __name__ == "__main__":
    for T in cfg.TEMPERATURE_CASES:
        Rradial_vs_rhot(T)