# 6_process_charge_data.py
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from fusion_tem import device as cfg

# =============================================================================
# 全局绘图配置 (来自 6.1_plot_charge.py)
# =============================================================================
plt.style.use('default')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
markersize = 4
plt.rcParams['axes.grid'] = False
plt.rcParams['legend.frameon'] = False  # False 不显示边框，True 显示边框
dpi = 900
plt.rcParams['savefig.transparent'] = True
colors = {"Helium": "tab:blue", "Hydrogen": "tab:green"}
# =============================================================================
# 数据分析函数 (来自 6.0_analyze_charge.py)
# =============================================================================
def generate_operation_summary(all_profiles_df: pd.DataFrame) -> pd.DataFrame:
    """
    从运行过程的原始数据文件中，自动计算并生成性能概要。
    """
    print("--- 正在从运行原始数据中自动生成性能概要... ---")
    summary_rows = []
    
    # 确定列名
    try:
        coolant_col = all_profiles_df.columns[0]
        qm_col = all_profiles_df.columns[1]
        time_col = all_profiles_df.columns[2]
        t_max_col = all_profiles_df.columns[3]
        t_RMS_col = all_profiles_df.columns[6] # 假设T_min在第6列
        p_max_col = all_profiles_df.columns[7] # 假设p_max在最后一列
    except IndexError:
        print("[✘] 错误: 数据文件中的列数不足。请确保文件包含所有必需的列。")
        return pd.DataFrame() # 返回空DataFrame

    # 根据“制冷剂”和“单管流量”进行分组
    for (coolant_index, qm), group_df in all_profiles_df.groupby(by=[coolant_col, qm_col]):
        
        # 刨除t=0的时刻，用于计算最大值
        group_after_t0 = group_df[group_df[time_col] > 0]
        if group_after_t0.empty:
            # 如果没有t>0的数据，则使用整个数据集，以防万一
            group_after_t0 = group_df
       
        # 1. 计算最大温度的最大值
        max_of_T_max = group_df[t_max_col].max()
        
        # 2. 计算温度均匀度的最大值 (T_max - T_min)
        max_uniformity = group_df[t_RMS_col].max()
        
        # 3. 计算最大压降 (从 p_max 列获取) 并转换为 bar
        max_pressure_pa = group_after_t0[p_max_col].max()
        max_pressure_bar = max_pressure_pa / 1e5
        
        coolant_name = "Hydrogen" if coolant_index == 2 else "Helium"
        
        summary_rows.append({
            "Coolant": coolant_name,
            "qm (g/s)": qm,
            "Max T (K)": max_of_T_max,
            "Max Temp Uniformity (K)": max_uniformity,
            "Max Pressure Drop (kPa)": (max_pressure_bar-cfg.P_REF)*1e2
        })
        
    summary_df = pd.DataFrame(summary_rows).sort_values(by=['Coolant', 'qm (g/s)'])
    print("  [✔] 性能概要生成完毕。")
    return summary_df

# =============================================================================
# 数据绘图函数 (改编自 6.1_plot_charge.py)
# =============================================================================
def plot_summary_data(df: pd.DataFrame, output_dir: Path):
    """根据生成的性能概要数据进行绘图并保存。"""
    print("\n--- 正在根据性能概要生成图表... ---")
    
    # 分离氦气和氢气
    df_helium = df[df["Coolant"] == "Helium"]
    df_hydrogen = df[df["Coolant"] == "Hydrogen"]
    colors = ["tab:blue", "tab:green"]
    labels = ["He", "H$_2$"]
    markers = ["o", "s"]

    # -------- 图1：Peak Temperature --------
    plt.figure(figsize=(8, 6))
    plt.plot(df_helium["qm (g/s)"], df_helium["Max T (K)"],
             label=labels[0], marker=markers[0], color=colors[0])
    plt.plot(df_hydrogen["qm (g/s)"], df_hydrogen["Max T (K)"],
             label=labels[1], marker=markers[1], color=colors[1])
        # 画出可行区域target
    ymin = 20
    ymax = 1.02*max(df_helium["Max T (K)"].max(), df_hydrogen["Max T (K)"].max())
    plt.axhspan(ymin, cfg.MAX_TEMP_RUN, color='green', alpha=0.15)
    plt.ylim(ymin, ymax)
    plt.xticks(sorted(df["qm (g/s)"].unique()))
    plt.xlabel("Mass flow rate (g/s)")
    plt.ylabel("Max temperature (K)")
    plt.legend()
    plt.tight_layout()
    fig_name = "plot_peak_temperature."+cfg.PLOT_FORMAT
    plot_path = output_dir / fig_name
    plt.savefig(plot_path, dpi=cfg.PLOT_DPI, bbox_inches="tight", transparent=cfg.PLOT_TRANSPARENT)
    plt.close()
    print(f"  [✔] 图表已保存: {plot_path}")

    # -------- 图2：Temperature Uniformity --------
    plt.figure(figsize=(8, 6))
    plt.plot(df_helium["qm (g/s)"], df_helium["Max Temp Uniformity (K)"],
             label=labels[0], marker=markers[0], color=colors[0])
    plt.plot(df_hydrogen["qm (g/s)"], df_hydrogen["Max Temp Uniformity (K)"],
             label=labels[1], marker=markers[1], color=colors[1])
    plt.xlabel("Mass flow rate (g/s)")
    plt.xticks(sorted(df["qm (g/s)"].unique()))
    plt.ylabel("Max temperature non-uniformity (K)")
    plt.legend()
    plt.tight_layout()
    fig_name = "plot_temperature_uniformity."+cfg.PLOT_FORMAT
    plot_path = output_dir / fig_name
    plt.savefig(plot_path, dpi=cfg.PLOT_DPI, bbox_inches="tight", transparent=cfg.PLOT_TRANSPARENT)
    plt.close()
    print(f"  [✔] 图表已保存: {plot_path}")

    # -------- 图3：Pressure Drop --------
    plt.figure(figsize=(8, 6))
    plt.plot(df_helium["qm (g/s)"], df_helium["Max Pressure Drop (kPa)"],
             label=labels[0], marker=markers[0], color=colors[0])
    plt.plot(df_hydrogen["qm (g/s)"], df_hydrogen["Max Pressure Drop (kPa)"],
             label=labels[1], marker=markers[1], color=colors[1])
    # 画出可行区域target
    plt.axhspan(0, 1, color='green', alpha=0.15)
    plt.ylim(0, 1.2*max(df_helium["Max Pressure Drop (kPa)"].max(), df_hydrogen["Max Pressure Drop (kPa)"].max()))
    plt.xticks(sorted(df["qm (g/s)"].unique()))
    plt.xlabel("Mass flow rate (g/s)")
    plt.ylabel("Max pressure drop (kPa)")
    plt.legend(frameon=False)
    plt.tight_layout()
    fig_name = "plot_pressure_drop."+cfg.PLOT_FORMAT
    plot_path = output_dir / fig_name
    plt.savefig(plot_path, dpi=cfg.PLOT_DPI, bbox_inches="tight", transparent=cfg.PLOT_TRANSPARENT)
    plt.close()
    print(f"  [✔] 图表已保存: {plot_path}")
    print("\n[✔] 所有图表生成完毕。")


# =============================================================================
# 主函数
# =============================================================================
def main():
    """主执行函数，加载数据，生成概要，保存到Excel，并绘图。"""
    base_path = Path(__file__).parent.resolve()
    input_dir = base_path / cfg.OPERATION_DATA_DIR
    output_tables_dir = base_path / cfg.OPERATION_OUTPUT_DIR
    output_figures_dir = cfg.OUTPUTS_FIGURES_DIR / "operation"
    output_tables_dir.mkdir(parents=True, exist_ok=True)
    output_figures_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 加载包含所有工况的总数据文件
    profiles_all_path = input_dir / cfg.OPERATION_PROFILE_ALL_FILE
    if not profiles_all_path.exists():
        print(f"错误: 找不到总数据文件: {profiles_all_path}")
        return
        
    try:
        # 读取原始数据，跳过前4行
        all_profiles_df = pd.read_excel(profiles_all_path, skiprows=4, header=0)
        all_profiles_df.columns = [col.strip() for col in all_profiles_df.columns]
    except Exception as e:
        print(f"错误: 读取或解析总数据文件失败: {e}")
        return

    # 2. 从原始数据中自动生成性能概要
    df_summary = generate_operation_summary(all_profiles_df)
    
    # 如果成功生成概要，则继续
    if not df_summary.empty:
        # 3. 将概要数据保存到Excel文件
        output_path = output_tables_dir / cfg.OPERATION_SUMMARY_FILE
        df_summary.to_excel(output_path, index=False)
        print(f"\n[✔] 运行性能概要已成功保存至: {output_path}")
        
        # 在控制台打印表格，方便快速查看
        print("\n--- 运行性能概要 ---")
        print(df_summary.to_string())

        # 4. 根据概要数据绘图
        plot_summary_data(df_summary, output_figures_dir)


if __name__ == "__main__":
    main()
