# 05_analyze_cooldown.py (最终重构版)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from fusion_tem import device as cfg

# =============================================================================
# 全局绘图配置
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
# =============================================================================
# 用户配置区
# =============================================================================
# 在这里手动指定您感兴趣、并希望生成详细降温曲线的单管质量流量 (g/s)
# 这个值也会在第一张性能图上被高亮显示
QMM_TO_ANALYZE = 0.2 
# =============================================================================

def generate_performance_summary(all_profiles_df: pd.DataFrame) -> pd.DataFrame:
    """
    [新增] 从原始的总数据文件中，自动计算并生成性能概要。
    """
    print("--- 正在从原始数据中自动生成性能概要... ---")
    summary_rows = []
    # 确定列名
    variable_cols = all_profiles_df.columns[:2]
    time_col = all_profiles_df.columns[2]
    t_max_col = all_profiles_df.columns[3]
    t_uniform_col = all_profiles_df.columns[6]
    p_max_col = all_profiles_df.columns[7] 
    
    # 根据第一列（应为制冷剂种类）和第二列（qmm）进行分组
    variable_cols = all_profiles_df.columns[:2]
    
    for (coolant_index, qmm), group_df in all_profiles_df.groupby(by=list(variable_cols)):
        # --- 降温时间计算逻辑 ---
        # 筛选出温度数据变得平稳，且温度低于50K的时间点
        # 这里采用温度变化率绝对值小于某个阈值（如0.01K/hour）作为“平稳”判据
        # 统一时间序列和温度序列，确保一一对应
        # 筛选出温度低于50K的数据点
        temp_array = group_df[t_max_col].values
        time_array = group_df[time_col].values
        mask = temp_array < 50
        temp_array_50K = temp_array[mask]
        time_array_50K = time_array[mask]
        # 计算温度变化率（K/s），阈值为0.01K/h = 0.01/3600 K/s
        temp_diff = np.diff(temp_array_50K)
        time_diff = np.diff(time_array_50K)
        temp_rate = temp_diff / time_diff
        # 找到温度变化率绝对值小于阈值的点，+1补齐索引
        stable_indices = np.where(np.abs(temp_rate) < (0.01/3600))[0] + 1
        # 只在温度低于50K且变化率平稳的点中筛选
        if len(stable_indices) > 0:
            # 由于group_df的索引和mask后的索引不同，需要用原始索引
            cooldown_time_s = time_array_50K[stable_indices[0]]
        else:
            cooldown_time_s = np.nan  # 没有找到满足条件的点
        
        cooldown_time_hours = cooldown_time_s / (3600 )
        # ---------------------------------------------
        
        # 最大压降的计算逻辑保持不变
        # 最大压力为数据最后一列，去除时间为0的数据
        #max_pressure_pa = group_after_t0[p_max_col].max()
        
        # 定义为去除第一行数据后的最大值
        max_pressure_kPa = group_df[p_max_col].iloc[1:].max() / 1e3
        
        max_T_uniform_K = group_df[t_uniform_col].max()
        
        coolant_name = "Hydrogen" if coolant_index == 1 else "Helium"
        
        summary_rows.append({
            "Coolant": coolant_name,
            "qmm (g/s)": qmm,
            "Cooldown Time (h)": cooldown_time_hours,
            "T_uniform (K)": max_T_uniform_K,
            "Pressure Drop (kPa)": max_pressure_kPa-cfg.P0_HE_BAR*100 if coolant_index == 2 else max_pressure_kPa-cfg.P0_H2_BAR*100
        })
        
    summary_df = pd.DataFrame(summary_rows).sort_values(by=['Coolant', 'qmm (g/s)'])
    print("  [✔] 性能概要生成完毕。",summary_df)
    return summary_df

def plot_cooldown_performance(df_summary: pd.DataFrame, qmm_to_highlight: float, output_dir: Path):
    """
    [重构] 绘制降温性能的双Y轴曲线图。
    现在可以同时绘制多种制冷剂的数据，并高亮用户指定的流量点。
    """
    print("--- 正在绘制多种制冷剂的性能对比曲线图... ---")
    
    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax2 = ax1.twinx()

    # 定义颜色和标记样式
    colors_time = {'Helium': 'tab:blue', 'Hydrogen': 'tab:green'}
    colors_pressure = {'Helium': 'tab:blue', 'Hydrogen': 'tab:green'}
    markers = {'Helium': 'o', 'Hydrogen': 's'}
    
    handles, labels = [], [] # 用于存储所有图例信息

    # 使用 groupby 分别处理每种制冷剂的数据
    for coolant_name, group_df in df_summary.groupby('Coolant'):
        qmm = group_df['qmm (g/s)']
        time_h = group_df['Cooldown Time (h)']
        pressure_drop = group_df['Pressure Drop (kPa)']
        if coolant_name == 'Helium':
            coolant = 'He'
        elif coolant_name == 'Hydrogen':
            coolant = 'H$_2$'
        
        # 绘制左侧Y轴：降温时间
        line1, = ax1.plot(qmm, time_h, color=colors_time.get(coolant_name,), 
                          marker=markers.get(coolant_name, 'x'), label=f'{coolant} Time', markersize=markersize)
        
        # 绘制右侧Y轴：压降
        line2, = ax2.plot(qmm, pressure_drop, color=colors_pressure.get(coolant_name), 
                          marker=markers.get(coolant_name, 'o'), linestyle='--', label=f'{coolant} Max pressure drop', markersize=markersize)
        
        # 收集图例信息
        handles.extend([line1, line2])

    # 设置坐标轴
    ax1.set_xlabel('Mass flow rate (g/s)')
    ax1.set_ylabel('Cooldown Time (h)')
    ax2.set_ylabel('Max pressure drop (kPa)')
    
    # 创建统一的图例
    ax1.legend(handles, [h.get_label() for h in handles], loc='upper center')
    
    plt.tight_layout()

    output_path = output_dir / f"cooldown_performance_comparison.{cfg.PLOT_FORMAT}"
    plt.savefig(output_path, format=cfg.PLOT_FORMAT, transparent=cfg.PLOT_TRANSPARENT, dpi=300)
    print(f"  [✔] 性能对比图已保存: {output_path.name}")
    plt.show()

def plot_cooldown_curve_comparison(target_qmm: float, all_profiles_df: pd.DataFrame, output_dir: Path):
    """
    [重构] 在同一个坐标系下，对比绘制氦气和氢气的详细降温曲线。
    横轴单位从秒转换为小时。
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    print(f"\n--- 准备为单管流量 qmm ≈ {target_qmm:.2f} g/s 对比绘制降温曲线... ---")
    # 1. 定义一个按制冷剂索引区分的 Colormap 字典
    #    2 -> Helium -> 蓝色系
    #    1 -> Hydrogen -> 绿色系
    color_maps = {
        2: plt.get_cmap('Blues'),
        1: plt.get_cmap('Greens')
    }
    # 2. 定义一个按温度类型区分的样式字典 (参考蓝本)
    style_map = {
        'T_max': {'marker': 'o', 'linestyle': '-'},
        'T_avg': {'marker': 's', 'linestyle': '--'},
        'T_min': {'marker': '^', 'linestyle': '-'}
    }
    
    coolants_to_plot = {'He': 2, 'H$_2$': 1}
    
    # 循环处理两种制冷剂
    for name, index in coolants_to_plot.items():
        coolant_df = all_profiles_df[all_profiles_df['Material Switch 2 index'] == index]
        if coolant_df.empty:
            print(f"  [!] 警告: 在数据文件中找不到 {name} (index={index}) 的数据，跳过。")
            continue
            
        available_qmms = coolant_df['qmm (g/s)'].unique()
        closest_qmm = available_qmms[np.argmin(np.abs(available_qmms - target_qmm))]
        print(f"  -> 对于 {name}, 找到最接近的目标流量为: {closest_qmm:.2f} g/s。")

        final_df = coolant_df[coolant_df['qmm (g/s)'] == closest_qmm].iloc[0:24*4]
        if final_df.empty: continue

        # --- [核心修改] ---
        # 1. 从 "Time (s)" 列读取时间数据
        time_seconds = final_df['Time (s)']
        # 2. 将时间单位从秒转换为小时
        time_hours = time_seconds / 3600
        # -------------------

        t_max = final_df['T_max (K)']
        t_min = final_df['T_min (K)']
        t_avg = final_df.get('T_ave (K)', pd.Series(dtype='float64'))

        # --- [核心修改] ---
        # 2. 根据制冷剂确定线型
        linestyle = '-' if name == 'Helium' else '--'
        
        # 3. 从 Colormap 中为 T_max, T_avg, T_min 选择不同的深浅颜色
        current_cmap = color_maps[index]
        color_max = current_cmap(0.9) # 最深
        color_avg = current_cmap(0.7)
        color_min = current_cmap(0.5) # 最浅
        
        # 4. 在绘图时，组合使用颜色和样式字典
        ax.plot(time_hours, t_max, label=f'{name} $T_{{\\mathrm{{max}}}}$', 
                color=color_max, markersize=markersize, **style_map['T_max'])
        
        ax.plot(time_hours, t_avg, label=f'{name} $T_{{\\mathrm{{ave}}}}$', 
                color=color_avg, markersize=markersize, **style_map['T_avg'])
                
        ax.plot(time_hours, t_min, label=f'{name} $T_{{\\mathrm{{min}}}}$', 
                color=color_min, markersize=markersize, **style_map['T_min'])
        
    
    # [修改] 更新X轴标签为小时 (h)
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Temperature (K)')
    ax.grid(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    output_path = output_dir / f"cooldown_curve_comparison_qmm{target_qmm:.1f}.{cfg.PLOT_FORMAT}"
    plt.savefig(output_path, format=cfg.PLOT_FORMAT, transparent=cfg.PLOT_TRANSPARENT, dpi=300)
    print(f"  [✔] 降温对比曲线图已保存: {output_path.name}")
    #plt.show()


def main():
    base_path = Path(__file__).parent.resolve()
    input_dir = base_path / cfg.COOLDOWN_DATA_DIR
    output_tables_dir = base_path / cfg.COOLDOWN_OUTPUT_DIR
    output_figures_dir = cfg.OUTPUTS_FIGURES_DIR / "cooldown"
    output_tables_dir.mkdir(parents=True, exist_ok=True)
    output_figures_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 加载包含所有工况的总数据文件
    profiles_all_path = input_dir / cfg.COOLDOWN_PROFILES_ALL_FILE
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
    df_summary = generate_performance_summary(all_profiles_df)
    df_summary.to_excel(output_tables_dir / "summary.xlsx", index=False)
    
    # 3. 使用生成的概要数据，绘制性能对比图
    if not df_summary.empty:
        plot_cooldown_performance(df_summary, QMM_TO_ANALYZE, output_figures_dir)
    
    # 4. 使用原始数据，绘制指定工况的详细降温曲线
    qmm_list = df_summary['qmm (g/s)'].unique()
    for qmm in qmm_list:
        plot_cooldown_curve_comparison(qmm, all_profiles_df, output_figures_dir)

if __name__ == "__main__":
    main()
