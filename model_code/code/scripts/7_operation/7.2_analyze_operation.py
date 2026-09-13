# 07_analyze_operation_v2.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
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
dpi = 900
plt.rcParams['savefig.transparent'] = True
colors = {"Helium": "tab:blue", "Hydrogen": "tab:green"}
plt.rcParams['axes.grid'] = False  # 取消网格线
plt.rcParams['legend.frameon'] = False  # False 不显示边框，True 显示边框
# =============================================================================


def analyze_single_case(group_df: pd.DataFrame) -> dict:
    """
    对单个工况（一个确定的制冷剂/流量/功率组合）的数据进行分析，
    提取所有需要的关键性能指标。
    """
    # 确定列名
    time_col, t_max_col, t_avg_col, t_rms_col = group_df.columns[3], group_df.columns[4], group_df.columns[5], group_df.columns[7]

    # 1. 计算失超前的平稳运行温度
    pre_quench_stable_df = group_df[(group_df[time_col] >= cfg.STABLE_TEMP_START_S) & (group_df[time_col] < cfg.QUENCH_START_S)]
    stable_temp_k = pre_quench_stable_df[t_max_col].mean() if not pre_quench_stable_df.empty else np.nan

    # 2. 读取失超期间的最大温度
    quench_df = group_df[(group_df[time_col] >= cfg.QUENCH_START_S) & (group_df[time_col] <= cfg.QUENCH_END_S)]
    peak_temp_k = quench_df[t_max_col].max() if not quench_df.empty else np.nan

    # 3. 计算温度恢复时间：定义为失超热源撤去后，最大温度回到21K所需的时间
    recovery_target_temp = 21.0  # 目标温度为21K
    post_quench_df = group_df[group_df[time_col] > cfg.QUENCH_END_S]
    recovered_points = post_quench_df[post_quench_df[t_max_col] <= recovery_target_temp]
    
    if recovered_points.empty:
        recovery_time_s = np.nan  # 未能恢复到21K
    else:
        time_at_recovery_s = recovered_points[time_col].iloc[0]
        recovery_time_s = time_at_recovery_s - cfg.QUENCH_END_S  # 从失超结束后开始计时
    

    # 4. 读取温度均匀度(T_RMS)的最大值
    max_t_rms = group_df[t_rms_col].max()
    
    return {
        "Stable Temp (K)": stable_temp_k,
        "Max T (K)": peak_temp_k,
        "Recovery Time (s)": recovery_time_s,
        "Max T (K)": max_t_rms,
    }

def generate_summary(all_profiles_df: pd.DataFrame) -> pd.DataFrame:
    """从原始数据中，生成包含所有工况分析结果的性能总览表。"""
    print("--- 正在从原始数据生成性能总览表... ---")
    summary_rows = []
    
    # 根据制冷剂、流量、失超功率这三个变量进行分组
    variable_cols = all_profiles_df.columns[:3]
    for params, group_df in all_profiles_df.groupby(by=list(variable_cols)):
        coolant_index, qm, pquench = params
        
        # 对每个工况的数据调用分析函数
        analysis_results = analyze_single_case(group_df)
        
        coolant_name = "Hydrogen" if coolant_index == 2 else "Helium"
        
        row_data = {
            "Coolant": coolant_name,
            "qm (g/s)": qm,
            "Pquench (W)": pquench,
        }
        row_data.update(analysis_results)
        summary_rows.append(row_data)
        
    summary_df = pd.DataFrame(summary_rows)
    print("  [✔] 性能总览表生成完毕。")
    return summary_df

def plot_summary_curves(summary_df: pd.DataFrame, variable_col: str, fixed_col: str, output_dir: Path):
    """根据生成的总览表，绘制性能曲线图。"""
    print(f"--- 正在绘制性能曲线图 (变量: {variable_col})... ---")
    
    # 定义要绘制的指标和对应的Y轴标签
    metrics_to_plot = {
        "Max T (K)": "Max Temperature (K)",
        "Recovery Time (s)": "Recovery Time (s)",
        "Max T (K)": "Max Temperature (K)",
        "Stable Temp (K)": "Stable Temperature (K)"
    }
   
    for metric_col, y_label in metrics_to_plot.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 分别绘制氦气和氢气的曲线
        for coolant, color in colors.items():
            coolant_data = summary_df[summary_df['Coolant'] == coolant]
            if not coolant_data.empty:
                ax.plot(coolant_data[variable_col], coolant_data[metric_col], marker='o', linestyle='-', color=color, label=coolant)
        if variable_col == 'qm (g/s)':
            x_label = '$\\dot{m}$ (g/s)'
        else:
            x_label = 'Power of transient quench (W)'
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        #ax.set_title(f"{y_label} vs. {variable_col}\n(Fixed {fixed_col})")
        ax.legend()        
        # 保存图像
        filename = f"{metric_col.replace(' ', '_').replace('(', '').replace(')', '')}_vs_{variable_col.replace(' ', '_').replace('/', '')}.{cfg.PLOT_FORMAT}"
        output_path = output_dir / filename
        plt.savefig(output_path, format=cfg.PLOT_FORMAT, transparent=cfg.PLOT_TRANSPARENT, dpi=300)
        plt.close(fig)
    print(f"  [✔] {len(metrics_to_plot)} 张性能曲线图已保存。")

def process_experiment_file(input_path: Path, output_tables_dir: Path, output_figures_dir: Path):
    """
    [新增] 这是一个高级函数，封装了对单个实验文件的完整处理流程。
    """
    if not input_path.exists():
        print(f"警告: 找不到输入文件 '{input_path}'，将跳过此项分析。")
        return

    print(f"\n{'='*30}\n处理实验文件: {input_path.name}\n{'='*30}")
    
    # 1. 加载数据
    try:
        all_profiles_df = pd.read_excel(input_path, skiprows=4, header=0)
        all_profiles_df.columns = [col.strip() for col in all_profiles_df.columns]
    except Exception as e:
        print(f"错误: 读取或解析文件失败: {e}")
        return

    # 2. 生成性能总览表
    summary_df = generate_summary(all_profiles_df)
    if summary_df.empty:
        print("错误: 未能从文件中生成有效的性能概要。")
        return
        
    # 3. 自动识别变量和固定量
    qm_list = summary_df['qm (g/s)'].unique()
    pq_list = summary_df['Pquench (W)'].unique()
    
    if len(qm_list) > 1 and len(pq_list) == 1:
        variable_col = 'qm (g/s)'
        fixed_val = pq_list[0]
        fixed_col_str = f'Pquench={int(fixed_val)}'
        print(f"  -> 自动识别: 变量为 qm, 固定 Pquench = {fixed_val} W")
    elif len(pq_list) > 1 and len(qm_list) == 1:
        variable_col = 'Pquench (W)'
        fixed_val = qm_list[0]
        fixed_col_str = f'qm={fixed_val}'
        print(f"  -> 自动识别: 变量为 Pquench, 固定 qm = {fixed_val} g/s")
    else:
        print("  -> 警告: 无法明确识别唯一的变量和固定量，跳过绘图。")
        # 即使无法绘图，依然保存总览表
        summary_path = output_tables_dir / f"summary_{input_path.stem}.xlsx"
        summary_df.to_excel(summary_path, index=False)
        print(f"  [✔] 完整的性能总览表已保存至: {summary_path}")
        return
        
    # 4. 创建专用的输出子文件夹
    exp_output_dir = output_tables_dir / f"Vary_{variable_col.split(' ')[0]}_Fixed_{fixed_col_str}"
    exp_figures_dir = output_figures_dir / f"Vary_{variable_col.split(' ')[0]}_Fixed_{fixed_col_str}"
    exp_output_dir.mkdir(parents=True, exist_ok=True)
    exp_figures_dir.mkdir(parents=True, exist_ok=True)
    
    # 5. 保存总览表
    summary_path = exp_output_dir / "performance_summary.xlsx"
    summary_df.to_excel(summary_path, index=False)
    print(f"  [✔] 性能总览表已保存至: {summary_path.name}")
    print(summary_df.to_string())

    # 6. 绘制性能曲线图
    plot_summary_curves(summary_df, variable_col, fixed_col_str, exp_figures_dir)


def main():
    """
    主执行函数，现在会自动处理所有预定义的实验文件。
    """
    base_path = Path(__file__).parent.resolve()
    input_dir = base_path / cfg.OPERATION_DATA_DIR
    output_tables_dir = base_path / cfg.OPERATION_OUTPUT_DIR
    output_figures_dir = cfg.OUTPUTS_FIGURES_DIR / "operation"
    output_tables_dir.mkdir(parents=True, exist_ok=True)
    output_figures_dir.mkdir(parents=True, exist_ok=True)
    
    # 自动处理“改变qm”的实验文件
    process_experiment_file(
        input_path=input_dir / cfg.OPERATION_FILE_VARY_QM,
        output_tables_dir=output_tables_dir,
        output_figures_dir=output_figures_dir
    )
    
    # 自动处理“改变Pq”的实验文件
    process_experiment_file(
        input_path=input_dir / cfg.OPERATION_FILE_VARY_PQ,
        output_tables_dir=output_tables_dir,
        output_figures_dir=output_figures_dir
    )

    print(f"\n[✔] 所有分析任务完成。结果保存在 '{output_tables_dir}' 的子文件夹中。")

if __name__ == "__main__":
    # 提示：为保持答案简洁，我省略了未改动的函数（如analyze_single_case等）。
    # 您只需用本回答提供的 process_experiment_file 和 main 函数，
    # 替换掉您脚本中的旧版本即可。
    main()
