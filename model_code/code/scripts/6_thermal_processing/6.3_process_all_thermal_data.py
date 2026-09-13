import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from fusion_tem import device as cfg


# =============================================================================
# 通用绘图样式配置
# =============================================================================
def setup_plot_style():
    """统一设置 Matplotlib 全局样式，供本模块所有图表复用。"""
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
    plt.rcParams['axes.grid'] = False
    plt.rcParams['legend.frameon'] = False
    plt.rcParams['savefig.transparent'] = True


# =============================================================================
# 零、降温工况：汇总与绘图（整合自 5.0_analyze_cooldown.py）
# =============================================================================
def generate_cooldown_summary(all_profiles_df: pd.DataFrame) -> pd.DataFrame:
    """
    从降温过程的原始数据文件中，自动计算并生成性能概要。
    列假定与 5.0_analyze_cooldown.py 一致：
      0: Coolant index, 1: qmm (g/s), 2: Time (s),
      3: T_max, 6: T_uniform, 7: p_max.
    """
    print("--- 正在从降温原始数据中自动生成性能概要... ---")
    summary_rows = []

    # 列名索引
    variable_cols = all_profiles_df.columns[:2]
    time_col = all_profiles_df.columns[2]
    t_max_col = all_profiles_df.columns[3]
    t_uniform_col = all_profiles_df.columns[6]
    p_max_col = all_profiles_df.columns[7]

    for (coolant_index, qmm), group_df in all_profiles_df.groupby(by=list(variable_cols)):
        # 1) 计算达到接近稳态的降温时间：温度 < 50K 且 dT/dt 很小
        temp_array = group_df[t_max_col].values
        time_array = group_df[time_col].values
        mask = temp_array < 50
        temp_array_50K = temp_array[mask]
        time_array_50K = time_array[mask]

        if len(temp_array_50K) > 1:
            temp_diff = np.diff(temp_array_50K)
            time_diff = np.diff(time_array_50K)
            temp_rate = temp_diff / time_diff
            stable_indices = np.where(np.abs(temp_rate) < (0.01 / 3600))[0] + 1
            if len(stable_indices) > 0:
                cooldown_time_s = time_array_50K[stable_indices[0]]
            else:
                cooldown_time_s = np.nan
        else:
            cooldown_time_s = np.nan

        cooldown_time_hours = cooldown_time_s / 3600.0 if np.isfinite(cooldown_time_s) else np.nan

        # 2) 最大压降（kPa），忽略第一行（通常是 t=0）
        if len(group_df) > 1:
            max_pressure_kPa = group_df[p_max_col].iloc[1:].max() / 1e3
        else:
            max_pressure_kPa = np.nan

        # 3) 最大温度非均匀度
        max_T_uniform_K = group_df[t_uniform_col].max()

        coolant_name = "Hydrogen" if coolant_index == 1 else "Helium"

        # 4) 相对于初始压力的压降
        if coolant_index == 2:
            # 2: Helium
            ref_bar = cfg.P0_HE_BAR
        else:
            # 1: Hydrogen
            ref_bar = cfg.P0_H2_BAR
        pressure_drop_kPa_rel = max_pressure_kPa - ref_bar * 100

        summary_rows.append(
            {
                "Coolant": coolant_name,
                "qmm (g/s)": qmm,
                "Cooldown Time (h)": cooldown_time_hours,
                "T_uniform (K)": max_T_uniform_K,
                "Pressure Drop (kPa)": pressure_drop_kPa_rel,
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(by=["Coolant", "qmm (g/s)"])
    print("  [✔] 降温性能概要生成完毕。")
    return summary_df


def plot_cooldown_performance(df_summary: pd.DataFrame, output_dir: Path):
    """
    绘制降温性能对比图：
      - Cooldown Time vs qmm
      - Pressure Drop vs qmm
      - T_uniform vs qmm
    """
    print("--- 正在绘制降温性能对比曲线图... ---")
    markersize = 4

    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax2 = ax1.twinx()

    colors_time = {"Helium": "tab:blue", "Hydrogen": "tab:green"}
    colors_pressure = {"Helium": "tab:blue", "Hydrogen": "tab:green"}
    markers = {"Helium": "o", "Hydrogen": "s"}

    handles = []

    for coolant_name, group_df in df_summary.groupby("Coolant"):
        qmm = group_df["qmm (g/s)"]
        time_h = group_df["Cooldown Time (h)"]
        pressure_drop = group_df["Pressure Drop (kPa)"]

        label_prefix = "He" if coolant_name == "Helium" else "H$_2$"

        line1, = ax1.plot(
            qmm,
            time_h,
            color=colors_time.get(coolant_name),
            marker=markers.get(coolant_name, "x"),
            label=f"{label_prefix} Time",
            markersize=markersize,
        )
        line2, = ax2.plot(
            qmm,
            pressure_drop,
            color=colors_pressure.get(coolant_name),
            marker=markers.get(coolant_name, "o"),
            linestyle="--",
            label=f"{label_prefix} Max pressure drop",
            markersize=markersize,
        )
        handles.extend([line1, line2])

    ax1.set_xlabel("Mass flow rate (g/s)")
    ax1.set_ylabel("Cooldown Time (h)")
    ax2.set_ylabel("Max pressure drop (kPa)")
    ax1.legend(handles, [h.get_label() for h in handles], loc="upper center")
    plt.tight_layout()

    output_path = output_dir / f"cooldown_performance_comparison.{cfg.PLOT_FORMAT}"
    plt.savefig(
        output_path,
        format=cfg.PLOT_FORMAT,
        transparent=cfg.PLOT_TRANSPARENT,
        dpi=cfg.PLOT_DPI,
    )
    print(f"  [✔] 降温性能对比图已保存: {output_path}")
    plt.close(fig)


def plot_cooldown_curve_comparison(
    target_qmm: float, all_profiles_df: pd.DataFrame, output_dir: Path
):
    """
    在同一个坐标系下，对比绘制氦气和氢气的详细降温曲线 T_max/T_avg/T_min vs Time(h)。
    直接复用 5.0_analyze_cooldown.py 中的逻辑。
    """
    markersize = 4
    fig, ax = plt.subplots(figsize=(12, 7))
    print(f"\n--- 准备为单管流量 qmm ≈ {target_qmm:.2f} g/s 对比绘制降温曲线... ---")

    color_maps = {2: plt.get_cmap("Blues"), 1: plt.get_cmap("Greens")}
    style_map = {
        "T_max": {"marker": "o", "linestyle": "-"},
        "T_avg": {"marker": "s", "linestyle": "--"},
        "T_min": {"marker": "^", "linestyle": "-"},
    }
    coolants_to_plot = {"He": 2, "H$_2$": 1}

    # 冷却剂索引列：在原始数据中为第1列（与 5.0_analyze_cooldown 保持一致）
    coolant_index_col = all_profiles_df.columns[0]

    for name, index in coolants_to_plot.items():
        coolant_df = all_profiles_df[all_profiles_df[coolant_index_col] == index]
        if coolant_df.empty:
            print(f"  [!] 警告: 在数据文件中找不到 {name} (index={index}) 的数据，跳过。")
            continue

        available_qmms = coolant_df["qmm (g/s)"].unique()
        closest_qmm = available_qmms[np.argmin(np.abs(available_qmms - target_qmm))]
        print(f"  -> 对于 {name}, 找到最接近的目标流量为: {closest_qmm:.2f} g/s。")

        final_df = coolant_df[coolant_df["qmm (g/s)"] == closest_qmm].iloc[0 : 24 * 4]
        if final_df.empty:
            continue

        time_seconds = final_df["Time (s)"]
        time_hours = time_seconds / 3600.0

        t_max = final_df["T_max (K)"]
        t_min = final_df["T_min (K)"]
        t_avg = final_df.get("T_ave (K)", pd.Series(dtype="float64"))

        current_cmap = color_maps[index]
        color_max = current_cmap(0.9)
        color_avg = current_cmap(0.7)
        color_min = current_cmap(0.5)

        ax.plot(
            time_hours,
            t_max,
            label=f'{name} $T_{{\\mathrm{{max}}}}$',
            color=color_max,
            markersize=markersize,
            **style_map["T_max"],
        )
        ax.plot(
            time_hours,
            t_avg,
            label=f'{name} $T_{{\\mathrm{{ave}}}}$',
            color=color_avg,
            markersize=markersize,
            **style_map["T_avg"],
        )
        ax.plot(
            time_hours,
            t_min,
            label=f'{name} $T_{{\\mathrm{{min}}}}$',
            color=color_min,
            markersize=markersize,
            **style_map["T_min"],
        )

    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Temperature (K)")
    ax.grid(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    output_path = output_dir / f"cooldown_curve_comparison_qmm{target_qmm:.1f}.{cfg.PLOT_FORMAT}"
    plt.savefig(
        output_path,
        format=cfg.PLOT_FORMAT,
        transparent=cfg.PLOT_TRANSPARENT,
        dpi=cfg.PLOT_DPI,
    )
    print(f"  [✔] 降温对比曲线图已保存: {output_path}")
    plt.close(fig)


def process_cooldown_data():
    """封装原 5.0_analyze_cooldown.py 的主流程。"""
    base_path = Path(__file__).parent.resolve()
    input_dir = base_path / cfg.COOLDOWN_DATA_DIR
    output_tables_dir = base_path / cfg.COOLDOWN_OUTPUT_DIR
    output_figures_dir = cfg.OUTPUTS_FIGURES_DIR / "cooldown"
    output_tables_dir.mkdir(parents=True, exist_ok=True)
    output_figures_dir.mkdir(parents=True, exist_ok=True)

    profiles_all_path = input_dir / cfg.COOLDOWN_PROFILES_ALL_FILE
    if not profiles_all_path.exists():
        print(f"[降温] 错误: 找不到总数据文件: {profiles_all_path}")
        return

    try:
        all_profiles_df = pd.read_excel(profiles_all_path, skiprows=4, header=0)
        all_profiles_df.columns = [col.strip() for col in all_profiles_df.columns]
    except Exception as e:
        print(f"[降温] 错误: 读取或解析总数据文件失败: {e}")
        return

    df_summary = generate_cooldown_summary(all_profiles_df)
    if df_summary.empty:
        return

    summary_path = output_tables_dir / "summary.xlsx"
    df_summary.to_excel(summary_path, index=False)
    print(f"[✔] 降温性能概要已保存至: {summary_path}")

    plot_cooldown_performance(df_summary, output_figures_dir)

    # 为每个 qmm 绘制详细降温曲线
    qmm_list = df_summary["qmm (g/s)"].unique()
    for qmm in qmm_list:
        plot_cooldown_curve_comparison(qmm, all_profiles_df, output_figures_dir)


# =============================================================================
# 一、充电工况：汇总与绘图（整合自 6.1_process_charge_data.py）
# =============================================================================
def generate_charge_summary(all_profiles_df: pd.DataFrame) -> pd.DataFrame:
    """
    从充电过程的原始数据文件中，自动计算并生成性能概要。
    """
    print("--- 正在从充电原始数据中自动生成性能概要... ---")
    summary_rows = []

    # 确定列名（保持与 6.1_process_charge_data.py 一致）
    try:
        coolant_col = all_profiles_df.columns[0]
        qmm_col = all_profiles_df.columns[1]
        qmj_col = all_profiles_df.columns[2]
        time_col = all_profiles_df.columns[3]
        t_max_col = all_profiles_df.columns[4]
        # 第7列为 T_max - T_min（温度均匀度）
        t_RMS_col = all_profiles_df.columns[7]
        # 第8列为 p_max
        p_max_col = all_profiles_df.columns[8]
    except IndexError:
        print("[✘] 错误: 充电数据文件中的列数不足。请确保文件包含所有必需的列。")
        return pd.DataFrame()

    # 根据“制冷剂 + 单管流量 + 总流量”进行分组
    for (coolant_index, qmm, qmj), group_df in all_profiles_df.groupby(
        by=[coolant_col, qmm_col, qmj_col]
    ):
        # 刨除 t=0 的时刻，用于计算最大值
        group_after_t0 = group_df[group_df[time_col] > 0]
        if group_after_t0.empty:
            group_after_t0 = group_df

        qm = qmm + qmj

        # 1. 最大温度
        max_of_T_max = group_df[t_max_col].max()

        # 2. 温度均匀度最大值
        max_uniformity = group_df[t_RMS_col].max()

        # 3. 最大压降（Pa -> bar -> kPa，相对 cfg.P_REF）
        max_pressure_pa = group_after_t0[p_max_col].max()
        max_pressure_bar = max_pressure_pa / 1e5

        coolant_name = "Hydrogen" if coolant_index == 2 else "Helium"

        summary_rows.append(
            {
                "Coolant": coolant_name,
                "qm (g/s)": qm,
                "Max T (K)": max_of_T_max,
                "Max Temp Uniformity (K)": max_uniformity,
                "Max Pressure Drop (kPa)": (max_pressure_bar - cfg.P_REF) * 1e2,
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        by=["Coolant", "qm (g/s)"]
    )
    print("  [✔] 充电性能概要生成完毕。")
    return summary_df


def plot_charge_summary(df: pd.DataFrame, output_dir: Path):
    """根据充电性能概要数据进行绘图并保存。"""
    print("\n--- 正在根据充电性能概要生成图表... ---")

    df_helium = df[df["Coolant"] == "Helium"]
    df_hydrogen = df[df["Coolant"] == "Hydrogen"]
    colors = ["tab:blue", "tab:green"]
    labels = ["He", "H$_2$"]
    markers = ["o", "s"]

    # 图1：Peak Temperature
    plt.figure(figsize=(8, 6))
    if not df_helium.empty:
        plt.plot(
            df_helium["qm (g/s)"],
            df_helium["Max T (K)"],
            label=labels[0],
            marker=markers[0],
            color=colors[0],
        )
    if not df_hydrogen.empty:
        plt.plot(
            df_hydrogen["qm (g/s)"],
            df_hydrogen["Max T (K)"],
            label=labels[1],
            marker=markers[1],
            color=colors[1],
        )
    ymin = 20
    ymax = 1.02 * max(
        df_helium["Max T (K)"].max() if not df_helium.empty else ymin,
        df_hydrogen["Max T (K)"].max() if not df_hydrogen.empty else ymin,
    )
    plt.axhspan(ymin, cfg.MAX_TEMP_RUN, color="green", alpha=0.15)
    plt.ylim(ymin, ymax)
    plt.xlabel("Mass flow rate (g/s)")
    plt.ylabel("Max temperature (K)")
    plt.legend()
    plt.tight_layout()
    plot_path = output_dir / f"plot_peak_temperature.{cfg.PLOT_FORMAT}"
    plt.savefig(
        plot_path,
        dpi=cfg.PLOT_DPI,
        bbox_inches="tight",
        transparent=cfg.PLOT_TRANSPARENT,
    )
    plt.close()
    print(f"  [✔] 图表已保存: {plot_path}")

    # 图2：Temperature Uniformity
    plt.figure(figsize=(8, 6))
    if not df_helium.empty:
        plt.plot(
            df_helium["qm (g/s)"],
            df_helium["Max Temp Uniformity (K)"],
            label=labels[0],
            marker=markers[0],
            color=colors[0],
        )
    if not df_hydrogen.empty:
        plt.plot(
            df_hydrogen["qm (g/s)"],
            df_hydrogen["Max Temp Uniformity (K)"],
            label=labels[1],
            marker=markers[1],
            color=colors[1],
        )
    plt.xlabel("Mass flow rate (g/s)")
    plt.ylabel("Max temperature non-uniformity (K)")
    plt.legend()
    plt.tight_layout()
    plot_path = output_dir / f"plot_temperature_uniformity.{cfg.PLOT_FORMAT}"
    plt.savefig(
        plot_path,
        dpi=cfg.PLOT_DPI,
        bbox_inches="tight",
        transparent=cfg.PLOT_TRANSPARENT,
    )
    plt.close()
    print(f"  [✔] 图表已保存: {plot_path}")

    # 图3：Pressure Drop
    plt.figure(figsize=(8, 6))
    if not df_helium.empty:
        plt.plot(
            df_helium["qm (g/s)"],
            df_helium["Max Pressure Drop (kPa)"],
            label=labels[0],
            marker=markers[0],
            color=colors[0],
        )
    if not df_hydrogen.empty:
        plt.plot(
            df_hydrogen["qm (g/s)"],
            df_hydrogen["Max Pressure Drop (kPa)"],
            label=labels[1],
            marker=markers[1],
            color=colors[1],
        )
    plt.axhspan(0, 1, color="green", alpha=0.15)
    max_pd = max(
        df_helium["Max Pressure Drop (kPa)"].max()
        if not df_helium.empty
        else 1,
        df_hydrogen["Max Pressure Drop (kPa)"].max()
        if not df_hydrogen.empty
        else 1,
    )
    plt.ylim(0, 1.2 * max_pd)
    plt.xlabel("Mass flow rate (g/s)")
    plt.ylabel("Max pressure drop (kPa)")
    plt.legend()
    plt.tight_layout()
    plot_path = output_dir / f"plot_pressure_drop.{cfg.PLOT_FORMAT}"
    plt.savefig(
        plot_path,
        dpi=cfg.PLOT_DPI,
        bbox_inches="tight",
        transparent=cfg.PLOT_TRANSPARENT,
    )
    plt.close()
    print(f"  [✔] 图表已保存: {plot_path}")
    print("[✔] 充电性能图表生成完毕。")


def process_charge_data():
    """封装原 6.1_process_charge_data.py 的主流程。"""
    base_path = Path(__file__).parent.resolve()
    input_dir = base_path / cfg.CHARGE_DATA_DIR
    output_tables_dir = base_path / cfg.CHARGE_OUTPUT_DIR
    output_figures_dir = cfg.OUTPUTS_FIGURES_DIR / "charge"
    output_tables_dir.mkdir(parents=True, exist_ok=True)
    output_figures_dir.mkdir(parents=True, exist_ok=True)

    profiles_all_path = input_dir / cfg.CHARGE_PROFILE_ALL_FILE
    if not profiles_all_path.exists():
        print(f"[充电] 错误: 找不到总数据文件: {profiles_all_path}")
        return

    try:
        all_profiles_df = pd.read_excel(profiles_all_path, skiprows=4, header=0)
        all_profiles_df.columns = [col.strip() for col in all_profiles_df.columns]
    except Exception as e:
        print(f"[充电] 错误: 读取或解析总数据文件失败: {e}")
        return

    df_summary = generate_charge_summary(all_profiles_df)
    if df_summary.empty:
        return

    output_path = output_tables_dir / cfg.CHARGE_SUMMARY_FILE
    df_summary.to_excel(output_path, index=False)
    print(f"\n[✔] 充电性能概要已成功保存至: {output_path}")
    print("\n--- 充电性能概要 ---")
    print(df_summary.to_string())

    plot_charge_summary(df_summary, output_figures_dir)


# =============================================================================
# 二、稳态运行工况：汇总与绘图（整合自 6.2_process_operation_data.py）
# =============================================================================
def generate_operation_summary(all_profiles_df: pd.DataFrame) -> pd.DataFrame:
    """
    从运行过程的原始数据文件中，自动计算并生成性能概要。
    """
    print("--- 正在从运行原始数据中自动生成性能概要... ---")
    summary_rows = []

    try:
        coolant_col = all_profiles_df.columns[0]
        qm_col = all_profiles_df.columns[1]
        time_col = all_profiles_df.columns[2]
        t_max_col = all_profiles_df.columns[3]
        t_RMS_col = all_profiles_df.columns[6]
        p_max_col = all_profiles_df.columns[7]
    except IndexError:
        print("[✘] 错误: 运行数据文件中的列数不足。请确保文件包含所有必需的列。")
        return pd.DataFrame()

    for (coolant_index, qm), group_df in all_profiles_df.groupby(
        by=[coolant_col, qm_col]
    ):
        group_after_t0 = group_df[group_df[time_col] > 0]
        if group_after_t0.empty:
            group_after_t0 = group_df

        max_of_T_max = group_df[t_max_col].max()
        max_uniformity = group_df[t_RMS_col].max()
        max_pressure_pa = group_after_t0[p_max_col].max()
        max_pressure_bar = max_pressure_pa / 1e5

        coolant_name = "Hydrogen" if coolant_index == 2 else "Helium"

        summary_rows.append(
            {
                "Coolant": coolant_name,
                "qm (g/s)": qm,
                "Max T (K)": max_of_T_max,
                "Max Temp Uniformity (K)": max_uniformity,
                "Max Pressure Drop (kPa)": (max_pressure_bar - cfg.P_REF) * 1e2,
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        by=["Coolant", "qm (g/s)"]
    )
    print("  [✔] 运行性能概要生成完毕。")
    return summary_df


def plot_operation_summary(df: pd.DataFrame, output_dir: Path):
    """根据运行性能概要数据进行绘图并保存。"""
    print("\n--- 正在根据运行性能概要生成图表... ---")

    df_helium = df[df["Coolant"] == "Helium"]
    df_hydrogen = df[df["Coolant"] == "Hydrogen"]
    colors = ["tab:blue", "tab:green"]
    labels = ["He", "H$_2$"]
    markers = ["o", "s"]

    # 图1：Peak Temperature
    plt.figure(figsize=(8, 6))
    if not df_helium.empty:
        plt.plot(
            df_helium["qm (g/s)"],
            df_helium["Max T (K)"],
            label=labels[0],
            marker=markers[0],
            color=colors[0],
        )
    if not df_hydrogen.empty:
        plt.plot(
            df_hydrogen["qm (g/s)"],
            df_hydrogen["Max T (K)"],
            label=labels[1],
            marker=markers[1],
            color=colors[1],
        )
    ymin = 20
    ymax = 1.02 * max(
        df_helium["Max T (K)"].max() if not df_helium.empty else ymin,
        df_hydrogen["Max T (K)"].max() if not df_hydrogen.empty else ymin,
    )
    plt.axhspan(ymin, cfg.MAX_TEMP_RUN, color="green", alpha=0.15)
    plt.ylim(ymin, ymax)
    plt.xticks(sorted(df["qm (g/s)"].unique()))
    plt.xlabel("Mass flow rate (g/s)")
    plt.ylabel("Max temperature (K)")
    plt.legend()
    plt.tight_layout()
    plot_path = output_dir / f"plot_peak_temperature.{cfg.PLOT_FORMAT}"
    plt.savefig(
        plot_path,
        dpi=cfg.PLOT_DPI,
        bbox_inches="tight",
        transparent=cfg.PLOT_TRANSPARENT,
    )
    plt.close()
    print(f"  [✔] 图表已保存: {plot_path}")

    # 图2：Temperature Uniformity
    plt.figure(figsize=(8, 6))
    if not df_helium.empty:
        plt.plot(
            df_helium["qm (g/s)"],
            df_helium["Max Temp Uniformity (K)"],
            label=labels[0],
            marker=markers[0],
            color=colors[0],
        )
    if not df_hydrogen.empty:
        plt.plot(
            df_hydrogen["qm (g/s)"],
            df_hydrogen["Max Temp Uniformity (K)"],
            label=labels[1],
            marker=markers[1],
            color=colors[1],
        )
    plt.xlabel("Mass flow rate (g/s)")
    plt.xticks(sorted(df["qm (g/s)"].unique()))
    plt.ylabel("Max temperature non-uniformity (K)")
    plt.legend()
    plt.tight_layout()
    plot_path = output_dir / f"plot_temperature_uniformity.{cfg.PLOT_FORMAT}"
    plt.savefig(
        plot_path,
        dpi=cfg.PLOT_DPI,
        bbox_inches="tight",
        transparent=cfg.PLOT_TRANSPARENT,
    )
    plt.close()
    print(f"  [✔] 图表已保存: {plot_path}")

    # 图3：Pressure Drop
    plt.figure(figsize=(8, 6))
    if not df_helium.empty:
        plt.plot(
            df_helium["qm (g/s)"],
            df_helium["Max Pressure Drop (kPa)"],
            label=labels[0],
            marker=markers[0],
            color=colors[0],
        )
    if not df_hydrogen.empty:
        plt.plot(
            df_hydrogen["qm (g/s)"],
            df_hydrogen["Max Pressure Drop (kPa)"],
            label=labels[1],
            marker=markers[1],
            color=colors[1],
        )
    plt.axhspan(0, 1, color="green", alpha=0.15)
    max_pd = max(
        df_helium["Max Pressure Drop (kPa)"].max()
        if not df_helium.empty
        else 1,
        df_hydrogen["Max Pressure Drop (kPa)"].max()
        if not df_hydrogen.empty
        else 1,
    )
    plt.ylim(0, 1.2 * max_pd)
    plt.xticks(sorted(df["qm (g/s)"].unique()))
    plt.xlabel("Mass flow rate (g/s)")
    plt.ylabel("Max pressure drop (kPa)")
    plt.legend()
    plt.tight_layout()
    plot_path = output_dir / f"plot_pressure_drop.{cfg.PLOT_FORMAT}"
    plt.savefig(
        plot_path,
        dpi=cfg.PLOT_DPI,
        bbox_inches="tight",
        transparent=cfg.PLOT_TRANSPARENT,
    )
    plt.close()
    print(f"  [✔] 图表已保存: {plot_path}")
    print("[✔] 运行性能图表生成完毕。")


def process_operation_data():
    """封装原 6.2_process_operation_data.py 的主流程。"""
    base_path = Path(__file__).parent.resolve()
    input_dir = base_path / cfg.OPERATION_DATA_DIR
    output_tables_dir = base_path / cfg.OPERATION_OUTPUT_DIR
    output_figures_dir = cfg.OUTPUTS_FIGURES_DIR / "operation"
    output_tables_dir.mkdir(parents=True, exist_ok=True)
    output_figures_dir.mkdir(parents=True, exist_ok=True)

    profiles_all_path = input_dir / cfg.OPERATION_PROFILE_ALL_FILE
    if not profiles_all_path.exists():
        print(f"[运行] 错误: 找不到总数据文件: {profiles_all_path}")
        return

    try:
        all_profiles_df = pd.read_excel(profiles_all_path, skiprows=4, header=0)
        all_profiles_df.columns = [col.strip() for col in all_profiles_df.columns]
    except Exception as e:
        print(f"[运行] 错误: 读取或解析总数据文件失败: {e}")
        return

    df_summary = generate_operation_summary(all_profiles_df)
    if df_summary.empty:
        return

    output_path = output_tables_dir / cfg.OPERATION_SUMMARY_FILE
    df_summary.to_excel(output_path, index=False)
    print(f"\n[✔] 运行性能概要已成功保存至: {output_path}")
    print("\n--- 运行性能概要 ---")
    print(df_summary.to_string())

    plot_operation_summary(df_summary, output_figures_dir)


# =============================================================================
# 三、失超工况（概要+曲线）：整合自 7.2_analyze_operation.py 与 7.3_analyze_quench.py
# =============================================================================
def analyze_quench_single_case(group_df: pd.DataFrame) -> dict:
    """对单个失超工况进行分析。"""
    time_col = group_df.columns[3]
    t_max_col = group_df.columns[4]
    p_max_col = group_df.columns[8]

    # 稳态温度（失超前）
    pre_quench_stable_df = group_df[
        (group_df[time_col] >= cfg.STABLE_TEMP_START_S)
        & (group_df[time_col] < cfg.QUENCH_START_S)
    ]
    stable_temp_k = (
        pre_quench_stable_df[t_max_col].mean()
        if not pre_quench_stable_df.empty
        else np.nan
    )

    # 失超期间最大温度
    quench_df = group_df[
        (group_df[time_col] >= cfg.QUENCH_START_S)
        & (group_df[time_col] <= cfg.QUENCH_END_S)
    ]
    peak_temp_k = quench_df[t_max_col].max() if not quench_df.empty else np.nan

    # 恢复时间（回到 21K）
    recovery_target_temp = 21.0
    post_quench_df = group_df[group_df[time_col] > cfg.QUENCH_END_S]
    recovered_points = post_quench_df[post_quench_df[t_max_col] <= recovery_target_temp]
    if recovered_points.empty:
        recovery_time_s = np.nan
    else:
        time_at_recovery_s = recovered_points[time_col].iloc[0]
        recovery_time_s = time_at_recovery_s - cfg.QUENCH_END_S

    # 最大压降（去除 t=0）
    filtered_df = group_df[group_df[time_col] > 0]
    max_p_Pa = filtered_df[p_max_col].max() if not filtered_df.empty else np.nan
    max_p_kPa = max_p_Pa / 1e3

    return {
        "Stable Temp (K)": stable_temp_k,
        "Max T (K)": peak_temp_k,
        "Recovery Time (s)": recovery_time_s,
        "Max Pressure Drop (kPa)": max_p_kPa - cfg.P_REF * 1e2,
    }


def generate_quench_summary(all_profiles_df: pd.DataFrame) -> pd.DataFrame:
    """生成失超工况的性能总览表。"""
    print("--- 正在从失超原始数据生成性能总览表... ---")
    summary_rows = []

    variable_cols = all_profiles_df.columns[:3]
    for params, group_df in all_profiles_df.groupby(by=list(variable_cols)):
        coolant_index, qm, pquench = params
        analysis_results = analyze_quench_single_case(group_df)

        # 注意：7.3 中 1 表示 Hydrogen，2 表示 Helium
        coolant_name = "Hydrogen" if coolant_index == 1 else "Helium"

        row_data = {
            "Coolant": coolant_name,
            "qm (g/s)": qm,
            "Pquench (W)": pquench,
        }
        row_data.update(analysis_results)
        summary_rows.append(row_data)

    summary_df = pd.DataFrame(summary_rows)
    print("  [✔] 失超性能总览表生成完毕。")
    return summary_df


def plot_quench_summary_curves(
    summary_df: pd.DataFrame, variable_col: str, fixed_col_str: str, output_dir: Path
):
    """根据失超性能总览表绘制性能曲线图。"""
    print(f"--- 正在绘制失超性能曲线图 (变量: {variable_col})... ---")

    metrics_to_plot = {
        "Max T (K)": "Max Temperature (K)",
        "Recovery Time (s)": "Recovery Time (s)",
        "Stable Temp (K)": "Stable Temperature (K)",
        "Max Pressure Drop (kPa)": "Max Pressure Drop (kPa)",
    }
    colors = {"Helium": "tab:blue", "Hydrogen": "tab:green"}

    for metric_col, y_label in metrics_to_plot.items():
        fig, ax = plt.subplots(figsize=(8, 6))

        for coolant, color in colors.items():
            coolant_data = summary_df[summary_df["Coolant"] == coolant]
            if not coolant_data.empty:
                plot_data = coolant_data.dropna(subset=[metric_col])
                ax.plot(
                    plot_data[variable_col],
                    plot_data[metric_col],
                    marker="o",
                    linestyle="-",
                    color=color,
                    label=coolant,
                )

        # 设置横轴刻度
        x_values = summary_df[variable_col].dropna().unique()
        x_values.sort()
        if len(x_values) > 1:
            interval = np.min(np.diff(x_values))
            xticks = np.arange(x_values.min(), x_values.max() + interval, interval)
            ax.set_xticks(xticks)

        if variable_col == "qm (g/s)":
            x_label = "Mass flow rate (g/s)"
        else:
            x_label = "Power of transient quench (W)"

        # 画出可行区域
        ax.autoscale(False)
        ymin = ax.get_ylim()[0]
        if metric_col == "Max Pressure Drop (kPa)":
            ax.axhspan(ymin, 1, color="green", alpha=0.15)
        elif metric_col == "Max T (K)":
            ax.axhspan(ymin, 21, color="green", alpha=0.15)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.legend(frameon=False)

        filename = (
            f"{metric_col.replace(' ', '_').replace('(', '').replace(')', '')}"
            f"_vs_{variable_col.split(' ')[0].replace('/', '')}.{cfg.PLOT_FORMAT}"
        )
        output_path = output_dir / filename
        plt.tight_layout()
        plt.savefig(
            output_path,
            bbox_inches="tight",
            format=cfg.PLOT_FORMAT,
            transparent=cfg.PLOT_TRANSPARENT,
            dpi=cfg.PLOT_DPI,
        )
        plt.close(fig)

    print(f"  [✔] {len(metrics_to_plot)} 张失超性能曲线图已保存。")


def plot_quench_recovery(all_profiles_df: pd.DataFrame, save_path: Path):
    """绘制失超后恢复过程的 T_max(t) 曲线。"""
    time_col = all_profiles_df.columns[3]
    t_max_col = all_profiles_df.columns[4]
    coolant_col = all_profiles_df.columns[0]

    plot_start_time = cfg.QUENCH_END_S - 20
    plot_end_time = cfg.QUENCH_END_S + 100

    unique_params = all_profiles_df.iloc[:, 1:3].drop_duplicates().values
    for flow, power in unique_params:
        plt.figure(figsize=(10, 6), dpi=cfg.PLOT_DPI)
        for coolant, color, label in [
            (2, "tab:blue", "Helium"),
            (1, "tab:green", "Hydrogen"),
        ]:
            mask = (
                (all_profiles_df[coolant_col] == coolant)
                & (all_profiles_df[all_profiles_df.columns[1]] == flow)
                & (all_profiles_df[all_profiles_df.columns[2]] == power)
            )
            df_plot = all_profiles_df[mask].sort_values(by=time_col)
            df_plot = df_plot[
                (df_plot[time_col] >= plot_start_time)
                & (df_plot[time_col] <= plot_end_time)
            ]
            if df_plot.empty:
                continue

            plt.plot(df_plot[time_col], df_plot[t_max_col], color=color, lw=2, label=label)

        plt.axhspan(0, 21, color="green", alpha=0.15)
        plt.xlabel("Time / s")
        plt.ylabel("Max Temperature / K")
        plt.legend()
        plt.tight_layout()

        save_dir = save_path.parent
        save_dir.mkdir(parents=True, exist_ok=True)
        fname = f"quench_recovery_flow={flow}gs-1_power={power}W.{cfg.PLOT_FORMAT}"
        full_save_path = save_dir / fname
        plt.savefig(
            full_save_path,
            dpi=cfg.PLOT_DPI,
            bbox_inches="tight",
            transparent=cfg.PLOT_TRANSPARENT,
        )
        plt.close()
        print(f"[失超] 恢复曲线已保存: {full_save_path}")


def process_quench_data():
    """封装 7.2 + 7.3 的失超分析流程（概要 + 曲线）。"""
    base_path = Path(__file__).parent.resolve()
    input_dir = base_path / cfg.QUENCH_DATA_DIR
    output_tables_dir = base_path / cfg.QUENCH_OUTPUT_DIR
    output_figures_dir = cfg.OUTPUTS_FIGURES_DIR / "quench"
    output_tables_dir.mkdir(parents=True, exist_ok=True)
    output_figures_dir.mkdir(parents=True, exist_ok=True)

    # 处理“改变 qm”的实验文件
    vary_qm_path = input_dir / cfg.QUENCH_FILE_VARY_QM
    if vary_qm_path.exists():
        try:
            df_qm = pd.read_excel(vary_qm_path, skiprows=4, header=0)
            df_qm.columns = [col.strip() for col in df_qm.columns]
        except Exception as e:
            print(f"[失超] 读取 VARY_QM 文件失败: {e}")
        else:
            summary_df = generate_quench_summary(df_qm)
            if not summary_df.empty:
                variable_col = "qm (g/s)" if len(summary_df["qm (g/s)"].unique()) > 1 else "Pquench (W)"
                fixed_vals = (
                    summary_df["Pquench (W)"].unique()
                    if variable_col == "qm (g/s)"
                    else summary_df["qm (g/s)"].unique()
                )
                fixed_col_str = (
                    f"Pquench={int(fixed_vals[0])}"
                    if variable_col == "qm (g/s)"
                    else f"qm={fixed_vals[0]}"
                )
                exp_output_dir = output_tables_dir / f"Vary_{variable_col.split(' ')[0]}_Fixed_{fixed_col_str}"
                exp_figures_dir = output_figures_dir / f"Vary_{variable_col.split(' ')[0]}_Fixed_{fixed_col_str}"
                exp_output_dir.mkdir(parents=True, exist_ok=True)
                exp_figures_dir.mkdir(parents=True, exist_ok=True)

                summary_path = exp_output_dir / "performance_summary_vary_qm.xlsx"
                summary_df.to_excel(summary_path, index=False)
                print(f"[失超] 性能总览表已保存至: {summary_path}")
                print(summary_df.to_string())

                plot_quench_summary_curves(summary_df, variable_col, fixed_col_str, exp_figures_dir)

                # 恢复曲线
                plot_quench_recovery(df_qm, exp_figures_dir / "quench_recovery_vary_qm.svg")
    else:
        print(f"[失超] 警告: 找不到 VARY_QM 文件: {vary_qm_path}")

    # 处理“改变 Pq”的实验文件
    vary_pq_path = input_dir / cfg.QUENCH_FILE_VARY_PQ
    if vary_pq_path.exists():
        try:
            df_pq = pd.read_excel(vary_pq_path, skiprows=4, header=0)
            df_pq.columns = [col.strip() for col in df_pq.columns]
        except Exception as e:
            print(f"[失超] 读取 VARY_PQ 文件失败: {e}")
        else:
            summary_df = generate_quench_summary(df_pq)
            if not summary_df.empty:
                variable_col = "qm (g/s)" if len(summary_df["qm (g/s)"].unique()) > 1 else "Pquench (W)"
                fixed_vals = (
                    summary_df["Pquench (W)"].unique()
                    if variable_col == "qm (g/s)"
                    else summary_df["qm (g/s)"].unique()
                )
                fixed_col_str = (
                    f"Pquench={int(fixed_vals[0])}"
                    if variable_col == "qm (g/s)"
                    else f"qm={fixed_vals[0]}"
                )
                exp_output_dir = output_tables_dir / f"Vary_{variable_col.split(' ')[0]}_Fixed_{fixed_col_str}"
                exp_figures_dir = output_figures_dir / f"Vary_{variable_col.split(' ')[0]}_Fixed_{fixed_col_str}"
                exp_output_dir.mkdir(parents=True, exist_ok=True)
                exp_figures_dir.mkdir(parents=True, exist_ok=True)

                summary_path = exp_output_dir / "performance_summary_vary_pq.xlsx"
                summary_df.to_excel(summary_path, index=False)
                print(f"[失超] 性能总览表已保存至: {summary_path}")
                print(summary_df.to_string())

                plot_quench_summary_curves(summary_df, variable_col, fixed_col_str, exp_figures_dir)

                # 恢复曲线
                plot_quench_recovery(df_pq, exp_figures_dir / "quench_recovery_vary_pq.svg")
    else:
        print(f"[失超] 警告: 找不到 VARY_PQ 文件: {vary_pq_path}")

    print(f"\n[✔] 所有失超分析任务完成。结果保存在 '{output_tables_dir}' 的子文件夹中。")


# =============================================================================
# 总入口
# =============================================================================
def main(
    run_cooldown: bool = True,
    run_charge: bool = True,
    run_operation: bool = True,
    run_quench: bool = True,
):
    """
    统一入口：
      - run_charge: 是否处理充电数据
      - run_operation: 是否处理稳态运行数据
      - run_quench: 是否处理失超数据
    """
    setup_plot_style()

    if run_cooldown:
        print("\n" + "=" * 60)
        print(">>> 开始处理【降温】数据")
        print("=" * 60)
        process_cooldown_data()

    if run_charge:
        print("\n" + "=" * 60)
        print(">>> 开始处理【充电】数据")
        print("=" * 60)
        process_charge_data()

    if run_operation:
        print("\n" + "=" * 60)
        print(">>> 开始处理【稳态运行】数据")
        print("=" * 60)
        process_operation_data()

    if run_quench:
        print("\n" + "=" * 60)
        print(">>> 开始处理【失超】数据")
        print("=" * 60)
        process_quench_data()


if __name__ == "__main__":
    # 默认全部执行；如需只跑其中一类，可在此修改参数
    main(run_cooldown=True, run_charge=True, run_operation=True, run_quench=True)


