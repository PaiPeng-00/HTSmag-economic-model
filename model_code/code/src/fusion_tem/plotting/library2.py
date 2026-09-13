# plot_library.py (统一绘图库 - 英文版)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import matplotlib as mpl
from pathlib import Path
from fusion_tem import device as cfg
from typing import Optional, Tuple, List
from scipy.interpolate import RegularGridInterpolator

# =============================================================================
# 绘图函数 1: 组合图 (成本构成 + 回收周期)
# =============================================================================
font_size = 20
font_size_title = 24
font_size_label = 20
font_size_tick = 20
font_size_legend = 20
font_size_legend_title = 20
font_size_legend_label = 20
font_size_legend_tick = 20

# =============================================================================
# 绘图函数 2: 厂用电热力图
# =============================================================================

def plot_parasitic_heatmap_single(
    df: pd.DataFrame,
    output_dir: Path,
    cmap: str,
    norm: mpl.colors.Normalize,
    scenario: str,
    temperature_K: float,
    draw_contours: bool = True,
    verbose: bool = True,
    filename_suffix: str = "",  # 可选的文件名后缀，用于区分不同配色方案
):
    """
    为单个场景和单个温度生成并保存一个热力图。
    主图不包含colorbar、坐标轴标签和刻度值。
    
    参数:
        df: 包含单个温度数据的DataFrame
        output_dir: 输出目录
        cmap: 颜色映射
        norm: 全局归一化对象（确保所有图使用同一个colorbar范围）
        scenario: 场景名称
        temperature_K: 温度值
    """
    font_size = 18
    setup_plot_style()
    
    label = f"{temperature_K}K"
    
    # 创建单个子图
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    
    # 准备数据：使用原始数据创建 pivot（基于 R_JOINT_SCAN_VALUES）
    pivot = df.pivot(index="Npw", columns="R_joint", values="r_parasitic_pct")
    
    # 获取原始网格（用于插值）
    R_joint_original = pivot.columns.values  # 原始 R_joint 值（Ω单位）
    Npw_original = pivot.index.values  # 原始 Npw 值
    Z_original = pivot.values  # 原始 Z 值
    
    # 使用对数数组创建绘图网格（匹配对数刻度）
    # 配置中的 R_JOINT_SCAN_VALUES 以 nOhm 为单位，这里转换为 Ohm 参与插值和绘图
    R_joint_plot = np.array(cfg.R_JOINT_SCAN_VALUES, dtype=float) * 1e-9
    Npw_plot = Npw_original  # Npw 保持不变
    
    # 创建插值器（使用原始网格）
    interp = RegularGridInterpolator(
        (Npw_original, R_joint_original),
        Z_original,
        bounds_error=False,
        fill_value=np.nan
    )
    
    # 创建绘图网格
    X_plot, Y_plot = np.meshgrid(R_joint_plot, Npw_plot, indexing='xy')
    
    # 插值到绘图网格
    Z_plot = interp((Y_plot, X_plot))
    
    # 使用对数网格绘图
    pcm = ax.pcolormesh(X_plot, Y_plot, Z_plot, cmap=cmap, norm=norm, shading="auto")
    ax.set_xscale('log')  # 设置横轴为对数刻度
    
    # 为了兼容后续代码，保留原始变量名（但使用插值后的值）
    X = X_plot
    Y = Y_plot
    Z = Z_plot
    
    clabels = []

    if draw_contours:

        levels = {'4.2K': [7, 8, 10, 15, 20, 50], '10.0K': [3, 4, 5, 6, 8, 10], '20.0K': [1.5, 2, 2.5, 3, 4, 5]}

        contour_levels = levels.get(label, [])
        
        # 限制等高线数量最多为5条
        if len(contour_levels) > 5:
            # 均匀选择5条，保留最小值和最大值
            if len(contour_levels) > 2:
                indices = np.linspace(0, len(contour_levels) - 1, 5, dtype=int)
                contour_levels = [contour_levels[i] for i in indices]
            else:
                contour_levels = contour_levels[:5]

        if contour_levels:
            # 先临时绘制所有等高线以获取路径信息，用于确定颜色
            temp_CS = ax.contour(X, Y, Z, levels=contour_levels, colors="black", linewidths=0.01)
            # 获取每条等高线的颜色（基于背景色亮度）
            level_colors = get_contour_colors_by_background(temp_CS, X, Y, Z, cmap, norm)
            # 清除临时等高线
            for collection in temp_CS.collections:
                try:
                    collection.remove()
                except:
                    pass
            
            def _pick_safe_label_positions_from_contour(
                axis: plt.Axes,
                contour_set,
                margin_axes: float = 0.08,
            ) -> List[Tuple[float, float]]:
                """
                从 contour_set 的路径中选择一个尽量远离边缘的点，用于手动放置标签。
                返回点列表（可能为空）。
                """
                try:
                    # contour_set.collections[0] 对应该 level 的 LineCollection
                    if not hasattr(contour_set, "collections") or len(contour_set.collections) == 0:
                        return []
                    col = contour_set.collections[0]
                    if not hasattr(col, "get_paths"):
                        return []
                    paths = col.get_paths()
                    if not paths:
                        return []

                    best = None
                    best_score = -1.0
                    for p in paths:
                        v = p.vertices
                        if v is None or len(v) < 5:
                            continue
                        # 抽样若干点，挑一个离边缘最远的
                        idxs = np.linspace(0, len(v) - 1, min(30, len(v)), dtype=int)
                        for i in idxs:
                            xdata, ydata = float(v[i, 0]), float(v[i, 1])
                            x_disp, y_disp = axis.transData.transform((xdata, ydata))
                            x_axes, y_axes = axis.transAxes.inverted().transform((x_disp, y_disp))
                            if (
                                x_axes < margin_axes
                                or x_axes > 1.0 - margin_axes
                                or y_axes < margin_axes
                                or y_axes > 1.0 - margin_axes
                            ):
                                continue
                            # score：距离四边的最小值，越大越好
                            score = min(x_axes, 1.0 - x_axes, y_axes, 1.0 - y_axes)
                            if score > best_score:
                                best_score = score
                                best = (xdata, ydata)
                    return [best] if best is not None else []
                except Exception:
                    return []

            # 按颜色逐条绘制等高线（并保证每条至少有一个可见标签）
            all_clabels = []
            for level in contour_levels:
                color = level_colors.get(level, 'black')  # 默认黑色
                CS_level = ax.contour(X, Y, Z, levels=[level], colors=color, linewidths=2)
                labels_level = ax.clabel(CS_level, inline=True, fontsize=font_size, fmt="%.1f", colors=color)

                # 若 matplotlib 自动放置失败（或后续可能被隐藏），尝试手动补一个“安全位置”的标签
                if not labels_level:
                    manual_pos = _pick_safe_label_positions_from_contour(ax, CS_level, margin_axes=0.08)
                    if manual_pos:
                        labels_level = ax.clabel(
                            CS_level,
                            inline=True,
                            fontsize=font_size,
                            fmt="%.1f",
                            colors=color,
                            manual=manual_pos,
                        )
                if labels_level:
                    all_clabels.extend(labels_level)
            
            clabels = all_clabels




    # 隐藏靠近坐标轴/边框的等高线标签，避免数值与轴线/边框交叉
    def _hide_labels_near_axes(texts, axis, margin_axes: float = 0.03, margin_data_y: float = 3.0):
        """
        将靠近坐标轴/边框的等高线标签隐藏，防止数字与坐标轴或边框交叉。
        
        参数:
            texts: clabel返回的文本对象列表
            axis: matplotlib axes对象
            margin_axes: 以坐标轴归一化坐标为单位的边缘留白（0-1），默认3%
            margin_data_y: 以数据坐标为单位的y轴下边缘留白（用于横轴检测），默认3.0
        """
        if not texts:
            return
        
        # 获取当前坐标轴的数据范围
        xlim = axis.get_xlim()
        ylim = axis.get_ylim()
        
        for txt in texts:
            xdata, ydata = txt.get_position()
            
            # 方法1：使用axes坐标检测（适用于所有边缘）
            x_disp, y_disp = axis.transData.transform((xdata, ydata))
            x_axes, y_axes = axis.transAxes.inverted().transform((x_disp, y_disp))
            
            # 方法2：使用数据坐标检测（特别针对横轴，因为y轴下限是2）
            # 如果标签的y坐标接近y轴下限，直接隐藏
            near_bottom = ydata < (ylim[0] + margin_data_y)
            near_top = ydata > (ylim[1] - margin_data_y)
            near_left = xdata < (xlim[0] + (xlim[1] - xlim[0]) * margin_axes)
            near_right = xdata > (xlim[1] - (xlim[1] - xlim[0]) * margin_axes)
            
            # 使用axes坐标检测（适用于所有边缘，更严格）
            near_edge_axes = (
                x_axes < margin_axes
                or x_axes > 1.0 - margin_axes
                or y_axes < margin_axes
                or y_axes > 1.0 - margin_axes
            )
            
            # 如果标签靠近任何边缘，隐藏它
            if near_bottom or near_top or near_left or near_right or near_edge_axes:
                txt.set_visible(False)
    
    # 应用标签隐藏函数
    if clabels:
        _hide_labels_near_axes(clabels, ax)

    # 设置坐标轴范围但不显示刻度和标签（对数轴，单位：Ohm，对应1nOhm到100nOhm）
    ax.set_xlim(1e-9, 100e-9)  # 1nOhm到100nOhm，单位转换为Ohm
    ax.set_ylim(2, 200)
    
    # 移除所有刻度值和标签
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # 移除坐标轴标签
    ax.set_xlabel("")
    ax.set_ylabel("")
    
    # 保留标题（包含场景和温度信息）
    # ax.set_title(f"{scenario} - {label}")

    # 调整布局
    fig.subplots_adjust(bottom=0.1, right=0.95, left=0.05, top=0.9)
    
    # 保存单个子图（不包含colorbar、坐标轴标签和刻度）
    suffix_str = f"_{filename_suffix}" if filename_suffix else ""
    plot_path = output_dir / f"parasitic_ratio_heatmap_{scenario}_{label}{suffix_str}.svg"
    plt.savefig(plot_path, dpi=cfg.PLOT_DPI, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    
    # 创建readme.txt文件，记录绘图信息
    readme_path = output_dir / "readme.txt"
    with open(readme_path, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"图片文件: {plot_path.name}\n")
        f.write(f"场景: {scenario}\n")
        f.write(f"温度: {temperature_K}K\n")
        f.write(f"\n坐标轴数据安排:\n")
        f.write(f"  横轴(X轴): R_joint (接头电阻), 单位: Ohm\n")
        f.write(f"    - 绘图数据范围: {X.min():.6e} ~ {X.max():.6e} Ohm\n")
        f.write(f"    - 显示方式: 对数刻度 (log scale)\n")
        f.write(f"    - 显示范围: 1e-9 ~ 100e-9 Ohm, 对应 1nOhm ~ 100nOhm\n")
        # 子图输出用于后续SVG拼接：子图本身会隐藏刻度；这里记录推荐显示刻度
        f.write(f"    - 刻度(推荐, nΩ): 1, 10, 100\n")
        f.write(f"    - 绘图网格: 使用对数均匀分布的数组 (R_JOINT_SCAN_VALUES_PLOT)\n")
        f.write(f"    - 计算网格: 使用原始整数数组 (R_JOINT_SCAN_VALUES)\n")
        f.write(f"  纵轴(Y轴): Npw (并联绕组数)\n")
        f.write(f"    - 数据范围: {Y.min():.1f} ~ {Y.max():.1f}\n")
        f.write(f"    - 显示范围: 2 ~ 200\n")
        try:
            y_ticks = getattr(cfg, "NPW_SCAN_VALUES", None)
            if y_ticks is not None:
                y_ticks_list = [int(v) for v in np.array(y_ticks).astype(float).tolist()]
                f.write(f"    - 刻度(推荐): {', '.join(map(str, y_ticks_list))}\n")
        except Exception:
            pass
        f.write(f"\n数据值列: r_parasitic_pct (寄生功率百分比)\n")
        if clabels:
            f.write(f"等高线级别: {', '.join([f'{l:.1f}' for l in contour_levels]) if 'contour_levels' in locals() else 'N/A'}\n")
        f.write(f"{'='*60}\n")



def save_parasitic_heatmap_colorbar(output_dir: Path, cmap: str, norm: mpl.colors.Normalize, filename_suffix: str = ""):
    """
    单独生成并保存colorbar（所有子图共享）。
    
    参数:
        output_dir: 输出目录
        cmap: 颜色映射
        norm: 归一化对象
        filename_suffix: 可选的文件名后缀，用于区分不同配色方案
    """
    setup_plot_style()
    
    # 创建一个新的ScalarMappable对象用于colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # 设置空数组
    
    fig_cbar = plt.figure(figsize=(0.5, 6))
    ax_cbar = fig_cbar.add_axes([0.1, 0.15, 0.3, 0.7])
    cbar = fig_cbar.colorbar(sm, cax=ax_cbar, label="Parasitic Power Fraction (%)")
    
    # 自定义 colorbar 刻度
    cbar.set_ticks([2, 4, 10, 20, 50])
    cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label("Parasitic Power Fraction (%)", fontsize=18, labelpad=15)
    
    suffix_str = f"_{filename_suffix}" if filename_suffix else ""
    cbar_path = output_dir / f"parasitic_ratio_heatmap_colorbar{suffix_str}.{cfg.PLOT_FORMAT}"
    plt.savefig(cbar_path, dpi=cfg.PLOT_DPI, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig_cbar)


def plot_cryo_mode_grid(
    df: pd.DataFrame,
    scenarios: list,
    temperatures: list,
    save_path: Path,
    cmap: str,
    norm: mpl.colors.Normalize,
    mode_label: str
):
    """
    为指定模式绘制一个 3x3 的子图矩阵，行对应情景、列对应温度。
    颜色表示制冷系统电功率。
    """
    setup_plot_style()
    scenarios = list(scenarios)
    temperatures = list(temperatures)
    n_rows = len(scenarios)
    n_cols = len(temperatures)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 5.0, n_rows * 5.0),
        sharex=True,
        sharey=True
    )
    axes = np.atleast_2d(axes)
    pcm = None
    subplot_records = []
    xlim_range = (np.log10(1), np.log10(100))
    ylim_range = (2, 200)

    for i, scenario in enumerate(scenarios):
        for j, temp in enumerate(temperatures):
            ax = axes[i, j]
            subset = df[
                (df["scenario"] == scenario) &
                (df["temperature_K"] == temp)
            ]
            if subset.empty:
                ax.axis("off")
                continue

            subset = subset.copy()
            subset["cryo_power_MW"] = subset["cryo_power_W"] / 1e6

            pivot = subset.pivot_table(
                index="Npw",
                columns="R_joint",
                values="cryo_power_MW",
                aggfunc="mean"
            ).sort_index().sort_index(axis=1)

            X = np.log10(pivot.columns.to_numpy() * 1e9)
            Y = pivot.index.to_numpy()
            Z = pivot.to_numpy()

            pcm = ax.pcolormesh(
                X, Y, Z,
                cmap=cmap,
                norm=norm,
                shading="auto"
            )

            z_min = np.nanmin(Z)
            z_max = np.nanmax(Z)
            contour_levels = None
            contour_num = 7
            if np.isfinite(z_min) and np.isfinite(z_max) and z_max > z_min:
                finite_positive = Z[np.isfinite(Z) & (Z > 0)]
                if finite_positive.size > 0:
                    z_min_pos = finite_positive.min()
                    z_max_pos = finite_positive.max()
                    if z_max_pos > z_min_pos:
                        if z_max_pos / z_min_pos >= 10:
                            contour_levels = np.geomspace(z_min_pos, z_max_pos, contour_num)
                        else:
                            contour_levels = np.linspace(z_min_pos, z_max_pos, contour_num)
                    else:
                        contour_levels = np.linspace(z_min, z_max, contour_num)
                else:
                    contour_levels = np.linspace(z_min, z_max, contour_num)
                XX, YY = np.meshgrid(X, Y)
                CS = ax.contour(
                    XX, YY, Z,
                    levels=contour_levels,
                    colors="black",
                    linewidths=1.2
                )
                ax.clabel(CS, inline=True, fontsize=font_size_tick, fmt="%.2f", colors="black")

            ax.set_xlim(xlim_range)
            ax.set_ylim(ylim_range)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("")
            ax.set_ylabel("")

            subplot_records.append({
                "scenario": scenario,
                "temperature": temp,
                "X": X,
                "Y": Y,
                "Z": Z,
                "contour_levels": contour_levels
            })

    fig.suptitle(mode_label, fontsize=32, y=0.99)
    
    if pcm is not None:
        # 当有 colorbar 时，使用 subplots_adjust 而不是 tight_layout
        # 因为手动添加的 add_axes 与 tight_layout 不兼容
        fig.subplots_adjust(right=0.88, top=0.96)
        cax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(pcm, cax=cax)
        cbar.ax.tick_params(labelsize=20)
        cbar.set_label("Cryogenic Power (MW)", fontsize=24)
    else:
        # 没有 colorbar 时，使用 tight_layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=cfg.PLOT_DPI)
    plt.close(fig)
    print(f"Cryogenic power grid saved to: {save_path}")

    # 保存单独的子图
    individual_dir = save_path.parent / f"{save_path.stem}_subplots"
    individual_dir.mkdir(parents=True, exist_ok=True)
    for record in subplot_records:
        fig_single, ax_single = plt.subplots(figsize=(5, 5))
        pcm_single = ax_single.pcolormesh(
            record["X"], record["Y"], record["Z"],
            cmap=cmap,
            norm=norm,
            shading="auto"
        )
        if record["contour_levels"] is not None:
            XX, YY = np.meshgrid(record["X"], record["Y"])
            CS_single = ax_single.contour(
                XX, YY, record["Z"],
                levels=record["contour_levels"],
                colors="black",
                linewidths=1.2
            )
            ax_single.clabel(CS_single, inline=True, fontsize=font_size_tick, fmt="%.2f", colors="black")

        ax_single.set_xlim(xlim_range)
        ax_single.set_ylim(ylim_range)
        ax_single.set_xticks([])
        ax_single.set_xticklabels([])
        ax_single.set_xlabel("")
        ax_single.set_yticks([])
        ax_single.set_ylabel("")
        for spine in ax_single.spines.values():
            spine.set_visible(True)

        single_path = individual_dir / f"{record['scenario']}_{record['temperature']:.1f}K.{cfg.PLOT_FORMAT}"
        fig_single.tight_layout()
        fig_single.savefig(single_path, dpi=cfg.PLOT_DPI)
        plt.close(fig_single)
        print(f"  └─ Saved subplot: {single_path}")

    # 保存共享 colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    colorbar_fig = plt.figure(figsize=(0.8, 4))
    colorbar_ax = colorbar_fig.add_axes([0.3, 0.05, 0.4, 0.9])
    colorbar = plt.colorbar(sm, cax=colorbar_ax)
    colorbar.set_label("Cryogenic Power (MW)")
    colorbar_path = save_path.parent / f"{save_path.stem}_colorbar.{cfg.PLOT_FORMAT}"
    colorbar_fig.savefig(colorbar_path, dpi=cfg.PLOT_DPI, bbox_inches='tight')
    plt.close(colorbar_fig)

# =============================================================================
# 辅助绘图函数 (图例、寄生功率vs温度等)
# =============================================================================

def setup_plot_style():
    """设置全局Matplotlib绘图样式"""
    plt.style.use('default')
    # 设置字体：优先使用 Arial（英文），如果遇到中文则回退到支持中文的字体
    # Windows 系统常见的中文字体：Microsoft YaHei, SimHei, SimSun
    # macOS 系统常见的中文字体：PingFang SC, STHeiti
    # Linux 系统常见的中文字体：WenQuanYi Micro Hei, Noto Sans CJK SC
    import platform
    system = platform.system()
    if system == 'Windows':
        # Windows 系统：优先 Arial，回退到 Microsoft YaHei（微软雅黑）
        plt.rcParams['font.sans-serif'] = ['Arial', 'Microsoft YaHei', 'SimHei', 'SimSun', 'DejaVu Sans']
    elif system == 'Darwin':  # macOS
        # macOS 系统：优先 Arial，回退到 PingFang SC
        plt.rcParams['font.sans-serif'] = ['Arial', 'PingFang SC', 'STHeiti', 'DejaVu Sans']
    else:  # Linux
        # Linux 系统：优先 Arial，回退到常见中文字体
        plt.rcParams['font.sans-serif'] = ['Arial', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
    # 设置默认字体族（如果系统支持 Arial，优先使用）
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.linewidth'] = 2
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20
    plt.rcParams['legend.fontsize'] = 20
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['axes.grid'] = False
    plt.rcParams['savefig.transparent'] = True
    plt.rcParams['legend.frameon'] = False

def plot_parasitic_power_vs_temp(excel_path: str, save_path: str):
    """绘制寄生功耗与温度的关系图。"""
    setup_plot_style()
    try:
        df = pd.read_excel(excel_path)
    except FileNotFoundError:
        print(f"Error: File not found at '{excel_path}'.")
        return
    power_stats = df.groupby('temperature_K')['r_parasitic_pct'].agg(['mean', 'std']).reset_index()
    fig, ax = plt.subplots(figsize=(10, 8))
    unique_temps = power_stats['temperature_K'].unique()
    colors = ['#386192', '#00BFFF', '#ACD8E5']
    ax.bar(power_stats['temperature_K'], power_stats['mean'], color=colors, width=0.8, edgecolor='black')
    ax.set_xlabel('Operating Temperature (K)', labelpad=15)
    ax.set_ylabel('Cryo-System Parasitic Power (%)', labelpad=15)
    ax.set_title('Parasitic Power vs. Operating Temperature', fontsize=22, y=1.05)
    ax.set_xticks(power_stats['temperature_K'])
    plt.tight_layout()
    plt.savefig(save_path, dpi=cfg.PLOT_DPI)
    plt.close(fig)
    print(f"Parasitic power chart saved to: {save_path}")

def save_color_legend(save_path: str):
    """生成并保存颜色图例"""
    setup_plot_style()
    color_map = {
        (4.2, 'He'): '#386192', (10, 'He'): '#00BFFF',
        (20, 'He'): '#ACD8E5', (20, 'H2'): '#A3D99F',
    }
    def sort_coolant_temp(combo_values):
        return (0 if combo_values[1] == 'He' else 1, combo_values[0])
    all_combos = sorted(color_map.keys(), key=sort_coolant_temp)
    handles = [mpatches.Patch(color=color_map[combo], label=f'{combo[0]}K - {combo[1]}') for combo in all_combos]
    fig = plt.figure(figsize=(10, 1.5))
    ax = fig.add_subplot(111)
    ax.axis('off')
    legend = ax.legend(
        handles=handles,
        loc='center',
        ncol=len(handles),
        frameon=False,
        fontsize=18,
        title='Operating Case (Temperature - Coolant)'
    )
    legend.get_frame().set_facecolor('none')
    legend.get_frame().set_edgecolor('none')
    legend.get_title().set_fontsize(18)
    plt.tight_layout()
    plt.savefig(save_path, dpi=cfg.PLOT_DPI, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"Color legend saved to: {save_path}")

def auto_contour_levels(
    Z_sub: np.ndarray,
    norm: mpl.colors.Normalize,
    n_min: int = 4,
    n_max: int = 5,
    q: float = 0.05,
    margin: float = 0.03,
    include_zero: bool = True,
    verbose: bool = False
) -> np.ndarray:
    """
    为单个子图自动生成4-6条等高线，在视觉空间（norm空间）中均匀分布。
    
    参数:
        Z_sub: 子图数据数组（2D）
        norm: Matplotlib归一化对象（必须支持inverse方法，如SymLogNorm、LogNorm）
        n_min: 最小等高线数量（默认4）
        n_max: 最大等高线数量（默认5）
        q: 分位数范围，用于避免极值挤压（默认0.05，即使用5%-95%分位数）
        margin: 视觉空间中的边距比例（默认0.03，即3%）
        include_zero: 如果数据跨0，是否将最接近0的等高线替换为0.0（默认True）
        verbose: 是否输出详细信息（默认False）
    
    返回:
        等高线级别数组（已排序，不包含zmin/zmax）
    """
    # 展平并去除NaN/inf
    z_flat = Z_sub.flatten()
    z_valid = z_flat[np.isfinite(z_flat)]
    
    # 如果有效点太少，返回空数组
    if len(z_valid) < 10:
        return np.array([])
    
    z_min = float(np.nanmin(z_valid))
    z_max = float(np.nanmax(z_valid))
    
    # 如果数据范围无效，返回空数组
    if not np.isfinite(z_min) or not np.isfinite(z_max) or z_max <= z_min:
        return np.array([])
    
    # 将数据转换到norm空间（视觉空间）
    try:
        # 获取norm的vmin和vmax
        vmin = norm.vmin if hasattr(norm, 'vmin') and np.isfinite(norm.vmin) else z_min
        vmax = norm.vmax if hasattr(norm, 'vmax') and np.isfinite(norm.vmax) else z_max
        
        # 将有效数据clip到norm范围，然后转换到视觉空间
        z_clipped = np.clip(z_valid, vmin, vmax)
        t_values = norm(z_clipped)
        # 确保是普通数组，不是 MaskedArray
        if isinstance(t_values, np.ma.MaskedArray):
            t_values = np.array(t_values.data[~t_values.mask] if t_values.mask is not np.ma.nomask else t_values.data)
        else:
            t_values = np.array(t_values)
    except Exception:
        # 如果norm转换失败，使用线性映射作为备用
        t_values = (z_valid - z_min) / (z_max - z_min) if (z_max > z_min) else np.zeros_like(z_valid)
        t_values = np.array(t_values)
    
    # 使用分位数范围避免极值挤压
    t_lo, t_hi = np.quantile(t_values, [q, 1 - q])
    
    # 如果分位数范围太小，使用全范围
    if t_hi <= t_lo:
        t_lo = np.min(t_values)
        t_hi = np.max(t_values)
    
    # 计算视觉空间的范围
    span = t_hi - t_lo
    if span <= 0:
        return np.array([])
    
    # 决定等高线数量（根据视觉空间范围自适应，但限制在n_min到n_max之间）
    n = int(np.clip(np.round(n_min + (n_max - n_min) * min(span / 0.5, 1.0)), n_min, n_max))
    
    # 在视觉空间中留边距并取等距点（不包含端点）
    t_levels = np.linspace(t_lo + margin * span, t_hi - margin * span, n)

    # 逆变换得到 raw levels
    try:
        if hasattr(norm, "inverse"):
            levels_raw = norm.inverse(t_levels)
        else:
            levels_raw = t_levels * (z_max - z_min) + z_min
    except Exception:
        levels_raw = t_levels * (z_max - z_min) + z_min

    print(f"levels_raw: {levels_raw}")
    print(f"z_min: {z_min}, z_max: {z_max}")
    print(f"n_min: {n_min}, n_max: {n_max}")
    print(f"q: {q}, margin: {margin}")
    print(f"include_zero: {include_zero}")
    print(f"verbose: {verbose}")
    # 主路径：raw → snap_levels_to_nice（内部再从整齐候选集中选 n 条）
    levels = snap_levels_to_nice(
        levels=levels_raw,
        z_min=z_min,
        z_max=z_max,
        n_min=n_min,
        prefer_integers=True,
        norm=norm,
        verbose=verbose,
    )
    # 输出z_max对应的参数组合
    Npw = np.array(cfg.NPW_SCAN_VALUES, dtype=int)
    R_joint_nOhm = np.array(cfg.R_JOINT_SCAN_VALUES, dtype=float)
    z_max_indices = np.argwhere(Z_sub == z_max)
    for idx in z_max_indices:
        print(f"z_max: {z_max}, levels: {levels}, z_max参数组合 (index): {tuple(idx)}", f"Npw: {Npw[idx[0]]}, R_joint_nOhm: {R_joint_nOhm[idx[1]]}")
    return levels


def pick_nice_levels_by_norm_targets(
    Z_sub: np.ndarray,
    norm: mpl.colors.Normalize,
    n_min: int = 5,
    n_max: int = 5,
    q: float = 0.05,
    margin: float = 0.03,
    include_zero: bool = True,
    verbose: bool = False  # 是否输出详细信息（默认False，只有"相对于最小值"才输出）
) -> np.ndarray:
    """
    在视觉空间（norm空间）中均匀选择目标位置，然后从多尺度整齐候选值中选择最接近的。
    
    参数:
        Z_sub: 子图数据数组（2D）
        norm: Matplotlib归一化对象（必须支持inverse方法，如SymLogNorm、LogNorm）
        n_min: 最小等高线数量（默认4）
        n_max: 最大等高线数量（默认5）
        q: 分位数范围，用于避免极值挤压（默认0.05，即使用5%-95%分位数）
        margin: 视觉空间中的边距比例（默认0.03，即3%）
        include_zero: 如果数据跨0，是否包含0.0（默认True）
    
    返回:
        等高线级别数组（已排序，不包含zmin/zmax，都是整齐数）
    """
    # 展平并去除NaN/inf
    z_flat = Z_sub.flatten()
    z_valid = z_flat[np.isfinite(z_flat)]
    
    # 如果有效点太少，返回空数组
    if len(z_valid) < 10:
        return np.array([])
    
    z_min = float(np.nanmin(z_valid))
    z_max = float(np.nanmax(z_valid))
    
    # 如果数据范围无效，返回空数组
    if not np.isfinite(z_min) or not np.isfinite(z_max) or z_max <= z_min:
        return np.array([])
    
    # 将数据转换到norm空间（视觉空间）
    try:
        # 获取norm的vmin和vmax
        vmin = norm.vmin if hasattr(norm, 'vmin') and np.isfinite(norm.vmin) else z_min
        vmax = norm.vmax if hasattr(norm, 'vmax') and np.isfinite(norm.vmax) else z_max
        
        # 将有效数据clip到norm范围，然后转换到视觉空间
        z_clipped = np.clip(z_valid, vmin, vmax)
        t_values = norm(z_clipped)
        # 确保是普通数组，不是 MaskedArray
        if isinstance(t_values, np.ma.MaskedArray):
            t_values = np.array(t_values.data[~t_values.mask] if t_values.mask is not np.ma.nomask else t_values.data)
        else:
            t_values = np.array(t_values)
    except Exception:
        # 如果norm转换失败，使用线性映射作为备用
        t_values = (z_valid - z_min) / (z_max - z_min) if (z_max > z_min) else np.zeros_like(z_valid)
        t_values = np.array(t_values)
    
    # 使用分位数范围避免极值挤压
    t_lo, t_hi = np.quantile(t_values, [q, 1 - q])
    
    # 如果分位数范围太小，使用全范围
    if t_hi <= t_lo:
        t_lo = np.min(t_values)
        t_hi = np.max(t_values)
    
    # 计算视觉空间的范围
    span = t_hi - t_lo
    if span <= 0:
        return np.array([])
    
    # 决定等高线数量（根据视觉空间范围自适应，但限制在n_min到n_max之间）
    n = int(np.clip(np.round(n_min + (n_max - n_min) * min(span / 0.5, 1.0)), n_min, n_max))

    
    # 在视觉空间中留边距并取等距点（不包含端点）- 这些是目标位置
    t_targets = np.linspace(t_lo + margin * span, t_hi - margin * span, n)
    
    # 生成多尺度的整齐候选值
    # 使用 {1, 2, 5} × 10^k 的多个 step，覆盖多个数量级
    data_range = z_max - z_min
    if data_range <= 0:
        return np.array([])
    
    # 分别确定 z_min 和 z_max 的数量级范围，确保覆盖整个数据范围
    # 对于 z_min 和 z_max，分别计算它们的数量级
    if abs(z_min) > 1e-10:
        k_min = int(np.floor(np.log10(abs(z_min))))
    else:
        k_min = int(np.floor(np.log10(max(abs(z_max), 1e-10)))) - 3  # 如果 z_min 很小，向下扩展3个数量级
    
    if abs(z_max) > 1e-10:
        k_max = int(np.ceil(np.log10(abs(z_max))))
    else:
        k_max = int(np.ceil(np.log10(max(abs(z_min), 1e-10)))) + 3  # 如果 z_max 很小，向上扩展3个数量级
    
    # 确保覆盖范围足够宽，至少覆盖从 z_min 到 z_max 的数量级
    k_min = min(k_min, int(np.floor(np.log10(max(abs(z_min), 1e-10)))))
    k_max = max(k_max, int(np.ceil(np.log10(max(abs(z_max), 1e-10)))))
    
    # 限制 k_min 不能太小，避免生成过小的 step 导致内存溢出
    # 根据数据范围动态限制：如果数据范围很大，k_min 不能太小
    # 最小 step 应该使得从 z_min 到 z_max 的候选值数量不超过 MAX_CANDIDATES_PER_STEP
    MAX_CANDIDATES_PER_STEP = 1000  # 每个 step 最多生成一定数量的候选值，避免内存溢出
    if data_range > 0:
        # 计算允许的最小 step：data_range / MAX_CANDIDATES_PER_STEP
        min_step_allowed = data_range / MAX_CANDIDATES_PER_STEP
        k_min_allowed = int(np.floor(np.log10(min_step_allowed)))
        # k_min 不能小于 k_min_allowed（即 step 不能小于 min_step_allowed）
        k_min = max(k_min, k_min_allowed)
        # 同时，k_min 不能小于 -6（即 step 不能小于 1e-6），作为硬性限制
        k_min = max(k_min, -6)
    
    # 生成候选值：使用多个 step（1, 2, 5）和多个数量级
    candidates = []
    
    # 从最小数量级到最大数量级，每个数量级使用 1, 2, 5 作为步长
    for k in range(k_min, k_max + 1):
        base = 10 ** k
        for multiplier in [1, 2, 5]:
            step = multiplier * base
            # 生成该 step 下的候选值（覆盖 z_min 到 z_max）
            grid_start = np.floor(z_min / step) * step
            grid_end = np.ceil(z_max / step) * step
            
            # 估算需要生成的候选值数量
            if step > 0:
                num_points = int((grid_end - grid_start) / step) + 1
            else:
                num_points = 0
            
            # 如果候选值数量过多，跳过这个 step（避免内存溢出）
            if num_points > MAX_CANDIDATES_PER_STEP:
                continue
            # 生成网格点
            if num_points > 0:
                grid_points = np.arange(grid_start, grid_end + step, step)
                candidates.extend(grid_points.tolist())
    
    # 如果 include_zero 且数据跨0，确保包含0
    if include_zero and z_min < 0 < z_max:
        if 0.0 not in candidates:
            candidates.append(0.0)
    
    # 如果数据范围较小（最大值和最小值之差在 5 以内），
    # 在整个数据范围内启用 0.5 间隔的候选等高线值，以获得更细粒度的等高线
    if data_range <= 5.0:
        half_step_candidates = []
        # 在 [z_min, z_max] 上以 0.5 为步长生成候选值
        start_i = int(np.floor(z_min * 2))
        end_i = int(np.ceil(z_max * 2))
        for i in range(start_i, end_i + 1):
            candidate = i * 0.5
            if z_min <= candidate <= z_max:
                half_step_candidates.append(candidate)
        candidates.extend(half_step_candidates)
    
    # 去重并排序
    candidates = np.unique(np.array(candidates))
    candidates = np.sort(candidates)
    
    # 过滤候选值：只保留在数据范围内的（留边距）
    eps = 1e-9 * data_range if data_range > 0 else 1e-9
    candidates = candidates[(candidates >= z_min + eps) & (candidates <= z_max - eps)]
    
    if len(candidates) == 0:
        return np.array([])
    
    # 将候选值映射到视觉空间
    try:
        if hasattr(norm, 'inverse'):
            # 对于候选值，我们需要将它们映射到视觉空间
            # 但 norm 通常是从数据空间到 [0,1] 的映射
            # 我们需要使用 norm 的 forward 映射
            t_candidates = norm(candidates)
            # 确保是普通数组
            if isinstance(t_candidates, np.ma.MaskedArray):
                t_candidates = np.array(t_candidates.data[~t_candidates.mask] if t_candidates.mask is not np.ma.nomask else t_candidates.data)
            else:
                t_candidates = np.array(t_candidates)
        else:
            # 如果没有 norm，使用线性映射
            t_candidates = (candidates - z_min) / (z_max - z_min) if (z_max > z_min) else np.zeros_like(candidates)
    except Exception:
        # 如果映射失败，使用线性映射
        t_candidates = (candidates - z_min) / (z_max - z_min) if (z_max > z_min) else np.zeros_like(candidates)
    
    # 使用分位数方法选择等高线：10%, 30%, 50%, 70%, 90%
    # 避免选择太接近边界（0或1）的值
    percentile_targets = [10, 30, 50, 75, 97]  # 分位数目标
    
    # 计算数据的分位数值（在数据空间中）
    z_flat = Z_sub.flatten()
    z_valid = z_flat[np.isfinite(z_flat)]
    percentile_values = np.percentile(z_valid, percentile_targets)
    
    # 将分位数值映射到视觉空间
    try:
        if hasattr(norm, 'inverse'):
            t_percentiles = norm(percentile_values)
            if isinstance(t_percentiles, np.ma.MaskedArray):
                t_percentiles = np.array(t_percentiles.data[~t_percentiles.mask] if t_percentiles.mask is not np.ma.nomask else t_percentiles.data)
            else:
                t_percentiles = np.array(t_percentiles)
        else:
            t_percentiles = (percentile_values - z_min) / (z_max - z_min) if (z_max > z_min) else np.zeros_like(percentile_values)
    except Exception:
        t_percentiles = (percentile_values - z_min) / (z_max - z_min) if (z_max > z_min) else np.zeros_like(percentile_values)
    
    # 从候选值中选择最接近分位数的值
    selected_levels = []
    last_selected_t = -np.inf
    
    for i, (percentile_val, t_percentile) in enumerate(zip(percentile_values, t_percentiles)):
        # 找到所有满足单调约束的候选值
        valid_mask = t_candidates > last_selected_t
        if not np.any(valid_mask):
            continue
        
        valid_t_candidates = t_candidates[valid_mask]
        valid_candidates = candidates[valid_mask]
        
        # 找到最接近分位数位置的候选值
        nearest_idx = np.argmin(np.abs(valid_t_candidates - t_percentile))
        selected_candidate = valid_candidates[nearest_idx]
        selected_t = valid_t_candidates[nearest_idx]
        
        selected_levels.append(selected_candidate)
        last_selected_t = selected_t
    
    if len(selected_levels) < 2:
        return np.array([])
    
    levels = np.array(selected_levels)
    # 如果条数不足，尝试添加更小的 step 来增加候选值密度
    if len(levels) < n_min:
        # 计算当前最小 step
        if len(candidates) > 1:
            min_step = np.min(np.diff(np.sort(candidates)))
        else:
            min_step = data_range / 10.0
        
        # 添加更小的 step（减半）
        new_step = min_step / 2.0
        new_step = nice_step(new_step)  # 确保是 nice step
        
        # 限制 new_step 不能太小，避免内存溢出
        min_step_allowed = data_range / MAX_CANDIDATES_PER_STEP if data_range > 0 else 1e-6
        new_step = max(new_step, min_step_allowed, 1e-6)  # 硬性限制：step 不能小于 1e-6
        
        # 生成新的候选值
        grid_start = np.floor(z_min / new_step) * new_step
        grid_end = np.ceil(z_max / new_step) * new_step
        
        # 估算需要生成的候选值数量
        if new_step > 0:
            num_points_new = int((grid_end - grid_start) / new_step) + 1
        else:
            num_points_new = 0
        
        # 如果候选值数量过多，跳过添加更小的 step
        if num_points_new > MAX_CANDIDATES_PER_STEP:
            if verbose:
                print(f"    跳过添加更小的 step={new_step:.6g}: 需要 {num_points_new} 个候选值，超过限制 {MAX_CANDIDATES_PER_STEP}")
            # 不添加新的候选值，直接使用现有的
            all_candidates = candidates.copy()
        else:
            new_candidates = np.arange(grid_start, grid_end + new_step, new_step)
            new_candidates = new_candidates[(new_candidates >= z_min + eps) & (new_candidates <= z_max - eps)]
            # 合并到现有候选值
            all_candidates = np.unique(np.concatenate([candidates, new_candidates]))
        
        # 如果数据范围在1-10之间，添加0.5的倍数候选值
        if z_min >= 1.0 and z_max <= 10.0:
            half_step_candidates = []
            for i in range(int(np.ceil(z_min * 2)), int(np.floor(z_max * 2)) + 1):
                candidate = i * 0.5
                if candidate >= z_min and candidate <= z_max:
                    half_step_candidates.append(candidate)
            all_candidates = np.unique(np.concatenate([all_candidates, half_step_candidates]))
        
        all_candidates = np.sort(all_candidates)
        
        # 重新映射到视觉空间
        try:
            if hasattr(norm, 'inverse'):
                t_all_candidates = norm(all_candidates)
                if isinstance(t_all_candidates, np.ma.MaskedArray):
                    t_all_candidates = np.array(t_all_candidates.data[~t_all_candidates.mask] if t_all_candidates.mask is not np.ma.nomask else t_all_candidates.data)
                else:
                    t_all_candidates = np.array(t_all_candidates)
            else:
                t_all_candidates = (all_candidates - z_min) / (z_max - z_min) if (z_max > z_min) else np.zeros_like(all_candidates)
        except Exception:
            t_all_candidates = (all_candidates - z_min) / (z_max - z_min) if (z_max > z_min) else np.zeros_like(all_candidates)
        
        # 重新选择：使用分位数方法
        percentile_values_new = np.percentile(z_valid, percentile_targets)
        
        # 将分位数值映射到视觉空间
        try:
            if hasattr(norm, 'inverse'):
                t_percentiles_new = norm(percentile_values_new)
                if isinstance(t_percentiles_new, np.ma.MaskedArray):
                    t_percentiles_new = np.array(t_percentiles_new.data[~t_percentiles_new.mask] if t_percentiles_new.mask is not np.ma.nomask else t_percentiles_new.data)
                else:
                    t_percentiles_new = np.array(t_percentiles_new)
            else:
                t_percentiles_new = (percentile_values_new - z_min) / (z_max - z_min) if (z_max > z_min) else np.zeros_like(percentile_values_new)
        except Exception:
            t_percentiles_new = (percentile_values_new - z_min) / (z_max - z_min) if (z_max > z_min) else np.zeros_like(percentile_values_new)
        
        selected_levels = []
        last_selected_t = -np.inf
        
        for percentile_val, t_percentile in zip(percentile_values_new, t_percentiles_new):
            valid_mask = t_all_candidates > last_selected_t
            if not np.any(valid_mask):
                continue
            
            valid_t_candidates = t_all_candidates[valid_mask]
            valid_candidates = all_candidates[valid_mask]
            
            nearest_idx = np.argmin(np.abs(valid_t_candidates - t_percentile))
            selected_candidate = valid_candidates[nearest_idx]
            selected_t = valid_t_candidates[nearest_idx]
            
            selected_levels.append(selected_candidate)
            last_selected_t = selected_t
        
        if len(selected_levels) > 0:
            levels = np.array(selected_levels)
    
    # 最终边界保护：确保所有等高线在数据范围内
    eps_final = max(eps, 1e-9 * data_range) if data_range > 0 else 1e-9
    levels = np.clip(levels, z_min + eps_final, z_max - eps_final)
    levels = np.unique(np.sort(levels))
    return levels


def snap_to_nice_number(x: float, data_range: Optional[float] = None, z_min: Optional[float] = None, z_max: Optional[float] = None) -> float:
    """
    将数值吸附到"整齐"的值：
    - 优先使用整数
    - 如果必须使用小数，保留一位有效数字
    - 考虑更多候选值，使结果更接近原始值
    - 根据数据范围自动选择粒度：范围大时使用更粗的粒度
    
    参数:
        x: 输入值
        data_range: 可选，数据范围（z_max - z_min），用于决定粒度
        z_min: 可选，数据最小值
        z_max: 可选，数据最大值
    
    返回:
        吸附后的整齐值
    """
    if x == 0.0:
        return 0.0
    
    abs_x = abs(x)
    sign = 1.0 if x >= 0 else -1.0
    
    # 计算数量级
    if abs_x > 0:
        k = int(np.floor(np.log10(abs_x)))
        magnitude = 10 ** k
    else:
        return 0.0
    
    # 根据数据范围决定粒度（粗/细）
    use_fine_grain = True
    if data_range is not None and z_min is not None and z_max is not None:
        if z_min > 0:
            range_ratio = z_max / z_min
            if range_ratio > 10:
                use_fine_grain = False
        else:
            if data_range is not None and data_range > 10:
                use_fine_grain = False

    if abs_x > 1.0:
        # 对于大于1的值：
        # - 1~10 区间始终允许细粒度（保留 4、6、9 等）；
        # - 更大数值根据 use_fine_grain 选择粗/细粒度。
        integer_candidates = []

        if magnitude >= 100:
            # 10^2 以上：根据粗/细模板生成候选值
            if use_fine_grain:
                base_template_10_100 = [10, 12, 15, 18, 20, 25, 30, 35, 40, 45,
                                        50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
            else:
                base_template_10_100 = [10, 20, 50, 80, 100]

            for template_val in base_template_10_100:
                candidate = template_val * (magnitude // 10)
                if candidate >= 1:
                    integer_candidates.append(int(candidate))
            if abs_x >= magnitude * 10:
                for template_val in base_template_10_100:
                    candidate_next = template_val * (magnitude // 10) * 10
                    if candidate_next >= 1:
                        integer_candidates.append(int(candidate_next))

        elif magnitude >= 10:
            # 10~100 区间
            if use_fine_grain:
                base_template_10_100 = [10, 12, 15, 18, 20, 25, 30, 35, 40, 45,
                                        50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
            else:
                base_template_10_100 = [10, 20, 50, 80, 100]
            integer_candidates.extend(base_template_10_100)
            if abs_x >= 8 * magnitude:
                for template_val in base_template_10_100:
                    integer_candidates.append(template_val * 10)

        elif magnitude >= 1:
            # 1~10 区间：始终使用 1~10 的整数，不受粗/细开关影响
            base_template_1_10 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            integer_candidates.extend(base_template_1_10)

        else:
            # 0.1~1 区间：按数量级缩放 1~10 模板
            base_template_1_10 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            for template_val in base_template_1_10:
                candidate = template_val * magnitude
                if candidate >= 1:
                    integer_candidates.append(int(candidate))

        integer_candidates = sorted(set(integer_candidates))
        # 优先选择不超过 z_max 的最近值（若提供），避免被后续 clip 推高
        if z_max is not None and len(integer_candidates) > 0:
            le_candidates = [c for c in integer_candidates if c <= z_max]
            if le_candidates:
                nearest_integer = min(le_candidates, key=lambda c: abs(abs_x - c))
            else:
                nearest_integer = min(integer_candidates, key=lambda c: abs(abs_x - c))
        else:
            nearest_integer = min(integer_candidates, key=lambda c: abs(abs_x - c))
        result = sign * float(nearest_integer)
    else:
        # 对于 <= 1 的值：细粒度用所有一位有效数字，粗粒度用 0.2/0.5/1.0（即2/5/10系列）
        normalized = abs_x / magnitude

        if use_fine_grain:
            candidates = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        else:
            candidates = [2.0, 5.0, 10.0]

        nearest_candidate = min(candidates, key=lambda c: abs(normalized - c))
        if nearest_candidate == 10.0:
            result = sign * 1.0 * (magnitude * 10)
        else:
            result = sign * nearest_candidate * magnitude
    
    return result


def nice_step(x: float) -> float:
    """
    返回最接近 x 的 {1,2,5,10} * 10^k 形式的"好看的"步长。
    
    参数:
        x: 输入值
    
    返回:
        最接近的 nice step（1, 2, 5, 10 或其 10 的幂次倍）
    """
    if x <= 0:
        return 1.0
    
    # 计算数量级
    magnitude = 10 ** np.floor(np.log10(x))
    
    # 归一化到 1-10 范围
    normalized = x / magnitude
    
    # 选择最接近的 {1, 2, 5, 10}
    if normalized <= 1.5:
        nice_normalized = 1.0
    elif normalized <= 3.5:
        nice_normalized = 2.0
    elif normalized <= 7.5:
        nice_normalized = 5.0
    else:
        nice_normalized = 10.0
    
    return nice_normalized * magnitude


def snap_levels_to_nice(
    levels: np.ndarray,
    z_min: float,
    z_max: float,
    n_min: int = 4,
    prefer_integers: bool = True,
    norm: Optional[mpl.colors.Normalize] = None,
    verbose: bool = False,
) -> np.ndarray:
    """
    从“整齐候选集”中直接选择 n 条（与 t_targets 视觉均匀对应），不再先 snap 再选。
    """
    if len(levels) == 0:
        return levels

    # 预处理
    levels = np.unique(np.sort(np.array(levels, dtype=float)))
    data_range = z_max - z_min
    if not np.isfinite(data_range) or data_range <= 0:
        return np.clip(levels, z_min, z_max)

    n = len(levels)

    # ---------- 生成候选：local grid（整数/整位数） + coarse 1-2-5 ----------
    def generate_125(zlo: float, zhi: float) -> np.ndarray:
        """在 [zlo,zhi] 内生成 1-2-5×10^k 的整齐刻度"""
        if zhi <= 0:
            return np.array([])
        eps = 1e-12
        lo = max(zlo, eps)
        k_min = int(np.floor(np.log10(lo)))
        k_max = int(np.ceil(np.log10(zhi)))
        vals = []
        for k in range(k_min - 1, k_max + 2):
            base = 10.0 ** k
            for m in [1.0, 2.0, 5.0]:
                v = m * base
                if zlo - eps <= v <= zhi + eps:
                    vals.append(v)
        return np.array(sorted(set(vals)), dtype=float)

    # 1) 先根据范围选择 local grid 的 step（越粗优先）
    step_list = [20.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.2]
    step_chosen = step_list[-1]
    for st in step_list:
        if np.floor(data_range / st) + 1 >= n:
            step_chosen = st
            break

    # 用 step_chosen 生成 local grid
    start = np.ceil(z_min / step_chosen) * step_chosen
    # 至少覆盖到 z_max
    k_max_local = int(np.floor((z_max - start) / step_chosen)) + 2
    grid_vals = start + step_chosen * np.arange(0, max(k_max_local, 1))
    eps_g = 1e-12 * max(1.0, abs(z_max))
    local_grid = grid_vals[(grid_vals > z_min + eps_g) & (grid_vals < z_max - eps_g)]

    # 2) 再叠加 coarse 1-2-5 × 10^k
    coarse_125 = generate_125(z_min, z_max)

    candidates = np.unique(
        np.concatenate(
            [
                local_grid.astype(float),
                coarse_125.astype(float),
            ]
        )
    )


    if len(candidates) == 0:
        # 理论上不会发生，因为 local_grid 一定能覆盖
        return np.clip(levels, z_min, z_max)

    # ---------- 视觉空间映射 ----------
    if norm is not None:
        try:
            t_candidates = norm(candidates)
            if isinstance(t_candidates, np.ma.MaskedArray):
                t_candidates = np.array(t_candidates.data[~t_candidates.mask] if t_candidates.mask is not np.ma.nomask else t_candidates.data)
            else:
                t_candidates = np.array(t_candidates)
        except Exception:
            norm = None  # fallback
    if norm is None:
        t_candidates = (candidates - z_min) / data_range

    # t_targets：保持视觉均匀（使用原始 levels 的 t 范围）
    if norm is not None:
        t_raw = norm(levels)
        if isinstance(t_raw, np.ma.MaskedArray):
            t_raw = np.array(t_raw.data[~t_raw.mask] if t_raw.mask is not np.ma.nomask else t_raw.data)
        else:
            t_raw = np.array(t_raw)
    else:
        t_raw = (levels - z_min) / data_range

    t_lo = float(np.min(t_raw))
    t_hi = float(np.max(t_raw))
    if not np.isfinite(t_lo) or not np.isfinite(t_hi) or t_hi <= t_lo:
        return np.clip(levels, z_min, z_max)

    t_targets = np.linspace(t_lo, t_hi, n)

    # t_gap_min：防止过于拥挤
    t_span = t_hi - t_lo
    t_gap_min = t_span / (n * 4.0)


    # ---------- 带回溯的选择 ----------
    def try_select(cand_vals: np.ndarray) -> Optional[np.ndarray]:
        if cand_vals.size == 0:
            return None
        try:
            t_cand = norm(cand_vals) if norm is not None else (cand_vals - z_min) / data_range
            if isinstance(t_cand, np.ma.MaskedArray):
                t_cand = np.array(t_cand.data[~t_cand.mask] if t_cand.mask is not np.ma.nomask else t_cand.data)
            else:
                t_cand = np.array(t_cand)
        except Exception:
            t_cand = (cand_vals - z_min) / data_range

        best_solution: Optional[np.ndarray] = None

        def backtrack(idx, prev_t, prev_val, chosen):
            nonlocal best_solution
            if idx == n:
                best_solution = np.array(chosen, dtype=float)
                return True
            tgt = t_targets[idx]
            order = np.argsort(np.abs(t_cand - tgt))
            for k in order:
                y = cand_vals[k]
                ty = t_cand[k]
                if idx > 0:
                    if y <= prev_val:
                        continue
                    if ty - prev_t < t_gap_min:
                        continue
                chosen.append(y)
                if backtrack(idx + 1, ty, y, chosen):
                    return True
                chosen.pop()
            return False

        ok = backtrack(0, t_lo - t_gap_min, z_min - data_range, [])
        return np.unique(np.sort(best_solution)) if ok and best_solution is not None else None

    # 1) 先用 1-2-5 + raw single-sig
    solution = try_select(candidates)

    # 2) 如果不够，再放宽：加入 mantissa [1,1.5,2,3,5,7] × 10^k
    if solution is None or len(solution) < n_min:
        if verbose:
            print("  - widen candidates with mantissas [1,1.5,2,3,5,7]")
        dense_vals = []
        if z_max > 0:
            k_min = int(np.floor(np.log10(max(z_min, 1e-12))))
            k_max = int(np.ceil(np.log10(z_max)))
            for k in range(k_min - 1, k_max + 2):
                base = 10.0 ** k
                for m in [1.0, 1.5, 2.0, 3.0, 5.0, 7.0]:
                    v = m * base
                    if z_min <= v <= z_max:
                        dense_vals.append(v)
        candidates_dense = np.unique(np.concatenate([candidates, np.array(dense_vals, dtype=float)]))
        solution = try_select(candidates_dense)

    # 3) 仍不足，再把所有 raw 都加入
    if (solution is None or len(solution) < n_min) and len(levels) > 0:
        if verbose:
            print("  - final fallback: add all raw levels to candidates")
        candidates_all = np.unique(np.concatenate([candidates, levels]))
        solution = try_select(candidates_all)

    if solution is not None and len(solution) > 0:
        # 上端锚定（Upper Anchor）：尽量把最后一条推到接近 z_max 的整齐值
        levels_chosen = np.array(solution, dtype=float)
        if levels_chosen.size >= 2 and candidates.size > 0:
            eps_cap = max(1e-9 * max(1.0, abs(z_max)), 1e-6 * data_range)
            upper_cap = z_max - eps_cap
            anchor_base = levels_chosen[-2]
            upper_pool = candidates[(candidates > anchor_base) & (candidates < upper_cap)]
            if upper_pool.size > 0:
                # 允许的最小间隔：不小于当前 t_gap_min
                anchor_level = levels_chosen[-1]
                if norm is not None:
                    try:
                        t_prev = float(norm(anchor_base))
                    except Exception:
                        t_prev = (anchor_base - z_min) / data_range
                else:
                    t_prev = (anchor_base - z_min) / data_range

                best_anchor = anchor_level
                for cand in np.sort(upper_pool)[::-1]:  # 从大到小尝试
                    if norm is not None:
                        try:
                            t_cand = float(norm(cand))
                        except Exception:
                            t_cand = (cand - z_min) / data_range
                    else:
                        t_cand = (cand - z_min) / data_range
                    if t_cand - t_prev >= t_gap_min:
                        best_anchor = cand
                        break
                levels_chosen[-1] = best_anchor

        return np.unique(np.sort(levels_chosen))

    # 保底：返回原始 levels（clip）
    fallback = np.clip(levels, z_min, z_max)
    if verbose:
        print("  - backtrack failed, fallback to clipped raw levels")
        print(f"  - fallback levels: {np.round(fallback, 4).tolist()}")
    return fallback


def filter_contour_levels_to_target_count(
    levels: np.ndarray,
    z_min: float,
    z_max: float,
    master_levels: Optional[np.ndarray] = None,
    target_min_count: int = 4,
    target_max_count: int = 5,
    verbose: bool = False
) -> np.ndarray:
    """
    筛选等高线到目标数量（4-5条）。
    策略：保留最大值、最小值，然后从中间均匀选择剩下的。
    
    参数:
        levels: 当前等高线级别数组（已排序）
        z_min: 数据最小值
        z_max: 数据最大值
        master_levels: 可选，主等高线数组，用于补充等高线
        target_min_count: 目标最小等高线数量（默认4条）
        target_max_count: 目标最大等高线数量（默认5条）
        verbose: 是否输出详细信息
    
    返回:
        筛选后的等高线级别数组
    """
    if len(levels) == 0:
        return levels
    
    levels = np.sort(np.array(levels, dtype=float))
    
    # 如果超过目标最大数量，筛选到目标数量
    if len(levels) > target_max_count:
        levels_before_limit = levels.copy()
        
        # 保留最大值和最小值
        min_level = levels[0]
        max_level = levels[-1]
        
        # 获取中间部分的等高线（排除最大值和最小值）
        if len(levels) > 2:
            middle_levels = levels[1:-1]
            
            # 需要从中间选择的数量：目标总数 - 2（最大值和最小值）
            target_middle_count = target_max_count - 2  # 5 - 2 = 3
            
            if len(middle_levels) >= target_middle_count:
                # 从中间等高线中均匀选择
                indices = np.linspace(0, len(middle_levels) - 1, target_middle_count, dtype=int)
                selected_middle = middle_levels[indices]
                levels = np.sort(np.concatenate([[min_level], selected_middle, [max_level]]))
            else:
                # 如果中间等高线不足，全部保留（总数会少于5条）
                levels = np.sort(np.concatenate([[min_level], middle_levels, [max_level]]))
        
        if verbose:
            print(f"    - 步骤5 [筛选等高线]: 从 {len(levels_before_limit)} 条筛选到 {len(levels)} 条（保留最大值和最小值，中间均匀选择）")
    
    # 如果少于目标最小数量，尝试从主数组中补充
    elif len(levels) < target_min_count:
        levels_before_limit = levels.copy()
        if len(levels) > 0 and master_levels is not None and len(master_levels) > 0:
            min_level = levels[0]
            max_level = levels[-1]
            
            # 从主数组中查找在数据范围内但不在当前等高线中的值
            candidate_levels = master_levels[
                (master_levels >= z_min) & 
                (master_levels <= z_max) &
                ~np.isin(master_levels, levels)
            ]
            
            if len(candidate_levels) > 0:
                # 需要补充的数量
                need_count = target_min_count - len(levels)
                if need_count > 0:
                    # 优先选择中间位置的等高线
                    if len(candidate_levels) >= need_count:
                        # 均匀选择
                        indices = np.linspace(0, len(candidate_levels) - 1, need_count, dtype=int)
                        selected = candidate_levels[indices]
                        levels = np.sort(np.concatenate([levels, selected]))
                    else:
                        # 如果候选值不足，全部添加
                        levels = np.sort(np.concatenate([levels, candidate_levels]))
                    
                    if verbose:
                        print(f"    - 步骤5 [补充等高线]: 从 {len(levels_before_limit)} 条补充到 {len(levels)} 条")
    
    # 最终检查：确保等高线数量在目标范围内
    if len(levels) > target_max_count:
        levels_before_limit = levels.copy()
        min_level = levels[0]
        max_level = levels[-1]
        if len(levels) > 2:
            middle_levels = levels[1:-1]
            target_middle_count = target_max_count - 2
            if len(middle_levels) >= target_middle_count:
                indices = np.linspace(0, len(middle_levels) - 1, target_middle_count, dtype=int)
                selected_middle = middle_levels[indices]
                levels = np.sort(np.concatenate([[min_level], selected_middle, [max_level]]))
            else:
                levels = np.sort(np.concatenate([[min_level], middle_levels, [max_level]]))
        
        if verbose:
            print(f"    - 最终检查 [限制等高线数量]: 从 {len(levels_before_limit)} 条强制限制到 {len(levels)} 条")
    
    return levels


def save_hatch_legend(save_path: str):
    """生成并保存格纹图例"""
    setup_plot_style()
    hatch_map = {
        'HTS Tape Cost': '/',
        'Power Supply Cost': '|',
        'Coolant Cost': '\\'
    }
    handles = [mpatches.Patch(facecolor='white', edgecolor='black', hatch=hatch, label=label) for label, hatch in hatch_map.items()]
    fig = plt.figure(figsize=(10, 1.5))
    ax = fig.add_subplot(111)
    ax.axis('off')
    legend = ax.legend(
        handles=handles,
        loc='center',
        ncol=len(handles),
        frameon=False,
        fontsize=18,
        title='Direct Magnet System Cost Component',
        handleheight=1.5
    )
    legend.get_frame().set_facecolor('none')
    legend.get_frame().set_edgecolor('none')
    legend.get_title().set_fontsize(18)
    plt.tight_layout()
    plt.savefig(save_path, dpi=cfg.PLOT_DPI, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"Hatch legend saved to: {save_path}")


def determine_contour_color_by_background(
    contour_path: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    cmap: str,
    norm: mpl.colors.Normalize,
    sample_points: int = 20,
    luminance_threshold: float = 0.5
) -> str:
    """
    根据等高线路径上背景色的亮度，动态确定等高线颜色。
    如果背景色较浅（亮度高），返回黑色；如果背景色较深（亮度低），返回白色。
    
    参数:
        contour_path: 等高线路径点数组，形状为 (N, 2)，每行为 (x, y) 坐标
        X: 数据网格的X坐标，形状为 (ny, nx)
        Y: 数据网格的Y坐标，形状为 (ny, nx)
        Z: 数据值数组，形状为 (ny, nx)，用于确定背景色
        cmap: 颜色映射名称（字符串）
        norm: Matplotlib归一化对象，用于将Z值映射到颜色
        sample_points: 在等高线路径上采样的点数（默认20）
        luminance_threshold: 亮度阈值（0-1），超过此值使用黑色，否则使用白色（默认0.5）
    
    返回:
        颜色字符串：'black' 或 'white'
    """
    if len(contour_path) == 0:
        return 'black'  # 默认返回黑色
    
    # 在等高线路径上均匀采样点
    if len(contour_path) > sample_points:
        indices = np.linspace(0, len(contour_path) - 1, sample_points, dtype=int)
        sampled_points = contour_path[indices]
    else:
        sampled_points = contour_path
    
    # 获取颜色映射对象
    if isinstance(cmap, str):
        colormap = plt.get_cmap(cmap)
    else:
        colormap = cmap
    
    # 计算所有采样点的平均亮度
    total_luminance = 0.0
    valid_points = 0
    
    for point in sampled_points:
        x, y = point[0], point[1]
        
        # 检查点是否在数据范围内
        if (X.min() <= x <= X.max()) and (Y.min() <= y <= Y.max()):
            # 使用双线性插值获取该点的Z值
            # 找到最近的网格点
            # 对于对数轴，需要特殊处理，但这里简化处理，使用线性插值
            
            # 找到x和y在网格中的位置
            # 对于X（可能是对数刻度），需要找到对应的列索引
            if X.shape[1] > 1:
                # 使用对数空间的插值
                x_log = np.log10(np.clip(x, X.min(), X.max()))
                x_min_log = np.log10(X.min())
                x_max_log = np.log10(X.max())
                x_frac = (x_log - x_min_log) / (x_max_log - x_min_log) if (x_max_log > x_min_log) else 0.5
                col_idx = x_frac * (X.shape[1] - 1)
            else:
                col_idx = 0
            
            # 对于Y（线性刻度），直接插值
            if Y.shape[0] > 1:
                y_frac = (y - Y.min()) / (Y.max() - Y.min()) if (Y.max() > Y.min()) else 0.5
                row_idx = y_frac * (Y.shape[0] - 1)
            else:
                row_idx = 0
            
            # 使用双线性插值获取Z值
            row_low = int(np.floor(row_idx))
            row_high = min(int(np.ceil(row_idx)), Z.shape[0] - 1)
            col_low = int(np.floor(col_idx))
            col_high = min(int(np.ceil(col_idx)), Z.shape[1] - 1)
            
            row_low = max(0, row_low)
            row_high = max(0, row_high)
            col_low = max(0, col_low)
            col_high = max(0, col_high)
            
            # 双线性插值权重
            row_weight = row_idx - row_low
            col_weight = col_idx - col_low
            
            # 获取四个角点的Z值
            z00 = Z[row_low, col_low] if (row_low < Z.shape[0] and col_low < Z.shape[1]) else np.nan
            z01 = Z[row_low, col_high] if (row_low < Z.shape[0] and col_high < Z.shape[1]) else np.nan
            z10 = Z[row_high, col_low] if (row_high < Z.shape[0] and col_low < Z.shape[1]) else np.nan
            z11 = Z[row_high, col_high] if (row_high < Z.shape[0] and col_high < Z.shape[1]) else np.nan
            
            # 双线性插值
            if not (np.isnan(z00) or np.isnan(z01) or np.isnan(z10) or np.isnan(z11)):
                z_interp = (z00 * (1 - row_weight) * (1 - col_weight) +
                           z01 * (1 - row_weight) * col_weight +
                           z10 * row_weight * (1 - col_weight) +
                           z11 * row_weight * col_weight)
                
                # 将Z值归一化到[0,1]范围（通过norm）
                try:
                    normalized_z = norm(z_interp)
                    if isinstance(normalized_z, np.ma.MaskedArray):
                        normalized_z = float(normalized_z.data[0]) if normalized_z.size > 0 else 0.5
                    else:
                        normalized_z = float(normalized_z)
                    
                    # 确保在[0,1]范围内
                    normalized_z = np.clip(normalized_z, 0.0, 1.0)
                    
                    # 获取颜色（RGBA格式）
                    rgba = colormap(normalized_z)
                    
                    # 计算亮度（使用标准的RGB到亮度转换公式）
                    # 亮度公式：L = 0.299*R + 0.587*G + 0.114*B
                    r, g, b = rgba[0], rgba[1], rgba[2]
                    luminance = 0.299 * r + 0.587 * g + 0.114 * b
                    
                    total_luminance += luminance
                    valid_points += 1
                except Exception:
                    # 如果归一化失败，跳过该点
                    pass
    
    # 计算平均亮度
    if valid_points > 0:
        avg_luminance = total_luminance / valid_points
    else:
        # 如果没有有效点，默认返回黑色
        return 'black'
    
    # 根据平均亮度决定颜色
    if avg_luminance >= luminance_threshold:
        return 'black'  # 背景色浅，使用黑色等高线
    else:
        return 'white'  # 背景色深，使用白色等高线


def get_contour_colors_by_background(
    contour_set,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    cmap: str,
    norm: mpl.colors.Normalize,
    sample_points: int = 20,
    luminance_threshold: float = 0.5
) -> dict:
    """
    为等高线集合中的每条等高线确定颜色（基于背景色亮度）。
    
    参数:
        contour_set: matplotlib contour对象（QuadContourSet）
        X: 数据网格的X坐标，形状为 (ny, nx)
        Y: 数据网格的Y坐标，形状为 (ny, nx)
        Z: 数据值数组，形状为 (ny, nx)
        cmap: 颜色映射名称（字符串）
        norm: Matplotlib归一化对象
        sample_points: 每条等高线路径上采样的点数（默认20）
        luminance_threshold: 亮度阈值（默认0.5）
    
    返回:
        字典：{level: color}，其中level是等高线值，color是'black'或'white'
    """
    level_to_color = {}
    
    if not contour_set or not hasattr(contour_set, 'allsegs'):
        return level_to_color
    
    # 遍历每条等高线
    for i, level in enumerate(contour_set.levels):
        segs = contour_set.allsegs[i]
        
        # 收集该等高线的所有路径点
        all_path_points = []
        for seg in segs:
            if len(seg) > 1:
                all_path_points.append(seg)
        
        if len(all_path_points) > 0:
            # 合并所有路径点
            combined_path = np.vstack(all_path_points)
            
            # 确定该等高线的颜色
            color = determine_contour_color_by_background(
                combined_path, X, Y, Z, cmap, norm, sample_points, luminance_threshold
            )
            level_to_color[level] = color
    
    return level_to_color

    # =============================================================================
# 绘图函数（新增）：ΔLCOE 热力图（风格与 parasitic heatmap 一致）
# =============================================================================

def plot_delta_lcoe_heatmap_single(
    df: pd.DataFrame,
    output_dir: Path,
    cmap: str,
    norm: mpl.colors.Normalize,
    scenario: str,
    temperature_K: float,
    value_col: str = "delta_lcoe_$/MWh",
    baseline_point: Optional[Tuple[float, float]] = None,# (Npw_baseline, R_joint_baseline)
    coolant: Optional[str] = None,  # 制冷剂类型，用于文件名区分
    contour_levels: Optional[np.ndarray] = None,  # 可选的等高线级别数组（如果为None，则自动生成）
    auto_generate_levels: bool = True,  # 是否自动生成等高线（默认True）
    draw_contours: bool = True,
    verbose: bool = True,
    filename_suffix: str = "",  # 可选的文件名后缀，用于区分不同配色方案

):
    """
    为单个场景和单个温度生成并保存 ΔLCOE 热力图（不含colorbar、坐标轴标签和刻度值）。
    画风与 plot_parasitic_heatmap_single 保持一致。

    参数:
        df: 包含单个温度数据的DataFrame
        output_dir: 输出目录
        cmap: 颜色映射
        norm: 全局归一化对象（建议 SymLogNorm）
        scenario: 场景名称
        temperature_K: 温度值
        value_col: 默认 'delta_lcoe_$/MWh'
        baseline_point: 可选，在图上用星号标注基准点 A
        coolant: 可选，制冷剂类型（如"He"或"H2"），用于文件名区分
        contour_levels: 可选的等高线级别数组（如果为None且auto_generate_levels=True，则自动生成）
        auto_generate_levels: 是否自动生成等高线（默认True）
    """
    font_size = 18
    setup_plot_style()

    # 生成label，如果提供了coolant则在文件名中包含
    if coolant is not None:
        label = f"{temperature_K}K_{coolant}"
    else:
        label = f"{temperature_K}K"

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    # 与 parasitic heatmap 相同的网格化方式：index=Npw, columns=R_joint
    pivot = df.pivot(index="Npw", columns="R_joint", values=value_col)
    
    # 获取原始网格（用于插值）
    R_joint_original = pivot.columns.values  # 原始 R_joint 值（Ω单位）
    Npw_original = pivot.index.values  # 原始 Npw 值
    Z_original = pivot.values  # 原始 Z 值
    
    # 使用对数数组创建绘图网格（匹配对数刻度）
    # 配置中的 R_JOINT_SCAN_VALUES 以 nOhm 为单位，这里转换为 Ohm 参与插值和绘图
    R_joint_plot = np.array(cfg.R_JOINT_SCAN_VALUES, dtype=float) * 1e-9
    Npw_plot = Npw_original  # Npw 保持不变
    
    # 创建插值器（使用原始网格）
    interp = RegularGridInterpolator(
        (Npw_original, R_joint_original),
        Z_original,
        bounds_error=False,
        fill_value=np.nan
    )
    
    # 创建绘图网格
    X_plot, Y_plot = np.meshgrid(R_joint_plot, Npw_plot, indexing='xy')
    
    # 插值到绘图网格
    Z_plot_interp = interp((Y_plot, X_plot))
    
    # 记录NaN位置（不可行点），不填充，后续用Hatch显示
    nan_mask = np.isnan(Z_plot_interp)
    
    # 对于绘图，使用masked array，NaN值不会被绘制（显示为白色/背景色）
    # 但我们需要用Hatch填充来明确标识这些不可行点
    Z_plot = np.ma.masked_array(Z_plot_interp, mask=nan_mask)

    # 使用对数网格绘图
    # X的单位是Ohm，范围对应1nOhm到100nOhm（即1e-9到100e-9 Ohm）
    
    # 绘制热力图（NaN区域会被mask掉，显示为背景色）
    ax.pcolormesh(X_plot, Y_plot, Z_plot, cmap=cmap, norm=norm, shading="auto")
    ax.set_xscale('log')  # 设置横轴为对数刻度
    
    # 为了兼容后续代码，保留原始变量名（但使用插值后的值）
    X = X_plot
    Y = Y_plot
    Z = Z_plot_interp  # 用于等高线等后续处理
    
    # 判断是否是"相对于最小值"的等高线（用于控制输出详细程度）
    is_delta_min = "delta_lcoe_min" in value_col or "relative to scenario minimum" in str(value_col).lower()
    
    # 自动生成等高线（如果启用）
    if draw_contours and auto_generate_levels and contour_levels is None:
        contour_levels = auto_contour_levels(
            Z_sub=Z,
            norm=norm,
            n_min=4,
            n_max=5,  # 最多5条等高线
            q=0.05,
            margin=0.03,
            include_zero=True,
            verbose=is_delta_min  # 只有"相对于最小值"才输出详细信息
        )
    
    # 如果仍然没有等高线，检查是否提供了
    if draw_contours and (contour_levels is None or len(contour_levels) == 0):
        if not auto_generate_levels:
            raise ValueError(
                f"plot_delta_lcoe_heatmap_single: contour_levels must be provided when auto_generate_levels=False. "
                f"Got None or empty array for scenario={scenario}, temperature_K={temperature_K}, coolant={coolant}"
            )
        # 如果自动生成失败，跳过等高线绘制
        contour_levels_to_use = np.array([])
    else:
        contour_levels_to_use = np.array(contour_levels, dtype=float)
    
    # 对NaN区域（不可行点）使用Hatch填充
    if nan_mask.any():
        # 使用pcolormesh的网格来绘制Hatch
        # pcolormesh使用边角点，所以需要扩展网格
        X_edges = np.zeros((len(Y) + 1, len(X[0]) + 1))
        Y_edges = np.zeros((len(Y) + 1, len(X[0]) + 1))
        
        # 计算边角点位置
        for i in range(len(Y) + 1):
            for j in range(len(X[0]) + 1):
                if i == 0:
                    if len(Y) > 1:
                        Y_edges[i, j] = Y[0, 0] - (Y[1, 0] - Y[0, 0]) / 2
                    else:
                        Y_edges[i, j] = Y[0, 0] - 1
                elif i == len(Y):
                    Y_edges[i, j] = Y[-1, 0] + (Y[-1, 0] - Y[-2, 0]) / 2 if len(Y) > 1 else Y[0, 0] + 1
                else:
                    Y_edges[i, j] = (Y[i-1, 0] + Y[i, 0]) / 2
                
                if j == 0:
                    if len(X[0]) > 1:
                        # 对数轴：使用几何平均
                        X_edges[i, j] = X[0, 0] / np.sqrt(X[0, 1] / X[0, 0])
                    else:
                        X_edges[i, j] = X[0, 0] * 0.9
                elif j == len(X[0]):
                    if len(X[0]) > 1:
                        # 对数轴：使用几何平均
                        X_edges[i, j] = X[0, -1] * np.sqrt(X[0, -1] / X[0, -2])
                    else:
                        X_edges[i, j] = X[0, 0] * 1.1
                else:
                    # 对数轴：使用几何平均
                    X_edges[i, j] = np.sqrt(X[0, j-1] * X[0, j])
        
        # 对每个NaN的网格单元绘制Hatch
        for i in range(len(Y)):
            for j in range(len(X[0])):
                if nan_mask[i, j]:
                    # 绘制Hatch填充的矩形
                    rect = mpatches.Rectangle(
                        (X_edges[i, j], Y_edges[i, j]),
                        X_edges[i, j+1] - X_edges[i, j],
                        Y_edges[i+1, j] - Y_edges[i, j],
                        facecolor='none',
                        edgecolor='black',
                        linewidth=0.5,
                        hatch='///',
                        alpha=0.5
                    )
                    ax.add_patch(rect)

    # --- 等高线：必须传入等高线级别，否则报错 ---
    if draw_contours and (contour_levels is None or len(contour_levels) == 0):
        raise ValueError(
            f"plot_delta_lcoe_heatmap_single: contour_levels must be provided. "
            f"Got None or empty array for scenario={scenario}, temperature_K={temperature_K}, coolant={coolant}"
        )
    
    contour_levels_to_use = np.array(contour_levels, dtype=float) if (draw_contours and contour_levels is not None) else np.array([])
    
    # 初始化filtered_levels，用于后续colorbar标注
    filtered_levels = np.array([])
    
    # 辅助函数：检测等高线标签是否与边框或已有等高线重合
    def _detect_overlapping_labels(texts, axis, levels_map: dict, margin_axes: float = 0.05, margin_data_y: float = 5.0):
        """
        检测哪些等高线标签与边框或已有等高线重合。
        
        参数:
            texts: clabel返回的文本对象列表
            axis: matplotlib axes对象
            levels_map: 标签对象到等高线值的映射字典
            margin_axes: 以坐标轴归一化坐标为单位的边缘留白（0-1），默认5%
            margin_data_y: 以数据坐标为单位的y轴下边缘留白（用于横轴检测），默认5.0
        
        返回:
            overlapping_levels: 需要移除的等高线值列表（因为标签与边框重合）
            overlapping_pairs: 需要移除的等高线值列表（因为标签与其他等高线标签重合）
        """
        if not texts:
            return [], []
        
        # 获取当前坐标轴的数据范围
        xlim = axis.get_xlim()
        ylim = axis.get_ylim()
        
        overlapping_levels = []  # 与边框重合的等高线
        label_positions = []  # 存储所有标签的位置和对应的等高线值
        
        # 获取renderer用于计算bbox
        try:
            renderer = axis.figure.canvas.get_renderer()
        except:
            renderer = None
        
        for txt in texts:
            if not txt.get_visible():
                continue
                
            xdata, ydata = txt.get_position()
            
            # 获取标签对应的等高线值（从映射中获取）
            level_value = levels_map.get(txt, None)
            if level_value is None:
                # 如果映射中没有，尝试从文本中解析
                try:
                    level_value = float(txt.get_text())
                except (ValueError, TypeError):
                    continue
            
            # 获取标签的bbox（用于检测与其他标签的重合）
            if renderer is not None:
                try:
                    bbox = txt.get_window_extent(renderer=renderer)
                except:
                    bbox = None
            else:
                bbox = None
            
            # 方法1：使用axes坐标检测（适用于所有边缘）
            x_disp, y_disp = axis.transData.transform((xdata, ydata))
            x_axes, y_axes = axis.transAxes.inverted().transform((x_disp, y_disp))
            
            # 方法2：使用数据坐标检测（特别针对横轴，因为y轴下限是2）
            near_bottom = ydata < (ylim[0] + margin_data_y)
            near_top = ydata > (ylim[1] - margin_data_y)
            near_left = xdata < (xlim[0] + (xlim[1] - xlim[0]) * margin_axes)
            near_right = xdata > (xlim[1] - (xlim[1] - xlim[0]) * margin_axes)
            
            # 使用axes坐标检测（适用于所有边缘，更严格）
            near_edge_axes = (
                x_axes < margin_axes
                or x_axes > 1.0 - margin_axes
                or y_axes < margin_axes
                or y_axes > 1.0 - margin_axes
            )
            
            # 如果标签靠近任何边缘，记录需要移除
            if near_bottom or near_top or near_left or near_right or near_edge_axes:
                overlapping_levels.append(level_value)
            
            # 存储标签位置用于检测与其他标签的重合
            if bbox is not None:
                label_positions.append({
                    'text': txt,
                    'bbox': bbox,
                    'level': level_value,
                    'x': xdata,
                    'y': ydata,
                    'x_axes': x_axes,
                    'y_axes': y_axes
                })
        
        # 检测标签之间的重合
        overlapping_pairs = []
        for i, label1 in enumerate(label_positions):
            if label1['level'] in overlapping_levels:
                continue  # 已经因为边框重合被标记为移除
            for j, label2 in enumerate(label_positions[i+1:], start=i+1):
                if label2['level'] in overlapping_levels:
                    continue  # 已经因为边框重合被标记为移除
                
                # 检测两个标签的bbox是否重叠
                bbox1 = label1['bbox']
                bbox2 = label2['bbox']
                
                # 计算两个bbox的重叠区域
                overlap_x = max(0, min(bbox1.x1, bbox2.x1) - max(bbox1.x0, bbox2.x0))
                overlap_y = max(0, min(bbox1.y1, bbox2.y1) - max(bbox1.y0, bbox2.y0))
                overlap_area = overlap_x * overlap_y
                
                # 如果重叠面积超过较小bbox面积的30%，认为重合
                min_area = min(bbox1.width * bbox1.height, bbox2.width * bbox2.height)
                if min_area > 0 and overlap_area / min_area > 0.3:
                    # 记录需要移除的等高线值（移除其中一个，保留较小的值）
                    val1 = label1['level']
                    val2 = label2['level']
                    # 移除较大的值（通常较大的值在图上位置更靠上或靠右，更容易与其他标签重合）
                    if abs(val1) > abs(val2):
                        overlapping_pairs.append(val1)
                    else:
                        overlapping_pairs.append(val2)
        
        return overlapping_levels, overlapping_pairs
    
    # 辅助函数：检测等高线路径是否相交
    def _detect_contour_intersections(contour_set):
        """
        检测等高线路径是否相交。
        
        参数:
            contour_set: matplotlib contour对象
        
        返回:
            intersecting_levels: 相交的等高线值列表
        """
        if not contour_set or not hasattr(contour_set, 'allsegs'):
            return []
        
        intersecting_levels = []
        all_paths = []
        level_to_paths = {}
        
        # 收集所有等高线路径
        for i, level in enumerate(contour_set.levels):
            segs = contour_set.allsegs[i]
            paths_for_level = []
            for seg in segs:
                if len(seg) > 1:
                    paths_for_level.append(seg)
                    all_paths.append((level, seg))
            if paths_for_level:
                level_to_paths[level] = paths_for_level
        
        # 检测路径相交
        def segments_intersect(seg1, seg2, tol=1e-6):
            """检测两条线段是否相交"""
            if len(seg1) < 2 or len(seg2) < 2:
                return False
            
            # 检查seg1的每条边是否与seg2的每条边相交
            for i in range(len(seg1) - 1):
                p1 = seg1[i]
                p2 = seg1[i + 1]
                for j in range(len(seg2) - 1):
                    q1 = seg2[j]
                    q2 = seg2[j + 1]
                    
                    # 计算两条线段的参数方程
                    # p = p1 + t*(p2-p1), q = q1 + s*(q2-q1)
                    # 求解 t 和 s
                    dx1 = p2[0] - p1[0]
                    dy1 = p2[1] - p1[1]
                    dx2 = q2[0] - q1[0]
                    dy2 = q2[1] - q1[1]
                    
                    denom = dx1 * dy2 - dy1 * dx2
                    if abs(denom) < tol:
                        continue  # 平行线段
                    
                    t = ((q1[0] - p1[0]) * dy2 - (q1[1] - p1[1]) * dx2) / denom
                    s = ((q1[0] - p1[0]) * dy1 - (q1[1] - p1[1]) * dx1) / denom
                    
                    # 检查交点是否在两条线段上
                    if 0 <= t <= 1 and 0 <= s <= 1:
                        # 检查交点是否不在端点（避免端点重合）
                        if (t > tol and t < 1 - tol) or (s > tol and s < 1 - tol):
                            return True
            return False
        
        # 检查不同等高线之间的路径是否相交
        levels_list = list(level_to_paths.keys())
        for i, level1 in enumerate(levels_list):
            if level1 in intersecting_levels:
                continue
            paths1 = level_to_paths[level1]
            for j, level2 in enumerate(levels_list[i+1:], start=i+1):
                if level2 in intersecting_levels:
                    continue
                paths2 = level_to_paths[level2]
                
                # 检查level1和level2的路径是否相交
                for path1 in paths1:
                    for path2 in paths2:
                        if segments_intersect(path1, path2):
                            # 如果相交,记录较大的等高线值（通常较大的值更容易与其他等高线相交）
                            if abs(level1) > abs(level2):
                                if level1 not in intersecting_levels:
                                    intersecting_levels.append(level1)
                            else:
                                if level2 not in intersecting_levels:
                                    intersecting_levels.append(level2)
                            break
                    if level1 in intersecting_levels or level2 in intersecting_levels:
                        break
        
        return intersecting_levels
    
    # 辅助函数：隐藏靠近坐标轴/边框的等高线标签，避免数值与轴线/边框交叉
    def _hide_labels_near_axes(texts, axis, margin_axes: float = 0.05, margin_data_y: float = 5.0):
        """
        将靠近坐标轴/边框的等高线标签隐藏，防止数字与坐标轴或边框交叉。
        
        参数:
            texts: clabel返回的文本对象列表
            axis: matplotlib axes对象
            margin_axes: 以坐标轴归一化坐标为单位的边缘留白（0-1），默认5%（增大以更严格）
            margin_data_y: 以数据坐标为单位的y轴下边缘留白（用于横轴检测），默认5.0
        """
        if not texts:
            return
        
        # 获取当前坐标轴的数据范围
        xlim = axis.get_xlim()
        ylim = axis.get_ylim()
        
        for txt in texts:
            xdata, ydata = txt.get_position()
            
            # 方法1：使用axes坐标检测（适用于所有边缘）
            x_disp, y_disp = axis.transData.transform((xdata, ydata))
            x_axes, y_axes = axis.transAxes.inverted().transform((x_disp, y_disp))
            
            # 方法2：使用数据坐标检测（特别针对横轴，因为y轴下限是2）
            # 如果标签的y坐标接近y轴下限，直接隐藏
            near_bottom = ydata < (ylim[0] + margin_data_y)
            near_top = ydata > (ylim[1] - margin_data_y)
            near_left = xdata < (xlim[0] + (xlim[1] - xlim[0]) * margin_axes)
            near_right = xdata > (xlim[1] - (xlim[1] - xlim[0]) * margin_axes)
            
            # 使用axes坐标检测（适用于所有边缘，更严格）
            near_edge_axes = (
                x_axes < margin_axes
                or x_axes > 1.0 - margin_axes
                or y_axes < margin_axes
                or y_axes > 1.0 - margin_axes
            )
            
            # 如果标签靠近任何边缘，隐藏它
            if near_bottom or near_top or near_left or near_right or near_edge_axes:
                txt.set_visible(False)

    # 绘制等高线：使用自动生成或传入的等高线级别（已经过处理，直接使用）
    if draw_contours and len(contour_levels_to_use) > 0:
        # 直接使用传入的等高线（已经通过auto_contour_levels生成，或在视觉空间中均匀分布）
        filtered_levels = contour_levels_to_use
        
        # 先对等高线级别去重（考虑浮点数精度，使用相对容差）
        filtered_levels = np.sort(filtered_levels)
        # 使用相对容差去重：如果两个值的相对差异小于1e-6，认为是重复的
        if len(filtered_levels) > 1:
            unique_mask = np.ones(len(filtered_levels), dtype=bool)
            for i in range(1, len(filtered_levels)):
                # 计算相对差异
                if abs(filtered_levels[i]) > 1e-10:
                    rel_diff = abs(filtered_levels[i] - filtered_levels[i-1]) / abs(filtered_levels[i])
                else:
                    rel_diff = abs(filtered_levels[i] - filtered_levels[i-1])
                # 如果相对差异太小，认为是重复的
                if rel_diff < 1e-6:
                    unique_mask[i] = False
            filtered_levels = filtered_levels[unique_mask]
        
        # 定义智能格式化函数，根据数值范围自动调整精度，确保格式化后不重复
        def format_contour_label(x, all_levels):
            """格式化等高线标签，根据数值范围自动调整精度，确保不重复，并去掉末尾的0"""
            # 辅助函数：去掉末尾的0和小数点（如果小数点后全是0）
            def remove_trailing_zeros(s):
                """去掉字符串末尾的0和小数点"""
                if '.' in s:
                    s = s.rstrip('0').rstrip('.')
                return s
            
            # 计算所有值的范围，确定需要的小数位数
            if len(all_levels) > 1:
                min_val = np.min(np.abs(all_levels[all_levels != 0])) if np.any(all_levels != 0) else abs(x)
                max_val = np.max(np.abs(all_levels))
                # 根据数值范围确定精度
                if max_val >= 10:
                    # 大数值，使用整数或1位小数
                    if abs(x - round(x)) > 0.05:
                        formatted = f"{x:.1f}"
                    else:
                        formatted = f"{int(round(x))}"
                elif max_val >= 1:
                    # 中等数值，使用1-2位小数
                    if min_val < 0.1:
                        formatted = f"{x:.2f}"
                    else:
                        formatted = f"{x:.1f}"
                elif max_val >= 0.1:
                    # 小数值，使用2-3位小数
                    if min_val < 0.01:
                        formatted = f"{x:.3f}"
                    else:
                        formatted = f"{x:.2f}"
                else:
                    # 很小数值，使用3-4位小数
                    if min_val < 0.001:
                        formatted = f"{x:.4f}"
                    else:
                        formatted = f"{x:.3f}"
                # 去掉末尾的0
                return remove_trailing_zeros(formatted)
            else:
                # 只有一个值，使用%.2g格式，然后去掉末尾的0
                formatted = f"{x:.2g}"
                return remove_trailing_zeros(formatted)
        
        # 检测格式化后是否有重复标签，如果有则移除重复的等高线
        # 使用字典记录格式化后的标签到原始值的映射
        formatted_to_level = {}
        unique_levels = []
        level_to_formatted = {}  # 记录每个level对应的格式化字符串
        
        for level in filtered_levels:
            formatted = format_contour_label(level, filtered_levels)
            if formatted not in formatted_to_level:
                # 没有重复，添加
                formatted_to_level[formatted] = level
                level_to_formatted[level] = formatted
                unique_levels.append(level)
            else:
                # 格式化后重复，比较两个值，保留更"整齐"的值
                existing_level = formatted_to_level[formatted]
                # 计算哪个值更接近整数（更整齐）
                existing_dist_to_int = abs(existing_level - round(existing_level))
                current_dist_to_int = abs(level - round(level))
                
                # 如果当前值更接近整数，替换
                if current_dist_to_int < existing_dist_to_int:
                    # 从unique_levels中移除旧值，添加新值
                    unique_levels = [l for l in unique_levels if l != existing_level]
                    unique_levels.append(level)
                    formatted_to_level[formatted] = level
                    level_to_formatted[level] = formatted
                    # 移除旧值的映射
                    if existing_level in level_to_formatted:
                        del level_to_formatted[existing_level]
                # 否则保留原有值，跳过当前值（不添加）
        
        # 重新排序
        filtered_levels = np.sort(np.array(unique_levels))
        
        # 限制等高线数量最多为5条
        if len(filtered_levels) > 5:
            # 如果包含0，保留0，然后从其他值中均匀选择
            if 0.0 in filtered_levels:
                non_zero_levels = filtered_levels[filtered_levels != 0.0]
                # 从非零值中选择4条（加上0共5条）
                if len(non_zero_levels) > 4:
                    # 均匀选择4条
                    indices = np.linspace(0, len(non_zero_levels) - 1, 4, dtype=int)
                    selected_non_zero = non_zero_levels[indices]
                    filtered_levels = np.sort(np.concatenate([[0.0], selected_non_zero]))
                else:
                    filtered_levels = np.sort(np.concatenate([[0.0], non_zero_levels]))
            else:
                # 没有0，均匀选择5条
                indices = np.linspace(0, len(filtered_levels) - 1, 5, dtype=int)
                filtered_levels = filtered_levels[indices]
        
        # 创建最终的格式化函数（使用已确定的格式化字符串，确保一致性）
        # 使用闭包捕获 filtered_levels 和 level_to_formatted
        def format_contour_label_final(x):
            """最终格式化函数，使用已确定的格式化字符串（已去掉末尾的0）"""
            # 首先检查是否在映射中
            if x in level_to_formatted:
                return level_to_formatted[x]
            else:
                # 如果不在映射中（可能是在重新绘制时新增的级别），使用智能格式化
                # 尝试找到最接近的已格式化级别
                if len(level_to_formatted) > 0:
                    closest_level = min(level_to_formatted.keys(), key=lambda l: abs(l - x))
                    # 如果非常接近（相对差异小于1%），使用相同的格式化
                    if abs(x) > 1e-10:
                        rel_diff = abs(x - closest_level) / abs(x)
                    else:
                        rel_diff = abs(x - closest_level)
                    if rel_diff < 0.01:
                        return level_to_formatted[closest_level]
                # 否则使用默认格式化（会自动去掉末尾的0）
                return format_contour_label(x, filtered_levels)
        
        # 绘制筛选后的等高线：先绘制所有等高线，然后检测重合并移除
        if len(filtered_levels) > 0:
            # 先临时绘制所有等高线以获取路径信息，用于确定颜色
            # 使用不可见的等高线（linewidths=0.01，几乎不可见）来获取路径
            temp_CS = ax.contour(X, Y, Z, levels=filtered_levels, colors="black", linewidths=0.01)
            # 获取每条等高线的颜色（基于背景色亮度）
            level_colors = get_contour_colors_by_background(temp_CS, X, Y, Z, cmap, norm)
            # 清除临时等高线
            for collection in temp_CS.collections:
                try:
                    collection.remove()
                except:
                    pass
            
            # 先绘制所有等高线（包括0等高线，不需要重复绘制）
            # 如果包含0等高线，使用加粗线条
            all_contour_objects = []  # 存储所有等高线对象
            all_text_objects = []  # 存储所有标签对象
            all_levels_map = {}  # 存储标签到等高线值的映射
            
            if 0.0 in filtered_levels:
                # 分别绘制0等高线（加粗）和其他等高线
                non_zero_levels = filtered_levels[filtered_levels != 0.0]
                if len(non_zero_levels) > 0:
                    # 为每条非零等高线使用动态颜色
                    for level in non_zero_levels:
                        color = level_colors.get(level, 'black')  # 默认黑色
                        CS_level = ax.contour(X, Y, Z, levels=[level], colors=color, linewidths=2)
                        all_contour_objects.append(CS_level)
                        texts_level = ax.clabel(
                            CS_level,
                            inline=True,
                            fontsize=font_size,
                            fmt=format_contour_label_final,
                            colors=color,
                        )
                        if texts_level:
                            all_text_objects.extend(texts_level)
                            # 建立标签到等高线值的映射
                            for txt in texts_level:
                                try:
                                    txt_value = float(txt.get_text())
                                    all_levels_map[txt] = level
                                except (ValueError, TypeError):
                                    all_levels_map[txt] = level
                # 绘制0等高线（加粗，使用动态颜色）
                zero_color = level_colors.get(0.0, 'black')  # 默认黑色
                CS_zero = ax.contour(X, Y, Z, levels=[0.0], colors=zero_color, linewidths=3)
                all_contour_objects.append(CS_zero)
                texts_zero = ax.clabel(
                    CS_zero,
                    inline=True,
                    fontsize=font_size,
                    fmt=format_contour_label_final,
                    colors=zero_color,
                )
                if texts_zero:
                    all_text_objects.extend(texts_zero)
                    for txt in texts_zero:
                        all_levels_map[txt] = 0.0
            else:
                # 没有0等高线，正常绘制所有等高线，每条使用动态颜色
                for level in filtered_levels:
                    color = level_colors.get(level, 'black')  # 默认黑色
                    CS_level = ax.contour(X, Y, Z, levels=[level], colors=color, linewidths=2)
                    all_contour_objects.append(CS_level)
                    texts_level = ax.clabel(
                        CS_level,
                        inline=True,
                        fontsize=font_size,
                        fmt=format_contour_label_final,
                        colors=color,
                    )
                    if texts_level:
                        all_text_objects.extend(texts_level)
                        # 建立标签到等高线值的映射
                        for txt in texts_level:
                            try:
                                txt_value = float(txt.get_text())
                                all_levels_map[txt] = level
                            except (ValueError, TypeError):
                                all_levels_map[txt] = level
            
            # 检测等高线路径是否相交
            intersecting_levels = []
            for contour_obj in all_contour_objects:
                intersecting = _detect_contour_intersections(contour_obj)
                intersecting_levels.extend(intersecting)
            intersecting_levels = list(set(intersecting_levels))
            
            # 检测重合的等高线标签
            if all_text_objects:
                # 需要移除的等高线值（因为标签与边框重合）
                overlapping_with_border, overlapping_with_labels = _detect_overlapping_labels(all_text_objects, ax, all_levels_map)
                
                # 合并需要移除的等高线值（包括相交的等高线）
                levels_to_remove = set(overlapping_with_border) | set(overlapping_with_labels) | set(intersecting_levels)
                
                # 如果检测到需要移除的等高线，重新绘制（移除重合的等高线）
                # 但至少保留最小值和最大值等高线
                if levels_to_remove:
                    min_level = filtered_levels[0] if len(filtered_levels) > 0 else None
                    max_level = filtered_levels[-1] if len(filtered_levels) > 0 else None
                    
                    # 从levels_to_remove中排除最小值和最大值
                    levels_to_remove = set(level for level in levels_to_remove if level != min_level and level != max_level)
                    
                    # 从filtered_levels中移除重合的等高线
                    filtered_levels_new = filtered_levels[~np.isin(filtered_levels, list(levels_to_remove))]
                    
                    # 确保最小值和最大值被保留
                    if min_level is not None and min_level not in filtered_levels_new:
                        filtered_levels_new = np.concatenate([[min_level], filtered_levels_new])
                    if max_level is not None and max_level not in filtered_levels_new:
                        filtered_levels_new = np.concatenate([filtered_levels_new, [max_level]])
                    filtered_levels_new = np.sort(filtered_levels_new)
                    
                    # 清除当前等高线和标签
                    # 等高线集合需要从axes中移除
                    for collection in all_contour_objects:
                        try:
                            collection.remove()
                        except:
                            pass
                    # 文本对象也需要移除
                    for txt in all_text_objects:
                        try:
                            txt.remove()
                        except:
                            pass
                    
                    # 重新绘制（只绘制不重合的等高线，使用动态颜色）
                    if len(filtered_levels_new) > 0:
                        if 0.0 in filtered_levels_new:
                            non_zero_levels_new = filtered_levels_new[filtered_levels_new != 0.0]
                            if len(non_zero_levels_new) > 0:
                                # 为每条等高线使用动态颜色
                                for level in non_zero_levels_new:
                                    color = level_colors.get(level, 'black')
                                    CS_level = ax.contour(X, Y, Z, levels=[level], colors=color, linewidths=2)
                                    texts_level = ax.clabel(
                                        CS_level,
                                        inline=True,
                                        fontsize=font_size,
                                        fmt=format_contour_label_final,
                                        colors=color,
                                    )
                                    if texts_level:
                                        _hide_labels_near_axes(texts_level, ax)
                            # 绘制0等高线（加粗，使用动态颜色）
                            zero_color = level_colors.get(0.0, 'black')
                            CS_zero = ax.contour(X, Y, Z, levels=[0.0], colors=zero_color, linewidths=3)
                            texts_zero = ax.clabel(
                                CS_zero,
                                inline=True,
                                fontsize=font_size,
                                fmt=format_contour_label_final,
                                colors=zero_color,
                            )
                            if texts_zero:
                                _hide_labels_near_axes(texts_zero, ax)
                        else:
                            # 没有0等高线，正常绘制所有等高线，使用动态颜色
                            for level in filtered_levels_new:
                                color = level_colors.get(level, 'black')
                                CS_level = ax.contour(X, Y, Z, levels=[level], colors=color, linewidths=2)
                                texts_level = ax.clabel(
                                    CS_level,
                                    inline=True,
                                    fontsize=font_size,
                                    fmt=format_contour_label_final,
                                    colors=color,
                                )
                                if texts_level:
                                    _hide_labels_near_axes(texts_level, ax)
                else:
                    # 没有重合，先隐藏靠近边框的标签
                    _hide_labels_near_axes(all_text_objects, ax)
                    
                    # 检测哪些等高线的所有标签都被隐藏了，或者根本没有标签
                    # 建立反向映射：从等高线值到标签列表
                    level_to_texts = {}
                    for txt, level in all_levels_map.items():
                        if level not in level_to_texts:
                            level_to_texts[level] = []
                        level_to_texts[level].append(txt)
                    
                    # 找出所有标签都被隐藏的等高线，以及没有标签的等高线
                    # 但至少保留最小值和最大值等高线（即使标签被隐藏）
                    levels_with_no_visible_labels = []
                    min_level = filtered_levels[0] if len(filtered_levels) > 0 else None
                    max_level = filtered_levels[-1] if len(filtered_levels) > 0 else None
                    
                    for level in filtered_levels:
                        # 跳过最小值和最大值，即使它们的标签被隐藏也要保留
                        if level == min_level or level == max_level:
                            continue
                        
                        if level in level_to_texts:
                            # 有标签，检查是否所有标签都被隐藏
                            texts = level_to_texts[level]
                            has_visible = any(txt.get_visible() for txt in texts)
                            if not has_visible:
                                levels_with_no_visible_labels.append(level)
                        else:
                            # 没有标签（clabel没有为这个等高线生成标签），也应该移除
                            levels_with_no_visible_labels.append(level)
                    
                    # 如果有等高线的所有标签都被隐藏或没有标签，移除这些等高线并重新绘制
                    # 但至少保留最小值和最大值
                    if levels_with_no_visible_labels:
                        # 从filtered_levels中移除没有可见标签的等高线
                        filtered_levels_new = filtered_levels[~np.isin(filtered_levels, levels_with_no_visible_labels)]
                        
                        # 确保最小值和最大值被保留
                        if min_level is not None and min_level not in filtered_levels_new:
                            filtered_levels_new = np.concatenate([[min_level], filtered_levels_new])
                        if max_level is not None and max_level not in filtered_levels_new:
                            filtered_levels_new = np.concatenate([filtered_levels_new, [max_level]])
                        filtered_levels_new = np.sort(filtered_levels_new)
                        
                        # 清除当前等高线和标签
                        for collection in all_contour_objects:
                            try:
                                collection.remove()
                            except:
                                pass
                        for txt in all_text_objects:
                            try:
                                txt.remove()
                            except:
                                pass
                        
                        # 重新绘制（只绘制有可见标签的等高线，使用动态颜色）
                        if len(filtered_levels_new) > 0:
                            if 0.0 in filtered_levels_new:
                                non_zero_levels_new = filtered_levels_new[filtered_levels_new != 0.0]
                                if len(non_zero_levels_new) > 0:
                                    # 为每条等高线使用动态颜色
                                    for level in non_zero_levels_new:
                                        color = level_colors.get(level, 'black')
                                        CS_level = ax.contour(X, Y, Z, levels=[level], colors=color, linewidths=2)
                                        texts_level = ax.clabel(
                                            CS_level,
                                            inline=True,
                                            fontsize=font_size,
                                            fmt=format_contour_label_final,
                                            colors=color,
                                        )
                                        if texts_level:
                                            _hide_labels_near_axes(texts_level, ax)
                                # 绘制0等高线（加粗，使用动态颜色）
                                zero_color = level_colors.get(0.0, 'black')
                                CS_zero = ax.contour(X, Y, Z, levels=[0.0], colors=zero_color, linewidths=3)
                                texts_zero = ax.clabel(
                                    CS_zero,
                                    inline=True,
                                    fontsize=font_size,
                                    fmt=format_contour_label_final,
                                    colors=zero_color,
                                )
                                if texts_zero:
                                    _hide_labels_near_axes(texts_zero, ax)
                            else:
                                # 没有0等高线，正常绘制所有等高线，使用动态颜色
                                for level in filtered_levels_new:
                                    color = level_colors.get(level, 'black')
                                    CS_level = ax.contour(X, Y, Z, levels=[level], colors=color, linewidths=2)
                                    texts_level = ax.clabel(
                                        CS_level,
                                        inline=True,
                                        fontsize=font_size,
                                        fmt=format_contour_label_final,
                                        colors=color,
                                    )
                                    if texts_level:
                                        _hide_labels_near_axes(texts_level, ax)

    # 与原图一致的显示范围（对数轴，单位：Ohm，对应1nOhm到100nOhm）
    ax.set_xlim(1e-9, 100e-9)  # 1nOhm到100nOhm，单位转换为Ohm
    ax.set_ylim(2, 200)

    # 移除所有刻度值和标签（与 parasitic heatmap 完全一致）
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlabel("")
    ax.set_ylabel("")

    # 可选：标注 baseline A 点（星标，白色填充+黑边，保证在深色上也可见）
    if baseline_point is not None:
        bNpw, bR = baseline_point
        bx = bR  # 使用原始值，对数轴会自动处理
        by = bNpw
        ax.scatter([bx], [by], marker="*", s=180, c="white", edgecolors="black", linewidths=1.5, zorder=10)

    fig.subplots_adjust(bottom=0.1, right=0.95, left=0.05, top=0.9)

    # 检测哪些等高线实际显示在图上
    displayed_levels = []
    displayed_levels_with_labels = []
    
    # 收集所有contour集合（QuadContourSet或LineCollection）
    for collection in ax.collections:
        try:
            # 检查是否是contour集合
            # QuadContourSet有levels属性
            if hasattr(collection, 'levels'):
                levels = collection.levels
                if levels is not None:
                    if isinstance(levels, (list, np.ndarray)):
                        displayed_levels.extend(levels)
                    else:
                        displayed_levels.append(levels)
            # 或者检查是否有_levels属性
            elif hasattr(collection, '_levels'):
                levels = collection._levels
                if levels is not None:
                    if isinstance(levels, (list, np.ndarray)):
                        displayed_levels.extend(levels)
                    else:
                        displayed_levels.append(levels)
        except Exception:
            pass
    
    # 去重并排序
    displayed_levels = np.unique(displayed_levels)
    displayed_levels = np.sort(displayed_levels)
    
    # 检测哪些等高线有可见的标签
    # 方法1：从ax.texts中查找（clabel创建的文本对象）
    for txt in ax.texts:
        if txt.get_visible():
            try:
                # 尝试从标签文本中解析等高线值
                txt_text = txt.get_text().strip()
                # 移除可能的格式化字符
                txt_text = txt_text.replace('$', '').replace('\\', '')
                txt_value = float(txt_text)
                # 找到最接近的等高线值
                if len(displayed_levels) > 0:
                    closest_level = min(displayed_levels, key=lambda x: abs(x - txt_value))
                    # 允许较大的误差，因为格式化可能导致数值变化
                    tolerance = max(abs(closest_level) * 0.2, 0.01)
                    if abs(closest_level - txt_value) < tolerance:
                        if closest_level not in displayed_levels_with_labels:
                            displayed_levels_with_labels.append(closest_level)
            except (ValueError, TypeError):
                pass
    
    # 方法2：如果displayed_levels为空，尝试从contour集合的路径中推断
    if len(displayed_levels) == 0:
        # 检查是否有contour路径
        for collection in ax.collections:
            try:
                if hasattr(collection, 'get_paths'):
                    paths = collection.get_paths()
                    if len(paths) > 0:
                        # 如果有路径，说明至少有一些等高线被绘制了
                        # 但我们无法直接从路径获取级别，所以使用原始级别
                        displayed_levels = contour_levels_to_use.copy()
                        break
            except Exception:
                pass
    
    displayed_levels_with_labels = np.sort(displayed_levels_with_labels)

    # 根据value_col生成不同的文件名，区分普通图、全局基准图和相对于最小值的图
    # 注意：相对于最小值的图保留为 SVG，便于后续进行矢量拼接；其余仍为 PDF。
    suffix_str = f"_{filename_suffix}" if filename_suffix else ""
    if "global" in value_col.lower():
        plot_path = output_dir / f"delta_lcoe_global_heatmap_{scenario}_{label}{suffix_str}.{cfg.PLOT_FORMAT}"
    elif "min" in value_col.lower():
        plot_path = output_dir / f"delta_lcoe_min_heatmap_{scenario}_{label}{suffix_str}.{cfg.PLOT_FORMAT}"
    else:
        plot_path = output_dir / f"delta_lcoe_heatmap_{scenario}_{label}{suffix_str}.{cfg.PLOT_FORMAT}"
    plt.savefig(plot_path, dpi=cfg.PLOT_DPI, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    
    # 创建readme.txt文件，记录绘图信息
    readme_path = output_dir / "readme.txt"
    with open(readme_path, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"图片文件: {plot_path.name}\n")
        f.write(f"场景: {scenario}\n")
        f.write(f"温度: {temperature_K}K\n")
        if coolant:
            f.write(f"制冷剂: {coolant}\n")
        f.write(f"\n坐标轴数据安排:\n")
        f.write(f"  横轴(X轴): R_joint (接头电阻), 单位: Ohm\n")
        f.write(f"    - 绘图数据范围: {X.min():.6e} ~ {X.max():.6e} Ohm\n")
        f.write(f"    - 显示方式: 对数刻度 (log scale)\n")
        f.write(f"    - 显示范围: 1e-9 ~ 100e-9 Ohm, 对应 1nOhm ~ 100nOhm\n")
        # 子图输出用于后续SVG拼接：子图本身会隐藏刻度；这里记录推荐显示刻度
        f.write(f"    - 刻度(推荐, nΩ): 1, 10, 100\n")
        f.write(f"    - 绘图网格: 使用对数均匀分布的数组 (R_JOINT_SCAN_VALUES_PLOT)\n")
        f.write(f"    - 计算网格: 使用原始整数数组 (R_JOINT_SCAN_VALUES)\n")
        f.write(f"  纵轴(Y轴): Npw (并联绕组数)\n")
        f.write(f"    - 数据范围: {Y.min():.1f} ~ {Y.max():.1f}\n")
        f.write(f"    - 显示范围: 2 ~ 200\n")
        try:
            y_ticks = getattr(cfg, "NPW_SCAN_VALUES", None)
            if y_ticks is not None:
                y_ticks_list = [int(v) for v in np.array(y_ticks).astype(float).tolist()]
                f.write(f"    - 刻度(推荐): {', '.join(map(str, y_ticks_list))}\n")
        except Exception:
            pass
        f.write(f"\n数据值列: {value_col}\n")
        if baseline_point:
            f.write(f"基准点: Npw={baseline_point[0]}, R_joint={baseline_point[1]} Ohm\n")
        if len(contour_levels_to_use) > 0:
            f.write(f"等高线级别: {', '.join([f'{l:.3g}' for l in contour_levels_to_use])}\n")
        f.write(f"{'='*60}\n")



def save_delta_lcoe_heatmap_colorbar(
    output_dir: Path,
    cmap: str,
    norm: mpl.colors.Normalize,
    label: str = r"$\Delta$LCOE ($/\mathrm{MWh}$)",
    data_min: Optional[float] = None,  # 实际数据的最小值（用于非对称刻度）
    data_max: Optional[float] = None,   # 实际数据的最大值（用于非对称刻度）
    is_global_ref: bool = False,  # 是否为全局基准colorbar
    filename_suffix: str = ""  # 可选的文件名后缀，用于区分不同配色方案
):
    """
    单独生成并保存 ΔLCOE 的共享 colorbar（所有子图共享）。
    版式/字体与 save_parasitic_heatmap_colorbar 一致。
    注意：数据集可能不是正负对称的，需要根据实际数据范围生成刻度。
    """
    setup_plot_style()

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    fig_cbar = plt.figure(figsize=(0.5, 4))
    ax_cbar = fig_cbar.add_axes([0.1, 0.15, 0.3, 0.7])
    cbar = fig_cbar.colorbar(sm, cax=ax_cbar)

    # tick：使用固定的刻度列表
    vmin = getattr(norm, "vmin", None)
    vmax = getattr(norm, "vmax", None)
    
    # 如果提供了data_min和data_max，优先使用它们来确定刻度范围
    if data_min is not None and data_max is not None and np.isfinite(data_min) and np.isfinite(data_max):
        # 使用提供的data_min和data_max来确定刻度范围
        tick_vmin = data_min
        tick_vmax = data_max
    else:
        tick_vmin = vmin
        tick_vmax = vmax
    
    # 根据label判断单位，如果是CNY/kWh，需要转换刻度
    is_cny_unit = "CNY" in label or "cny" in label.lower()
    USD_TO_CNY_EXCHANGE_RATE = 7.2  # 1美元 = 7.2人民币
    
    # 如果是CNY/kWh单位，使用CNY版本的固定刻度列表
    if is_cny_unit:
        # 使用CNY/kWh单位的固定刻度列表
        fixed_ticks = getattr(cfg, 'LCOE_CONTOUR_FIXED_LEVELS_CNY', 
                              np.array([0, 0.001, 0.002, 0.003, 0.004, 0.004, 0.005, 0.01, 0.02, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6]))
    else:
        # 固定的刻度列表（美元/MWh单位）
        fixed_ticks = getattr(cfg, 'LCOE_CONTOUR_FIXED_LEVELS', 
                              np.array([-40, -20, -10, -5, -3, -2, -1, -0.5, -0.2, -0.1, -0.05, 
                                       0, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 40, 65, 80, 100, 120, 140, 160, 180, 200]))
    
    # 过滤掉超出范围的刻度
    if (tick_vmin is not None) and (tick_vmax is not None) and np.isfinite(tick_vmin) and np.isfinite(tick_vmax):
        ticks = fixed_ticks[(fixed_ticks >= tick_vmin) & (fixed_ticks <= tick_vmax)]
    else:
        ticks = fixed_ticks
        
        # 设置刻度，根据值的大小选择合适的格式，禁用科学计数法
        if len(ticks) > 0:
            cbar.set_ticks(ticks)
            # 对于整数使用整数格式，对于小数使用小数格式
            # 使用普通数字格式，不使用科学计数法
            def format_tick(x, pos):
                # 使用普通格式，不使用科学计数法
                # 处理所有数值范围，确保不使用科学计数法
                abs_x = abs(x)
                if abs_x == 0.0:
                    return "0"
                elif abs_x < 0.0001:
                    # 非常小的数值，使用5位小数
                    return f"{x:.5f}"
                elif abs_x < 0.001:
                    # 小于0.001的数值，使用4位小数
                    return f"{x:.4f}"
                elif abs_x < 0.01:
                    # 小于0.01的数值，使用3位小数
                    return f"{x:.3f}"
                elif abs_x < 0.1:
                    # 小于0.1的数值，使用2位小数
                    return f"{x:.2f}"
                elif abs_x < 1:
                    # 小于1的数值，使用2位小数
                    return f"{x:.2f}"
                else:
                    # 大于等于1的数值
                    if abs_x >= 1 and abs(x - int(x)) < 1e-10:
                        # 整数，直接显示整数
                        return str(int(x))
                    else:
                        # 非整数，使用2位小数
                        return f"{x:.2f}"
            
            # 使用FuncFormatter格式化刻度（FuncFormatter返回的字符串不会使用科学计数法）
            formatter = ticker.FuncFormatter(format_tick)
            cbar.ax.yaxis.set_major_formatter(formatter)
            
            # 明确禁用科学计数法和偏移量
            # 隐藏偏移量文本（如果存在）
            cbar.ax.yaxis.offsetText.set_visible(False)
            
            # 确保刻度标签不使用科学计数法
            # 通过直接设置每个刻度标签的文本和属性
            # 注意：需要在formatter设置之后，但在保存之前执行
            def update_tick_labels():
                """更新所有刻度标签，确保不使用科学计数法"""
                for tick in cbar.ax.yaxis.get_major_ticks():
                    tick_value = tick.get_loc()
                    # 使用我们的格式化函数直接设置标签文本
                    tick_label = format_tick(tick_value, None)
                    # 设置标签文本
                    tick.label1.set_text(tick_label)
                    # 确保标签可见
                    tick.label1.set_visible(True)
            
            # 在保存之前更新标签
            # 使用fig_cbar.canvas.draw_idle()来触发更新，但不阻塞
            try:
                fig_cbar.canvas.draw_idle()
                update_tick_labels()
            except:
                # 如果draw_idle失败，直接更新标签
                update_tick_labels()
    
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label(label, fontsize=18, labelpad=15)
    
    # 设置colorbar只在一侧显示刻度（左侧）
    cbar.ax.yaxis.set_tick_params(labelleft=True, labelright=False)

    # 在保存之前，强制更新所有刻度标签，确保不使用科学计数法
    if len(ticks) > 0:
        # 重新定义format_tick函数（确保在作用域内）
        def format_tick_final(x, pos):
            """格式化刻度标签，不使用科学计数法"""
            abs_x = abs(x)
            if abs_x == 0.0:
                return "0"
            elif abs_x < 0.0001:
                return f"{x:.5f}"
            elif abs_x < 0.001:
                return f"{x:.4f}"
            elif abs_x < 0.01:
                return f"{x:.3f}"
            elif abs_x < 0.1:
                return f"{x:.2f}"
            elif abs_x < 1:
                return f"{x:.2f}"
            else:
                if abs_x >= 1 and abs(x - int(x)) < 1e-10:
                    return str(int(x))
                else:
                    return f"{x:.2f}"
        
        # 强制绘制一次，让formatter生效
        fig_cbar.canvas.draw()
        
        # 再次更新所有刻度标签，直接设置文本
        for tick in cbar.ax.yaxis.get_major_ticks():
            tick_value = tick.get_loc()
            # 使用格式化函数生成标签文本
            tick_label = format_tick_final(tick_value, None)
            # 直接设置左侧标签文本，覆盖任何自动格式化
            tick.label1.set_text(tick_label)
            # 确保左侧标签可见
            tick.label1.set_visible(True)
            # 隐藏右侧标签
            tick.label2.set_visible(False)
            # 禁用LaTeX渲染（如果可能）
            try:
                tick.label1.set_usetex(False)
            except:
                pass
        
        # 再次绘制以确保更改生效
        fig_cbar.canvas.draw()
    
    # 根据is_global_ref参数或label判断colorbar类型（使用英文关键词，避免中文字符）
    suffix_str = f"_{filename_suffix}" if filename_suffix else ""
    if is_global_ref or "Global Ref" in label or "global" in label.lower():
        cbar_path = output_dir / f"delta_lcoe_global_heatmap_colorbar{suffix_str}.{cfg.PLOT_FORMAT}"
    elif "min" in label.lower() or "relative to scenario minimum" in label.lower():
        cbar_path = output_dir / f"delta_lcoe_min_heatmap_colorbar{suffix_str}.{cfg.PLOT_FORMAT}"
    else:
        cbar_path = output_dir / f"delta_lcoe_heatmap_colorbar{suffix_str}.{cfg.PLOT_FORMAT}"
    
    plt.savefig(cbar_path, dpi=cfg.PLOT_DPI, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig_cbar)
