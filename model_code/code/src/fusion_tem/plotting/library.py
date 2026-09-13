# plot_library.py (统一绘图库 - 英文版)

import os as _os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import matplotlib as mpl
from matplotlib import patheffects
from pathlib import Path
from fusion_tem import device as cfg
from typing import Optional, Tuple

# =============================================================================
# 辅助函数：根据底色深浅设置等高线颜色
# =============================================================================

def calculate_luminance(rgb: Tuple[float, float, float]) -> float:
    """
    计算 RGB 颜色的亮度（luminance）。
    
    使用相对亮度公式：L = 0.299*R + 0.587*G + 0.114*B
    返回值范围：0（最暗）到 1（最亮）
    
    Args:
        rgb: RGB 颜色元组，值范围 0-1
    
    Returns:
        亮度值（0-1）
    """
    r, g, b = rgb
    return 0.299 * r + 0.587 * g + 0.114 * b


def get_contour_color_from_background(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    contour_level: float,
    cmap: mpl.colors.Colormap,
    norm: mpl.colors.Normalize,
    threshold: float = 0.5
) -> str:
    """
    根据等高线路径上的背景色亮度，确定等高线应该使用的颜色。
    
    方法：
    1. 在等高线路径上采样多个点
    2. 获取每个点的 Z 值，通过 norm 和 cmap 获取背景色
    3. 计算背景色的平均亮度
    4. 如果平均亮度 < threshold，返回 'white'（深色背景用白色等高线）
    5. 否则返回 'black'（浅色背景用黑色等高线）
    
    Args:
        X: X 坐标网格
        Y: Y 坐标网格
        Z: Z 值网格
        contour_level: 等高线级别
        cmap: 颜色映射
        norm: 归一化对象
        threshold: 亮度阈值（默认 0.5）
    
    Returns:
        等高线颜色字符串：'white' 或 'black'
    """
    # 在等高线路径上采样点
    # 使用 Z 值接近 contour_level 的点作为采样点
    mask = np.abs(Z - contour_level) < (np.nanmax(Z) - np.nanmin(Z)) * 0.1
    
    if not mask.any():
        # 如果没有找到接近的点，使用所有有效点
        mask = np.isfinite(Z)
    
    if not mask.any():
        return 'black'  # 默认返回黑色
    
    # 获取采样点的 Z 值
    sampled_Z = Z[mask]
    
    # 通过 norm 归一化
    normalized_values = norm(sampled_Z)
    
    # 处理超出范围的值
    normalized_values = np.clip(normalized_values, 0, 1)
    
    # 通过 cmap 获取颜色
    colors = cmap(normalized_values)
    
    # 计算所有采样点的平均亮度
    luminances = []
    for color in colors:
        if len(color) >= 3:
            rgb = (color[0], color[1], color[2])
            lum = calculate_luminance(rgb)
            luminances.append(lum)
    
    if not luminances:
        return 'black'
    
    avg_luminance = np.mean(luminances)
    
    # 根据平均亮度返回颜色
    if avg_luminance < threshold:
        return 'white'  # 深色背景用白色等高线
    else:
        return 'black'  # 浅色背景用黑色等高线


def get_contour_colors_for_levels(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    levels: np.ndarray,
    cmap: mpl.colors.Colormap,
    norm: mpl.colors.Normalize,
    threshold: float = 0.3
) -> list:
    """
    为多个等高线级别计算对应的颜色列表。
    
    Args:
        X: X 坐标网格
        Y: Y 坐标网格
        Z: Z 值网格
        levels: 等高线级别数组
        cmap: 颜色映射
        norm: 归一化对象
        threshold: 亮度阈值（默认 0.5）
    
    Returns:
        颜色列表，每个元素对应一个等高线级别
    """
    colors = []
    for level in levels:
        color = get_contour_color_from_background(
            X, Y, Z, level, cmap, norm, threshold
        )
        colors.append(color)
    return colors


# =============================================================================
# 绘图函数 1: 组合图 (成本构成 + 回收周期)
# =============================================================================
font_size = 20
font_size_title = 24
font_size_label = 20
font_size_tick = cfg.HEATMAP_FONT_SIZE_TICK
font_size_legend = 20
font_size_legend_title = 20
font_size_legend_label = 20
font_size_legend_tick = 20

# =============================================================================
# 绘图函数 2: 厂用电热力图
# =============================================================================

def plot_parasitic_heatmap_single(df: pd.DataFrame, output_dir: Path, cmap: str, norm: mpl.colors.Normalize, 
                                   scenario: str, temperature_K: float, filename_suffix: str = "", 
                                   use_stroke: bool = False):
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
        filename_suffix: 可选的文件名后缀，用于区分不同配色方案
    """
    font_size = cfg.HEATMAP_FONT_SIZE
    setup_plot_style()
    
    label = f"{temperature_K}K"
    
    # 创建单个子图
    fig, ax = plt.subplots(1, 1, figsize=cfg.HEATMAP_FIGSIZE_SINGLE)
    
    # 准备数据：先去除重复的 (Npw, R_joint) 组合
    # 如果有重复，取第一个值（或平均值）
    df_clean = df.copy()
    if df_clean.duplicated(subset=['Npw', 'R_joint']).any():
        # 如果有重复，对每个 (Npw, R_joint) 组合取平均值
        df_clean = df_clean.groupby(['Npw', 'R_joint'], as_index=False)['r_parasitic_pct'].mean()
    
    # 准备数据
    pivot = df_clean.pivot(index="Npw", columns="R_joint", values="r_parasitic_pct")
    X, Y = np.meshgrid(pivot.columns, pivot.index)
    Z = pivot.values

    X_log = np.log10(X * 1e9)
    
    
    # 直接使用中心点坐标，使 pcolormesh 和 contour 都使用中心点，二者一致
    # 获取 colormap 对象（支持 config 中的自定义/截取版配色方案名）
    cmap_resolved = cfg.resolve_cmap(cmap) if hasattr(cfg, "resolve_cmap") else cmap
    if isinstance(cmap_resolved, str):
        try:
            cmap_obj = plt.get_cmap(cmap_resolved)
        except ValueError:
            # 如果 matplotlib 不支持，尝试从 seaborn 获取
            try:
                import seaborn as sns
                cmap_obj = sns.color_palette(cmap_resolved, as_cmap=True)
            except (ImportError, ValueError):
                # 如果都失败，使用默认配色
                cmap_obj = plt.get_cmap("viridis")
    else:
        cmap_obj = cmap_resolved
    
    # 绘制热力图（使用中心点坐标，shading="auto"）
    # 这样 pcolormesh 和 contour 都使用中心点，二者完全一致
    pcm = ax.pcolormesh(X_log, Y, Z, cmap=cmap_obj, norm=norm, shading="auto", alpha=cfg.HEATMAP_ALPHA)
    
    # 绘制等高线（使用中心点坐标，即原始数据点坐标）
    levels = {'4.2K': [2.2, 2.5, 3, 4, 6, 10, 20, 40], '10.0K': [1, 1.5, 2, 3, 5, 8, 12, 20], '20.0K': [0.5, 0.6, 0.8, 1, 1.5, 2, 3, 5]}
    contour_levels = levels.get(label, [])
    
    # 绘制等高线
    # 隐藏靠近坐标轴/边框的等高线标签，避免数值与轴线/边框交叉
    def _hide_labels_near_axes(texts, axis, margin_axes: float = 0.05, margin_data_y: float = 5.0):
        """
        将靠近坐标轴/边框的等高线标签隐藏，防止数字与坐标轴或边框交叉。
        
        参数:
            texts: clabel返回的文本对象列表
            axis: matplotlib axes对象
            margin_axes: 以坐标轴归一化坐标为单位的边缘留白（0-1），默认5%
            margin_data_y: 以数据坐标为单位的y轴下边缘留白（用于横轴检测），默认5.0
        
        返回:
            bool: 如果所有标签都被隐藏了，返回True；否则返回False
        """
        if not texts:
            return False
        
        # 获取当前坐标轴的数据范围
        xlim = axis.get_xlim()
        ylim = axis.get_ylim()
        
        hidden_count = 0
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
                hidden_count += 1

        # 如果所有标签都被隐藏了，返回True
        return hidden_count == len(texts)
    
    if len(contour_levels) > 0:
        if use_stroke:
            # 使用描边技术：使用path_effects为黑色等高线添加白色描边
            CS = None
            clabels = []
            for i, level in enumerate(contour_levels):
                # 绘制黑色主线条
                CS_level = ax.contour(X_log, Y, Z, levels=[level], colors='black', linewidths=cfg.HEATMAP_CONTOUR_LINEWIDTH)
                if CS is None:
                    CS = CS_level
                # 关键: 必须先 clabel(inline=True) 在线条上打断、留出间隙放标签，
                # 再加白色描边；否则描边在 clabel 之前会填满 inline 间隙，导致线条压住数字。
                texts_level = ax.clabel(CS_level, inline=True, fontsize=font_size, fmt=cfg.HEATMAP_CONTOUR_FMT_PARASITIC, colors='black')
                # 为等高线集合添加白色描边效果（在 clabel 之后，保留间隙）
                for collection in CS_level.collections:
                    collection.set_path_effects([
                        patheffects.withStroke(
                            linewidth=cfg.CONTOUR_STROKE_LINEWIDTH,
                            foreground=cfg.CONTOUR_STROKE_FOREGROUND,
                            alpha=cfg.CONTOUR_STROKE_ALPHA
                        ),
                        patheffects.Normal()
                    ])
                if texts_level:
                    # 为每个文字标签添加白色描边
                    for txt in texts_level:
                        txt.set_path_effects([
                            patheffects.withStroke(
                                linewidth=cfg.LABEL_STROKE_LINEWIDTH,
                                foreground=cfg.LABEL_STROKE_FOREGROUND,
                                alpha=cfg.LABEL_STROKE_ALPHA
                            ),
                            patheffects.Normal()
                        ])
                    # 立即检查并隐藏靠近边缘的标签
                    all_hidden = _hide_labels_near_axes(texts_level, ax)
                    # 如果这个level的所有标签都被隐藏了，移除对应的等高线
                    if all_hidden:
                        for collection in CS_level.collections:
                            collection.remove()
                    else:
                        clabels.extend(texts_level)
        else:
            # 根据背景色亮度动态选择颜色（浅色背景用黑色，深色背景用白色）
            level_colors_list = get_contour_colors_for_levels(X_log, Y, Z, np.array(contour_levels), cmap_obj, norm)
            
            # 为每条等高线绘制不同颜色
            CS = None
            clabels = []
            for i, level in enumerate(contour_levels):
                color = level_colors_list[i] if i < len(level_colors_list) else 'black'  # 默认使用黑色
                CS_level = ax.contour(X_log, Y, Z, levels=[level], colors=color, linewidths=cfg.HEATMAP_CONTOUR_LINEWIDTH)
                if CS is None:
                    CS = CS_level
                texts_level = ax.clabel(CS_level, inline=True, fontsize=font_size, fmt=cfg.HEATMAP_CONTOUR_FMT_PARASITIC, colors=color)
                if texts_level:
                    # 立即检查并隐藏靠近边缘的标签
                    all_hidden = _hide_labels_near_axes(texts_level, ax)
                    # 如果这个level的所有标签都被隐藏了，移除对应的等高线
                    if all_hidden:
                        for collection in CS_level.collections:
                            collection.remove()
                    else:
                        clabels.extend(texts_level)
    else:
        CS = None
        clabels = None
    

    # 设置坐标轴范围但不显示刻度和标签

    xlim_min = np.min(df["R_joint"])*1e9
    xlim_max = np.max(df["R_joint"])*1e9
    ax.set_xlim(np.log10(xlim_min), np.log10(xlim_max))
    ylim_min = np.min(df["Npw"])
    ylim_max = np.max(df["Npw"])
    ax.set_ylim(ylim_min, ylim_max)
    
    # 移除所有刻度值和标签
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # 移除坐标轴标签
    ax.set_xlabel("")
    ax.set_ylabel("")
    
    # 设置边框（与AF热力图一致）
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(3)
    
    # 保留标题（包含场景和温度信息）
    # ax.set_title(f"{scenario} - {label}")

    # 调整布局（与AF热力图一致：去除白边，让子图完全填满figure）
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    ax.set_position([0, 0, 1, 1]) 
    
    # 保存单个子图（不包含colorbar、坐标轴标签和刻度）
    # 使用与AF热力图相同的保存方式：bbox_inches='tight'，pad_inches=0.0（裁剪空白边缘）
    suffix_str = f"_{filename_suffix}" if filename_suffix else ""
    plot_path = output_dir / f"parasitic_ratio_heatmap_{scenario}_{label}{suffix_str}.{cfg.PLOT_FORMAT}"
    plt.savefig(plot_path, dpi=cfg.PLOT_DPI, bbox_inches='tight', pad_inches=0.0)
    plt.close(fig)
    print(f"Heatmap saved to: {plot_path}")


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
    
    # 支持 config 中的自定义/截取版配色名（如 "YlGnBu_trunc_0p8"）
    cmap_resolved = cfg.resolve_cmap(cmap) if hasattr(cfg, "resolve_cmap") else cmap
    
    # 创建一个新的ScalarMappable对象用于colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap_resolved, norm=norm)
    sm.set_array([])  # 设置空数组
    
    fig_cbar = plt.figure(figsize=cfg.HEATMAP_FIGSIZE_COLORBAR)
    ax_cbar = fig_cbar.add_axes([0.1, 0.15, 0.3, 0.7])
    cbar = fig_cbar.colorbar(sm, cax=ax_cbar, label="Parasitic Power Fraction (%)")
    
    # 自定义 colorbar 刻度
    cbar.set_ticks([2, 4, 10, 20, 50])
    cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
    cbar.ax.tick_params(labelsize=cfg.HEATMAP_FONT_SIZE_COLORBAR)
    cbar.set_label("Parasitic Power Fraction (%)", fontsize=cfg.HEATMAP_FONT_SIZE_COLORBAR, labelpad=15)
    
    # 根据filename_suffix构建文件名
    suffix_str = f"_{filename_suffix}" if filename_suffix else ""
    cbar_path = output_dir / f"parasitic_ratio_heatmap_colorbar{suffix_str}.{cfg.PLOT_FORMAT}"
    plt.savefig(cbar_path, dpi=cfg.PLOT_DPI, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig_cbar)


def plot_availability_heatmap_single(
    rho_turn_vals, npw_vals, Z_af_ref, output_dir: Path, cmap: str,
    norm: mpl.colors.Normalize, scenario: str, temperature_K: float,
    contour_levels, filename_suffix: str = "", use_stroke: bool = False,
    fmt: str | None = None, fname_stem: str = "AF_ref_heatmap",
    smooth_sigma: float = 0.6,
):
    """相对可用度(AF_ref)热力图 —— 采用与 Fig5/6(delta) 一致的 library 渲染风格。

    - 对数 x 轴（匝间电阻率 μΩ·cm²）和线性 y 轴（并绕根数），细网格(260)插值使等高线平滑;
    - 黑等高线 + 白描边(use_stroke), clabel 内联标注, 只隐藏贴边标签但**绝不删线**
      (保住 0.10 这类贴边的语义等高线);
    - 裸面板(无刻度/标签, 黑边框 lw3), 供 9.0 的 af_ref 拼接器加共享轴。

    参数:
        rho_turn_vals: 1D, 匝间电阻率 (μΩ·cm²)
        npw_vals:      1D, 并绕根数
        Z_af_ref:      2D, shape=(len(npw), len(rho)), 相对可用度值
        contour_levels: 固定语义级别(如 [0.1,0.3,0.6,0.9,0.95,0.99])
    """
    from scipy.interpolate import RegularGridInterpolator as _RGI
    setup_plot_style()
    font_size = cfg.HEATMAP_FONT_SIZE
    if fmt is None:
        fmt = getattr(cfg, "HEATMAP_CONTOUR_FMT_AF", "%.2f")
    fig, ax = plt.subplots(1, 1, figsize=cfg.HEATMAP_FIGSIZE_SINGLE)

    x_src = np.log10(np.asarray(rho_turn_vals, dtype=float))   # log10(rho_turn)
    y_src = np.asarray(npw_vals, dtype=float)                  # linear Npw
    Z = np.asarray(Z_af_ref, dtype=float)                      # (npw, rho)

    # 轻高斯平滑(与 Fig5/6 的 _smooth_delta_surface 同理): 压掉粗网格(charge-time 离散步)
    # 造成的低值等高线台阶/折线。NaN 安全(归一化), sigma≈0.8 很轻, 不移动主等高线。
    if smooth_sigma and smooth_sigma > 0:
        from scipy.ndimage import gaussian_filter
        finite = np.isfinite(Z)
        if finite.all():
            Z = gaussian_filter(Z, sigma=smooth_sigma, mode="nearest")
        else:
            w = gaussian_filter(finite.astype(float), sigma=smooth_sigma, mode="nearest")
            Zs = gaussian_filter(np.where(finite, Z, 0.0), sigma=smooth_sigma, mode="nearest")
            Z = np.where(finite, Zs / np.maximum(w, 1e-9), np.nan)

    cmap_obj = cfg.resolve_cmap(cmap) if hasattr(cfg, "resolve_cmap") else cmap
    if isinstance(cmap_obj, str):
        try:
            cmap_obj = plt.get_cmap(cmap_obj)
        except ValueError:
            cmap_obj = plt.get_cmap("viridis")

    # 细网格插值(log-log): 与 Fig5/6 一致, 让等高线平滑(2.9 原为 40, 这里 260)
    fx = np.linspace(x_src.min(), x_src.max(), 260)
    fy = np.linspace(y_src.min(), y_src.max(), 260)
    # 单调三次(PCHIP)张量插值: AF_ref 沿 rho、Npw 均单调 -> C1 光滑且**不过冲**。
    # (线性在非均匀 log 网格的十进制边界处斜率突变->"肘部"折点; cubic 在陡处过冲->波浪; PCHIP 两者都避。)
    from scipy.interpolate import PchipInterpolator as _PCHIP
    _Z1 = np.empty((len(y_src), len(fx)))         # 先沿 rho 插到细 rho
    for _i in range(len(y_src)):
        _Z1[_i] = _PCHIP(x_src, Z[_i], extrapolate=True)(fx)
    Zf = np.empty((len(fy), len(fx)))             # 再沿 Npw 插到细 Npw
    for _j in range(len(fx)):
        Zf[:, _j] = _PCHIP(y_src, _Z1[:, _j], extrapolate=True)(fy)
    FX, FY = np.meshgrid(fx, fy)

    ax.pcolormesh(FX, FY, Zf, cmap=cmap_obj, norm=norm, shading="auto", alpha=cfg.HEATMAP_ALPHA, rasterized=True)  # 栅格化热力图(等高线仍矢量), 避免 260x260 cell 撑爆 SVG(128MB->~百KB)

    # 只隐藏贴边标签(不删线), margin 与 _hide_labels_near_axes 一致(0.05)
    def _hide_label_if_near_edge(txt, axis, margin_axes=0.05):
        xdata, ydata = txt.get_position()
        xd, yd = axis.transData.transform((xdata, ydata))
        xa, ya = axis.transAxes.inverted().transform((xd, yd))
        if xa < margin_axes or xa > 1 - margin_axes or ya < margin_axes or ya > 1 - margin_axes:
            txt.set_visible(False)

    lw = getattr(cfg, "HEATMAP_CONTOUR_LINEWIDTH_AF", cfg.HEATMAP_CONTOUR_LINEWIDTH)
    stroke_lw = getattr(cfg, "CONTOUR_STROKE_LINEWIDTH_AF", cfg.CONTOUR_STROKE_LINEWIDTH)
    for level in contour_levels:
        level_lw = 2.0 * lw if np.isclose(float(level), 0.99) else lw
        CS = ax.contour(FX, FY, Zf, levels=[float(level)], colors="black", linewidths=level_lw)
        texts = ax.clabel(CS, inline=True, fontsize=font_size, fmt=fmt, colors="black")
        if use_stroke:
            # mpl>=3.8: 描边直接加在 ContourSet 上。切勿用 CS.collections —— 该弃用 shim
            # 会隐藏 CS 本体、另挂"影子 LineCollection"渲染, 导致后续 remove()/set_paths()
            # 只作用于本体而画面不变(救援重定位曾因此完全失效)。
            CS.set_path_effects([
                patheffects.withStroke(linewidth=stroke_lw,
                                       foreground=cfg.CONTOUR_STROKE_FOREGROUND,
                                       alpha=cfg.CONTOUR_STROKE_ALPHA),
                patheffects.Normal(),
            ])
            for txt in (texts or []):
                txt.set_path_effects([
                    patheffects.withStroke(linewidth=cfg.LABEL_STROKE_LINEWIDTH,
                                           foreground=cfg.LABEL_STROKE_FOREGROUND,
                                           alpha=cfg.LABEL_STROKE_ALPHA),
                    patheffects.Normal(),
                ])
        for txt in (texts or []):
            _hide_label_if_near_edge(txt, ax)   # 只隐藏标签, 线保留

        # 救援重定位: clabel 自动落点若恰在贴边禁区(如 0.99 线的浅段)会被全部隐藏,
        # 留下"有线无号"。做法: 删掉带旧切口的线与隐藏标签, 重画全新等高线, 然后**手工**
        # 在离四边最远的路径点放标签并在标签处开缺口。
        # (不能用 clabel(manual=...): mpl 的 add_label_near 有 bug —— 标签放在指定点,
        #  inline 切口却切在路径起点附近, 造成"标签压线 + 别处断线"。)
        # 仅当整条线最深点也进不了禁区外(<0.055)才维持隐藏 —— 仍绝不删线。
        if (texts is not None) and len(texts) > 0 and not any(t.get_visible() for t in texts):
            _trans = (ax.transData + ax.transAxes.inverted()).transform
            best_pt, best_d, best_ki = None, 0.0, None
            for _path in CS.get_paths():
                _v = _path.vertices
                if len(_v) < 2:
                    continue
                _va = _trans(_v)
                _d = np.minimum.reduce([_va[:, 0], 1.0 - _va[:, 0], _va[:, 1], 1.0 - _va[:, 1]])
                _k = int(np.argmax(_d))
                if _d[_k] > best_d:
                    best_d, best_pt = float(_d[_k]), (float(_v[_k, 0]), float(_v[_k, 1]))
            if best_pt is not None and best_d >= 0.055:
                CS.remove()               # 移除带旧切口的线(内部会一并移除其 labelTexts, 勿手动先删)
                CS = ax.contour(FX, FY, Zf, levels=[float(level)], colors="black", linewidths=level_lw)
                if use_stroke:
                    CS.set_path_effects([  # 同上: 不可用 CS.collections
                        patheffects.withStroke(linewidth=stroke_lw,
                                               foreground=cfg.CONTOUR_STROKE_FOREGROUND,
                                               alpha=cfg.CONTOUR_STROKE_ALPHA),
                        patheffects.Normal(),
                    ])
                # --- 手工标签 + 手工开缺口(对新线做一次性路径手术) ---
                from matplotlib.path import Path as _MplPath
                _lab = (fmt % float(level)) if isinstance(fmt, str) else str(level)
                # 在最深点找新线上的最近顶点及局部切线角(显示坐标, 保证视觉角度正确)
                _paths_new = CS.get_paths()
                _pd = ax.transData.transform  # data->display
                _ptd = _pd(np.asarray(best_pt))
                _bi, _bk, _bdist = 0, 0, np.inf
                for _pi, _path in enumerate(_paths_new):
                    _vd = _pd(_path.vertices)
                    _dist = np.hypot(_vd[:, 0] - _ptd[0], _vd[:, 1] - _ptd[1])
                    _k = int(np.argmin(_dist))
                    if _dist[_k] < _bdist:
                        _bi, _bk, _bdist = _pi, _k, float(_dist[_k])
                _v = _paths_new[_bi].vertices
                _vd = _pd(_v)
                _k0, _k1 = max(_bk - 2, 0), min(_bk + 2, len(_v) - 1)
                _ang = np.degrees(np.arctan2(_vd[_k1, 1] - _vd[_k0, 1], _vd[_k1, 0] - _vd[_k0, 0]))
                if _ang > 90: _ang -= 180
                if _ang < -90: _ang += 180
                _txt = ax.text(_v[_bk, 0], _v[_bk, 1], _lab, fontsize=font_size, color="black",
                               ha="center", va="center", rotation=_ang, rotation_mode="anchor")
                if use_stroke:
                    _txt.set_path_effects([
                        patheffects.withStroke(linewidth=cfg.LABEL_STROKE_LINEWIDTH,
                                               foreground=cfg.LABEL_STROKE_FOREGROUND,
                                               alpha=cfg.LABEL_STROKE_ALPHA),
                        patheffects.Normal(),
                    ])
                # 实测标签包围盒 -> 沿线删除盒内(+边距)顶点, 插入 MOVETO 缺口
                try:
                    _rend = fig.canvas.get_renderer()
                except AttributeError:
                    fig.canvas.draw()
                    _rend = fig.canvas.get_renderer()
                _bb = _txt.get_window_extent(renderer=_rend)
                _half = 0.5 * float(np.hypot(_bb.width, _bb.height)) * 0.72 + 3.0  # 近似半长+边距
                _dist_all = np.hypot(_vd[:, 0] - _vd[_bk, 0], _vd[:, 1] - _vd[_bk, 1])
                _cut = _dist_all <= _half
                # 只保留与 _bk 连通的那一段 cut(避免路径折返时误删远处顶点)
                _lo = _bk
                while _lo - 1 >= 0 and _cut[_lo - 1]: _lo -= 1
                _hi = _bk
                while _hi + 1 < len(_v) and _cut[_hi + 1]: _hi += 1
                _codes = _paths_new[_bi].codes
                _seg_a, _seg_b = _v[:_lo], _v[_hi + 1:]
                _newv, _newc = [], []
                for _seg in (_seg_a, _seg_b):
                    if len(_seg) >= 2:
                        _newv.append(_seg)
                        _c = np.full(len(_seg), _MplPath.LINETO); _c[0] = _MplPath.MOVETO
                        _newc.append(_c)
                if _newv:
                    _paths_new[_bi] = _MplPath(np.vstack(_newv), np.concatenate(_newc))
                    CS.set_paths(_paths_new)

    ax.set_xlim(x_src.min(), x_src.max())
    ax.set_ylim(y_src.min(), y_src.max())
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(""); ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(3)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    ax.set_position([0, 0, 1, 1])

    suffix_str = f"_{filename_suffix}" if filename_suffix else ""
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / f"{fname_stem}_{scenario}_Top_{temperature_K}K{suffix_str}.{cfg.PLOT_FORMAT}"
    plt.savefig(plot_path, dpi=cfg.PLOT_DPI, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)
    print(f"AF_ref heatmap saved to: {plot_path}")
    return plot_path


def plot_cryo_power_heatmap_single(
    df: pd.DataFrame,
    output_dir: Path,
    cmap: str,
    norm: mpl.colors.Normalize,
    mode_label: str,
    temperature_K: float,
    value_col: str = "cryo_power_MWe",
    filename_suffix: str = "",
):
    """
    为单个运行模式、单个温度绘制制冷功率热力图（单位：MWe）。
    横轴：接头电阻，纵轴：并绕根数。主图不包含 colorbar、坐标轴标签和刻度。

    参数:
        df: 包含 Npw, R_joint 和 value_col 的 DataFrame
        output_dir: 输出目录
        cmap: 颜色映射
        norm: 归一化对象（如 LogNorm）
        mode_label: 模式标签（用于文件名，如 "pulse", "dwell", "static"）
        temperature_K: 温度值（如 4.2, 10.0, 20.0）
        value_col: 数值列名，单位 MWe
        filename_suffix: 可选的文件名后缀
    """
    font_size = cfg.HEATMAP_FONT_SIZE
    setup_plot_style()

    label = f"{temperature_K}K"
    if temperature_K == 10.0 or temperature_K == 20.0:
        label = f"{int(temperature_K)}K"

    fig, ax = plt.subplots(1, 1, figsize=cfg.HEATMAP_FIGSIZE_SINGLE)

    df_clean = df.copy()
    if df_clean.duplicated(subset=["Npw", "R_joint"]).any():
        df_clean = df_clean.groupby(["Npw", "R_joint"], as_index=False)[value_col].mean()

    pivot = df_clean.pivot(index="Npw", columns="R_joint", values=value_col)
    X, Y = np.meshgrid(pivot.columns, pivot.index)
    Z = pivot.values
    X_log = np.log10(X * 1e9)

    cmap_resolved = cfg.resolve_cmap(cmap) if hasattr(cfg, "resolve_cmap") else cmap
    if isinstance(cmap_resolved, str):
        try:
            cmap_obj = plt.get_cmap(cmap_resolved)
        except ValueError:
            try:
                import seaborn as sns
                cmap_obj = sns.color_palette(cmap_resolved, as_cmap=True)
            except (ImportError, ValueError):
                cmap_obj = plt.get_cmap("viridis")
    else:
        cmap_obj = cmap_resolved

    ax.pcolormesh(X_log, Y, Z, cmap=cmap_obj, norm=norm, shading="auto", alpha=cfg.HEATMAP_ALPHA)

    z_min, z_max = np.nanmin(Z), np.nanmax(Z)
    if np.isfinite(z_min) and np.isfinite(z_max) and z_max > z_min:
        finite_pos = Z[np.isfinite(Z) & (Z > 0)]
        if finite_pos.size > 0 and finite_pos.max() > finite_pos.min():
            n_contour = 6
            contour_levels = np.geomspace(finite_pos.min(), finite_pos.max(), n_contour)
            XX, YY = np.meshgrid(X_log[0, :] if X_log.ndim > 1 else X_log, Y[:, 0] if Y.ndim > 1 else Y)
            if XX.shape != Z.shape:
                XX, YY = np.meshgrid(np.unique(X_log), np.unique(Y))
            CS = ax.contour(XX, YY, Z, levels=contour_levels, colors="black", linewidths=cfg.HEATMAP_CONTOUR_LINEWIDTH_CRYO)
            ax.clabel(CS, inline=True, fontsize=font_size, fmt=cfg.HEATMAP_CONTOUR_FMT_CRYO, colors="black")

    xlim_min = np.min(df["R_joint"]) * 1e9
    xlim_max = np.max(df["R_joint"]) * 1e9
    ax.set_xlim(np.log10(xlim_min), np.log10(xlim_max))
    ylim_min = np.min(df["Npw"])
    ylim_max = np.max(df["Npw"])
    ax.set_ylim(ylim_min, ylim_max)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlabel("")
    ax.set_ylabel("")

    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(3)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    ax.set_position([0, 0, 1, 1])

    suffix_str = f"_{filename_suffix}" if filename_suffix else ""
    safe_mode = mode_label.replace(" ", "_").lower()
    plot_path = output_dir / f"cryo_power_MWe_heatmap_{safe_mode}_{label}{suffix_str}.{cfg.PLOT_FORMAT}"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=cfg.PLOT_DPI, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)
    print(f"Cryo power (MWe) heatmap saved to: {plot_path}")


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
        figsize=(n_cols * cfg.HEATMAP_FIGSIZE_SINGLE[0], n_rows * cfg.HEATMAP_FIGSIZE_SINGLE[1]),
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
                shading="auto",
                alpha=cfg.HEATMAP_ALPHA
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
                    linewidths=cfg.HEATMAP_CONTOUR_LINEWIDTH_CRYO
                )
                ax.clabel(CS, inline=True, fontsize=font_size_tick, fmt=cfg.HEATMAP_CONTOUR_FMT_CRYO, colors="black")

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

    fig.suptitle(mode_label, fontsize=cfg.HEATMAP_FONT_SIZE_TITLE, y=0.99)
    
    if pcm is not None:
        # 当有 colorbar 时，使用 subplots_adjust 而不是 tight_layout
        # 因为手动添加的 add_axes 与 tight_layout 不兼容
        fig.subplots_adjust(right=0.88, top=0.96)
        cax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(pcm, cax=cax)
        cbar.ax.tick_params(labelsize=cfg.HEATMAP_FONT_SIZE_TICK)
        cbar.set_label("Cryogenic Power (MW)", fontsize=cfg.HEATMAP_FONT_SIZE_COLORBAR_TITLE)
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
        fig_single, ax_single = plt.subplots(figsize=cfg.HEATMAP_FIGSIZE_SINGLE)
        pcm_single = ax_single.pcolormesh(
            record["X"], record["Y"], record["Z"],
            cmap=cmap,
            alpha=cfg.HEATMAP_ALPHA,
            norm=norm,
            shading="auto"
        )
        if record["contour_levels"] is not None:
            XX, YY = np.meshgrid(record["X"], record["Y"])
            CS_single = ax_single.contour(
                XX, YY, record["Z"],
                levels=record["contour_levels"],
                colors="black",
                linewidths=cfg.HEATMAP_CONTOUR_LINEWIDTH_CRYO
            )
            ax_single.clabel(CS_single, inline=True, fontsize=font_size_tick, fmt=cfg.HEATMAP_CONTOUR_FMT_CRYO, colors="black")

        ax_single.set_xlim(xlim_range)
        ax_single.set_ylim(ylim_range)
        ax_single.set_xticks([])
        ax_single.set_xticklabels([])
        ax_single.set_xlabel("")
        ax_single.set_yticks([])
        ax_single.set_ylabel("")
        
        # 设置边框（与AF热力图一致）
        for spine in ax_single.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(3)
            spine.set_visible(True)

        # 调整布局（与AF热力图一致：去除白边，让子图完全填满figure）
        fig_single.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        ax_single.set_position([0, 0, 1, 1])
        
        single_path = individual_dir / f"{record['scenario']}_{record['temperature']:.1f}K.{cfg.PLOT_FORMAT}"
        # 使用与AF热力图相同的保存方式：bbox_inches='tight'，pad_inches=0.0（裁剪空白边缘）
        fig_single.savefig(single_path, dpi=cfg.PLOT_DPI, bbox_inches='tight', pad_inches=0.0)
        plt.close(fig_single)
        print(f"  └─ Saved subplot: {single_path}")

    # 保存共享 colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    colorbar_fig = plt.figure(figsize=cfg.HEATMAP_FIGSIZE_COLORBAR)
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
    ax.set_title('Parasitic Power vs. Operating Temperature', fontsize=cfg.HEATMAP_FONT_SIZE_TITLE, y=1.05)
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
        fontsize=cfg.HEATMAP_FONT_SIZE_COLORBAR,
        title='Operating Case (Temperature - Coolant)'
    )
    legend.get_frame().set_facecolor('none')
    legend.get_frame().set_edgecolor('none')
    legend.get_title().set_fontsize(cfg.HEATMAP_FONT_SIZE_COLORBAR)
    plt.tight_layout()
    plt.savefig(save_path, dpi=cfg.PLOT_DPI, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"Color legend saved to: {save_path}")

def auto_contour_levels(
    Z_sub: np.ndarray,
    norm: mpl.colors.Normalize,
    n_min: int = 4,
    n_max: int = 6,
    q: float = 0.08,
    margin: float = 0.05,
    include_zero: bool = True,
    verbose: bool = False  # 是否输出详细信息（默认False，只有"相对于最小值"才输出）
) -> np.ndarray:
    """
    为单个子图自动生成4-6条等高线，在视觉空间（norm空间）中均匀分布。
    
    参数:
        Z_sub: 子图数据数组（2D或1D）
        norm: Matplotlib归一化对象（必须支持inverse方法，如SymLogNorm、LogNorm）
        n_min: 最小等高线数量（默认4）
        n_max: 最大等高线数量（默认6）
        q: 分位数范围，用于避免极值挤压（默认0.08，即使用8%-92%分位数）
        margin: 视觉空间中的边距比例（默认0.05，即5%）
        include_zero: 如果数据跨0，是否将最接近0的等高线替换为0.0（默认True）
        verbose: 是否输出详细信息（默认False，只有"相对于最小值"才输出）
    
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
    # 注意：norm通常期望输入在[vmin, vmax]范围内，但我们可以直接使用
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
    except Exception as e:
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
    # 如果范围很大，使用更多等高线；如果范围很小，使用较少等高线
    n = int(np.clip(np.round(n_min + (n_max - n_min) * min(span / 0.5, 1.0)), n_min, n_max))
    
    # 在视觉空间中留边距并取等距点（不包含端点）
    t_levels = np.linspace(t_lo + margin * span, t_hi - margin * span, n)
    
    # 逆变换回数据空间
    try:
        if hasattr(norm, 'inverse'):
            levels = norm.inverse(t_levels)
        else:
            # 如果没有inverse方法，使用线性逆变换（备用方案）
            levels = t_levels * (z_max - z_min) + z_min
    except Exception as e:
        # 如果逆变换失败，使用线性逆变换（备用方案）
        levels = t_levels * (z_max - z_min) + z_min
    
    # 确保levels在数据范围内（留小边距避免与边框重合）
    eps = 1e-9 * (z_max - z_min) if (z_max > z_min) else 1e-9
    levels = np.clip(levels, z_min + eps, z_max - eps)
    
    # 去重并排序
    levels = np.unique(levels)
    levels = np.sort(levels)
    
    # 如果include_zero=True且数据跨0，将最接近0的等高线替换为0.0
    if include_zero and z_min < 0 < z_max and len(levels) > 0:
        # 找到最接近0的等高线
        closest_idx = np.argmin(np.abs(levels))
        closest_level = levels[closest_idx]
        
        # 如果最接近的等高线足够接近0（在数据范围的1%内），替换为0.0
        data_range = z_max - z_min
        if abs(closest_level) < 0.01 * data_range:
            levels[closest_idx] = 0.0
            levels = np.unique(np.sort(levels))
    
    return levels


# ===== 非均匀 ΔLCOE 热力图的自动等高线 =====================================
# 分布特点: 主体在低端(最优盆地~0), 极薄的极值尾在不可行边界(可达数百), 颜色用
# SymLogNorm 在 vmax 饱和。手工逐面板调参归纳出的规则(见 grill-me 设计记录):
#   floor = ≥底边(最低Npw行)delta 的最小整齐档  -> 跳 bottom-stub, 最低线不贴底
#   cap   = 内部性≥DELTA_CAP_INTERIORITY 的最高整齐档 -> 顶标签必显、不缩成贴角细弧
#   中间   = floor→cap 在视觉(norm)空间等距 + snap 到整齐阶梯, 共 ~n 条
_NICE_MANTISSA = (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8)
_NICE_LADDER = np.array(sorted({m * 10.0 ** k for k in range(-4, 5) for m in _NICE_MANTISSA}))
# 内部性阈值: 标签中心须距四边 ≥ 该比例才不被 _hide_labels_near_axes(margin=0.05)隐藏;
# 取 0.12 (≈2.4×隐藏margin) 保证清晰、并自动避开贴角细弧。
DELTA_CAP_INTERIORITY = 0.12
# ΔLCOE 分档的分位数窗口(绘图面积加权): floor 覆盖 P_LO、cap 覆盖 P_HI, 中间几何铺 nice 档。
# 分布极底重 -> 取 [10, 90] 让最低档≈10%数据在其下(低值区有线)、不浪费在<10%的高值细尾。
DELTA_Q_LO = 10.0
DELTA_Q_HI = 90.0


def _contour_interiority(FX, FY, Zi, level, x0, x1, y0, y1) -> float:
    """某 level 的等高线'最长段的最大内部性'(该段各点到四边归一化最小距离的最大值)。

    0 = 贴边(标签会被隐藏), 越大越靠图内。用于封顶时判断'标签能否显示'。
    在独立 Figure 上算, 不触碰当前 pyplot 状态。
    """
    from matplotlib.figure import Figure
    try:
        axt = Figure().subplots()
        cs = axt.contour(FX, FY, Zi, levels=[float(level)])
    except Exception:
        return 0.0
    best = 0.0
    blen = -1.0
    for p in cs.get_paths():
        v = p.vertices
        if len(v) < 2:
            continue
        xn = (v[:, 0] - x0) / (x1 - x0)
        yn = (v[:, 1] - y0) / (y1 - y0)
        seg_len = float(np.hypot(np.diff(xn), np.diff(yn)).sum())
        if seg_len > blen:                       # 只看最长段(标签落在最长段上)
            blen = seg_len
            best = float(np.minimum.reduce([xn, 1.0 - xn, yn, 1.0 - yn]).max())
    return best


def _contour_low_metrics(FX, FY, Zi, level, x0, x1, y0, y1,
                         y_margin: float = 0.10, x_frac: float = 0.3):
    """返回 (是否贴底横跑, 最长段归一化长度)。

    - 贴底横跑: 有顶点落在底边 y_margin(默认0.10, 约 Npw≈6-7)内, 且这些点 x 跨度 > x_frac。
      用于抓低值等高线折向底边的下支(标签无处放 -> 无标注贴底线)。
    - 最长段长度: 用于识别孤立短弧(如最优附近的小环)。
    在独立 Figure 上算, 不触碰当前 pyplot 状态。
    """
    from matplotlib.figure import Figure
    try:
        axt = Figure().subplots()
        cs = axt.contour(FX, FY, Zi, levels=[float(level)])
    except Exception:
        return (False, 1.0)
    bhug = False
    maxlen = 0.0
    for p in cs.get_paths():
        v = p.vertices
        if len(v) < 2:
            continue
        xn = (v[:, 0] - x0) / (x1 - x0)
        yn = (v[:, 1] - y0) / (y1 - y0)
        maxlen = max(maxlen, float(np.hypot(np.diff(xn), np.diff(yn)).sum()))
        near = yn < y_margin
        if int(near.sum()) >= 2:
            xnn = xn[near]
            if float(xnn.max() - xnn.min()) > x_frac:
                bhug = True
    return (bhug, maxlen)


def _prune_unlabeled_bottom_paths(ax, CS, texts, bottom_band: float = 0.06) -> None:
    """就地删除'整段极贴底(yn.max<bottom_band)、且无可见标签'的等高线子路径。

    典型来源: 细网格插值/平滑在最底一两行(Npw≈5-7)引入微小起伏, 使某档等高线除主线外还在
    最底部多出一小段; 该段标签必然贴底被 `_hide_labels_near_axes` 隐藏 -> 无标注贴底线。
    须在 clabel + 隐藏贴边标签之后调用。mpl≥3.8 把同一 level 的所有分支合并进一条 compound
    Path(以 MOVETO 分隔子路径), 故先按 MOVETO 拆子路径再逐段判断:
      - 候选: 子路径最高点 yn.max() < bottom_band(整段压在最底部);
      - 删除: 候选段的包围盒内没有任何**可见**标签。
    阈值取 0.06(而非更大)是关键: inline 标签把主线断成两段时, 靠下那段仍从底部升到断口、
    yn.max 通常 >0.06 -> 不被误删(避免"半截线"); 只有真正整段极贴底的额外小段才删。
    """
    from matplotlib.path import Path as _MplPath
    try:
        inv = ax.transAxes.inverted()
        trans = ax.transData
    except Exception:
        return

    vis = []
    for t in (texts or []):
        try:
            if t.get_visible():
                vis.append(inv.transform(trans.transform([t.get_position()]))[0])
        except Exception:
            pass
    vis = np.asarray(vis) if vis else np.empty((0, 2))

    def _subverts(p):
        """按 MOVETO 把 compound Path 拆成若干 (Nx2) 子路径顶点数组。"""
        v = np.asarray(p.vertices)
        codes = p.codes
        if codes is None:
            return [v]
        idx = [i for i, c in enumerate(codes) if c == _MplPath.MOVETO]
        idx.append(len(v))
        return [v[idx[k]:idx[k + 1]] for k in range(len(idx) - 1) if idx[k + 1] - idx[k] >= 1]

    def _keep_sub(sv) -> bool:
        if len(sv) < 2:
            return True
        frac = inv.transform(trans.transform(sv))
        xn, yn = frac[:, 0], frac[:, 1]
        if float(yn.max()) >= bottom_band:
            return True                      # 主支/升起的弧 -> 保留
        if vis.size:                         # 贴底候选: 有可见标签则保留
            inbox = ((vis[:, 0] >= xn.min() - 0.05) & (vis[:, 0] <= xn.max() + 0.05)
                     & (vis[:, 1] >= yn.min() - 0.05) & (vis[:, 1] <= yn.max() + 0.05))
            if bool(np.any(inbox)):
                return True
        return False                         # 贴底且无可见标签 -> 删

    def _rebuild(p):
        subs = [sv for sv in _subverts(p) if _keep_sub(sv)]
        if not subs:
            return None
        verts = np.concatenate(subs)
        codes = []
        for sv in subs:
            codes.append(_MplPath.MOVETO)
            codes.extend([_MplPath.LINETO] * (len(sv) - 1))
        return _MplPath(verts, np.asarray(codes, dtype=np.uint8))

    def _prune_coll(coll):
        new_paths = []
        for p in coll.get_paths():
            rp = _rebuild(p)
            if rp is not None:
                new_paths.append(rp)
        coll.set_paths(new_paths)

    colls = getattr(CS, "collections", None)
    try:
        if colls:
            for coll in colls:
                _prune_coll(coll)
        else:
            _prune_coll(CS)
    except Exception:
        pass


def _auto_levels_nonuniform(Z_sub, norm, y_log, x_log,
                            cap_interiority: float = DELTA_CAP_INTERIORITY,
                            n: int = 5) -> np.ndarray:
    """非均匀 ΔLCOE 热力图的等高线级别 —— **按数据分布定档, 尊重坐标轴**(2026-07-01 重构)。

    这些面板 delta 分布极"底重": 50–70% 的数据挤在低值区、长尾很细。旧法在 floor→max 之间
    几何铺档, 导致所有档堆在稀疏的高值尾巴上、数据密集的低值区反而无线(如 S1/4.2K 最低档 40,
    却有 63% 数据在其下)。新法:
      - 在**绘图均匀网格(线性 Npw × log Rj, 只取可行 cell)**上取分位数 —— 线性 Npw 匹配渲染纵轴,
        故分位数是**绘图面积加权**的, 天然尊重坐标轴;
      - floor = 覆盖 P10 的最小整齐档(≈10% 数据在其下, 低值区有线); cap = 覆盖 P90 的整齐档
        (不把档浪费在 <10% 的细高尾); 中间在 nice 阶梯 {1,1.2,1.5,2,2.5,3,4,5,6,8}×10^k 上
        几何均匀取 n 档。
    渲染层 `_prune_unlabeled_bottom_paths`(level 粒度) 兜底处理个别无标注贴底子路径。
    """
    z_all = Z_sub[np.isfinite(Z_sub)]
    if z_all.size < 10:
        return np.array([])
    from scipy.interpolate import RegularGridInterpolator as _RGI
    feas = np.isfinite(Z_sub)
    dmax = float(np.nanmax(z_all))
    Zfill = np.where(feas, Z_sub, dmax)
    npw_lin = np.power(10.0, y_log)                        # 线性 Npw(匹配渲染纵轴)
    fy = np.linspace(float(npw_lin.min()), float(npw_lin.max()), 240)
    fx = np.linspace(float(x_log.min()), float(x_log.max()), 240)  # log Rj(匹配渲染横轴)
    gv = _RGI((y_log, x_log), Zfill, bounds_error=False, fill_value=None)
    gm = _RGI((y_log, x_log), feas.astype(float), bounds_error=False, fill_value=0.0)
    FX, FY = np.meshgrid(fx, fy)
    qy = np.log10(FY)                                      # 查询用 log Npw
    Zi = gv((qy, FX))
    Mi = gm((qy, FX)) >= 0.5                               # 可行掩膜(剔除 recirc>50 的填充值)
    zf = Zi[np.isfinite(Zi) & Mi]
    if zf.size < 10:
        zf = z_all
    ladder = _NICE_LADDER
    plo, phi = np.percentile(zf, [DELTA_Q_LO, DELTA_Q_HI])
    zmin = float(zf.min())
    _loc = ladder[ladder >= max(float(plo), zmin)]
    lo = float(_loc.min()) if _loc.size else zmin
    _hic = ladder[ladder >= float(phi)]
    hi = float(_hic.min()) if _hic.size else float(zf.max())
    if hi <= lo:                                           # 极窄范围: cap 抬一档
        _h2 = ladder[ladder > lo]
        hi = float(_h2.min()) if _h2.size else lo
    cands = sorted(float(c) for c in ladder if lo <= c <= hi)
    if not cands:
        return np.array([round(lo, 4)])
    if len(cands) <= int(n):
        sel = cands                                        # 阶梯档数已 <= 目标, 全取
    else:
        idx = np.unique(np.linspace(0, len(cands) - 1, int(n)).round().astype(int))
        sel = [cands[i] for i in idx]                      # 几何均匀取 n 档(含首尾)
    return np.array(sorted({round(s, 4) for s in sel}))


def auto_contour_levels(
    Z_sub: np.ndarray,
    norm: mpl.colors.Normalize,
    n_min: int = 4,
    n_max: int = 6,
    q: float = 0.08,
    margin: float = 0.05,
    include_zero: bool = True,
    verbose: bool = False,
    y_coords_log=None,
    x_coords_log=None,
    cap_interiority: float = DELTA_CAP_INTERIORITY,
) -> np.ndarray:
    """
    为单个子图自动生成4-6条等高线，在视觉空间（norm空间）中均匀分布。
    
    参数:
        Z_sub: 子图数据数组（2D）
        norm: Matplotlib归一化对象（必须支持inverse方法，如SymLogNorm、LogNorm）
        n_min: 最小等高线数量（默认4）
        n_max: 最大等高线数量（默认6）
        q: 分位数范围，用于避免极值挤压（默认0.05，即使用5%-95%分位数）
        margin: 视觉空间中的边距比例（默认0.05，即5%）
        include_zero: 如果数据跨0，是否将最接近0的等高线替换为0.0（默认True）
        verbose: 是否输出详细信息（默认False）
        y_coords_log/x_coords_log: 提供时启用非均匀热力图新法(见 _auto_levels_nonuniform)

    返回:
        等高线级别数组（已排序，不包含zmin/zmax）
    """
    # === 新方法(非均匀热力图): 给了坐标轴 -> floor(底边)+cap(可标注性)+视觉等距 ===
    if y_coords_log is not None and x_coords_log is not None:
        try:
            lv = _auto_levels_nonuniform(
                Z_sub, norm,
                np.asarray(y_coords_log, dtype=float),
                np.asarray(x_coords_log, dtype=float),
                cap_interiority=cap_interiority,
                n=n_max,
            )
            if lv is not None and len(lv) > 0:
                if verbose:
                    print(f"[auto_contour_levels] 非均匀法 levels={np.round(lv, 4).tolist()}")
                return lv
        except Exception as _e:
            if verbose:
                print(f"[auto_contour_levels] 非均匀法失败, 回退旧法: {_e}")
        # 落空则继续走下面的旧分位数法

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

    # 输出 raw levels 和对应的视觉空间分数位
    # if verbose or True:  # 总是输出，方便调试
    #     print(f"\n[auto_contour_levels] Raw levels 计算:")
    #     print(f"  - 视觉空间分数位 (t_levels): {np.round(t_levels, 4).tolist()}")
    #     print(f"  - 对应的 raw levels: {np.round(levels_raw, 4).tolist()}")
    #     print(f"  - 数据范围: [{z_min:.4g}, {z_max:.4g}]")
    #     print(f"  - 视觉空间范围: [{t_lo:.4f}, {t_hi:.4f}], span={span:.4f}")
    #     print(f"  - 等高线数量: {n}")

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
    return levels


def pick_nice_levels_by_norm_targets(
    Z_sub: np.ndarray,
    norm: mpl.colors.Normalize,
    n_min: int = 5,
    n_max: int = 6,
    q: float = 0.08,
    margin: float = 0.05,
    include_zero: bool = True,
    verbose: bool = False  # 是否输出详细信息（默认False，只有"相对于最小值"才输出）
) -> np.ndarray:
    """
    在视觉空间（norm空间）中均匀选择目标位置，然后从多尺度整齐候选值中选择最接近的。
    
    参数:
        Z_sub: 子图数据数组（2D）
        norm: Matplotlib归一化对象（必须支持inverse方法，如SymLogNorm、LogNorm）
        n_min: 最小等高线数量（默认4）
        n_max: 最大等高线数量（默认6）
        q: 分位数范围，用于避免极值挤压（默认0.08，即使用8%-92%分位数）
        margin: 视觉空间中的边距比例（默认0.05，即5%）
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

    # if verbose:
    #     print("\n[contour] snap_levels_to_nice()")
    #     print(f"  - data range: [{z_min:.4g}, {z_max:.4g}], n={n}")
    #     print(f"  - raw levels: {np.round(levels, 4).tolist()}")

    # 1) 先根据范围选择 local grid 的 step（越粗优先）
    step_list = [20.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.2]
    step_chosen = step_list[-1]
    for st in step_list:
        if np.floor(data_range / st) > n:
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

    # if verbose:
    #     print(f"  - step_chosen for local grid: {step_chosen}")
    #     print(f"  - local_grid: {np.round(local_grid,4).tolist()}")
    #     print(f"  - coarse 1-2-5: {np.round(coarse_125,4).tolist()}")
    #     print(f"  - candidates (local grid + 1-2-5): {np.round(candidates,4).tolist()}")

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
    # print(f"  - t_span: {t_span}")
    t_gap_min = t_span / (n * 3.0)

    # if verbose:
    #     print(f"  - t_gap_min: {t_gap_min}")
    #     print(f"  - t_targets: {np.round(t_targets, 4).tolist()}")

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
            # 增加上端缓冲距离，避免锚定得太接近最大值（使用数据范围的2%作为缓冲）
            eps_cap = max(1e-9 * max(1.0, abs(z_max)), 0.08 * data_range)
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
                    if t_cand - t_prev >= 2*t_gap_min:
                        best_anchor = cand
                        break
                levels_chosen[-1] = best_anchor

        # if verbose:
        #     print(f"  - final chosen levels (with upper anchor): {np.round(levels_chosen, 4).tolist()}")
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
        fontsize=cfg.HEATMAP_FONT_SIZE_COLORBAR,
        title='Direct Magnet System Cost Component',
        handleheight=1.5
    )
    legend.get_frame().set_facecolor('none')
    legend.get_frame().set_edgecolor('none')
    legend.get_title().set_fontsize(cfg.HEATMAP_FONT_SIZE_COLORBAR)
    plt.tight_layout()
    plt.savefig(save_path, dpi=cfg.PLOT_DPI, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"Hatch legend saved to: {save_path}")

    # =============================================================================
# 绘图函数（新增）：ΔLCOE 热力图（风格与 parasitic heatmap 一致）
# =============================================================================

def _smooth_delta_surface(Z, scenario=None, temperature_K=None, coolant=None, sigma=1.2):
    """对 ΔLCOE 网格做轻度 NaN 感知高斯平滑, 抹掉离散充电时间/可用率(AF)台阶造成的非物理小台阶。

    Z: 2D 数组, 行=Npw(升序), 列=R_joint(升序)。
    物理: 固定 Npw 时 ΔLCOE 沿 R_joint 单调增; 固定 R_joint 时沿 Npw 是 U 形(最优 Npw 处最小)。
    AF 最近邻查表会在该光滑曲面上叠加 ~0.07 $/MWh 的小台阶, 使等高线出现折线/小拐折。
    早期版本用 cummax 强制单调, 但 cummax 会把台阶压成"平台"(相等值), 等高线穿过平台仍会折;
    这里改用归一化(NaN 感知)高斯平滑, 把小台阶抹成连续过渡、不产生平台, 保留 U 形与最优结构。
    改动远小于颜色尺度; 颜色(粗网格)与等高线(后续细网格插值)都用平滑后的 Z。自检 R_joint 残余非单调。
    """
    if Z is None or getattr(Z, "ndim", 0) != 2:
        return Z
    try:
        from scipy.ndimage import gaussian_filter
    except Exception:
        return Z
    Z = np.asarray(Z, dtype=float)
    M = (~np.isnan(Z)).astype(float)
    if M.sum() < 8:
        return Z
    Zf = np.where(np.isnan(Z), 0.0, Z)
    out = gaussian_filter(Zf, sigma) / np.maximum(gaussian_filter(M, sigma), 1e-9)
    out[M < 0.5] = np.nan
    # 自检: R_joint 方向应单调增; 报告残余非单调对(平滑后应≈0)与最大改动
    rj_bad = 0
    for i in range(out.shape[0]):
        v = out[i, :][~np.isnan(out[i, :])]
        if v.size >= 2:
            rj_bad += int(np.sum(np.diff(v) < -1e-6))
    max_change = float(np.nanmax(np.abs(out - Z))) if np.isfinite(out).any() else 0.0
    tag = f"{scenario} {temperature_K}K {coolant}".strip()
    if max_change > 0:
        print(f"  [等高线平滑] {tag}: 高斯平滑 sigma={sigma}, 最大改动 {max_change:.4f} $/MWh, R_joint残余非单调 {rj_bad}")
    return out



def _safe_clabel(ax, CS, *, edge_margin: float = 0.10, **kwargs):
    """clabel 包装: 强制把标签放在离四边至少 edge_margin(轴归一化)处。

    背景: clabel(inline=True) 会挖断等值线给标签让位。matplotlib 自选的位置常落在
    画面边缘 —— 此时若 _hide_labels_near_axes 再把标签藏掉, 缺口就成了"断了却什么
    都没有"的空档(Fig. 5 面板 G 的 level 3/6/10 即如此); 即便标签保留, 落在角上也会
    把等值线与边界的交点挖掉(level 1 的标签落在 R_j=1.36, Npw=16.5, 正好挖掉左下角
    与左边界的连接)。这里改为自己选位: 只在离边界足够远的顶点放标签, 找不到就不放,
    等值线因此保持连续。见工作文档 §14.51。
    """
    try:
        paths = list(CS.get_paths())
    except AttributeError:                      # 旧 matplotlib
        paths = [q for c in CS.collections for q in c.get_paths()]
    positions = []
    for path in paths:
        v = np.asarray(path.vertices, dtype=float)
        if len(v) < 2 or not np.isfinite(v).all():
            continue
        frac = ax.transAxes.inverted().transform(ax.transData.transform(v))
        ok = ((frac[:, 0] > edge_margin) & (frac[:, 0] < 1.0 - edge_margin)
              & (frac[:, 1] > edge_margin) & (frac[:, 1] < 1.0 - edge_margin))
        if not ok.any():
            continue
        # 取离四边最远的顶点, 使标签周围留白最大
        clearance = np.minimum.reduce([frac[:, 0], 1.0 - frac[:, 0],
                                       frac[:, 1], 1.0 - frac[:, 1]])
        positions.append(tuple(v[int(np.argmax(np.where(ok, clearance, -1.0)))]))
    if not positions:
        return []
    kwargs.pop("manual", None)
    return ax.clabel(CS, manual=positions, **kwargs)


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
    filename_suffix: str = "",  # 可选的文件名后缀，用于区分不同配色方案
    use_stroke: bool = False,  # 是否使用描边技术（Halo/Stroke）
    draw_contours: bool = True,  # False 时只画热力图(颜色+不可行区), 不画等高线/标签(用于调参预览的"无等高线"版)
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
    font_size = cfg.HEATMAP_FONT_SIZE
    setup_plot_style()

    # 生成label，如果提供了coolant则在文件名中包含
    if coolant is not None:
        label = f"{temperature_K}K_{coolant}"
    else:
        label = f"{temperature_K}K"

    fig, ax = plt.subplots(1, 1, figsize=cfg.HEATMAP_FIGSIZE_SINGLE)

    # 与 parasitic heatmap 相同的网格化方式：index=Npw, columns=R_joint
    pivot = df.pivot(index="Npw", columns="R_joint", values=value_col)
    X, Y = np.meshgrid(pivot.columns, pivot.index)
    Z = pivot.values

    # 轻度高斯平滑: 抹平离散 AF/充电时间台阶造成的非物理小台阶, 使等高线不再有折线/拐折。
    # (早期用 cummax 强制单调, 但会压出"平台"→等高线穿平台仍折; 改用平滑, 不产生平台。)
    # 颜色与等高线都用平滑后的 Z; 改动远小于颜色尺度。
    Z = _smooth_delta_surface(Z, scenario=scenario, temperature_K=temperature_K, coolant=coolant)

    # 记录NaN位置（不可行点），不填充，后续用Hatch显示
    nan_mask = np.isnan(Z)
    
    # 与原图一致：把 R_joint 转为 nΩ 并取 log10
    X_log = np.log10(X * 1e9)
    
    # 直接使用中心点坐标，使 pcolormesh 和 contour 都使用中心点，二者一致
    # 对于绘图，使用masked array，NaN值不会被绘制（显示为白色/背景色）
    # 但我们需要用Hatch填充来明确标识这些不可行点
    Z_plot = np.ma.masked_array(Z, mask=nan_mask)
    
    # 获取 colormap 对象（支持 config 中的自定义/截取版配色方案名）
    cmap_resolved = cfg.resolve_cmap(cmap) if hasattr(cfg, "resolve_cmap") else cmap
    if isinstance(cmap_resolved, str):
        try:
            cmap_obj = plt.get_cmap(cmap_resolved)
        except ValueError:
            # 如果 matplotlib 不支持，尝试从 seaborn 获取
            try:
                import seaborn as sns
                cmap_obj = sns.color_palette(cmap_resolved, as_cmap=True)
            except (ImportError, ValueError):
                # 如果都失败，使用默认配色
                cmap_obj = plt.get_cmap("viridis")
    else:
        cmap_obj = cmap_resolved
    
    # 绘制热力图（使用中心点坐标，shading="auto"）
    # 这样 pcolormesh 和 contour 都使用中心点，二者完全一致
    ax.pcolormesh(X_log, Y, Z_plot, cmap=cmap_obj, norm=norm, shading="auto", alpha=cfg.HEATMAP_ALPHA)
    
    # 判断是否是"相对于最小值"的等高线（用于控制输出详细程度）
    is_delta_min = "delta_lcoe_min" in value_col or "relative to scenario minimum" in str(value_col).lower()
    
    # 自动生成等高线（如果启用）
    if auto_generate_levels and contour_levels is None:
        # 所有面板统一用标准自适应等高线(同 10K 的画法)。n_max 由 6 降到 5, 让数据范围宽的
        # 4.2K 列等高线不致过密、标签互不压线。(不再对 S1/4.2K 特殊加密。)
        contour_levels = auto_contour_levels(
            Z_sub=Z,
            norm=norm,
            n_min=4,
            n_max=5,
            q=0.08,
            margin=0.03,
            include_zero=True,
            verbose=is_delta_min,  # 只有"相对于最小值"才输出详细信息
            # 传坐标轴 -> 启用非均匀热力图新法(floor=底边, cap=可标注性, 视觉等距)
            y_coords_log=np.log10(pivot.index.values.astype(float)),
            x_coords_log=np.log10(pivot.columns.values.astype(float) * 1e9),
        )
    
    # 如果仍然没有等高线，检查是否提供了
    if contour_levels is None or len(contour_levels) == 0:
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
        X_log_edges = np.zeros((len(Y) + 1, len(X_log[0]) + 1))
        Y_edges = np.zeros((len(Y) + 1, len(X_log[0]) + 1))
        
        # 计算边角点位置
        for i in range(len(Y) + 1):
            for j in range(len(X_log[0]) + 1):
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
                    if len(X_log[0]) > 1:
                        X_log_edges[i, j] = X_log[0, 0] - (X_log[0, 1] - X_log[0, 0]) / 2
                    else:
                        X_log_edges[i, j] = X_log[0, 0] - 0.1
                elif j == len(X_log[0]):
                    X_log_edges[i, j] = X_log[0, -1] + (X_log[0, -1] - X_log[0, -2]) / 2 if len(X_log[0]) > 1 else X_log[0, 0] + 0.1
                else:
                    X_log_edges[i, j] = (X_log[0, j-1] + X_log[0, j]) / 2
        
        # 对每个NaN的网格单元绘制Hatch
        for i in range(len(Y)):
            for j in range(len(X_log[0])):
                if nan_mask[i, j]:
                    # 绘制Hatch填充的矩形
                    rect = mpatches.Rectangle(
                        (X_log_edges[i, j], Y_edges[i, j]),
                        X_log_edges[i, j+1] - X_log_edges[i, j],
                        Y_edges[i+1, j] - Y_edges[i, j],
                        facecolor='none',
                        edgecolor='black',
                        linewidth=0.5,
                        hatch='///',
                        alpha=0.5
                    )
                    ax.add_patch(rect)

    # --- 等高线级别：缺失时, 自适应模式下用数据分位数兜底(保证出图), 非自适应才报错 ---
    if contour_levels is None or len(contour_levels) == 0:
        if not auto_generate_levels:
            raise ValueError(
                f"plot_delta_lcoe_heatmap_single: contour_levels must be provided. "
                f"Got None or empty array for scenario={scenario}, temperature_K={temperature_K}, coolant={coolant}"
            )
        # auto_contour_levels 对该格退化分布返回空: 用正值分位数兜底; 无正值则跳过等高线(全hatch)
        _zf = Z[np.isfinite(Z)]
        _zp = _zf[_zf > 0]
        if len(_zp) >= 5:
            contour_levels = np.unique(np.percentile(_zp, [20, 40, 60, 80, 95]))
        elif len(_zp) > 0:
            contour_levels = np.unique(_zp)
        else:
            contour_levels = np.array([])

    contour_levels_to_use = contour_levels

    # draw_contours=False: 清空级别 -> 跳过下面整段等高线绘制(其 if len(..)>0 守卫), 只留热力图+hatch
    if not draw_contours:
        contour_levels_to_use = np.array([])

    # 初始化filtered_levels，用于后续colorbar标注
    filtered_levels = np.array([])

    # --- 细网格插值: 仅用于等高线 ---
    # 颜色(pcolormesh)与不可行区(hatch)已用粗网格画好(保留扫描分辨率)。粗网格上近水平的
    # 低等高线会呈阶梯/折线; 这里在 log-log 空间把(已单调化的) Z 上采样到密网格, 让等高线
    # 平滑。等高线级别已在上面用粗网格算好, 此处只替换坐标(X_log,Y)与 Z 供后续 contour 使用。
    try:
        from scipy.interpolate import RegularGridInterpolator as _RGI
        _y_src = np.log10(pivot.index.values.astype(float))            # log10(Npw)
        _x_src = np.log10(pivot.columns.values.astype(float) * 1e9)    # log10(R_joint nΩ)
        _feas = ~np.isnan(Z)
        if _feas.sum() >= 8 and len(_y_src) >= 3 and len(_x_src) >= 3:
            _Zfill = np.where(_feas, Z, float(np.nanmax(Z)))           # 填充不可行区便于插值, 之后再遮罩
            _fy = np.linspace(_y_src.min(), _y_src.max(), 260)
            _fx = np.linspace(_x_src.min(), _x_src.max(), 260)
            _gv = _RGI((_y_src, _x_src), _Zfill, bounds_error=False, fill_value=None)
            _gm = _RGI((_y_src, _x_src), _feas.astype(float), method="nearest",
                       bounds_error=False, fill_value=0.0)
            _FX, _FY = np.meshgrid(_fx, _fy)
            _Zf = _gv((_FY, _FX))
            _Zf[_gm((_FY, _FX)) < 0.5] = np.nan                       # 不可行区不画等高线
            X_log = _FX
            Y = 10.0 ** _FY
            Z = _Zf
    except Exception as _e:
        print(f"  [等高线插值] 跳过(回退粗网格): {_e}")
    
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
    
    # 辅助函数：隐藏靠近坐标轴/边框的等高线标签，避免数值与轴线/边框交叉
    def _hide_labels_near_axes(texts, axis, margin_axes: float = 0.05, margin_data_y: float = 5.0):
        """
        将靠近坐标轴/边框的等高线标签隐藏，防止数字与坐标轴或边框交叉。
        
        参数:
            texts: clabel返回的文本对象列表
            axis: matplotlib axes对象
            margin_axes: 以坐标轴归一化坐标为单位的边缘留白（0-1），默认5%（增大以更严格）
            margin_data_y: 以数据坐标为单位的y轴下边缘留白（用于横轴检测），默认5.0
        
        返回:
            bool: 如果所有标签都被隐藏了，返回True；否则返回False
        """
        if not texts:
            return False
        
        # 获取当前坐标轴的数据范围
        xlim = axis.get_xlim()
        ylim = axis.get_ylim()
        
        hidden_count = 0
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
                hidden_count += 1

        # 如果所有标签都被隐藏了，返回True
        return hidden_count == len(texts)

    # 绘制等高线：使用自动生成或传入的等高线级别（已经过处理，直接使用）
    if len(contour_levels_to_use) > 0:
        # 直接使用传入的等高线（已经通过auto_contour_levels生成，或在视觉空间中均匀分布）
        filtered_levels = contour_levels_to_use
        
        # 定义智能格式化函数，避免科学计数法
        def format_contour_label(x):
            """格式化等高线标签，避免科学计数法"""
            # 如果值很大（>=1000）或很小（<0.01且>0），使用科学计数法
            if abs(x) >= 1000:
                # 对于大值，尝试使用整数或一位小数
                if abs(x) % 1 < 1e-6:  # 接近整数
                    return f"{int(x)}"
                else:
                    return f"{x:.1f}"
            elif 0 < abs(x) < 0.01:
                # 对于很小的值，使用科学计数法
                return f"{x:.2e}"
            else:
                # 对于中等值，使用普通数字格式，最多2位小数
                # 如果接近整数，显示整数
                if abs(x) % 1 < 1e-6:
                    return f"{int(x)}"
                elif abs(x) < 1:
                    return f"{x:.2f}"
                elif abs(x) < 10:
                    return f"{x:.1f}"
                else:
                    return f"{x:.0f}"
        
        # 绘制筛选后的等高线：先绘制所有等高线，然后检测重合并移除
        if len(filtered_levels) > 0:
            # 先绘制所有等高线（包括0等高线，不需要重复绘制）
            # 如果包含0等高线，使用加粗线条
            all_contour_objects = []  # 存储所有等高线对象
            all_text_objects = []  # 存储所有标签对象
            all_levels_map = {}  # 存储标签到等高线值的映射
            
            if use_stroke:
                # 使用描边技术：使用path_effects为黑色等高线添加白色描边
                if 0.0 in filtered_levels:
                    # 分别绘制0等高线（加粗）和其他等高线
                    non_zero_levels = filtered_levels[filtered_levels != 0.0]
                    if len(non_zero_levels) > 0:
                        for level in non_zero_levels:
                            # 绘制黑色主线条
                            CS_level = ax.contour(X_log, Y, Z, levels=[level], colors='black', linewidths=cfg.HEATMAP_CONTOUR_LINEWIDTH)
                            all_contour_objects.append(CS_level)
                            # 先 clabel(inline) 打断线条留间隙，再描边（否则描边填满间隙→压线）
                            texts_level = _safe_clabel(ax, 
                                CS_level,
                                inline=True,
                                fontsize=font_size,
                                fmt=format_contour_label,
                                colors='black',
                            )
                            # 为等高线集合添加白色描边效果（在 clabel 之后，保留间隙）
                            for collection in CS_level.collections:
                                collection.set_path_effects([
                                    patheffects.withStroke(
                                        linewidth=cfg.CONTOUR_STROKE_LINEWIDTH,
                                        foreground=cfg.CONTOUR_STROKE_FOREGROUND,
                                        alpha=cfg.CONTOUR_STROKE_ALPHA
                                    ),
                                    patheffects.Normal()
                                ])
                            if texts_level:
                                # 为每个文字标签添加白色描边
                                for txt in texts_level:
                                    txt.set_path_effects([
                                        patheffects.withStroke(
                                            linewidth=cfg.LABEL_STROKE_LINEWIDTH,
                                            foreground=cfg.LABEL_STROKE_FOREGROUND,
                                            alpha=cfg.LABEL_STROKE_ALPHA
                                        ),
                                        patheffects.Normal()
                                    ])
                                # 立即检查并隐藏靠近边缘的标签
                                all_hidden = _hide_labels_near_axes(texts_level, ax)
                                # 如果这个level的所有标签都被隐藏了，移除对应的等高线
                                if all_hidden:
                                    for collection in CS_level.collections:
                                        collection.remove()
                                else:
                                    all_text_objects.extend(texts_level)
                                    _prune_unlabeled_bottom_paths(ax, CS_level, texts_level)  # 删无标注贴底线(盆地下支)
                                    for txt in texts_level:
                                        all_levels_map[txt] = level
                    # 绘制0等高线（加粗）
                    CS_zero = ax.contour(X_log, Y, Z, levels=[0.0], colors='black', linewidths=cfg.HEATMAP_CONTOUR_LINEWIDTH_ZERO)
                    all_contour_objects.append(CS_zero)
                    texts_zero = _safe_clabel(ax, 
                        CS_zero,
                        inline=True,
                        fontsize=font_size,
                        fmt=format_contour_label,
                        colors='black',
                    )
                    # 为0等高线添加白色描边效果（在 clabel 之后，保留间隙）
                    for collection in CS_zero.collections:
                        collection.set_path_effects([
                            patheffects.withStroke(
                                linewidth=cfg.CONTOUR_STROKE_LINEWIDTH_ZERO,
                                foreground=cfg.CONTOUR_STROKE_FOREGROUND,
                                alpha=cfg.CONTOUR_STROKE_ALPHA
                            ),
                            patheffects.Normal()
                        ])
                    if texts_zero:
                        for txt in texts_zero:
                            txt.set_path_effects([
                                patheffects.withStroke(
                                    linewidth=cfg.LABEL_STROKE_LINEWIDTH,
                                    foreground=cfg.LABEL_STROKE_FOREGROUND,
                                    alpha=cfg.LABEL_STROKE_ALPHA
                                ),
                                patheffects.Normal()
                            ])
                        # 立即检查并隐藏靠近边缘的标签
                        all_hidden = _hide_labels_near_axes(texts_zero, ax)
                        # 如果这个level的所有标签都被隐藏了，移除对应的等高线
                        if all_hidden:
                            for collection in CS_zero.collections:
                                collection.remove()
                        else:
                            all_text_objects.extend(texts_zero)
                            for txt in texts_zero:
                                all_levels_map[txt] = 0.0
                else:
                    # 没有0等高线，正常绘制所有等高线
                    for level in filtered_levels:
                        # 绘制黑色主线条
                        CS_level = ax.contour(X_log, Y, Z, levels=[level], colors='black', linewidths=cfg.HEATMAP_CONTOUR_LINEWIDTH)
                        all_contour_objects.append(CS_level)
                        # 先 clabel(inline) 打断线条留间隙，再描边
                        texts_level = _safe_clabel(ax, 
                            CS_level,
                            inline=True,
                            fontsize=font_size,
                            fmt=format_contour_label,
                            colors='black',
                        )
                        # 为等高线集合添加白色描边效果（在 clabel 之后，保留间隙）
                        for collection in CS_level.collections:
                            collection.set_path_effects([
                                patheffects.withStroke(
                                    linewidth=cfg.CONTOUR_STROKE_LINEWIDTH,
                                    foreground=cfg.CONTOUR_STROKE_FOREGROUND,
                                    alpha=cfg.CONTOUR_STROKE_ALPHA
                                ),
                                patheffects.Normal()
                            ])
                        if texts_level:
                            for txt in texts_level:
                                txt.set_path_effects([
                                    patheffects.withStroke(
                                        linewidth=cfg.LABEL_STROKE_LINEWIDTH,
                                        foreground=cfg.LABEL_STROKE_FOREGROUND,
                                        alpha=cfg.LABEL_STROKE_ALPHA
                                    ),
                                    patheffects.Normal()
                                ])
                            # 立即检查并隐藏靠近边缘的标签
                            all_hidden = _hide_labels_near_axes(texts_level, ax)
                            # 如果这个level的所有标签都被隐藏了，移除对应的等高线
                            if all_hidden:
                                for collection in CS_level.collections:
                                    collection.remove()
                            else:
                                all_text_objects.extend(texts_level)
                                _prune_unlabeled_bottom_paths(ax, CS_level, texts_level)  # 删无标注贴底线(盆地下支)
                                all_levels_map.update({txt: level for txt in texts_level})
            else:
                # 根据背景色亮度动态选择颜色（浅色背景用黑色，深色背景用白色）
                level_colors_list = get_contour_colors_for_levels(X_log, Y, Z, filtered_levels, cmap_obj, norm)
                level_colors_dict = {level: level_colors_list[i] for i, level in enumerate(filtered_levels)}
                
                if 0.0 in filtered_levels:
                    # 分别绘制0等高线（加粗）和其他等高线
                    non_zero_levels = filtered_levels[filtered_levels != 0.0]
                    if len(non_zero_levels) > 0:
                        # 为每条等高线绘制不同颜色
                        for level in non_zero_levels:
                            color = level_colors_dict.get(level, 'black')
                            CS_level = ax.contour(X_log, Y, Z, levels=[level], colors=color, linewidths=cfg.HEATMAP_CONTOUR_LINEWIDTH)
                            all_contour_objects.append(CS_level)
                            texts_level = _safe_clabel(ax, 
                                CS_level,
                                inline=True,
                                fontsize=font_size,
                                fmt=format_contour_label,
                                colors=color,
                            )
                            if texts_level:
                                # 立即检查并隐藏靠近边缘的标签
                                all_hidden = _hide_labels_near_axes(texts_level, ax)
                                # 如果这个level的所有标签都被隐藏了，移除对应的等高线
                                if all_hidden:
                                    for collection in CS_level.collections:
                                        collection.remove()
                                else:
                                    all_text_objects.extend(texts_level)
                                    _prune_unlabeled_bottom_paths(ax, CS_level, texts_level)  # 删无标注贴底线(盆地下支)
                                    # 建立标签到等高线值的映射
                                    for txt in texts_level:
                                        all_levels_map[txt] = level
                    # 绘制0等高线（加粗）
                    color_zero = level_colors_dict.get(0.0, 'black')
                    CS_zero = ax.contour(X_log, Y, Z, levels=[0.0], colors=color_zero, linewidths=cfg.HEATMAP_CONTOUR_LINEWIDTH_ZERO)
                    all_contour_objects.append(CS_zero)
                    texts_zero = _safe_clabel(ax, 
                        CS_zero,
                        inline=True,
                        fontsize=font_size,
                        fmt=format_contour_label,
                        colors=color_zero,
                    )
                    if texts_zero:
                        all_text_objects.extend(texts_zero)
                        for txt in texts_zero:
                            all_levels_map[txt] = 0.0
                else:
                    # 没有0等高线，正常绘制所有等高线
                    # 为每条等高线绘制不同颜色
                    for level in filtered_levels:
                        color = level_colors_dict.get(level, 'black')
                        CS_level = ax.contour(X_log, Y, Z, levels=[level], colors=color, linewidths=cfg.HEATMAP_CONTOUR_LINEWIDTH)
                        all_contour_objects.append(CS_level)
                        texts_level = _safe_clabel(ax, 
                            CS_level,
                            inline=True,
                            fontsize=font_size,
                            fmt=format_contour_label,
                            colors=color,
                        )
                        if texts_level:
                            # 立即检查并隐藏靠近边缘的标签
                            all_hidden = _hide_labels_near_axes(texts_level, ax)
                            # 如果这个level的所有标签都被隐藏了，移除对应的等高线
                            if all_hidden:
                                for collection in CS_level.collections:
                                    collection.remove()
                            else:
                                all_text_objects.extend(texts_level)
                                _prune_unlabeled_bottom_paths(ax, CS_level, texts_level)  # 删无标注贴底线(盆地下支)
                                all_levels_map.update({txt: level for txt in texts_level})
            
            # 检测重合的等高线标签
            if all_text_objects:
                # 需要移除的等高线值（因为标签与边框重合）
                overlapping_with_border, overlapping_with_labels = _detect_overlapping_labels(all_text_objects, ax, all_levels_map)
                
                # 合并需要移除的等高线值
                levels_to_remove = set(overlapping_with_border) | set(overlapping_with_labels)
                
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
                    
                    # 重新绘制（只绘制不重合的等高线）
                    if len(filtered_levels_new) > 0:
                        if use_stroke:
                            # 使用描边技术：使用path_effects为黑色等高线添加白色描边
                            if 0.0 in filtered_levels_new:
                                non_zero_levels_new = filtered_levels_new[filtered_levels_new != 0.0]
                                if len(non_zero_levels_new) > 0:
                                    for level in non_zero_levels_new:
                                        CS_level = ax.contour(X_log, Y, Z, levels=[level], colors='black', linewidths=cfg.HEATMAP_CONTOUR_LINEWIDTH)
                                        # 为等高线集合添加白色描边效果
                                        for collection in CS_level.collections:
                                            collection.set_path_effects([
                                                patheffects.withStroke(
                                                    linewidth=cfg.CONTOUR_STROKE_LINEWIDTH,
                                                    foreground=cfg.CONTOUR_STROKE_FOREGROUND,
                                                    alpha=cfg.CONTOUR_STROKE_ALPHA
                                                ),
                                                patheffects.Normal()
                                            ])
                                        texts_level = _safe_clabel(ax, CS_level, inline=True, fontsize=font_size, fmt=format_contour_label, colors='black')
                                        if texts_level:
                                            for txt in texts_level:
                                                txt.set_path_effects([
                                                    patheffects.withStroke(
                                                        linewidth=cfg.LABEL_STROKE_LINEWIDTH,
                                                        foreground=cfg.LABEL_STROKE_FOREGROUND,
                                                        alpha=cfg.LABEL_STROKE_ALPHA
                                                    ),
                                                    patheffects.Normal()
                                                ])
                                            # 立即检查并隐藏靠近边缘的标签
                                            all_hidden = _hide_labels_near_axes(texts_level, ax)
                                            # 如果这个level的所有标签都被隐藏了，移除对应的等高线
                                            if all_hidden:
                                                for collection in CS_level.collections:
                                                    collection.remove()
                                CS_zero = ax.contour(X_log, Y, Z, levels=[0.0], colors='black', linewidths=cfg.HEATMAP_CONTOUR_LINEWIDTH_ZERO)
                                # 为0等高线添加白色描边效果
                                for collection in CS_zero.collections:
                                    collection.set_path_effects([
                                        patheffects.withStroke(
                                            linewidth=cfg.CONTOUR_STROKE_LINEWIDTH_ZERO,
                                            foreground=cfg.CONTOUR_STROKE_FOREGROUND,
                                            alpha=cfg.CONTOUR_STROKE_ALPHA
                                        ),
                                        patheffects.Normal()
                                    ])
                                texts_zero = _safe_clabel(ax, CS_zero, inline=True, fontsize=font_size, fmt=format_contour_label, colors='black')
                                if texts_zero:
                                    for txt in texts_zero:
                                        txt.set_path_effects([
                                            patheffects.withStroke(
                                                linewidth=cfg.LABEL_STROKE_LINEWIDTH,
                                                foreground=cfg.LABEL_STROKE_FOREGROUND,
                                                alpha=cfg.LABEL_STROKE_ALPHA
                                            ),
                                            patheffects.Normal()
                                        ])
                                    # 立即检查并隐藏靠近边缘的标签
                                    all_hidden = _hide_labels_near_axes(texts_zero, ax)
                                    # 如果这个level的所有标签都被隐藏了，移除对应的等高线
                                    if all_hidden:
                                        for collection in CS_zero.collections:
                                            collection.remove()
                            else:
                                for level in filtered_levels_new:
                                    CS_level = ax.contour(X_log, Y, Z, levels=[level], colors='black', linewidths=cfg.HEATMAP_CONTOUR_LINEWIDTH)
                                    # 为等高线集合添加白色描边效果
                                    for collection in CS_level.collections:
                                        collection.set_path_effects([
                                            patheffects.withStroke(
                                                linewidth=cfg.CONTOUR_STROKE_LINEWIDTH,
                                                foreground=cfg.CONTOUR_STROKE_FOREGROUND,
                                                alpha=cfg.CONTOUR_STROKE_ALPHA
                                            ),
                                            patheffects.Normal()
                                        ])
                                    texts_level = _safe_clabel(ax, CS_level, inline=True, fontsize=font_size, fmt=format_contour_label, colors='black')
                                    if texts_level:
                                        for txt in texts_level:
                                            txt.set_path_effects([
                                                patheffects.withStroke(
                                                    linewidth=cfg.LABEL_STROKE_LINEWIDTH,
                                                    foreground=cfg.LABEL_STROKE_FOREGROUND,
                                                    alpha=cfg.LABEL_STROKE_ALPHA
                                                ),
                                                patheffects.Normal()
                                            ])
                                        # 立即检查并隐藏靠近边缘的标签
                                        all_hidden = _hide_labels_near_axes(texts_level, ax)
                                        # 如果这个level的所有标签都被隐藏了，移除对应的等高线
                                        if all_hidden:
                                            for collection in CS_level.collections:
                                                collection.remove()
                        else:
                            # 获取每条等高线的颜色（基于背景色亮度）
                            level_colors_list_new = get_contour_colors_for_levels(X_log, Y, Z, filtered_levels_new, cmap_obj, norm)
                            level_colors_dict_new = {level: level_colors_list_new[i] for i, level in enumerate(filtered_levels_new)}
                            
                            if 0.0 in filtered_levels_new:
                                non_zero_levels_new = filtered_levels_new[filtered_levels_new != 0.0]
                                if len(non_zero_levels_new) > 0:
                                    # 为每条等高线绘制不同颜色
                                    for level in non_zero_levels_new:
                                        color = level_colors_dict_new.get(level, 'black')
                                        CS_level = ax.contour(X_log, Y, Z, levels=[level], colors=color, linewidths=cfg.HEATMAP_CONTOUR_LINEWIDTH)
                                        texts_level = _safe_clabel(ax, CS_level, inline=True, fontsize=font_size, fmt=format_contour_label, colors=color)
                                        if texts_level:
                                            # 立即检查并隐藏靠近边缘的标签
                                            all_hidden = _hide_labels_near_axes(texts_level, ax)
                                            # 如果这个level的所有标签都被隐藏了，移除对应的等高线
                                            if all_hidden:
                                                for collection in CS_level.collections:
                                                    collection.remove()
                                # 绘制0等高线（加粗）
                                color_zero = level_colors_dict_new.get(0.0, 'black')
                                CS_zero = ax.contour(X_log, Y, Z, levels=[0.0], colors=color_zero, linewidths=cfg.HEATMAP_CONTOUR_LINEWIDTH_ZERO)
                                texts_zero = _safe_clabel(ax, CS_zero, inline=True, fontsize=font_size, fmt=format_contour_label, colors=color_zero)
                                if texts_zero:
                                    # 立即检查并隐藏靠近边缘的标签
                                    all_hidden = _hide_labels_near_axes(texts_zero, ax)
                                    # 如果这个level的所有标签都被隐藏了，移除对应的等高线
                                    if all_hidden:
                                        for collection in CS_zero.collections:
                                            collection.remove()
                            else:
                                # 没有0等高线，正常绘制所有等高线
                                # 为每条等高线绘制不同颜色
                                for level in filtered_levels_new:
                                    color = level_colors_dict_new.get(level, 'black')
                                    CS_level = ax.contour(X_log, Y, Z, levels=[level], colors=color, linewidths=cfg.HEATMAP_CONTOUR_LINEWIDTH)
                                    texts_level = _safe_clabel(ax, CS_level, inline=True, fontsize=font_size, fmt=format_contour_label, colors=color)
                                    if texts_level:
                                        # 立即检查并隐藏靠近边缘的标签
                                        all_hidden = _hide_labels_near_axes(texts_level, ax)
                                        # 如果这个level的所有标签都被隐藏了，移除对应的等高线
                                        if all_hidden:
                                            for collection in CS_level.collections:
                                                collection.remove()
                else:
                    # 没有重合，先隐藏靠近边框的标签（这里不需要移除等高线，因为后面会统一处理）
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
                        # filtered_levels 可能是 Python list(布尔索引会报 "only integer scalar arrays"),先转 ndarray
                        _fl_arr = np.asarray(filtered_levels)
                        filtered_levels_new = _fl_arr[~np.isin(_fl_arr, levels_with_no_visible_labels)]
                        
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
                        
                        # 重新绘制（只绘制有可见标签的等高线）
                        if len(filtered_levels_new) > 0:
                            if use_stroke:
                                # 使用描边技术：使用path_effects为黑色等高线添加白色描边
                                if 0.0 in filtered_levels_new:
                                    non_zero_levels_new = filtered_levels_new[filtered_levels_new != 0.0]
                                    if len(non_zero_levels_new) > 0:
                                        for level in non_zero_levels_new:
                                            CS_level = ax.contour(X_log, Y, Z, levels=[level], colors='black', linewidths=cfg.HEATMAP_CONTOUR_LINEWIDTH)
                                            # 为等高线集合添加白色描边效果
                                            for collection in CS_level.collections:
                                                collection.set_path_effects([
                                                    patheffects.withStroke(
                                                        linewidth=cfg.CONTOUR_STROKE_LINEWIDTH,
                                                        foreground=cfg.CONTOUR_STROKE_FOREGROUND,
                                                        alpha=cfg.CONTOUR_STROKE_ALPHA
                                                    ),
                                                    patheffects.Normal()
                                                ])
                                            texts_level = _safe_clabel(ax, CS_level, inline=True, fontsize=font_size, fmt=format_contour_label, colors='black')
                                            if texts_level:
                                                for txt in texts_level:
                                                    txt.set_path_effects([
                                                        patheffects.withStroke(
                                                            linewidth=cfg.LABEL_STROKE_LINEWIDTH,
                                                            foreground=cfg.LABEL_STROKE_FOREGROUND,
                                                            alpha=cfg.LABEL_STROKE_ALPHA
                                                        ),
                                                        patheffects.Normal()
                                                    ])
                                                # 立即检查并隐藏靠近边缘的标签
                                                all_hidden = _hide_labels_near_axes(texts_level, ax)
                                                # 如果这个level的所有标签都被隐藏了，移除对应的等高线
                                                if all_hidden:
                                                    for collection in CS_level.collections:
                                                        collection.remove()
                                    CS_zero = ax.contour(X_log, Y, Z, levels=[0.0], colors='black', linewidths=cfg.HEATMAP_CONTOUR_LINEWIDTH_ZERO)
                                    # 为0等高线添加白色描边效果
                                    for collection in CS_zero.collections:
                                        collection.set_path_effects([
                                            patheffects.withStroke(
                                                linewidth=cfg.CONTOUR_STROKE_LINEWIDTH_ZERO,
                                                foreground=cfg.CONTOUR_STROKE_FOREGROUND,
                                                alpha=cfg.CONTOUR_STROKE_ALPHA
                                            ),
                                            patheffects.Normal()
                                        ])
                                    texts_zero = _safe_clabel(ax, CS_zero, inline=True, fontsize=font_size, fmt=format_contour_label, colors='black')
                                    if texts_zero:
                                        for txt in texts_zero:
                                            txt.set_path_effects([
                                                patheffects.withStroke(
                                                    linewidth=cfg.LABEL_STROKE_LINEWIDTH,
                                                    foreground=cfg.LABEL_STROKE_FOREGROUND,
                                                    alpha=cfg.LABEL_STROKE_ALPHA
                                                ),
                                                patheffects.Normal()
                                            ])
                                        # 立即检查并隐藏靠近边缘的标签
                                        all_hidden = _hide_labels_near_axes(texts_zero, ax)
                                        # 如果这个level的所有标签都被隐藏了，移除对应的等高线
                                        if all_hidden:
                                            for collection in CS_zero.collections:
                                                collection.remove()
                                else:
                                    for level in filtered_levels_new:
                                        CS_level = ax.contour(X_log, Y, Z, levels=[level], colors='black', linewidths=cfg.HEATMAP_CONTOUR_LINEWIDTH)
                                        # 为等高线集合添加白色描边效果
                                        for collection in CS_level.collections:
                                            collection.set_path_effects([
                                                patheffects.withStroke(
                                                    linewidth=cfg.CONTOUR_STROKE_LINEWIDTH,
                                                    foreground=cfg.CONTOUR_STROKE_FOREGROUND,
                                                    alpha=cfg.CONTOUR_STROKE_ALPHA
                                                ),
                                                patheffects.Normal()
                                            ])
                                        texts_level = _safe_clabel(ax, CS_level, inline=True, fontsize=font_size, fmt=format_contour_label, colors='black')
                                        if texts_level:
                                            for txt in texts_level:
                                                txt.set_path_effects([
                                                    patheffects.withStroke(
                                                        linewidth=cfg.LABEL_STROKE_LINEWIDTH,
                                                        foreground=cfg.LABEL_STROKE_FOREGROUND,
                                                        alpha=cfg.LABEL_STROKE_ALPHA
                                                    ),
                                                    patheffects.Normal()
                                                ])
                                            # 立即检查并隐藏靠近边缘的标签
                                            all_hidden = _hide_labels_near_axes(texts_level, ax)
                                            # 如果这个level的所有标签都被隐藏了，移除对应的等高线
                                            if all_hidden:
                                                for collection in CS_level.collections:
                                                    collection.remove()
                            else:
                                # 获取每条等高线的颜色（基于背景色亮度）
                                level_colors_list_new = get_contour_colors_for_levels(X_log, Y, Z, filtered_levels_new, cmap_obj, norm)
                                level_colors_dict_new = {level: level_colors_list_new[i] for i, level in enumerate(filtered_levels_new)}
                                
                                if 0.0 in filtered_levels_new:
                                    non_zero_levels_new = filtered_levels_new[filtered_levels_new != 0.0]
                                    if len(non_zero_levels_new) > 0:
                                        # 为每条等高线绘制不同颜色
                                        for level in non_zero_levels_new:
                                            color = level_colors_dict_new.get(level, 'black')
                                            CS_level = ax.contour(X_log, Y, Z, levels=[level], colors=color, linewidths=cfg.HEATMAP_CONTOUR_LINEWIDTH)
                                            texts_level = _safe_clabel(ax, CS_level, inline=True, fontsize=font_size, fmt=format_contour_label, colors=color)
                                            if texts_level:
                                                # 立即检查并隐藏靠近边缘的标签
                                                all_hidden = _hide_labels_near_axes(texts_level, ax)
                                                # 如果这个level的所有标签都被隐藏了，移除对应的等高线
                                                if all_hidden:
                                                    for collection in CS_level.collections:
                                                        collection.remove()
                                    # 绘制0等高线（加粗）
                                    color_zero = level_colors_dict_new.get(0.0, 'black')
                                    CS_zero = ax.contour(X_log, Y, Z, levels=[0.0], colors=color_zero, linewidths=cfg.HEATMAP_CONTOUR_LINEWIDTH_ZERO)
                                    texts_zero = _safe_clabel(ax, CS_zero, inline=True, fontsize=font_size, fmt=format_contour_label, colors=color_zero)
                                    if texts_zero:
                                        # 立即检查并隐藏靠近边缘的标签
                                        all_hidden = _hide_labels_near_axes(texts_zero, ax)
                                        # 如果这个level的所有标签都被隐藏了，移除对应的等高线
                                        if all_hidden:
                                            for collection in CS_zero.collections:
                                                collection.remove()
                                else:
                                    # 没有0等高线，正常绘制所有等高线
                                    # 为每条等高线绘制不同颜色
                                    for level in filtered_levels_new:
                                        color = level_colors_dict_new.get(level, 'black')
                                        CS_level = ax.contour(X_log, Y, Z, levels=[level], colors=color, linewidths=cfg.HEATMAP_CONTOUR_LINEWIDTH)
                                        texts_level = _safe_clabel(ax, CS_level, inline=True, fontsize=font_size, fmt=format_contour_label, colors=color)
                                        if texts_level:
                                            # 立即检查并隐藏靠近边缘的标签
                                            all_hidden = _hide_labels_near_axes(texts_level, ax)
                                            # 如果这个level的所有标签都被隐藏了，移除对应的等高线
                                            if all_hidden:
                                                for collection in CS_level.collections:
                                                    collection.remove()

    # 与原图一致的显示范围
    xlim_min = np.min(df["R_joint"])*1e9
    xlim_max = np.max(df["R_joint"])*1e9
    ax.set_xlim(np.log10(xlim_min), np.log10(xlim_max))
    ylim_min = np.min(df["Npw"])
    ylim_max = np.max(df["Npw"])
    ax.set_ylim(ylim_min, ylim_max)

    # FIG5_YLOG=1: Npw 纵轴改对数(用于 Fig. 5-1)。
    # 线性 1-200 轴下 Npw 1-10 只占面板高度 4.5%, Npw 1-40(含 LCOE 谷底)占 19.6%,
    # 低 Npw 侧的"充电延迟墙"被压扁到看不出。改 log 后 1-10 占 43.5%、1-40 占 80%,
    # 才能同时看清双侧夹逼: 低 Npw 侧电感大->充电延迟->可用度下降,
    # 高 Npw 侧接头多->低温电耗上升。默认关闭, 不影响既有 Fig. 5/6。
    if _os.environ.get("FIG5_YLOG", "0") == "1":
        ax.set_yscale("log")

    # 移除所有刻度值和标签（与 parasitic heatmap 完全一致）
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlabel("")
    ax.set_ylabel("")

    # 设置边框（与AF热力图一致）
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(3)

    # 可选：标注 baseline A 点（星标，白色填充+黑边，保证在深色上也可见）
    if baseline_point is not None:
        bNpw, bR = baseline_point
        bx = np.log10(bR * 1e9)
        by = bNpw
        ax.scatter([bx], [by], marker="*", s=180, c="white", edgecolors="black", linewidths=1.5, zorder=10)

    # 调整布局（与AF热力图一致：去除白边，让子图完全填满figure）
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    ax.set_position([0, 0, 1, 1])

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
    # 使用与AF热力图相同的保存方式：bbox_inches='tight'，pad_inches=0.0（裁剪空白边缘）
    plt.savefig(plot_path, dpi=cfg.PLOT_DPI, bbox_inches='tight', pad_inches=0.0)
    plt.close(fig)
    print(f"Heatmap saved to: {plot_path}")


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

    # 支持 config 中的自定义/截取版配色名（如 "YlGnBu_trunc_0p8"）
    cmap_resolved = cfg.resolve_cmap(cmap) if hasattr(cfg, "resolve_cmap") else cmap
    sm = plt.cm.ScalarMappable(cmap=cmap_resolved, norm=norm)
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
    
    cbar.ax.tick_params(labelsize=cfg.HEATMAP_FONT_SIZE_COLORBAR)
    cbar.set_label(label, fontsize=cfg.HEATMAP_FONT_SIZE_COLORBAR, labelpad=15)
    
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
    

    # 根据filename_suffix构建文件名后缀
    suffix_str = f"_{filename_suffix}" if filename_suffix else ""
    
    if is_global_ref or ("Global Ref" in label) or ("global" in label.lower()):
        cbar_path = output_dir / f"delta_lcoe_global_heatmap_colorbar{suffix_str}.{cfg.PLOT_FORMAT}"
    elif "min" in label.lower() or "relative to scenario minimum" in label.lower():
        cbar_path = output_dir / f"delta_lcoe_min_heatmap_colorbar{suffix_str}.{cfg.PLOT_FORMAT}"
    else:
        cbar_path = output_dir / f"delta_lcoe_heatmap_colorbar{suffix_str}.{cfg.PLOT_FORMAT}"
    
    plt.savefig(cbar_path, dpi=cfg.PLOT_DPI, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig_cbar)
