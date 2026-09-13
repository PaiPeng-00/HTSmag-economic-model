# magnet_heat_load_analyzer.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import OrderedDict
import sys
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties
import matplotlib.colors as mcolors

# 导入配置和计算模型
try:
    from fusion_tem import device as cfg
    from fusion_tem.cryo import heat_load as hlm
except ImportError as e:
    print(f"[错误] 无法导入所需的模块: {e}")
    print("请确保 config.py 和 heat_load_model.py 文件与此脚本位于同一目录下。")
    sys.exit(1)

# =============================================================================
# 绘图字号配置 - 统一管理所有绘图函数的字号设置
# =============================================================================
FONT_CONFIG = {
    'base_font_size': 20,           # 基础字号
    'axes_label_size': 20,          # 坐标轴标签字号
    'tick_label_size': 20,          # 刻度标签字号
    'legend_font_size': 20,         # 图例字号（主图中的图例）
    'legend_standalone_size': 20,   # 独立图例字号（须与面板内字号一致，见 LEGEND_MAX_WIDTH_PT）
    'percent_label_size': 20,       # 百分比标注字号
    'total_label_size': 20,         # 总热负荷标注字号
    'config_name_size': 20,         # 配置名称标注字号
}
# 图例宽度上限(pt)。4.11 拼图时 available_width = total_w - left(40) - right(30)
# ≈ 1330 pt；图例一旦超宽就会被整体缩放，字号随之偏离上面的设定值
# (曾把 24 pt 压成 18.93 pt，与面板内的 20 pt 并存 -> 三种字号)。
# 留 30 pt 余量，并由 save_legend_only 自动选列数来满足它。
LEGEND_MAX_WIDTH_PT = 1300.0
LEGEND_NCOL_CANDIDATES = (4, 3, 2)
# 图例条目间距。3 列时相邻条目仍有轻微重叠 -> 加大列间距与行间距。
# 加宽后若超出 LEGEND_MAX_WIDTH_PT, 自动降到 2 列(12 项 -> 6 行), 不会触发缩放。
LEGEND_COLUMNSPACING = 3.0   # 列间距(原 1.5)
LEGEND_LABELSPACING = 1.1    # 行间距(原 matplotlib 默认 0.5)
LEGEND_HANDLETEXTPAD = 0.8   # 色条与文字的间距(原 0.5)
# 为了绘制时间序列热负荷图，需要设置更大的字号
FONT_CONFIG_TIME_SERIES = {
    'base_font_size': 24,           # 基础字号
    'axes_label_size': 24,          # 坐标轴标签字号
    'tick_label_size': 24,          # 刻度标签字号
    'legend_font_size': 24,         # 图例字号（主图中的图例）
    'legend_standalone_size': 28,   # 独立图例字号（单独保存的图例，需要更大以保持视觉一致）
    'percent_label_size': 22,       # 百分比标注字号
    'total_label_size': 24,         # 总热负荷标注字号
    'config_name_size': 24,         # 配置名称标注字号
}

# 应用全局matplotlib设置
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': FONT_CONFIG['base_font_size'],
    'axes.labelsize': FONT_CONFIG['axes_label_size'],
    'xtick.labelsize': FONT_CONFIG['tick_label_size'],
    'ytick.labelsize': FONT_CONFIG['tick_label_size'],
    'legend.fontsize': FONT_CONFIG['legend_font_size'],
    'axes.linewidth': 3,
})
# =============================================================================
# 1. 热负荷计算编排模块 (HeatLoadCalculator)
# =============================================================================

# ---------------------------------------------------------------------------
# 出版标签格式（2026-07-26 由 scratchpad/build_figures_160mm_v6_fig23_clean 上移）
#   - 总热负荷不带单位后缀 "W"（单位在图注中给出）
#   - 组分百分比取整到整数，ROUND_HALF_UP，与原 v6 的 Decimal 口径一致
# 原先这两项要靠对冻结 SVG 做字形手术才能实现，现在在生成时直接定型。
# ---------------------------------------------------------------------------
from decimal import Decimal, ROUND_HALF_UP as _ROUND_HALF_UP


def _fmt_total(watts: float) -> str:
    """总热负荷标签：千分位，无单位后缀。"""
    return f"{watts:,.0f}"


def _fmt_pct(value_pct: float) -> str:
    """组分占比标签：四舍五入到整数百分比。"""
    q = Decimal(repr(float(value_pct))).quantize(Decimal("1"), rounding=_ROUND_HALF_UP)
    return f"{q}%"


class HeatLoadCalculator:
    """
    根据不同的工况和磁体配置，调用 heat_load_model.py 中的函数来计算热负荷。
    """
    def _preprocess_comsol_data(self, T_op, Npw, rhot_uOhm_cm2, loss_file_name):
        """
        从单个Excel文件的sheet中读取COMSOL数据。
        不同温度的数据存储在同一个文件的不同sheet中（如 "4.2K", "10K", "20K"）。
        路径：cfg.HEAT_DATA_DIR / loss_file_name
        
        注意：此方法已更新为使用单个文件多sheet的格式，与 _preprocess_charging_data_by_temp 保持一致。
        """
        # 新的路径结构：直接使用HEAT_DATA_DIR，不再按温度分文件夹
        file_path = cfg.HEAT_DATA_DIR / loss_file_name
        
        if not file_path.exists():
            print(f"  [!] 警告: COMSOL 数据文件未找到: {file_path}")
            return pd.Series(dtype=float)
        
        # Sheet名称：根据温度确定（如 "4.2K", "10K", "20K"）
        sheet_name = f"{T_op}K"
        
        try:
            # 尝试读取指定sheet，如果不存在则尝试读取第一个sheet
            try:
                df = pd.read_excel(file_path, sheet_name=sheet_name, header=0, skiprows=lambda x: x < 5)
            except (ValueError, KeyError):
                # 如果指定sheet不存在，尝试读取第一个sheet
                df = pd.read_excel(file_path, sheet_name=0, header=0, skiprows=lambda x: x < 5)
                print(f"  [!] 警告: Sheet '{sheet_name}' not found in {file_path}, using first sheet")
            
            # 数据文件格式：
            # - 第0列：rhot值 (uOhm·cm²)
            # - 第1列：Npw值
            # - 第2列：时间（秒）
            # - 第3列：损耗值（W）
            
            # 筛选出匹配Npw和rhot的数据
            rhot_col = df.columns[0]
            npw_col = df.columns[1]
            
            # 找到最接近的rhot值（因为可能不是精确匹配）
            available_rhots = df[rhot_col].unique()
            closest_rhot = available_rhots[np.argmin(np.abs(available_rhots - rhot_uOhm_cm2))]
            
            # 筛选出匹配Npw和rhot的数据
            df_filtered = df[(df[rhot_col] == closest_rhot) & (df[npw_col] == Npw)].copy()
            
            if df_filtered.empty:
                print(f"  [!] 警告: 未找到匹配的数据 (Npw={Npw}, rhot={rhot_uOhm_cm2}uOhm·cm²) in {file_path}")
                return pd.Series(dtype=float)
            
            # 时间列在第2列（索引为2），单位为秒，转换为小时
            df_filtered['Time (h)'] = df_filtered.iloc[:, 2] / 3600
            # 损耗值列在第3列（索引为3）
            value_col_name = df_filtered.columns[3]
            
            return df_filtered.set_index('Time (h)')[value_col_name]

        except Exception as e:
            print(f"  [!] 错误: 读取COMSOL文件 {file_path} 失败: {e}")
            return pd.Series(dtype=float)

    def _preprocess_charging_data_by_temp(self, T_op, Npw, rhot_uOhm_cm2, loss_file_name):
        """
        从单个Excel文件的sheet中读取充电损耗数据。
        不同温度的数据存储在同一个文件的不同sheet中（如 "4.2K", "10K", "20K"）。
        路径：cfg.HEAT_DATA_DIR / loss_file_name
        投稿文件名与 Data S2/S3 保持一致。
        
        数据文件格式：
        - 第0列：rhot值 (uOhm·cm²)
        - 第1列：Npw值
        - 第2列：时间（秒）
        - 第3列：损耗值（W）
        
        需要根据Npw和rhot筛选出对应的数据。
        """
        # 新的路径结构：直接使用HEAT_DATA_DIR，不再按温度分文件夹
        file_path = cfg.HEAT_DATA_DIR / loss_file_name
        
        if not file_path.exists():
            print(f"  [!] 警告: 充电损耗数据文件未找到: {file_path}")
            return pd.Series(dtype=float)
        
        # Sheet名称：根据温度确定（如 "4.2K", "10K", "20K"）
        sheet_name = f"{T_op}K"
        
        try:
            # 尝试读取指定sheet，如果不存在则尝试读取第一个sheet
            try:
                df = pd.read_excel(file_path, sheet_name=sheet_name, header=0, skiprows=lambda x: x < 5)
            except (ValueError, KeyError):
                # 如果指定sheet不存在，尝试读取第一个sheet
                df = pd.read_excel(file_path, sheet_name=0, header=0, skiprows=lambda x: x < 5)
                print(f"  [!] 警告: Sheet '{sheet_name}' not found in {file_path}, using first sheet")
            
            # 筛选出匹配Npw和rhot的数据
            # 第0列是rhot值，第1列是Npw值
            rhot_col = df.columns[0]
            npw_col = df.columns[1]
            
            # 找到最接近的rhot值（因为可能不是精确匹配）
            available_rhots = df[rhot_col].unique()
            closest_rhot = available_rhots[np.argmin(np.abs(available_rhots - rhot_uOhm_cm2))]
            
            # 筛选出匹配Npw和rhot的数据，使用.copy()创建副本以避免SettingWithCopyWarning
            df_filtered = df[(df[rhot_col] == closest_rhot) & (df[npw_col] == Npw)].copy()
            
            if df_filtered.empty:
                print(f"  [!] 警告: 未找到匹配的数据 (Npw={Npw}, rhot={rhot_uOhm_cm2}uOhm·cm²) in {file_path}")
                return pd.Series(dtype=float)
            
            # 时间列在第2列（索引为2），单位为秒，转换为小时
            df_filtered['Time (h)'] = df_filtered.iloc[:, 2] / 3600
            # 损耗值列在第3列（索引为3）
            value_col_name = df_filtered.columns[3] # 损耗值列
            
            return df_filtered.set_index('Time (h)')[value_col_name]

        except Exception as e:
            print(f"  [!] 错误: 读取充电损耗文件 {file_path} 失败: {e}")
            return pd.Series(dtype=float)

    def _get_power_at_time(self, df_series, t):
        """简单的线性插值函数，用于获取特定时间的损耗。"""
        if df_series is None or df_series.empty: return 0
        times = df_series.index.to_numpy()
        idx = np.searchsorted(times, t, side='left')
        if idx == 0: return df_series.iloc[0]
        if idx >= len(times): return df_series.iloc[-1]
        return df_series.iloc[idx - 1]

    def calculate_loads(self, mode: str, Npw, R_p2p_joint: float, 
                        T_op: float, Ip: float, L_tot: float, rhot: float = 0):
        """
        计算热负荷的主函数。
        """
        # --- T_op 温区热负荷 ---
        top_loads = OrderedDict()
        
        is_operating = (mode in ['operation', 'charging'])
        
        # 调用 heat_load_model 中的函数
        #print(Npw, R_p2p_joint, Ip, L_tot)
        top_loads["Coil-to-coil joint Joule heat"] = hlm.pancake_joint_heat(Npw, R_p2p_joint, Ip) if is_operating else 0
        top_loads["Su-su joint Joule heat"] = hlm.coil_internal_joint_heat(
            Npw, L_total_m=L_tot, L_single_tape=cfg.LEN_PER_SINGEL_REBCO, Ip=Ip, R_ss=cfg.R_SU_JOINT) if is_operating else 0
        top_loads["Nuclear heat"] = hlm.nuclear_heating() if mode == 'operation' else 0
        
        top_loads["Radiative heat"] = hlm.radiation_heat(cfg.A_cryostat,  cfg.eps, cfg.T_HIGH, T_op)
        
        Q_cond_77K, Q_joule_77K, Q_hts_cond = hlm.current_lead_heat(Npw=Npw, Top=T_op, Ip=Ip)
        top_loads["Conductive heat - HTS current leads"] = Q_hts_cond

        # 其他静态热负荷
        coolant_pipes = hlm.pipe_heat(cfg.N_cool_pipe, cfg.d_in_cool, cfg.d_out_cool, cfg.L_cool, cfg.T_HIGH, T_op)
        other_pipes = hlm.pipe_heat(cfg.N_aux_pipe, cfg.d_in_aux, cfg.d_out_aux, cfg.L_aux, cfg.T_HIGH, T_op)
        other_static = hlm.misc_heat()
        
        top_loads["Conductive heat - coolant transfer lines"] = coolant_pipes
        top_loads["Conductive heat - auxiliary leads"] = other_pipes
        top_loads["Other sources"] = other_static

        # 暂态损耗 (仅充电模式)
        if mode == 'charging':
            # 对于充电模式，从工作簿对应温度 sheet 读取数据
            # 4.2K、10K、20K 数据分别位于 Data S3 和 Data S2 工作簿的同名温度 sheet
            # 文件包含所有Npw和rhot组合的数据，需要根据参数筛选
            rhot_uOhm_cm2 = rhot * 1e10  # 转换为uOhm·cm²
            mag_loss_file = "data_s3_magnetization_loss.xlsx"
            radial_loss_file = "data_s2_radial_loss.xlsx"
            mag_loss_series = self._preprocess_charging_data_by_temp(T_op, Npw, rhot_uOhm_cm2, mag_loss_file)
            radial_loss_series = self._preprocess_charging_data_by_temp(T_op, Npw, rhot_uOhm_cm2, radial_loss_file)
            
            snapshot_time = cfg.CHARGE_HOURS
            top_loads["Magnetisation loss"] = self._get_power_at_time(mag_loss_series, snapshot_time)
            top_loads["Radial loss"] = self._get_power_at_time(radial_loss_series, snapshot_time)
        else:
            top_loads["Magnetisation loss"] = 0
            top_loads["Radial loss"] = 0

        # --- 77K 温区热负荷 ---
        loads_77k = OrderedDict()    
        loads_77k["Joule heat - resistive current leads"] = Q_joule_77K if is_operating else 0
        loads_77k["Conductive heat - resistive current leads"] = Q_cond_77K

        return pd.Series(top_loads), pd.Series(loads_77k)
    # +++ 新增方法: 计算充电过程的时间序列热负荷 +++
    def calculate_transient_timeseries(self, Npw, R_p2p_joint: float, T_op: float, Ip: float, rhot: float, L_tot: float):
        """
        计算充电全过程的时间序列热负荷 (仅 T_op 温区)。
        """
        # 定义时间点
        time_points = np.linspace(0, cfg.CHARGE_HOURS * 1.2, 101) # 多算20%时间看稳态
        
        # 预加载随时间变化的损耗数据
        # 对于充电模式，从工作簿对应温度 sheet 读取数据
        rhot_uOhm_cm2 = rhot * 1e10  # 转换为uOhm·cm²
        mag_loss_file = "data_s3_magnetization_loss.xlsx"
        radial_loss_file = "data_s2_radial_loss.xlsx"
        mag_loss_series = self._preprocess_charging_data_by_temp(T_op, Npw, rhot_uOhm_cm2, mag_loss_file)
        radial_loss_series = self._preprocess_charging_data_by_temp(T_op, Npw, rhot_uOhm_cm2, radial_loss_file)

        all_timesteps_data = []
        for t in time_points:
            loads = OrderedDict()
            current_ratio = min(t / cfg.CHARGE_HOURS, 1.0)
            current = Ip * current_ratio
            # 随时间变化的热源
            loads["Coil-to-coil joint Joule heat"] = hlm.pancake_joint_heat(Npw, R_p2p_joint, Ip) * (current_ratio**2)
            loads["Su-su joint Joule heat"] = hlm.coil_internal_joint_heat(Npw, L_total_m=L_tot, Ip=Ip) * (current_ratio**2)


            loads["Magnetisation loss"] = self._get_power_at_time(mag_loss_series, t)
            loads["Radial loss"] = self._get_power_at_time(radial_loss_series, t)
            
            # 静态热源 (不随时间变化)
            loads["Radiative heat"] = hlm.radiation_heat(cfg.A_cryostat,  cfg.eps, cfg.T_HIGH, T_op)
            Q_cond_77K, Q_joule_77K, Q_hts_cond = hlm.current_lead_heat(Npw=Npw, Top=T_op, Ip=Ip)
            loads["Conductive heat - HTS current leads"] = Q_hts_cond
            loads["Conductive heat - coolant transfer lines"] = hlm.pipe_heat(cfg.N_cool_pipe, cfg.d_in_cool, cfg.d_out_cool, cfg.L_cool, cfg.T_HIGH, T_op)
            loads["Conductive heat - auxiliary leads"] = hlm.pipe_heat(cfg.N_aux_pipe, cfg.d_in_aux, cfg.d_out_aux, cfg.L_aux, cfg.T_HIGH, T_op)
            loads["Other sources"] = hlm.misc_heat()

            # 核热在充电时为0
            loads["Nuclear heat"] = 0
            
            all_timesteps_data.append(loads)
            
        df = pd.DataFrame(all_timesteps_data, index=np.round(time_points, 3))
        df.index.name = 'Time (h)'
        return df

# =============================================================================
# 2. 绘图模块 (HeatLoadPlotter) - (与上一版基本相同)
# =============================================================================
class HeatLoadPlotter:
    """
    负责所有绘图任务，特别是双柱并列堆叠图。
    """
    def __init__(self, output_dir: Path):
        self.output_path = output_dir
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.COLORS = ['#CCE092', '#8DAFDB', '#FABA90', '#FFD865',
                        '#B4C7E7', '#EDEDED', '#2F5597', '#F8B8CC', 
                        '#FBE0EA', '#D3C6F1', '#FFDAC1', '#C4EEDF']
        self.HEAT_SOURCE_COLORS = OrderedDict([
            ("Coil-to-coil joint Joule heat", self.COLORS[0]),
            ("Su-su joint Joule heat", self.COLORS[1]),
            ("Nuclear heat", self.COLORS[2]),
            ("Magnetisation loss", self.COLORS[3]),
            ("Radial loss", self.COLORS[4]),
            ("Radiative heat", self.COLORS[5]),
            ("Conductive heat - HTS current leads", self.COLORS[6]),
            ("Conductive heat - coolant transfer lines", self.COLORS[7]),
            ("Conductive heat - auxiliary leads", self.COLORS[8]),
            ("Other sources", self.COLORS[9]),
            ("Joule heat - resistive current leads", self.COLORS[10]),
            ("Conductive heat - resistive current leads", self.COLORS[11])
        ])

    def _get_text_color_for_background(self, bg_color):
        """
        根据背景颜色返回合适的文本颜色（白色或黑色）。
        如果背景颜色很深，返回白色；否则返回黑色。
        
        Args:
            bg_color: 背景颜色（可以是hex字符串、RGB元组等）
            
        Returns:
            str: 'white' 或 'black'
        """
        # 将颜色转换为RGB值（0-1范围）
        try:
            rgb = mcolors.to_rgb(bg_color)
        except (ValueError, TypeError):
            # 如果颜色转换失败，默认返回黑色
            return 'black'
        
        # 计算相对亮度（使用ITU-R BT.709标准）
        # L = 0.299*R + 0.587*G + 0.114*B
        # 这里rgb已经是0-1范围，所以直接计算
        luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        
        # 如果亮度小于0.5（即较暗），使用白色文本；否则使用黑色文本
        return 'white' if luminance < 0.5 else 'black'

    def plot_dual_stacked_bar(self, top_loads: pd.Series, loads_77k: pd.Series, config_name: str, T_op: float, mode: str):
        fig, ax = plt.subplots(figsize=(1.2, 6))
        bar_width, x_pos = 0.3, 0

        bottom_top = 0
        for source, value in top_loads.items():
            if value > 0:
                color = self.HEAT_SOURCE_COLORS.get(source, '#000000')
                ax.bar(x_pos - bar_width/2, value, bar_width, bottom=bottom_top,
                       label=source, color=color,  edgecolor='black')
                bottom_top += value

        bottom_77k = 0
        for source, value in loads_77k.items():
            if value > 0:
                color = self.HEAT_SOURCE_COLORS.get(source, '#000000')
                ax.bar(x_pos + bar_width/2, value, bar_width, bottom=bottom_77k,
                       label=source, color=color, edgecolor='black')
                bottom_77k += value

        total_top, total_77k = top_loads.sum(), loads_77k.sum()
        ax.text(x_pos - bar_width/2, total_top, _fmt_total(total_top), 
                ha='center', va='bottom', fontsize=FONT_CONFIG['total_label_size'])
        ax.text(x_pos + bar_width/2, total_77k, _fmt_total(total_77k), 
                ha='center', va='bottom', fontsize=FONT_CONFIG['total_label_size'])
        
        ax.set_xticks([x_pos - bar_width/2, x_pos + bar_width/2])
        ax.set_xticklabels([f'T_op = {T_op} K', '77 K Level'])
        ax.set_ylabel('Heat Load (W)')
        
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left',
                  fontsize=FONT_CONFIG['legend_font_size'])

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()

        filename = f"{config_name}_{mode}_{T_op}K.svg"
        save_path = self.output_path / filename
        plt.savefig(save_path, dpi=cfg.PLOT_DPI, bbox_inches='tight',transparent=True)
        print(f"  [✔] 图像已保存: {save_path}")
        plt.close(fig)

        # +++ 新增方法 1: 生成独立图例 +++
    def save_legend_only(self):
        """
        生成一个只包含所有热源图例的独立图片。
        """
        handles = [
            Line2D([0], [0], color=color, lw=10, label=label)
            for label, color in self.HEAT_SOURCE_COLORS.items()
        ]
        
        # 列数自动选择：从多到少试，取第一个宽度不超过 LEGEND_MAX_WIDTH_PT 的。
        # 目的是让 4.11 拼图时无需缩放图例——一旦缩放，图例字号就不再等于
        # FONT_CONFIG['legend_standalone_size']，会与面板内字号打架。
        # 列数少了自然行数多，条目间也不再挤到重叠。
        n_items = len(handles)
        legend_font = FontProperties(size=FONT_CONFIG['legend_standalone_size'], family='Arial')

        fig_legend = None
        for n_cols in LEGEND_NCOL_CANDIDATES:
            n_rows = (n_items + n_cols - 1) // n_cols
            if fig_legend is not None:
                plt.close(fig_legend)
            # 画布给足宽度，多余部分靠 bbox_inches='tight' 裁掉
            fig_legend = plt.figure(figsize=(30, max(2.0, n_rows * 1.4)))
            legend = fig_legend.legend(handles=handles, loc='center', frameon=True, edgecolor='white',
                                       ncol=n_cols, prop=legend_font, handlelength=2.0,
                                       handletextpad=LEGEND_HANDLETEXTPAD,
                                       columnspacing=LEGEND_COLUMNSPACING,
                                       labelspacing=LEGEND_LABELSPACING)
            fig_legend.canvas.draw()
            # window_extent 单位是像素，按 figure dpi 折算成 pt
            w_pt = (legend.get_window_extent(fig_legend.canvas.get_renderer()).width
                    * 72.0 / fig_legend.dpi)
            if w_pt <= LEGEND_MAX_WIDTH_PT or n_cols == LEGEND_NCOL_CANDIDATES[-1]:
                break
        print(f"  图例: {n_items} 项 / {n_cols} 列 / {n_rows} 行，宽 {w_pt:.0f}pt "
              f"(上限 {LEGEND_MAX_WIDTH_PT:.0f}pt)"
              + ("" if w_pt <= LEGEND_MAX_WIDTH_PT else "  [警告] 仍超宽，拼图时会被缩放"))

        save_path = self.output_path / cfg.LEGEND_IMAGE_FILE
        # 先保存，然后裁剪空白边缘
        fig_legend.savefig(save_path, dpi=cfg.PLOT_DPI, bbox_inches='tight', pad_inches=0.1, transparent=True)
        plt.close(fig_legend)
        
        # 裁剪SVG文件的空白边缘
        try:
            import xml.etree.ElementTree as ET
            import re
            
            def _parse_size(size_str: str) -> float:
                """解析SVG尺寸字符串为数值"""
                if size_str is None:
                    return 0.0
                m = re.findall(r"[0-9.]+", str(size_str))
                return float(m[0]) if m else 0.0
            
            # 读取保存的SVG文件
            tree = ET.parse(str(save_path))
            root_svg = tree.getroot()
            
            # 获取SVG命名空间
            svg_ns = None
            for prefix, uri in root_svg.attrib.items():
                if prefix.startswith('xmlns') and 'svg' in uri.lower():
                    svg_ns = uri
                    break
            if svg_ns is None:
                svg_ns = 'http://www.w3.org/2000/svg'
            
            ET.register_namespace('', svg_ns)
            
            # 获取当前尺寸和viewBox
            current_width = _parse_size(root_svg.get('width', '0'))
            current_height = _parse_size(root_svg.get('height', '0'))
            viewbox = root_svg.get('viewBox', f"0 0 {current_width} {current_height}")
            viewbox_parts = [float(x) for x in viewbox.split()]
            vb_x, vb_y, vb_w, vb_h = viewbox_parts if len(viewbox_parts) == 4 else (0, 0, current_width, current_height)
            
            # 使用图例的bbox信息（如果可用）
            # 由于matplotlib已经使用了bbox_inches='tight'，viewBox应该已经比较紧凑
            # 但我们仍然可以进一步优化，通过分析实际内容
            
            # 查找所有文本和图形元素，计算边界
            min_x = float('inf')
            min_y = float('inf')
            max_x = float('-inf')
            max_y = float('-inf')
            
            def find_bounds(elem, ns, offset_x=0, offset_y=0):
                """递归查找元素的边界"""
                nonlocal min_x, min_y, max_x, max_y
                
                # 处理transform
                tx, ty = offset_x, offset_y
                if 'transform' in elem.attrib:
                    transform_str = elem.attrib['transform']
                    # 提取translate
                    translate_match = re.search(r'translate\(([^)]+)\)', transform_str)
                    if translate_match:
                        coords = [float(x.strip()) for x in translate_match.group(1).replace(',', ' ').split()]
                        if len(coords) >= 2:
                            tx += coords[0]
                            ty += coords[1]
                
                # 检查元素位置
                x, y = None, None
                if 'x' in elem.attrib:
                    try:
                        x = float(elem.attrib['x']) + tx
                    except (ValueError, TypeError):
                        pass
                if 'y' in elem.attrib:
                    try:
                        y = float(elem.attrib['y']) + ty
                    except (ValueError, TypeError):
                        pass
                
                # 对于文本元素
                if elem.tag.endswith('text') and elem.text and elem.text.strip():
                    if x is not None and y is not None:
                        font_size = float(elem.attrib.get('font-size', 12))
                        text = elem.text.strip()
                        text_width = len(text) * font_size * 0.6
                        text_height = font_size * 1.2
                        min_x = min(min_x, x)
                        min_y = min(min_y, y - text_height)
                        max_x = max(max_x, x + text_width)
                        max_y = max(max_y, y)
                
                # 对于矩形、路径等图形元素
                if 'width' in elem.attrib and 'height' in elem.attrib:
                    try:
                        w = float(elem.attrib['width'])
                        h = float(elem.attrib['height'])
                        if x is not None and y is not None:
                            min_x = min(min_x, x)
                            min_y = min(min_y, y)
                            max_x = max(max_x, x + w)
                            max_y = max(max_y, y + h)
                    except (ValueError, TypeError):
                        pass
                
                # 递归处理子元素
                for child in elem:
                    find_bounds(child, ns, tx, ty)
            
            # 查找所有元素的边界
            for elem in root_svg:
                find_bounds(elem, svg_ns)
            
            # 如果找到了有效边界，进行裁剪
            if min_x != float('inf') and min_y != float('inf'):
                # 添加边距（保留一些空白）
                padding = 20
                min_x = max(0, min_x - padding)
                min_y = max(0, min_y - padding)
                max_x = min(vb_w, max_x + padding)
                max_y = min(vb_h, max_y + padding)
                
                # 计算裁剪后的尺寸
                cropped_w = max_x - min_x
                cropped_h = max_y - min_y
                
                # 更新viewBox和尺寸
                root_svg.set('viewBox', f"{min_x} {min_y} {cropped_w} {cropped_h}")
                root_svg.set('width', f"{cropped_w}pt")
                root_svg.set('height', f"{cropped_h}pt")
                
                # 保存裁剪后的文件
                tree.write(str(save_path), encoding='utf-8', xml_declaration=True)
                #print(f"\n[✔] 独立图例已保存到: {save_path} (已裁剪空白边缘: {current_width:.1f}pt x {current_height:.1f}pt -> {cropped_w:.1f}pt x {cropped_h:.1f}pt)")
            else:
                # 如果无法找到边界，至少减小pad_inches后的边距
                # 使用更小的viewBox（基于当前viewBox，但稍微缩小）
                padding = 10
                root_svg.set('viewBox', f"{padding} {padding} {vb_w - 2*padding} {vb_h - 2*padding}")
                root_svg.set('width', f"{vb_w - 2*padding}pt")
                root_svg.set('height', f"{vb_h - 2*padding}pt")
                tree.write(str(save_path), encoding='utf-8', xml_declaration=True)
                #print(f"\n[✔] 独立图例已保存到: {save_path} (已裁剪部分空白边缘)")
        except Exception as e:
            #print(f"\n[✔] 独立图例已保存到: {save_path}")
            print(f"  [警告] 裁剪空白边缘时出错: {e}")

    # +++ 新增方法 2: 绘制相对值堆叠图 +++
    def plot_dual_normalized_stacked_bar(self, top_loads: pd.Series, loads_77k: pd.Series, config_name: str, T_op: float, mode: str):
        
        output_path = self.output_path
        output_path.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(2.5, 8))
        bar_width, x_pos = 0.35, 0
        
        total_top = top_loads.sum()
        if total_top > 0:
            top_loads_pct = (top_loads / total_top) * 100
            bottom_top = 0
            for source, value_pct in top_loads_pct.items():
                if value_pct > 0:
                    color = self.HEAT_SOURCE_COLORS.get(source, '#000000')
                    ax.bar(x_pos - bar_width/2, value_pct, bar_width, bottom=bottom_top,
                           color=color, edgecolor='black')
                    if value_pct > 5: # 仅标注大于5%的项
                        # 根据背景颜色自动选择文本颜色
                        text_color = self._get_text_color_for_background(color)
                        ax.text(x_pos - bar_width/2, bottom_top + value_pct/2, _fmt_pct(value_pct), 
                                ha='center', va='center', fontsize=FONT_CONFIG['percent_label_size'], color=text_color)
                    bottom_top += value_pct
            # 与柱子同轴：柱心在 x_pos - bar_width/2（见上面的 ax.bar）。
            # 原先写 bar_width*0.7，等于朝外偏了 0.2 个柱宽。
            ax.text(x_pos - bar_width/2, 101, _fmt_total(total_top),
                    ha='center', va='bottom', fontsize=FONT_CONFIG['total_label_size'])
        
        total_77k = loads_77k.sum()
        if total_77k > 0:
            loads_77k_pct = (loads_77k / total_77k) * 100
            bottom_77k = 0
            for source, value_pct in loads_77k_pct.items():
                if value_pct > 0:
                    color = self.HEAT_SOURCE_COLORS.get(source, '#000000')
                    ax.bar(x_pos + bar_width/2, value_pct, bar_width, bottom=bottom_77k,
                           color=color, edgecolor='black',)
                    if value_pct > 5:
                        # 根据背景颜色自动选择文本颜色
                        text_color = self._get_text_color_for_background(color)
                        ax.text(x_pos + bar_width/2, bottom_77k + value_pct/2, _fmt_pct(value_pct), 
                                ha='center', va='center', fontsize=FONT_CONFIG['percent_label_size'], color=text_color)
                    bottom_77k += value_pct
            ax.text(x_pos + bar_width/2, 101, _fmt_total(total_77k),
                    ha='center', va='bottom', fontsize=FONT_CONFIG['total_label_size'])

        ax.set_ylim(0, 115)
        ax.axis('off') # 关闭坐标轴
        #ax.set_title(f'Heat Load for {config_name} ({mode.capitalize()} Mode) - Relative')
        
        # 在底部手动添加标签
        #ax.text(x_pos - bar_width/2, -5, f'T_op = {T_op} K', ha='center', va='top', fontsize=FONT_CONFIG['config_name_size'])
        ax.text(x_pos , -5, config_name, ha='center', va='top', fontsize=FONT_CONFIG['config_name_size'])

        plt.tight_layout()

        filename = f"{config_name}_{mode}_{T_op}K_relative.svg"
        save_path = output_path / filename
        plt.savefig(save_path, dpi=cfg.PLOT_DPI, bbox_inches='tight',transparent=True)
        #print(f"  [✔] 相对值图像已保存: {save_path}")
        plt.close(fig)

        # +++ 新增方法 3: 绘制随时间变化的堆叠面积图 +++
    def plot_stacked_area_over_time(self, df: pd.DataFrame, config_name: str,  T_op: float):
        """
        绘制 T_op 温区热负荷随时间变化的堆叠面积图。
        """
        if df.empty:
            print(f"    [!] 警告: 用于绘制时间序列图的数据为空，跳过。")
            return
        
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # 准备数据和颜色
        x = df.index
        # 注意：这里需要转置DataFrame，使得每一行是一个热源的时间序列
        y = df.T.values 
        labels = df.columns
        colors = [self.HEAT_SOURCE_COLORS.get(label, '#000000') for label in labels]

        # 绘制堆叠面积图
        ax.stackplot(x, y, labels=labels, colors=colors, alpha=0.8)

        # 绘制总热负荷曲线
        total_heat = df.sum(axis=1)
        ax.plot(x, total_heat, color='black', linestyle='--', linewidth=2, label='Total Heat Load')

        # 标注斜坡结束线
        ax.axvline(x=cfg.CHARGE_HOURS, color='gray', linestyle='-.', linewidth=1.5, label='End of Current Ramp')

        ax.set_xlabel('Time (h)')
        ax.set_ylabel('Heat Load (W)')
        #ax.set_title(f'T_op Heat Load vs. Time for {config_name} (Charging Mode, T_op={T_op}K)')
        ax.grid(True, which='both', linestyle=':', linewidth=0.7)
        ax.set_xlim(left=0, right=cfg.CHARGE_HOURS*1.2)
        ax.set_xlabel('Time (h)', fontsize=FONT_CONFIG_TIME_SERIES['axes_label_size'])
        ax.set_ylabel('Heat Load (W)', fontsize=FONT_CONFIG_TIME_SERIES['total_label_size'])
        ax.set_ylim(bottom=0)

        # 获取图例项并反转顺序以匹配堆叠顺序
        handles, labels = ax.get_legend_handles_labels()
        #ax.legend(handles[::-1], labels[::-1], loc='upper left', bbox_to_anchor=(1.02, 1.0))

        plt.tight_layout() # 为图例留出空间
        
        filename = f"{config_name}_charging_{T_op}K_timeseries.svg"
        save_path = self.output_path / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        #print(f"  [✔] 时间序列图像已保存: {save_path}")
        plt.close(fig)



class ExcelReporter:
    """
    收集所有计算结果并将其写入一个Excel文件。
    """
    def __init__(self, filename: Path):
        self.filename = filename
        self.records = [] # 用于存储每一行的数据

    def add_record(self, config_name: str, mode: str, T_op: float, params: dict, 
                     top_loads: pd.Series, loads_77k: pd.Series):
        """
        为一次计算结果添加一条记录。

        Args:
            config_name (str): 配置名称
            mode (str): 运行工况
            T_op (float): 运行温度
            params (dict): 输入的参数字典
            top_loads (pd.Series): T_op 温区的热负荷
            loads_77k (pd.Series): 77K 温区的热负荷
        """
        # 基础信息 (输入参数)
        record = OrderedDict()
        record['Config Name'] = config_name
        record['Mode'] = mode
        record['T_op (K)'] = T_op
        record.update(params) # 添加 Npw, R_p2p_joint 等参数

        total_top = top_loads.sum()
        total_77k = loads_77k.sum()

        # 计算结果 (输出热负荷)
        # 将Series转换为字典，并为列名添加后缀以区分
        for source, value in top_loads.items():
            record[f'{source} (W)'] = value
            pct_value = (value / total_top * 100) if total_top else 0
            record[f'{source} (%)'] = f"{pct_value:.2f}%"
        for source, value in loads_77k.items():
            record[f'{source} (W)'] = value
            pct_value = (value / total_77k * 100) if total_77k else 0
            record[f'{source} (%)'] = f"{pct_value:.2f}%"

        # 添加总计
        record['Total T_op Load (W)'] = total_top
        record['Total 77K Load (W)'] = total_77k

        # Top=20 K情况下，需要的制冷剂流速
        record['H2 flow rate (g/s)'] = top_loads.sum() / cfg.ENTHALPY_H2_KJ_KG
        record['He flow rate (g/s)'] = top_loads.sum() / cfg.ENTHALPY_HE_KJ_KG
        
        self.records.append(record)

    def save(self):
        """
        将所有收集到的记录保存到Excel文件。
        """
        if not self.records:
            print("[信息] 没有记录可以保存到Excel。")
            return

        # 将记录列表转换为DataFrame
        df = pd.DataFrame(self.records)
        df.fillna(0, inplace=True) # 用0填充缺失值 (例如某些工况下不存在的热源)

        try:
            df.to_excel(self.filename, index=False, engine='openpyxl')
            print(f"\n[✔] 汇总报告已成功保存到: {self.filename}")
        except Exception as e:
            print(f"\n[错误] 保存Excel文件失败: {e}")
            print("       请确保您已安装 'openpyxl' 库 (pip install openpyxl)")
# =============================================================================
# 3. 主执行模块
# =============================================================================
if __name__ == "__main__":
    
    # -- 定义磁体配置 --
    configs = {
        'A':  {'npw': 5, 'rhot': 5000e-10}, 
        'B':  {'npw': 10,  'rhot': 5000e-10},
        'C':  {'npw': 20, 'rhot': 5000e-10},
        'D':  {'npw': 200, 'rhot': 5000e-10},


        'A1': {'npw': 5, 'rhot': 5000e-10},
        'B1': {'npw': 10, 'rhot': 5000e-10}, 'B2': {'npw': 10, 'rhot': 1500e-10},
        
        'C1': {'npw': 20, 'rhot': 5000e-10}, 'C2': {'npw': 20, 'rhot': 1500e-10},
        'C3': {'npw': 20, 'rhot': 100e-10},  'C4': {'npw': 20, 'rhot': 50e-10},

        'D1': {'npw': 200, 'rhot': 5000e-10}, 'D2': {'npw': 200, 'rhot': 1500e-10},
        'D3': {'npw': 200, 'rhot': 100e-10},  'D4': {'npw': 200, 'rhot': 50e-10},   

        #'E1': {'npw': 100, 'rhot': 5000e-10}, 'E2': {'npw': 100, 'rhot': 1500e-10},
        #'E3': {'npw': 100, 'rhot': 100e-10},  'E4': {'npw': 100, 'rhot': 50e-10},   

    }

    plot_configs = {
        'B2': {'npw': 20, 'rhot': 1500e-10},
    }

    R_p2p_joint_list = [1e-9, 10e-9, 100e-9]
    # ✅ 在最开始就创建 reporter，只创建一次
    global_reporter = ExcelReporter(Path.cwd() / cfg.HEAT_LOAD_OUTPUT_DIR / "heatload_2bars_results.xlsx")
    # -- 遍历 R_joint 列表（最外层循环）--
    for R_p2p_joint in R_p2p_joint_list:
        print(f"\n{'='*60}")
        print(f"正在处理 R_joint = {R_p2p_joint*1e9} nOhm")
        print(f"{'='*60}")
        # -- 遍历磁体配置 --
        for name, config_params in configs.items():
            Npw = config_params['npw']
            rhot = config_params['rhot']
            params = {
                'Npw': Npw, 'R_p2p_joint': R_p2p_joint, 'rhot': rhot, 
            }
            print(f"\n--- 正在处理配置: {name} ---")
            
            # static 和 operation 模式
            for mode in ['static', 'operation']:
                for T_op in [4.2, 10.0, 20.0]:
                    output_dir = Path(cfg.OUTPUTS_FIGURES_DIR) / "heatload" / f"Top={T_op}K_Rj={R_p2p_joint*1e9}nOhm" / f"config={name}"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    calculator = HeatLoadCalculator()
                    plotter = HeatLoadPlotter(output_dir)
                    
                    # -- 程序开始时，首先生成并保存一次图例 --
                    plotter.save_legend_only()
                    params['Ip'] = cfg.Ip_list[T_op]  # 单根运行温度随着温度变化
                    params['L_tot'] = cfg.L_HTS_TF_m[T_op]  # 单个TF的HTS带材长度随着温度变化
                    print(f"  Calculating for: {mode} mode at {T_op}K...")
                    # 1. 计算
                    top_loads, loads_77k = calculator.calculate_loads(mode=mode, T_op=T_op, **params)
                    # 2. 绘图，调用两种绘图方法 ++
                    # plotter.plot_dual_stacked_bar(top_loads, loads_77k, name, T_op, mode)
                    plotter.plot_dual_normalized_stacked_bar(top_loads, loads_77k, name, T_op, mode)

                    # 3. 添加记录到报告
                    global_reporter.add_record(name, mode, T_op, params, top_loads, loads_77k)

            # charging 模式
            mode = 'charging'
            # 充电情况，现在支持4.2K、10K和20K三种温度
            for T_op in [4.2, 10.0, 20.0]:
                output_dir = Path(cfg.OUTPUTS_FIGURES_DIR) / "heatload" / f"Top={T_op}K_Rj={R_p2p_joint*1e9}nOhm" / f"config={name}"
                output_dir.mkdir(parents=True, exist_ok=True)
                calculator = HeatLoadCalculator()
                plotter = HeatLoadPlotter(output_dir)
                
                # -- 程序开始时，首先生成并保存一次图例 --
                plotter.save_legend_only()
                
                print(f"  Calculating for: {mode} mode at {T_op}K...")
                params['Ip'] = cfg.Ip_list[T_op] 
                params['L_tot'] = cfg.L_HTS_TF_m[T_op]
                # 1. 计算充电结束时的快照，用于柱状图和报告
                top_loads, loads_77k = calculator.calculate_loads(mode=mode, T_op=T_op, **params)
                plotter.plot_dual_normalized_stacked_bar(top_loads, loads_77k, name, T_op, mode)
                global_reporter.add_record(name, mode, T_op, params, top_loads, loads_77k)
                # 2. +++ 计算完整的时间序列数据并绘图 +++
                print(f"  Calculating timeseries for: {mode} mode at {T_op}K...")
                timeseries_df = calculator.calculate_transient_timeseries(**params, T_op=T_op)
                plotter.plot_stacked_area_over_time(timeseries_df, name, T_op=T_op)
    # --- 循环结束后，保存Excel报告 ---
    global_reporter.save()
    print(f"\n[✔] 所有分析和绘图已完成。结果保存在文件夹 '{output_dir}' 中。")
