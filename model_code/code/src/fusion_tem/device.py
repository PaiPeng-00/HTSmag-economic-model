# ==============================================================================
# === 全局配置文件 (config.py) ===
# ==============================================================================
# 请将此文件放置在项目根目录下，所有其他脚本将从此导入参数。
from pathlib import Path
import logging
import numpy as np

from fusion_tem.economic.price_basis import (
    BACKGROUND_CAPITAL_USD_BY_SCENARIO,
    POWER_SUPPLY_PRICE_2025_USD_PER_A,
)

# ----------------------------------------------------------------------
# Repository paths
# ----------------------------------------------------------------------
# 本模块从 code/config.py 移入 code/src/fusion_tem/device.py;
# parents[2] 让 REPO_ROOT 仍指向 code/ (= src 的上级), 保持 data/ outputs/ configs/ 路径不变。
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = REPO_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = REPO_ROOT / "data" / "processed"
import os as _os_dev
# 非 arc 器件输出到 outputs/<device>/ (与 ARC 分开存放); arc 保持 outputs/
_DEV_TAG = _os_dev.environ.get("FUSION_DEVICE", "arc").lower()
OUTPUTS_DIR = REPO_ROOT / "outputs" if _DEV_TAG == "arc" else REPO_ROOT / "outputs" / _DEV_TAG
OUTPUTS_FIGURES_DIR = OUTPUTS_DIR / "figures"
OUTPUTS_TABLES_DIR = OUTPUTS_DIR / "tables"
OUTPUTS_LOGS_DIR = OUTPUTS_DIR / "logs"
CONFIGS_DIR = REPO_ROOT / "configs"

# ------------------------------------------------------------------------------
# 器件选择 (Device selection)
# 从 configs/devices/{device}.yaml 加载"器件相关原始参数"(几何/运行点/电厂/数据文件)。
# 用环境变量 FUSION_DEVICE 切换托卡马克构型, 默认 arc:
#     FUSION_DEVICE=sparc python scan_full_grid.py
# 新增器件只需加一个 yaml, 无需改代码。派生量(V_magnet 等)仍由本文件自动计算。
# ------------------------------------------------------------------------------
import os as _os
try:
    import yaml as _yaml
except ImportError:
    _yaml = None
DEVICE = _os.environ.get("FUSION_DEVICE", "arc").lower()
_dev_yaml = REPO_ROOT.parent / "configs" / "devices" / f"{DEVICE}.yaml"
if not _dev_yaml.exists():
    _dev_yaml = CONFIGS_DIR / "devices" / f"{DEVICE}.yaml"   # 备选位置
_DEV = (_yaml.safe_load(_dev_yaml.read_text(encoding="utf-8")) or {}) if (_yaml is not None and _dev_yaml.exists()) else {}
def _dev(key, default):
    """取器件参数; 对 dict(如 Ip_list) 把数值键统一转 float, 以匹配 Top=4.2/10.0/20.0。"""
    v = _DEV.get(key, default)
    return {float(k): val for k, val in v.items()} if isinstance(v, dict) else v

# ------------------------------------------------------------------------------
# 绘图样式 (E 类): 从 configs/plotting.yaml 读取, 缺失则用下方代码内缺省值。
# 这些参数只影响渲染, 不影响任何数值结果。见 0-思路/ARC-参数归属清单.md。
# ------------------------------------------------------------------------------
_PLOT_CFG_PATH = CONFIGS_DIR / "plotting.yaml"
_PLOT = {}
if _yaml is not None and _PLOT_CFG_PATH.exists():
    _PLOT = _yaml.safe_load(_PLOT_CFG_PATH.read_text(encoding="utf-8")) or {}


def _plot(key, default):
    """取绘图样式参数; yaml 缺键时回落到代码内缺省, 保证行为不变。

    yaml 没有元组类型, figsize 一类会读成 list; 按缺省值的类型还原, 保证
    迁移前后 `type(cfg.X)` 完全一致。
    """
    v = _PLOT.get(key, default)
    if isinstance(default, tuple) and isinstance(v, list):
        return tuple(v)
    return v

logging.getLogger(__name__).info(
    "device=%s (configs/devices/%s.yaml: %s)",
    DEVICE,
    DEVICE,
    "loaded" if _DEV else "missing; using built-in ARC defaults",
)

# ------------------------------------------------------------------------------
# 1. 磁体几何与物理参数 (Magnet Geometry & Physics)
# ------------------------------------------------------------------------------
# 这些参数定义了超导磁体的基本物理结构。
# 它们在电感计算和电阻计算中都会被用到。

# 饼状线圈数量 (Number of Pancakes)
NP = _dev("NP", 12)   # ARC: Nc=12 (= FEM/T-A 饼数, was SPARC 16)

# D形线圈直段长度 (m)
L1 = _dev("L1", 7.24)   # ARC FEM (was SPARC 3.0)

# D形线圈最内层小圆弧半径 (m)
R1 = _dev("R1", 0.30)   # ARC FEM (was SPARC 0.4)
R1_BASE = R1

# 线圈径向堆叠总厚度 (m)
# 这是所有匝在径向上的总宽度，用于计算匝间距。
R2 = _dev("R2", 0.64)   # ARC WP径向厚 = FEM dr_tf (统一; was SPARC 0.3)

# 单个饼状线圈的轴向厚度 (m)
WID = _dev("WID", 12e-3)   # ARC FEM 带宽 (was SPARC 4mm)

# 饼状线圈之间的轴向距离 (m)
DIST = _dev("DIST", 3e-3)   # ARC FEM 饼间距 (was SPARC 10mm)

# 单匝D型线圈的平均长度 (m)
L_TAPE_PER_TURN = (L1/2 + (R1+R2/2)*np.pi/2 + (R1+L1/2+R2/2)*np.pi/2)*2
# 系统总安匝数或等效总匝数 (Total nominal turns for the system)
# 这个值会根据并绕根数(Npw)被分配到每个物理匝上。
N_TOTAL_TAPE = _dev("N_TOTAL_TAPE", 1750) #20K 每饼带材数 = FEM Nt(20K) (was SPARC 2000); 安匝 Nt*NP*Ip=1750*12*400=8.4MA
L_tot = L_TAPE_PER_TURN * N_TOTAL_TAPE *NP # 单个TF磁体的总带材长度 

# TF磁体个数
Ntf = _dev("Ntf", 18)
S_large = (R1+R2)**2*np.pi/2 + (R1+R2+L1/2)**2*np.pi/2

S_small = (R2)**2*np.pi/2 + (R2+L1/2)**2*np.pi/2

S_magnet = S_large - S_small +R2*L1
H_magnet = (DIST+WID)*NP
V_magnet = S_magnet*H_magnet # 磁体体积


#低温恒温器的表面积
R_cryostat_bottom = (L1/2+R1+R2)+R2   # 低温恒温器底面大半径
r_cryostat_bottom = (R1+R2)  + R2     # 低温恒温器底面小半径
# 低温恒温器底面周长
C_cryostat_bottom = L1+ np.pi*(R_cryostat_bottom+r_cryostat_bottom) 

# 低温恒温器底面面积
A_cryostat_bottom = L1* (R2*2+R1) + (np.pi*R_cryostat_bottom**2 + np.pi*r_cryostat_bottom**2)/2 

H_cryostat = H_magnet*1.5 # 低温恒温器高度,1.5倍磁体高度

# 低温恒温器侧壁面积
A_cryostat_side = C_cryostat_bottom*H_cryostat 

# 低温恒温器表面积
A_cryostat = A_cryostat_bottom*2 + A_cryostat_side  


# ------------------------------------------------------------------------------
# 2. 导体与材料参数 (Conductor & Material)
# ------------------------------------------------------------------------------
# 每个饼状线圈所用带材的标称长度 (m)
# 这个值将用于计算 su-su 接头的数量
LEN_PER_SINGEL_REBCO = _dev("LEN_PER_SINGEL_REBCO", 200.0)
# 单个线圈的带材长度 (m)
L_TAPE_PER_PANCAKE = L_TAPE_PER_TURN * N_TOTAL_TAPE

# 匝内(tape-to-tape)电阻率 (Ω·m²)
# 假设是并绕带材之间的接触电阻率，通常较低。
RHO_TAPE = _dev("RHO_TAPE", 50e-10)

RRR = _dev("RRR", 100)   #铜纯度
# ------------------------------------------------------------------------------
# 3. 模拟控制参数 (Simulation Control)
# ------------------------------------------------------------------------------
# 这些参数用于控制计算的精度、范围和过程。
# -------------------------------------------------------------------------
# Temperature case definitions (for charging comparisons)
# Each case only changes: Ntape_coil (number of parallel tape roots used per coil)
#                         Ip (target current per conductor or per-design)
# Geometry (NP, L1, R1
#, etc.) is unchanged.
# Key is the float temperature in K (use same keys as Ip_list, L_HTS_TAPE_LIST).
# -------------------------------------------------------------------------
# 不同温度下Ip的目标值
Ip_list = _dev("Ip_list", {4.2: 700.0, 10.0: 583.0, 20.0: 400.0})   # ARC FEM 单带电流(载流比0.7), was SPARC {350,300,210}

Nt_list = _dev("Nt_list", {4.2: 1000 , 10.0: 1200 , 20.0: 1750  })   # ARC FEM 带材数/饼, was SPARC {1200,1400,2000}

# 一个TF磁体的总带材长度
L_HTS_TF_m = {T: Nt * L_TAPE_PER_TURN * NP for T, Nt in Nt_list.items()}
# TF系统的总带材长度,km
L_HTS_SYSTEM_km = {T: L_HTS_TF_m[T] * Ntf *1e-3 for T in L_HTS_TF_m.keys()}

Ic0_cm = _dev("Ic0_cm", 400)   # 77K 0T 临界电流 A/cm (ARC 12mm带材, HTS成本kAm参考): 400 A/cm * 1.2 cm = 480 A
Ic0 = Ic0_cm * WID *100 # 12mm带宽带材的实际临界电流 (A) = 480 A
#经济模型所需的 kAm（千安·米）
kAm_HTS_TAPE_LIST = {T: Ic0 * L for T, L in L_HTS_SYSTEM_km.items()}
TEMPERATURE_CASES = {
    4.2: {
        'label': '4.2K',
        # 每个线圈使用的带材总根数（你需要根据设计填入合适的值）
        'Ntape_coil': Nt_list.get(4.2, 1000),
        # 这里 Ip_list 已经在文件中定义，为方便也可以在 case 中重复覆盖
        'Ip': Ip_list.get(4.2, 700.0)
    },
    10.0: {
        'label': '10K',
        'Ntape_coil': Nt_list.get(10.0, 1200),
        'Ip': Ip_list.get(10.0, 583.0)
    },
    20.0: {
        'label': '20K',
        'Ntape_coil': Nt_list.get(20.0, 1750),
        'Ip': Ip_list.get(20.0, 400.0)
    }
}

# --- 电感计算参数 ---
# 每匝离散点数
POINTS_PER_TURN = _dev("POINTS_PER_TURN", 200)
# 互感计算时的分组数 (Grouping number for mutual inductance calculation)
# 根据 Npw 动态返回 ng 的值
def get_ng_for_npw(npw) -> int:
    """
    根据并绕根数 (Npw) 动态确定分组数 (ng)。
    这个函数封装了您提出的逻辑。
    """
    if npw <= 10:
        return 10
    elif npw <= 50:
        return 2
    elif npw <= 100:
        return 1
    else:
        # 为超出范围的 Npw 提供一个默认值
        return 1

# --- 充电模拟参数 ---
# 目标电流 (A), 指的是流经单根导体的电流值
I_TARGET = 500.0
# 充电持续时间 (小时)
CHARGE_HOURS = _dev("CHARGE_HOURS", 96)
# 稳态持续时间 (小时)
# 充电 ODE 的稳态段积分上限。2026-07-26 由 2000x 提到 6000x: 改用正确的 ARC 电感矩阵
# 后(见工作文档 §14.39), 充电时间整体变长约 2.4 倍, 最极端的 Npw=1/rho=10 需要约
# 372,096 h, 原 192,096 h 窗口解不出 -> calculate_time_to_999 返回 None -> 被兜底成
# cfg.charge_hours_max(120 h), 反而通过了 ">120 h" 判据。加宽窗口后该点真解出 372,096 h,
# 正确判为不可行; 所有原本已收敛的格子结果逐位不变(已实测)。
STEADY_HOURS = 6000 * CHARGE_HOURS

# ------------------------------------------------------------------------------
# 4. 参数扫描列表 (Parameter Scan Lists)
# ------------------------------------------------------------------------------
# 定义了进行参数化分析时需要遍历的列表。

# 实验一：需要扫描的并绕根数(Npw)列表
NPW_LIST_EXP1 = [ 100, 50, 20, 10, 5,2,1]
# 实验一：固定的匝间电阻率 (Ω·m²)

# 实验二：需要扫描的匝间电阻率(rho_turn)列表 (Ω·m²)
# 实验二：固定的并绕根数

# LCOE 绘图使用的固定匝间电阻率 (uOhm·cm²)

#  5. 热负荷分析参数

# 制冷剂焓值参数 (单位: kJ/kg)，不再乘以1000
ENTHALPY_HE_KJ_KG = 106.13 - 100.18
ENTHALPY_H2_KJ_KG = 24.142 - 15.036

T_HIGH = _dev("T_HIGH", 300)

P_fusion_W = _dev("P_fusion_W", 525e6) #525MW ARC (was SPARC 140MW)
eat_conv = _dev("eat_conv", 0.40) #热电转化效率 ARC (was 0.35)
# 总热功率 MWth = 聚变功率 × 包层能量倍增 (ARC: 708/525=1.349)。默认按此因子从 P_fusion 推导,
# 器件 yaml 可用 FUSION_OUTPUT_MWTH 显式覆盖。供经济模型 (gross=MWth×eat_conv) 用。
fusion_output_MWth = _dev("FUSION_OUTPUT_MWTH", round(P_fusion_W / 1e6 * (708.0 / 525.0), 1))
# 非磁体直接成本（plant-level LCOE 的 C0）。ARC 默认采用换算后的 S2 constant-2025-US$ 值；其他器件可在 YAML 覆盖。
C0_nonmagnet_USD = _dev("C0_NONMAGNET_USD", BACKGROUND_CAPITAL_USD_BY_SCENARIO["S2"])
# ------------------------------------------------------------------------------
# 6. 输入/输出路径名 (I/O Paths)
# ------------------------------------------------------------------------------
# 定义了保存结果的文件夹名称。

# 单个TF的电感计算结果的输出文件夹名
INDUCTANCE_OUTPUT_DIR = DATA_RAW_DIR / "inductance"

# TF系统的电感计算结果的输出文件夹名
TF_SYSTEM_INDUCTANCE_OUTPUT_DIR = OUTPUTS_FIGURES_DIR / "inductance"
TF_SYSTEM_MATRIX = _dev("TF_SYSTEM_MATRIX", "TF_system_L_matrix_ARC.xlsx")   # ARC 18x18 单匝互感 (CP4, R2=0.64; was SPARC TF_system_L_matrix.xlsx)
# 充电模拟结果的输出文件夹名
CHARGING_SIM_OUTPUT_DIR = OUTPUTS_TABLES_DIR / "charging"

#TF 系统的充电模拟结果输出文件名
TF_SYSTEM_CHARGING_SIM_OUTPUT_DIR = OUTPUTS_TABLES_DIR / "charging_tf"
TF_SYSTEM_CHARGING_SIM_OUTPUT_FILE_Npw1_200 = 'charging_time999_summary_TF_system_Npw=1_to_200.xlsx'
TF_SYSTEM_CHARGING_SIM_OUTPUT_PLOT_FILE_Npw1_200 = 'charging_time999_heatmap_contour_TF_system_Npw=1-200.svg'
# COMSOL ODE 文件的输出文件夹名
COMSOL_ODE_OUTPUT_DIR = OUTPUTS_TABLES_DIR / "comsol"
# ------------------------------------------------------------------------------
# 6. 热负荷分析参数 (Heat Load Analysis)
# ------------------------------------------------------------------------------
# 这些参数用于第四步的热负荷计算。

# --- 静态热源 (W) ---
N_cool_pipe = _dev("N_cool_pipe", 2)   # 制冷管道数量
d_in_cool = _dev("d_in_cool", 0.02)   # 制冷管道内径
d_out_cool = _dev("d_out_cool", 0.022)   # 制冷管道外径
L_cool = _dev("L_cool", 1.0)   # 制冷管道长度

N_aux_pipe = _dev("N_aux_pipe", 2)   # 其他辅助管道数量
d_in_aux = _dev("d_in_aux", 0.01)   # 其他辅助管道内径
d_out_aux = _dev("d_out_aux", 0.012)   # 其他辅助管道外径
L_aux = _dev("L_aux", 1.0)   # 其他辅助管道长度

#--- 低温恒温器辐射热：热绝缘包裹参数 ---
eps = _dev("eps", 0.03)   #辐射热的发射率
N_LAYER_CRYOSTAT = _dev("N_LAYER_CRYOSTAT", 30)   # 低温恒温器热绝缘包裹层数

# --- 核热与接头参数 ---

NUCLEAR_POWER_DENSITY = _dev("NUCLEAR_POWER_DENSITY", 421.0)          # 核功率密度 (W/m^3) ARC: 722W/TF ÷ V_magnet=1.72m3 (was SPARC 600)
VOLUME_M3 = V_magnet                   # 受核热影响的体积 (m^3)
JOINT_WIDTH = _dev("JOINT_WIDTH", 12.0)   # 接头宽度 (mm)
JOINT_LENGTH = _dev("JOINT_LENGTH", 200.0)   # 接头长度 (mm)
JOINT_PRESSURE = _dev("JOINT_PRESSURE", 75.0)   # 接头压力 (MPa)
R_SU_JOINT = _dev("R_SU_JOINT", 10e-9)   # Su-Su接头电阻 (Ω)

# REBCO段的引线热负荷计算参数
carrying_factor = _dev("carrying_factor", 0.7)   # 载流比
Time_transfer = _dev("Time_transfer", 5.0)   # 离网放电时间 t_f, s (Fry 2024 §III.D: minimum 5 s discharge)
Temp_transfer = _dev("Temp_transfer", 150.0)   # 稳定体绝热定尺的峰值温度上限 T_f, K
                                       # 2026-07-26 由 100 改为 150: 正文式(S19)与
                                       # Fry et al. 2024 §III.D 均为 "peak temperature
                                       # below 150 K"。见工作文档 §14.44。
T_HOT_HTS_LEAD = _dev("T_HOT_HTS_LEAD", 77.0)   #REBCO引线的热端温度,K
L_LEAD_HTS = _dev("L_LEAD_HTS", 0.5)   #REBCO引线的长度,m
# --- 铜引线参数 ---
design_factor = _dev("design_factor", 5e6)   # 引线设计因子,I*L/A ≈ 5e6 A/m (经验设计值)
INTK_CU30 = _dev("INTK_CU30", 42853.56)   # 30K下铜的积分导热率 (W/m)
LEN_LEAD_CU = _dev("LEN_LEAD_CU", 1.0)   # 引线的铜部分的长度 (m)
N_LEAD = _dev("N_LEAD", 2)   # 引线数量

#--- 其他静态热源 ---
Q_OTHER_SOURCES = _dev("Q_OTHER_SOURCES", 10.0)   # 其他静态热源


ceff_77to20 = _dev("ceff_77to20", 5)   # 77K热量转换为->20K的等效热量

# --- 模拟时间控制 (h) ---

TOTAL_SIMULATION_TIME_H = _dev("TOTAL_SIMULATION_TIME_H", 192)   # 总模拟时长
TIME_STEP_H = _dev("TIME_STEP_H", 1)   # 时间步长

# --- 冷却剂参数 ---
P0_HE_BAR = _dev("P0_HE_BAR", 10.0)   # 氦气初始压力,bar
P0_H2_BAR = _dev("P0_H2_BAR", 10.0)   # 氢气初始压力,bar
P_REF = _dev("P_REF", 10.0)   # 参考压力,bar

#运行最大温度
MAX_TEMP_RUN = _dev("MAX_TEMP_RUN", 21)

# --- 为时间序列图选择的特定分析对象 ---
NPW_FOR_TIMESERIES_PLOT = [10, 100]       # 分析Npw时，为哪些值绘制时间序列图
RHOT_FOR_TIMESERIES_PLOT = [1500, 5000]    # 分析rhot时，为哪些值绘制时间序列图

# --- 输入/输出路径 ---
HEAT_DATA_DIR = DATA_RAW_DIR / "heat_input"               # 热负荷输入数据根文件夹
#为不同实验定义输入数据的子文件夹名称
HEAT_DATA_SUBDIR_NPW = 'variable_Npw'
HEAT_DATA_SUBDIR_RHOT = 'variable_rhot'

HEAT_LOAD_OUTPUT_DIR = OUTPUTS_TABLES_DIR / "heatload"   # 热负荷分析结果输出文件夹

LEGEND_IMAGE_FILE = "heat_load_legend.svg" # 独立图例文件名
MAG_LOSS_FILE = _dev("MAG_LOSS_FILE", "data_s3_magnetization_loss.xlsx")       # 磁化损耗数据文件名
RADIAL_LOSS_FILE = _dev("RADIAL_LOSS_FILE", "data_s2_radial_loss.xlsx") # 径向损耗数据文件名
# 处理后的数据 输入/输出路径 (在末尾新增)
PREPROCESSED_DATA_DIR = DATA_PROCESSED_DIR / "heat_input"
PREPROCESSED_FILENAME = 'preprocessed_loss_data.xlsx'
# ------------------------------------------------------------------------------
# 7. 全局绘图输出格式 (Global Plotting Output Format)
#  根据全局设定，定义所有图表的输出格式
# ------------------------------------------------------------------------------
PLOT_FORMAT = _plot("PLOT_FORMAT", 'svg')
# 统一图像分辨率（用于所有 plt.savefig）
# 保持原来的高分辨率，不改变图像外观；我们只通过修改文件扩展名来控制输出格式（SVG / PDF）
PLOT_DPI = _plot("PLOT_DPI", 1800)
PLOT_TRANSPARENT = _plot("PLOT_TRANSPARENT", True)

# 可选：Inkscape 路径（用于 SVG -> PDF 导出）
# 若不设置，将尝试从系统 PATH 查找 inkscape
INKSCAPE_EXE = _os_dev.environ.get("INKSCAPE_EXE", "inkscape")

# 8. 降温分析参数 (Cooldown Analysis Parameters)
COOLDOWN_DATA_DIR = DATA_RAW_DIR / "cooldown"
COOLDOWN_OUTPUT_DIR = OUTPUTS_TABLES_DIR / "cooldown"

# 9. 充电性能分析参数 (Charging Performance Analysis)
CHARGE_DATA_DIR = DATA_RAW_DIR / "charge"
CHARGE_OUTPUT_DIR = OUTPUTS_TABLES_DIR / "charge"

# 10. 运行性能分析参数 (Operation Performance Analysis)
OPERATION_DATA_DIR = DATA_RAW_DIR / "operation"
OPERATION_OUTPUT_DIR = OUTPUTS_TABLES_DIR / "operation"

#11. 失超工况分析参数 (quench Analysis)
QUENCH_DATA_DIR = DATA_RAW_DIR / "quench"

QUENCH_OUTPUT_DIR = OUTPUTS_TABLES_DIR / "quench"

# [修改] 为两个不同的实验指定各自的数据文件名

# [新增] 失超事件的时间定义 (单位: 秒)

# [新增] 稳态温度的判定标准


#12. 规模化分析参数 (Scaling Analysis)
SCALING_DATA_DIR = DATA_RAW_DIR / "scaling_input"
SCALING_OUTPUT_DIR = OUTPUTS_TABLES_DIR / "scaling"

#14. Techno-economic analysis parameters
ECONOMIC_DATA_DIR = DATA_RAW_DIR / "economic_input"
ECONOMIC_OUTPUT_DIR = OUTPUTS_TABLES_DIR / "economic"
ECONOMIC_FIGURES_DIR = OUTPUTS_FIGURES_DIR / "economic"

PARASITIC_RATIO_OUTPUT_DIR = 'parasitic_ratio'
# 热力图子目录：分别存放 USD 图、CNY 图和中间数据（Excel）
PARASITIC_RATIO_USD_DIR = 'USD'
PARASITIC_RATIO_CNY_DIR = 'CNY'
PARASITIC_RATIO_DATA_DIR = 'data'
# 可供选择的绘制热力图的色彩方案
color_schemes = _plot("color_schemes", ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'Blues', 'YlGnBu', 'PuBuGn', 'mako', 'rocket'])
# 进行绘图的热力图的色彩方案（active）
# 说明：
# - "YlGnBu_trunc_0p8" 表示截取 YlGnBu 的 [0.0, 0.8] 区间（压缩最深色端），但在同一配色方案内保持一致。
# 快速调试模式 FAST_FIG=1: 只画 YlGnBu_trunc_0p8 一套配色 (出图快 ~4 倍)。定稿全量出图时不设此变量。
if _os.environ.get("FAST_FIG", "").lower() in ("1", "true", "yes"):
    color_schemes = ["YlGnBu_trunc_0p8"]
else:
    color_schemes = ["cividis", "Blues", "YlGnBu", "YlGnBu_trunc_0p8"]


def resolve_cmap(cmap):
    """
    将 config 中的“自定义配色方案名”解析为 matplotlib Colormap 对象。
    - 传入 str 时：支持内置/Seaborn colormap 名，以及本项目自定义的截取版 colormap 名。
    - 传入 Colormap 时：原样返回。
    """
    # 已经是 colormap 对象则直接返回
    try:
        import matplotlib as _mpl  # 局部导入，避免 config 顶层强依赖
        if isinstance(cmap, _mpl.colors.Colormap):
            return cmap
    except Exception:
        # matplotlib 不可用时，退化为原样返回
        return cmap

    if not isinstance(cmap, str):
        return cmap

    # --- 自定义截取版配色：YlGnBu 的 [0.0, 0.8] ---
    if cmap == "YlGnBu_trunc_0p8":
        try:
            import numpy as _np
            import matplotlib.pyplot as _plt
            from matplotlib.colors import ListedColormap as _ListedColormap

            _cmap_full = _plt.get_cmap("YlGnBu")
            _colors = _cmap_full(_np.linspace(0.0, 0.8, 256))
            return _ListedColormap(_colors, name="YlGnBu_trunc_0p8")
        except Exception:
            # 若失败则退回原始 YlGnBu
            return "YlGnBu"

    # 默认：交给调用方自行处理（plt.get_cmap / seaborn fallback）
    return cmap

# 进行绘图的热力图的色彩方案
cmap_parasitic_ratio = _plot("cmap_parasitic_ratio", 'cividis')
# LCOE等高线计算方法：'adaptive'（自适应，基于数据分布频率）或 'fixed'（固定值）
LCOE_CONTOUR_METHOD = _plot("LCOE_CONTOUR_METHOD", 'fixed') # 可选：'adaptive' 或 'fixed'
# LCOE等高线固定值数组（当LCOE_CONTOUR_METHOD='fixed'时使用）
# 用户指定的固定值：[-3, -1, -0.5, -0.1, 0, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 40]
# 注意：已扩展范围以覆盖更大数据范围，并确保最小间隔>=0.1
# ΔLCOE_min 恒 ≥0（相对情景全局最小值），不需要负值。用好看的全局固定级别，
# 由调用处显式传入(绕过 auto_contour_levels 自适应)，避免窄范围面板挤出 25,26,27 这种聚集级别。
# 低端 2 间隔(2,4,6,8,10) 给窄范围面板足够覆盖，高端 5 间隔(15,20,25,30,40) 保持稀疏整洁。
LCOE_CONTOUR_FIXED_LEVELS = np.array([
    2, 4, 6, 8, 10, 15, 20, 25, 30, 40
])
# CNY/kWh单位的LCOE等高线固定值数组（用于CNY版本的热力图）
LCOE_CONTOUR_FIXED_LEVELS_CNY = np.array([
    0, 0.001, 0.002, 0.003, 0.004, 0.004, 0.005,
    0.01, 0.016, 0.02, 0.03, 0.04, 0.05,
    0.09, 0.1, 0.12, 0.13, 0.14, 0.16,
    0.2, 0.3, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7,
    1, 1.2, 1.4, 1.6
])

# ΔLCOE 相关的全局配置参数
# >50% 的点视为不可行，用于mask并在图中画白色斜线格子
DELTA_LCOE_MASK_PARASITIC_PCT = _plot("DELTA_LCOE_MASK_PARASITIC_PCT", 50.0)
# 取 |ΔLCOE| 的 99.5 分位来限定色标范围，避免极端值压缩色标
DELTA_LCOE_ABS_PCTL = _plot("DELTA_LCOE_ABS_PCTL", 99.5)
# 汇率：美元对人民币（用于 LCOE 单位转换）
USD_TO_CNY_EXCHANGE_RATE = 7.2  # 1美元 = 7.2人民币

# 成本分析结果输出文件夹
COST_OUTPUT_DIR = 'cost_output'

ECONOMIC_SUMMARY_FILE = f'cost&heat_summary_all.xlsx'
ECONOMIC_SUMMARY_FIG = 'cost_summary_all.svg'
HOURS_PER_YEAR = 8760 # 每年小时数
# 年度运行计划参数

pulse_hours = 2.0             # 每个生产脉冲时长 (h)
dwell_hours = 10.0 / 60.0     # 每个间隔/充电时长 (h) ; 默认 10 min
maintenance_hours_per_year = 24 * 30 * 2  # 默认: 每年 2 个月 = 1440 h
nuclear_decay_factor = 0.1   # 间隔期间的核热衰减因子

# 进行成本分析的时候，确定的参数
Npw_TARGET = 50              # 目标并绕根数
R_p2p_joint_TARGET = 10e-9   # 目标匝间电阻率
charge_hours_max = 24*5      # 目标充电时长，5天

#电源设备的成本
# 电源设备成本：500 A 参考电源价格为 10,000 US$，折合 20 US$/A。
POWER_SUPPLY_REFERENCE_CURRENT_A = 500.0
POWER_SUPPLY_REFERENCE_COST_USD = (
    POWER_SUPPLY_PRICE_2025_USD_PER_A * POWER_SUPPLY_REFERENCE_CURRENT_A
)
power_supply_price_perA = POWER_SUPPLY_REFERENCE_COST_USD / POWER_SUPPLY_REFERENCE_CURRENT_A

#每年生产小时数
production_hours_per_year = (HOURS_PER_YEAR-maintenance_hours_per_year)*pulse_hours/(pulse_hours+dwell_hours )

# --- 运行场景定义（统一提供给经济模型与可用因子绘图脚本） ---
# C0_nonmagnet_USD (非磁体 BoP 直接资本, 全厂 LCOE 的 C0): 随技术成熟度递减, 与寿命/HTS价一致。
#   文献锚点: Wade-based 2010-US$ scenario anchors are normalized once to constant 2025 US$.
#   映射: S1/S2/S3 = 6.200979/4.133986/2.066993 billion 2025 US$.
#   S4/S5/S6 是 S2 的 HTS 价变体(Fig6)，统一沿用 S2 的运行时间表和
#   2025-US$ 背景资本，只改变直接定义的 HTS 情景价格。
#   HTS 价格为 100/50/10 constant-2025-US$/(kA·m)，不作历史价格 CPI 换算。
SCENARIO_DEFINITIONS = {
    "S1": {
        "label": "S1",
        "fwb_damage_limit_dpa": 50.0,
        "fwb_lifetime_fpy": 1.0,
        "major_scheduled_days_per_fpy": 16.93,
        "minor_scheduled_days_per_fpy": 6.05,
        "unplanned_unavailability": 0.07,
        "plant_availability_requirement": 0.80,
        "tf_cycles_per_year": 1.0,
        "tcool_h": 12 * 24.0,
        "twarm_h": 6 * 24.0,
        "kdis": 1.0,
        "tau_pulse_h": 1,
        "tau_dwell_h": 1,
        "C0_nonmagnet_USD": BACKGROUND_CAPITAL_USD_BY_SCENARIO["S1"],   # present/FOAK: 上界
    },
    "S2": {
        "label": "S2",
        "fwb_damage_limit_dpa": 100.0,
        "fwb_lifetime_fpy": 2.0,
        "major_scheduled_days_per_fpy": 8.47,
        "minor_scheduled_days_per_fpy": 6.05,
        "unplanned_unavailability": 0.07,
        "plant_availability_requirement": 0.80,
        "tf_cycles_per_year": 1.0,
        "tcool_h": 12 * 24.0,
        "twarm_h": 6 * 24.0,
        "kdis": 1.0,
        "tau_pulse_h": 1,
        "tau_dwell_h": 0.25,
        "C0_nonmagnet_USD": BACKGROUND_CAPITAL_USD_BY_SCENARIO["S2"],   # neutral: 中心(PROCESS B27)
    },
    "S3": {
        "label": "S3",
        "fwb_damage_limit_dpa": 200.0,
        "fwb_lifetime_fpy": 4.0,
        "major_scheduled_days_per_fpy": 4.23,
        "minor_scheduled_days_per_fpy": 6.05,
        "unplanned_unavailability": 0.07,
        "plant_availability_requirement": 0.80,
        "tf_cycles_per_year": 1.0,
        "tcool_h": 12 * 24.0,
        "twarm_h": 6 * 24.0,
        "kdis": 1.0,
        "tau_pulse_h": 1.0,
        "tau_dwell_h": 0.063,
        "C0_nonmagnet_USD": BACKGROUND_CAPITAL_USD_BY_SCENARIO["S3"],   # advanced/learned: 下界
    },
    "S4": {
        "label": "S4",
        "fwb_damage_limit_dpa": 100.0,
        "fwb_lifetime_fpy": 2.0,
        "major_scheduled_days_per_fpy": 8.47,
        "minor_scheduled_days_per_fpy": 6.05,
        "unplanned_unavailability": 0.07,
        "plant_availability_requirement": 0.80,
        "tf_cycles_per_year": 1.0,
        "tcool_h": 12 * 24.0, # 同S2
        "twarm_h": 6 * 24.0,  # 同S2
        "kdis": 1.0,          # 同S2
        "tau_pulse_h": 1,     # 同S2
        "tau_dwell_h": 0.25,  # 同S2
        "C0_nonmagnet_USD": BACKGROUND_CAPITAL_USD_BY_SCENARIO["S2"],   # Fig6 HTS 价单变量
    },
    "S5": {
        "label": "S5",
        "fwb_damage_limit_dpa": 100.0,
        "fwb_lifetime_fpy": 2.0,
        "major_scheduled_days_per_fpy": 8.47,
        "minor_scheduled_days_per_fpy": 6.05,
        "unplanned_unavailability": 0.07,
        "plant_availability_requirement": 0.80,
        "tf_cycles_per_year": 1.0,
        "tcool_h": 12 * 24.0, # 同S2
        "twarm_h": 6 * 24.0,  # 同S2
        "kdis": 1.0,          # 同S2
        "tau_pulse_h": 1,     # 同S2
        "tau_dwell_h": 0.25,  # 同S2
        "C0_nonmagnet_USD": BACKGROUND_CAPITAL_USD_BY_SCENARIO["S2"],   # Fig6 HTS 价单变量
    },
    "S6": {
        "label": "S6",
        "fwb_damage_limit_dpa": 100.0,
        "fwb_lifetime_fpy": 2.0,
        "major_scheduled_days_per_fpy": 8.47,
        "minor_scheduled_days_per_fpy": 6.05,
        "unplanned_unavailability": 0.07,
        "plant_availability_requirement": 0.80,
        "tf_cycles_per_year": 1.0,
        "tcool_h": 12 * 24.0, # 同S2
        "twarm_h": 6 * 24.0,  # 同S2
        "kdis": 1.0,          # 同S2
        "tau_pulse_h": 1,     # 同S2
        "tau_dwell_h": 0.25,  # 同S2
        "C0_nonmagnet_USD": BACKGROUND_CAPITAL_USD_BY_SCENARIO["S2"],   # Fig6 HTS 价单变量
    },
}

N_facility_list = range(1,51) # 在一个地点规模化建设的装置数量，共享冷厂，减小制冷耗电，1-50台
N_cooldown_per_year = 0 # 每年降温次数，假设不降温，在维护期间也是保持低温
E_cooldown_Wh = 0 # 每次降温能量，每次降温能量

# =============================================================================
# 15. 经济分析参数扫描配置 (Economic Analysis Parameter Scan Configuration)
# =============================================================================
# R_joint绘图值：对数均匀分布的数组（用于绘图，匹配对数刻度）
# 范围：1nOhm 到 100nOhm
R_JOINT_SCAN_VALUES = np.array([
    1.000, 1.166, 1.359, 1.585, 1.848, 2.154,
    2.512, 2.929, 3.415, 3.981, 4.642, 5.412,
    6.310, 7.356, 8.577, 10.000, 11.659, 13.594,
    15.849, 18.478, 21.544, 25.119, 29.286, 34.145,
    39.811, 46.416, 54.117, 63.096, 73.564, 85.770,
    100.000
])  # nOhm
# Npw扫描值：[1, 10, 20, 30, ..., 200]
# NPW_SCAN_VALUES = np.concatenate([[1], np.arange(10, 201, 10)])
NPW_SCAN_VALUES = np.concatenate([np.arange(1, 21, 1), np.arange(10, 201, 10)])

# 参数扫描的温度-制冷剂对
SCAN_TEMP_COOLANT_PAIRS = [(4.2, "He"), (10.0, "He"), (20.0, "He"), (20.0, "H2")]

# 经济分析的运行工况
ECONOMIC_OPERATING_CONDITIONS = [(4.2, "He"), (10.0, "He"), (20.0, "He"), (20.0, "H2")]

# 全局基准LCOE参考点（用于全局相对LCOE图）
GLOBAL_REFERENCE_SCENARIO = 'S3'
GLOBAL_REFERENCE_TEMP = 20.0
GLOBAL_REFERENCE_COOLANT = 'H2'

# SymLogNorm参数（用于ΔLCOE热力图）
DELTA_LCOE_LINTHRESH_USD = _plot("DELTA_LCOE_LINTHRESH_USD", 0.1) # USD/MWh单位的线性阈值
DELTA_LCOE_LINSCALE_USD = _plot("DELTA_LCOE_LINSCALE_USD", 0.5) # USD/MWh单位的线性缩放因子
DELTA_LCOE_LINTHRESH_CNY = _plot("DELTA_LCOE_LINTHRESH_CNY", 0.001) # CNY/kWh单位的线性阈值
DELTA_LCOE_LINSCALE_CNY = _plot("DELTA_LCOE_LINSCALE_CNY", 0.5) # CNY/kWh单位的线性缩放因子

# 等高线级别选择参数
CONTOUR_TARGET_COUNT = _plot("CONTOUR_TARGET_COUNT", 5) # 每个子图的目标等高线数量
CONTOUR_MIN_SPACING_USD = _plot("CONTOUR_MIN_SPACING_USD", 0.1) # USD/MWh单位的最小等高线间隔
CONTOUR_MIN_SPACING_CNY = _plot("CONTOUR_MIN_SPACING_CNY", 0.001) # CNY/kWh单位的最小等高线间隔

# 等高线和标签描边参数
USE_STROKE = _plot("USE_STROKE", True) # 设置为 True 以启用描边技术（黑色等高线+白色描边）
CONTOUR_STROKE_LINEWIDTH = _plot("CONTOUR_STROKE_LINEWIDTH", 4) # 普通等高线描边线宽
CONTOUR_STROKE_LINEWIDTH_ZERO = _plot("CONTOUR_STROKE_LINEWIDTH_ZERO", 5) # 0等高线描边线宽（更粗）
CONTOUR_STROKE_LINEWIDTH_AF = _plot("CONTOUR_STROKE_LINEWIDTH_AF", 2.0) # AF热力图等高线描边线宽
CONTOUR_STROKE_FOREGROUND = _plot("CONTOUR_STROKE_FOREGROUND", 'white') # 描边颜色
CONTOUR_STROKE_ALPHA = _plot("CONTOUR_STROKE_ALPHA", 0.9) # 描边透明度
LABEL_STROKE_LINEWIDTH = _plot("LABEL_STROKE_LINEWIDTH", 4) # 文字标签描边线宽
LABEL_STROKE_FOREGROUND = _plot("LABEL_STROKE_FOREGROUND", 'white') # 文字标签描边颜色
LABEL_STROKE_ALPHA = _plot("LABEL_STROKE_ALPHA", 0.8) # 文字标签描边透明度

# 热力图透明度参数
HEATMAP_ALPHA = _plot("HEATMAP_ALPHA", 0.8) # 热力图透明度（0.0-1.0）

# 热力图字体大小参数
HEATMAP_FONT_SIZE = _plot("HEATMAP_FONT_SIZE", 18) # 热力图主要字体大小（等高线标签、坐标轴标签等）
HEATMAP_FONT_SIZE_TICK = _plot("HEATMAP_FONT_SIZE_TICK", 20) # 热力图刻度字体大小
HEATMAP_FONT_SIZE_TITLE = _plot("HEATMAP_FONT_SIZE_TITLE", 24) # 热力图标题字体大小
HEATMAP_FONT_SIZE_COLORBAR = _plot("HEATMAP_FONT_SIZE_COLORBAR", 18) # 热力图色条标签字体大小
HEATMAP_FONT_SIZE_COLORBAR_TITLE = _plot("HEATMAP_FONT_SIZE_COLORBAR_TITLE", 24) # 热力图色条标题字体大小

# 热力图图形大小参数
HEATMAP_FIGSIZE_SINGLE = _plot("HEATMAP_FIGSIZE_SINGLE", (5, 5)) # 单个热力图图形大小（宽，高）
HEATMAP_FIGSIZE_GRID = _plot("HEATMAP_FIGSIZE_GRID", (15, 5)) # 网格热力图图形大小（宽，高）
HEATMAP_FIGSIZE_COLORBAR = _plot("HEATMAP_FIGSIZE_COLORBAR", (0.5, 6)) # 色条图形大小（宽，高）
HEATMAP_FIGSIZE_COLORBAR_LARGE = _plot("HEATMAP_FIGSIZE_COLORBAR_LARGE", (3.0, 6.0)) # 大型色条图形大小（宽，高）

# 热力图等高线参数
HEATMAP_CONTOUR_LINEWIDTH = _plot("HEATMAP_CONTOUR_LINEWIDTH", 2) # 普通等高线线条宽度
HEATMAP_CONTOUR_LINEWIDTH_ZERO = _plot("HEATMAP_CONTOUR_LINEWIDTH_ZERO", 3) # 0等高线线条宽度（更粗）
HEATMAP_CONTOUR_LINEWIDTH_AF = _plot("HEATMAP_CONTOUR_LINEWIDTH_AF", 2) # AF热力图等高线线条宽度
HEATMAP_CONTOUR_LINEWIDTH_CRYO = _plot("HEATMAP_CONTOUR_LINEWIDTH_CRYO", 2) # 低温热力图等高线线条宽度

# 热力图等高线标签格式
HEATMAP_CONTOUR_FMT_PARASITIC = _plot("HEATMAP_CONTOUR_FMT_PARASITIC", '%.1f') # 寄生功率等高线标签格式
HEATMAP_CONTOUR_FMT_LCOE = _plot("HEATMAP_CONTOUR_FMT_LCOE", '%.2f') # LCOE等高线标签格式（自适应）
HEATMAP_CONTOUR_FMT_AF = _plot("HEATMAP_CONTOUR_FMT_AF", '%.2f') # AF热力图等高线标签格式
HEATMAP_CONTOUR_FMT_CRYO = _plot("HEATMAP_CONTOUR_FMT_CRYO", '%.2f') # 低温热力图等高线标签格式






