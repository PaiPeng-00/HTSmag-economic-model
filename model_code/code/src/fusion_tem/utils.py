# mutual_inductance_equ_Dshape_v3_refined.py
import pandas as pd
import numpy as np
from numba import cuda
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
from fusion_tem import device as cfg
# --- 全局字体大小配置 ---
# 将此代码块放在脚本的开头
plt.rcParams['font.family'] = 'Arial' 
# 全局设置坐标轴标签的字号
plt.rcParams['axes.labelsize'] = 20  # 例如，从12号增大到14号
# 全局设置图表总标题的字号
plt.rcParams['axes.titlesize'] = 20  # 例如，从16号增大到18号
# 全局设置x轴刻度数字的字号
plt.rcParams['xtick.labelsize'] = 20
# 全局设置y轴刻度数字的字号
plt.rcParams['ytick.labelsize'] = 20
# 全局设置图例的字号
plt.rcParams['legend.fontsize'] = 20

mapsize = 12
# ========================
# 1. 几何构建模块
# ========================

def generate_equivalent_group_D_turn(
    L1: float, R1_base: float, ng, dR: float, center: Tuple[float, float, float],
    points_straight = 100, points_arc = 50
) -> np.ndarray:
    """
    生成一组 D 形线圈匝的等效路径点。
    该函数将 ng 匝线圈等效为具有平均几何尺寸的单匝线圈。

    [REFINEMENT] 增加了输入参数校验。
    """
    # [REFINEMENT] 输入参数校验
    assert L1 > 0, "D形线圈直段长度 L1 必须为正数"
    assert R1_base > 0, "基础半径 R1_base 必须为正数"
    assert ng > 0, "匝数 ng 必须为正整数"
    assert points_straight > 1 and points_arc > 1, "离散点数必须大于1"

    x0, y0, z0 = center

    # 计算 ng 匝的平均半径
    R1_start = R1_base
    R1_end = R1_base + (ng - 1) * dR
    R1_avg = (R1_start + R1_end) / 2
    R_arc_avg = L1 / 2 + R1_avg

    # (1) 上方直段
    straight_x = np.linspace(-L1 / 2, L1 / 2, points_straight)
    straight_y = np.full_like(straight_x, R1_avg)

    # (2) 右侧小圆弧
    theta_right = np.linspace(np.pi / 2, 0, points_arc)
    arc_right_x = L1 / 2 + R1_avg * np.cos(theta_right)
    arc_right_y = R1_avg * np.sin(theta_right)

    # (3) 下方大半圆
    theta_bottom = np.linspace(0, -np.pi, points_straight)
    arc_bottom_x = R_arc_avg * np.cos(theta_bottom)
    arc_bottom_y = R_arc_avg * np.sin(theta_bottom)

    # (4) 左侧小圆弧
    theta_left = np.linspace(np.pi, np.pi / 2, points_arc)
    arc_left_x = -L1 / 2 + R1_avg * np.cos(theta_left)
    arc_left_y = R1_avg * np.sin(theta_left)

    x = np.concatenate([straight_x, arc_right_x, arc_bottom_x, arc_left_x])
    y = np.concatenate([straight_y, arc_right_y, arc_bottom_y, arc_left_y])
    z = np.full_like(x, z0)

    return np.vstack((x + x0, y + y0, z)).T

def build_multicoil_Dshape_equiv(
    Np, Nt, Nm, ng, L1: float, R1_base: float, dR: float,
    thickness: float, dist: float, points_per_turn
) -> Tuple[List[np.ndarray], List[np.ndarray], List[Tuple[int, int]]]:
    """
    构建完整的多线圈系统几何模型。
    [REFINEMENT] 增加了类型提示和输入校验。
    """
    # [REFINEMENT] 输入参数校验
    assert all(arg > 0 for arg in [Np, Nt, Nm, ng, L1, R1_base, thickness, dist, points_per_turn]), "所有数值参数必须为正"
    assert Nt % Nm == 0, "总匝数 Nt 必须能被模块数 Nm 整除"

    all_pts_list, all_weights, all_info = [], [], []
    points_straight = int(points_per_turn / (2 + (np.pi + 1) / (1 + L1 / (2*R1_base))))
    points_arc = int((points_per_turn - 2 * points_straight) / 2)

    for coil_idx in range(Np):
        z_center = coil_idx * (thickness + dist)
        z_positions = np.linspace(-thickness / 2, thickness / 2, Nm) + z_center

        for m_idx in range(Nm):
            start_turn_module = m_idx * (Nt // Nm)
            end_turn_module = (m_idx + 1) * (Nt // Nm)
            
            for g_start_turn in range(start_turn_module, end_turn_module, ng):
                g_turns = min(ng, end_turn_module - g_start_turn)
                if g_turns == 0: continue

                g_R1_base = R1_base + g_start_turn * dR
                pts = generate_equivalent_group_D_turn(
                    L1=L1, R1_base=g_R1_base, ng=g_turns, dR=dR,
                    center=(0.0, 0.0, z_positions[m_idx]),
                    points_straight=points_straight, points_arc=points_arc
                )
                all_pts_list.append(pts)
                all_weights.append(np.array([g_turns], dtype=np.float32))
                all_info.append((coil_idx, m_idx))

    return all_pts_list, all_weights, all_info

def expand_weights(weight_group: np.ndarray, target_len) -> np.ndarray:
    """将组权重扩展到与该组的线元数量匹配的数组。"""
    if not isinstance(weight_group, np.ndarray) or weight_group.size == 0:
        raise ValueError("weight_group 必须是一个非空的 NumPy 数组。")
    return np.full(target_len, weight_group[0], dtype=np.float32)

# ========================
# 2. GPU 计算模块
# ========================

@cuda.jit
def mutual_inductance_kernel(dl_i, mid_i, w_i, dl_j, mid_j, w_j, result_array):
    """
    CUDA 核函数，用于并行计算两组线元之间的互感贡献。
    此函数在 GPU 的每个线程上执行，计算 dl_i[idx_i] 与所有 dl_j 之间的互感贡献。
    """
    idx_i = cuda.grid(1)
    if idx_i >= mid_i.shape[0]:
        return

    r1x, r1y, r1z = mid_i[idx_i]
    dl1x, dl1y, dl1z = dl_i[idx_i]
    wi = w_i[idx_i]

    # 累加来自线圈 j 所有线元的贡献
    total_contribution = 0.0
    for idx_j in range(mid_j.shape[0]):
        r2x, r2y, r2z = mid_j[idx_j]
        dl2x, dl2y, dl2z = dl_j[idx_j]
        wj = w_j[idx_j]

        dx, dy, dz = r2x - r1x, r2y - r1y, r2z - r1z
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)

        # 避免奇异点（线元重合或过近），保证数值稳定性
        if dist > 1e-9:
            dot_product = dl1x * dl2x + dl1y * dl2y + dl1z * dl2z
            total_contribution += (dot_product / dist) * wi * wj

    # 使用原子加法，确保多线程安全地更新最终结果
    cuda.atomic.add(result_array, 0, total_contribution)


def compute_mutual_inductance_gpu(
    dl_i: np.ndarray, mid_i: np.ndarray, w_i: np.ndarray,
    dl_j: np.ndarray, mid_j: np.ndarray, w_j: np.ndarray
) -> float:
    """
    在 GPU 上计算两个线圈组之间的互感。
    返回: 计算出的互感值 (单位: H)。
    """
    threads_per_block = 128
    blocks_per_grid = (mid_i.shape[0] + threads_per_block - 1) // threads_per_block

    result_dev = cuda.to_device(np.array([0.0], dtype=np.float32))
    
    dl_i_dev, mid_i_dev, w_i_dev = cuda.to_device(dl_i), cuda.to_device(mid_i), cuda.to_device(w_i)
    dl_j_dev, mid_j_dev, w_j_dev = cuda.to_device(dl_j), cuda.to_device(mid_j), cuda.to_device(w_j)
    
    mutual_inductance_kernel[blocks_per_grid, threads_per_block](
        dl_i_dev, mid_i_dev, w_i_dev,
        dl_j_dev, mid_j_dev, w_j_dev,
        result_dev
    )
    cuda.synchronize()

    total_M_contribution = result_dev.copy_to_host()[0]
    
    mu0 = 4e-7 * np.pi
    return (mu0 / (4 * np.pi)) * total_M_contribution

def compute_group_matrix_gpu(
    pts_list: List[np.ndarray], weights_list: List[np.ndarray]
) -> np.ndarray:
    """计算"线圈组-线圈组"的完整互感矩阵。"""
    N = len(pts_list)
    if N == 0:
        return np.array([])
    M = np.zeros((N, N), dtype=np.float64)

    print("正在预处理几何数据...")
    preprocessed_data = []
    for p_i, w_group_i in zip(pts_list, weights_list):
        dl_i = np.diff(p_i, axis=0, append=p_i[0:1, :])
        mid_i = 0.5 * (p_i + np.roll(p_i, -1, axis=0))
        w_i = expand_weights(w_group_i, mid_i.shape[0])
        preprocessed_data.append((
            dl_i.astype(np.float32), 
            mid_i.astype(np.float32), 
            w_i.astype(np.float32)
        ))

    print(f"开始计算 {N}x{N} 组-组互感矩阵...")
    for i in range(N):
        for j in range(i, N): # 利用对称性 M_ij = M_ji
            dl_i, mid_i, w_i = preprocessed_data[i]
            dl_j, mid_j, w_j = preprocessed_data[j]
            
            val = compute_mutual_inductance_gpu(dl_i, mid_i, w_i, dl_j, mid_j, w_j)
            
            M[i, j] = val
            if i != j:
                M[j, i] = val

    print("组-组矩阵计算完成。")
    return M

def aggregate_matrix_to_pancakes(
    group_matrix: np.ndarray, info: List[Tuple[int, int]], Np
) -> np.ndarray:
    """将组-组矩阵聚合成饼状线圈-饼状线圈矩阵。"""
    if group_matrix.size == 0:
        return np.array([])
        
    pancake_matrix = np.zeros((Np, Np), dtype=np.float64)
    info_array = np.array(info)
    
    print("正在将组矩阵聚合成饼状线圈矩阵...")
    for i in range(Np):
        for j in range(i, Np):
            groups_in_pancake_i = np.where(info_array[:, 0] == i)[0]
            groups_in_pancake_j = np.where(info_array[:, 0] == j)[0]
            
            sub_matrix = group_matrix[np.ix_(groups_in_pancake_i, groups_in_pancake_j)]
            total_inductance = np.sum(sub_matrix)
            
            pancake_matrix[i, j] = total_inductance
            if i != j:
                pancake_matrix[j, i] = total_inductance

    print("矩阵聚合完成。")
    return pancake_matrix

# ========================
# 3. 可视化模块
# ========================

def plot_multicoil_paths(pts_list: List[np.ndarray], title: str = '多线圈等效路径'):
    """以 3D 形式绘制所有线圈的几何路径。"""
    if not pts_list:
        print("警告：路径列表为空，无法绘图。")
        return
        
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    colors = plt.cm.viridis(np.linspace(0, 1, len(pts_list)))
    
    for idx, pts in enumerate(pts_list):
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], label=f'线圈组 {idx}', linewidth=1.5, color=colors[idx])

    ax.set_xlabel('X (m)'), ax.set_ylabel('Y (m)'), ax.set_zlabel('Z (m)')
    ax.set_title(title, fontsize=16), ax.legend(), ax.grid(True)
    ax.autoscale_view()
    plt.tight_layout(), plt.show()

def plot_inductance_heatmap(
    matrix: np.ndarray, title: str = 'Inductance Matrix', save_path: Optional[str] = None
):
    """将电感矩阵可视化为热力图。"""
    if matrix.size == 0:
        print("警告：输入的电感矩阵为空，无法生成热力图。")
        return
    
    # 自定义格式化函数：最多4位有效数字，不使用科学计数法，控制显示长度避免重叠
    def format_annotation(x):
        """Format Fig. S4 values without scientific notation."""
        if pd.isna(x) or x == 0:
            return "0"
        abs_x = abs(x)
        if abs_x >= 100:
            return f"{int(round(x))}"
        if abs_x >= 10:
            return f"{x:.1f}"
        if abs_x >= 1:
            return f"{x:.2f}"
        # Publication Fig. S4 contract: values below 1 H use two decimals.
        return f"{x:.2f}"

    plt.figure(figsize=(20, 20))   
    # --- 核心修改 ---
    # 1. 获取矩阵的维度 N
    n_coils = matrix.shape[0]
    
    # 2. 创建从 1 到 N 的标签
    #    例如，如果 N=8, tick_labels 将是 [1, 2, 3, 4, 5, 6, 7, 8]
    tick_labels = range(1, n_coils + 1)
    
    # 3. 将矩阵转换为带有新索引的 DataFrame，这能让
    #    seaborn自动使用我们想要的标签。
    df_to_plot = pd.DataFrame(matrix, index=tick_labels, columns=tick_labels)
    
    # 4. 创建格式化后的标注矩阵（最多4位有效数字，不使用科学计数法）
    annot_matrix = np.array([[format_annotation(val) for val in row] for row in df_to_plot.values])
    
    # 在调用 `heatmap` 时传入我们处理过的数据
    # 对电感矩阵取对数（以10为底），避免log(0)报错，先将0替换为极小值
    # 颜色和图例采用对数，但标注显示原始数据（格式化后）
    # Plot the original inductance values on a linear scale.  The colorbar
    # uses readable decimal labels rather than lg(Inductance).
    def format_colorbar(x, _pos):
        return format_annotation(x)

    ax = sns.heatmap(
        df_to_plot,
        annot=False,
        cmap='YlGnBu',
        linewidths=.5,
        square=True,
        cbar_kws={
            "shrink": 0.85,
            "format": FuncFormatter(format_colorbar),
        },
    )
    # The final publication SVG rescales the panel to 160 mm.  Explicit
    # source-space padding keeps the axes title, tick labels, and colour-bar
    # title visually separated after that normalization.
    cbar = ax.collections[0].colorbar
    cbar.set_label("Inductance [H]", labelpad=18)
    cbar.ax.tick_params(pad=4)
    # Annotate every cell; choose text colour from the linear colour scale.
    _lo = float(np.nanmin(df_to_plot.values))
    _hi = float(np.nanmax(df_to_plot.values))
    _mid = 0.5 * (_lo + _hi)
    for _i in range(n_coils):
        for _j in range(n_coils):
            _color = 'white' if df_to_plot.values[_i, _j] > _mid else 'black'
            ax.text(
                _j + 0.5,
                _i + 0.5,
                format_annotation(df_to_plot.values[_i, _j]),
                ha='center',
                va='center',
                fontsize=13,
                color=_color,
            )
    ax.set_xticks(np.arange(n_coils) + 0.5)
    ax.set_xticklabels(tick_labels, rotation=0, ha='center', fontsize=16)
    ax.set_yticks(np.arange(n_coils) + 0.5)
    ax.set_yticklabels(tick_labels, rotation=0, ha='right', va='center', fontsize=16)
    ax.tick_params(axis='x', pad=4)
    ax.tick_params(axis='y', pad=4)

    #plt.title(title, fontsize=16)
    # 更新坐标轴标题以反映新的编号方式
    ax.set_xlabel("Coil Number", labelpad=16)
    ax.set_ylabel("Coil Number", labelpad=18)
    
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['axes.unicode_minus'] = False
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"热力图已保存至: {save_path}")
    else:
        plt.show()
    plt.close()

def LEGACY_DISCRETE_RADIAL_MODEL(Npw, Ntape_coil, rho_turn: float) -> float:
    """V6.3 modulo-based radial model, retained only for V6.4 regression audits."""
    N_total_layers = Ntape_coil
    pitch = cfg.R2 / N_total_layers if N_total_layers > 0 else 0
    k_indices = np.arange(N_total_layers)
    rho_values = np.where(k_indices % Npw == 0, rho_turn, cfg.RHO_TAPE)
    R_k_values = cfg.R1 + k_indices * pitch
    Dk_values = cfg.L1 + np.pi * R_k_values + np.pi * (R_k_values + cfg.L1 / 2)
    return float(np.sum(rho_values / (cfg.WID * Dk_values)))


def continuous_effective_turn_rho_values(Npw, Ntape_coil, rho_turn: float) -> tuple[np.ndarray, dict]:
    """Return the V6.4 homogeneous radial-resistivity field and its contract audit.

    ``Npw`` remains an integer tape count per physical turn.  Only the effective
    number of turn interfaces is homogenized; no rounding or modulo allocation
    is used in this central V6.4 model.
    """
    if not isinstance(Npw, (int, np.integer)) or Npw < 1:
        raise ValueError("Npw must be a positive integer number of parallel tapes per turn")
    if Ntape_coil < 1:
        raise ValueError("Ntape_coil must be positive")
    n_turn_eff = Ntape_coil / Npw
    n_tt_eff = n_turn_eff - 1.0
    if n_tt_eff < 0:
        raise ValueError("N_tt_eff must be non-negative; require Npw <= Ntape_coil")
    f_tt = n_tt_eff / (Ntape_coil - 1) if Ntape_coil > 1 else 0.0
    if not 0.0 <= f_tt <= 1.0:
        raise ValueError("continuous turn-interface fraction is outside [0, 1]")
    rho_values = np.full(Ntape_coil, cfg.RHO_TAPE, dtype=float)
    rho_values[0] = rho_turn
    if Ntape_coil > 1:
        rho_values[1:] = cfg.RHO_TAPE + f_tt * (rho_turn - cfg.RHO_TAPE)
    rho_turn_weight = 1.0 + (Ntape_coil - 1) * f_tt
    audit = {
        "N_turn_eff": n_turn_eff,
        "N_tt_eff": n_tt_eff,
        "f_tt": f_tt,
        "continuous_rho_turn_weight": rho_turn_weight,
        "expected_rho_turn_weight": n_turn_eff,
        "weight_conservation_residual": rho_turn_weight - n_turn_eff,
        "rho_inner_boundary": float(rho_values[0]),
    }
    return rho_values, audit


def calculate_radial_resistance(Npw, Ntape_coil, rho_turn: float) -> float:
    """Calculate one pancake's V6.4 continuous effective-turn radial resistance.

    The existing geometric map remains unchanged and is linear in the layer
    resistivities.  V6.3's discrete model is deliberately retained above for
    regression only and is never called by this central V6.4 calculation.
    """
    N_total_layers = Ntape_coil
    pitch = cfg.R2 / N_total_layers if N_total_layers > 0 else 0
    k_indices = np.arange(N_total_layers)
    rho_values, _ = continuous_effective_turn_rho_values(Npw, Ntape_coil, rho_turn)

    # 2. 计算每一层的物理周长 (Dk)
    # 这个计算基于每一层的物理位置，与 Npw 无关
    R_k_values = cfg.R1 + k_indices * pitch
    Dk_values = cfg.L1 + np.pi * R_k_values + np.pi * (R_k_values + cfg.L1 / 2)

    # 3. 计算每一层的电阻 R_layer = rho / Area，并求和
    # Area = 导体宽度 * 该层周长
    layer_resistances = rho_values / (cfg.WID * Dk_values)
    return float(np.sum(layer_resistances))

def calculate_t2t_resistance(Npw, rho_turn: float) -> float:
    """
    计算单个饼状线圈的等效匝间电阻。

    [MODIFIED] 修正了计算逻辑：
    1.  匝间电阻是径向所有 并绕匝和并绕匝电阻的串联总和。
    2.  Npw 决定了在这些层中，"匝间电阻率" (rho_turn) 出现的频率。
    """
    # 总的径向层数是固定的
    N_total_layers = cfg.N_TOTAL_TAPE
    
    # 每层的物理厚度增量 (pitch)
    pitch = cfg.R2 / N_total_layers if N_total_layers > 0 else 0

    # 创建一个从 0 到 N_total_layers-1 的索引数组，代表每一层
    k_indices = np.arange(N_total_layers)

    # --- 核心逻辑修正 ---
    # 1. 判断每一层边界的属性
    # 如果一个层的索引号 k 是 Npw 的倍数，那么它代表一个新的物理匝的开始，
    # 其内边界是“匝-匝”边界，应使用 rho_turn。
    # 其他所有层都是同一个物理匝内部的“带-带”边界，使用 rho_tape=0。
    is_turn_boundary = (k_indices % Npw == 0)
    rho_values = np.where(is_turn_boundary, rho_turn, 0)

    # 2. 计算每一层的物理周长 (Dk)
    # 这个计算基于每一层的物理位置，与 Npw 无关
    R_k_values = cfg.R1_BASE + k_indices * pitch
    Dk_values = cfg.L1 + np.pi * R_k_values + np.pi * (R_k_values + cfg.L1 / 2)

    # 3. 计算每一层的电阻 R_layer = rho / Area，并求和
    # Area = 导体宽度 * 该层周长
    layer_resistances = rho_values / (cfg.WID * Dk_values)
    Rt2t_total = np.sum(layer_resistances)
    
    return  Rt2t_total
