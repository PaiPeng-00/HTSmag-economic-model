# mutual_inductance_equ_Dshape_v3_refined.py
import pandas as pd
import numpy as np
from numba import cuda
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import config as cfg
# --- Global font configuration ---
# Apply at module import so all plots share the same style
plt.rcParams['font.family'] = 'Arial' 
# Axes labels
plt.rcParams['axes.labelsize'] = 20
# Figure titles
plt.rcParams['axes.titlesize'] = 20
# Tick labels
plt.rcParams['xtick.labelsize'] = 20
# Global y-axis tick label size
plt.rcParams['ytick.labelsize'] = 20
# Legend fonts
plt.rcParams['legend.fontsize'] = 20

mapsize = 12
# ========================
# 1. Geometry construction utilities
# ========================

def generate_equivalent_group_D_turn(
    L1: float, R1_base: float, ng, dR: float, center: Tuple[float, float, float],
    points_straight = 100, points_arc = 50
) -> np.ndarray:
    """
    Generate equivalent path points for a group of D-shaped coil turns.
    This replaces `ng` individual turns with a single equivalent turn
    using averaged geometric parameters.

    [REFINEMENT] Adds basic input validation.
    """
    # Input validation
    assert L1 > 0, "Straight length L1 must be positive."
    assert R1_base > 0, "Base radius R1_base must be positive."
    assert ng > 0, "Number of turns ng must be positive."
    assert points_straight > 1 and points_arc > 1, "Number of points must exceed 1."

    x0, y0, z0 = center

    # Average radius over ng turns
    R1_start = R1_base
    R1_end = R1_base + (ng - 1) * dR
    R1_avg = (R1_start + R1_end) / 2
    R_arc_avg = L1 / 2 + R1_avg

    # (1) Top straight segment
    straight_x = np.linspace(-L1 / 2, L1 / 2, points_straight)
    straight_y = np.full_like(straight_x, R1_avg)

    # (2) Right-side small arc
    theta_right = np.linspace(np.pi / 2, 0, points_arc)
    arc_right_x = L1 / 2 + R1_avg * np.cos(theta_right)
    arc_right_y = R1_avg * np.sin(theta_right)

    # (3) Bottom large arc
    theta_bottom = np.linspace(0, -np.pi, points_straight)
    arc_bottom_x = R_arc_avg * np.cos(theta_bottom)
    arc_bottom_y = R_arc_avg * np.sin(theta_bottom)

    # (4) Left-side small arc
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
    Build a complete multi-coil equivalent-geometry model.

    [REFINEMENT] Adds type hints and basic input validation.
    """
    # Input validation
    assert all(arg > 0 for arg in [Np, Nt, Nm, ng, L1, R1_base, thickness, dist, points_per_turn]), \
        "All numeric parameters must be positive."
    assert Nt % Nm == 0, "Total turns Nt must be divisible by Nm (number of modules)."

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
    """Expand a group weight into an array matching the number of segments."""
    if not isinstance(weight_group, np.ndarray) or weight_group.size == 0:
        raise ValueError("weight_group must be a non-empty NumPy array.")
    return np.full(target_len, weight_group[0], dtype=np.float32)

# ========================
# 2. GPU computation (mutual inductance)
# ========================

@cuda.jit
def mutual_inductance_kernel(dl_i, mid_i, w_i, dl_j, mid_j, w_j, result_array):
    """
    CUDA kernel to accumulate mutual-inductance contributions between two
    groups of line elements.

    Each GPU thread handles one element of coil i and loops over all
    elements of coil j.
    """
    idx_i = cuda.grid(1)
    if idx_i >= mid_i.shape[0]:
        return

    r1x, r1y, r1z = mid_i[idx_i]
    dl1x, dl1y, dl1z = dl_i[idx_i]
    wi = w_i[idx_i]

    # Accumulate contributions from all elements of coil j
    total_contribution = 0.0
    for idx_j in range(mid_j.shape[0]):
        r2x, r2y, r2z = mid_j[idx_j]
        dl2x, dl2y, dl2z = dl_j[idx_j]
        wj = w_j[idx_j]

        dx, dy, dz = r2x - r1x, r2y - r1y, r2z - r1z
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)

        # Skip near-coincident elements to avoid numerical singularities
        if dist > 1e-9:
            dot_product = dl1x * dl2x + dl1y * dl2y + dl1z * dl2z
            total_contribution += (dot_product / dist) * wi * wj

    # Atomic add to safely update the global result
    cuda.atomic.add(result_array, 0, total_contribution)


def compute_mutual_inductance_gpu(
    dl_i: np.ndarray, mid_i: np.ndarray, w_i: np.ndarray,
    dl_j: np.ndarray, mid_j: np.ndarray, w_j: np.ndarray
) -> float:
    """
    Compute mutual inductance between two groups of line elements on the GPU.

    Returns:
        Mutual inductance in henry (H).
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
    """Compute the full mutual-inductance matrix between all coil groups."""
    N = len(pts_list)
    if N == 0:
        return np.array([])
    M = np.zeros((N, N), dtype=np.float64)

    print("Pre-processing geometry data for GPU computation ...")
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

    print(f"Computing {N}x{N} group-to-group mutual-inductance matrix on GPU ...")
    for i in range(N):
        for j in range(i, N):  # use symmetry M_ij = M_ji
            dl_i, mid_i, w_i = preprocessed_data[i]
            dl_j, mid_j, w_j = preprocessed_data[j]
            
            val = compute_mutual_inductance_gpu(dl_i, mid_i, w_i, dl_j, mid_j, w_j)
            
            M[i, j] = val
            if i != j:
                M[j, i] = val

    print("Finished computing group-to-group inductance matrix.")
    return M

def aggregate_matrix_to_pancakes(
    group_matrix: np.ndarray, info: List[Tuple[int, int]], Np
) -> np.ndarray:
    """Aggregate group-to-group matrix into a pancake-to-pancake matrix."""
    if group_matrix.size == 0:
        return np.array([])
        
    pancake_matrix = np.zeros((Np, Np), dtype=np.float64)
    info_array = np.array(info)
    
    print("Aggregating group matrix into pancake matrix ...")
    for i in range(Np):
        for j in range(i, Np):
            groups_in_pancake_i = np.where(info_array[:, 0] == i)[0]
            groups_in_pancake_j = np.where(info_array[:, 0] == j)[0]
            
            sub_matrix = group_matrix[np.ix_(groups_in_pancake_i, groups_in_pancake_j)]
            total_inductance = np.sum(sub_matrix)
            
            pancake_matrix[i, j] = total_inductance
            if i != j:
                pancake_matrix[j, i] = total_inductance

    print("Aggregation finished.")
    return pancake_matrix

# ========================
# 3. Visualisation helpers
# ========================

def plot_multicoil_paths(pts_list: List[np.ndarray], title: str = 'Equivalent multi-coil paths'):
    """Plot all coil paths in 3D."""
    if not pts_list:
        print("Warning: empty path list; nothing to plot.")
        return
        
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    colors = plt.cm.viridis(np.linspace(0, 1, len(pts_list)))
    
    for idx, pts in enumerate(pts_list):
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], label=f'Coil group {idx}', linewidth=1.5, color=colors[idx])

    ax.set_xlabel('X (m)'), ax.set_ylabel('Y (m)'), ax.set_zlabel('Z (m)')
    ax.set_title(title, fontsize=16), ax.legend(), ax.grid(True)
    ax.autoscale_view()
    plt.tight_layout(), plt.show()

def plot_inductance_heatmap(
    matrix: np.ndarray, title: str = 'Inductance Matrix', save_path: Optional[str] = None
):
    """Visualise an inductance matrix as a heatmap."""
    if matrix.size == 0:
        print("Warning: input inductance matrix is empty; cannot plot heatmap.")
        return
    
    # Custom formatting: up to 4–6 significant digits, prefer fixed notation
    def format_annotation(x):
        """Format annotation values with limited significant digits."""
        if pd.isna(x) or x == 0:
            return "0"
        
        abs_x = abs(x)
        
        # Choose formatting based on magnitude to keep labels compact
        # while preserving a reasonable number of significant digits.

        # ≥ 10000: round to hundreds (e.g. 12300)
        if abs_x >= 10000:
            rounded = round(x, -2)
            return f"{int(rounded)}"
        
        # ≥ 1000: round to integer (e.g. 1234)
        elif abs_x >= 1000:
            return f"{int(round(x))}"
        
        # ≥ 100: 1 decimal place (e.g. 123.4)
        elif abs_x >= 100:
            return f"{x:.1f}"
        
        # ≥ 10: 2 decimal places (e.g. 12.34)
        elif abs_x >= 10:
            return f"{x:.2f}"
        
        # ≥ 1: 3 decimal places
        elif abs_x >= 1:
            return f"{x:.3f}"
        
        # ≥ 0.1: 3 decimal places
        elif abs_x >= 0.1:
            return f"{x:.3f}"
        
        # ≥ 0.01: 4 decimal places
        elif abs_x >= 0.01:
            return f"{x:.4f}"
        
        # ≥ 0.001: 5 decimal places
        elif abs_x >= 0.001:
            return f"{x:.5f}"
        
        # < 0.001: 6 decimal places
        else:
            return f"{x:.6f}"
    
    plt.figure(figsize=(20, 20))   
    # 1) Matrix dimension
    n_coils = matrix.shape[0]
    
    # 2) Tick labels from 1..N (human-friendly indexing)
    tick_labels = range(1, n_coils + 1)
    
    # 3) Convert to DataFrame so seaborn uses our labels
    df_to_plot = pd.DataFrame(matrix, index=tick_labels, columns=tick_labels)
    
    # 4) Build formatted annotation matrix
    annot_matrix = np.array([[format_annotation(val) for val in row] for row in df_to_plot.values])
    
    # 5) Take log10 of inductance (avoid log(0) by replacing zeros)
    #    Colours use log10, annotations show formatted original values.
    matrix_log = np.log10(np.where(df_to_plot.values > 0, df_to_plot.values, 1e-12))
    df_log = pd.DataFrame(matrix_log, index=df_to_plot.index, columns=df_to_plot.columns)
    # Plot heatmap: colour & colourbar use log10(inductance),
    # annotations show formatted original inductance.
    ax = sns.heatmap(
        df_log,
        annot=annot_matrix,          # formatted original inductance
        fmt="",                      # annotations are pre-formatted strings
        cmap='YlGnBu',
        annot_kws={"size": 18},
        linewidths=.5,
        cbar_kws={"label": "lg (Inductance [H])", "shrink": 0.85}
    )
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # Title can be added externally; here we just label axes.
    plt.xlabel("Coil Number")
    plt.ylabel("Coil Number")
    
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['axes.unicode_minus'] = False
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to: {save_path}")
    else:
        plt.show()
    plt.close()

def calculate_radial_resistance(Npw, Ntape_coil, rho_turn: float) -> float:
    """
    Compute the equivalent radial resistance of a single pancake coil.

    1. The total resistance is the series sum of all radial layers.
    2. Npw determines how often the turn-to-turn resistivity rho_turn
       appears in those layers.
    """
    # Total number of radial layers
    N_total_layers = Ntape_coil
    
    # Radial thickness increment per layer (pitch)
    pitch = cfg.R2 / N_total_layers if N_total_layers > 0 else 0

    # Layer indices 0 .. N_total_layers-1
    k_indices = np.arange(N_total_layers)

    # Determine which layers correspond to actual turn boundaries.
    # If k is a multiple of Npw, that boundary is between turns (use rho_turn);
    # otherwise it is within a turn (tape-to-tape, use rho_tape).
    is_turn_boundary = (k_indices % Npw == 0)
    rho_values = np.where(is_turn_boundary, rho_turn, cfg.RHO_TAPE)

    # Physical perimeter Dk for each layer; depends on geometry only.
    R_k_values = cfg.R1 + k_indices * pitch
    Dk_values = cfg.L1 + np.pi * R_k_values + np.pi * (R_k_values + cfg.L1 / 2)

    # Electrical resistance per layer R_layer = rho / Area, sum over layers.
    # Area = tape width * perimeter of that layer.
    layer_resistances = rho_values / (cfg.WID * Dk_values)
    Rr_total = np.sum(layer_resistances)
    
    return Rr_total

def calculate_t2t_resistance(Npw, rho_turn: float) -> float:
    """
    Compute the equivalent turn-to-turn resistance of a single pancake.

    [MODIFIED] Logic:
    1. Turn-to-turn resistance is the series sum of all parallel-turn
       boundaries in the radial stack.
    2. Npw determines how often the boundary resistivity rho_turn appears.
    """
    # Total number of radial layers
    N_total_layers = cfg.N_TOTAL_TAPE
    
    # Radial thickness increment per layer (pitch)
    pitch = cfg.R2 / N_total_layers if N_total_layers > 0 else 0

    # Layer indices 0 .. N_total_layers-1
    k_indices = np.arange(N_total_layers)

    # Determine which layers are turn boundaries (k multiple of Npw).
    # Those boundaries use rho_turn; intra-turn boundaries use rho_tape = 0.
    is_turn_boundary = (k_indices % Npw == 0)
    rho_values = np.where(is_turn_boundary, rho_turn, 0)

    # Physical perimeter Dk for each layer (independent of Npw)
    R_k_values = cfg.R1_BASE + k_indices * pitch
    Dk_values = cfg.L1 + np.pi * R_k_values + np.pi * (R_k_values + cfg.L1 / 2)

    # Layer resistance R_layer = rho / Area, sum over all layers.
    # Area = tape width * perimeter for that layer.
    layer_resistances = rho_values / (cfg.WID * Dk_values)
    Rt2t_total = np.sum(layer_resistances)
    
    return  Rt2t_total