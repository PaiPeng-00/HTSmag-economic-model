import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import solve_ivp
from pathlib import Path
from matplotlib.colors import LogNorm, PowerNorm
# Import configuration directly from config.py
import config as cfg
# Import calculate_radial_resistance directly from utils.py
from utils import calculate_radial_resistance

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'          # regular font
plt.rcParams['mathtext.it'] = 'Arial:italic'   # italic
plt.rcParams['mathtext.bf'] = 'Arial:bold'     # bold
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['savefig.transparent'] = True
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'


# 1. Define value ranges for Npw and rho_turn
# Npw_values = np.concatenate((np.arange(1, 10, 1), np.arange(10, 60, 10), np.arange(60, 110, 20)))
Npw_values = np.concatenate((np.arange(1, 100, 1),
                            np.arange(100, 201, 10)))

rho_turn_values_uOhm_cm2 = np.concatenate(
    (
        np.arange(10, 110, 10),
        np.arange(100, 2000, 100),
        np.arange(2000, 11000, 1000),
    )
)  # turn‑to‑turn resistivity from 10 to 1e4 μΩ·cm²
rho_turn_values_Ohm_m2 = rho_turn_values_uOhm_cm2 * 1e-10  # convert to Ω·m² (1 μΩ·cm² = 1e-10 Ω·m²)

# Fixed Npw and rho_turn grids used for plotting
fixed_Npw_values = np.concatenate((np.arange(1, 10, 1),
                                    np.arange(10, 21, 2),
                                    np.arange(30, 61, 10),
                                    np.arange(60, 210, 20)))
fixed_rho_turn_values_uOhm_cm2 = np.concatenate((np.arange(10, 110, 10),
                                                np.arange(500, 2000, 500),
                                                np.arange(2000, 11000, 1000)))

# Custom tick locations used in plots
desired_npw_ticks = [1, 5, 10,  20, 100, 200]
desired_rho_ticks = [20, 50, 100,   1500, 5000, 10000]
# Core function copied from 2.0_simulate_charging.py
def simulate_charging_system(L_matrix: np.ndarray, R_values: list, npw,
                             I_target_local=None, Ntape_coil_local=None):
    """
    Simulate TF‑system charging with an inductance matrix and radial resistances.

    New parameters:
      - I_target_local: (float) target current per conductor (A) at this temperature; falls back to cfg.I_TARGET if None
      - Ntape_coil_local: (float) total tape count per coil; falls back to cfg.N_TOTAL_TAPE if None
    """
    n = L_matrix.shape[0]
    # Use external values if provided; otherwise fall back to global config
    I_target_used = I_target_local if I_target_local is not None else cfg.I_TARGET
    total_tape_used = Ntape_coil_local if Ntape_coil_local is not None else cfg.N_TOTAL_TAPE

    # Previously: I_steady_total = cfg.I_TARGET * ntape
    # Now: use I_target_used * npw (parallel strands npw)
    I_steady_total = I_target_used * npw

    n = L_matrix.shape[0]
    # Charging and steady‑state duration
    charge_duration_s = cfg.CHARGE_HOURS * 3600
    ramp_rate = I_steady_total / charge_duration_s if charge_duration_s > 0 else 0
    total_duration_s = charge_duration_s + cfg.STEADY_HOURS * 3600

    # Handle singular inductance matrix by adding small diagonal regularization if needed
    try:
        L_inv = np.linalg.inv(L_matrix)
    except np.linalg.LinAlgError:
        epsilon = 1e-9
        L_regularized = L_matrix + np.eye(n) * epsilon
        try:
            L_inv = np.linalg.inv(L_regularized)
        except np.linalg.LinAlgError:
            print("Error: inductance matrix remains singular after regularization; cannot solve.")
            return None, None, None, None, None

    R_diag = np.diag(R_values)

    # ODE system for coupled inductive‑resistive circuit
    def ode_system(t, I_L):
        I_total_t = min(ramp_rate * t, I_steady_total)
        dI_L_dt = L_inv @ (R_diag @ (np.full(n, I_total_t) - I_L))
        return dI_L_dt

    # Phase 1: charging ramp
    t_eval_charge = np.linspace(0, charge_duration_s, int(cfg.CHARGE_HOURS * 60) + 1)
    initial_I_L = np.zeros(n)
    
    solution_charge = solve_ivp(
        fun=ode_system,
        t_span=[0, charge_duration_s],
        y0=initial_I_L,
        t_eval=t_eval_charge,
        method='Radau',  # stiff solver
        rtol=1e-4,
        atol=1e-4,
    )
    if not solution_charge.success:
        print(f"Error: charging stage solve failed - {solution_charge.message}")
        return None, None, None, None, None

    # Phase 2: (quasi) steady‑state
    t_eval_steady = np.linspace(charge_duration_s, total_duration_s, int(cfg.STEADY_HOURS * 60) + 1)
    initial_I_L_steady = solution_charge.y[:, -1]  # initial condition: end of charging stage

    solution_steady = solve_ivp(
        fun=ode_system,
        t_span=[charge_duration_s, total_duration_s],
        y0=initial_I_L_steady,
        t_eval=t_eval_steady,
        method='Radau',
        rtol=1e-4,
        atol=1e-4
    )
    if not solution_steady.success:
        print(f"Error: steady‑state stage solve failed - {solution_steady.message}")
        return None, None, None, None, None
        
    # Merge both stages
    t_sol = np.concatenate([solution_charge.t, solution_steady.t[1:]])
    I_L_sol = np.hstack([solution_charge.y, solution_steady.y[:, 1:]]).T

    # Post‑processing: total current, radial current and power
    I_total_over_time = np.minimum(ramp_rate * t_sol, I_steady_total)
    I_R_sol = I_total_over_time[:, np.newaxis] - I_L_sol
    P_R_sol = I_R_sol**2 * R_values

    return t_sol, I_L_sol, I_R_sol, P_R_sol, I_total_over_time

def calculate_time_to_999(t_seconds, I_L_solution, npw, I_target_per_conductor=None):
    """
    Compute the time for the total azimuthal current to reach 99.9% of its
    target value.

    Args:
        t_seconds (np.ndarray): time array in seconds.
        I_L_solution (np.ndarray): azimuthal current solution, shape (n_times, n_pancakes).
        npw (int): number of parallel strands (Npw).
        I_target_per_conductor (float): target current per conductor (A). If None, uses cfg.I_TARGET.

    Returns:
        float: time to 99.9% of target current in hours. If not reached within
        the simulated window, returns the final simulation time (lower bound).
    """
    if t_seconds is None or I_L_solution is None:
        return None

    if I_target_per_conductor is None:
        I_target_per_conductor = cfg.I_TARGET

    n_pancakes = I_L_solution.shape[1]
    total_target_current = I_target_per_conductor * npw * n_pancakes
    target_999_current = 0.999 * total_target_current

    total_azimuthal_current_vs_time = np.sum(I_L_solution, axis=1)

    indices = np.where(total_azimuthal_current_vs_time >= target_999_current)

    if len(indices[0]) > 0:
        first_index = indices[0][0]
        time_to_999_seconds = t_seconds[first_index]
        return time_to_999_seconds / 3600.0
    else:
        # Did not reach 99.9% during simulation; return end time as a lower bound
        # (true time_999_h is likely larger than the simulated duration).
        return float(t_seconds[-1]) / 3600.0
        
def plot_charging_time_heatmap_fixed(df_full, fixed_Npw_values, fixed_rho_values, output_path):
    """Plot a heatmap for a subset of fixed Npw and rho_turn values."""
    # Post‑process DataFrame read from Excel
    df_from_excel = df_full
    # Ensure index is integer
    df_from_excel.index = np.round(df_from_excel.index.values, 0).astype(int)
    df_from_excel.columns = df_from_excel.columns.astype(int)

    # Convert fixed rho values to int as well
    fixed_rho_values = np.round(fixed_rho_values, 0).astype(int)

    # Filter to selected rows/columns
    df_filtered = df_from_excel.loc[
        df_from_excel.index.isin(fixed_rho_values),
        fixed_Npw_values]

    # Match AF heatmap convention: x‑axis rho_turn, y‑axis Npw
    matrix = df_filtered.values.T           # shape: Npw x rho
    npw_values = df_filtered.columns.values
    rho_values = df_filtered.index.values

    plt.figure(figsize=(6, 6))
    norm = PowerNorm(
        gamma=0.2,
        vmin=np.nanmin(matrix[matrix > 0]),
        vmax=np.nanmax(matrix)
    )

    ax = sns.heatmap(
        matrix,
        norm=norm,
        annot=False,
        fmt=".2f",
        cmap="cividis",
        cbar=False,
        xticklabels=[str(int(x)) for x in rho_values],
        yticklabels=[f"{int(y)}" for y in npw_values],
    )

    # Keep Npw increasing from bottom to top, consistent with AF plots
    ax.invert_yaxis()

    # Add contour lines
    X_plot, Y_plot = np.meshgrid(np.arange(len(rho_values)),
                                 np.arange(len(npw_values)))
    levels = [96,  97,100,  120, 
               200, 500, 1000, 5000, 10000]
    if cfg.CHARGE_HOURS == 24:
        levels = [
            24,
            24.5,
            25,
            30,
            50,
            100,
            150,
            200,
            300,
            500,
            1000,
            5000,
            10000,
        ]
    contour_lines = ax.contour(
        X_plot + 0.5, Y_plot + 0.5, matrix,
        levels=levels, colors='white', linestyles='dashed', linewidths=1.5
    )
    ax.clabel(contour_lines, inline=True, fontsize=14, fmt='%.2f')



    # Find tick positions for selected rho (x) and Npw (y)
    xtick_positions = [i for i, val in enumerate(rho_values) if val in desired_rho_ticks]
    ytick_positions = [i for i, val in enumerate(npw_values) if val in desired_npw_ticks]

    # Apply custom ticks and labels
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels([str(v) for v in desired_rho_ticks if v in rho_values], rotation=0)
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels([str(v) for v in desired_npw_ticks if v in npw_values], rotation=0)

    plt.xlabel('Turn-to-turn resistivity (μΩ·cm²)')
    plt.ylabel('Number of parallel-stacked')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def charge999(T, Ip_case, Ntape_coil_case):
    """Main driver to generate charging‑time grids and corresponding heatmaps."""

    # Paths
    script_dir = Path(__file__).parent.resolve()
    output_dir = script_dir / cfg.TF_SYSTEM_CHARGING_SIM_OUTPUT_DIR / f"Temp_{T}K_Ip_{Ip_case}A_Ntape_coil_{Ntape_coil_case}_charge999"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_excel_path = output_dir / cfg.TF_SYSTEM_CHARGING_SIM_OUTPUT_FILE_Npw1_200
    figure_dir = Path(cfg.OUTPUTS_FIGURES_DIR) / "charging_tf" / f"Temp_{T}K_Ip_{Ip_case}A_Ntape_coil_{Ntape_coil_case}_charge999"
    figure_dir.mkdir(parents=True, exist_ok=True)
    output_plot_path = figure_dir / cfg.TF_SYSTEM_CHARGING_SIM_OUTPUT_PLOT_FILE_Npw1_200
    inductance_dir = script_dir / cfg.INDUCTANCE_OUTPUT_DIR

    # 2) Try to read existing Excel data if present (resume)
    df_existing = None
    if output_excel_path.exists():
        print(f"Detected existing data file: {output_excel_path}, trying to read...")
        try:
            df_existing = pd.read_excel(output_excel_path, index_col=0)
            # Ensure dtypes for indices and columns are consistent
            df_existing.columns = df_existing.columns.astype(int)
            df_existing.index = df_existing.index.astype(float) 
            print("Successfully loaded existing data.")
        except Exception as e:
            print(f"Failed to read existing data: {e}. Recomputing from scratch.")
            df_existing = None

    # 3) Determine the union of all Npw and rho_turn values to build the DataFrame
    all_npw_values = set(Npw_values)
    all_rho_turn_values_uOhm_cm2 = set(rho_turn_values_uOhm_cm2.round(0))

    if df_existing is not None:
        all_npw_values.update(df_existing.columns.values)
        all_rho_turn_values_uOhm_cm2.update(df_existing.index.values)

    # Convert sets back to sorted lists/arrays for ordered DataFrame axes
    all_npw_values_sorted = sorted(list(all_npw_values))
    all_rho_turn_values_uOhm_cm2_sorted = sorted(list(all_rho_turn_values_uOhm_cm2))

    # 4) Initialize charging_time_df on the unified grid
    charging_time_df = pd.DataFrame(index=all_rho_turn_values_uOhm_cm2_sorted, 
                                    columns=all_npw_values_sorted)
    charging_time_df.index.name = 'rho_turn(μΩ·cm²)'
    charging_time_df.columns.name = 'Npw (Npw)'

    # 5) Merge existing data (if any) into the unified DataFrame
    if df_existing is not None:
        charging_time_df.update(df_existing)
        print("Existing data merged into the new DataFrame layout.")
    else:
        print("No existing data found; computing from scratch.")

    # 6) Read L_single_wound_matrix
    L_single_wound_file = inductance_dir / f"TF_system_L_matrix.xlsx"

    L_single_wound_matrix = None
    if L_single_wound_file.exists():
        L_single_wound_matrix = pd.read_excel(L_single_wound_file, header=None).values
        print(f"Successfully read L_single_wound_matrix from: {L_single_wound_file}")
    else:
        print(f"Warning: L_single_wound_matrix file not found: {L_single_wound_file}")
        print("Using a default example matrix for demonstration. "
              "Please provide the inductance file for accurate results.")
        # Fall back to an example matrix so the code can still run
        L_single_wound_matrix = np.diag(np.full(cfg.Ntf, 5e-3)) + \
                                np.diag(np.full(cfg.Ntf - 1, 0.5e-3), k=1) + \
                                np.diag(np.full(cfg.Ntf - 1, 0.5e-3), k=-1)

    # Ensure matrix dimensions match cfg.Ntf
    if L_single_wound_matrix.shape[0] != cfg.Ntf or L_single_wound_matrix.shape[1] != cfg.Ntf:
        print(f"Warning: inductance matrix shape {L_single_wound_matrix.shape} "
              f"does not match cfg.Ntf={cfg.Ntf}.")
        print("Using a default example matrix for demonstration. "
              "Please check the inductance file or cfg.Ntf settings.")
        L_single_wound_matrix = np.diag(np.full(cfg.Ntf, 5e-3)) + \
                                np.diag(np.full(cfg.Ntf - 1, 0.5e-3), k=1) + \
                                np.diag(np.full(cfg.Ntf - 1, 0.5e-3), k=-1)

    # 7) Loop over the grid and compute charging time for each combination
    print("Start computing charging‑time grid...")
    for i, rho_turn in enumerate(rho_turn_values_Ohm_m2):
        for j, npw in enumerate(Npw_values):
            # Current cell value (use .at for fast scalar access)
            current_rho_label = (rho_turn*1e10).round(0)
            current_npw_label = npw
            current_cell_value = charging_time_df.at[current_rho_label, current_npw_label]
        
            # Skip if this cell is already filled
            if pd.notna(current_cell_value):
                print(
                    f"  Skip Npw={npw}, rho_turn={rho_turn/1e-10:.0f} μΩ·cm² "
                    f"(existing value: {current_cell_value:.2f} h)"
                )
                continue
            print(f"  Computing Npw={npw}, rho_turn={rho_turn/1e-10:.0f} μΩ·cm² ...")
            
            # Scale inductance matrix with Npw and tape count:
            #   L_parallel = L_single * (Ntape_coil_case * cfg.NP / Npw)^2
            L_matrix_current = L_single_wound_matrix *(Ntape_coil_case*cfg.NP/npw)**2
            # Compute radial resistance; TF‑level resistance is sum over pancakes
            Rr = calculate_radial_resistance(npw, Ntape_coil_case, rho_turn) *cfg.NP
            R_values = [Rr] * cfg.Ntf  # assume identical radial resistance for each pancake

            # Simulate charging process
            t, I_L, _, _, _ = simulate_charging_system(
                L_matrix_current, R_values, npw=npw,
                I_target_local=Ip_case, Ntape_coil_local=Ntape_coil_case)
            
            # Compute 99.9% charging time
            time_999_h = calculate_time_to_999(t, I_L, npw=npw, I_target_per_conductor=Ip_case)
            
            # Store result in DataFrame
            charging_time_df.loc[rho_turn_values_uOhm_cm2[i].round(0), npw] = time_999_h
            
            if time_999_h is None:
                print("    Did not reach 99.9% of target current within simulation window.")
            else:
                print(f"    Charging time: {time_999_h:.2f} h")

    print("\nCharging‑time grid computed.")

    # Save data to Excel
    charging_time_df.to_excel(output_excel_path)
    print(f"Charging‑time data saved to: {output_excel_path}")

    # Reload from Excel (for demonstration of the read‑back workflow)
    df_from_excel = pd.read_excel(output_excel_path, index_col=0)
    df_from_excel.columns = df_from_excel.columns.astype(int)
    df_from_excel.index = df_from_excel.index.astype(float)

    print(fixed_rho_turn_values_uOhm_cm2)
    # Plot heatmap for the fixed subset
    plot_charging_time_heatmap_fixed(
        df_from_excel,
        fixed_Npw_values,
        fixed_rho_turn_values_uOhm_cm2,
        output_plot_path
    )

    print(f"\nHeatmap saved to {output_plot_path}")

if __name__ == "__main__":
    for temp in cfg.TEMPERATURE_CASES:
        Ip_case = cfg.TEMPERATURE_CASES[temp]['Ip']
        Ntape_coil_case = cfg.TEMPERATURE_CASES[temp]['Ntape_coil']
        charge999(temp, Ip_case, Ntape_coil_case)

