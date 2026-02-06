# 02_simulate_charging.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pathlib import Path

import config as cfg
from utils import calculate_radial_resistance

plt.style.use('default')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
color_charge_hour = 'black'

def simulate_charging_system(L_matrix: np.ndarray, R_values: list, npw,
                             I_target_local=None, Ntape_coil_local=None):
    """Optional I_target_local, Ntape_coil_local; else use cfg."""
    n = L_matrix.shape[0]
    I_target_used = I_target_local if I_target_local is not None else cfg.I_TARGET
    total_tape_used = Ntape_coil_local if Ntape_coil_local is not None else cfg.N_TOTAL_TAPE
    I_steady_total = I_target_used * npw
    charge_duration_s = cfg.CHARGE_HOURS * 3600
    ramp_rate = I_steady_total / charge_duration_s if charge_duration_s > 0 else 0
    total_duration_s = charge_duration_s + cfg.STEADY_HOURS * 3600
    try:
        L_inv = np.linalg.inv(L_matrix)
    except np.linalg.LinAlgError:
        epsilon = 1e-9
        L_regularized = L_matrix + np.eye(n) * epsilon
        try:
            L_inv = np.linalg.inv(L_regularized)
        except np.linalg.LinAlgError:
            print("Error: matrix still singular after regularization.")
            return None, None, None, None, None
    R_diag = np.diag(R_values)

    def ode_system(t, I_L):
        I_total_t = min(ramp_rate * t, I_steady_total)
        dI_L_dt = L_inv @ (R_diag @ (np.full(n, I_total_t) - I_L))
        return dI_L_dt

    t_eval_charge = np.linspace(0, charge_duration_s, int(cfg.CHARGE_HOURS * 60) + 1)
    initial_I_L = np.zeros(n)
    solution_charge = solve_ivp(
        fun=ode_system,
        t_span=[0, charge_duration_s],
        y0=initial_I_L,
        t_eval=t_eval_charge,
        method='Radau',
        rtol=1e-4,
        atol=1e-4
    )
    if not solution_charge.success:
        print(f"Error: charge phase failed - {solution_charge.message}")
        return None, None, None, None, None

    t_eval_steady = np.linspace(charge_duration_s, total_duration_s, int(cfg.STEADY_HOURS * 60) + 1)
    initial_I_L_steady = solution_charge.y[:, -1]
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
        print(f"Error: steady phase failed - {solution_steady.message}")
        return None, None, None, None, None
    t_sol = np.concatenate([solution_charge.t, solution_steady.t[1:]])
    I_L_sol = np.hstack([solution_charge.y, solution_steady.y[:, 1:]]).T
    I_total_over_time = np.minimum(ramp_rate * t_sol, I_steady_total)
    I_R_sol = I_total_over_time[:, np.newaxis] - I_L_sol
    P_R_sol = I_R_sol**2 * R_values

    return t_sol, I_L_sol, I_R_sol, P_R_sol, I_total_over_time

def calculate_time_to_999(t_seconds, I_L_solution, npw, I_target_per_conductor=None):
    """Return time (hours) to reach 99.9% of total target current, or None."""
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
        return None
def plot_current_ratios(results_dict, title, output_path_prefix, charge_end_time=None):
    """Plot radial and azimuthal current ratio (%); mark end of charge."""
    if not results_dict:
        print("No data to plot.")
        return

    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(results_dict)))

    for idx, (label, (t_h, I_L, I_R, I_total)) in enumerate(results_dict.items()):
        total_L = np.sum(I_L, axis=1)
        total_R = np.sum(I_R, axis=1)
        total_all = total_L + total_R
        total_all[total_all == 0] = np.nan

        ratio_R = total_R / total_all * 100
        ratio_L = total_L / total_all * 100

        plt.plot(t_h, ratio_R, lw=2, color=colors[idx], label=f"{label} (Radial)")
        plt.plot(t_h, ratio_L, "--", lw=2, color=colors[idx], label=f"{label} (Azimuthal)")

    if charge_end_time is not None:
        plt.axvline(x=charge_end_time, ls="--", c=color_charge_hour, alpha=0.6, label="End of charge")

    plt.xlabel("Time (h)")
    plt.ylabel("Current Ratio (%)")
    #plt.title(title)
    plt.gcf().subplots_adjust(top=0.88)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 105)
    plt.tight_layout()
    plt.savefig(f"{output_path_prefix}_Current_Ratio.svg", dpi=300)
    plt.close()

def plot_simulation_results(t_h, I_L, I_R, P_R, title_prefix, output_path):
    """Plot currents and power."""
    n = I_L.shape[1]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for i in range(n):
        ax1.plot(t_h, I_L[:, i], label=f'Coil {i+1}')
        ax2.plot(t_h, I_R[:, i], '--', label=f'Coil {i+1}')
    ax1.axvline(x=cfg.CHARGE_HOURS, ls='--', c=color_charge_hour, alpha=0.7)
    ax2.axvline(x=cfg.CHARGE_HOURS, ls='--', c=color_charge_hour, alpha=0.7)
    ax1.set_ylabel('Azimuthal Current (A)'), ax1.set_title(f'{title_prefix} - Inductor Currents'), ax1.grid(True), ax1.legend()
    ax2.set_ylabel('Radial Current (A)'), ax2.set_title(f'{title_prefix} - Resistor Currents'), ax2.grid(True), ax2.legend()
    plt.xlabel('Time (h)'), plt.tight_layout()
    plt.savefig(output_path / "currents.svg", dpi=300)
    # print(f"Saved: {output_path / 'currents.svg'}")
    plt.close()

def plot_radial_current_ratio_multi(results_dict, title, output_path_prefix, charge_end_time=None):
    """
    """Plot radial current ratio (%) vs time; one curve per (Npw, rho_turn)."""
    if not results_dict:
        print("No data for radial ratio.")
        return

    plt.figure(figsize=(8, 6))
    colors = plt.cm.plasma(np.linspace(0, 1, len(results_dict)))

    for idx, (label, (t_h, I_L, I_R, I_total)) in enumerate(results_dict.items()):
        total_L = np.sum(I_L, axis=1)
        total_R = np.sum(I_R, axis=1)
        total_all = total_L + total_R
        total_all[total_all == 0] = np.nan

        ratio_R = total_R / total_all * 100
        plt.plot(t_h, ratio_R, lw=2, color=colors[idx], label=f"{label}")

    if charge_end_time is not None:
        plt.axvline(x=charge_end_time, ls="--", c=color_charge_hour, alpha=0.6, label="End of charge")

    plt.xlabel("Time (h)")
    plt.ylabel("Radial current ratio (%)")
    #plt.title(title + " — Radial Component")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(0, 105)
    plt.tight_layout()
    plt.savefig(f"{output_path_prefix}_Radial_Current_Ratio.svg", dpi=300)
    plt.close()

def plot_azimuthal_current_ratio_multi(results_dict, title, output_path_prefix, charge_end_time=None):
    """
    """Plot azimuthal current ratio (%) vs time; one curve per (Npw, rho_turn)."""
    if not results_dict:
        print("No data for azimuthal ratio.")
        return

    plt.figure(figsize=(8, 6))
    colors = plt.cm.cividis(np.linspace(0, 1, len(results_dict)))

    for idx, (label, (t_h, I_L, I_R, I_total)) in enumerate(results_dict.items()):
        total_L = np.sum(I_L, axis=1)
        total_R = np.sum(I_R, axis=1)
        total_all = total_L + total_R
        total_all[total_all == 0] = np.nan

        ratio_L = total_L / total_all * 100
        plt.plot(t_h, ratio_L, lw=2, color=colors[idx], label=f"{label}")

    if charge_end_time is not None:
        plt.axvline(x=charge_end_time, ls="--", c=color_charge_hour, alpha=0.6, label="End of charge")

    plt.xlabel("Time (h)")
    plt.ylabel("Azimuthal current ratio (%)")
    #plt.title(title + " — Azimuthal Component")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(0, 105)
    plt.tight_layout()
    plt.savefig(f"{output_path_prefix}_Azimuthal_Current_Ratio.svg", dpi=300)
    plt.close()

def plot_charging_time(df, x_col, y_col, title, output_path, log_x=False):
    """Plot charging time vs a variable."""
    plt.figure(figsize=(10, 6))
    plt.plot(df[x_col], df[y_col], 'o-', label='Time to 99.9%')
    if log_x:
        plt.xscale('log')
    plt.xlabel(x_col)
    plt.ylabel(y_col.replace('_', ' ') + ' (hours)')
    #plt.title(title)
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")

# =============================================================================
def main():
    """Run both experiments (vary Npw; vary rho_turn)."""
    script_dir = Path(__file__).parent.resolve()
    inductance_dir = script_dir / cfg.INDUCTANCE_OUTPUT_DIR
    base_output = script_dir / cfg.TF_SYSTEM_CHARGING_SIM_OUTPUT_DIR
    base_output.mkdir(exist_ok=True)
    if not inductance_dir.exists():
        print(f"Error: inductance dir not found: {inductance_dir}")
        print("Run 01_calculate_inductance.py first.")
        return
    if not base_output.exists():
        print(f"Error: output dir not found: {base_output}")
        return
    for T, case in cfg.TEMPERATURE_CASES.items():
        label = case['label']
        Ntape_coil_case = case['Ntape_coil']
        Ip_case = case['Ip']

        print("\n" + "-"*60)
        print(f"Running temperature case: {label} (T={T} K)  Ip={Ip_case}, Ntape_coil={Ntape_coil_case}")
        print("-"*60)

        # Per-temperature output dir
        output_for_T = base_output / f"T_{str(T).replace('.','p')}"
        output_for_T.mkdir(parents=True, exist_ok=True)
        print("\n" + "="*50)
        print("Experiment 1: vary Npw, fixed resistivity")
        print("="*50)
        results_exp1 = {}
        exp1_summary_data = []
        output_dir_exp1 = output_for_T / f"Exp1_Vary_Npw_fix_rhot={cfg.RHO_TURN_EXP1_FIXED*1e10}"
        output_dir_exp1.mkdir(parents=True, exist_ok=True)
        for npw in cfg.NPW_LIST_EXP1:
            
            print(f"\n--- Npw = {npw} ---")        
            output_dir_exp = output_dir_exp1 / f"Npw_{npw}"
            output_dir_exp.mkdir(parents=True, exist_ok=True)
            ng_current = cfg.get_ng_for_npw(npw)
            print(f"\nComputing Npw = {npw} (ng = {ng_current})...")
    
            matrix_file = inductance_dir / f"TF_system_L_matrix.xlsx"
            if not matrix_file.exists():
                print(f"Warning: inductance file not found {matrix_file}, skip Npw={npw}")
                continue
            L_matrix = pd.read_excel(matrix_file, header=None).values
            Nt_local = Ntape_coil_case / npw
            Nturn_magnet = Nt_local * cfg.NP
            L_matrix = L_matrix * Nturn_magnet**2
            print(L_matrix)
            
            Rr = calculate_radial_resistance(
                npw, Ntape_coil=Ntape_coil_case, rho_turn=cfg.RHO_TURN_EXP1_FIXED) * cfg.NP
            R_values = [Rr] * cfg.Ntf
            print(f"Radial resistance per TF: {Rr*1e6:.2f} μΩ")
            print(Ip_case, Ntape_coil_case)
            t, I_L, I_R, P_R, I_total = simulate_charging_system(
                L_matrix, R_values, npw=npw,
                I_target_local=Ip_case, Ntape_coil_local=Ntape_coil_case)
            if t is None: continue
            time_999_h = calculate_time_to_999(
                t, I_L, npw, I_target_per_conductor=Ip_case)
            if time_999_h is not None:
                print(f"Time to 99.9%: {time_999_h:.2f} h")
            else:
                print("Did not reach 99.9% within simulation time.")
            exp1_summary_data.append({'Npw': npw, 'Radial_Resistance_uOhm': Rr * 1e6, 'Time_to_99.9%_h': time_999_h})
            
            t_h = t / 3600
            results_exp1[f"Npw = {npw}"] = (t_h, I_L, I_R, I_total)
            plot_simulation_results(t_h, I_L, I_R, P_R, f"Npw={npw}", output_dir_exp)

        # Plot ratio curves
        plot_current_ratios(results_exp1, "Exp1 - Vary Npw", output_dir_exp1 / "Exp1_Vary_Npw", charge_end_time=cfg.CHARGE_HOURS)
        plot_radial_current_ratio_multi(results_exp1, "Exp1 - Vary Npw", output_dir_exp1 / "Exp1_Vary_Npw", charge_end_time=cfg.CHARGE_HOURS)
        plot_azimuthal_current_ratio_multi(results_exp1, "Exp1 - Vary Npw", output_dir_exp1 / "Exp1_Vary_Npw", charge_end_time=cfg.CHARGE_HOURS)



        df_exp1 = pd.DataFrame(exp1_summary_data)
        excel_path_exp1 = output_dir_exp1 / "Exp1_charging_time_summary.xlsx"
        df_exp1.to_excel(excel_path_exp1, index=False)
        print(f"\nExp1 summary saved: {excel_path_exp1.name}")
        plot_charging_time(df_exp1.dropna(), 'Npw', 'Time_to_99.9%_h', 'Time to 99.9% Charge vs. Npw', output_dir_exp1 / "Exp1_charging_time_vs_Npw.svg")

        print("\n" + "="*50)
        print(f"Experiment 2: vary rho_turn, fixed Npw={cfg.NPW_EXP2_FIXED}")
        print("="*50)
        exp2_summary_data = []
        npw_fixed = cfg.NPW_EXP2_FIXED
        output_dir_exp2 = output_for_T / f"Exp2_Vary_Rho_fix_Npw={npw_fixed}"
        output_dir_exp2.mkdir(parents=True, exist_ok=True)
        results_exp2 = {}
        matrix_file_exp2 = inductance_dir / f"TF_system_L_matrix.xlsx"
        if not matrix_file_exp2.exists():
            print(f"Error: inductance file for Npw={npw_fixed} not found, cannot run Exp2.")
            return
        L_matrix_exp2 = pd.read_excel(matrix_file_exp2, header=None).values

        Nt_local = Ntape_coil_case / npw_fixed
        Nturn_magnet = Nt_local * cfg.NP
        L_matrix_exp2 = L_matrix * Nturn_magnet**2
        
        
        for rho_turn in cfg.RHO_TURN_LIST_EXP2:
            rho_turn_uOhm_cm2 = rho_turn * 1e10
            print(f"\n--- rho_turn = {rho_turn_uOhm_cm2:.0f} μΩ·cm² ---")

            output_dir_exp = output_dir_exp2 / f"Rho_{rho_turn_uOhm_cm2:.0f}"
            output_dir_exp.mkdir(parents=True, exist_ok=True)
            # Radial resistance per TF
            Rr = calculate_radial_resistance(
                npw_fixed, Ntape_coil= Ntape_coil_case, rho_turn=rho_turn) * cfg.NP
            R_values = [Rr] * cfg.Ntf
            print(f"Radial resistance: {Rr*1e6:.2f} μΩ")

            t, I_L, I_R, P_R, I_total = simulate_charging_system(L_matrix_exp2, 
                        R_values, npw=npw_fixed,
                        I_target_local=Ip_case, Ntape_coil_local=Ntape_coil_case)
            if t is None: continue

            time_999_h = calculate_time_to_999(
                t, I_L, npw, I_target_per_conductor=Ip_case)
            if time_999_h is not None:
                print(f"Time to 99.9%: {time_999_h:.2f} h")
            else:
                print("Did not reach 99.9% within simulation time.")
            exp2_summary_data.append({'rho_turn_uOhm_cm2': rho_turn_uOhm_cm2, 'Radial_Resistance_uOhm': Rr * 1e6, 'Time_to_99.9%_h': time_999_h})

            t_h = t / 3600
            label = f"ρ_turn = {rho_turn_uOhm_cm2:.0f} μΩ·cm²"
            results_exp2[label] = (t_h, I_L, I_R, I_total)
            plot_simulation_results(t_h, I_L, I_R, P_R, f"Rho={rho_turn_uOhm_cm2:.0f}", output_dir_exp)
        df_exp2 = pd.DataFrame(exp2_summary_data)
        excel_path_exp2 = output_dir_exp2 / "Exp2_charging_time_summary.xlsx"
        df_exp2.to_excel(excel_path_exp2, index=False)
        print(f"\nExp2 summary saved: {excel_path_exp2.name}")
        plot_charging_time(df_exp2.dropna(), 'rho_turn_uOhm_cm2', 'Time_to_99.9%_h', 'Time to 99.9% Charge vs. Turn Resistivity', output_dir_exp2 / "Exp2_charging_time_vs_Rho.svg", log_x=True)

            # Plot
        plot_current_ratios(results_exp2, "Exp2 - Vary Rho_turn", output_dir_exp2 / "Exp2_Vary_Rho", charge_end_time=cfg.CHARGE_HOURS)
        plot_radial_current_ratio_multi(results_exp2, "Exp2 - Vary Rho_turn", output_dir_exp2 / "Exp2_Vary_Rho", charge_end_time=cfg.CHARGE_HOURS)
        plot_azimuthal_current_ratio_multi(results_exp2, "Exp2 - Vary Rho_turn", output_dir_exp2 / "Exp2_Vary_Rho", charge_end_time=cfg.CHARGE_HOURS)



if __name__ == '__main__':
    main()