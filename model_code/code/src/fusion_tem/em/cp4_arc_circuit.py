"""CP4: ARC 18-coil TF-system circuit model.

(1) Scale the validated SPARC 18x18 single-turn inductance matrix to ARC by the
    single-turn self-inductance ratio L_ARC/L_SPARC (loop formula with equivalent
    radius r_eq). The 18-coil toroidal COUPLING PATTERN (M_ij/M_ii) is geometry-
    invariant (both machines have 18 toroidally-symmetric coils), so only the scale
    changes; this preserves the validated structure while moving to ARC size.
(2) Run the charging ODE (reusing simulate_charging_system) for representative
    (Npw, rho_turn) to get charging time and radial loss; compare SPARC vs ARC.
(3) Save ARC matrix + emit the COMSOL M11..M18 values and ODE file.
"""
import numpy as np
import pandas as pd
from pathlib import Path

from fusion_tem import device as cfg
from analysis_simulate_charging_tf_system import simulate_charging_system, calculate_time_to_999
from fusion_tem.utils import calculate_radial_resistance

HERE = Path(__file__).parent
mu0 = 4 * np.pi * 1e-7

# ---- equivalent radii (CP1/CP2) and conductor bundle minor radius ----
r_eq_S, R2_S = 1.78, 0.30      # SPARC
r_eq_A, R2_A = 4.14, 0.64      # ARC: R2 = WP radial thickness = Mf FEM dr_tf = 0.64 m (kept consistent across FEM / TA / inductance)
def L_loop(R, a):              # single-turn loop self-inductance ~ mu0 R (ln(8R/a)-2)
    return R * (np.log(8 * R / a) - 2.0)
scale = L_loop(r_eq_A, R2_A / 2) / L_loop(r_eq_S, R2_S / 2)
print(f"self-inductance scale L_ARC/L_SPARC = {scale:.3f}  (r_eq {r_eq_S}->{r_eq_A} m)")

# ---- load validated SPARC 18x18, scale to ARC ----
M_S = pd.read_excel(HERE / "data/raw/inductance/TF_system_L_matrix.xlsx", header=None).values
M_A = M_S * scale
print(f"SPARC M_ii = {np.diag(M_S).mean():.3e} H  ->  ARC M_ii = {np.diag(M_A).mean():.3e} H")
out = HERE / "data/raw/inductance/TF_system_L_matrix_ARC.xlsx"
pd.DataFrame(M_A).to_excel(out, header=False, index=False)
print(f"saved ARC matrix -> {out.name}")

print("\n=== COMSOL par5 M11..M18 (ARC, single-turn, first row) ===")
for k in range(8):
    print(f"  M1{k+1} = {M_A[0,k]:.3e}[H]")

# ---- ARC allocation (CP3), mirror original recipe exactly ----
AT_coil = 120 * 70e3
Ip_20K = 660.0
total_tapes_magnet = AT_coil / Ip_20K          # 12720 tapes per TF magnet
cfg.NP = 16                                     # pancakes (model default; ARC WP layers)
Ntape_per_pancake = total_tapes_magnet / cfg.NP # 795 (Ntape_coil_case in original)
cfg.R2 = R2_A                                   # ARC WP radial thickness (radial-resistance pitch)

print(f"\nARC: total_tapes/magnet={total_tapes_magnet:.0f}, NP={cfg.NP}, Ntape/pancake={Ntape_per_pancake:.0f}, Ip={Ip_20K} A")
print("\n=== charging time t_charge(99.9%) and radial loss: ARC ===")
print(f"{'Npw':>5} {'rho[uOhm.cm2]':>14} {'Nturn_magnet':>13} {'L_eff[H]':>10} {'t_charge[h]':>12} {'P_rad_pk[W]':>12}")
for Npw in [10, 50, 100]:
    Nt_local = Ntape_per_pancake / Npw
    Nturn_magnet = Nt_local * cfg.NP            # turns per magnet
    L_eff = M_A * Nturn_magnet ** 2            # scale single-turn matrix by Nturn^2
    for rho_uohm in [50, 5000]:
        rho = rho_uohm * 1e-10                  # uOhm.cm2 -> Ohm.m2
        Rr = calculate_radial_resistance(Npw, Ntape_per_pancake, rho) * cfg.NP  # x NP pancakes
        R_values = [Rr] * cfg.Ntf
        res = simulate_charging_system(L_eff, R_values, npw=Npw,
                                       I_target_local=Ip_20K, Ntape_coil_local=Ntape_per_pancake)
        t_s, I_L, I_R, P_R, _ = res
        tc_h = calculate_time_to_999(t_s, I_L, Npw, I_target_per_conductor=Ip_20K)
        tc_h = tc_h if tc_h is not None else float('nan')   # HOURS
        pr = np.nanmax(P_R) if P_R is not None else float('nan')
        print(f"{Npw:5d} {rho_uohm:14.0f} {Nturn_magnet:13.0f} {L_eff[0,0]:10.2f} {tc_h:12.2f} {pr:12.1f}")
