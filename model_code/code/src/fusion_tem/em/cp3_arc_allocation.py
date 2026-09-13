"""CP3: ARC ampere-turn allocation.

Fixed: ampere-turns/coil = 8.40 MA (= 120 cables x 70 kA, reproduces B0=9.2 T).
Npw (parallel-stacked tapes/turn) is the SCAN VARIABLE. Per-tape current Ip is set
by the Jc margin (Mangiarotti Ic x load factor), temperature dependent. Then:

    total_tapes/coil = AT_coil / Ip(T)         (fixed for given T)
    N_turns(Npw)     = total_tapes / Npw       (turns x parallel-tapes = const)

Cross-check vs ARC published: 12 mm tape, Ip~660 A @20K, total REBCO ~5730 km.
"""
import numpy as np

# ---- Mangiarotti Ic temperature ratios (pure-perp), from CP1 ----
BFLOOR = 1e-6; TLOW = 4.2; DT = 22.0 - 4.2
def jc_perp_42(b): b = max(b, BFLOOR); return 3268.2 * b ** (-0.6442)
def f_perp(b): b = max(b, BFLOOR); return 0.6781 - 0.0107 * b
def jc_perp_T(T, b): return jc_perp_42(b) * np.exp(-(T - TLOW) * np.log(1 / f_perp(b)) / DT)

AT_coil = 120 * 70e3          # 8.40 MA per coil (fixed, sets B0=9.2 T)
N_TF = 18
turn_perimeter = 25.98        # m, ARC mean coil circumference (PROCESS B27)

# ARC 20 K reference per-tape current (70 kA / 106 tapes)
Ip_20K_ref = 70e3 / 106.0     # ~660 A
# operating perpendicular field for the Jc margin (tape ~parallel to B, small B_perp)
Bperp_op = 3.0                # representative B_perp [T] at operating layer (refine w/ FEM)

temps = [4.2, 10.0, 20.0]
# Ip(T) scales with Ic(T) at operating field (more current capacity when colder)
Ip_T = {T: Ip_20K_ref * jc_perp_T(T, Bperp_op) / jc_perp_T(20.0, Bperp_op) for T in temps}

print("=== ARC per-tape current & total tape count by temperature ===")
print(f"{'T[K]':>5} {'Ip[A]':>7} {'total_tapes/coil':>17} {'N_turns@Npw=106':>16}")
total_tapes = {}
for T in temps:
    tt = AT_coil / Ip_T[T]
    total_tapes[T] = tt
    print(f"{T:5.1f} {Ip_T[T]:7.0f} {tt:17.0f} {tt/106:16.1f}")

# cross-check total REBCO tape length vs ARC published 5730 km
tt20 = total_tapes[20.0]
L_total_km = tt20 * turn_perimeter * N_TF / 1e3
print(f"\ncross-check total REBCO tape length @20K = {L_total_km:.0f} km  (ARC paper 5730 km)")

print("\n=== allocation table: scan Npw, N_turns = total_tapes/Npw (20 K) ===")
print(f"{'Npw':>5} {'N_turns':>9} {'Iop/turn[kA]':>13} {'tape_len/coil[km]':>18}")
for Npw in [1, 5, 10, 20, 50, 106, 150, 200]:
    Nturns = total_tapes[20.0] / Npw
    Iop_turn = Npw * Ip_T[20.0] / 1e3
    L_coil = total_tapes[20.0] * turn_perimeter / 1e3
    print(f"{Npw:5d} {Nturns:9.0f} {Iop_turn:13.1f} {L_coil:18.1f}")

print("\nNote: total_tapes (hence tape length & cost) fixed by AT/Ip; only the")
print("Npw<->N_turns split changes. This is the 'ampere-turn product fixed' constraint.")
