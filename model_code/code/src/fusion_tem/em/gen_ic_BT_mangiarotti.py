"""Generate Ic(B_perp, T) interpolation tables for the COMSOL T-A model from the
Mangiarotti exp-temp Jc(B,theta,T) model — formulas copied verbatim from PROCESS
process/models/superconductors.py (jc_mangiarotti_exp_temp_layer), valid 4.2-30 K.

Output: Jc_layer in A/mm^2. Convert to per-tape Ic [A] by multiplying by the REBCO
layer cross-section of ONE tape: A_rebco = tape_width * rebco_layer_thickness.
"""
import numpy as np

BFLOOR = 1e-6
TLOW = 4.2
TREF = 22.0
DT = TREF - TLOW  # 17.8 K


def jc_perp_42(b):
    b = max(b, BFLOOR)
    return 3268.2 * b ** (-0.6442)


def jc_para_42(b):
    b = max(b, BFLOOR)
    return max(4086.0 - 71.859 * b, 1e-6)


def f_perp(b):
    b = max(b, BFLOOR)
    return 0.6781 - 0.0107 * b


def f_para(b):
    b = max(b, BFLOOR)
    return 0.7808 - 0.0105 * b


def theta0(b):
    b = max(b, BFLOOR)
    return max(11.99 - 0.069 * b, 1e-6)


def theta_deg(bpar, bperp):
    return float(np.arctan2(abs(bpar), abs(bperp)) * 180.0 / np.pi)


def jc_layer(T, bpar, bperp):
    """Layer Jc [A/mm^2] at temperature T, parallel field bpar, perpendicular field bperp."""
    b = float(np.hypot(bpar, bperp))
    b = max(b, BFLOOR)
    th = theta_deg(bpar, bperp)
    fp = f_perp(b)
    fa = f_para(b)
    t0 = theta0(b)
    tsp = DT / np.log(1.0 / fp)
    tsa = DT / np.log(1.0 / fa)
    jcp = jc_perp_42(b) * np.exp(-(T - TLOW) / tsp)
    jca = jc_para_42(b) * np.exp(-(T - TLOW) / tsa)
    return jcp + (jca - jcp) * np.exp(-(90.0 - th) / t0)


if __name__ == "__main__":
    # anchor check: 4.2 K must reproduce the published anchors exactly
    print("anchor check  perp 4.2K @5T: %.1f (anchor %.1f)" % (jc_layer(4.2, 0, 5), jc_perp_42(5)))
    print("anchor check  para 4.2K @5T: %.1f (anchor %.1f)" % (jc_layer(4.2, 5, 0), jc_para_42(5)))
    print()

    temps = [4.2, 10.0, 20.0]
    bperp_grid = [0.5, 1, 2, 3, 5, 7, 10, 15, 20, 23]

    print("=== Jc_layer [A/mm^2], pure-perpendicular (Bpar=0) ===")
    print("%6s %9s %9s %9s" % ("Bperp", "4.2K", "10K", "20K"))
    for bp in bperp_grid:
        print("%6.1f %9.1f %9.1f %9.1f" % (bp, jc_layer(4.2, 0, bp), jc_layer(10, 0, bp), jc_layer(20, 0, bp)))

    print()
    print("=== operating: Bpar=20 T (parallel), scan small Bperp (anisotropy) ===")
    print("%6s %9s %9s %9s" % ("Bperp", "4.2K", "10K", "20K"))
    for bp in [0.1, 0.5, 1, 2, 3]:
        print("%6.1f %9.1f %9.1f %9.1f" % (bp, jc_layer(4.2, 20, bp), jc_layer(10, 20, bp), jc_layer(20, 20, bp)))
