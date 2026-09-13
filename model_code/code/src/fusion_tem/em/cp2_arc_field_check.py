"""CP2: analytic ARC toroidal-field self-consistency check + FEM parameter set.

Validates that the ARC ampere-turn / geometry numbers reproduce B0=9.2 T and
Bmax=23 T BEFORE running the FEM, so the FEM run is a confirmation not a search.
"""
import numpy as np

mu0 = 4 * np.pi * 1e-7
# ARC published (Sorbom 2015, Table 1 + §4.2.1)
R0 = 3.3          # major radius [m]
a = 1.13          # minor radius [m]
B0_pub = 9.2      # on-axis toroidal field [T]
Bmax_pub = 23.0   # peak on-conductor field [T]
N_TF = 18
cables_per_coil = 120
I_cable = 70e3    # A

# ampere-turns per coil
AT_coil = cables_per_coil * I_cable
print(f"ampere-turns/coil = {cables_per_coil} x {I_cable/1e3:.0f} kA = {AT_coil/1e6:.2f} MA")

# B0 from ampere-turns: B0 = mu0 * N_TF * AT_coil / (2 pi R0)
B0_calc = mu0 * N_TF * AT_coil / (2 * np.pi * R0)
print(f"B0(from AT) = {B0_calc:.2f} T   (published {B0_pub} T)  -> {'OK' if abs(B0_calc-B0_pub)<0.3 else 'CHECK'}")

# inverse: AT needed for B0=9.2
AT_need = B0_pub * 2 * np.pi * R0 / (mu0 * N_TF)
print(f"AT/coil needed for 9.2 T = {AT_need/1e6:.2f} MA  (have {AT_coil/1e6:.2f} MA)")

# Bmax: inboard conductor radius from B ~ 1/r  ->  r_cond = R0 * B0/Bmax
r_cond = R0 * B0_pub / Bmax_pub
print(f"inboard conductor radius for Bmax=23 T: r = R0*B0/Bmax = {r_cond:.2f} m")
print(f"  => inboard radial gap R0 - r = {R0 - r_cond:.2f} m (plasma a={a} + blanket/VV/build)")

# FEM (mf_3D_Dshape_Bcen.m) parameter set for ARC
# coil1.N = Nt*Nc/2 ; current Iop ; need Nt*Nc/2 * Iop = AT_coil
# choose Nc (pancakes) and Iop to hit AT, with 12mm tape
print("\n=== FEM mf_3D_Dshape_Bcen.m ARC parameter set ===")
Nc = 24           # pancakes (choose); ARC WP is graded 120 cables -> represent as turns
Iop = 70e3        # per-turn (cable) current [A]
N_coil = AT_coil / Iop
Nt_fem = 2 * N_coil / Nc
print(f"  Ntf  = {N_TF}")
print(f"  Iop  = {Iop/1e3:.0f}e3   % per-cable current [A]")
print(f"  Nc   = {Nc}        % pancakes (coil1.N = Nt*Nc/2)")
print(f"  Nt   = {Nt_fem:.1f}      % turns/pancake so Nt*Nc/2*Iop = {AT_coil/1e6:.2f} MA")
print(f"  R10  = {r_cond:.2f}      % inner radius of TF inboard leg [m]")
print(f"  R1   = 0.30       % D inner-arc radius [m] (tune)")
print(f"  L1   = {2*a*1.84:.2f}       % straight section ~ 2*a*kappa [m] (tune)")
print(f"  wid  = 12[mm]     % ARC tape width")
print(f"  thk  = 0.1[mm]    % tape thickness (12mm x 0.1mm REBCO CICC)")
print("  target: central B -> 9.2 T, peak |B| on inboard conductor -> ~23 T")
