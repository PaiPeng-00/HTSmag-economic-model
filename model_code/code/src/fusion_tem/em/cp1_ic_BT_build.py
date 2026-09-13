"""CP1: build Ic(B_perp, T) for the COMSOL T-A model, and cross-check the
Mangiarotti temperature dependence against the existing validated discrete tables.

Constitutive law in the T-A model (verified):
    Ic   = IcB20(abs(Br)) * wid/10mm * equ_turn     # Ic for a 10 mm reference width
    Jc   = Ic / (wid * thHTS)                        # A/m^2 into the E-J power law
So the tables are "Ic [A] for a 10 mm-wide tape" vs perpendicular field Br.
"""
import numpy as np

# ---- Mangiarotti exp-temp layer Jc [A/mm^2] (verbatim from PROCESS) ----
BFLOOR = 1e-6; TLOW = 4.2; DT = 22.0 - 4.2
def jc_perp_42(b): b = max(b, BFLOOR); return 3268.2 * b ** (-0.6442)
def jc_para_42(b): b = max(b, BFLOOR); return max(4086.0 - 71.859 * b, 1e-6)
def f_perp(b): b = max(b, BFLOOR); return 0.6781 - 0.0107 * b
def f_para(b): b = max(b, BFLOOR); return 0.7808 - 0.0105 * b
def theta0(b): b = max(b, BFLOOR); return max(11.99 - 0.069 * b, 1e-6)
def jc_layer(T, bpar, bperp):
    b = max(float(np.hypot(bpar, bperp)), BFLOOR)
    th = float(np.arctan2(abs(bpar), abs(bperp)) * 180 / np.pi)
    tsp = DT / np.log(1 / f_perp(b)); tsa = DT / np.log(1 / f_para(b))
    jcp = jc_perp_42(b) * np.exp(-(T - TLOW) / tsp)
    jca = jc_para_42(b) * np.exp(-(T - TLOW) / tsa)
    return jcp + (jca - jcp) * np.exp(-(90 - th) / theta0(b))

# ---- existing validated discrete tables [Ic in A, 10 mm tape] vs B_perp [T] ----
B_disc = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13])
IcB4  = np.array([4434,3295,2601,2195,1912,1702,1540,1403,1293,1192,1100,1040,975,916])
IcB20_full = {0:2540,0.5:1781,1:1376,1.5:1187,2:1037,2.5:945,3:872,3.5:806,4:751,
              4.5:711,5:664,5.5:631,6:609,6.5:577,7:562,8:549,9:538,10:529}
IcB20 = np.array([2540,1376,1037,872,751,664,609,562,549,538,529,np.nan,np.nan,np.nan])
IcB30 = np.array([2264,1273,974,803,686,596,523,462,410,363,324,289,258,230])

print("=== Cross-check: temperature ratio Ic(T)/Ic(20K) at fixed B_perp ===")
print("        Discrete tables          Mangiarotti (pure-perp)")
print(" B[T]   4.2/20   30/20      4.2/20   10/20   30/20")
for i,b in enumerate(B_disc):
    if b>10: continue
    disc_42 = IcB4[i]/IcB20[i] if not np.isnan(IcB20[i]) else np.nan
    disc_30 = IcB30[i]/IcB20[i] if not np.isnan(IcB20[i]) else np.nan
    m20=jc_layer(20,0,b); m_42=jc_layer(4.2,0,b)/m20; m_10=jc_layer(10,0,b)/m20; m_30=jc_layer(30,0,b)/m20
    print(f"{b:5.0f}   {disc_42:6.3f}  {disc_30:6.3f}     {m_42:6.3f} {m_10:6.3f} {m_30:6.3f}")

print()
print("=== Candidate A: discrete-table direct 2D interp (gives 10K by T-interp 4.2<->20) ===")
def ic_disc(b, T):
    # piecewise-linear in T across {4.2,20,30} discrete tables, linear in B
    def at(tab, b): return float(np.interp(b, B_disc, tab))
    if T<=4.2: return at(IcB4,b)
    if T<=20: w=(T-4.2)/(20-4.2);
    if T<=20:
        return (1-w)*at(IcB4,b)+w*at(np.nan_to_num(IcB20,nan=529),b)
    w=(T-20)/(30-20); return (1-w)*at(np.nan_to_num(IcB20,nan=529),b)+w*at(IcB30,b)
print(" B[T]   Ic@4.2  Ic@10   Ic@20   (A, 10mm) [discrete-interp]")
for b in [0,1,2,3,5,7,10]:
    print(f"{b:5.0f}  {ic_disc(b,4.2):7.0f} {ic_disc(b,10):7.0f} {ic_disc(b,20):7.0f}")

print()
print("=== Candidate B: anchor IcB20(validated) x Mangiarotti T-ratio ===")
def ic_anchored(b, T):
    ic20 = float(np.interp(b, [0,1,2,3,4,5,6,7,8,9,10],
                  [2540,1376,1037,872,751,664,609,562,549,538,529]))
    ratio = jc_layer(T,0,b)/jc_layer(20,0,b)
    return ic20*ratio
print(" B[T]   Ic@4.2  Ic@10   Ic@20   (A, 10mm) [Mangiarotti-scaled]")
for b in [0,1,2,3,5,7,10]:
    print(f"{b:5.0f}  {ic_anchored(b,4.2):7.0f} {ic_anchored(b,10):7.0f} {ic_anchored(b,20):7.0f}")
