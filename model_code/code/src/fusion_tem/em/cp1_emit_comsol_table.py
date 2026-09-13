"""CP1 final: emit the Ic(B_perp, T) 2D table for COMSOL (Candidate B —
validated IcB20@20K anchor x Mangiarotti exp-temp T-ratio), incl. 10 K.

Outputs:
  data/processed/Ic_BperpT_mangiarotti.csv  (long table: T, Bperp, Ic_10mm_A)
  data/processed/Ic_BperpT_comsol_grid.txt  (COMSOL grid Interpolation: spreadsheet form)
  prints the COMSOL Analytic function expression.
Convention: Ic [A] for a 10 mm-wide tape vs perpendicular field Br [T] (model scales by wid/10mm).
"""
import numpy as np
from pathlib import Path

BFLOOR = 1e-6; TLOW = 4.2; DT = 22.0 - 4.2
def jc_perp_42(b): b = max(b, BFLOOR); return 3268.2 * b ** (-0.6442)
def f_perp(b): b = max(b, BFLOOR); return 0.6781 - 0.0107 * b
def jc_perp_T(T, b):
    tsp = DT / np.log(1 / f_perp(b))
    return jc_perp_42(b) * np.exp(-(T - TLOW) / tsp)

# validated 20 K anchor (existing IcB20, Ic[A] for 10 mm tape)
B20 = [0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,8,9,10]
I20 = [2540,1781,1376,1187,1037,945,872,806,751,711,664,631,609,577,562,549,538,529]
def ic20(b): return float(np.interp(b, B20, I20))

def ic_BT(b, T):
    """Ic [A, 10mm] = IcB20(b) * Jc_perp_mang(T,b)/Jc_perp_mang(20,b)."""
    return ic20(b) * jc_perp_T(T, b) / jc_perp_T(20.0, b)

OUT = Path("data/processed"); OUT.mkdir(parents=True, exist_ok=True)
temps = [4.2, 10.0, 20.0]
bgrid = np.round(np.arange(0.0, 10.01, 0.5), 2)

# long table
rows = ["T_K,Bperp_T,Ic_10mm_A"]
for T in temps:
    for b in bgrid:
        rows.append(f"{T},{b},{ic_BT(b,T):.2f}")
(OUT / "Ic_BperpT_mangiarotti.csv").write_text("\n".join(rows), encoding="utf-8")

# COMSOL grid Interpolation (spreadsheet): first row = T columns, first col = Bperp
grid = ["% Ic(Bperp[T], T[K]) for 10mm tape; rows=Bperp, cols=T",
        "Bperp\\T," + ",".join(f"{T}" for T in temps)]
for b in bgrid:
    grid.append(f"{b}," + ",".join(f"{ic_BT(b,T):.2f}" for T in temps))
(OUT / "Ic_BperpT_comsol_grid.txt").write_text("\n".join(grid), encoding="utf-8")

print("=== Ic(Bperp,T) [A, 10mm tape] — written to data/processed/ ===")
print(f"{'Bperp':>6} {'4.2K':>8} {'10K':>8} {'20K':>8}")
for b in [0,1,2,3,5,7,10]:
    print(f"{b:6.1f} {ic_BT(b,4.2):8.0f} {ic_BT(b,10):8.0f} {ic_BT(b,20):8.0f}")

print("\n=== COMSOL Analytic function (closed-form, drop-in) ===")
print("Define Jc_perp_mang(T,b) and use Ic(Br,T):")
print("  f_perp(b)   = 0.6781 - 0.0107*b")
print("  Jcp(T,b)    = 3268.2*b^(-0.6442) * exp(-(T-4.2)*log(1/f_perp(b))/17.8)")
print("  IcB20a(b)   = <piecewise from existing IcB20 table>")
print("  Ic(Br,T)    = IcB20a(abs(Br)) * Jcp(T,abs(Br))/Jcp(20,abs(Br))  [A, 10mm]")
print("  -> model:   Ic_used = Ic(Br,T)*wid/10[mm]*equ_turn ; Jc = Ic_used/(wid*thHTS)")
