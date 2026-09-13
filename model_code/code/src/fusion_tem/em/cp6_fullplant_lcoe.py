"""CP6: full-plant LCOE = magnet-LCOE + C0*CRF/E_net.

Answers the Joule editor's question: does adding the fixed non-magnet capital C0
change the conclusion? -> It AMPLIFIES the penalty, because a worse magnet design
lowers net export E_net and the fixed C0 is spread over less electricity.

    dLCOE_full(i,*) = (C0*CRF)*(1/E_net(i) - 1/E_net(*)) + dLCOE_mag(i,*)

ARC plant: gross = thermal 708 MW x 0.40 = 283 MWe. C0 uses the S2 background-capital anchor normalized once to constant 2025 US$.
"""
import numpy as np

from fusion_tem.economic.price_basis import BACKGROUND_CAPITAL_USD_BY_SCENARIO

def crf(r, N): return r / (1.0 - (1.0 + r) ** (-N))

# ---- ARC plant (CP6 overrides) ----
P_thermal_MW = 708.0          # ARC total thermal power (incl. blanket multiplication)
eta = 0.40                    # ARC thermal-electric efficiency
gross_MWe = P_thermal_MW * eta
r_other = 0.30                # non-TF-cryogenic auxiliary scenario fraction
print(f"ARC gross = {P_thermal_MW} MWth x {eta} = {gross_MWe:.0f} MWe  (SPARC was 49 MWe)")

# ---- fixed non-magnet capital C0 (constant 2025 US$) ----
C0 = BACKGROUND_CAPITAL_USD_BY_SCENARIO["S2"]
r_disc = 0.08
scenarios = {'S1': 20, 'S2': 30, 'S3': 50}

# ---- demonstration: full-plant LCOE penalty of bad vs good magnet design ----
# availability per scenario (illustrative; real values come from CP4 charging + scan)
avail = {'S1': 0.38, 'S2': 0.60, 'S3': 0.85}
# magnet design corners: recirculating cryo fraction r_cryo
designs = {'good (20K, ~optimal)': 0.02, 'mid': 0.15, 'bad (4.2K, unfavorable)': 0.50}

print("\n=== full-plant LCOE C0-contribution by scenario & design ===")
for sc, N in scenarios.items():
    CRF = crf(r_disc, N)
    C0_annual = C0 * CRF
    A = avail[sc]
    gross_mwh = gross_MWe * 8760 * A
    print(f"\n-- {sc} (N={N}y, CRF={CRF:.4f}, avail={A}, gross={gross_mwh/1e6:.3f} TWh/y) --")
    print(f"   {'design':>26} {'r_cryo':>7} {'E_net[TWh]':>11} {'C0_LCOE[$/MWh]':>15}")
    ref_enet = gross_mwh * (1 - designs['good (20K, ~optimal)'] - r_other)
    for name, rc in designs.items():
        e_net = gross_mwh * (1 - rc - r_other)
        c0_lcoe = C0_annual / e_net if e_net > 0 else np.inf
        print(f"   {name:>26} {rc:7.2f} {e_net/1e6:11.3f} {c0_lcoe:15.1f}")
    # delta vs good design
    e_bad = gross_mwh * (1 - designs['bad (4.2K, unfavorable)'] - r_other)
    e_good = gross_mwh * (1 - designs['good (20K, ~optimal)'] - r_other)
    dlcoe_C0 = C0_annual * (1/e_bad - 1/e_good)
    print(f"   => C0-driven dLCOE(bad - good) = +{dlcoe_C0:.1f} $/MWh  (ON TOP of magnet dLCOE)")

# ---- C0 sensitivity (+/-50%) for S2 bad-vs-good ----
print("\n=== C0 sensitivity (S2, bad-vs-good) ===")
N = scenarios['S2']; CRF = crf(r_disc, N); A = avail['S2']
gross_mwh = gross_MWe*8760*A
e_bad = gross_mwh*(1-0.50-r_other); e_good = gross_mwh*(1-0.02-r_other)
for f in [0.5, 1.0, 1.5]:
    d = (C0*f)*CRF*(1/e_bad - 1/e_good)
    print(f"   C0={C0*f/1e9:.1f}B$ -> C0-driven dLCOE(bad-good) = +{d:.1f} $/MWh")

print("\nConclusion: any fixed non-magnet C0 only AMPLIFIES the LCOE penalty of a")
print("worse magnet design (lower E_net). The editor's concern strengthens the result.")
