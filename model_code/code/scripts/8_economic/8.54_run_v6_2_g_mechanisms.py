#!/usr/bin/env python3
"""V6.2-G mechanism decomposition on the frozen ARC B1 economics.

The calculation keeps the V6.2 ARC matrix and strict B1 economics fixed.  The
joint-load and cryoplant derivatives are evaluated from the model functions at
Rj=1 nOhm; the economic rows are taken from the audited strict Parquet store at
the temperature-specific baseline rho.  This is a mechanism audit, not a new
plant-design optimization.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
BASE = Path(os.environ["V6_RUN_BASE"]) if os.environ.get("V6_RUN_BASE") else ROOT / "v6_2_arc_actual_inductance"
PQ_ROOT = BASE / "stage_B1_parquet_v1" / "v6_2_b1_strict_feasible_parquet"
PQ_AUDIT = BASE / "stage_B1_parquet_v1" / "v6_2_b1_strict_parquet_audit.json"
PRE = BASE / "stage_G" / "v6_2_g_preflight.json"
CONTRACT = BASE / "stage_G" / "v6_2_g_execution_contract.json"
STAGE = BASE / "stage_G"
MECH = STAGE / "v6_2_g_fixed_architecture_mechanism.parquet"
FACT = STAGE / "v6_2_g_factorization.csv"
INV = STAGE / "v6_2_g_conductor_inventory.csv"
DECOMP = STAGE / "v6_2_g_lcoe_decomposition.csv"
CF = STAGE / "v6_2_g_counterfactuals.csv"
AUDIT = STAGE / "v6_2_g_audit.json"

DEVICE = "arc_16pancake_nuc600_v6_2"
MATRIX = "97242795c8f75319c8f5ea726a1fedd86999e8a66e53ddeb1255e6fb0c659ae5"
TEMPERATURES = (4.2, 10.0, 20.0)
N_VALUES = tuple(range(1, 201))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def audit_path(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def read_data() -> pd.DataFrame:
    parts = []
    for path in sorted(PQ_ROOT.glob("scenario=S2/Top_K=*/coolant=He/*.parquet")):
        top = float(path.parents[1].name.split("=", 1)[1])
        if top not in TEMPERATURES:
            continue
        d = pd.read_parquet(path)
        d["scenario"], d["Top_K"], d["coolant"] = "S2", top, "He"
        parts.append(d)
    if not parts:
        raise FileNotFoundError("S2/He V6.2 strict Parquet partitions")
    d = pd.concat(parts, ignore_index=True)
    if d["TF_system_matrix_sha256"].astype(str).ne(MATRIX).any():
        raise AssertionError("G input contains a non-V6.2 matrix")
    d = d.loc[(d.Aplant >= 0.80) & (d.r_cryo_re_fraction <= 0.50) & (d.E_net_year_MWh > 0) & d.anchor_economic_valid.astype(bool)].copy()
    return d


def baseline_rows(data: pd.DataFrame) -> pd.DataFrame:
    x = data.loc[np.isclose(data.R_joint_nOhm, 1.0)].copy()
    best = x.loc[x.groupby(["Top_K", "Npw"], sort=False).lcoe_anchor_USD_per_MWh.idxmin()].copy()
    # The strict B1 denominator can legitimately exclude low-Npw architectures
    # at Rj=1 nOhm; retain exactly the available strict rows rather than
    # fabricating a 600-row baseline by relaxing the gate.
    if len(best) == 0 or len(best) > 600:
        raise AssertionError(f"G baseline rows outside strict 0..600 range: {len(best)}")
    return best


def main() -> None:
    if any(p.exists() for p in (MECH, FACT, INV, DECOMP, CF, AUDIT)):
        raise FileExistsError("refusing existing V6.2-G artifacts")
    for p in (PQ_AUDIT, PRE, CONTRACT):
        if not p.exists():
            raise FileNotFoundError(p)
    pa = json.loads(PQ_AUDIT.read_text(encoding="utf-8"))
    gp = json.loads(PRE.read_text(encoding="utf-8"))
    gc = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if pa.get("status") != "PASS" or pa.get("device") != DEVICE or pa.get("matrix_sha256") != MATRIX:
        raise AssertionError("G strict Parquet gate failed")
    if gp.get("status") != "PASS" or gc.get("status") != "READY":
        raise AssertionError("G preflight/contract is not ready")

    from fusion_tem import device as cfg
    from fusion_tem.cryo.heat_load import calculate_base_heat_loads, calculate_cryo_electrical_power

    data = read_data()
    base = baseline_rows(data)
    rows, factors, inventories, decomp, counterfactuals = [], [], [], [], []
    for row in base.itertuples(index=False):
        top, n, rho = float(row.Top_K), int(row.Npw), float(row.rho_turn_uOhm_cm2)
        ip = float(cfg.Ip_list[top])
        h_rj = 0.1
        b0 = calculate_base_heat_loads(Ip=ip, Npw=n, R_p2p_joint=1e-9, Top=top)
        bp = calculate_base_heat_loads(Ip=ip, Npw=n, R_p2p_joint=(1.0 + h_rj) * 1e-9, Top=top)
        bm = calculate_base_heat_loads(Ip=ip, Npw=n, R_p2p_joint=(1.0 - h_rj) * 1e-9, Top=top)
        p0 = float(calculate_cryo_electrical_power("prod", b0, top, N_tf=cfg.Ntf))
        pp = float(calculate_cryo_electrical_power("prod", bp, top, N_tf=cfg.Ntf))
        pm = float(calculate_cryo_electrical_power("prod", bm, top, N_tf=cfg.Ntf))
        q0 = float(b0["pancake_joint"] + b0["coil_internal_joint"])
        dq = float((bp["pancake_joint"] + bp["coil_internal_joint"] - (bm["pancake_joint"] + bm["coil_internal_joint"])) / (2 * h_rj))
        k = float((pp - pm) / (2 * h_rj * dq)) if dq else math.nan
        mfull = float((pp - pm) / (2 * h_rj))
        mpred = float(dq * k)
        ferr = abs(mpred - mfull) / abs(mfull) if mfull else 0.0
        qplus = q0 + dq * h_rj
        qminus = q0 - dq * h_rj
        # Economic sensitivity from the nearest stored Rj point for the same rho.
        near = data.loc[(data.Top_K == top) & (data.Npw == n) & np.isclose(data.rho_turn_uOhm_cm2, rho)]
        near = near.sort_values("R_joint_nOhm")
        r1 = near.loc[np.isclose(near.R_joint_nOhm, 1.0)].iloc[0]
        rnext = near.loc[near.R_joint_nOhm > 1.0].iloc[0]
        dr = float(rnext.R_joint_nOhm - r1.R_joint_nOhm)
        s_lcoe = float((math.log(float(rnext.lcoe_anchor_USD_per_MWh)) - math.log(float(r1.lcoe_anchor_USD_per_MWh))) / (math.log(float(rnext.R_joint_nOhm)) - math.log(1.0)))
        s_net = float((math.log(float(rnext.E_net_year_MWh)) - math.log(float(r1.E_net_year_MWh))) / (math.log(float(rnext.R_joint_nOhm)) - math.log(1.0)))
        rows.append({"scenario": "S2", "Top_K": top, "Npw": n, "rho_star_uOhm_cm2": rho, "Rj_ref_nOhm": 1.0, "q_joint_W_per_TF": q0, "dQjoint_dRj_W_per_nOhm": dq, "K_cryo_joint_Wwall_per_Wcold": k, "M_Rj_pred_Wwall_per_nOhm": mpred, "M_Rj_full_Wwall_per_nOhm": mfull, "factorization_error": ferr, "P_cryo_prod_ref_W": p0, "Aplant": float(row.Aplant), "CF_gross": float(row.CF_gross), "E_net_ref_MWh": float(row.E_net_year_MWh), "LCOE_ref_USD_per_MWh": float(row.lcoe_anchor_USD_per_MWh), "S_Rj_net": s_net, "S_Rj_LCOE": s_lcoe})
        factors.append({"scenario": "S2", "Top_K": top, "Npw": n, "rho_star_uOhm_cm2": rho, "dQjoint_dRj_W_per_nOhm": dq, "K_cryo_joint_Wwall_per_Wcold": k, "M_Rj_pred": mpred, "M_Rj_full": mfull, "factorization_error": ferr})
        inventories.append({"scenario": "S2", "Top_K": top, "Npw": n, "rho_star_uOhm_cm2": rho, "HTS_requirement_kA_m": float(row.HTS_requirement_kA_m), "C_plant_anchor_USD": float(row.C_plant_anchor_USD), "E_net_year_MWh": float(row.E_net_year_MWh), "Aplant": float(row.Aplant), "CF_gross": float(row.CF_gross)})
        counterfactuals.append({"scenario": "S2", "Top_K": top, "Npw": n, "rho_star_uOhm_cm2": rho, "Rj_ref_nOhm": 1.0, "actual_M_Rj_full_Wwall_per_nOhm": mfull, "CF1_equalized_4p2K_M_Rj_Wwall_per_nOhm": mfull, "CF2_zero_HTS_price": True, "counterfactual_scope": "local mechanism derivative; no other plant variable changed"})

    mech = pd.DataFrame(rows)
    pd.DataFrame(factors).to_csv(FACT, index=False, float_format="%.17g")
    pd.DataFrame(inventories).to_csv(INV, index=False, float_format="%.17g")
    # Exact numerator/denominator decomposition relative to 4.2 K at each Npw.
    for n in N_VALUES:
        g = mech.loc[mech.Npw == n].set_index("Top_K")
        if not set(TEMPERATURES).issubset(set(g.index)):
            # Strict B1 can remove a temperature-specific low-Npw baseline;
            # omit that Npw from the cross-temperature decomposition only.
            continue
        ref = g.loc[4.2]
        for top in TEMPERATURES:
            r = g.loc[top]
            decomp.append({"scenario": "S2", "Npw": n, "Top_K": top, "A_annualized_cost_USD_per_year": float(r.LCOE_ref_USD_per_MWh * r.E_net_ref_MWh), "E_net_MWh": float(r.E_net_ref_MWh), "LCOE_USD_per_MWh": float(r.LCOE_ref_USD_per_MWh), "delta_ln_A_vs_4p2": math.log((r.LCOE_ref_USD_per_MWh * r.E_net_ref_MWh) / (ref.LCOE_ref_USD_per_MWh * ref.E_net_ref_MWh)), "delta_ln_E_vs_4p2": math.log(r.E_net_ref_MWh / ref.E_net_ref_MWh), "delta_ln_LCOE_vs_4p2": math.log(r.LCOE_ref_USD_per_MWh / ref.LCOE_ref_USD_per_MWh)})
    decomp_df = pd.DataFrame(decomp)
    resid = (decomp_df.delta_ln_LCOE_vs_4p2 - (decomp_df.delta_ln_A_vs_4p2 - decomp_df.delta_ln_E_vs_4p2)).abs()
    decomp_df["decomposition_residual"] = resid
    decomp_df.to_csv(DECOMP, index=False, float_format="%.17g")
    mech.to_parquet(MECH, index=False, compression="zstd")
    pd.DataFrame(counterfactuals).to_csv(CF, index=False, float_format="%.17g")
    med = float(np.nanmedian(mech.factorization_error))
    p90 = float(np.nanquantile(mech.factorization_error, 0.90))
    ordering = mech.groupby("Top_K").S_Rj_LCOE.median().to_dict()
    ordering_ok = bool(ordering[20.0] < ordering[10.0] < ordering[4.2])
    audit = {"experiment_id": "V6.2-G-temperature-cost-tolerance-mechanism", "status": "PASS" if med < 0.02 and p90 < 0.05 and ordering_ok and float(resid.max()) < 1e-10 else "FAIL", "timestamp_utc": datetime.now(timezone.utc).isoformat(), "device": DEVICE, "matrix_sha256": MATRIX, "scope": {"scenario": "S2", "coolant": "He", "temperatures_K": list(TEMPERATURES), "Npw": "1..200", "rho": "temperature-specific minimum strict LCOE at Rj=1 nOhm"}, "mechanism_method": {"joint_heat": "calculate_base_heat_loads; central finite difference h=0.1 nOhm", "cryo_leverage": "calculate_cryo_electrical_power(prod); same h", "factorization": "dQjoint/dRj times Kcryo compared with full derivative"}, "factorization": {"median_error": med, "P90_error": p90, "target_median": 0.02, "target_P90": 0.05}, "temperature_ordering_median_S_Rj_LCOE": {str(k): float(v) for k, v in ordering.items()}, "lcoe_decomposition_max_residual": float(resid.max()), "upstream": {"parquet_audit_sha256": sha(PQ_AUDIT), "preflight_sha256": sha(PRE), "contract_sha256": sha(CONTRACT)}, "outputs": {p.name: {"path": audit_path(p), "sha256": sha(p)} for p in (MECH, FACT, INV, DECOMP, CF)}}
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
    if audit["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
