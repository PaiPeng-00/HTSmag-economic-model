#!/usr/bin/env python3
"""V6.2-H: operating-temperature crossover map on the frozen S2/He model."""
from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BASE = Path(os.environ["V6_RUN_BASE"]) if os.environ.get("V6_RUN_BASE") else ROOT / "v6_2_arc_actual_inductance"
PQ_ROOT = BASE / "stage_B1_parquet_v1" / "v6_2_b1_strict_feasible_parquet"
PQ_AUDIT = BASE / "stage_B1_parquet_v1" / "v6_2_b1_strict_parquet_audit.json"
G_AUDIT = BASE / "stage_G" / "v6_2_g_audit.json"
CONTRACT = BASE / "stage_H" / "v6_2_h_execution_contract.json"
STAGE = BASE / "stage_H"
PREF = STAGE / "v6_2_h_preferred_temperature_map.parquet"
BREAK_R = STAGE / "v6_2_h_pairwise_break_even_Rj.csv"
BREAK_P = STAGE / "v6_2_h_pairwise_break_even_price.csv"
NEAR = STAGE / "v6_2_h_temperature_nearoptimality.csv"
AUDIT = STAGE / "v6_2_h_audit.json"

DEVICE = "arc_16pancake_nuc600_v6_2"
MATRIX = "97242795c8f75319c8f5ea726a1fedd86999e8a66e53ddeb1255e6fb0c659ae5"
TEMPS = (4.2, 10.0, 20.0)
PRICES = np.arange(10.0, 101.0, 1.0)
N_VALUES = tuple(range(1, 201))
R_VALUES = np.geomspace(1.0, 100.0, 121)
REP_N = (20, 50, 100, 200)
REP_R = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0)
M_HTS = 2.8436625


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def audit_path(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def load_strict() -> pd.DataFrame:
    parts = []
    for path in sorted(PQ_ROOT.glob("scenario=S2/Top_K=*/coolant=He/*.parquet")):
        top = float(path.parents[1].name.split("=", 1)[1])
        if top not in TEMPS:
            continue
        d = pd.read_parquet(path, columns=["Npw", "rho_turn_uOhm_cm2", "R_joint_nOhm", "E_net_year_MWh", "HTS_requirement_kA_m", "C_plant_anchor_USD", "annual_noncapital_cost_USD", "Aplant", "CF_gross", "r_cryo_re_fraction", "anchor_economic_valid", "TF_system_matrix_sha256"])
        d["Top_K"], d["scenario"], d["coolant"] = top, "S2", "He"
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    if d["TF_system_matrix_sha256"].astype(str).ne(MATRIX).any():
        raise AssertionError("H input matrix mismatch")
    d = d.loc[(d.Aplant >= 0.80) & (d.r_cryo_re_fraction <= 0.50) & (d.E_net_year_MWh > 0) & d.anchor_economic_valid.astype(bool)].copy()
    return d


def main() -> None:
    if any(p.exists() for p in (PREF, BREAK_R, BREAK_P, NEAR, AUDIT)):
        raise FileExistsError("refusing existing V6.2-H artifacts")
    for p in (PQ_AUDIT, G_AUDIT, CONTRACT):
        if not p.exists():
            raise FileNotFoundError(p)
    pa = json.loads(PQ_AUDIT.read_text(encoding="utf-8")); ga = json.loads(G_AUDIT.read_text(encoding="utf-8")); hc = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if pa.get("status") != "PASS" or pa.get("device") != DEVICE or pa.get("matrix_sha256") != MATRIX:
        raise AssertionError("H Parquet gate failed")
    if ga.get("status") != "PASS":
        raise SystemExit("BLOCKED: V6.2-G is not PASS")
    if hc.get("status") != "BLOCKED_UPSTREAM_V6_2_G_NOT_RUN":
        raise AssertionError("unexpected H contract state")

    import importlib.util
    anchor_path = Path(__file__).with_name("8.5_recompute_v2_anchor_economics.py")
    spec = importlib.util.spec_from_file_location("v62_h_anchor", anchor_path); anchor = importlib.util.module_from_spec(spec); assert spec and spec.loader; spec.loader.exec_module(anchor)
    crf = float(anchor.capital_recovery_factor(anchor.DISCOUNT_RATE_BY_SCENARIO["S2"], anchor.PROJECT_YEARS_BY_SCENARIO["S2"]))
    d = load_strict()
    # The price perturbation is linear in HTS requirement and exact under the
    # frozen incremental-anchor definition.
    ref = d.loc[(d.Top_K == 10.0) & (d.Npw == 19) & np.isclose(d.R_joint_nOhm, 1.0) & np.isclose(d.rho_turn_uOhm_cm2, 10000.0)]
    if ref.empty:
        raise AssertionError("H frozen reference row missing")
    qref = float(ref.iloc[0].HTS_requirement_kA_m)
    d["a"] = (crf * (d.C_plant_anchor_USD - M_HTS * 50.0 * (d.HTS_requirement_kA_m - qref)) + d.annual_noncapital_cost_USD) / d.E_net_year_MWh
    d["b"] = crf * M_HTS * (d.HTS_requirement_kA_m - qref) / d.E_net_year_MWh
    # Compute the rho-optimized plant LCOE for each price, then preferred T.
    pref_parts = []
    for price in PRICES:
        x = d.assign(LCOE_USD_per_MWh=d.a + d.b * price)
        best = x.loc[x.groupby(["Top_K", "Npw", "R_joint_nOhm"], sort=False).LCOE_USD_per_MWh.idxmin(), ["Top_K", "Npw", "R_joint_nOhm", "rho_turn_uOhm_cm2", "LCOE_USD_per_MWh"]].copy()
        best.rename(columns={"rho_turn_uOhm_cm2": "best_rho_turn_uOhm_cm2"}, inplace=True)
        best["HTS_price_USD_per_kAm"] = price
        pref_parts.append(best)
    best_all = pd.concat(pref_parts, ignore_index=True)
    piv = best_all.pivot_table(index=["HTS_price_USD_per_kAm", "Npw", "R_joint_nOhm"], columns="Top_K", values="LCOE_USD_per_MWh", aggfunc="min").reset_index()
    piv.columns = [str(c) if not isinstance(c, float) else f"LCOE_{c:g}K_USD_per_MWh" for c in piv.columns]
    for t in TEMPS:
        col = f"LCOE_{t:g}K_USD_per_MWh"
        if col not in piv.columns:
            raise AssertionError(f"missing H temperature column {col}")
    vals = piv[[f"LCOE_{t:g}K_USD_per_MWh" for t in TEMPS]].to_numpy(float)
    piv["T_preferred_K"] = np.asarray(TEMPS)[np.nanargmin(vals, axis=1)]
    finite_min = np.nanmin(vals, axis=1)
    for pct in (1.0, 5.0):
        near = vals <= finite_min[:, None] * (1.0 + pct / 100.0)
        piv[f"nearoptimal_{int(pct)}pct"] = ["{" + ",".join(f"{t:g}" for t, ok in zip(TEMPS, row) if ok) + "}" for row in near]
    piv.to_parquet(PREF, index=False, compression="zstd")
    # Root-based pairwise Rj curves at representative N and prices.
    roots_r = []
    for n in REP_N:
        for price in PRICES:
            g = piv.loc[(piv.Npw == n) & (piv.HTS_price_USD_per_kAm == price)].sort_values("R_joint_nOhm")
            for t1, t2 in ((4.2, 10.0), (10.0, 20.0), (4.2, 20.0)):
                y1 = g[f"LCOE_{t1:g}K_USD_per_MWh"].to_numpy(float); y2 = g[f"LCOE_{t2:g}K_USD_per_MWh"].to_numpy(float); diff = y1 - y2
                for i in range(len(R_VALUES) - 1):
                    if np.isfinite(diff[i]) and np.isfinite(diff[i + 1]) and diff[i] == 0:
                        root = float(R_VALUES[i]); resid = 0.0
                    elif np.isfinite(diff[i]) and np.isfinite(diff[i + 1]) and diff[i] * diff[i + 1] < 0:
                        lo, hi = float(R_VALUES[i]), float(R_VALUES[i + 1])
                        for _ in range(40):
                            mid = math.sqrt(lo * hi)
                            j = i if abs(mid - R_VALUES[i]) < abs(R_VALUES[i + 1] - mid) else i + 1
                            # Linear interpolation in log-Rj is used only for root localization in this stored grid.
                            frac = (math.log(mid) - math.log(lo)) / (math.log(hi) - math.log(lo))
                            dm = diff[i] + frac * (diff[i + 1] - diff[i])
                            if diff[i] * dm <= 0: hi = mid
                            else: lo = mid
                        root = math.sqrt(lo * hi); resid = float(abs(dm) / max(min(abs(y1[i]), abs(y2[i])), 1e-30))
                        roots_r.append({"Npw": n, "HTS_price_USD_per_kAm": price, "T1_K": t1, "T2_K": t2, "Rj_break_even_nOhm": root, "root_residual": resid, "censored": False})
                        break
                else:
                    roots_r.append({"Npw": n, "HTS_price_USD_per_kAm": price, "T1_K": t1, "T2_K": t2, "Rj_break_even_nOhm": math.nan, "root_residual": math.nan, "censored": True})
    pd.DataFrame(roots_r).to_csv(BREAK_R, index=False, float_format="%.17g")
    # Price roots at fixed representative joint resistance.
    roots_p = []
    for n in REP_N:
        for rj in REP_R:
            g = piv.loc[(piv.Npw == n) & np.isclose(piv.R_joint_nOhm, rj)].sort_values("HTS_price_USD_per_kAm")
            for t1, t2 in ((4.2, 10.0), (10.0, 20.0), (4.2, 20.0)):
                y1 = g[f"LCOE_{t1:g}K_USD_per_MWh"].to_numpy(float); y2 = g[f"LCOE_{t2:g}K_USD_per_MWh"].to_numpy(float); diff = y1 - y2
                signs = np.flatnonzero(np.isfinite(diff[:-1]) & np.isfinite(diff[1:]) & (diff[:-1] * diff[1:] <= 0))
                if len(signs):
                    i = int(signs[0]); p = float(PRICES[i] + (PRICES[i + 1] - PRICES[i]) * (-diff[i]) / (diff[i + 1] - diff[i])) if diff[i + 1] != diff[i] else float(PRICES[i]); roots_p.append({"Npw": n, "Rj_nOhm": rj, "T1_K": t1, "T2_K": t2, "HTS_price_break_even_USD_per_kAm": p, "censored": False})
                else:
                    roots_p.append({"Npw": n, "Rj_nOhm": rj, "T1_K": t1, "T2_K": t2, "HTS_price_break_even_USD_per_kAm": math.nan, "censored": True})
    pd.DataFrame(roots_p).to_csv(BREAK_P, index=False, float_format="%.17g")
    near_rows = []
    for pct in (1.0, 5.0):
        col = f"nearoptimal_{int(pct)}pct"
        for t in TEMPS:
            token = f"{t:g}"
            near_rows.append({"temperature_K": t, "penalty_pct": pct, "fraction_of_map": float(piv[col].astype(str).str.contains(token, regex=False).mean()), "N_rows": len(piv)})
    pd.DataFrame(near_rows).to_csv(NEAR, index=False, float_format="%.17g")
    audit = {"experiment_id": "V6.2-H-operating-temperature-crossover", "status": "PASS", "timestamp_utc": datetime.now(timezone.utc).isoformat(), "device": DEVICE, "matrix_sha256": MATRIX, "scope": {"scenario": "S2", "coolant": "He", "Npw": "1..200", "Rj_grid": "121-point log [1,100]", "HTS_price_grid": "10..100 constant 2025 USD kA^-1 m^-1", "rho": "reoptimized over strict V6.2 B1 rows"}, "roots": {"Rj": "representative Npw and full price grid; log-grid localization", "price": "representative Npw/Rj; linear-price localization"}, "rows": {"preferred_map": len(piv), "Rj_roots": len(roots_r), "price_roots": len(roots_p)}, "upstream": {"G_audit_sha256": sha(G_AUDIT), "parquet_audit_sha256": sha(PQ_AUDIT)}, "outputs": {p.name: {"path": audit_path(p), "sha256": sha(p)} for p in (PREF, BREAK_R, BREAK_P, NEAR)}}
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
