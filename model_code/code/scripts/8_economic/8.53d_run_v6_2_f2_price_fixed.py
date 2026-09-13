#!/usr/bin/env python3
"""V6.2-F2 launcher using a compact frozen-anchor lookup for direct midpoints."""
from __future__ import annotations
import importlib.util, json, math
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('v62_f2_impl_d',HERE/'8.53_run_v6_2_f2_price.py'); impl=importlib.util.module_from_spec(spec); assert spec and spec.loader; spec.loader.exec_module(impl)
LOOKUP=impl.STAGE/'v6_2_f2_anchor_lookup.parquet'; _L=None

def build_lookup():
    if LOOKUP.exists(): return
    parts=[]
    for p in sorted(impl.PQ_ROOT.glob('scenario=S2/Top_K=*/coolant=He/*.parquet')):
        top=float(p.parents[1].name.split('=',1)[1])
        if top not in impl.TEMPERATURES: continue
        d=pd.read_parquet(p,columns=['Npw','rho_turn_uOhm_cm2','R_joint_nOhm','Aplant','CF_gross','HTS_requirement_kA_m','C_plant_anchor_USD','annual_noncapital_cost_USD','E_net_year_MWh'])
        d['Top_K']=top; parts.append(d.loc[np.isclose(d.R_joint_nOhm,1.0)])
    d=pd.concat(parts,ignore_index=True).drop_duplicates(['Top_K','Npw','rho_turn_uOhm_cm2'])
    d.to_parquet(LOOKUP,index=False,compression='zstd')

def direct_worker(job):
    global _L
    if impl._FAST is None: impl._FAST=impl.load_module('v62_f2_fast_d',impl.FAST)
    if _L is None: _L=pd.read_parquet(LOOKUP)
    job=dict(job); job['raw_root']=str(impl.RAW_DIR); direct=impl._FAST.worker(job); out=[]; qref=float(job['q_ref']); crf=float(job['crf'])
    for rec in direct:
        raw=impl.ROOT/rec['raw_path']; d=pd.read_csv(raw,low_memory=False); d['Top_K']=float(job['Top_K']); d['Npw']=int(job['Npw'])
        d=d.merge(_L[['Top_K','Npw','rho_turn_uOhm_cm2','HTS_requirement_kA_m','C_plant_anchor_USD','annual_noncapital_cost_USD']],on=['Top_K','Npw','rho_turn_uOhm_cm2'],how='left')
        aplant=pd.to_numeric(d['Aplant'],errors='coerce'); rc=pd.to_numeric(d['r_cryo_re_fraction'],errors='coerce'); net=pd.to_numeric(d['E_net_year_MWh'],errors='coerce')
        valid=(d['status'].astype(str).str.lower().eq('success')&np.isfinite(aplant)&(aplant>=.80)&np.isfinite(rc)&(rc<=.5)&np.isfinite(net)&(net>0)&np.isfinite(d['C_plant_anchor_USD'])&np.isfinite(d['HTS_requirement_kA_m'])&np.isfinite(d['annual_noncapital_cost_USD']))
        for rv,g in d.groupby('R_joint_nOhm',sort=True):
            g=g.loc[valid.loc[g.index]].copy()
            for price in impl.PRICES:
                if g.empty: out.append({'HTS_price_USD_per_kAm':price,'Top_K':float(job['Top_K']),'Npw':int(job['Npw']),'R_joint_nOhm':float(rv),'feasible':False,'lcoe':math.nan,'best_rho':math.nan}); continue
                g['lcoe_price']=impl.price_lcoe(g,price,qref,crf); g=g.loc[np.isfinite(g.lcoe_price)&(g.lcoe_price>0)]
                if g.empty: out.append({'HTS_price_USD_per_kAm':price,'Top_K':float(job['Top_K']),'Npw':int(job['Npw']),'R_joint_nOhm':float(rv),'feasible':False,'lcoe':math.nan,'best_rho':math.nan})
                else:
                    z=g.loc[g.lcoe_price.idxmin()]; out.append({'HTS_price_USD_per_kAm':price,'Top_K':float(job['Top_K']),'Npw':int(job['Npw']),'R_joint_nOhm':float(rv),'feasible':True,'lcoe':float(z.lcoe_price),'best_rho':float(z.rho_turn_uOhm_cm2)})
    return out

impl.direct_worker=direct_worker
if __name__=='__main__':
    impl.STAGE.mkdir(parents=True,exist_ok=True); build_lookup(); impl.main()
