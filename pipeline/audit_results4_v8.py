"""Offline Figure 5 inputs and independent temperature/price audits."""
from pathlib import Path
import argparse,hashlib,json,ast
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SCI=ROOT/'scientific_results_v7_splice_equivalent_20260906'
SCENARIOS=['S1','S2','S3']
BASE=np.array([1122.1472612313785,321.97916120106885,78.82074592882216])
KEY=['Npw','rho_turn_uOhm_cm2','R_joint_nOhm']
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=SCI/'experiments/results4_temperature_price_v8');a=p.parse_args()
    if a.output.exists(): raise SystemExit('STOP_OUTPUT_ALREADY_EXISTS')
    sources=[]; fixed=[]; minima=[]; all_keys=[]; boundaries=[]
    source_dir=SCI/'experiments/full_realization_robustness_matrix_v7/realization_ledgers'
    manifest=json.loads((ROOT/'public_release/postprocessing_v8_20260911/MANIFEST.json').read_text(encoding='utf-8'))
    hashes={Path(x['path']).name:x['sha256'] for x in manifest['files'] if x['path'].endswith('.parquet')}
    for path in sorted(source_dir.glob('*.parquet')):
        digest=sha(path); assert digest==hashes[path.name]
        cols=KEY+['Top_K','coolant']+[f'{stem}_{s}' for stem in ['Aplant','E_net_year_MWh','lcoe_anchor_USD_per_MWh'] for s in SCENARIOS]
        d=pd.read_parquet(path,columns=cols); assert len(d)==200*61*121 and not d.duplicated(KEY).any()
        avail=d[[f'Aplant_{s}' for s in SCENARIOS]].ge(.8).all(axis=1)
        good=np.isfinite(d[[f'lcoe_anchor_USD_per_MWh_{s}' for s in SCENARIOS]]).all(axis=1)
        good &= d[[f'lcoe_anchor_USD_per_MWh_{s}' for s in SCENARIOS]].gt(0).all(axis=1)
        good &= d[[f'E_net_year_MWh_{s}' for s in SCENARIOS]].gt(0).all(axis=1)
        top=float(d.Top_K.iloc[0]);coolant=d.coolant.iloc[0]
        idx=d.loc[avail&good,'lcoe_anchor_USD_per_MWh_S2'].idxmin();best=d.loc[idx]
        minima.append({'Top_K':top,'coolant':coolant,'S2_minimum_USD_MWh':float(best.lcoe_anchor_USD_per_MWh_S2),
                       **{k:float(best[k]) for k in KEY},'N_common_avail':int(avail.sum())})
        if coolant=='He':
            eligible=d.loc[d.Npw.eq(200)&d.R_joint_nOhm.eq(1),KEY].copy()
            eligible['Top_K']=top;eligible['common_avail']=avail.loc[eligible.index];eligible['all_economically_valid']=good.loc[eligible.index]
            assert len(eligible)==61;all_keys.append(eligible)
            f=d.loc[d.Npw.eq(200)&d.rho_turn_uOhm_cm2.eq(10000)].sort_values('R_joint_nOhm').copy()
            assert len(f)==121
            delta=f[[f'lcoe_anchor_USD_per_MWh_{s}' for s in SCENARIOS]].to_numpy()/BASE-1
            f['delta_max_pct']=delta.max(axis=1)*100
            valid=avail.loc[f.index].to_numpy()&good.loc[f.index].to_numpy()
            passed=valid&(delta<=.1).all(axis=1)
            first_fail=int(np.flatnonzero(~passed)[0]) if not passed.all() else len(f)
            tol=float(f.R_joint_nOhm.iloc[first_fail-1]) if first_fail else None
            nonmonotonic=bool(np.any(passed[first_fail:]))
            controls=[]
            if first_fail<len(f):
                controls=[s for i,s in enumerate(SCENARIOS) if delta[first_fail,i]>.1]
            boundaries.append({'Top_K':top,'Npw':200,'rho_turn_uOhm_cm2':10000,'coolant':'He',
                  'last_contiguous_pass_nOhm':tol,'first_fail_nOhm':None if passed.all() else float(f.R_joint_nOhm.iloc[first_fail]),
                  'upper_censored':bool(passed.all()),'nonmonotonic':nonmonotonic,'first_failure_scenarios':controls,
                  'first_failure_availability':bool(not avail.loc[f.index[first_fail]]) if first_fail<len(f) else None,
                  'first_failure_validity':bool(not good.loc[f.index[first_fail]]) if first_fail<len(f) else None})
            f['pass_10pct']=passed
            for i,s in enumerate(SCENARIOS): f[f'delta_{s}_pct']=delta[:,i]*100
            fixed.append(f)
        sources.append({'path':str(path.relative_to(ROOT)),'sha256':digest})
    raw=SCI/'stage_F2/v6_2_f2_anchor_lookup.parquet'
    f2audit=json.loads((SCI/'stage_F2/v6_2_f2_audit.json').read_text(encoding='utf-8'));assert f2audit['status']=='PASS'
    crf=f2audit['frozen_reference']['CRF']
    code=ROOT/'model_code/code/scripts/8_economic/8.53_run_v6_2_f2_price.py'
    tree=ast.parse(code.read_text(encoding='utf-8-sig'))
    mhts=next(float(ast.literal_eval(n.value)) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='M_HTS' for t in n.targets))
    anchor=pd.read_parquet(raw,filters=[('Npw','==',200),('R_joint_nOhm','==',1)])
    assert len(anchor)==183
    anchor=anchor.merge(pd.concat(all_keys),on=KEY+['Top_K'],validate='one_to_one')
    assert anchor.common_avail.all() and anchor.all_economically_valid.all()
    raw_price=SCI/'figure_panel_data/main/Fig4C_HTS_price_tolerance_v6_7.csv'
    prior=pd.read_csv(raw_price);prices=[]
    for price in [10.,50.,100.]:
        tmp=anchor.copy()
        tmp['lcoe_price']=(crf*(tmp.C_plant_anchor_USD+mhts*(price-50)*tmp.HTS_requirement_kA_m)+tmp.annual_noncapital_cost_USD)/tmp.E_net_year_MWh
        best=tmp.loc[tmp.groupby('Top_K').lcoe_price.idxmin()].sort_values('Top_K')
        ref=float(best.lcoe_price.min());preferred=float(best.loc[best.lcoe_price.idxmin(),'Top_K'])
        for _,r in best.iterrows():
            prev=prior.loc[prior.HTS_price_USD_per_kAm.eq(price)&prior.Top_K.eq(r.Top_K)].iloc[0]
            penalty=(r.lcoe_price/ref-1)*100
            assert np.isclose(r.lcoe_price,prev.lcoe_price,rtol=0,atol=1e-9)
            assert np.isclose(penalty,prev.lcoe_increase_above_minimum_pct,rtol=0,atol=1e-9)
            assert preferred==prev.preferred_temperature_K
            prices.append({'HTS_price_USD_per_kAm':price,'Top_K':r.Top_K,'lcoe_price':r.lcoe_price,
                           'lcoe_premium_pct':penalty,'preferred_temperature_K':preferred,
                           'selected_rho_turn_uOhm_cm2':r.rho_turn_uOhm_cm2})
    requirements=anchor.groupby('Top_K').HTS_requirement_kA_m.agg(['min','max'])
    assert np.allclose(requirements['min'],requirements['max'],rtol=1e-10)
    conductor_increase=float((requirements.loc[20,'min']/requirements.loc[4.2,'min']-1)*100)
    b42=next(x for x in boundaries if x['Top_K']==4.2);b20=next(x for x in boundaries if x['Top_K']==20)
    assert [b42['last_contiguous_pass_nOhm'],b20['last_contiguous_pass_nOhm']]==[3.548,36.869]
    result={'status':'PASS','physical_model_rerun':False,'main_comparison':'He, Npw=200, rho_turn=10000; fixed across S1-S3',
         'baselines':BASE.tolist(),'boundaries':boundaries,'ratio_20_to_4p2':b20['last_contiguous_pass_nOhm']/b42['last_contiguous_pass_nOhm'],
         'S2_temperature_minima':minima,'REBCO_increase_20_vs_4p2_pct':conductor_increase,
         'price_sensitivity':{'status':'PASS','all_183_candidates_common_availability_and_validity':True,
           'original_9_price_records_unchanged':True,'rho_reoptimized':True,
           'normalization':'At each price, relative to minimum among three temperatures; independent sensitivity, not robustness criterion',
           'CRF':crf,'M_HTS':mhts,'rows':prices},
         'sources':sources+[{'path':str(raw.relative_to(ROOT)),'sha256':sha(raw)},{'path':str(raw_price.relative_to(ROOT)),'sha256':sha(raw_price)},{'path':str(code.relative_to(ROOT)),'sha256':sha(code)}]}
    a.output.mkdir(parents=True)
    pd.concat(fixed).to_csv(a.output/'Fig5A_fixed_high_current_curves.csv',index=False)
    pd.DataFrame(boundaries).to_csv(a.output/'Fig5A_contiguous_boundaries.csv',index=False)
    pd.DataFrame(prices).to_csv(a.output/'Fig5B_price_sensitivity.csv',index=False)
    pd.DataFrame(minima).to_csv(a.output/'S2_temperature_minima_common_availability.csv',index=False)
    (a.output/'RESULTS4_AUDIT.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['sources','price_sensitivity']},indent=2))
if __name__=='__main__': main()

