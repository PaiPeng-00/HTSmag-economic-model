"""Read-only scientific reducer: identify the frozen minimum's exact parent set."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SCI=ROOT/'scientific_results_v7_splice_equivalent_20260906/experiments'
BASE=np.array([1122.1472612313785,321.97916120106885,78.82074592882216])
SCENARIOS=['S1','S2','S3']
KEY=['Npw','rho_turn_uOhm_cm2','R_joint_nOhm','Top_K','coolant']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=SCI/'methods_populations_v8');a=p.parse_args()
    assert not a.output.exists(),'Refuse to overwrite existing audit'
    prior=json.loads((SCI/'results4_temperature_price_v8/RESULTS4_AUDIT.json').read_text(encoding='utf-8'))
    expected={str(Path(x['path'])):x['sha256'] for x in prior['sources']}
    domains=['scenario_valid_full_grid','scenario_availability_and_valid','common_availability_scenario_valid','common_availability_all_valid']
    best={(d,s):None for d in domains for s in SCENARIOS}
    counts={d:{s:0 for s in SCENARIOS} for d in domains};total=avail=invalid=0
    passes=np.zeros(4,dtype=np.int64);sources=[]
    for source in sorted((SCI/'full_realization_robustness_matrix_v7/realization_ledgers').glob('*.parquet')):
        digest=sha(source);assert digest==expected[str(source.relative_to(ROOT))]
        f=pd.read_parquet(source,columns=KEY+[f'{v}_{s}' for s in SCENARIOS for v in ['Aplant','E_net_year_MWh','lcoe_anchor_USD_per_MWh']])
        assert len(f)==200*61*121 and not f.duplicated(KEY).any()
        assert [f[k].nunique() for k in KEY[:3]]==[200,61,121]
        aa=f[[f'Aplant_{s}' for s in SCENARIOS]].to_numpy()>=.8
        ll=f[[f'lcoe_anchor_USD_per_MWh_{s}' for s in SCENARIOS]].to_numpy()
        ee=f[[f'E_net_year_MWh_{s}' for s in SCENARIOS]].to_numpy()
        vv=(ee>0)&np.isfinite(ll)&(ll>0);ac=aa.all(axis=1);vc=vv.all(axis=1)
        total+=len(f);avail+=int(ac.sum());invalid+=int((ac&~vc).sum())
        for eps_index,eps in enumerate([.01,.05,.10,.20]):passes[eps_index]+=int((ac&vc&((ll/BASE-1)<=eps).all(axis=1)).sum())
        for j,s in enumerate(SCENARIOS):
            for d,mask in zip(domains,[vv[:,j],aa[:,j]&vv[:,j],ac&vv[:,j],ac&vc]):
                counts[d][s]+=int(mask.sum());idx=f.loc[mask,f'lcoe_anchor_USD_per_MWh_{s}'].idxmin()
                value=float(ll[idx,j]);old=best[(d,s)]
                if old is None or value<old['LCOE_USD_MWh']:
                    best[(d,s)]={'domain':d,'scenario':s,'LCOE_USD_MWh':value,**f.loc[idx,KEY].to_dict(),
                                  'Aplant_S1_S2_S3':f.loc[idx,[f'Aplant_{x}' for x in SCENARIOS]].tolist()}
        sources.append({'path':str(source.relative_to(ROOT)),'sha256':digest,'rows':len(f)})
    assert total==5904800 and avail==4904614 and invalid==28654
    assert passes.tolist()==[229177,2838587,4068226,4468887]
    for d in domains[1:]:assert np.allclose([best[(d,s)]['LCOE_USD_MWh'] for s in SCENARIOS],BASE,rtol=0,atol=1e-12)
    rows=[{'margin_pct':e,'N_pass':int(n),'fraction_all_pct':float(n/total*100),'fraction_availability_pct':float(n/avail*100)} for e,n in zip([1,5,10,20],passes)]
    result={'status':'PASS','physical_model_rerun':False,'N_all':total,'N_availability':avail,'availability_pct':avail/total*100,
            'N_invalid_within_availability':invalid,'frozen_baselines':BASE.tolist(),
            'frozen_code_parent_domain':'scenario_availability_and_valid',
            'code_evidence':'pipeline/run_full_realization_robustness_matrix_v7.py: reference_domain = valid & frame[availability_pass_s]',
            'common_availability_minima_equal_frozen':True,'minima':list(best.values()),'domain_counts':counts,'retention':rows,'sources':sources}
    a.output.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(a.output/'Table_S4_economic_robustness.csv',index=False)
    pd.DataFrame(best.values()).to_csv(a.output/'baseline_parent_domains.csv',index=False)
    (a.output/'METHODS_POPULATION_AUDIT.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['sources','domain_counts','code_evidence']},ensure_ascii=False))
if __name__=='__main__':main()
