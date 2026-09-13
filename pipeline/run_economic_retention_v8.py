"""Frozen full-realization economic retention for Figure 4, without model runs."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SCI=ROOT/'scientific_results_v7_splice_equivalent_20260906'
PARENT=SCI/'experiments/full_realization_robustness_matrix_v7'
SCENARIOS=('S1','S2','S3')
BASELINES=np.array([1122.1472612313785,321.97916120106885,78.82074592882216])
EPS=(.01,.05,.10,.20)
KEYS=['Npw','rho_turn_uOhm_cm2','R_joint_nOhm','Top_K','coolant']

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4<<20),b''): h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,default=SCI/'experiments/economic_retention_v8'); args=p.parse_args()
    if args.output.exists(): raise SystemExit('STOP_OUTPUT_ALREADY_EXISTS: '+str(args.output))
    source_audit=json.loads((PARENT/'FULL_REALIZATION_ROBUSTNESS_MATRIX_V7.json').read_text(encoding='utf-8'))
    assert source_audit['status']=='PASS'
    assert np.array_equal(np.array([source_audit['baselines'][s] for s in SCENARIOS]),BASELINES)
    manifest=json.loads((ROOT/'public_release/postprocessing_v8_20260911/MANIFEST.json').read_text(encoding='utf-8'))
    hashes={Path(e['path']).name:e['sha256'] for e in manifest['files'] if e['path'].endswith('.parquet')}
    a_cols=[f'Aplant_{s}' for s in SCENARIOS]; l_cols=[f'lcoe_anchor_USD_per_MWh_{s}' for s in SCENARIOS]
    e_cols=[f'E_net_year_MWh_{s}' for s in SCENARIOS]; r_cols=[f'r_cryo_re_fraction_{s}' for s in SCENARIOS]
    counts=np.zeros(4,dtype=np.int64); all_count=0; avail_count=0; invalid_count=0
    cryo_avail=[]; cryo_10=[]; sources=[]
    for source in sorted((PARENT/'realization_ledgers').glob('*.parquet')):
        digest=sha(source); assert digest==hashes[source.name]
        d=pd.read_parquet(source,columns=KEYS+a_cols+l_cols+e_cols+r_cols+['availability_pass_all','delta_max'])
        assert len(d)==200*61*121 and not d.duplicated(KEYS).any()
        assert tuple(d[k].nunique() for k in KEYS)==(200,61,121,1,1)
        avail=d[a_cols].ge(.8).all(axis=1).to_numpy()
        assert np.array_equal(avail,d.availability_pass_all.to_numpy())
        lcoe=d[l_cols].to_numpy(); enet=d[e_cols].to_numpy()
        valid=(np.isfinite(lcoe)&(lcoe>0)&np.isfinite(enet)&(enet>0)).all(axis=1)
        penalty=lcoe/BASELINES-1
        pass_counts=[]
        for j,eps in enumerate(EPS):
            direct=avail&valid&(penalty<=eps).all(axis=1)
            stored=avail&(d.delta_max.to_numpy()<=eps)
            assert np.array_equal(direct,stored), 'STOP_RECOMPUTED_GATE_MISMATCH'
            pass_counts.append(int(direct.sum())); counts[j]+=direct.sum()
            if eps==.1:
                values=d.loc[direct,r_cols].to_numpy().ravel()
                assert np.isfinite(values).all(); cryo_10.append(values)
        values=d.loc[avail,r_cols].to_numpy().ravel(); assert np.isfinite(values).all(); cryo_avail.append(values)
        all_count+=len(d); avail_count+=int(avail.sum()); invalid_count+=int((avail&~valid).sum())
        sources.append({'path':str(source.relative_to(ROOT)),'sha256':digest,'N_all':len(d),'N_avail':int(avail.sum()),'threshold_counts':pass_counts})
        print(source.name, int(avail.sum()), pass_counts,flush=True)
    assert len(sources)==4 and all_count==5_904_800 and avail_count==4_904_614
    assert list(counts[:3])==[229_177,2_838_587,4_068_226]
    assert np.diff(counts).min()>=0 and counts[-1]<=avail_count-invalid_count
    retention=[{'epsilon_pct':eps*100,'N_pass':int(n),'N_availability':avail_count,'N_all':all_count,
                'f_econ_given_avail_pct':float(n/avail_count*100),'f_all_pct':float(n/all_count*100)} for eps,n in zip(EPS,counts)]
    burden=[]; names=['r_cryo_min_pct','r_cryo_P10_pct','r_cryo_median_pct','r_cryo_P90_pct','r_cryo_max_pct']
    for label,n,parts in [('availability',avail_count,cryo_avail),('availability_and_10pct',int(counts[2]),cryo_10)]:
        values=np.concatenate(parts); assert len(values)==3*n
        q=np.quantile(values,[0,.1,.5,.9,1],method='linear')*100
        burden.append({'population':label,'N_realizations':n,'N_scenario_records':len(values),**dict(zip(names,map(float,q)))})
    for key in names: assert np.isclose(burden[0][key],source_audit['primary']['Y1_all'][key],rtol=0,atol=1e-12)
    assert np.allclose([burden[1][key] for key in names],[.6013,.9019,1.6456,4.0950,8.6129],rtol=0,atol=.00005)
    margins=[{'scenario':s,'minimum_lcoe_USD_MWh':float(b),'criterion_pct':10.,'margin_USD_MWh':float(.10*b)} for s,b in zip(SCENARIOS,BASELINES)]
    report={'status':'PASS','created_utc':datetime.now(timezone.utc).isoformat(),
            'unit':'Complete five-variable realization; identical parameters across S1-S3; equal weight',
            'availability':'Aplant >= 0.80 under all three scenarios',
            'economic_validity':'Positive E_net and finite positive LCOE under all three scenarios',
            'criterion':'max_s(LCOE_i_s / frozen_global_LCOE_star_s - 1) <= epsilon',
            'no_reoptimization':True,'refrigeration_filter':False,'physical_model_rerun':False,
            'N_all':all_count,'N_availability':avail_count,'N_availability_economic_invalid':invalid_count,
            'retention':retention,'refrigeration':burden,'margins':margins,'sources':sources,'script_sha256':sha(Path(__file__))}
    args.output.mkdir(parents=True)
    pd.DataFrame(retention).to_csv(args.output/'economic_retention_1_5_10_20.csv',index=False)
    pd.DataFrame(burden).to_csv(args.output/'refrigeration_before_after_10pct.csv',index=False)
    pd.DataFrame(margins).to_csv(args.output/'scenario_10pct_margins.csv',index=False)
    (args.output/'ECONOMIC_RETENTION_V8.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    print(json.dumps({'retention':retention,'refrigeration':burden},indent=2))

if __name__=='__main__': main()
