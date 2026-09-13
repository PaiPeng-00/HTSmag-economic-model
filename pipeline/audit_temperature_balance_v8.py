"""Frozen-data temperature balance, with weights specified BEFORE availability screening."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
SCI=ROOT/'scientific_results_v7_splice_equivalent_20260906/experiments'
OUT=SCI/'temperature_balance_v8'
BASE=np.array([1122.1472612313785,321.97916120106885,78.82074592882216])
EPS=[1,5,10,20]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    global OUT
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir',type=Path,help='New directory for an independent reproduction; existing results are never overwritten')
    args=ap.parse_args()
    if args.output_dir is not None:OUT=args.output_dir.resolve()
    assert not OUT.exists(),'Refuse to overwrite a completed sensitivity'
    prior=json.loads((SCI/'methods_populations_v8/METHODS_POPULATION_AUDIT.json').read_text(encoding='utf-8'))
    rows=[];sources=[]
    for source in prior['sources']:
        path=ROOT/source['path'];assert sha(path)==source['sha256']
        keys=['Npw','rho_turn_uOhm_cm2','R_joint_nOhm','Top_K','coolant']
        df=pd.read_parquet(path,columns=keys+[f'{v}_{s}' for s in ['S1','S2','S3'] for v in ['Aplant','E_net_year_MWh','lcoe_anchor_USD_per_MWh']])
        assert len(df)==200*61*121 and not df.duplicated(keys).any()
        assert [df[k].nunique() for k in keys]==[200,61,121,1,1]
        temp=float(df.Top_K.iloc[0]);coolant=str(df.coolant.iloc[0]);w=.5 if temp==20 else 1.
        a=df[[f'Aplant_{s}' for s in ['S1','S2','S3']]].to_numpy()
        e=df[[f'E_net_year_MWh_{s}' for s in ['S1','S2','S3']]].to_numpy()
        l=df[[f'lcoe_anchor_USD_per_MWh_{s}' for s in ['S1','S2','S3']]].to_numpy()
        av=(a>=.8).all(axis=1);valid=((e>0)&np.isfinite(l)&(l>0)).all(axis=1)
        penalty=(l/BASE-1).max(axis=1)
        for eps in EPS:
            p=av&valid&(penalty<=eps/100)
            # Independent test uses three explicit comparisons, not max(penalty).
            independent=av&valid
            for j in range(3):independent=independent&((l[:,j]/BASE[j]-1)<=eps/100)
            assert np.array_equal(p,independent)
            rows.append({'Top_K':temp,'coolant':coolant,'margin_pct':eps,'N_all':len(df),'N_availability':int(av.sum()),'N_invalid_within_availability':int((av&~valid).sum()),'N_pass':int(p.sum()),'weight_per_realization_relative':w})
        sources.append(source)
    per=pd.DataFrame(rows);out=[]
    assert set(zip(per.Top_K,per.coolant))=={(4.2,'He'),(10.,'He'),(20.,'He'),(20.,'H2')}
    for eps,expected in zip(EPS,[229177,2838587,4068226,4468887]):
        f=per[per.margin_pct==eps];assert f.N_all.sum()==5904800 and f.N_availability.sum()==4904614 and f.N_pass.sum()==expected
        w=f.weight_per_realization_relative.to_numpy()
        n=float(np.dot(w,f.N_pass));d=float(np.dot(w,f.N_availability))
        # Diagnostic only: exactly one-third of the CONDITIONED population per temperature.
        bytemp=f.groupby('Top_K')[['N_pass','N_availability']].sum()
        postbalanced=float((bytemp.N_pass/bytemp.N_availability).mean()*100)
        ordinary=expected/4904614*100
        out.append({'margin_pct':eps,'equal_realization_pct':ordinary,'temperature_balanced_pct':n/d*100,'change_percentage_points':n/d*100-ordinary,'weighted_passing_count_relative':n,'weighted_availability_denominator_relative':d,'post_availability_equal_temperature_pct_diagnostic':postbalanced})
    OUT.mkdir(parents=True)
    per.to_csv(OUT/'temperature_coolant_counts.csv',index=False)
    pd.DataFrame(out).to_csv(OUT/'temperature_balanced_retention.csv',index=False)
    result={'status':'PASS','physical_model_rerun':False,'frozen_baselines':BASE.tolist(),'weight_definition':'Before filtering: 4.2K He=1, 10K He=1, 20K He=0.5, 20K H2=0.5. Common scale cancels; full-grid temperature shares are one third each.','conditional_denominator':'Sum of preassigned weights over common availability, including economically invalid realizations.','post_screen_diagnostic':'Optional exact balance of the three temperature groups AFTER availability; not the primary sensitivity definition.','sources':sources,'counts':rows,'retention':out}
    (OUT/'TEMPERATURE_BALANCE_AUDIT.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    print(json.dumps({'status':'PASS','retention':out},ensure_ascii=False))
if __name__=='__main__':main()
