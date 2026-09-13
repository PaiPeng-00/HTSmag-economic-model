"""User-selected all-resistivity conservative Figure 5 from stored audit data."""
from pathlib import Path
import argparse, json, hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/'scientific_results_v7_splice_equivalent_20260906/experiments'
SOURCE=EXP/'fig5_rho_invariance_v8'
OUT=EXP/'fig5_conservative_v8'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    global SOURCE, OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir',type=Path,default=SOURCE)
    parser.add_argument('--output-dir',type=Path,default=OUT)
    args=parser.parse_args();SOURCE=args.source_dir.resolve();OUT=args.output_dir.resolve()
    if OUT.exists():raise SystemExit('STOP: output already exists; select a new --output-dir')
    OUT.mkdir(parents=True)
    audit=json.loads((SOURCE/'AUDIT.json').read_text(encoding='utf-8')); assert audit['status']=='PASS'
    d=pd.read_csv(SOURCE/'rho_boundary_summary.csv')
    d['Rj_tol_nOhm']=d.min_valid_nOhm.where(d.N_valid.eq(61))
    d['upper_censored']=d.N_censored.eq(61)&d.N_valid.eq(61)
    h=pd.read_csv(SOURCE/'N50_N200_all_rho_heatmaps.csv')
    heat=h.groupby(['Npw','Top_K','R_joint_nOhm'],as_index=False).agg(delta_max_pct=('delta_max_pct','max'),all_availability=('all_availability','all'),all_valid_economics=('all_valid_economics','all'),N_rho=('rho_turn_uOhm_cm2','size'))
    assert heat.N_rho.eq(61).all() and len(heat)==2*3*121
    for (n,t),g in heat.groupby(['Npw','Top_K']):
        g=g.sort_values('R_joint_nOhm')
        passing=g.all_availability & g.all_valid_economics & g.delta_max_pct.le(10)
        count=np.logical_and.accumulate(passing).sum()
        calc=g.R_joint_nOhm.iloc[count-1] if count else np.nan
        expected=d.loc[d.Npw.eq(n)&d.Top_K.eq(t),'Rj_tol_nOhm'].iloc[0]
        assert np.isclose(calc,expected,equal_nan=True)
    price=EXP/'results4_temperature_price_v8/Fig5B_price_sensitivity.csv'
    (OUT/'Fig5C_price_sensitivity.csv').write_bytes(price.read_bytes())
    d.to_csv(OUT/'Fig5B_conservative_boundaries.csv',index=False)
    heat.to_csv(OUT/'Fig5A_conservative_heatmaps.csv',index=False)
    result=dict(status='PASS',physical_model_rerun=False,criterion_pct=10,
        rule='Minimum tolerance over all 61 turn-contact resistivities; any missing tolerance makes the aggregate unavailable.',
        heatmap_rule='Maximum scenario-relative LCOE increase over S1-S3 and all 61 turn-contact resistivities; all-rho validity required.',
        reference_LCOE_USD_per_MWh=audit['reference_LCOE_USD_per_MWh'],
        examples=d.loc[d.Npw.isin([50,200])].to_dict('records'),
        first_valid_Npw={str(t):int(g.loc[g.Rj_tol_nOhm.notna(),'Npw'].min()) for t,g in d.groupby('Top_K')},
        price_C_unchanged_sha256=sha(price),sources={p.name:sha(p) for p in [SOURCE/'AUDIT.json',SOURCE/'rho_boundary_summary.csv',SOURCE/'N50_N200_all_rho_heatmaps.csv']})
    (OUT/'AUDIT.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
