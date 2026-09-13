"""First-crossing linear-Rj interpolation of stored, unfiltered He LCOE curves.

Interpolate each scenario and resistivity separately BEFORE minimization.
Never alter the raw ledgers, references, sample counts, or Fig. 5C data.
"""
from pathlib import Path
import argparse, hashlib, json, shutil
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'scientific_results_v7_splice_equivalent_20260906/experiments'
SCENARIOS = ['S1', 'S2', 'S3']
KEY = ['Npw', 'rho_turn_uOhm_cm2', 'R_joint_nOhm']

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def crossings(r, delta, availability, energy, valid, eps):
    """Arrays have shape (curve, node). Unresolved invalid brackets stay undefined."""
    passed = availability & energy & valid & (delta <= eps)
    prefix = np.logical_and.accumulate(passed, axis=1)
    n = prefix.sum(axis=1)
    m = len(r); idx = np.arange(len(n))
    low = np.maximum(n - 1, 0); high = np.minimum(n, m - 1)
    dlo, dhi = delta[idx, low], delta[idx, high]
    bracket = (n > 0) & (n < m)
    interpolable = (bracket & availability[idx, high] & energy[idx, high]
                    & valid[idx, high] & np.isfinite(dlo) & np.isfinite(dhi)
                    & (dlo <= eps) & (dhi > eps))
    estimate = np.full(len(n), np.nan)
    estimate[n == m] = r[-1]
    estimate[interpolable] = (r[low[interpolable]] +
        (eps-dlo[interpolable])/(dhi[interpolable]-dlo[interpolable]) *
        (r[high[interpolable]]-r[low[interpolable]]))
    status = np.full(len(n), 'initial_failure', dtype=object)
    status[n == m] = 'upper_censored'
    status[bracket] = 'unresolved_non_economic_bracket'
    status[interpolable] = 'interpolated'
    return pd.DataFrame(dict(Rj_tol_nOhm=estimate, status=status,
        upper_censored=n == m, n_initial_passing_nodes=n,
        r_lo_nOhm=np.where(n > 0, r[low], np.nan),
        r_hi_nOhm=np.where(n < m, r[high], np.nan),
        delta_lo=np.where(n > 0, dlo, np.nan),
        delta_hi=np.where(n < m, dhi, np.nan),
        availability_hi=availability[idx, high], energy_hi=energy[idx, high],
        valid_LCOE_hi=valid[idx, high],
        nonmonotonic=np.any(passed & ~prefix, axis=1)))

def self_test():
    r=np.array([1.,2.,4.]); d=np.array([[0.,.08,.16],[.2,.3,.4],[0.,.01,.02],[0.,np.nan,.2],[0.,.2,0.]])
    yes=np.ones_like(d,dtype=bool)
    q=crossings(r,d,yes,yes,np.isfinite(d),.1)
    assert np.isclose(q.Rj_tol_nOhm[0],2.5)
    assert q.status.tolist()==['interpolated','initial_failure','upper_censored','unresolved_non_economic_bracket','interpolated']
    assert q.nonmonotonic[4] and np.isclose(q.Rj_tol_nOhm[4],1.5)
    # A controlling-curve switch: min(individual crossings) != crossing(max endpoints).
    dc=np.array([[0.,.2],[.09,.11]]); yc=np.ones_like(dc,dtype=bool)
    a=crossings(np.array([1.,2.]),dc,yc,yc,yc,.1)
    b=crossings(np.array([1.,2.]),dc.max(axis=0)[None,:],yc[:1],yc[:1],yc[:1],.1)
    assert np.isclose(a.Rj_tol_nOhm.min(),1.5) and not np.isclose(b.Rj_tol_nOhm.iloc[0],1.5)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=EXP/'fig5_interpolated_v8')
    args=parser.parse_args(); out=args.output_dir.resolve()
    if out.exists(): raise SystemExit('STOP: output exists; select a new --output-dir')
    self_test()
    old=json.loads((EXP/'fig5_conservative_v8/AUDIT.json').read_text(encoding='utf-8'))
    refs=old['reference_LCOE_USD_per_MWh']
    provenance=json.loads((EXP/'fig5_rho_invariance_v8/AUDIT.json').read_text(encoding='utf-8'))
    curves=[]; sources=[]
    for identity in provenance['sources']:
        source=ROOT/identity['path'].replace('\\','/'); assert sha(source)==identity['sha256']
        cols=KEY+['Top_K','coolant']+[f'{stem}_{s}' for stem in ['Aplant','E_net_year_MWh','lcoe_anchor_USD_per_MWh'] for s in SCENARIOS]
        d=pd.read_parquet(source,columns=cols).sort_values(KEY)
        assert len(d)==200*61*121 and not d.duplicated(KEY).any()
        assert [d[k].nunique() for k in KEY]==[200,61,121]
        assert d.coolant.eq('He').all() and d.Top_K.nunique()==1
        r=np.sort(d.R_joint_nOhm.unique()); assert r[0]==1 and r[-1]==100
        assert np.array_equal(d.R_joint_nOhm.to_numpy().reshape(-1,121),np.tile(r,(12200,1)))
        t=float(d.Top_K.iloc[0]); base=d.drop_duplicates(KEY[:2])[KEY[:2]].reset_index(drop=True)
        for s in SCENARIOS:
            a=d[f'Aplant_{s}'].to_numpy().reshape(-1,121)
            en=d[f'E_net_year_MWh_{s}'].to_numpy().reshape(-1,121)
            lc=d[f'lcoe_anchor_USD_per_MWh_{s}'].to_numpy().reshape(-1,121)
            delta=lc/refs[s]-1
            for eps in [.01,.05,.10,.20]:
                q=crossings(r,delta,a>=.8,en>0,np.isfinite(lc)&(lc>0),eps)
                q=pd.concat([base,q],axis=1);q['Top_K']=t;q['scenario']=s;q['epsilon']=eps
                curves.append(q.loc[q.Npw.ge(20)].copy())
        sources.append(identity)
        print(f'He {t:g} K: {len(d):,} unfiltered rows verified; raw-response interpolation complete.',flush=True)
    c=pd.concat(curves,ignore_index=True)
    rows=[]
    for key,g in c.groupby(['Npw','Top_K','rho_turn_uOhm_cm2','epsilon'],sort=True):
        assert len(g)==3
        valid=g.Rj_tol_nOhm.notna().all(); z=g.loc[g.Rj_tol_nOhm.idxmin()] if valid else None
        row=dict(zip(['Npw','Top_K','rho_turn_uOhm_cm2','epsilon'],key))
        row.update(Rj_tol_nOhm=float(z.Rj_tol_nOhm) if valid else np.nan,
            upper_censored=bool(valid and g.upper_censored.all()),
            controlling_scenario=z.scenario if valid and not g.upper_censored.all() else '',
            status=z.status if valid else ('initial_failure' if g.status.eq('initial_failure').any() else 'unresolved_non_economic_bracket'),
            nonmonotonic=bool(g.nonmonotonic.any()))
        for k in ['r_lo_nOhm','r_hi_nOhm','delta_lo','delta_hi']:
            row[k]=float(z[k]) if valid else np.nan
        rows.append(row)
    individuals=pd.DataFrame(rows); summaries=[]
    for key,g in individuals.groupby(['Npw','Top_K','epsilon'],sort=True):
        assert len(g)==61
        valid=g.Rj_tol_nOhm.notna().all(); z=g.loc[g.Rj_tol_nOhm.idxmin()] if valid else None
        row=dict(zip(['Npw','Top_K','epsilon'],key))
        row.update(N_rho=61,N_valid=int(g.Rj_tol_nOhm.notna().sum()),
            Rj_tol_nOhm=float(z.Rj_tol_nOhm) if valid else np.nan,
            min_valid_nOhm=g.Rj_tol_nOhm.min(),max_valid_nOhm=g.Rj_tol_nOhm.max(),
            upper_censored=bool(valid and g.upper_censored.all()),
            N_censored=int(g.upper_censored.sum()),N_nonmonotonic=int(g.nonmonotonic.sum()),
            controlling_scenario=z.controlling_scenario if valid and not g.upper_censored.all() else '',
            controlling_rho_uOhm_cm2=float(z.rho_turn_uOhm_cm2) if valid and not g.upper_censored.all() else np.nan)
        for k in ['r_lo_nOhm','r_hi_nOhm','delta_lo','delta_hi']: row[k]=float(z[k]) if valid else np.nan
        summaries.append(row)
    summary=pd.DataFrame(summaries)
    unresolved=c.loc[c.status.eq('unresolved_non_economic_bracket')]
    # Scientific gate: do not publish a fallback node as an interpolated threshold.
    if len(unresolved):
        raise RuntimeError(f'STOP: {len(unresolved)} non-economic first-failure intervals require review')
    out.mkdir(parents=True)
    c.to_parquet(out/'scenario_rho_crossings.parquet',index=False)
    c.loc[c.Npw.isin([50,200])].to_csv(out/'N50_N200_scenario_rho_crossings.csv',index=False)
    individuals.to_csv(out/'individual_rho_interpolated_boundaries.csv',index=False)
    summary.to_csv(out/'all_margins_conservative_boundaries.csv',index=False)
    summary.loc[summary.epsilon.eq(.1)].to_csv(out/'Fig5B_conservative_boundaries.csv',index=False)
    for name in ['Fig5A_conservative_heatmaps.csv','Fig5C_price_sensitivity.csv']:
        shutil.copyfile(EXP/'fig5_conservative_v8'/name,out/name)
    sensitivity=individuals.loc[individuals.Npw.eq(200)&individuals.rho_turn_uOhm_cm2.eq(10000)]
    sensitivity.to_csv(out/'high_current_fixed_rho_sensitivity.csv',index=False)
    def clean_records(d): return json.loads(d.to_json(orient='records',double_precision=15))
    report=dict(status='PASS',physical_model_rerun=False,criterion_pct=10,
        interpolation_coordinate='R_joint_nOhm (linear, not logarithmic)',
        rule='Minimum of individually interpolated scenario/resistivity crossings; any initial failure makes the conservative tolerance undefined.',
        heatmap_rule=old['heatmap_rule'],reference_LCOE_USD_per_MWh=refs,sources=sources,
        N_scenario_rho_margin_curves=len(c),N_unresolved_invalid_brackets=len(unresolved),
        N_nonmonotonic_curves=int(c.nonmonotonic.sum()),
        examples=clean_records(summary.loc[summary.epsilon.eq(.1)&summary.Npw.isin([50,200])]),
        fixed_rho_sensitivity=clean_records(sensitivity),
        price_C_unchanged_sha256=sha(out/'Fig5C_price_sensitivity.csv'),
        heatmap_A_unchanged_sha256=sha(out/'Fig5A_conservative_heatmaps.csv'),
        original_sample_counts_and_refrigeration_statistics_unchanged=True)
    (out/'AUDIT.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(report['examples'],indent=2));print(sensitivity.to_string(index=False))

if __name__=='__main__': main()
