"""Postprocess unfiltered stored realizations; no physical-model evaluation."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCI = ROOT / 'scientific_results_v7_splice_equivalent_20260906'
OUT = SCI / 'experiments/fig5_rho_invariance_v8'
SCENARIOS = ['S1', 'S2', 'S3']
REFERENCE = np.array([1122.1472612313785, 321.97916120106885, 78.82074592882216])
KEY = ['Npw', 'rho_turn_uOhm_cm2', 'R_joint_nOhm']

def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUT,
                        help='New output directory; existing directories are never overwritten.')
    OUT = parser.parse_args().output_dir.resolve()
    if OUT.exists():
        raise SystemExit('STOP: output already exists')
    manifest = json.loads((ROOT / 'public_release/postprocessing_v8_20260911/MANIFEST.json').read_text(encoding='utf-8'))
    hashes = {Path(x['path']).name: x['sha256'] for x in manifest['files'] if x['path'].endswith('.parquet')}
    summaries, rows, heatmaps, sources = [], [], [], []
    for source in sorted((SCI / 'experiments/full_realization_robustness_matrix_v7/realization_ledgers').glob('*.parquet')):
        identity = pd.read_parquet(source, columns=['Top_K', 'coolant'])
        assert identity.Top_K.nunique() == identity.coolant.nunique() == 1
        if identity.coolant.iloc[0] != 'He':
            continue
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        assert digest == hashes[source.name]
        cols = KEY + ['Top_K', 'coolant'] + [f'{stem}_{s}' for stem in ['Aplant','E_net_year_MWh','lcoe_anchor_USD_per_MWh'] for s in SCENARIOS]
        d = pd.read_parquet(source, columns=cols).sort_values(KEY)
        assert len(d) == 200*61*121 and not d.duplicated(KEY).any()
        assert [d[k].nunique() for k in KEY] == [200,61,121]
        assert d.groupby(KEY[:2]).size().eq(121).all()
        temperature = float(d.Top_K.iloc[0])
        r = np.sort(d.R_joint_nOhm.unique())
        assert r[0] == 1 and r[-1] == 100
        av = d[[f'Aplant_{s}' for s in SCENARIOS]].to_numpy() >= .8
        en = d[[f'E_net_year_MWh_{s}' for s in SCENARIOS]].to_numpy() > 0
        lc = d[[f'lcoe_anchor_USD_per_MWh_{s}' for s in SCENARIOS]].to_numpy()
        valid = en & np.isfinite(lc) & (lc > 0)
        delta = lc / REFERENCE - 1
        passing = (av & valid & (delta <= .1)).all(axis=1).reshape(200*61,121)
        n_consecutive = np.logical_and.accumulate(passing, axis=1).sum(axis=1)
        base = d.drop_duplicates(KEY[:2])[KEY[:2]].reset_index(drop=True)
        base['Top_K'] = temperature
        base['Rj_tol_nOhm'] = np.where(n_consecutive > 0, r[np.maximum(n_consecutive-1,0)], np.nan)
        base['valid_tolerance'] = n_consecutive > 0
        base['upper_censored'] = n_consecutive == 121
        base['nonmonotonic'] = np.any(passing & ~np.logical_and.accumulate(passing,axis=1),axis=1)
        base['first_fail_nOhm'] = np.where(n_consecutive < 121,r[np.minimum(n_consecutive,120)],np.nan)
        for stem, array in [('availability', av), ('economic_validity', valid), ('LCOE',delta <= .1)]:
            first = array.reshape(200*61,121,3)[np.arange(200*61),np.minimum(n_consecutive,120),:]
            base['first_failure_'+stem] = ['|'.join(s for s,v in zip(SCENARIOS, values) if not v) if n < 121 else '' for values,n in zip(first,n_consecutive)]
        base = base.loc[base.Npw.ge(20)].copy()
        rows.append(base)
        for n, group in base.groupby('Npw'):
            assert len(group) == 61
            nvalid = int(group.valid_tolerance.sum())
            n_unique = int(group.Rj_tol_nOhm.nunique())
            summaries.append(dict(Npw=int(n),Top_K=temperature,N_rho=61,N_valid=nvalid,
                                  min_valid_nOhm=None if nvalid==0 else float(group.Rj_tol_nOhm.min()),
                                  max_valid_nOhm=None if nvalid==0 else float(group.Rj_tol_nOhm.max()),
                                  N_distinct_valid_boundaries=n_unique,
                                  all_61_valid_and_equal=bool(nvalid==61 and n_unique==1),
                                  N_censored=int(group.upper_censored.sum()),N_nonmonotonic=int(group.nonmonotonic.sum())))
        h=d.loc[d.Npw.isin([50,200])].copy()
        h['delta_max_pct']=100*delta[d.Npw.isin([50,200])].max(axis=1)
        h['all_availability']=av[d.Npw.isin([50,200])].all(axis=1)
        h['all_valid_economics']=valid[d.Npw.isin([50,200])].all(axis=1)
        heatmaps.append(h)
        sources.append(dict(path=str(source.relative_to(ROOT)),sha256=digest,rows=len(d),axis_cardinalities=[200,61,121]))
        print(f'He {temperature:g} K: complete ledger audited',flush=True)
    assert sorted({x['Top_K'] for x in summaries}) == [4.2,10,20]
    summary = pd.DataFrame(summaries)
    details = pd.concat(rows,ignore_index=True)
    assert len(details) == 181*3*61 and len(summary)==181*3
    result = dict(status='PASS',physical_model_rerun=False,reference_LCOE_USD_per_MWh=dict(zip(SCENARIOS,REFERENCE.tolist())),
                  sources=sources,N_base_temperature_rho_rows=len(details),
                  N_groups_all_61_valid_and_equal=int(summary.all_61_valid_and_equal.sum()),
                  N_groups_with_missing_tolerance=int(summary.N_valid.lt(61).sum()),
                  N_groups_with_distinct_valid_boundaries=int(summary.N_distinct_valid_boundaries.gt(1).sum()),
                  rho_can_be_omitted_for_full_requested_range=bool(summary.all_61_valid_and_equal.all()),
                  six_examples=summary.loc[summary.Npw.isin([50,200])].to_dict('records'),
                  first_all_61_equal_Npw_by_temperature={str(t): int(g.loc[g.all_61_valid_and_equal,'Npw'].min()) if g.all_61_valid_and_equal.any() else None for t,g in summary.groupby('Top_K')})
    OUT.mkdir(parents=True)
    details.to_csv(OUT/'individual_rho_boundaries.csv',index=False)
    summary.to_csv(OUT/'rho_boundary_summary.csv',index=False)
    pd.concat(heatmaps).to_csv(OUT/'N50_N200_all_rho_heatmaps.csv',index=False)
    (OUT/'AUDIT.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__ == '__main__':
    main()
