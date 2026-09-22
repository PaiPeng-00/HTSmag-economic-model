"""Recompute S3 and check S4/S5 against V10; optionally compare a fresh raw scan."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import reproduce
from pipeline import build_v10_supplementary_inputs as reducer


def equal(actual, expected, keys):
    actual=actual[expected.columns].sort_values(keys).reset_index(drop=True)
    expected=expected.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_exact=False,rtol=2e-9,atol=1e-8)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--raw',type=Path,help='Optional complete freshly solved scan; enables full S4/S5 value comparison')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    data=args.data.resolve()
    reproduce.verify_manifest(data)
    reproduce.hydrate(data)
    base=data/'figure_data/supplementary_figures'
    fresh,cases=reducer.charging()
    equal(fresh,pd.read_parquet(base/'FigS3/charging_heatload_Rj10.parquet'),['configuration_id','heat_source','time_h'])
    for token in ['4p2','10']:
        f=pd.read_parquet(base/f'FigS4/heatload_components_{token}K.parquet')
        assert len(f)==200*121*3 and not f.duplicated(reducer.KEY).any()
        np.testing.assert_allclose(f.Q_nuclear_W,reducer.cfg.V_mag*reducer.cfg.NUCLEAR_POWER_DENSITY,rtol=2e-12)
        np.testing.assert_allclose(f.Q_total_Tc_W/reducer.cfg.Ntf,f.Q_joint_W+f.Q_background_W,rtol=2e-12)
    power=pd.read_parquet(base/'FigS5/cryo_power_grid.parquet')
    assert len(power)==3*200*121 and not power.duplicated(reducer.KEY).any()
    assert np.isfinite(power[reducer.POWER]).all().all() and (power[reducer.POWER]>0).all().all()
    if args.raw:
        heat,fresh_power=reducer.scan_panels(args.raw)
        for temp,token in [(4.2,'4p2'),(10.,'10')]:
            equal(heat.loc[heat.Top_K.eq(temp)],pd.read_parquet(base/f'FigS4/heatload_components_{token}K.parquet'),reducer.KEY)
        equal(fresh_power,power,reducer.KEY)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    report={'status':'PASS','S3_all_11_cases_recomputed':True,'S4_V10_nuclear_heat_and_balance':True,
            'S5_grid_positive_finite':True,'S4_S5_complete_fresh_raw_comparison':'PASS' if args.raw else 'NOT RUN',
            'cases':cases}
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
