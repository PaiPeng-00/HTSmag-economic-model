"""Replay the V8 postprocessing from the frozen realization ledgers in this bundle.

Groups
  full, YT, YR, YJ          original four-group reducers (ledgers only)
  retention                 economic retention at 1/5/10/20 % (run_economic_retention_v8.py)
  results4                  S2 temperature minima and price sensitivity (audit_results4_v8.py)
  populations               Table S4 populations (audit_methods_populations_v8.py)
  temperature-balance       temperature-balanced weighting (audit_temperature_balance_v8.py)
  fig5                      interpolated conservative joint tolerance (interpolate_fig5_tolerance_v8.py)

Every output directory must be new; bundled reference results are never overwritten.
Example:  python -B run_analysis.py --group fig5 --output rerun_fig5
"""
from pathlib import Path
import argparse, importlib.util, json, runpy, sys

ROOT = Path(__file__).resolve().parent
EXP = ROOT / 'scientific_results_v7_splice_equivalent_20260906/experiments'
LEDGERS = EXP / 'full_realization_robustness_matrix_v7/realization_ledgers'
DERIVED = {'retention': ('run_economic_retention_v8.py', '--output'),
           'results4': ('audit_results4_v8.py', '--output'),
           'populations': ('audit_methods_populations_v8.py', '--output'),
           'temperature-balance': ('audit_temperature_balance_v8.py', '--output-dir'),
           'fig5': ('interpolate_fig5_tolerance_v8.py', '--output-dir')}
FOUR = {'full': 'run_full_realization_robustness_matrix_v7.py', 'YT': 'run_paired_temperature_effect_v7.py',
        'YR': 'run_joint_tolerance_temperature_ratio_v7.py', 'YJ': 'run_high_current_joint_resistance_yj_matrix_v7.py'}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--group', choices=list(FOUR) + list(DERIVED), required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    out = a.output.resolve()
    if out.exists():
        raise SystemExit('STOP: output already exists')
    sys.path.insert(0, str(ROOT / 'pipeline'))
    if a.group in DERIVED:
        script, flag = DERIVED[a.group]
        sys.argv = [script, flag, str(out)]
        runpy.run_path(str(ROOT / 'pipeline' / script), run_name='__main__')
        return
    spec = importlib.util.spec_from_file_location('frozen_analysis', ROOT / 'pipeline' / FOUR[a.group])
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    m.ROOT = ROOT; m.OUT = out
    m.PARENT_AUDIT = EXP / 'full_realization_robustness_matrix_v7/FULL_REALIZATION_ROBUSTNESS_MATRIX_V7.json'
    low, high = LEDGERS / 'realizations_T4p2_He.parquet', LEDGERS / 'realizations_T20p0_He.parquet'
    if a.group == 'full':
        paths = sorted(LEDGERS.glob('*.parquet'))
        if len(paths) != 4:
            raise SystemExit('STOP: expected four ledgers')
        out.mkdir(parents=True)
        population = m.build_population_matrix(paths); scenarios, weights = m.threshold_matrices(paths); rj = m.rj_boundaries(paths)
        for frame, name in [(population, 'Y0_Y3_population_matrix.csv'), (scenarios, 'scenario_threshold_matrix.csv'),
                            (weights, 'weighting_sensitivity_matrix.csv'), (rj, 'Rj_max_temperature_threshold_matrix.csv')]:
            frame.to_csv(out / name, index=False, float_format='%.17g')
        print(json.dumps({'status': 'PASS', 'group': 'full', 'model_rerun': False}))
        return
    if a.group == 'YT':
        m.SOURCE_4P2 = low; m.SOURCE_20 = high
    if a.group == 'YR':
        m.LOW = low; m.HIGH = high
    if a.group == 'YJ':
        m.SOURCES = {'4.2K': low, '20K': high}
    m.main()


if __name__ == '__main__':
    main()
