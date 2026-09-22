"""Regenerate numerical inputs for Figs. S3-S5 from the current V10 model.

Fig. S3 uses the eleven caption-defined cases, the same circuit equations,
and the same linear-in-time magnetization interpolation as the full scan.
Figs. S4/S5 are exact selections from a newly solved full scan, never from
historical figure tables. No model parameter or physical equation is changed.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / 'model/model_code/code'
sys.path.insert(0, str(CODE / 'src'))
os.environ.setdefault('FUSION_DEVICE', 'arc_16pancake_nuc600_v6_2')
from fusion_tem import device as cfg
from fusion_tem.cryo import heat_load as hlm
from fusion_tem.utils import calculate_radial_resistance

CASES = {'A1':(5,5000.),'B1':(10,5000.),'B2':(10,1500.),
         'C1':(20,5000.),'C2':(20,1500.),'C3':(20,100.),'C4':(20,50.),
         'D1':(200,5000.),'D2':(200,1500.),'D3':(200,100.),'D4':(200,50.)}
KEY = ['Top_K','coolant','scenario','Npw','rho_turn_uOhm_cm2','R_joint_nOhm']
HEAT = ['Q_coil_internal_joint_W','Q_pancake_joint_W','Q_nuclear_W','Q_radiation_W',
        'Q_current_leads_HTS_W','Q_pipes_coolant_W','Q_pipes_aux_W','Q_quench_W',
        'Q_misc_W','Q_total_Tc_W','Q_total_77K_W','P_cryo_electric_W','P_cryo_charge_peak_W']
POWER = ['P_cryo_prod_W','P_cryo_dwell_W','P_cryo_static_W']


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def charging():
    solver = module('supp_charge', CODE / 'scripts/2_charging/2.5_charge_time999_TF_system_all_Npw=1-200.py')
    scan = module('supp_scan', CODE / 'scripts/8_economic/scan_full_grid.py')
    mag = scan.load_magnetization_losses_for_temp(20.)
    if not mag:
        raise ValueError('Magnetization input did not load')
    matrix = pd.read_excel(CODE / 'data/raw/inductance' / cfg.TF_SYSTEM_MATRIX, header=None).to_numpy(float)
    frames, audit = [], []
    for index, (config, (npw, rho)) in enumerate(CASES.items()):
        L = matrix * (cfg.Nt_list[20.] * cfg.NP / npw)**2
        R = calculate_radial_resistance(npw, cfg.Nt_list[20.], rho*1e-10)*cfg.NP
        t, current, _, power, _ = solver.simulate_charging_system(L, [R]*cfg.Ntf, npw=npw,
            I_target_local=cfg.Ip_list[20.], Ntape_coil_local=cfg.Nt_list[20.], steady_hours=500.)
        duration = solver.calculate_time_to_999(t, current, npw=npw, I_target_per_conductor=cfg.Ip_list[20.])
        if duration is None:
            raise ValueError(f'No charging crossing: {config}')
        keep = t/3600. <= 1.2*cfg.CHARGE_HOURS
        times = t[keep]/3600.
        supply = np.minimum(times/cfg.CHARGE_HOURS, 1.)
        nearest = min(mag, key=lambda k: abs(k[0]-npw)/max(npw,1) + abs(k[1]-rho)/max(rho,1e-6))
        series = mag[nearest].sort_index()
        sources = {'radial':power[keep].sum(axis=1)/cfg.Ntf,
            'magnetization':np.interp(times,series.index.to_numpy(float),series.to_numpy(float)),
            'lead':np.full(len(times),hlm.current_lead_heat(Npw=npw,Top=20.,Ip=cfg.Ip_list[20.])[2]),
            'radiative':np.full(len(times),hlm.radiation_heat(cfg.A_cryostat,cfg.eps,cfg.T_HIGH,20.)),
            'coolant_pipe':np.full(len(times),hlm.pipe_heat(cfg.N_cool_pipe,cfg.d_in_cool,cfg.d_out_cool,cfg.L_cool,cfg.T_HIGH,20.)),
            'auxiliary_pipe':np.full(len(times),hlm.pipe_heat(cfg.N_aux_pipe,cfg.d_in_aux,cfg.d_out_aux,cfg.L_aux,cfg.T_HIGH,20.)),
            'other_background':np.full(len(times),hlm.misc_heat()),
            'coil_joint':hlm.pancake_joint_heat(npw,10e-9,cfg.Ip_list[20.])*supply**2,
            'internal_joint':hlm.coil_internal_joint_heat(npw,L_total_m=cfg.L_HTS_TF_m[20.],Ip=cfg.Ip_list[20.])*supply**2}
        for source, values in sources.items():
            frames.append(pd.DataFrame({'panel_id':chr(65+index),'configuration_id':config,
                'temperature_K':20.,'Rj_nOhm':10.,'Npw':npw,'rho_turn_uOhm_cm2':rho,
                'charging_time_h':duration,'current_ramp_end_h':cfg.CHARGE_HOURS,
                'display_end_h':1.2*cfg.CHARGE_HOURS,'time_h':times,'heat_source':source,
                'temperature_stage':'magnet','heat_load_W':values}))
        audit.append({'case':config,'charging_time_h':duration,'magnetization_source':list(nearest)})
    return pd.concat(frames, ignore_index=True), audit


def scan_panels(raw):
    heats, powers = [], []
    for frame in pd.read_csv(raw, usecols=KEY+HEAT+POWER, chunksize=200_000, low_memory=False):
        base = frame.coolant.eq('He') & np.isclose(frame.rho_turn_uOhm_cm2,10000.)
        selected = base & frame.Top_K.isin([4.2,10.])
        if selected.any():
            heats.append(frame.loc[selected,KEY+HEAT].copy())
        selected = base & frame.scenario.eq('S1')
        if selected.any():
            powers.append(frame.loc[selected,KEY+POWER].copy())
    heat, power = pd.concat(heats,ignore_index=True), pd.concat(powers,ignore_index=True)
    assert len(heat) == 2*200*121*3 and len(power) == 3*200*121
    assert not heat.duplicated(KEY).any() and not power.duplicated(KEY).any()
    heat['Q_joint_W'] = heat.Q_coil_internal_joint_W + heat.Q_pancake_joint_W
    heat['Q_background_W'] = heat.Q_total_Tc_W/cfg.Ntf - heat.Q_joint_W
    assert np.allclose(heat.Q_nuclear_W,cfg.V_mag*cfg.NUCLEAR_POWER_DENSITY,rtol=1e-12)
    assert ((heat.Q_background_W-heat.Q_nuclear_W)>0).all()
    return heat, power


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New directory for supplementary_figures and audit')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    for name in ['FigS3','FigS4','FigS5']:
        (out/'supplementary_figures'/name).mkdir(parents=True)
    f3, cases = charging()
    f3.to_parquet(out/'supplementary_figures/FigS3/charging_heatload_Rj10.parquet',index=False)
    heat, power = scan_panels(args.raw)
    for temperature, token in [(4.2,'4p2'),(10.,'10')]:
        heat.loc[heat.Top_K.eq(temperature)].sort_values(KEY).to_parquet(
            out/f'supplementary_figures/FigS4/heatload_components_{token}K.parquet',index=False)
    power.sort_values(KEY).to_parquet(out/'supplementary_figures/FigS5/cryo_power_grid.parquet',index=False)
    audit = {'status':'PASS','version':'V10','raw_source':str(args.raw.resolve()),
             'Q_nuclear_W_per_TF':float(cfg.V_mag*cfg.NUCLEAR_POWER_DENSITY),
             'system_to_single_TF_divisor':int(cfg.Ntf),'FigS3_cases':cases,
             'FigS3_interpolation':'same linear time interpolation as scan_full_grid.py',
             'FigS3_rows':len(f3),'FigS4_rows':len(heat),'FigS5_rows':len(power),
             'unit_contract':'Q_total_Tc_W and electrical power are system totals; other heat components are per TF. Q_background_W includes nuclear heat.',
             'files':{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*.parquet')}}
    (out/'SUPPLEMENTARY_INPUT_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(audit,indent=2))


if __name__ == '__main__':
    main()
