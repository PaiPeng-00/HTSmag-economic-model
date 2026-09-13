#!/usr/bin/env python3
"""Run one deterministic V6.1-B (scenario, temperature/coolant) production chunk."""
from __future__ import annotations
import argparse, importlib.util, json, os
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
DEVICE='arc_16pancake_nuc600'
# Fields required by V6.1-B feasibility, V6.1-C transient audit, and V6-G/H.
COMPACT_COLUMNS=(
'Top_K,coolant,scenario,Npw,rho_turn_uOhm_cm2,R_joint_nOhm,status,invalid_reason,'
'C0_nonmagnet_USD,project_lifetime_years,discount_rate,CRF,Charging_time_999_h,'
'EM_source_Npw,EM_source_rho_turn_uOhm_cm2,EM_coupling_method,Mag_loss_at_charge_W,Radial_loss_at_charge_W,'
'Mag_loss_energy_MWh,Radial_loss_energy_MWh,P_cryo_electric_W,COP_Tc,COP_77,Q_coil_internal_joint_W,'
'Q_pancake_joint_W,Q_nuclear_W,Q_radiation_W,Q_current_leads_HTS_W,Q_pipes_coolant_W,Q_pipes_aux_W,'
'Q_quench_W,Q_misc_W,Q_total_Tc_W,Q_total_77K_W,P_cryo_charge_peak_W,R_radial_per_pancake_Ohm,'
'R_radial_per_TF_Ohm,R_radial_system_Ohm,CAPEX_mag_direct_USD,CAPEX_mag_installed_USD,Tape_cost_USD,'
'HTS_price_2025USD_per_kAm,Coolant_fill_cost_USD,Power_supply_cost_USD,AF,r_cryo_re,r_cryo_re_fraction,'
'P_cryo_prod_W,P_cryo_dwell_W,P_cryo_static_W,P_cryo_coolwarm_W,P_cryo_excdec_W,P_cryo_charge_average_W,'
'E_cryo_charge_event_MWh,E_cryo_discharge_event_MWh,E_cryo_excdis_year_MWh,E_other_year_MWh,E_cryo_year_MWh,'
'E_gross_year_MWh,E_net_year_MWh,E_cryo_TF_annual_MWh,net_energy_annual_MWh,LCOE_plant_USD_per_MWh,'
'LCOE_magnet_only_USD_per_MWh,LCOE_fullplant_USD_per_MWh,cryo_efficiency_model,cryo_reference_temperature_K,'
'cryo_rated_4p5eq_kW,cryo_eta_raw_fraction_carnot,cryo_eta_cap_fraction_carnot,cryo_eta_rated_fraction_carnot,'
'cryo_eta_cap_active,cryo_rated_margin,cryo_part_load_factor,non_tf_aux_fraction,energy_closure_error_MWh'
)
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--scenario',choices=['S1','S2','S3'],required=True); ap.add_argument('--top',type=float,required=True); ap.add_argument('--coolant',choices=['He','H2'],required=True); ap.add_argument('--stage',type=Path,required=True); ap.add_argument('--cache',type=Path,required=True); args=ap.parse_args()
 if args.coolant=='H2' and args.top!=20.: raise ValueError('H2 is defined only at 20 K')
 os.environ['FUSION_DEVICE']=DEVICE; os.environ['CIRCUIT_SCALAR_CACHE_CSV']=str(args.cache.resolve()); os.environ['SCAN_COMPACT_COLUMNS']=COMPACT_COLUMNS; os.environ['SCAN_FLUSH_INTERVAL']='5000'; os.environ['SCAN_PROGRESS_INTERVAL']='10000'; os.environ['SCAN_DEFER_FINALIZE']='1'
 scan_path=Path(__file__).with_name('scan_full_grid.py'); spec=importlib.util.spec_from_file_location('v61b_scan',scan_path); mod=importlib.util.module_from_spec(spec); assert spec.loader; spec.loader.exec_module(mod)
 if mod.cfg.DEVICE!=DEVICE: raise RuntimeError('device mismatch')
 stage=args.stage.resolve(); stage.mkdir(parents=True,exist_ok=True); label=f'v6_1_b_{args.scenario}_{args.top:g}K_{args.coolant}'
 mod.TEMP_COOLANT_PAIRS=[(args.top,args.coolant)]; mod.SCENARIOS=[args.scenario]; mod.NPW_RANGE=np.arange(1,201,dtype=int); mod.RHO_TURN_UOHM_CM2=np.geomspace(10.,10000.,61); mod.R_JOINT_NOHM=np.geomspace(1.,100.,121)
 mod.OUTPUT_CSV=stage/f'{label}.csv'; mod.OUTPUT_XLSX=stage/f'{label}.xlsx'; mod.OUTPUT_MANIFEST=stage/f'{label}.manifest.json'
 expected=200*61*121
 (stage/f'{label}.start.json').write_text(json.dumps({'experiment_id':'V6.1-B','status':'RUNNING','device':DEVICE,'expected_rows':expected,'compact_column_count':len(COMPACT_COLUMNS.split(',')),'cache':str(args.cache.resolve())},indent=2),encoding='utf-8')
 mod.main()
if __name__=='__main__': main()
