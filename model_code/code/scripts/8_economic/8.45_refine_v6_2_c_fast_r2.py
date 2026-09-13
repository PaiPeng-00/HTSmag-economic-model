#!/usr/bin/env python3
"""V6.2-C fast direct log-Rj bisection: six persistent workers, 3-decimal Rj serialization parity, and Parquet scalar audit."""
from __future__ import annotations
import concurrent.futures, hashlib, importlib.util, json, math, os, subprocess
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
V6_BASE=Path(os.environ['V6_RUN_BASE']) if os.environ.get('V6_RUN_BASE') else ROOT/'v6_2_arc_actual_inductance'
STAGE=V6_BASE/'stage_C_fast_r2'
SOURCE_STAGE=V6_BASE/'stage_C'
STAGE_A=V6_BASE/'stage_A'
BRACKETS=SOURCE_STAGE/'v6_2_c_s2_joint_tolerance_grid_brackets.csv'
BRACKET_AUDIT=SOURCE_STAGE/'v6_2_c_grid_bracket_audit.json'
RAW_DIR=STAGE/'v6_2_c_direct_bisection_raw'
EVALS=STAGE/'v6_2_c_s2_direct_bisection_evaluations.csv'
BOUNDARIES=STAGE/'v6_2_c_joint_tolerance_boundaries.csv'
SUMMARY=STAGE/'v6_2_c_tolerance_factor_summary.csv'
BINDING=STAGE/'v6_2_c_binding_constraints.csv'
AUDIT=STAGE/'v6_2_c_direct_bisection_audit.json'
RUNNING=STAGE/'v6_2_c_direct_bisection.running.json'
DEVICE='arc_16pancake_nuc600_v6_2'
MATRIX_SHA256='97242795c8f75319c8f5ea726a1fedd86999e8a66e53ddeb1255e6fb0c659ae5'
CACHE=STAGE_A/'v6_2_a_circuit_scalars.csv'
AVAIL_AUDIT=STAGE_A/'v6_2_a_availability_audit.json'
SCAN=Path(__file__).with_name('scan_full_grid.py')
ANCHOR=Path(__file__).with_name('8.5_recompute_v2_anchor_economics.py')
RHO=np.geomspace(10.0,10000.0,61)
CRITERIA=('feas','temp_1pct','temp_5pct','temp_10pct','global_1pct','global_5pct','global_10pct')
COMPACT='Top_K,coolant,scenario,Npw,rho_turn_uOhm_cm2,R_joint_nOhm,status,invalid_reason,TF_system_matrix_file,TF_system_matrix_sha256,Charging_time_999_h,Mag_loss_at_charge_W,Radial_loss_at_charge_W,Mag_loss_energy_MWh,Radial_loss_energy_MWh,P_cryo_electric_W,P_cryo_charge_peak_W,Aplant,pulse_duty_factor,CF_gross,r_cryo_re_fraction,E_gross_year_MWh,E_net_year_MWh,LCOE_plant_USD_per_MWh,CAPEX_mag_installed_USD,Power_supply_cost_USD,HTS_price_2025USD_per_kAm,E_cryo_year_MWh,energy_closure_error_MWh'
MAX_ITERATIONS=50
LOG_HALF_ERROR_TARGET=math.log(1.001)
WORKERS=6
_SCAN_MODULE=None
_SCAN_CIRCUIT_CACHE=None
_SCAN_MAGNETIZATION_CACHE={}

def sha256(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(16*1024*1024),b''): h.update(b)
 return h.hexdigest()

def audit_path(p:Path)->str:
 return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)

def load_module(name:str,path:Path):
 s=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(s); assert s and s.loader; s.loader.exec_module(m); return m

def direct_anchor(raw:pd.DataFrame)->pd.DataFrame:
 os.environ['FUSION_DEVICE']=DEVICE
 from fusion_tem import device as cfg
 from fusion_tem.economic.lcoe import define_parameters
 from fusion_tem.economic.cost_boundary import FUSION_POWER_MWTH
 from fusion_tem.economic.price_basis import POWER_SUPPLY_PRICE_2025_USD_PER_A
 a=load_module(f'v62_anchor_{os.getpid()}',ANCHOR)
 p=define_parameters(); d=raw.copy()
 if cfg.DEVICE!=DEVICE or not d['TF_system_matrix_sha256'].eq(MATRIX_SHA256).all(): raise RuntimeError('V6.2 direct provenance mismatch')
 top=pd.to_numeric(d.Top_K,errors='raise'); scenario=d.scenario.astype(str)
 hts=top.map({float(k):float(v) for k,v in p['hts_kAm_by_temperature'].items()}).astype(float)
 density=pd.Series([p['coolant_density_kg_m3'][c][float(t)] for c,t in zip(d.coolant,top)],index=d.index,dtype=float)
 mass=density*float(p['cryo_loop_volume_m3'])
 ps=pd.to_numeric(d.Power_supply_cost_USD,errors='coerce')/float(POWER_SUPPLY_PRICE_2025_USD_PER_A)
 c_hts=hts*scenario.map(a.HTS_PRICE_BY_SCENARIO).astype(float)
 c_ps=ps*float(a.POWER_SUPPLY_PRICE_USD_PER_A)
 c_fill=mass*d.coolant.map({'He':175.0,'H2':10.0}).astype(float)
 ref_hts=scenario.map({s:float(p['hts_kAm_by_temperature'][10.0])*float(a.HTS_PRICE_BY_SCENARIO[s]) for s in ('S1','S2','S3')}).astype(float)
 ref_ps=scenario.map({s:float(cfg.Ip_list[10.0])*19.0/float(cfg.carrying_factor)*float(a.POWER_SUPPLY_PRICE_USD_PER_A) for s in ('S1','S2','S3')}).astype(float)
 ref_mass=float(p['cryo_loop_volume_m3'])*float(p['coolant_density_kg_m3']['He'][10.0])
 delta=float(a.M_HTS)*(c_hts-ref_hts)+float(a.M_PS)*(c_ps-ref_ps)+float(a.M_COOLANT_FILL)*(c_fill-ref_mass*175.0)
 capital=scenario.map({s:float(a.build_capital_anchor(s).full_anchor_usd) for s in ('S1','S2','S3')}).astype(float)+delta
 rate=scenario.map(a.DISCOUNT_RATE_BY_SCENARIO).astype(float); years=scenario.map(a.PROJECT_YEARS_BY_SCENARIO).astype(int)
 crf=pd.Series([a.capital_recovery_factor(r,y) for r,y in zip(rate,years)],index=d.index)
 gross=pd.to_numeric(d.E_gross_year_MWh,errors='coerce'); net=pd.to_numeric(d.E_net_year_MWh,errors='coerce')
 annual=(pd.to_numeric(d.CF_gross,errors='coerce')*float(cfg.HOURS_PER_YEAR)*float(FUSION_POWER_MWTH)*scenario.map(a.CORE_VOM_BY_SCENARIO).astype(float)+gross*float(a.PCS_VOM_USD_PER_MWH_E)+float(a.annual_pcs_fom_usd())+c_fill*float(a.COOLANT_REPLENISH_FRACTION))
 valid=np.isfinite(net)&(net>0)&np.isfinite(capital)&np.isfinite(annual)
 lcoe=pd.Series(np.nan,index=d.index,dtype=float); lcoe.loc[valid]=(crf.loc[valid]*capital.loc[valid]+annual.loc[valid])/net.loc[valid]
 d['lcoe_anchor_USD_per_MWh']=lcoe; d['anchor_economic_valid']=valid
 return d

def worker(job:dict)->list[dict]:
 global _SCAN_MODULE, _SCAN_CIRCUIT_CACHE, _SCAN_MAGNETIZATION_CACHE
 scenario=str(job.get('scenario','S2')); coolant=str(job.get('coolant','He')); top=float(job['Top_K']); n=int(job['Npw']); it=int(job['iteration']); rj=tuple(float(x) for x in job['R_joint_nOhm'])
 raw_root=Path(job.get('raw_root', RAW_DIR))
 raw=raw_root/f'iter_{it:02d}'/scenario/f'T{top:g}K_N{n:03d}.csv'; raw.parent.mkdir(parents=True,exist_ok=True)
 if raw.exists(): raise FileExistsError(raw)
 os.environ.update({'FUSION_DEVICE':DEVICE,'CIRCUIT_SCALAR_CACHE_CSV':str(CACHE),'SCAN_COMPACT_COLUMNS':COMPACT,'SCAN_FLUSH_INTERVAL':'25000','SCAN_PROGRESS_INTERVAL':'100000','SCAN_DEFER_FINALIZE':'1','WRITE_SCAN_XLSX':'0'})
 if _SCAN_MODULE is None:
  scan=load_module(f'v62c_fast_scan_{os.getpid()}',SCAN)
  if scan.cfg.DEVICE!=DEVICE: raise RuntimeError(f'device mismatch: {scan.cfg.DEVICE}')
  _SCAN_CIRCUIT_CACHE=scan.load_circuit_scalar_cache(scan.CIRCUIT_SCALAR_CACHE)
  _original_mag_loader=scan.load_magnetization_losses_for_temp
  def _cached_mag(top_value):
   if top_value not in _SCAN_MAGNETIZATION_CACHE: _SCAN_MAGNETIZATION_CACHE[top_value]=_original_mag_loader(top_value)
   return _SCAN_MAGNETIZATION_CACHE[top_value]
  scan.load_circuit_scalar_cache=lambda _path: _SCAN_CIRCUIT_CACHE
  scan.load_magnetization_losses_for_temp=_cached_mag
  _SCAN_MODULE=scan
 else:
  scan=_SCAN_MODULE
 scan.TEMP_COOLANT_PAIRS=[(top,coolant)]; scan.SCENARIOS=[scenario]; scan.NPW_RANGE=np.array([n]); scan.R_JOINT_NOHM=np.array(rj); scan.RHO_TURN_UOHM_CM2=RHO
 scan.OUTPUT_CSV=raw; scan.OUTPUT_XLSX=raw.with_suffix('.xlsx'); scan.OUTPUT_MANIFEST=raw.with_suffix('.manifest.json'); scan.main()
 d=direct_anchor(pd.read_csv(raw,low_memory=False)); rs=sha256(raw); ans=[]
 strict=(d.status.astype(str).str.lower().eq('success')&d.anchor_economic_valid.astype(bool)&np.isfinite(d.Aplant)&(d.Aplant>=.80)&np.isfinite(pd.to_numeric(d.r_cryo_re_fraction,errors='coerce'))&(pd.to_numeric(d.r_cryo_re_fraction,errors='coerce')<=.5)&np.isfinite(pd.to_numeric(d.E_net_year_MWh,errors='coerce'))&(pd.to_numeric(d.E_net_year_MWh,errors='coerce')>0)&np.isfinite(d.lcoe_anchor_USD_per_MWh)&(d.lcoe_anchor_USD_per_MWh>0))
 for rv,g in d.groupby('R_joint_nOhm',sort=True):
  z=g.loc[strict.loc[g.index]]
  if z.empty: ans.append({'scenario':scenario,'Top_K':top,'coolant':coolant,'Npw':n,'R_joint_nOhm':float(rv),'feasible':False,'best_rho_turn_uOhm_cm2':math.nan,'Aplant':math.nan,'CF_gross':math.nan,'r_cryo_re_fraction':math.nan,'E_net_year_MWh':math.nan,'lcoe_anchor_USD_per_MWh':math.nan,'raw_path':audit_path(raw),'raw_sha256':rs})
  else:
   b=z.loc[z.lcoe_anchor_USD_per_MWh.idxmin()]; ans.append({'scenario':scenario,'Top_K':top,'coolant':coolant,'Npw':n,'R_joint_nOhm':float(rv),'feasible':True,'best_rho_turn_uOhm_cm2':float(b.rho_turn_uOhm_cm2),'Aplant':float(b.Aplant),'CF_gross':float(b.CF_gross),'r_cryo_re_fraction':float(b.r_cryo_re_fraction),'E_net_year_MWh':float(b.E_net_year_MWh),'lcoe_anchor_USD_per_MWh':float(b.lcoe_anchor_USD_per_MWh),'raw_path':audit_path(raw),'raw_sha256':rs})
 return ans

def passed(x:dict,c:str,tref:float,gref:float)->bool:
 if not x['feasible']: return False
 if c=='feas': return True
 pct=float(c.split('_')[1].replace('pct','')); ref=tref if c.startswith('temp_') else gref
 return float(x['lcoe_anchor_USD_per_MWh'])<=ref*(1+pct/100)

def main()->None:
 STAGE.mkdir(parents=True, exist_ok=True)
 for p in (BRACKETS,BRACKET_AUDIT,CACHE,AVAIL_AUDIT):
  if not p.exists(): raise FileNotFoundError(p)
 if any(p.exists() for p in (EVALS,BOUNDARIES,SUMMARY,BINDING,AUDIT,RAW_DIR,RUNNING)): raise FileExistsError('V6.2-C direct outputs already exist')
 ba=json.loads(BRACKET_AUDIT.read_text(encoding='utf-8')); aa=json.loads(AVAIL_AUDIT.read_text(encoding='utf-8'))
 if ba.get('status')!='PASS_GRID_BRACKETS_NOT_FINAL_BOUNDARIES' or ba.get('device')!=DEVICE or ba.get('matrix_sha256')!=MATRIX_SHA256 or aa.get('status')!='PASS' or aa.get('device')!=DEVICE: raise AssertionError('upstream V6.2 C gate failure')
 if float(aa['Aplant_requirement']) != 0.80: raise AssertionError('unexpected V7 plant-availability requirement')
 refs={float(k):float(v) for k,v in ba['lcoe_references_USD_per_MWh'].items()}; gref=float(ba['global_reference_USD_per_MWh']); b=pd.read_csv(BRACKETS); started=datetime.now(timezone.utc)
 RUNNING.write_text(json.dumps({'experiment_id':'V6.2-C-direct-log-bisection','status':'RUNNING','device':DEVICE,'matrix_sha256':MATRIX_SHA256,'workers':WORKERS,'started_utc':started.isoformat()},indent=2),encoding='utf-8')
 states={}
 for row in b.to_dict('records'):
  for c in CRITERIA:
   key=(float(row['Top_K']),int(row['Npw']),c); low=bool(row[f'low_reference_infeasible_{c}']); cen=bool(row[f'right_censored_{c}'])
   states[key]={'mode':'low_ref' if low else ('right_censored' if cen else 'bisection'),'lo':math.nan if low or cen else float(row[f'Rj_max_{c}_grid_lower_nOhm']),'hi':math.nan if low or cen else float(row[f'Rj_bracket_{c}_upper_nOhm']),'iterations':0}
 RAW_DIR.mkdir(parents=True); ev=[]
 with concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS) as pool:
  for it in range(1,MAX_ITERATIONS+1):
   groups={}; midpoint={}
   for key,s in states.items():
    if s['mode']!='bisection' or .5*math.log(s['hi']/s['lo'])<=LOG_HALF_ERROR_TARGET: continue
    mid=round(math.sqrt(s['lo']*s['hi']),3)
    if not s['lo']<mid<s['hi']: raise AssertionError(f'rounding failure {key}')
    midpoint[key]=mid; groups.setdefault((key[0],key[1]),set()).add(mid)
   if not midpoint: break
   jobs=[{'Top_K':t,'Npw':n,'R_joint_nOhm':sorted(v),'iteration':it} for (t,n),v in sorted(groups.items())]; rec={}
   for result in pool.map(worker,jobs):
    for x in result: rec[(x['Top_K'],x['Npw'],x['R_joint_nOhm'])]=x; ev.append({'iteration':it,**x})
   for key,mid in midpoint.items():
    x=rec[(key[0],key[1],mid)]; s=states[key]
    if passed(x,key[2],refs[key[0]],gref): s['lo']=mid
    else: s['hi']=mid
    s['iterations']+=1
   print(f'[V6.2-C] iteration {it}: {len(midpoint)} criteria, {len(jobs)} direct model batches',flush=True)
 unresolved=[k for k,s in states.items() if s['mode']=='bisection' and .5*math.log(s['hi']/s['lo'])>LOG_HALF_ERROR_TARGET]
 if unresolved: raise AssertionError(f'unresolved bisections: {len(unresolved)}')
 pd.DataFrame(ev).to_parquet(STAGE/'v6_2_c_s2_direct_bisection_evaluations.parquet',index=False,compression='zstd'); outs=[]; binds=[]
 for row in b.to_dict('records'):
  o={'scenario':'S2','Top_K':float(row['Top_K']),'coolant':'He','Npw':int(row['Npw']),'R_ref_nOhm':1.0,'L_star_temperature_USD_per_MWh':refs[float(row['Top_K'])],'L_star_global_USD_per_MWh':gref}
  for c in CRITERIA:
   s=states[(float(row['Top_K']),int(row['Npw']),c)]; low=s['mode']=='low_ref'; cen=s['mode']=='right_censored'; lo=math.nan if low or cen else s['lo']; hi=math.nan if low or cen else s['hi']; v=math.nan if low else (100.0 if cen else math.sqrt(lo*hi)); err=math.nan if low else (0.0 if cen else math.exp(.5*math.log(hi/lo))-1)
   o.update({f'Rj_max_{c}_nOhm':v,f'F_R_{c}':v,f'Rj_lower_{c}_nOhm':100.0 if cen else lo,f'Rj_upper_{c}_nOhm':math.nan if cen else hi,f'relative_boundary_error_{c}':err,f'low_reference_infeasible_{c}':low,f'right_censored_{c}':cen,f'bisection_iterations_{c}':s['iterations'],f'binding_condition_{c}':row[f'binding_condition_{c}'],f'first_interval_reentry_{c}':bool(row[f'first_interval_reentry_{c}'])})
   binds.append({'scenario':'S2','Top_K':float(row['Top_K']),'coolant':'He','Npw':int(row['Npw']),'criterion':c,'binding_condition':row[f'binding_condition_{c}'],'low_reference_infeasible':low,'right_censored':cen,'first_interval_reentry':bool(row[f'first_interval_reentry_{c}'])})
  outs.append(o)
 out=pd.DataFrame(outs); out.to_csv(BOUNDARIES,index=False,encoding='utf-8',float_format='%.17g'); bd=pd.DataFrame(binds); bd.to_csv(BINDING,index=False,encoding='utf-8')
 ss=[]
 for (t,c),g in bd.groupby(['Top_K','criterion'],sort=True):
  v=out.loc[out.Top_K.eq(t),f'F_R_{c}'].dropna().to_numpy(float); ss.append({'scenario':'S2','Top_K':t,'coolant':'He','criterion':c,'N_total':200,'N_low_reference_infeasible':int(g.low_reference_infeasible.sum()),'N_right_censored':int(g.right_censored.sum()),'N_finite_boundary':len(v),'F_R_median':float(np.median(v)) if len(v) else math.nan,'F_R_p10':float(np.quantile(v,.1)) if len(v) else math.nan,'F_R_p90':float(np.quantile(v,.9)) if len(v) else math.nan})
 pd.DataFrame(ss).to_csv(SUMMARY,index=False,encoding='utf-8',float_format='%.17g')
 errs=[math.exp(.5*math.log(s['hi']/s['lo']))-1 for s in states.values() if s['mode']=='bisection']; audit={'experiment_id':'V7-C-direct-log-bisection','status':'PASS','timestamp_utc':datetime.now(timezone.utc).isoformat(),'device':DEVICE,'matrix_sha256':MATRIX_SHA256,'scenario':'S2','coolant':'He','git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'source':{'grid_brackets':{'path':audit_path(BRACKETS),'sha256':sha256(BRACKETS)},'circuit_cache':{'path':audit_path(CACHE),'sha256':sha256(CACHE)}},'plant_availability_requirement':0.80,'temperature_references_USD_per_MWh':refs,'global_reference_USD_per_MWh':gref,'bisection':{'space':'log_Rj','workers':WORKERS,'maximum_observed_midpoint_relative_error':max(errs) if errs else 0.0,'target':0.001,'rho_optimization':'minimum frozen-B1-anchor LCOE among 61 direct-evaluated strict-feasible rho samples'},'outputs':{p.name:{'path':audit_path(p),'sha256':sha256(p)} for p in (STAGE/'v6_2_c_s2_direct_bisection_evaluations.parquet',BOUNDARIES,SUMMARY,BINDING)},'started_utc':started.isoformat()}; AUDIT.write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); RUNNING.unlink(); print(json.dumps(audit,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__': main()
