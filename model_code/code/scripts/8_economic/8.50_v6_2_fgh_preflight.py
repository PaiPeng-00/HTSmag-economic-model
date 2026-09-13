#!/usr/bin/env python3
"""V6.2-only preflight contract for F/G/H experiments.

Replaces the obsolete V6.1-authoritative gate.  It does not perform physics;
it makes every downstream module reject old devices, matrices, or economics.
"""
from __future__ import annotations
import argparse, hashlib, json, os
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
BASE=Path(os.environ['V6_RUN_BASE']) if os.environ.get('V6_RUN_BASE') else ROOT/'v6_2_arc_actual_inductance'
DEVICE='arc_16pancake_nuc600_v6_2'
MATRIX='97242795c8f75319c8f5ea726a1fedd86999e8a66e53ddeb1255e6fb0c659ae5'

def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def audit_path(p:Path)->str:
 return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)

def read(path:Path)->dict:
 if not path.exists(): raise FileNotFoundError(path)
 return json.loads(path.read_text(encoding='utf-8'))

def check(path:Path, expected='PASS')->dict:
 a=read(path)
 if a.get('status')!=expected: raise AssertionError(f'{path.name}: {a.get("status")} != {expected}')
 if a.get('device')!=DEVICE: raise AssertionError(f'{path.name}: non-V6.2 device')
 if 'matrix_sha256' in a and a['matrix_sha256']!=MATRIX: raise AssertionError(f'{path.name}: non-ARC matrix')
 return a

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--module',choices=('F','G','H'),required=True); ap.add_argument('--stage',type=Path,required=True); args=ap.parse_args()
 a=check(BASE/'stage_A'/'v6_2_a_availability_audit.json')
 csv_b1_audit=BASE/'stage_B1'/'v6_2_b1_and_strict_feasibility_audit.json'
 pq_audit=BASE/'stage_B1_parquet_v1'/'v6_2_b1_strict_parquet_audit.json'
 b1_audit=pq_audit if pq_audit.exists() else csv_b1_audit
 b=check(b1_audit)
 c=check(BASE/'stage_C_fast_r2'/'v6_2_c_direct_bisection_audit.json')
 d=check(BASE/'stage_D'/'v6d_sampling_audit.json')
 e=check(BASE/'stage_E_fast'/'v6e_cross_scenario_audit.json')
 parquet=None
 if pq_audit.exists():
  parquet=read(pq_audit)
  if parquet.get('status')!='PASS' or parquet.get('device')!=DEVICE or parquet.get('matrix_sha256')!=MATRIX: raise AssertionError('V6.2 strict Parquet audit invalid')
 if args.module=='H':
  g=args.stage.parent/'stage_G'/'v6_2_g_audit.json'
  if not g.exists() or read(g).get('status')!='PASS': raise SystemExit('BLOCKED: V6.2-H requires V6.2-G PASS; V6.1 artifacts are not admissible')
 args.stage.mkdir(parents=True,exist_ok=True)
 out=args.stage/f'v6_2_{args.module.lower()}_preflight.json'
 if out.exists(): raise FileExistsError(out)
 payload={'experiment':f'V6.2-{args.module}','status':'PASS','timestamp_utc':datetime.now(timezone.utc).isoformat(),'device':DEVICE,'matrix_sha256':MATRIX,'economic_boundary':'incremental_anchor_v1_2025usd','rho_grid':'61 log-uniform points 10-10000 uOhm cm2','rj_grid':'121 log points 1-100 nOhm plus direct log-space bisection','temperature_grid_K':[4.2,10.0,20.0],'forbidden':['V6.1 outputs','arc_16pancake_nuc600 default device','SPARC-scaled mutual-inductance matrix'],'upstream':{'A':{'path':audit_path(BASE/'stage_A'/'v6_2_a_availability_audit.json'),'sha256':sha(BASE/'stage_A'/'v6_2_a_availability_audit.json')},'B1':{'path':audit_path(b1_audit),'sha256':sha(b1_audit)},'C':{'path':audit_path(BASE/'stage_C_fast_r2'/'v6_2_c_direct_bisection_audit.json'),'sha256':sha(BASE/'stage_C_fast_r2'/'v6_2_c_direct_bisection_audit.json')},'D':{'path':audit_path(BASE/'stage_D'/'v6d_sampling_audit.json'),'sha256':sha(BASE/'stage_D'/'v6d_sampling_audit.json')},'E':{'path':audit_path(BASE/'stage_E_fast'/'v6e_cross_scenario_audit.json'),'sha256':sha(BASE/'stage_E_fast'/'v6e_cross_scenario_audit.json')}},'strict_parquet_audit':audit_path(pq_audit) if parquet else None}
 out.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print(json.dumps(payload,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
