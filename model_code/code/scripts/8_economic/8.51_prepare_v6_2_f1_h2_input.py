#!/usr/bin/env python3
"""V6.2-F1 input preparation: reuse the audited 20 K H2 B-grid subset.

No duplicate physical grid is allowed: the compact V6.2-B 20 K H2 block already
contains the exact circuit/availability calculation.  This step extracts only
strict-feasible candidates from the V6.2 B1 Parquet store.
"""
from __future__ import annotations
import hashlib,json,os
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd
import pyarrow.dataset as ds

ROOT=Path(__file__).resolve().parents[3]; BASE=Path(os.environ['V6_RUN_BASE']) if os.environ.get('V6_RUN_BASE') else ROOT/'v6_2_arc_actual_inductance'
PRE=BASE/'stage_F'/'v6_2_f_preflight.json'; B1=BASE/'stage_B1_parquet_v1'/'v6_2_b1_strict_parquet_audit.json'
ROOTPQ=BASE/'stage_B1_parquet_v1'/'v6_2_b1_strict_feasible_parquet'
OUT=BASE/'stage_F'/'v6_2_f1_s2_20K_H2_strict.parquet'; AUDIT=BASE/'stage_F'/'v6_2_f1_h2_input_audit.json'
def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def audit_path(p):
 return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
def main():
 if OUT.exists() or AUDIT.exists():raise FileExistsError('refusing existing V6.2-F1 input')
 pre=json.loads(PRE.read_text(encoding='utf-8')); a=json.loads(B1.read_text(encoding='utf-8'))
 if pre.get('status')!='PASS' or a.get('status')!='PASS' or pre.get('device')!='arc_16pancake_nuc600_v6_2' or a.get('device')!='arc_16pancake_nuc600_v6_2':raise AssertionError('V6.2-F gate failed')
 parts=sorted((ROOTPQ/'scenario=S2'/'Top_K=20'/'coolant=H2').glob('*.parquet'))
 if not parts: raise FileNotFoundError('S2/20K/H2 strict Parquet partition')
 d=pd.concat([pd.read_parquet(part) for part in parts],ignore_index=True)
 if d.empty or len(d)%121:raise AssertionError(f'unexpected F1 subset rows: {len(d)}')
 d.to_parquet(OUT,index=False,compression='zstd')
 audit={'experiment':'V6.2-F1-H2-input','status':'PASS','timestamp_utc':datetime.now(timezone.utc).isoformat(),'device':'arc_16pancake_nuc600_v6_2','matrix_sha256':a['matrix_sha256'],'source_strict_parquet_audit_sha256':sha(B1),'filters':{'scenario':'S2','Top_K':20.0,'coolant':'H2'},'rows':len(d),'output':{'path':audit_path(OUT),'sha256':sha(OUT)}}
 AUDIT.write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(audit,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
