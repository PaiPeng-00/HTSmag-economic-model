#!/usr/bin/env python3
from __future__ import annotations
import json, os, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]; BASE=Path(os.environ['V6_RUN_BASE']) if os.environ.get('V6_RUN_BASE') else ROOT/'v6_2_arc_actual_inductance'; F2=BASE/'stage_F2'/'v6_2_f2_audit.json'; G=BASE/'stage_G'/'v6_2_g_audit.json'; H=BASE/'stage_H'/'v6_2_h_audit.json'; LOG=BASE/'v6_2_continuous_pipeline.json'
def run(name):
 p=subprocess.run([sys.executable,str(Path(__file__).with_name(name))],cwd=ROOT,check=False)
 if p.returncode: raise SystemExit(f'{name} failed: {p.returncode}')
def main():
 x={'pipeline':'V6.2-F2-G-H','status':'RUNNING','started_utc':datetime.now(timezone.utc).isoformat(),'gates':{'A':'PASS','B1':'PASS','C':'PASS','D':'PASS','E':'PASS','F1':'PASS'}}; LOG.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 run('8.53d_run_v6_2_f2_price_fixed.py')
 if not F2.exists() or json.loads(F2.read_text(encoding='utf-8')).get('status')!='PASS': raise SystemExit('F2 did not pass')
 x['gates']['F2']='PASS'; LOG.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 run('8.54_run_v6_2_g_mechanisms.py')
 if not G.exists() or json.loads(G.read_text(encoding='utf-8')).get('status')!='PASS': raise SystemExit('G did not pass; H remains blocked')
 x['gates']['G']='PASS'; LOG.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 run('8.55_run_v6_2_h_crossover.py')
 if not H.exists() or json.loads(H.read_text(encoding='utf-8')).get('status')!='PASS': raise SystemExit('H did not pass')
 x['gates']['H']='PASS'; x['status']='PASS'; x['completed_utc']=datetime.now(timezone.utc).isoformat(); LOG.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print(json.dumps(x,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
