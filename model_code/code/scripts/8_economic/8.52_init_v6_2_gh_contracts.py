#!/usr/bin/env python3
"""Create V6.2-only G/H execution contracts; never fall back to V6.1."""
from __future__ import annotations
import json,os
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]; BASE=Path(os.environ['V6_RUN_BASE']) if os.environ.get('V6_RUN_BASE') else ROOT/'v6_2_arc_actual_inductance'
def main():
 gpre=BASE/'stage_G'/'v6_2_g_preflight.json'; outg=BASE/'stage_G'/'v6_2_g_execution_contract.json'; outh=BASE/'stage_H'/'v6_2_h_execution_contract.json'
 if not outg.exists():
  p=json.loads(gpre.read_text(encoding='utf-8'))
  outg.write_text(json.dumps({'experiment':'V6.2-G','status':'READY','device':p['device'],'matrix_sha256':p['matrix_sha256'],'source_contract':'V6.2 strict B1 Parquet plus C/D/E PASS; no V6.1 input','primary':{'scenario':'S2','coolant':'He','temperatures_K':[4.2,10.0,20.0],'rho_treatment':'reoptimize all 61 candidates','Npw':'1..200','Rj':'C direct boundaries and direct calls as needed'},'mechanisms':['joint cold-load slope','cryo wall-plug amplification','net-electricity benefit','HTS conductor-capital penalty'],'required_outputs':['v6_2_g_fixed_architecture_mechanism.parquet','v6_2_g_factorization.csv','v6_2_g_audit.json'],'timestamp_utc':datetime.now(timezone.utc).isoformat()},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 (BASE/'stage_H').mkdir(parents=True,exist_ok=True)
 if not outh.exists():
  outh.write_text(json.dumps({'experiment':'V6.2-H','status':'BLOCKED_UPSTREAM_V6_2_G_NOT_RUN','device':'arc_16pancake_nuc600_v6_2','source_contract':'V6.2 only; V6.1 authoritative outputs are explicitly forbidden','unblock':'v6_2_arc_actual_inductance/stage_G/v6_2_g_audit.json with status PASS','planned_sweep':{'HTS_price_USD_per_kAm':'explicit price grid','Rj_nOhm':'1-100 plus root refinement','temperatures_K':[4.2,10.0,20.0],'rho_treatment':'reoptimize all 61 candidates'},'timestamp_utc':datetime.now(timezone.utc).isoformat()},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(outg);print(outh)
if __name__=='__main__':main()
