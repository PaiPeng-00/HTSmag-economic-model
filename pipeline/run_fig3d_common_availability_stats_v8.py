"""Fig. 3D: nine refrigeration summaries after one common availability screen.

Offline reducer only. All five realization variables remain explicit. No model,
economic-validity, LCOE, refrigeration-fraction filter, or reoptimization call.
20 K pools He/H2 individual realizations with equal weights, not group medians.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCI = ROOT/'scientific_results_v7_splice_equivalent_20260906'
PARENT = SCI/'experiments/full_realization_robustness_matrix_v7'
SCENARIOS = ('S1','S2','S3')
KEYS = ['Npw','rho_turn_uOhm_cm2','R_joint_nOhm']
AVAIL = [f'Aplant_{s}' for s in SCENARIOS]
CRYO = [f'r_cryo_re_fraction_{s}' for s in SCENARIOS]
NAMES = ['min_pct','P10_pct','median_pct','P90_pct','max_pct']
OPTIONS = [('T4p2_He',4.2,'He'),('T10p0_He',10.,'He'),('T20p0_He',20.,'He'),('T20p0_H2',20.,'H2')]

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4<<20),b''): h.update(block)
    return h.hexdigest()

def stats(values):
    assert np.isfinite(values).all(), 'STOP_NONFINITE_RETAINED_REFRIGERATION'
    assert (values>0).all(), 'STOP_NONPOSITIVE_RETAINED_REFRIGERATION'
    return dict(zip(NAMES,map(float,np.quantile(values,[0,.1,.5,.9,1],method='linear')*100)))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=SCI/'experiments/fig3d_common_availability_v8')
    args=parser.parse_args(); out=args.output.resolve()
    if out.exists(): raise SystemExit('STOP_OUTPUT_ALREADY_EXISTS: '+str(out))
    parent=json.loads((PARENT/'FULL_REALIZATION_ROBUSTNESS_MATRIX_V7.json').read_text(encoding='utf-8'))
    assert parent['status']=='PASS' and parent['validation']['merged_realization_rows']==5_904_800
    manifest=json.loads((ROOT/'public_release/postprocessing_v8_20260911/MANIFEST.json').read_text(encoding='utf-8'))
    known_hashes={Path(x['path']).name:x['sha256'] for x in manifest['files'] if x['path'].endswith('.parquet')}
    protected=[ROOT/'word_version/main_EN_full_WORD_v8.docx',ROOT/'current/submission/main_EN_full.tex',ROOT/'current/submission/main_CN_full.tex',
               ROOT/'current/formal_figures/Fig2.pdf',ROOT/'current/formal_figures_cn/Fig2.pdf']
    protected_before={str(p.relative_to(ROOT)):sha(p) for p in protected}
    populations=[]; arrays={(s,t):[] for s in SCENARIOS for t in (4.2,10.,20.)}
    maxima=[]; axes_reference=None; N_all=0; N_retained=0
    for tag,temp,coolant in OPTIONS:
        path=PARENT/f'realization_ledgers/realizations_{tag}.parquet'
        source_hash=sha(path)
        assert source_hash==known_hashes[path.name], 'STOP_SOURCE_HASH_DRIFT'
        frame=pd.read_parquet(path,columns=KEYS+['Top_K','coolant']+AVAIL+CRYO+['availability_pass_all','economic_valid_all'])
        assert len(frame)==200*61*121 and not frame.duplicated(KEYS).any(), 'STOP_FULL_GRID_KEYS'
        assert frame['Top_K'].eq(temp).all() and frame['coolant'].eq(coolant).all()
        axes=[np.sort(frame[k].unique()) for k in KEYS]
        assert tuple(map(len,axes))==(200,61,121)
        assert [(a[0],a[-1]) for a in axes]==[(1,200),(10.,10_000.),(1.,100.)]
        if axes_reference is None: axes_reference=axes
        else: assert all(np.array_equal(x,y) for x,y in zip(axes_reference,axes))
        assert np.isfinite(frame[AVAIL].to_numpy()).all()
        keep=frame[AVAIL].ge(.80).all(axis=1)
        assert np.array_equal(keep.to_numpy(),frame['availability_pass_all'].to_numpy())
        selected=frame.loc[keep]
        N_all+=len(frame); N_retained+=len(selected)
        populations.append({'Top_K':temp,'coolant':coolant,'N_all':len(frame),'N_retained':len(selected),
                            'N_rejected_availability':int((~keep).sum()),
                            'N_retained_economic_invalid':int((~selected['economic_valid_all']).sum()),
                            'source':str(path.relative_to(ROOT)),'sha256':source_hash,'duplicate_keys':0,
                            'scenario_specific_availability_counts':{s:int(frame[f'Aplant_{s}'].ge(.8).sum()) for s in SCENARIOS}})
        for s in SCENARIOS:
            values=selected[f'r_cryo_re_fraction_{s}'].to_numpy()
            assert np.isfinite(values).all()
            arrays[s,temp].append(values)
            maximum=values.max()
            ties=selected.loc[selected[f'r_cryo_re_fraction_{s}'].eq(maximum),KEYS+['Top_K','coolant']+AVAIL].copy()
            ties.insert(0,'scenario',s); ties['r_cryo_pct']=maximum*100
            maxima.extend(ties.to_dict(orient='records'))
        print(f'{tag}: all={len(frame)}, retained={len(selected)}, rejected={int((~keep).sum())}',flush=True)
    assert N_all==5_904_800 and N_retained==4_904_614
    rows=[]; overall_arrays=[]
    for s in SCENARIOS:
        for temp in (4.2,10.,20.):
            values=np.concatenate(arrays[s,temp]); overall_arrays.append(values)
            rows.append({'group':f'YD-{s}-{temp:g}','scenario':s,'Top_K':temp,
                         'coolants':'He+H2' if temp==20. else 'He','N':len(values),**stats(values)})
    summary=pd.DataFrame(rows)
    for temp in (4.2,10.,20.): assert summary.loc[summary.Top_K.eq(temp),'N'].nunique()==1
    for s in SCENARIOS: assert summary.loc[summary.scenario.eq(s),'N'].sum()==N_retained
    combined=np.concatenate(overall_arrays)
    assert len(combined)==3*N_retained
    overall={'group':'YD-all-scenario-records','N_realizations':N_retained,'N_scenario_records':len(combined),**stats(combined)}
    for name in NAMES:
        expected=parent['primary']['Y1_all']['r_cryo_'+name]
        assert np.isclose(overall[name],expected,rtol=0,atol=1e-12), (name,overall[name],expected)
    maximum=overall['max_pct']
    argmax=pd.DataFrame([r for r in maxima if r['r_cryo_pct']==maximum])
    assert len(argmax)>0
    immutable={str(p.relative_to(ROOT)):sha(p) for p in protected}
    assert immutable==protected_before, 'STOP_UNRELATED_ARTIFACT_CHANGED'
    audit={'status':'PASS','created_utc':datetime.now(timezone.utc).isoformat(),
           'definition':'Full five-variable realizations with Aplant_S1 >= 0.80 AND Aplant_S2 >= 0.80 AND Aplant_S3 >= 0.80',
           'weighting':'Equal weight per complete realization; 20 K pools He and H2 rows directly',
           'quantile_method':'NumPy linear quantile; fractions multiplied by 100 after quantile',
           'no_extra_filters':['E_net','LCOE','economic_valid_all','r_cryo<=0.50','scenario-specific screen','reoptimization'],
           'N_all':N_all,'N_retained':N_retained,'N_retained_scenario_records':len(combined),
           'partitions':populations,'summary':rows,'overall':overall,'argmax':argmax.to_dict(orient='records'),
           'manuscript_and_figure_hashes_unchanged':immutable,'physical_model_rerun':False,
           'figures_modified':False,'manuscripts_modified':False,'script_sha256':sha(Path(__file__))}
    out.mkdir(parents=True)
    summary.to_csv(out/'YD_nine_group_summary.csv',index=False)
    pd.DataFrame([overall]).to_csv(out/'YD_overall_summary.csv',index=False)
    argmax.to_csv(out/'YD_argmax.csv',index=False)
    (out/'YD_AUDIT.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    lines=['# Fig. 3D 共同 availability 总体：九组统计','',
           '母总体：三个情景均满足 Aplant≥0.80 的完整五变量 realization；等权。',
           f'全扫描 {N_all:,}；保留 {N_retained:,}；总体分位数合并 {len(combined):,} 条情景记录。',
           '20 K 直接合并 He 与 H2 个体，不平均两种 coolant 的分位数；不施加经济有效性、LCOE 或制冷比例筛选。','',
           '| 组 | N | min (%) | P10 (%) | median (%) | P90 (%) | max (%) |','|---|---:|---:|---:|---:|---:|---:|']
    for row in rows: lines.append('| '+row['group']+' | '+f"{row['N']:,}"+' | '+' | '.join(f'{row[k]:.6f}' for k in NAMES)+' |')
    lines+=['','总体 [min, P10, median, P90, max] (%) = '+str([overall[k] for k in NAMES]),'',
            '最大值精确并列数：'+str(len(argmax)),'','```json',json.dumps(audit['argmax'],ensure_ascii=False,indent=2),'```','',
            '注意：这不是 YT 的固定三变量、He-only、跨温度共同合格配对总体，不能据此给出纯温度的 paired reduction。',
            '本步未改 Word、中英文 TeX 或 Fig.3，未重跑物理模型。']
    (out/'YD_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(summary.to_string(index=False)); print(json.dumps({'overall':overall,'argmax':audit['argmax'],'output':str(out)},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
