#!/usr/bin/env python3
"""Python-only hierarchical five-panel replacement candidate for Fig. 6."""
from pathlib import Path
import json, hashlib
import numpy as np, pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs'/'v6_hts_temperature_tolerance_arc16pancake_nuc600'
SENS=OUT/'v6b_sensitivity_summary.csv'; BND=OUT/'v6c_joint_tolerance_boundaries.csv'; COV=OUT/'v6d_tolerance_coverage.csv'; B1=ROOT/'outputs'/'target_price_window'/'manuscript_b1'/'tables'/'B1_robust_architecture_coverage.csv'
DIR=OUT/'fig6_v6_temperature_tolerance_hierarchical'; BASE=DIR/'Fig6_hierarchical_temperature_tolerance'; SRC=DIR/'Fig6_hierarchical_source_data.csv'; AUD=DIR/'Fig6_hierarchical_audit.json'
TOPS=(4.2,10.,20.); COL={4.2:'#0F4D92',10.:'#239A9A',20.:'#D55E3A'}
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':7,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'legend.frameon':False})
def h(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def lab(ax,s): ax.text(-.13,1.035,s,transform=ax.transAxes,fontweight='bold',fontsize=9,va='bottom')
def split(ax,x,y,ok,**kw):
 ix=np.flatnonzero(ok)
 for seg in np.split(ix,np.flatnonzero(np.diff(ix)>1)+1):
  if len(seg): ax.plot(x[seg],y[seg],**kw)
def main():
 for p in (SENS,BND,COV,B1):
  if not p.exists(): raise FileNotFoundError(p)
 if DIR.exists(): raise FileExistsError(DIR)
 s,b,c,b1=map(pd.read_csv,(SENS,BND,COV,B1)); DIR.mkdir()
 pd.concat([s.assign(panel='A_B'),b.assign(panel='C'),c.assign(panel='D'),b1.assign(panel='E')],ignore_index=True,sort=False).to_csv(SRC,index=False)
 fig=plt.figure(figsize=(7.2,8.15),constrained_layout=True); gs=fig.add_gridspec(3,2,height_ratios=[1,2.2,1])
 a,bx=fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1]); hero=fig.add_subplot(gs[1,:]); d,e=fig.add_subplot(gs[2,0]),fig.add_subplot(gs[2,1])
 for ax,metric,yl,title in [(a,'S_Rj_cryo',r'$S_{R_j}^{\mathrm{cryo}}$','Refrigeration sensitivity'),(bx,'S_Rj_LCOE',r'$S_{R_j}^{\mathrm{LCOE}}$','LCOE sensitivity')]:
  f=s[s.metric.eq(metric)].sort_values('Top_K')
  for i,t in enumerate(TOPS):
   r=f[f.Top_K.eq(t)].iloc[0]; ax.vlines(i,r.p25,r.p75,color=COL[t],lw=5,alpha=.32); ax.vlines(i,r.p25,r.p75,color=COL[t],lw=1); ax.scatter(i,r['median'],s=25,c=COL[t]); ax.scatter(i,r.p90,s=20,marker='D',facecolors='white',edgecolors=COL[t],lw=.8)
  ax.set(xticks=range(3),xticklabels=['4.2 K','10 K','20 K'],ylabel=yl,title=title); lab(ax,'A' if ax is a else 'B')
  if ax is a:
   ax.text(.98,.96,'median: dot\nIQR: bar\nP90: diamond',transform=ax.transAxes,ha='right',va='top',fontsize=6); ax.annotate('-77% median sensitivity',xy=(2,float(f[f.Top_K.eq(20.)]['median'].iloc[0])),xytext=(1.35,ax.get_ylim()[1]*.82),fontsize=6.1,arrowprops={'arrowstyle':'-','color':'#777777','lw':.6})
  else: ax.annotate('-93% median sensitivity',xy=(2,float(f[f.Top_K.eq(20.)]['median'].iloc[0])),xytext=(1.05,ax.get_ylim()[1]*.78),fontsize=6.1,arrowprops={'arrowstyle':'-','color':'#777777','lw':.6})
 for t in TOPS:
  q=b[b.Top_K.eq(t)].sort_values('Npw'); x=q.Npw.to_numpy(float); g=pd.to_numeric(q.Rj_max_global_5pct_nOhm,errors='coerce').to_numpy(float); tr=pd.to_numeric(q.Rj_max_temp_5pct_nOhm,errors='coerce').to_numpy(float); ok=np.isfinite(g); okt=np.isfinite(tr)
  split(hero,x,g,ok,color=COL[t],lw=2.2,label=f'{t:g} K global-relative'); split(hero,x,tr,okt,color=COL[t],lw=.85,ls='--',alpha=.9)
  both=ok&okt; hero.fill_between(x,np.where(both,g,np.nan),np.where(both,tr,np.nan),color=COL[t],alpha=.13)
  if ok[-1]: hero.annotate(f'{t:g} K: {g[-1]:.1f} nOhm',(200,g[-1]),xytext=(5,0),textcoords='offset points',color=COL[t],fontsize=6.5,va='center')
  if t==20 and okt[-1]: hero.annotate(f'{tr[-1]:.1f} nOhm within-temperature',(200,tr[-1]),xytext=(5,1),textcoords='offset points',color=COL[t],fontsize=6,va='bottom')
 hero.set_yscale('log'); hero.set(xlim=(1,200),ylim=(.9,120),xlabel=r'$N_{pw}$',ylabel=r'$R_{j,max}^{5\%}$ (nOhm)',title='Joint-resistance tolerance for near-optimal plant performance'); lab(hero,'C'); hero.legend(loc='lower left',ncol=3,fontsize=6)
 hero.text(.025,.91,'solid: global-relative 5%\ndashed + band: within-temperature 5%',transform=hero.transAxes,fontsize=6.2,va='top'); hero.annotate('about 9x',xy=(198, pd.to_numeric(b[b.Top_K.eq(20.)].sort_values('Npw').Rj_max_global_5pct_nOhm.iloc[-1])),xytext=(170,4),arrowprops={'arrowstyle':'<->','color':'#777','lw':.7},ha='center',fontsize=8,color='#555')
 f=c[c.criterion.eq('temp_5pct')]; x=np.arange(3)
 eng=np.array([float(f[(f.Top_K.eq(t)) & (f.measure.eq('log_Rj')) & (f.domain.eq('engineering_1_to_10_nOhm'))].coverage.iloc[0])*100 for t in TOPS])
 stress=np.array([float(f[(f.Top_K.eq(t)) & (f.measure.eq('log_Rj')) & (f.domain.eq('stress_1_to_100_nOhm'))].coverage.iloc[0])*100 for t in TOPS])
 d.plot(x,eng,color='#333',lw=1,marker='o',ms=5,label='log-$R_j$, 1-10 nOhm'); d.plot(x,stress,color='#777',lw=1,marker='o',mfc='white',ms=5,label='log-$R_j$, 1-100 nOhm'); d.set(xticks=x,xticklabels=['4.2 K','10 K','20 K'],ylim=(0,105),ylabel=r'$C_{T,5\%}$ (%)',title='Tolerance-space coverage'); d.legend(fontsize=5.7,loc='lower left'); lab(d,'D')
 b1=b1.sort_values('tolerance_pct'); e.plot(b1.tolerance_pct,b1.coverage_pct,color='#777',lw=.8); e.scatter(b1.tolerance_pct,b1.coverage_pct,s=[26,54,26],c=['#777','#0F4D92','#777'],zorder=3)
 for r in b1.itertuples(index=False): e.annotate(f'{r.coverage_pct:.1f}%',(r.tolerance_pct,r.coverage_pct),xytext=(2,5),textcoords='offset points',fontsize=6.3,fontweight='bold' if r.tolerance_pct==5 else 'normal')
 e.set(xlim=(0,10.8),ylim=(50,100),xlabel='Allowed relative LCOE penalty (%)',ylabel='Coverage of common-feasible architectures (%)',title='Cross-scenario near-optimality'); lab(e,'E')
 for ext,kw in [('svg',{}),('pdf',{}),('tiff',{'dpi':600}),('png',{'dpi':180})]: fig.savefig(BASE.with_suffix('.'+ext),bbox_inches='tight',**kw)
 plt.close(fig)
 AUD.write_text(json.dumps({'status':'PASS','backend':'Python','contract':{'core_conclusion':'20 K widens within-temperature tolerance, whereas global commercial tolerance distinguishes it from 10 K.','archetype':'hierarchical quantitative grid','panels':'A/B sensitivities; C hero near-optimal tolerance; D coverage; E cross-scenario context'},'inputs':{p.name:h(p) for p in (SENS,BND,COV,B1)},'outputs':{p.name:h(p) for p in [SRC,*[BASE.with_suffix('.'+x) for x in ('svg','pdf','tiff','png')]]}},indent=2)+'\n')
if __name__=='__main__': main()



