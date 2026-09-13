#!/usr/bin/env python3
"""Python-only V6-F secondary-sensitivity diagnostic figure."""
from pathlib import Path
import json
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs'/'v6_hts_temperature_tolerance_arc16pancake_nuc600'
COOL=OUT/'v6f_coolant_sensitivity.csv'; PRICE=OUT/'v6f_hts_price_coverage.csv'
BASE=OUT/'v6f_diagnostic'; SOURCE=OUT/'v6f_diagnostic_source_data.csv'; AUDIT=OUT/'v6f_diagnostic_audit.json'
mpl.rcParams.update({'font.family':'Arial','font.size':7,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
cool=pd.read_csv(COOL); price=pd.read_csv(PRICE)
source=pd.concat([cool.assign(dataset='coolant_boundary'),price.assign(dataset='price_coverage')],ignore_index=True,sort=False); source.to_csv(SOURCE,index=False)
fig,ax=plt.subplots(1,2,figsize=(7.2,2.75),constrained_layout=True)
for coolant,color in [('He','#4C78A8'),('H2','#E07A5F')]:
 d=cool[cool.coolant.eq(coolant)].sort_values('Npw'); ax[0].plot(d.Npw,d.Rj_max_condition_5pct_nOhm,color=color,label=coolant,lw=1.4)
ax[0].set_yscale('log'); ax[0].set(xlabel=r'$N_{pw}$',ylabel=r'$R_{j,max}^{5\%}$ (n$\Omega$)',title='20 K coolant sensitivity',xlim=(1,200),ylim=(1,100)); ax[0].legend(title='Coolant',fontsize=6,title_fontsize=6)
for top,color in [(4.2,'#4C78A8'),(10.0,'#59A14F'),(20.0,'#E15759')]:
 d=price[price.Top_K.eq(top)].sort_values('HTS_price_USD_per_kAm'); ax[1].plot(d.HTS_price_USD_per_kAm,d.coverage_log_Rj_1_to_10_pct,marker='o',ms=3,color=color,label=f'{top:g} K')
ax[1].set_xscale('log'); ax[1].set(xlabel='HTS price (2025 USD/(kA m))',ylabel='Global-5% coverage (%)',title='HTS-price sensitivity (S2, He)',xlim=(8,125),ylim=(0,105)); ax[1].legend(fontsize=6)
for i,a in enumerate(ax): a.text(-.16,1.04,chr(97+i),transform=a.transAxes,fontweight='bold',fontsize=9)
for ext,kw in [('svg',{}),('pdf',{}),('tiff',{'dpi':600}),('png',{'dpi':220})]: fig.savefig(BASE.with_suffix('.'+ext),bbox_inches='tight',**kw)
plt.close(fig)
audit={'experiment_id':'V6-F-diagnostic-python','status':'PASS','backend':'Python/matplotlib','core_claim':'20 K coolant replacement and HTS price change are secondary sensitivity tests; reported limits remain direct-model global/condition-relative boundaries.','inputs':[COOL.name,PRICE.name],'outputs':[BASE.with_suffix('.'+x).name for x in ('svg','pdf','tiff','png')]+[SOURCE.name]}
AUDIT.write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
