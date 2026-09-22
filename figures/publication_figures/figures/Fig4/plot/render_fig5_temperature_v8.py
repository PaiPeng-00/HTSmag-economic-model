"""Displayed Figure 5: all-rho conservative A/B, unchanged price sensitivity C."""
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
import argparse,hashlib,json,os
import numpy as np,pandas as pd,fitz
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.colors import LinearSegmentedColormap,Normalize
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator,NullLocator,FuncFormatter
from panel_b_v7_style import draw_panel_b

ROOT=Path(__file__).resolve().parents[5]
DATA=ROOT/'results/figure_inputs/current/main_figures/Fig5'
W,H=180,162
COLORS={4.2:'#8DCEF3',10.0:'#D0E1B5',20.0:'#FBDABB'}
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--language',choices=['EN','CN'],default='CN' if os.environ.get('V68_CN_FONT')=='1' else 'EN')
    a=p.parse_args();cn=a.language=='CN';lang=lambda en,zh:zh if cn else en
    out=ROOT/f'figures/publication_figures/build_v8/{a.language}/Fig4';out.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(DATA/'panel_A_conservative_heatmaps.csv');price=pd.read_csv(DATA/'panel_C_price_sensitivity.csv')
    bounds=pd.read_csv(DATA/'panel_B_conservative_boundaries.csv')
    font='Times New Roman' if cn else 'Arial'
    mpl.rcParams.update({'font.family':[font,'SimSun'] if cn else [font],
        'font.size':7,'pdf.fonttype':42,'mathtext.fontset':'custom','mathtext.rm':font,
        'mathtext.it':font+':italic','mathtext.bf':font+':bold','axes.linewidth':.6,'axes.edgecolor':'black'})
    fig=plt.figure(figsize=(W/25.4,H/25.4),facecolor='white')
    def axes(x,y,w,h): return fig.add_axes([x/W,(H-y-h)/H,w/W,h/H])
    def title(x,y,letter,en,zh):
        fig.text(x/W,1-y/H,letter,fontsize=9,fontweight='bold',va='top')
        heading=fig.text((x+5)/W,1-y/H,lang(en,zh),fontsize=7.6,fontweight='bold',va='top')
        # SimSun has no bold face; match the stroked CN headings used in panel B.
        if cn: heading.set_path_effects([path_effects.Stroke(linewidth=.32,foreground='black'),path_effects.Normal()])
    title(4,3,'A','Joint-resistance tolerance across all turn-to-turn contact resistivities','覆盖全部匝间接触电阻率的接头电阻容限')
    title(103,93,'C','Temperature preference and conductor price','温度选择与导体价格')
    cmap=LinearSegmentedColormap.from_list('penalty',['#FFFFFF','#C7E8FA','#61B2E3'])
    cax=axes(18,20,61,3)
    cb=fig.colorbar(mpl.cm.ScalarMappable(norm=Normalize(0,20),cmap=cmap),cax=cax,orientation='horizontal',ticks=[0,5,10,20])
    cb.ax.set_xticklabels(['0','5','10','20+']);cb.ax.tick_params(labelsize=6,length=2,direction='in');cb.outline.set_linewidth(.4)
    fig.text(48.5/W,1-14/H,lang('Maximum LCOE increase (%)','最大 LCOE 增幅（%）'),ha='center',va='top',fontsize=6.2)
    key=axes(107,14,64,14);key.set(xlim=(0,1),ylim=(0,1));key.axis('off')
    key.plot(.08,.73,'v',color='black',ms=3)
    key.text(.21,.73,lang('Interpolated 10% limit','插值 10% 容限'),fontsize=6.2,va='center')
    key.add_patch(Rectangle((.025,.11),.10,.21,facecolor='#F1F2F2',edgecolor='#777777',hatch='////',lw=.4))
    key.text(.21,.215,lang('Invalid LCOE','LCOE 无效'),fontsize=6.2,va='center')
    axs=[]
    for n,xpos in [(50,18),(200,108)]:
        ax=axes(xpos,37,65,42);axs.append(ax)
        ax.set_title(rf'$N_{{\mathrm{{pw}}}}={n}$',fontsize=8,pad=4)
        for row,t in enumerate([20.,10.,4.2]):
            f=d.loc[d.Npw.eq(n)&d.Top_K.eq(t)].sort_values('R_joint_nOhm')
            r=f.R_joint_nOhm.to_numpy();edges=np.r_[1,np.sqrt(r[:-1]*r[1:]),100]
            assert f.all_availability.all()
            valid=f.all_valid_economics.to_numpy() & np.isfinite(f.delta_max_pct.to_numpy())
            ax.pcolormesh(edges,[row,row+1],np.where(valid,f.delta_max_pct.to_numpy(),np.nan)[None,:],cmap=cmap,vmin=0,vmax=20,rasterized=False,shading='flat',edgecolors='face',linewidth=.05,antialiased=False)
            for j in np.flatnonzero(~valid):
                ax.add_patch(Rectangle((edges[j],row),edges[j+1]-edges[j],1,facecolor='#F1F2F2',edgecolor='#999999',hatch='////',lw=0))
            bound=bounds.loc[bounds.Npw.eq(n)&bounds.Top_K.eq(t)].iloc[0]
            x=bound.Rj_tol_nOhm;assert np.isfinite(x)
            ax.plot(x,row+.64,'v',color='black',ms=4,zorder=5,clip_on=False)
            label='≥100' if bound.upper_censored else str(Decimal(str(x)).quantize(Decimal('.1'),rounding=ROUND_HALF_UP))
            ax.text(x/1.12 if x>50 else x*1.13,row+.64,label,fontsize=6.6,va='center',ha='right' if x>50 else 'left')
            ax.plot([1,100],[row,row],color='black',lw=.4)
        ax.set(xscale='log',xlim=(1,100),ylim=(0,3),yticks=[.5,1.5,2.5],yticklabels=['20 K','10 K','4.2 K'])
        ax.xaxis.set_major_locator(FixedLocator([1,2,5,10,20,50,100]));ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:g}'));ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlabel(lang(r'Joint resistance, $R_{\mathrm{j}}$ (n$\Omega$)','接头电阻 (nΩ)'),fontsize=7,labelpad=3)
    tx,panel_b_style_audit=draw_panel_b(fig,bounds,W,H,cn)
    bx=axes(116,108,57,40);temps=[4.2,10.,20.];ps=[10.,50.,100.]
    width=.23
    for k,t in enumerate(temps):
        vals=price.loc[price.Top_K.eq(t)].set_index('HTS_price_USD_per_kAm').loc[ps]
        bx.bar(np.arange(3)+(k-1)*width,vals.lcoe_premium_pct,width=width,color=COLORS[t],edgecolor='none',label=f'{t:g} K',zorder=3)
        zero=np.isclose(vals.lcoe_premium_pct,0)
        bx.plot((np.arange(3)+(k-1)*width)[zero],vals.lcoe_premium_pct.to_numpy()[zero],'o',color=COLORS[t],mec='#777777',mew=.3,ms=2.5,clip_on=False,zorder=5)
    for j,pp in enumerate(ps):
        rows=price.loc[price.HTS_price_USD_per_kAm.eq(pp)]
        bx.text(j,4.57,f'{rows.preferred_temperature_K.iloc[0]:g} K',ha='center',fontsize=6.5)
    bx.set(xlim=(-.55,2.55),ylim=(0,5),xticks=[0,1,2],xticklabels=['10','50','100'],yticks=[0,1,2,3,4,5])
    bx.set_xlabel(lang('HTS conductor price\n'+r'(US\$ kA$^{-1}$ m$^{-1}$)','HTS 导体价格\n'+r'(US\$ kA$^{-1}$ m$^{-1}$)'),fontsize=7,labelpad=3)
    bx.set_ylabel(lang('LCOE increase above minimum (%)','LCOE 相对最低值增幅（%）'),fontsize=7.0,labelpad=3)
    bx.legend(loc='lower center',bbox_to_anchor=(.5,1.045),ncol=3,frameon=False,fontsize=6,handlelength=1,columnspacing=.8,handletextpad=.4)
    for axt in axs+[bx]:
        for sp in axt.spines.values():sp.set_visible(True);sp.set_linewidth(.6);sp.set_color('black')
        axt.tick_params(direction='in',top=True,right=True,width=.6,length=2.5,labelsize=6.5,pad=3)
    import sys
    sys.path.insert(0,str(ROOT/'figures/publication_figures/programs'))
    from manuscript_axis_style_v8 import apply_manuscript_axis_fonts
    axis_style=apply_manuscript_axis_fonts(fig,W)
    fig.savefig(out/'Fig4.pdf');plt.close(fig)
    with fitz.open(out/'Fig4.pdf') as doc:
        page=doc[0]
        outlined_svg=page.get_svg_image(text_as_path=True)
        (out/'Fig4.svg').write_text(outlined_svg,encoding='utf-8')
        (out/'Fig4.word.svg').write_text(outlined_svg,encoding='utf-8')
        page.get_pixmap(matrix=fitz.Matrix(600/72,600/72),alpha=False).save(out/'Fig4.png')
        page.get_pixmap(matrix=fitz.Matrix(220/72,220/72),alpha=False).save(out/'Fig4_preview.png')
    input_hashes={p.name:sha(p) for p in DATA.iterdir() if p.is_file()}
    report={'status':'STRUCTURE_PASS_VISUAL_PENDING','display_figure':5,'asset':'Fig4','panels':['A','B','C'],
            'main_A_temperatures_K':[4.2,10,20],'main_A_Npw':[50,200],
            'main_AB_rho_aggregation':'minimum over S1-S3 and all 61 rho_turn values',
            'main_A_heatmap_aggregation':'cross-scenario conservative envelope',
            'main_AB_coolant':'He','criterion_pct':10,
            'scenario_specific_LCOE_references':True,
            'panel_C_independent_S2_price_sensitivity_unchanged':True,
            'panel_B_style':panel_b_style_audit,'axis_style':axis_style,
            'canonical_input_sha256':input_hashes,
            'outputs':{f'Fig4.{ext}':sha(out/f'Fig4.{ext}') for ext in ['pdf','png','svg']},'canvas_mm':[W,H]}
    (out/'FIG5_RENDER_AUDIT.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'language':a.language,'status':report['status']}))
if __name__=='__main__':main()
