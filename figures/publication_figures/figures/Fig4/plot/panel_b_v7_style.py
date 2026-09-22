"""Reuse the v7 joint-tolerance B painter with v8 conservative boundary data.

Style authority: v7/figures/publication_figures/figures/Fig4/plot/approved_renderer_v17.py,
add_panel_b_overlay, and v7/figures/publication_figures/programs/palette_b.py.
Only the data adapter, semantic labels and NaN-safe segmentation are updated.
The original B geometry is translated down by 20 mm on the current canvas.
No legacy PDF overlay, data optimization or physical-model evaluation is used.
"""
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
import matplotlib as mpl
import matplotlib.patheffects as path_effects
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

COLORS={4.2:'#8DCEF3',10.0:'#D0E1B5',20.0:'#FBDABB'}
SOURCE_SHA256='b00b74f7782d84a16f543249509637ac64c22f413b3975e9aa17d85ab174ec3b'
PALETTE_SHA256='6545797664f49babf7c2aaa32301991be3d824c7b943d5d7f22347a9b4c1000d'

def draw_panel_b(fig,bounds,width_mm,height_mm,cn=False):
    # Display-only crop: preserve the evaluated-domain onset in the audit.
    cutoffs={str(t):int(bounds.loc[bounds.Top_K.eq(t) & bounds.Rj_tol_nOhm.notna(),'Npw'].min())
             for t in [4.2,10.,20.]}
    bounds=bounds.loc[bounds.Npw.between(50,200)].copy()
    font='Times New Roman' if cn else 'Arial'
    lang=lambda en,zh:zh if cn else en
    pt=25.4/72
    shift=20.0
    with mpl.rc_context({'font.family':[font,'SimSun'] if cn else font,
                         'mathtext.fontset':'custom',
                         'mathtext.rm':font,'mathtext.it':font+':italic','mathtext.bf':font+':bold'}):
        left,right=38.24*pt,242.34*pt
        top,bottom=226.20*pt+shift,365.10*pt+shift
        ax=fig.add_axes([left/width_mm,(height_mm-bottom)/height_mm,
                         (right-left)/width_mm,(bottom-top)/height_mm],facecolor='none')
        ax.set_zorder(2)
        ax.set(xscale='log',yscale='log',xlim=(50,200),ylim=(1,100))
        ax.grid(axis='y',color='#DFE5EF',linewidth=.35,zorder=0)
        common=bounds.pivot(index='Npw',columns='Top_K',values='Rj_tol_nOhm').sort_index()
        assert common.shape==(151,3)
        # Keep the full ordered index: missing tolerances must create gaps.
        present=np.isfinite(common[[4.2,10.,20.]].to_numpy()).all(axis=1)
        x=common.index.to_numpy(float)
        r4=common[4.2].to_numpy(float);r10=common[10.].to_numpy(float);r20=common[20.].to_numpy(float)
        assert np.all(r4[present]<=r10[present]) and np.all(r10[present]<=r20[present])
        for low,high,t in [(r10,r20,20.),(r4,r10,10.),(np.ones_like(r4),r4,4.2)]:
            ax.fill_between(x,low,high,where=present,color=COLORS[t],alpha=.035,linewidth=0,zorder=1)
        for t in [4.2,10.,20.]:
            rows=bounds.loc[bounds.Top_K.eq(t)].sort_values('Npw')
            y=rows.Rj_tol_nOhm.to_numpy(float)
            finite=np.isfinite(y)
            uncensored=finite & ~rows.upper_censored.to_numpy(bool)
            censored=finite & rows.upper_censored.to_numpy(bool)
            # The curve rises to the 100-nOhm top edge where the limit leaves the
            # scanned range; no plateau is drawn along the frame (V8 Step 199).
            y_line=np.where(uncensored,y,np.nan)
            edge=censored & (np.r_[uncensored[1:],False] | np.r_[False,uncensored[:-1]])
            y_line[edge]=100.
            ax.plot(rows.Npw,y_line,color=COLORS[t],linewidth=1.80,
                    solid_capstyle='round',zorder=4,label=f'{t:g} K')
            markers=uncensored & rows.Npw.isin([50.,100.,150.,200.]).to_numpy()
            ax.plot(rows.Npw.to_numpy()[markers],y[markers],linestyle='none',marker='o',
                    markersize=2.7,markeredgewidth=0,color=COLORS[t],clip_on=False,zorder=5)
            endpoint=rows.loc[rows.Npw.eq(200)].iloc[0]
            assert np.isfinite(endpoint.Rj_tol_nOhm) and not endpoint.upper_censored
            label=str(Decimal(str(endpoint.Rj_tol_nOhm)).quantize(Decimal('.1'),rounding=ROUND_HALF_UP))
            ax.annotate(label,xy=(200.,endpoint.Rj_tol_nOhm),xytext=(-2.,-1.),
                        textcoords='offset points',ha='right',va='top',fontsize=6.3,color='#2F3E4E',zorder=6)
        ax.xaxis.set_major_locator(FixedLocator([50,100,200]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value,pos:f'{value:g}'))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_locator(FixedLocator([1,2,5,10,20,50,100]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value,pos:f'{value:g}'))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.tick_params(which='major',direction='in',labelsize=6.5,width=.60,length=3.,pad=2.,top=False,right=False)
        for spine in ax.spines.values():
            spine.set_visible(True);spine.set_linewidth(.65);spine.set_color('#000000');spine.set_zorder(30)
        ax.set_xlabel(lang(r'Parallel tapes per turn, $N_{\mathrm{pw}}$',r'每匝并联带材数，$N_{\mathrm{pw}}$'),
                      fontsize=7.0,labelpad=4.,fontfamily='SimSun' if cn else font)
        ax.set_ylabel(lang('Maximum tolerable joint resistance (nΩ)',r'最大可容忍接头电阻（$\mathrm{n}\Omega$）'),
                      fontsize=7.0,labelpad=4.,fontfamily='SimSun' if cn else font)
        legend=ax.legend(loc='lower left',bbox_to_anchor=(.04,.04),fontsize=6.3,frameon=True,
                         borderpad=.30,labelspacing=.42,handlelength=2.2,handletextpad=.45,borderaxespad=.2)
        frame=legend.get_frame();frame.set_facecolor('white');frame.set_edgecolor('#DDDAD4')
        frame.set_linewidth(.55);frame.set_alpha(.95)
        fig.text(11.69*pt/width_mm,1-((200.81089+7.4)*pt+shift)/height_mm,'B',
                 ha='right',va='top',fontsize=8.5,fontweight='bold',color='black')
        heading=fig.text(18.78*pt/width_mm,1-((201.21714+7.4)*pt+shift)/height_mm,
                        lang('Conservative joint-resistance limit','保守接头电阻容限'),
                        ha='left',va='top',fontsize=7.7,fontweight='bold',color='black')
        if cn:heading.set_path_effects([path_effects.Stroke(linewidth=.32,foreground='black'),path_effects.Normal()])
    audit={'authority':'v7 approved_renderer_v17.py add_panel_b_overlay + palette_b.py',
           'renderer_source_sha256':SOURCE_SHA256,'palette_source_sha256':PALETTE_SHA256,
           'xscale':ax.get_xscale(),'yscale':ax.get_yscale(),'xlim':list(ax.get_xlim()),'ylim':list(ax.get_ylim()),
           'axes_size_mm':[right-left,bottom-top],'translation_y_mm':shift,'colors':COLORS,
           'curve_width_pt':1.8,'marker_size_pt':2.7,'grid_width_pt':.35,'fill_alpha':.035,
           'axis_label_pt':7.0,'tick_label_pt':6.5,
           'legend':'lower left; original v7 frame and spacing','first_valid_Npw':cutoffs,
           'display_Npw':[50,200],'onset_vertical_lines_shown':False,
           'missing_tolerance':'NaN kept on full ordered grid; no crossing of missing nodes',
           'upper_censoring':'curve rises to the 100-nOhm top edge at the censoring transition; no plateau or lower-bound markers',
           'annotation_precision':'one decimal (V8 Step 199)',
           'data_source':'Fig5B_conservative_boundaries.csv; all-rho, cross-scenario 10% criterion'}
    return ax,audit
