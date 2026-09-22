"""Axis typography at the final 166-mm manuscript text width.

Keep each approved canvas/layout. Compensate for its placement scale so the
printed axis labels are 7 pt and ticks 6.5 pt, including Fig. 5 panel B.
"""
TEXT_WIDTH_MM=166.0
AXIS_LABEL_PT=7.0
TICK_LABEL_PT=6.5

def apply_manuscript_axis_fonts(fig, canvas_width_mm, placement_fraction=1.0):
    scale=canvas_width_mm/(TEXT_WIDTH_MM*placement_fraction)
    axes=[]
    for ax in fig.axes:
        if not ax.axison or ax.get_label()=='<colorbar>':
            continue
        ax.xaxis.label.set_fontsize(AXIS_LABEL_PT*scale)
        ax.yaxis.label.set_fontsize(AXIS_LABEL_PT*scale)
        ax.tick_params(axis='both',which='both',labelsize=TICK_LABEL_PT*scale)
        axes.append(ax)
    return {'axis_label_printed_pt':AXIS_LABEL_PT,'tick_label_printed_pt':TICK_LABEL_PT,
            'source_font_scale':scale,'canvas_width_mm':canvas_width_mm,
            'placement_fraction':placement_fraction,'axes_count':len(axes)}
