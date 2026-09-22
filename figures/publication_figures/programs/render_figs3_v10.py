#!/usr/bin/env python3
"""Render V10 Fig. S3, preserving the approved heat-history layout and style."""
from __future__ import annotations

import base64, hashlib, json, os, shutil, subprocess
from pathlib import Path

import fitz
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from matplotlib.patches import Patch
from matplotlib.ticker import NullLocator

ROOT=Path("."); DATA=ROOT; OUT=ROOT
MM=25.4; BLUE="#3A7B9B"; GREEN="#5F7D3A"; PURPLE="#8D6A9F"; ORANGE="#D79A42"; GREY="#DCE3EA"
CM=LinearSegmentedColormap.from_list("arc",["#F4F7FA","#D7EAF2","#A9D7DE","#68A8B5","#286D86"])
# Historical formal Supplementary palette and legend order.  The V6.7 source
# keys are mapped onto the previously approved heat-load visual vocabulary.
HEAT_SOURCE_COLORS={
 "coil_joint":"#CCE092","internal_joint":"#8DAFDB",
 "magnetization":"#FFD865","radial":"#B4C7E7","radiative":"#EDEDED",
 "lead":"#2F5597","coolant_pipe":"#F8B8CC",
 "auxiliary_pipe":"#FBE0EA","other_background":"#D3C6F1",
}
HEAT_SOURCE_LABELS={
 "coil_joint":"Full-current joint Joule heat","internal_joint":"Internal tape-splice Joule heat",
 "magnetization":"Magnetization loss","radial":"Radial loss","radiative":"Radiative heat",
 "lead":"Conductive heat - HTS current leads",
 "coolant_pipe":"Conductive heat - coolant transfer lines",
 "auxiliary_pipe":"Conductive heat - auxiliary leads","other_background":"Other sources",
}
mpl.rcParams.update({"font.family":"Arial","font.sans-serif":["Arial","Helvetica","DejaVu Sans"],
 "font.size":7,"pdf.fonttype":42,"svg.fonttype":"none","axes.linewidth":.55,"savefig.facecolor":"white"})

def sha(p):
 h=hashlib.sha256();
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def panel(ax,letter,title):
 ax.text(-.16,1.13,letter,transform=ax.transAxes,fontsize=9,fontweight="bold",va="top")
 ax.text(-.01,1.13,title,transform=ax.transAxes,fontsize=7.8,fontweight="bold",va="top")
def style(ax):
 ax.spines[["top","right"]].set_visible(False);ax.tick_params(length=2.5,width=.5,labelsize=6.2,pad=1.5)
def save(fig,name,tight=True,rasterize_for_secondary=None):
 OUT.mkdir(parents=True,exist_ok=True); pdf=OUT/f"{name}.pdf";svg=OUT/f"{name}.svg";png=OUT/f"{name}.png"
 kwargs={"bbox_inches":"tight"} if tight else {}
 fig.savefig(pdf,**kwargs)
 for artist in rasterize_for_secondary or []:artist.set_rasterized(True)
 fig.savefig(svg,dpi=600,**kwargs)
 fig.savefig(png,dpi=600,**kwargs);plt.close(fig)
 doc=fitz.open(pdf);pix=doc[0].get_pixmap(matrix=fitz.Matrix(2,2),alpha=False);pix.save(OUT/f"{name}_preview.png");doc.close()
 return {"pdf":sha(pdf),"svg":sha(svg),"png":sha(png)}
def grid(df,x,y,z):
 xs=np.sort(df[x].unique());ys=np.sort(df[y].unique());a=df.pivot(index=y,columns=x,values=z).reindex(index=ys,columns=xs).to_numpy();return xs,ys,a
def bounded_edges(centres,log=False):
 centres=np.asarray(centres,dtype=float);edges=np.empty(centres.size+1,dtype=float)
 edges[1:-1]=np.sqrt(centres[:-1]*centres[1:]) if log else .5*(centres[:-1]+centres[1:])
 edges[0]=centres[0];edges[-1]=centres[-1]
 return edges
def engineering_contour_levels(values,count=4):
 values=np.asarray(values,dtype=float);values=values[np.isfinite(values)&(values>0)]
 lo,hi=np.quantile(values,[.12,.90]);exponents=np.arange(np.floor(np.log10(values.min()))-1,np.ceil(np.log10(values.max()))+1)
 candidates=np.unique(np.concatenate([m*10.0**exponents for m in (1,2,3,5)]));candidates=candidates[(candidates>1.12*values.min())&(candidates<.92*values.max())]
 targets=np.geomspace(lo,hi,count);levels=[]
 for target in targets:
  level=float(candidates[np.argmin(np.abs(np.log(candidates)-np.log(target)))])
  if level not in levels:levels.append(level)
 if len(levels)<3:
  indices=np.linspace(0,len(candidates)-1,min(4,len(candidates))).round().astype(int);levels=[float(v) for v in candidates[indices]]
 return levels[:4]
def format_engineering_level(level):
 level=float(level)
 if level>=1000:return f"{level:,.0f}"
 if level>=1:return f"{level:g}"
 return f"{level:.2f}".rstrip("0").rstrip(".")
def label_contours_once(ax,contours,levels):
 fractions=np.linspace(.60,.32,len(levels));placed=0
 for index,(level,fraction) in enumerate(zip(levels,fractions)):
  raw=contours.allsegs[index];segments=[raw] if isinstance(raw,np.ndarray) and raw.ndim==2 else [segment for segment in raw if len(segment)>2]
  if not segments:continue
  segment=max(segments,key=len);display=ax.transData.transform(segment);steps=np.linalg.norm(np.diff(display,axis=0),axis=1);distance=np.r_[0,np.cumsum(steps)]
  target=fraction*distance[-1];bbox=ax.bbox;valid=np.where((display[:,0]>=bbox.x0+28)&(display[:,0]<=bbox.x1-28)&(display[:,1]>=bbox.y0+20)&(display[:,1]<=bbox.y1-32))[0]
  position=int(valid[np.argmin(np.abs(distance[valid]-target))]) if valid.size else int(np.clip(np.searchsorted(distance,target),1,len(segment)-2));position=int(np.clip(position,1,len(segment)-2));point=segment[position]
  tangent=display[position+1]-display[position-1];angle=np.degrees(np.arctan2(tangent[1],tangent[0]));angle=angle-180 if angle>90 else angle+180 if angle<-90 else angle
  label=ax.text(point[0],point[1],format_engineering_level(level),ha="center",va="center",rotation=angle,rotation_mode="anchor",fontsize=5.2,color="black",zorder=6,clip_on=True)
  label.set_path_effects([pe.Stroke(linewidth=2.0,foreground="white"),pe.Normal()]);placed+=1
 return placed
def heat(ax,df,x,y,z,logx=False,logc=False,levels=None):
 xs,ys,a=grid(df,x,y,z); norm=LogNorm(max(np.nanmin(a[a>0]),1e-9),np.nanmax(a)) if logc else Normalize(np.nanmin(a),np.nanmax(a))
 m=ax.pcolormesh(xs,ys,a,cmap=CM,norm=norm,shading="nearest",rasterized=True)
 if levels:
  ax.contour(xs,ys,a,levels=levels,colors="white",linewidths=2.2)
  c=ax.contour(xs,ys,a,levels=levels,colors="black",linewidths=.8)
  labs=ax.clabel(c,fmt="%g",fontsize=5.7,inline=False)
  for lab in labs:lab.set_path_effects([pe.Stroke(linewidth=1.5,foreground="white"),pe.Normal()])
 if logx:ax.set_xscale("log");ax.xaxis.set_minor_locator(NullLocator())
 style(ax);return m

def _charge_history_figure(data,name):
 fig=plt.figure(figsize=(160/MM,157.776402/MM),facecolor="white")
 secondary_raster_artists=[]
 panel_rects=[(.10,.72,.15,.19),(.425,.72,.15,.19),(.75,.72,.15,.19),(.08,.47,.15,.19),(.30,.47,.15,.19),(.52,.47,.15,.19),(.74,.47,.15,.19),(.08,.22,.15,.19),(.30,.22,.15,.19),(.52,.22,.15,.19),(.74,.22,.15,.19)]
 configs=sorted(data.configuration_id.unique());present=[s for s in HEAT_SOURCE_COLORS if s in set(data.heat_source)]
 xmax=float(data.display_end_h.iloc[0]) if "display_end_h" in data else float(data.time_h.max())
 for rect,cfg,letter in zip(panel_rects,configs,"ABCDEFGHIJK"):
  ax=fig.add_axes(rect);g=data[data.configuration_id.eq(cfg)]
  q=g.groupby(["time_h","heat_source"],as_index=False).heat_load_W.first();pv=q.pivot(index="time_h",columns="heat_source",values="heat_load_W").fillna(0.0)
  cols=[c for c in present if c in pv.columns];polys=ax.stackplot(pv.index,[pv[c].to_numpy() for c in cols],colors=[HEAT_SOURCE_COLORS[c] for c in cols],linewidth=0,alpha=.8)
  # Keep the formal PDF as vector artwork.  Rasterizing each small panel at
  # Matplotlib's default 100 dpi made the stacked areas and total curve visibly
  # blurred.  Only the secondary SVG/PNG exports rasterize the dense fills, at
  # the explicit 600-dpi export resolution used by save().
  secondary_raster_artists.extend(polys)
  total=pv[cols].sum(axis=1);ax.plot(pv.index,total,color="black",ls="--",lw=.7)
  ramp_end=float(g.current_ramp_end_h.iloc[0]) if "current_ramp_end_h" in g else 96.0
  ax.axvline(ramp_end,color="gray",ls="-.",lw=.55)
  ax.set_xlim(0,xmax);ax.set_xticks([0,50,100]);ax.set_ylim(0,float(total.max())*1.06);ax.grid(True,ls=":",lw=.35,color="#B8B8B8")
  ax.set_xlabel("Time (h)",fontsize=6.2,labelpad=1);ax.set_ylabel("Heat load (W)",fontsize=6.2,labelpad=1)
  npw=int(g.Npw.iloc[0]);rho=float(g.rho_turn_uOhm_cm2.iloc[0]);rho_label=f"{rho:,.0f}"
  ax.set_title(f"Npw={npw}; ρturn={rho_label}",fontsize=5.4,pad=2.0)
  ax.tick_params(length=2,width=.45,labelsize=5.8,pad=1);ax.text(-.32,1.16,letter,transform=ax.transAxes,fontsize=9,fontweight="bold",va="top")
 handles=[Patch(facecolor=HEAT_SOURCE_COLORS[c],edgecolor="none",alpha=.8,label=HEAT_SOURCE_LABELS[c]) for c in present]
 fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.5,.015),ncol=2,frameon=False,fontsize=6.3,columnspacing=1.8,handlelength=2.1)
 save(fig,name,tight=False,rasterize_for_secondary=secondary_raster_artists)

if __name__ == "__main__":
 import argparse
 from publication_style import activate_style
 parser=argparse.ArgumentParser()
 parser.add_argument("--data",type=Path,required=True)
 parser.add_argument("--output",type=Path,required=True)
 args=parser.parse_args()
 DATA=args.data.resolve(); OUT=args.output.resolve(); ROOT=OUT
 activate_style("SI_DENSE","FigS3")
 _charge_history_figure(pd.read_parquet(DATA / "supplementary_figures/FigS3/charging_heatload_Rj10.parquet"), "FigS3")
