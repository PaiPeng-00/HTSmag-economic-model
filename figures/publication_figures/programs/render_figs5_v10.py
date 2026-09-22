#!/usr/bin/env python3
"""Render V10 Fig. S5, preserving the approved cryogenic-power grid style."""
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
 "coil_joint":"Coil-to-coil joint Joule heat","internal_joint":"Internal tape-splice Joule heat",
 "magnetization":"Magnetisation loss","radial":"Radial loss","radiative":"Radiative heat",
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
 # V10's narrower 20-K pulse interval can contain only two coarse levels.
 # Add intermediate engineering levels without altering any plotted data.
 if len(candidates)<3:
  candidates=np.unique(np.concatenate([m*10.0**exponents for m in (1,1.5,2,2.5,3,4,5,6,8)]));candidates=candidates[(candidates>1.12*values.min())&(candidates<.92*values.max())]
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

def figs7_cryo_power_grid():
 d=pd.read_parquet(DATA/"supplementary_figures/FigS5/cryo_power_grid.parquet");fig,axs=plt.subplots(3,3,figsize=(166/MM,145/MM),sharex=True,sharey=True);fig.subplots_adjust(hspace=.48,wspace=.30,left=.10,right=.88,bottom=.14,top=.95)
 mode_cols=[("Pulse","P_cryo_prod_W"),("Dwell","P_cryo_dwell_W"),("Static","P_cryo_static_W")]
 vals=np.concatenate([d[c].to_numpy()/1e6 for _,c in mode_cols]);pos=vals[vals>0];norm=LogNorm(float(np.nanmin(pos)),float(np.nanmax(pos)));panel_audit={};static20_audit=None;panel_meshes=[]
 for i,t in enumerate([4.2,10,20]):
  for j,(mode,col) in enumerate(mode_cols):
   ax=axs[i,j];raw=d[d.Top_K.eq(t)].groupby(["R_joint_nOhm","Npw"],as_index=False)[col].median();raw[col]=raw[col]/1e6;xs,ys,a=grid(raw,"R_joint_nOhm","Npw",col)
   # Register every colour value at its authoritative (Rj, Npw) source node.
   # Gouraud shading interpolates only between those vertices and fills exactly
   # the closed source domain [1, 100] x [1, 200].  Do not manufacture N+1
   # edges inside the endpoint interval: that shifts Npw=1 to 1.4975 and moves
   # Rj=100 inward, even though the outer colour boundary appears correct.
   # Keep the QuadMesh vector in PDF/SVG.  Rasterizing at Matplotlib's default
   # 100 dpi rounds each panel to an integer-pixel image; the former PDF image
   # was shifted upward by 0.6825 pt and exceeded the top axis by 0.8495 pt,
   # versus only 0.3347 pt at the right edge.
   m=ax.pcolormesh(xs,ys,a,cmap=CM,norm=norm,shading="gouraud",rasterized=False,clip_on=True,antialiased=False)
   panel_meshes.append(m)
   mesh_coordinates=np.asarray(m.get_coordinates());expected_x=np.broadcast_to(xs[None,:],a.shape);expected_y=np.broadcast_to(ys[:,None],a.shape)
   registration_error=float(max(np.max(np.abs(mesh_coordinates[...,0]-expected_x)),np.max(np.abs(mesh_coordinates[...,1]-expected_y))))
   ax.set_xscale("log");ax.set_xlim(xs[0],xs[-1]);ax.set_ylim(ys[0],ys[-1]);ax.xaxis.set_minor_locator(NullLocator())
   lower_display=ax.transData.transform((xs[0],ys[0]));upper_display=ax.transData.transform((xs[-1],ys[-1]));bbox=ax.bbox
   boundary_display_error=float(max(abs(lower_display[0]-bbox.x0),abs(lower_display[1]-bbox.y0),abs(upper_display[0]-bbox.x1),abs(upper_display[1]-bbox.y1)))
   levels=engineering_contour_levels(a);ax.contour(xs,ys,a,levels=levels,colors="white",linewidths=1.9)
   contours=ax.contour(xs,ys,a,levels=levels,colors="black",linewidths=.65);label_count=label_contours_once(ax,contours,levels)
   key=chr(65+i*3+j);panel_audit[key]={"temperature_K":t,"mode":mode,"unit":"MW","contour_levels":levels,"contour_count":len(levels),"label_count":label_count,"shading":"gouraud","coordinate_semantics":"source values are colour vertices; interpolation is confined between adjacent source vertices","Rj_colour_vertex_min":float(mesh_coordinates[...,0].min()),"Rj_colour_vertex_max":float(mesh_coordinates[...,0].max()),"Npw_colour_vertex_min":float(mesh_coordinates[...,1].min()),"Npw_colour_vertex_max":float(mesh_coordinates[...,1].max()),"source_coordinate_registration_error":registration_error,"data_boundary_to_axes_boundary_error_px":boundary_display_error,"artist_clip_on":bool(m.get_clip_on()),"artist_clip_box_present":bool(m.get_clip_box() is not None)}
   if t==20 and mode=="Static":
    diffs=np.diff(a,axis=1);crossings=[]
    for x,zcol in zip(xs,a.T):
     if float(np.nanmin(zcol))<=.5<=float(np.nanmax(zcol)) and np.all(np.diff(zcol)>=0):crossings.append([float(x),float(np.interp(.5,zcol,ys))])
    static20_audit={"quantity":"P_cryo_static","unit":"MW","fixed_Npw_Rj_positive_step_count":int(np.sum(diffs>1e-12)),"fixed_Npw_Rj_negative_step_count":int(np.sum(diffs<-1e-12)),"Rj_dependence":"non-increasing","contour_level_MW":.5,"contour_crossing_Npw_at_Rj_1":crossings[0][1] if crossings else None,"contour_crossing_Npw_at_Rj_100":crossings[-1][1] if crossings else None,"contour_slope_sign":"positive" if crossings and crossings[-1][1]>crossings[0][1] else "non-positive","mechanism":"Static cold load excludes joint heat, but the frozen green_rated model uses one design-rated efficiency sized from the maximum simultaneous load. Increasing Rj raises rated capacity and eta_rated, so the same static cold load requires less electrical power; at constant electrical power this requires larger Npw."}
   panel(ax,key,f"{t:g} K, {mode}");style(ax)
 for ax in axs[-1,:]:ax.set_xlabel(r"Joint resistance, $R_{\mathrm{j}}$ (n$\Omega$)")
 for ax in axs[:,0]:ax.set_ylabel(r"Parallel tapes per turn, $N_{\mathrm{pw}}$");ax.set_yticks([1,50,100,150,200])
 cax=fig.add_axes([.90,.18,.018,.64]);ticks=[.1,1,10,100];cb=fig.colorbar(m,cax=cax,ticks=ticks);cb.ax.set_yticklabels(["0.1","1","10","100"]);cb.ax.yaxis.set_minor_locator(NullLocator());cb.set_label("TF refrigeration electrical power (MW)",fontsize=6.5)
 save(fig,"FigS5",rasterize_for_secondary=panel_meshes)
 pdf_doc=fitz.open(OUT/"FigS5.pdf");pdf_images=pdf_doc[0].get_images(full=True);pdf_doc.close();panel_raster_images=[image for image in pdf_images if image[2]>50 and image[3]>50]
 passed=all(3<=item["contour_count"]<=4 and item["label_count"]==item["contour_count"] and item["shading"]=="gouraud" and item["Rj_colour_vertex_min"]==1 and item["Rj_colour_vertex_max"]==100 and item["Npw_colour_vertex_min"]==1 and item["Npw_colour_vertex_max"]==200 and item["source_coordinate_registration_error"]==0 and item["data_boundary_to_axes_boundary_error_px"]<1e-9 and item["artist_clip_on"] and item["artist_clip_box_present"] for item in panel_audit.values()) and static20_audit is not None and static20_audit["fixed_Npw_Rj_positive_step_count"]==0 and len(panel_raster_images)==0
 audit={"status":"PASS" if passed else "FAIL","version":"V10","figure":"FigS5","backend":"Python","scientific_data_modified":False,"colour_unit":"MW","pdf_rendering_contract":{"panel_quadmesh":"vector","panel_raster_image_count":len(panel_raster_images),"total_raster_image_count":len(pdf_images),"remaining_raster_role":"colourbar only","superseded_rasterized_pdf_measurement_pt":{"axes_bbox":[34.5125,282.05612,136.465256,366.129134],"image_bbox":[34.56,282.738626,136.8,366.978626],"top_overshoot":0.849492,"right_overshoot":0.334744,"bottom_inset":0.682506,"left_inset":0.0475,"root_cause":"100-dpi rasterized-artist integer-pixel rounding produced an asymmetric PDF image transform"}},"colour_boundary_contract":{"Rj_min":1,"Rj_max":100,"Npw_min":1,"Npw_max":200,"mapping":"authoritative grid nodes are colour vertices; Gouraud interpolation occurs only inside the closed source domain","superseded_mapping":{"method":"N+1 edges forced inside endpoint interval","Npw_1_was_rendered_at":1.4975,"Npw_200_was_rendered_at":199.5025,"Rj_1_was_rendered_at":1.0192118462630158,"Rj_100_was_rendered_at":98.11502914399428,"effect":"vertical half-cell misregistration plus inward horizontal and vertical endpoint compression"}},"static_20K_semantic_audit":static20_audit,"panels":panel_audit}
 audit_path=ROOT/"audits/figures/FIGS5_CONTOUR_BOUNDARY_AUDIT_V6_7.json";audit_path.parent.mkdir(parents=True,exist_ok=True);audit_path.write_text(json.dumps(audit,indent=2)+"\n",encoding="utf-8")
 if not passed:raise RuntimeError("FIGS5_CONTOUR_BOUNDARY_AUDIT_FAIL")

if __name__ == "__main__":
 import argparse
 from publication_style import activate_style
 parser=argparse.ArgumentParser()
 parser.add_argument("--data",type=Path,required=True)
 parser.add_argument("--output",type=Path,required=True)
 args=parser.parse_args()
 DATA=args.data.resolve(); OUT=args.output.resolve(); ROOT=OUT
 activate_style("SI_DENSE","FigS7")
 figs7_cryo_power_grid()
