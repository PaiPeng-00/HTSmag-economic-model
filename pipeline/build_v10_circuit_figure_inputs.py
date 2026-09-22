"""Build Fig. 2B/C/D and Fig. 3C from the complete recomputed circuit cache."""
from pathlib import Path
import argparse
import json
import os
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault("FUSION_DEVICE","arc_16pancake_nuc600_v6_2")
sys.path.insert(0,str(ROOT/"model/model_code/code/src"))
from fusion_tem import device as cfg
from fusion_tem.economic.plant_availability import allocate_plant_annual_schedule

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    f=pd.read_csv(args.cache)
    keys=["Top_K","Npw","rho_turn_uOhm_cm2"]
    assert len(f)==36600 and not f.duplicated(keys).any()
    assert f.circuit_solver_status.eq("success").all()
    b=f.loc[f.Top_K.eq(10),keys+["Charging_time_999_h"]].copy()
    schedules = []
    for scenario in ("S1", "S2", "S3"):
        p = cfg.SCENARIO_DEFINITIONS[scenario]
        for row in f.itertuples(index=False):
            s = allocate_plant_annual_schedule(charge_hours=row.Charging_time_999_h,
                major_days_per_fpy=p["major_scheduled_days_per_fpy"], minor_days_per_fpy=p["minor_scheduled_days_per_fpy"],
                tf_cycles_per_year=p["tf_cycles_per_year"], cooldown_hours_per_cycle=p["tcool_h"],
                warmup_hours_per_cycle=p["twarm_h"], discharge_to_charge_ratio=p["kdis"],
                unplanned_unavailability=p["unplanned_unavailability"], pulse_hours_per_cycle=p["tau_pulse_h"],
                dwell_hours_per_cycle=p["tau_dwell_h"])
            schedules.append({"scenario": scenario, "Top_K": row.Top_K, "Npw": row.Npw,
                              "rho_turn_uOhm_cm2": row.rho_turn_uOhm_cm2, "Aplant": s.plant_availability,
                              "pulse_duty_factor": s.pulse_duty_factor, "CF_gross": s.gross_capacity_factor})
    schedules = pd.DataFrame(schedules)
    c = schedules.loc[schedules.Top_K.eq(10)].copy()
    d = []
    for (scenario, temp, rho), group in schedules.groupby(["scenario", "Top_K", "rho_turn_uOhm_cm2"]):
        passed = group.loc[group.Aplant.ge(.8)]
        d.append({"scenario": scenario, "Top_K": temp, "rho_turn_uOhm_cm2": rho,
                  "Npw_min_Aplant080": int(passed.Npw.min()) if len(passed) else np.nan,
                  "has_Aplant080": bool(len(passed)), "Aplant_at_Npw200": float(group.loc[group.Npw.eq(200), "Aplant"].iloc[0]),
                  "boundary_definition": "smallest_direct_Npw_with_Aplant_ge_0.80; no interpolation"})
    d = pd.DataFrame(d)

    radial=f.loc[f.Top_K.eq(10),["Npw","rho_turn_uOhm_cm2"]].copy()
    radial["radial_charging_energy_kWh_TF"]=f.loc[f.Top_K.eq(10),"Radial_loss_energy_MWh"].to_numpy()*1000
    fig2=out/"main_figures/Fig2"; fig3=out/"main_figures/Fig3"
    fig2.mkdir(parents=True); fig3.mkdir(parents=True)
    b.to_csv(fig2/"panel_B_charging_time.csv",index=False)
    c.to_csv(fig2/"panel_C_annual_availability.csv",index=False)
    d.to_csv(fig2/"panel_D_availability_boundary.csv",index=False)
    radial.to_parquet(fig3/"panel_C_radial_charging_loss.parquet",index=False)
    (out/"CIRCUIT_FIGURE_INPUT_AUDIT.json").write_text(json.dumps({
        "status":"PASS","circuit_cases":len(f),"Fig2B_C_temperature_K":10,
        "Fig3C_temperature_K":10,"Fig2D":"all three temperatures and scenarios",
        "radial_energy_unit":"kWh per TF magnet"
    },indent=2)+"\n",encoding="utf-8")

if __name__=="__main__": main()

