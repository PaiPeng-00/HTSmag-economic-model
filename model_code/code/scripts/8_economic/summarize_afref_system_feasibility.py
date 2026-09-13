"""Export manuscript-facing validation tables for revised feasibility."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


THRESHOLDS = (0.90, 0.95, 0.99)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv, low_memory=False)

    sensitivity_rows = []
    for scenario, group in df.groupby("scenario", sort=True):
        af_max = float(group["AF_max_scenario"].iloc[0])
        for threshold in THRESHOLDS:
            suffix = f"{int(round(100 * threshold)):03d}"
            availability = group[f"feasible_availability_{suffix}"].astype(bool)
            joint = (
                availability
                & group["feasible_cryo_050"].astype(bool)
                & group["finite_lcoe"].astype(bool)
                & group["no_model_error"].astype(bool)
            )
            feasible = group.loc[joint]
            optimum = feasible.loc[feasible["LCOE_plant_USD_per_MWh"].idxmin()]
            sensitivity_rows.append(
                {
                    "scenario": scenario,
                    "AF_threshold": threshold,
                    "AF_max_scenario": af_max,
                    "availability_pass_count": int(availability.sum()),
                    "joint_feasible_count": int(joint.sum()),
                    "optimum_Top_K": float(optimum["Top_K"]),
                    "optimum_coolant": optimum["coolant"],
                    "optimum_Npw": int(optimum["Npw"]),
                    "optimum_rho_turn_uOhm_cm2": float(optimum["rho_turn_uOhm_cm2"]),
                    "optimum_R_joint_nOhm": float(optimum["R_joint_nOhm"]),
                    "optimum_LCOE_USD_per_MWh": float(optimum["LCOE_plant_USD_per_MWh"]),
                    "feasible_Npw_min": int(feasible["Npw"].min()),
                    "feasible_Npw_max": int(feasible["Npw"].max()),
                    "feasible_R_joint_max_nOhm": float(feasible["R_joint_nOhm"].max()),
                }
            )
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(args.output_dir / "afref_threshold_sensitivity_by_scenario.csv", index=False)

    cross = df[
        df["feasible_availability_099"].astype(bool)
        & df["charging_time_warning_120h"].astype(bool)
    ]
    cross_table = cross.groupby("scenario", sort=True).agg(
        design_count=("Npw", "size"),
        charging_time_h_min=("Charging_time_999_h", "min"),
        charging_time_h_max=("Charging_time_999_h", "max"),
        Npw_min=("Npw", "min"),
        Npw_max=("Npw", "max"),
        rho_turn_uOhm_cm2_min=("rho_turn_uOhm_cm2", "min"),
        rho_turn_uOhm_cm2_max=("rho_turn_uOhm_cm2", "max"),
        R_joint_nOhm_min=("R_joint_nOhm", "min"),
        R_joint_nOhm_max=("R_joint_nOhm", "max"),
    ).reset_index()
    cross_table.to_csv(args.output_dir / "afref099_charge120_crosscheck.csv", index=False)

    old = (
        (df["Charging_time_999_h"] <= 120.0)
        & (df["r_cryo_re"] <= 50.0)
    )
    window_rows = []
    for definition, mask in (("old_charge120_and_rcryo50", old), ("new_joint", df["feasible_joint"])):
        for scenario, group in df.loc[mask].groupby("scenario", sort=True):
            window_rows.append(
                {
                    "definition": definition,
                    "scenario": scenario,
                    "design_count": int(len(group)),
                    "Npw_min": int(group["Npw"].min()),
                    "Npw_max": int(group["Npw"].max()),
                    "rho_turn_uOhm_cm2_min": float(group["rho_turn_uOhm_cm2"].min()),
                    "rho_turn_uOhm_cm2_max": float(group["rho_turn_uOhm_cm2"].max()),
                    "R_joint_nOhm_min": float(group["R_joint_nOhm"].min()),
                    "R_joint_nOhm_max": float(group["R_joint_nOhm"].max()),
                }
            )
    pd.DataFrame(window_rows).to_csv(
        args.output_dir / "old_new_feasible_design_windows.csv", index=False
    )

    ref_rows = []
    for scenario, group in df.groupby("scenario", sort=True):
        joint = group.loc[group["feasible_joint"].astype(bool)]
        optimum = joint.loc[joint["LCOE_plant_USD_per_MWh"].idxmin()]
        # The historical delta-LCOE plotting reference used only r_cryo_re<=50%.
        old_group = group.loc[
            group["LCOE_plant_USD_per_MWh"].notna()
            & (group["r_cryo_re"] <= 50.0)
        ]
        old_min = float(old_group["LCOE_plant_USD_per_MWh"].min())
        new_min = float(optimum["LCOE_plant_USD_per_MWh"])
        ref_rows.append(
            {
                "scenario": scenario,
                "Top_K": float(optimum["Top_K"]),
                "coolant": optimum["coolant"],
                "Npw": int(optimum["Npw"]),
                "rho_turn_uOhm_cm2": float(optimum["rho_turn_uOhm_cm2"]),
                "R_joint_nOhm": float(optimum["R_joint_nOhm"]),
                "AF_ref_system": float(optimum["AF_ref_system"]),
                "net_export_fraction": float(optimum["net_export_fraction"]),
                "charging_time_h": float(optimum["Charging_time_999_h"]),
                "old_reference_LCOE_USD_per_MWh": old_min,
                "new_reference_LCOE_USD_per_MWh": new_min,
                "old_feasible_max_delta_USD_per_MWh": float(
                    (old_group["LCOE_plant_USD_per_MWh"] - old_min).max()
                ),
                "new_joint_max_delta_USD_per_MWh": float(
                    (joint["LCOE_plant_USD_per_MWh"] - new_min).max()
                ),
            }
        )
    pd.DataFrame(ref_rows).to_csv(
        args.output_dir / "joint_lcoe_reference_designs.csv", index=False
    )

    print(f"Validation tables written to {args.output_dir}")


if __name__ == "__main__":
    main()
