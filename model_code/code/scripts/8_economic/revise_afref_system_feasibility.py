"""Create a versioned Data S1 with the revised system-feasibility schema."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fusion_tem.economic.feasibility import (
    AVAILABILITY_THRESHOLDS,
    add_joint_lcoe_reference,
    apply_system_feasibility,
)


def build_summary(before: pd.DataFrame, after: pd.DataFrame) -> dict:
    summary: dict[str, object] = {
        "row_count": int(len(after)),
        "scenario_af_max": {
            key: float(value)
            for key, value in after.groupby("scenario")["AF_max_scenario"].first().items()
        },
        "availability_pass_counts": {},
        "joint_feasible_counts": {},
        "cross_check_af099_charge_gt_120h": {},
        "lcoe_reference_designs": {},
        "delta_lcoe_max_comparison": {},
    }

    for threshold in AVAILABILITY_THRESHOLDS:
        suffix = f"{int(round(100 * threshold)):03d}"
        availability = after[f"feasible_availability_{suffix}"]
        joint = (
            availability
            & after["feasible_cryo_050"]
            & after["finite_lcoe"]
            & after["no_model_error"]
        )
        summary["availability_pass_counts"][f"AF_ref_system>={threshold:.2f}"] = int(
            availability.sum()
        )
        summary["joint_feasible_counts"][f"AF_ref_system>={threshold:.2f}"] = int(
            joint.sum()
        )

    cross = after["feasible_availability_099"] & after["charging_time_warning_120h"]
    cross_rows = after.loc[cross]
    summary["cross_check_af099_charge_gt_120h"] = {
        "count": int(cross.sum()),
        "charging_time_h_min": float(cross_rows["Charging_time_999_h"].min()),
        "charging_time_h_max": float(cross_rows["Charging_time_999_h"].max()),
        "Npw_min": int(cross_rows["Npw"].min()),
        "Npw_max": int(cross_rows["Npw"].max()),
        "rho_turn_uOhm_cm2_min": float(cross_rows["rho_turn_uOhm_cm2"].min()),
        "rho_turn_uOhm_cm2_max": float(cross_rows["rho_turn_uOhm_cm2"].max()),
    }

    design_cols = [
        "Top_K",
        "coolant",
        "Npw",
        "rho_turn_uOhm_cm2",
        "R_joint_nOhm",
        "LCOE_plant_USD_per_MWh",
        "AF_ref_system",
        "r_cryo_re_fraction",
        "net_export_fraction",
        "Charging_time_999_h",
    ]
    for scenario, group in after.groupby("scenario", sort=True):
        joint = group.loc[group["feasible_joint"]]
        optimum = joint.loc[joint["LCOE_plant_USD_per_MWh"].idxmin(), design_cols]
        summary["lcoe_reference_designs"][scenario] = {
            key: value.item() if hasattr(value, "item") else value
            for key, value in optimum.items()
        }

        old_valid = before[
            (before["scenario"] == scenario)
            & np.isfinite(before["LCOE_plant_USD_per_MWh"])
            & np.isfinite(before["r_cryo_re"])
            & (before["r_cryo_re"] <= 50.0)
        ]
        old_min = float(old_valid["LCOE_plant_USD_per_MWh"].min())
        new_min = float(joint["LCOE_plant_USD_per_MWh"].min())
        summary["delta_lcoe_max_comparison"][scenario] = {
            "old_reference_USD_per_MWh": old_min,
            "new_reference_USD_per_MWh": new_min,
            "old_feasible_max_delta_USD_per_MWh": float(
                (old_valid["LCOE_plant_USD_per_MWh"] - old_min).max()
            ),
            "new_joint_feasible_max_delta_USD_per_MWh": float(
                (joint["LCOE_plant_USD_per_MWh"] - new_min).max()
            ),
        }

    fig_slice = (
        (after["rho_turn_uOhm_cm2"] == 10000.0)
        & (after["Npw"] >= 5)
    )
    summary["fig5_6_display_slice"] = {
        "rows": int(fig_slice.sum()),
        "availability_failures_at_099": int(
            (fig_slice & ~after["feasible_availability_099"]).sum()
        ),
        "cryo_050_failures": int(
            (fig_slice & ~after["feasible_cryo_050"]).sum()
        ),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--summary-json", type=Path, required=True)
    args = parser.parse_args()

    before = pd.read_csv(args.input_csv)
    after, _ = apply_system_feasibility(before)
    after, _ = add_joint_lcoe_reference(after)
    summary = build_summary(before, after)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    after.to_csv(args.output_csv, index=False)
    args.summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {len(after):,} rows to {args.output_csv}")
    print(f"Wrote summary to {args.summary_json}")


if __name__ == "__main__":
    main()
