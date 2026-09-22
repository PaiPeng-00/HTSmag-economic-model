"""System-level availability and TF-cryogenic feasibility definitions.

This module is the single manuscript-facing implementation of the P0
feasibility boundary. Charging time and net-export fraction remain diagnostics;
neither replaces the explicit TF cryogenic recirculation criterion.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


PRIMARY_PLANT_AVAILABILITY_REQUIREMENT = 0.80
MAX_CRYO_RECIRCULATION_FRACTION = 0.50
MIN_NET_EXPORT_FRACTION = 0.20
CHARGING_TIME_WARNING_H = 120.0
PRIMARY_LCOE_COLUMN = "LCOE_plant_USD_per_MWh"
LEGACY_MAGNET_LCOE_COLUMN = "LCOE_magnet_only_USD_per_MWh"
LEGACY_FULLPLANT_LCOE_COLUMN = "LCOE_fullplant_legacy_USD_per_MWh"

def _finite(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return pd.Series(np.isfinite(values.to_numpy()), index=series.index)


def _cryo_fraction(out: pd.DataFrame) -> pd.Series:
    """Return canonical r_cryo,re as a fraction, preserving legacy percent data."""
    if "r_cryo_re_fraction" in out.columns:
        return pd.to_numeric(out["r_cryo_re_fraction"], errors="coerce")
    if "r_cryo_re" not in out.columns:
        raise KeyError("Missing feasibility column: r_cryo_re_fraction")
    # Historical Data S1 stores this legacy compatibility field in percent.
    return pd.to_numeric(out["r_cryo_re"], errors="coerce") / 100.0


def apply_system_feasibility(
    df: pd.DataFrame,
    *,
    availability_threshold: float = PRIMARY_PLANT_AVAILABILITY_REQUIREMENT,
    cryo_threshold: float = MAX_CRYO_RECIRCULATION_FRACTION,
    net_export_threshold: float = MIN_NET_EXPORT_FRACTION,
    charging_warning_h: float = CHARGING_TIME_WARNING_H,
) -> tuple[pd.DataFrame, pd.Series]:
    """Add the V7 absolute plant-availability and feasibility columns.

    ``feasible_joint`` requires ``Aplant >= 0.80``, TF refrigeration no
    greater than 50% of gross generation, finite LCOE, and no model error.
    The 120-h charging marker remains diagnostic only.
    """
    required = {
        "scenario",
        "Aplant",
        "E_net_year_MWh",
        "E_gross_year_MWh",
        "Charging_time_999_h",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise KeyError(f"Missing feasibility columns: {missing}")

    out = df.copy()
    status_error = (
        out["status"].astype(str).str.lower().eq("error")
        if "status" in out.columns
        else pd.Series(False, index=out.index)
    )
    out["model_error"] = status_error
    out["no_model_error"] = ~out["model_error"]

    aplant = pd.to_numeric(out["Aplant"], errors="coerce")
    valid_availability = _finite(aplant) & out["no_model_error"]
    availability_max = (
        pd.DataFrame({"scenario": out.loc[valid_availability, "scenario"], "Aplant": aplant[valid_availability]})
        .groupby("scenario", sort=True)["Aplant"]
        .max()
    )
    out["Aplant_max_scenario"] = out["scenario"].map(availability_max)
    out["feasible_plant_availability_080"] = (
        valid_availability & (aplant >= availability_threshold)
    )

    cryo_fraction = _cryo_fraction(out)
    out["r_cryo_re_fraction"] = cryo_fraction
    out["feasible_cryo_050"] = (
        _finite(cryo_fraction) & (cryo_fraction <= cryo_threshold)
    )
    # Compatibility alias used by earlier figure scripts.
    out["feasible_r_cryo_re"] = out["feasible_cryo_050"]

    gross = pd.to_numeric(out["E_gross_year_MWh"], errors="coerce")
    net = pd.to_numeric(out["E_net_year_MWh"], errors="coerce")
    valid_energy = _finite(gross) & _finite(net) & (gross > 0.0)
    out["net_export_fraction"] = np.where(valid_energy, net / gross, np.nan)
    out["feasible_net_export"] = (
        _finite(out["net_export_fraction"])
        & (out["net_export_fraction"] >= net_export_threshold)
    )

    lcoe_col = PRIMARY_LCOE_COLUMN
    if lcoe_col in out.columns:
        out["finite_lcoe"] = _finite(out[lcoe_col])
    else:
        out["finite_lcoe"] = True

    primary_col = "feasible_plant_availability_080"
    out["feasible_joint"] = (
        out[primary_col]
        & out["feasible_cryo_050"]
        & out["finite_lcoe"]
        & out["no_model_error"]
    )

    charge_time = pd.to_numeric(out["Charging_time_999_h"], errors="coerce")
    out["charging_time_warning_120h"] = (
        _finite(charge_time) & (charge_time > charging_warning_h)
    )

    # Manuscript-facing alias: feasible always means the joint P0 gate.
    out["feasible"] = out["feasible_joint"]
    out["feasibility_flags"] = ""
    out.loc[~out[primary_col], "feasibility_flags"] = "availability"
    cryo_fail = ~out["feasible_cryo_050"]
    out.loc[cryo_fail, "feasibility_flags"] = out.loc[
        cryo_fail, "feasibility_flags"
    ].map(lambda value: f"{value};cryo_050" if value else "cryo_050")
    out.loc[~out["finite_lcoe"], "feasibility_flags"] = out.loc[
        ~out["finite_lcoe"], "feasibility_flags"
    ].map(lambda value: f"{value};nonfinite_lcoe" if value else "nonfinite_lcoe")
    out.loc[out["model_error"], "feasibility_flags"] = out.loc[
        out["model_error"], "feasibility_flags"
    ].map(lambda value: f"{value};model_error" if value else "model_error")
    out.loc[out["charging_time_warning_120h"], "feasibility_flags"] = out.loc[
        out["charging_time_warning_120h"], "feasibility_flags"
    ].map(lambda value: f"{value};charge_time_warning" if value else "charge_time_warning")

    return out, availability_max


def add_joint_lcoe_reference(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Add scenario-specific plant-v2 references and auditable legacy deltas."""
    required = {"scenario", "feasible_joint", PRIMARY_LCOE_COLUMN}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise KeyError(f"Missing LCOE reference columns: {missing}")

    out = df.copy()
    lcoe = pd.to_numeric(out[PRIMARY_LCOE_COLUMN], errors="coerce")
    valid = out["feasible_joint"].astype(bool) & _finite(lcoe)
    references = out.loc[valid].groupby("scenario")[PRIMARY_LCOE_COLUMN].min()
    missing_scenarios = sorted(set(out["scenario"].dropna().unique()) - set(references.index))
    if missing_scenarios:
        raise ValueError(f"No jointly feasible LCOE reference for: {missing_scenarios}")

    out["LCOE_reference_joint_feasible"] = out["scenario"].map(references)
    out["delta_LCOE_joint_feasible_USD_per_MWh"] = (
        lcoe - out["LCOE_reference_joint_feasible"]
    )
    out["delta_LCOE_min_USD_per_MWh"] = out[
        "delta_LCOE_joint_feasible_USD_per_MWh"
    ]

    legacy_specs = (
        (
            LEGACY_MAGNET_LCOE_COLUMN,
            "LCOE_reference_magnet_only_legacy_USD_per_MWh",
            "delta_LCOE_magnet_only_legacy_USD_per_MWh",
        ),
        (
            LEGACY_FULLPLANT_LCOE_COLUMN,
            "LCOE_reference_fullplant_legacy_USD_per_MWh",
            "delta_LCOE_fullplant_legacy_USD_per_MWh",
        ),
    )
    for column, reference_column, delta_column in legacy_specs:
        if column not in out.columns:
            continue
        values = pd.to_numeric(out[column], errors="coerce")
        legacy_valid = out["feasible_joint"].astype(bool) & _finite(values)
        legacy_reference = out.loc[legacy_valid].groupby("scenario")[column].min()
        out[reference_column] = out["scenario"].map(legacy_reference)
        out[delta_column] = values - out[reference_column]

    return out, references
