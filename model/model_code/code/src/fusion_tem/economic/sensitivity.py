"""One-at-a-time non-TF auxiliary energy-boundary sensitivity."""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from fusion_tem.economic.feasibility import (
    add_joint_lcoe_reference,
    apply_system_feasibility,
)


OTHER_AUX_FRACTIONS = (0.20, 0.25, 0.30, 0.35)


def _require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = sorted(set(columns).difference(df.columns))
    if missing:
        raise KeyError(f"Missing sensitivity columns: {missing}")


def recalculate_energy_boundary(
    df: pd.DataFrame,
    *,
    other_aux_fraction: float | None = None,
) -> pd.DataFrame:
    """Recalculate net energy and plant-v2 LCOE for a new non-TF fraction.

    The TF cryogenic energy is held fixed. Charge and discharge are defined as
    identical full load profiles, so there is no discharge scaling sensitivity.
    Capital and non-coolant operating costs remain unchanged.
    """
    required = {
        "E_gross_year_MWh",
        "E_other_year_MWh",
        "E_cryo_year_MWh",
        "E_net_year_MWh",
        "LCOE_plant_USD_per_MWh",
    }
    _require_columns(df, required)
    out = df.copy()

    gross = pd.to_numeric(out["E_gross_year_MWh"], errors="coerce")
    old_net = pd.to_numeric(out["E_net_year_MWh"], errors="coerce")
    old_lcoe = pd.to_numeric(out["LCOE_plant_USD_per_MWh"], errors="coerce")
    annual_cost = old_lcoe * old_net
    cryo = pd.to_numeric(out["E_cryo_year_MWh"], errors="coerce")

    if other_aux_fraction is None:
        other = pd.to_numeric(out["E_other_year_MWh"], errors="coerce")
        r_other = pd.to_numeric(out.get("r_other", np.nan), errors="coerce")
    else:
        r_value = float(other_aux_fraction)
        if not 0 <= r_value < 1:
            raise ValueError("other_aux_fraction must lie in [0, 1)")
        r_other = pd.Series(r_value, index=out.index, dtype=float)
        other = r_value * gross

    net = gross - other - cryo
    lcoe = np.where(net > 0.0, annual_cost / net, np.nan)

    out["r_other"] = r_other
    out["E_other_year_MWh"] = other
    out["E_net_year_MWh"] = net
    out["LCOE_plant_USD_per_MWh"] = lcoe
    out["r_cryo_re_fraction"] = np.where(gross > 0.0, cryo / gross, np.nan)
    out["r_cryo_re"] = 100.0 * out["r_cryo_re_fraction"]
    out["energy_closure_error_MWh"] = gross - other - cryo - net
    return out


def _finalize_sensitivity_case(
    df: pd.DataFrame,
    *,
    sensitivity_type: str,
    sensitivity_value: float,
) -> pd.DataFrame:
    out, _ = apply_system_feasibility(df)
    out, _ = add_joint_lcoe_reference(out)
    out["sensitivity_type"] = sensitivity_type
    out["sensitivity_value"] = float(sensitivity_value)
    return out


def run_other_aux_sensitivity(
    df: pd.DataFrame,
    *,
    values: Iterable[float] = OTHER_AUX_FRACTIONS,
) -> pd.DataFrame:
    cases = []
    for value in values:
        case = recalculate_energy_boundary(df, other_aux_fraction=float(value))
        cases.append(
            _finalize_sensitivity_case(
                case,
                sensitivity_type="r_other",
                sensitivity_value=float(value),
            )
        )
    return pd.concat(cases, ignore_index=True)