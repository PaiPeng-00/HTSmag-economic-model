"""Incremental-anchor capital accounting for the ARC economic model.

The module contains economic post-processing only.  It deliberately does not
import or modify electromagnetic, charging, loss, or cryogenic calculations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


PRICE_BASIS_YEAR = 2025
CAPITAL_BOUNDARY_VERSION = "incremental_anchor_v1_2025usd"
REFERENCE_MAGNET_ID = "arc16_T10K_He_Npw19_rhoturn10000_Rj1nOhm"

WADE_TOTAL_CAPITAL_2010_MUSD = 4220.6
WADE_COIL_DIRECT_2010_MUSD = 264.4
PRICE_FACTOR_2010_TO_2025 = 1.4764234875

HTS_CAPITALIZATION_FACTORS = (1.5, 1.15, 1.099, 1.5)
PS_CAPITALIZATION_FACTORS = (1.1, 1.15, 1.099, 1.5)
M_HTS = 2.8436625
M_PS = 2.0853525
M_COOLANT_FILL = 1.0

MATURITY_SCALE_BY_SCENARIO: Mapping[str, float] = {
    "S1": 1.0,
    "S2": 2.0 / 3.0,
    "S3": 1.0 / 3.0,
    "S4": 2.0 / 3.0,
    "S5": 2.0 / 3.0,
    "S6": 2.0 / 3.0,
}

DISCOUNT_RATE_BY_SCENARIO: Mapping[str, float] = {
    "S1": 0.10, "S2": 0.07, "S3": 0.03,
    "S4": 0.07, "S5": 0.07, "S6": 0.07,
}
PROJECT_YEARS_BY_SCENARIO: Mapping[str, int] = {
    "S1": 20, "S2": 30, "S3": 40,
    "S4": 30, "S5": 30, "S6": 30,
}
HTS_PRICE_BY_SCENARIO: Mapping[str, float] = {
    "S1": 100.0, "S2": 50.0, "S3": 10.0,
    "S4": 100.0, "S5": 50.0, "S6": 10.0,
}
CORE_VOM_BY_SCENARIO: Mapping[str, float] = {
    "S1": 5.28, "S2": 3.17, "S3": 1.06,
    "S4": 3.17, "S5": 3.17, "S6": 3.17,
}

HELIUM_PRICE_USD_PER_KG = 175.0
HYDROGEN_PRICE_USD_PER_KG = 10.0
COOLANT_REPLENISH_FRACTION = 0.25
POWER_SUPPLY_PRICE_USD_PER_A = 47.59
NON_TF_AUX_FRACTION = 0.32
PCS_CAPITAL_USD_PER_KWE = 792.4
PCS_FOM_USD_PER_KWE_YEAR = 19.81
PCS_VOM_USD_PER_MWH_E = 1.84
GROSS_ELECTRIC_POWER_MWE = 283.2


@dataclass(frozen=True)
class CapitalAnchor:
    scenario: str
    maturity_scale: float
    full_anchor_usd: float
    capitalized_reference_coil_account_usd: float
    noncoil_background_usd: float


@dataclass(frozen=True)
class CapitalizedIncrement:
    delta_hts_direct_usd: float
    delta_hts_capitalized_usd: float
    delta_ps_direct_usd: float
    delta_ps_capitalized_usd: float
    delta_coolant_fill_usd: float
    delta_mag_anchor_usd: float


def _product(values: tuple[float, ...]) -> float:
    result = 1.0
    for value in values:
        result *= value
    return result


def validate_frozen_multipliers() -> None:
    if abs(_product(HTS_CAPITALIZATION_FACTORS) - M_HTS) >= 1e-12:
        raise AssertionError("HTS capitalization multiplier drifted")
    if abs(_product(PS_CAPITALIZATION_FACTORS) - M_PS) >= 1e-12:
        raise AssertionError("power-supply capitalization multiplier drifted")


def build_capital_anchor(scenario: str) -> CapitalAnchor:
    """Recompute a scenario anchor from the frozen Wade-Leuer primitives."""
    validate_frozen_multipliers()
    scale = float(MATURITY_SCALE_BY_SCENARIO[scenario])
    full_s1 = WADE_TOTAL_CAPITAL_2010_MUSD * 1e6 * PRICE_FACTOR_2010_TO_2025
    coil_s1 = (
        WADE_COIL_DIRECT_2010_MUSD * 1e6 * M_HTS
        * PRICE_FACTOR_2010_TO_2025
    )
    full = full_s1 * scale
    coil = coil_s1 * scale
    return CapitalAnchor(
        scenario=scenario,
        maturity_scale=scale,
        full_anchor_usd=full,
        capitalized_reference_coil_account_usd=coil,
        noncoil_background_usd=full - coil,
    )


def capital_recovery_factor(rate: float, years: float) -> float:
    if rate <= 0.0:
        return 1.0 / years
    growth = (1.0 + rate) ** years
    return rate * growth / (growth - 1.0)


def compute_capitalized_magnet_increment(
    *,
    hts_direct_usd: float,
    hts_reference_direct_usd: float,
    ps_direct_usd: float,
    ps_reference_direct_usd: float,
    coolant_fill_usd: float,
    coolant_reference_fill_usd: float,
) -> CapitalizedIncrement:
    delta_hts = float(hts_direct_usd) - float(hts_reference_direct_usd)
    delta_ps = float(ps_direct_usd) - float(ps_reference_direct_usd)
    delta_coolant = float(coolant_fill_usd) - float(coolant_reference_fill_usd)
    delta_hts_cap = M_HTS * delta_hts
    delta_ps_cap = M_PS * delta_ps
    delta_coolant_cap = M_COOLANT_FILL * delta_coolant
    return CapitalizedIncrement(
        delta_hts_direct_usd=delta_hts,
        delta_hts_capitalized_usd=delta_hts_cap,
        delta_ps_direct_usd=delta_ps,
        delta_ps_capitalized_usd=delta_ps_cap,
        delta_coolant_fill_usd=delta_coolant_cap,
        delta_mag_anchor_usd=delta_hts_cap + delta_ps_cap + delta_coolant_cap,
    )


def coolant_price_usd_per_kg(coolant: str) -> float:
    if coolant == "He":
        return HELIUM_PRICE_USD_PER_KG
    if coolant == "H2":
        return HYDROGEN_PRICE_USD_PER_KG
    raise KeyError(f"unsupported coolant: {coolant}")


def annual_pcs_fom_usd() -> float:
    return GROSS_ELECTRIC_POWER_MWE * 1000.0 * PCS_FOM_USD_PER_KWE_YEAR

