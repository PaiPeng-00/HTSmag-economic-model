"""Auditable plant-level economic boundary for the ARC-class model.

This module contains only financial and annual-energy accounting. It does not
change the electromagnetic, cryogenic, availability, or feasibility models.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from fusion_tem.economic.price_basis import (
    ANNUAL_COOLANT_REPLENISH_FRACTION,
    BACKGROUND_CAPITAL_USD_BY_SCENARIO,
    CORE_VOM_USD_PER_MWH_TH_BY_SCENARIO,
    HTS_PRICE_2025_USD_PER_KAM_BY_SCENARIO,
    HTS_PRICE_CONVERSION_METHOD,
    PCS_CAPITAL_COST_USD_PER_KWE,
    PCS_FOM_FRACTION_PER_YEAR,
    PCS_VOM_USD_PER_MWH_E,
    PRICE_BASIS_YEAR,
)


COST_BOUNDARY_VERSION = "plant_v4_core_pcs_coolant_2025usd_direct_hts"

# 电厂功率量的唯一来源是器件配置(configs/devices/<device>.yaml -> device.py)。
# 2026-07-26 前这三个常量在本模块硬编码, 与 device.py 的 P_fusion_W / eat_conv /
# fusion_output_MWth 语义重复: 改 yaml 而不改这里就会静默不一致。见工作文档 §14.45。
#
# 这里直接读器件 yaml, 而**不** import device.py: device.py 在 import 时读环境变量
# FUSION_DEVICE, 若本模块先把它导进来, 器件配置会退回默认值; 而模块导入顺序由
# 调用方(含 unittest 的字母序发现)决定, 不可控。直接读 yaml 没有这个耦合。
import os as _os  # noqa: E402
from pathlib import Path as _Path  # noqa: E402


def _load_device_yaml() -> dict:
    device = _os.environ.get("FUSION_DEVICE", "arc").lower()
    root = _Path(__file__).resolve().parents[3]      # -> code/
    for base in (root / "configs", root.parent / "configs"):
        path = base / "devices" / f"{device}.yaml"
        if path.is_file():
            try:
                import yaml
            except ImportError:
                return {}
            return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {}


_DEVICE_CFG = _load_device_yaml()
FUSION_POWER_MWTH = float(_DEVICE_CFG.get("P_fusion_W", 525e6)) / 1e6
_EAT_CONV = float(_DEVICE_CFG.get("eat_conv", 0.40))
# 包层后热功率与聚变热功率的比例沿用 ARC 的 708/525
THERMAL_POWER_AFTER_BLANKET_MWTH = float(
    _DEVICE_CFG.get("FUSION_OUTPUT_MWTH", round(FUSION_POWER_MWTH * (708.0 / 525.0), 1))
)
GROSS_ELECTRIC_POWER_MWE = THERMAL_POWER_AFTER_BLANKET_MWTH * _EAT_CONV


def validate_price_only_scenarios(
    *,
    time_scenarios: Mapping[str, Mapping[str, Any]],
    project_lifetime_years: Mapping[str, float],
    coolant_prices: Mapping[str, Mapping[str, float]],
    hts_prices: Mapping[str, float],
    discount_rate_by_scenario: Mapping[str, float],
) -> None:
    """Require S4-S6 to differ from S2 only in HTS conductor price."""
    for scenario in ("S4", "S5", "S6"):
        if time_scenarios[scenario] != time_scenarios["S2"]:
            raise ValueError(f"{scenario} must inherit the complete S2 schedule and C0")
        if project_lifetime_years[scenario] != project_lifetime_years["S2"]:
            raise ValueError(f"{scenario} must inherit the S2 project lifetime")
        if coolant_prices[scenario] != coolant_prices["S2"]:
            raise ValueError(f"{scenario} must inherit the S2 coolant prices")
        if (
            BACKGROUND_CAPITAL_USD_BY_SCENARIO[scenario]
            != BACKGROUND_CAPITAL_USD_BY_SCENARIO["S2"]
        ):
            raise ValueError(f"{scenario} must inherit the S2 background capital")
        if (
            CORE_VOM_USD_PER_MWH_TH_BY_SCENARIO[scenario]
            != CORE_VOM_USD_PER_MWH_TH_BY_SCENARIO["S2"]
        ):
            raise ValueError(f"{scenario} must inherit the S2 core VOM")
        if discount_rate_by_scenario[scenario] != discount_rate_by_scenario["S2"]:
            raise ValueError(f"{scenario} must inherit the S2 discount rate")
    observed_prices = [float(hts_prices[key]) for key in ("S4", "S5", "S6")]
    expected_prices = [
        float(HTS_PRICE_2025_USD_PER_KAM_BY_SCENARIO[key])
        for key in ("S4", "S5", "S6")
    ]
    if not np.allclose(observed_prices, expected_prices, rtol=0.0, atol=1e-12):
        raise ValueError(
            f"Unexpected constant-2025-US$ S4-S6 HTS prices: {observed_prices}; "
            f"expected {expected_prices}"
        )


def calculate_modeled_plant_costs(
    *,
    scenario: str,
    annual_hours_prod: float,
    E_net_year_MWh: float,
    CAPEX_mag_direct_USD: float,
    CAPEX_mag_installed_USD: float,
    coolant_fill_cost_USD: float,
    CRF: float,
    fusion_power_MWth: float = FUSION_POWER_MWTH,
    gross_electric_power_MWe: float = GROSS_ELECTRIC_POWER_MWE,
) -> dict[str, Any]:
    """Calculate the constant-2025-USD plant-v4 numerator and legacy diagnostics.

    ``CAPEX_PCS_reference_USD`` is a reference base used only to calculate
    annual PCS fixed O&M. It is deliberately excluded from every CRF-multiplied
    CAPEX term because the fixed background capital C0 already contains PCS and
    the balance of plant.
    """
    if scenario not in BACKGROUND_CAPITAL_USD_BY_SCENARIO:
        raise KeyError(f"Unknown economic scenario: {scenario}")

    C0_background_USD = float(BACKGROUND_CAPITAL_USD_BY_SCENARIO[scenario])
    core_vom_rate = float(CORE_VOM_USD_PER_MWH_TH_BY_SCENARIO[scenario])
    annual_hours_prod = float(annual_hours_prod)
    E_net_year_MWh = float(E_net_year_MWh)

    E_fusion_th_year_MWh = float(fusion_power_MWth) * annual_hours_prod
    E_gross_year_MWh = float(gross_electric_power_MWe) * annual_hours_prod

    OPEX_core_VOM_USD_per_year = E_fusion_th_year_MWh * core_vom_rate
    OPEX_PCS_VOM_USD_per_year = (
        E_gross_year_MWh * PCS_VOM_USD_PER_MWH_E
    )
    CAPEX_PCS_reference_USD = (
        float(gross_electric_power_MWe)
        * 1000.0
        * PCS_CAPITAL_COST_USD_PER_KWE
    )
    OPEX_PCS_FOM_USD_per_year = (
        CAPEX_PCS_reference_USD * PCS_FOM_FRACTION_PER_YEAR
    )
    OPEX_coolant_VOM_USD_per_year = (
        float(coolant_fill_cost_USD)
        * ANNUAL_COOLANT_REPLENISH_FRACTION
    )
    OPEX_total_modeled_USD_per_year = (
        OPEX_core_VOM_USD_per_year
        + OPEX_PCS_VOM_USD_per_year
        + OPEX_PCS_FOM_USD_per_year
        + OPEX_coolant_VOM_USD_per_year
    )

    Annualized_CAPEX_mag_USD_per_year = (
        float(CAPEX_mag_installed_USD) * float(CRF)
    )
    Annualized_C0_USD_per_year = C0_background_USD * float(CRF)
    Annualized_CAPEX_total_modeled_USD_per_year = (
        Annualized_CAPEX_mag_USD_per_year + Annualized_C0_USD_per_year
    )

    numerator_items = (
        Annualized_CAPEX_mag_USD_per_year,
        Annualized_C0_USD_per_year,
        OPEX_core_VOM_USD_per_year,
        OPEX_PCS_VOM_USD_per_year,
        OPEX_PCS_FOM_USD_per_year,
        OPEX_coolant_VOM_USD_per_year,
    )
    numerator_finite = bool(np.isfinite(numerator_items).all())
    valid = bool(np.isfinite(E_net_year_MWh) and E_net_year_MWh > 0.0 and numerator_finite)
    invalid_reason = None
    if not np.isfinite(E_net_year_MWh):
        invalid_reason = "nonfinite_E_net_year_MWh"
    elif E_net_year_MWh <= 0.0:
        invalid_reason = "nonpositive_E_net_year_MWh"
    elif not numerator_finite:
        invalid_reason = "nonfinite_cost_numerator"

    if valid:
        LCOE_plant_USD_per_MWh = (
            Annualized_CAPEX_total_modeled_USD_per_year
            + OPEX_total_modeled_USD_per_year
        ) / E_net_year_MWh
        LCOE_magnet_only_USD_per_MWh = (
            Annualized_CAPEX_mag_USD_per_year
            + OPEX_coolant_VOM_USD_per_year
        ) / E_net_year_MWh
        # Preserve the exact pre-v2 full-plant formula for comparison.
        LCOE_fullplant_legacy_USD_per_MWh = (
            Annualized_CAPEX_total_modeled_USD_per_year
            + OPEX_coolant_VOM_USD_per_year
        ) / E_net_year_MWh
    else:
        LCOE_plant_USD_per_MWh = np.nan
        LCOE_magnet_only_USD_per_MWh = np.nan
        LCOE_fullplant_legacy_USD_per_MWh = np.nan

    return {
        "cost_boundary_version": COST_BOUNDARY_VERSION,
        "monetary_price_basis_year": PRICE_BASIS_YEAR,
        "monetary_values_constant_2025_usd": True,
        "hts_conductor_price_source_year": 2025,
        "hts_conductor_price_cpi_factor": 1.0,
        "hts_conductor_price_conversion_method": HTS_PRICE_CONVERSION_METHOD,
        "C0_background_USD": C0_background_USD,
        "CAPEX_mag_direct_USD": float(CAPEX_mag_direct_USD),
        "CAPEX_mag_installed_USD": float(CAPEX_mag_installed_USD),
        "CAPEX_PCS_reference_USD": CAPEX_PCS_reference_USD,
        "Annualized_CAPEX_mag_USD_per_year": Annualized_CAPEX_mag_USD_per_year,
        "Annualized_C0_USD_per_year": Annualized_C0_USD_per_year,
        "Annualized_CAPEX_total_modeled_USD_per_year": (
            Annualized_CAPEX_total_modeled_USD_per_year
        ),
        "E_fusion_th_year_MWh": E_fusion_th_year_MWh,
        "E_gross_year_MWh": E_gross_year_MWh,
        "OPEX_core_VOM_USD_per_year": OPEX_core_VOM_USD_per_year,
        "OPEX_PCS_VOM_USD_per_year": OPEX_PCS_VOM_USD_per_year,
        "OPEX_PCS_FOM_USD_per_year": OPEX_PCS_FOM_USD_per_year,
        "OPEX_coolant_VOM_USD_per_year": OPEX_coolant_VOM_USD_per_year,
        "OPEX_total_modeled_USD_per_year": OPEX_total_modeled_USD_per_year,
        "LCOE_plant_USD_per_MWh": LCOE_plant_USD_per_MWh,
        "LCOE_magnet_only_USD_per_MWh": LCOE_magnet_only_USD_per_MWh,
        "LCOE_fullplant_legacy_USD_per_MWh": (
            LCOE_fullplant_legacy_USD_per_MWh
        ),
        "economic_invalid_reason": invalid_reason,
    }
