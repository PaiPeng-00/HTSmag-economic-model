"""Regression tests for the plant_v4_core_pcs_coolant_2025usd_direct_hts economic boundary."""
import os
import unittest

import numpy as np
import pandas as pd

os.environ["FUSION_DEVICE"] = "arc_16pancake_nuc600"

from fusion_tem.economic.cost_boundary import (
    ANNUAL_COOLANT_REPLENISH_FRACTION,
    BACKGROUND_CAPITAL_USD_BY_SCENARIO,
    CORE_VOM_USD_PER_MWH_TH_BY_SCENARIO,
    COST_BOUNDARY_VERSION,
    FUSION_POWER_MWTH,
    GROSS_ELECTRIC_POWER_MWE,
    PCS_CAPITAL_COST_USD_PER_KWE,
    PCS_FOM_FRACTION_PER_YEAR,
    PCS_VOM_USD_PER_MWH_E,
    THERMAL_POWER_AFTER_BLANKET_MWTH,
    calculate_modeled_plant_costs,
)
from fusion_tem.economic.feasibility import apply_system_feasibility
from fusion_tem.economic.lcoe import compute_case, define_parameters


class PlantOpexBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parameters = define_parameters()
        cls.hours = 4000.0
        cls.net = 700_000.0
        cls.mag_capex = 300_000_000.0
        cls.coolant_fill = 2_000_000.0
        cls.crf = 0.08
        cls.cost = calculate_modeled_plant_costs(
            scenario="S2",
            annual_hours_prod=cls.hours,
            E_net_year_MWh=cls.net,
            CAPEX_mag_direct_USD=cls.mag_capex,
            CAPEX_mag_installed_USD=cls.mag_capex,
            coolant_fill_cost_USD=cls.coolant_fill,
            CRF=cls.crf,
        )

    def test_01_opex_component_identity(self):
        c = self.cost
        expected = (
            c["OPEX_core_VOM_USD_per_year"]
            + c["OPEX_PCS_VOM_USD_per_year"]
            + c["OPEX_PCS_FOM_USD_per_year"]
            + c["OPEX_coolant_VOM_USD_per_year"]
        )
        self.assertAlmostEqual(c["OPEX_total_modeled_USD_per_year"], expected)

    def test_02_lcoe_numerator_identity(self):
        c = self.cost
        expected = (
            self.crf
            * (
                c["CAPEX_mag_installed_USD"]
                + c["C0_background_USD"]
            )
            + c["OPEX_total_modeled_USD_per_year"]
        )
        self.assertAlmostEqual(c["LCOE_plant_USD_per_MWh"] * self.net, expected)

    def test_03_core_vom_uses_525_mwth_not_708_mwth(self):
        c = self.cost
        rate = CORE_VOM_USD_PER_MWH_TH_BY_SCENARIO["S2"]
        self.assertEqual(FUSION_POWER_MWTH, 525.0)
        self.assertEqual(THERMAL_POWER_AFTER_BLANKET_MWTH, 708.0)
        self.assertAlmostEqual(
            c["OPEX_core_VOM_USD_per_year"],
            525.0 * self.hours * rate,
        )
        self.assertNotAlmostEqual(
            c["OPEX_core_VOM_USD_per_year"],
            708.0 * self.hours * rate,
        )

    def test_04_pcs_vom_uses_annual_gross_generation(self):
        c = self.cost
        expected_gross = GROSS_ELECTRIC_POWER_MWE * self.hours
        self.assertAlmostEqual(c["E_gross_year_MWh"], expected_gross)
        self.assertAlmostEqual(
            c["OPEX_PCS_VOM_USD_per_year"],
            expected_gross * PCS_VOM_USD_PER_MWH_E,
        )

    def test_05_pcs_reference_is_only_an_fom_basis(self):
        c = self.cost
        expected_reference = (
            GROSS_ELECTRIC_POWER_MWE
            * 1000.0
            * PCS_CAPITAL_COST_USD_PER_KWE
        )
        self.assertAlmostEqual(c["CAPEX_PCS_reference_USD"], expected_reference)
        self.assertAlmostEqual(
            c["OPEX_PCS_FOM_USD_per_year"],
            expected_reference * PCS_FOM_FRACTION_PER_YEAR,
        )
        self.assertAlmostEqual(
            c["Annualized_CAPEX_total_modeled_USD_per_year"],
            self.crf
            * (self.mag_capex + BACKGROUND_CAPITAL_USD_BY_SCENARIO["S2"]),
        )
        self.assertNotAlmostEqual(
            c["Annualized_CAPEX_total_modeled_USD_per_year"],
            self.crf
            * (
                self.mag_capex
                + BACKGROUND_CAPITAL_USD_BY_SCENARIO["S2"]
                + expected_reference
            ),
        )

    def test_06_s4_s6_are_strict_s2_price_only_cases(self):
        p = self.parameters
        for scenario in ("S4", "S5", "S6"):
            self.assertEqual(
                p["time_scenarios"][scenario],
                p["time_scenarios"]["S2"],
            )
            self.assertEqual(
                p["tech_scenario_to_years"][scenario],
                p["tech_scenario_to_years"]["S2"],
            )
            self.assertEqual(
                p["coolant_price_per_kg_by_scenario"][scenario],
                p["coolant_price_per_kg_by_scenario"]["S2"],
            )
            self.assertEqual(
                p["background_capital_USD_by_scenario"][scenario],
                p["background_capital_USD_by_scenario"]["S2"],
            )
            self.assertEqual(
                p["core_vom_USD_per_MWh_th_by_scenario"][scenario],
                p["core_vom_USD_per_MWh_th_by_scenario"]["S2"],
            )
            self.assertEqual(
                p["discount_rate_by_scenario"][scenario],
                p["discount_rate_by_scenario"]["S2"],
            )

    def test_07_other_energy_is_pulse_gross_only(self):
        result, _ = compute_case(
            tech_scenario="S2",
            year=self.parameters["tech_scenario_to_years"]["S2"],
            temperature_K=20.0,
            coolant="He",
            Npw=20,
            R_p2p_joint=10e-9,
            p=self.parameters,
            charge_time_h=10.0,
        )
        pulse_gross = (
            result["gross_power_output_MWe"] * result["annual_hours_prod"]
        )
        self.assertAlmostEqual(result["gross_energy_annual_MWh"], pulse_gross)
        self.assertAlmostEqual(
            result["other_aux_energy_annual_MWh"],
            self.parameters["non_tf_aux_fraction"] * pulse_gross,
        )
        self.assertNotAlmostEqual(
            result["other_aux_energy_annual_MWh"],
            self.parameters["non_tf_aux_fraction"]
            * result["gross_power_output_MWe"]
            * 8760.0,
        )

    def test_08_plant_lcoe_not_below_magnet_only_legacy(self):
        self.assertEqual(self.cost["cost_boundary_version"], COST_BOUNDARY_VERSION)
        self.assertGreaterEqual(
            self.cost["LCOE_plant_USD_per_MWh"],
            self.cost["LCOE_magnet_only_USD_per_MWh"],
        )

    def test_09_fixed_opex_terms_depend_only_on_scenario_and_pulse_time(self):
        alternate = calculate_modeled_plant_costs(
            scenario="S2",
            annual_hours_prod=self.hours,
            E_net_year_MWh=self.net - 10_000.0,
            CAPEX_mag_direct_USD=self.mag_capex * 2.0,
            CAPEX_mag_installed_USD=self.mag_capex * 2.0,
            coolant_fill_cost_USD=self.coolant_fill * 3.0,
            CRF=self.crf,
        )
        for field in (
            "OPEX_core_VOM_USD_per_year",
            "OPEX_PCS_VOM_USD_per_year",
            "OPEX_PCS_FOM_USD_per_year",
        ):
            self.assertAlmostEqual(self.cost[field], alternate[field])
        self.assertAlmostEqual(
            alternate["OPEX_coolant_VOM_USD_per_year"],
            self.coolant_fill
            * 3.0
            * ANNUAL_COOLANT_REPLENISH_FRACTION,
        )
        self.assertNotEqual(
            self.cost["OPEX_coolant_VOM_USD_per_year"],
            alternate["OPEX_coolant_VOM_USD_per_year"],
        )

    def test_10_joint_feasible_rows_have_finite_lcoe_and_positive_net(self):
        valid = self.cost
        invalid = calculate_modeled_plant_costs(
            scenario="S2",
            annual_hours_prod=self.hours,
            E_net_year_MWh=0.0,
            CAPEX_mag_direct_USD=self.mag_capex,
            CAPEX_mag_installed_USD=self.mag_capex,
            coolant_fill_cost_USD=self.coolant_fill,
            CRF=self.crf,
        )
        frame = pd.DataFrame(
            {
                "scenario": ["S2", "S2"],
                "status": ["success", "success"],
                "Aplant": [0.81, 0.81],
                "r_cryo_re_fraction": [0.1, 0.1],
                "E_net_year_MWh": [self.net, 0.0],
                "E_gross_year_MWh": [
                    valid["E_gross_year_MWh"],
                    invalid["E_gross_year_MWh"],
                ],
                "Charging_time_999_h": [500.0, 1.0],
                "LCOE_plant_USD_per_MWh": [
                    valid["LCOE_plant_USD_per_MWh"],
                    invalid["LCOE_plant_USD_per_MWh"],
                ],
            }
        )
        revised, _ = apply_system_feasibility(frame)
        feasible = revised.loc[revised["feasible_joint"]]
        self.assertFalse(feasible.empty)
        self.assertTrue(np.isfinite(feasible["LCOE_plant_USD_per_MWh"]).all())
        self.assertTrue((feasible["E_net_year_MWh"] > 0.0).all())
        self.assertTrue(revised.loc[0, "charging_time_warning_120h"])
        self.assertTrue(revised.loc[0, "feasible_joint"])
        self.assertFalse(revised.loc[1, "feasible_joint"])


if __name__ == "__main__":
    unittest.main()
