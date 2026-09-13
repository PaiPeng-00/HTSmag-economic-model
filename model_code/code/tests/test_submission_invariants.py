"""Regression tests for manuscript-facing P0 model definitions."""
import inspect
import math
import os
import re
import unittest

import numpy as np
import pandas as pd

os.environ["FUSION_DEVICE"] = "arc_16pancake_nuc600"

from fusion_tem import device as cfg
from fusion_tem.cryo import heat_load
from fusion_tem.economic import lcoe
from fusion_tem.economic.capital_anchor import NON_TF_AUX_FRACTION
from fusion_tem.economic.feasibility import (
    add_joint_lcoe_reference,
    apply_system_feasibility,
)
from fusion_tem.economic.near_optimal import (
    compute_robust_architecture_window,
    compute_scenario_architecture_regret,
    summarize_robust_windows,
)
from fusion_tem.economic.sensitivity import recalculate_energy_boundary
from fusion_tem.economic.price_basis import (
    HTS_PRICE_2025_USD_PER_KAM_BY_SCENARIO,
)


class SubmissionInvariantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parameters = lcoe.define_parameters()

    def test_gross_electric_power_basis(self):
        self.assertEqual(cfg.P_fusion_W, 525e6)
        self.assertEqual(cfg.fusion_output_MWth, 708.0)
        self.assertEqual(cfg.eat_conv, 0.40)
        self.assertAlmostEqual(cfg.fusion_output_MWth * cfg.eat_conv, 283.2)

    def test_power_supply_reference_price(self):
        self.assertEqual(cfg.POWER_SUPPLY_REFERENCE_CURRENT_A, 500.0)
        self.assertEqual(cfg.POWER_SUPPLY_REFERENCE_COST_USD, 10_000.0)
        self.assertEqual(cfg.power_supply_price_perA, 20.0)
        self.assertEqual(self.parameters["power_supply_unit_price"], 20.0)
        self.assertEqual(cfg.carrying_factor, 0.70)

    def test_tfmc_scaled_coolant_volume(self):
        expected_m3 = 1.5 * (0.34 / 0.9053) * cfg.V_magnet * cfg.Ntf
        self.assertAlmostEqual(self.parameters["cryo_loop_volume_m3"], expected_m3)
        self.assertTrue(math.isclose(expected_m3, 23.2037737078, rel_tol=1e-9))

    def test_price_sensitivity_scenarios_are_s2_single_variable(self):
        s2 = self.parameters["time_scenarios"]["S2"]
        for scenario in ("S4", "S5", "S6"):
            self.assertEqual(self.parameters["time_scenarios"][scenario], s2)
            self.assertEqual(
                self.parameters["tech_scenario_to_years"][scenario],
                self.parameters["tech_scenario_to_years"]["S2"],
            )
            self.assertEqual(
                self.parameters["coolant_price_per_kg_by_scenario"][scenario],
                self.parameters["coolant_price_per_kg_by_scenario"]["S2"],
            )
        self.assertEqual(
            [
                self.parameters["hts_price_per_kAm_by_scenario"][scenario]
                for scenario in ("S4", "S5", "S6")
            ],
            [HTS_PRICE_2025_USD_PER_KAM_BY_SCENARIO[s] for s in ("S4", "S5", "S6")],
        )

    def test_public_result_schema_uses_only_current_metrics(self):
        result, _ = lcoe.compute_case(
            tech_scenario="S2",
            year=self.parameters["tech_scenario_to_years"]["S2"],
            temperature_K=4.2,
            coolant="He",
            Npw=30,
            R_p2p_joint=10e-9,
            p=self.parameters,
            charge_time_h=100.0,
        )
        self.assertAlmostEqual(result["gross_power_output_MWe"], 283.2)
        expected_power_supply_cost = 750.0 * 30 / 0.70 * 20.0
        self.assertAlmostEqual(result["power_supply_cost_$"], expected_power_supply_cost)
        self.assertIn("capex_mag_$", result)
        self.assertIn("lcoe_plant_$/MWh", result)
        self.assertIn("cryo_energy_charge_annual_MWh", result)
        self.assertIn("cryo_energy_discharge_annual_MWh", result)
        self.assertIn("cryo_energy_dynamic_charge_annual_MWh", result)
        self.assertIn("cryo_energy_dynamic_discharge_annual_MWh", result)
        self.assertEqual(
            result["cost_boundary_version"], "plant_v4_core_pcs_coolant_2025usd_direct_hts"
        )
        self.assertIn("LCOE_magnet_only_USD_per_MWh", result)
        self.assertIn("OPEX_total_modeled_USD_per_year", result)
        forbidden = {
            "P_fusion_electric_W",
            "r_cryo_re_production",
            "capex_mag_direct_$",
            "capex_mag_installed_$",
        }
        self.assertTrue(forbidden.isdisjoint(result))

    def test_radiation_heat_is_28_W_per_tf_cryostat(self):
        expected = {
            4.2: 27.9882393848,
            10.0: 27.9882059066,
            20.0: 27.9876876059,
        }
        for temperature_K, expected_W in expected.items():
            observed = heat_load.radiation_heat(
                cfg.A_cryostat, cfg.eps, cfg.T_HIGH, temperature_K
            )
            self.assertAlmostEqual(observed, expected_W, places=6)
            self.assertGreaterEqual(observed, 27.0)
            self.assertLessEqual(observed, 29.0)

    def test_transient_charge_energy_uses_full_system_before_cop(self):
        base = heat_load.calculate_base_heat_loads(
            Ip=cfg.Ip_list[20.0],
            Npw=20,
            R_p2p_joint=10e-9,
            Top=20.0,
        )
        metrics = heat_load.calculate_charge_cryo_electrical_energy(
            base_heat_loads=base,
            Top=20.0,
            time_h=np.array([0.0, 1.0, 2.0]),
            magnetization_loss_W=np.array([0.0, 10.0, 20.0]),
            radial_loss_W=np.array([0.0, 100.0, 200.0]),
            N_tf=cfg.Ntf,
        )
        self.assertGreater(metrics["event_energy_MWh"], 0.0)
        self.assertGreater(metrics["dynamic_event_energy_MWh"], 0.0)
        self.assertAlmostEqual(
            metrics["event_energy_MWh"],
            metrics["base_event_energy_MWh"]
            + metrics["dynamic_event_energy_MWh"],
        )
        self.assertGreater(metrics["peak_heat_Tc_W_per_TF"], base["radiation"] + 200.0)

    def test_discharge_mirrors_complete_charge_profile_and_duration(self):
        self.assertNotIn("charge_stage_power_cap_W", self.parameters)
        self.assertNotIn("other_recirc_fraction", self.parameters)
        self.assertNotIn("discharge_dynamic_factor", self.parameters)
        self.assertNotIn(
            "discharge_dynamic_factor", inspect.signature(lcoe.compute_case).parameters
        )
        self.assertEqual(
            self.parameters["non_tf_aux_fraction"],
            NON_TF_AUX_FRACTION,
        )
        for scenario in ("S1", "S2", "S3", "S4", "S5", "S6"):
            self.assertEqual(self.parameters["time_scenarios"][scenario]["kdis"], 1.0)

        base_result, _ = lcoe.compute_case(
            tech_scenario="S2",
            year=self.parameters["tech_scenario_to_years"]["S2"],
            temperature_K=20.0,
            coolant="He",
            Npw=20,
            R_p2p_joint=10e-9,
            p=self.parameters,
            charge_time_h=100.0,
        )
        base_event = base_result["cryo_energy_charge_base_event_MWh"]
        dynamic_event = 0.25
        charge_event = base_event + dynamic_event
        result, _ = lcoe.compute_case(
            tech_scenario="S2",
            year=self.parameters["tech_scenario_to_years"]["S2"],
            temperature_K=20.0,
            coolant="He",
            Npw=20,
            R_p2p_joint=10e-9,
            p=self.parameters,
            charge_time_h=100.0,
            charge_cryo_energy_event_MWh=charge_event,
            charge_cryo_average_power_W=charge_event * 1e6 / 100.0,
        )
        tf_cycles = self.parameters["time_scenarios"]["S2"]["tf_cycles_per_year"]

        self.assertAlmostEqual(result["cryo_energy_charge_event_MWh"], charge_event)
        self.assertAlmostEqual(
            result["cryo_energy_discharge_event_MWh"], charge_event
        )
        self.assertAlmostEqual(
            result["cryo_energy_discharge_base_event_MWh"],
            result["cryo_energy_charge_base_event_MWh"],
        )
        self.assertAlmostEqual(
            result["cryo_energy_discharge_dynamic_event_MWh"],
            result["cryo_energy_charge_dynamic_event_MWh"],
        )
        self.assertAlmostEqual(
            result["cryo_power_discharge_average_W"],
            result["cryo_power_charge_average_W"],
        )
        self.assertAlmostEqual(result["charge_time_event_h"], 100.0)
        self.assertAlmostEqual(result["discharge_time_event_h"], 100.0)
        self.assertAlmostEqual(result["annual_hours_excdec"], 2.0 * tf_cycles * 100.0)
        self.assertAlmostEqual(
            result["cryo_energy_dynamic_charge_annual_MWh"],
            result["cryo_energy_dynamic_discharge_annual_MWh"],
        )
        self.assertAlmostEqual(
            result["cryo_energy_excdec_annual_MWh"],
            2.0 * tf_cycles * charge_event,
        )
        self.assertEqual(result["charge_discharge_profile_ratio"], 1.0)
        self.assertAlmostEqual(
            result["other_aux_energy_annual_MWh"],
            self.parameters["non_tf_aux_fraction"]
            * result["gross_energy_annual_MWh"],
        )
        self.assertAlmostEqual(
            result["net_energy_annual_MWh"],
            result["gross_energy_annual_MWh"]
            - result["other_aux_energy_annual_MWh"]
            - result["cryo_energy_annual_MWh"],
        )

    def test_all_six_scenarios_return_finite_central_results(self):
        for scenario in ("S1", "S2", "S3", "S4", "S5", "S6"):
            result, _ = lcoe.compute_case(
                tech_scenario=scenario,
                year=self.parameters["tech_scenario_to_years"][scenario],
                temperature_K=20.0,
                coolant="He",
                Npw=20,
                R_p2p_joint=10e-9,
                p=self.parameters,
                charge_time_h=1.0,
            )
            self.assertTrue(np.isfinite(result["lcoe_plant_$/MWh"]))
            self.assertAlmostEqual(
                result["net_energy_annual_MWh"],
                result["gross_energy_annual_MWh"]
                - result["other_aux_energy_annual_MWh"]
                - result["cryo_energy_annual_MWh"],
            )

    def test_absolute_plant_availability_and_exact_joint_feasibility_boundaries(self):
        sample = pd.DataFrame(
            {
                "scenario": ["S1", "S1", "S2", "S2", "S3", "S3", "S4", "S5", "S6"],
                "status": ["success"] * 9,
                "Aplant": [0.81, 0.81, 0.80, 0.79, 0.90, 0.79, 0.80, 0.80, 0.80],
                "r_cryo_re": [50.0, 50.1, 50.0, 49.0, 50.0, 50.0, 50.0, 50.0, 50.0],
                "E_net_year_MWh": [20.0, 19.9, 20.0, 19.0, 20.0, 20.0, 20.0, 0.0, 20.0],
                "E_gross_year_MWh": [100.0] * 9,
                "Charging_time_999_h": [121.0, 120.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
                "LCOE_plant_USD_per_MWh": [10.0, 11.0, 20.0, 21.0, 30.0, 31.0, 22.0, 23.0, 24.0],
            }
        )
        revised, availability_max = apply_system_feasibility(sample)
        self.assertEqual(
            availability_max.loc[["S1", "S2", "S3"]].to_dict(),
            {"S1": 0.81, "S2": 0.8, "S3": 0.9},
        )
        self.assertEqual(revised.loc[6, "Aplant_max_scenario"], 0.8)
        self.assertTrue(revised.loc[0, "feasible_cryo_050"])
        self.assertFalse(revised.loc[1, "feasible_cryo_050"])
        self.assertFalse(revised.loc[1, "feasible_joint"])
        self.assertFalse(revised.loc[3, "feasible_joint"])
        self.assertFalse(revised.loc[3, "feasible_net_export"])
        self.assertTrue(revised.loc[7, "feasible_joint"])
        self.assertFalse(revised.loc[7, "feasible_net_export"])
        self.assertTrue(revised.loc[0, "charging_time_warning_120h"])
        self.assertFalse(revised.loc[1, "charging_time_warning_120h"])
        self.assertTrue((revised["feasible"] == revised["feasible_joint"]).all())
        self.assertTrue(revised["no_model_error"].all())

        with_delta, refs = add_joint_lcoe_reference(revised)
        self.assertEqual(
            refs.to_dict(),
            {"S1": 10.0, "S2": 20.0, "S3": 30.0, "S4": 22.0, "S5": 23.0, "S6": 24.0},
        )
        mins = with_delta.loc[with_delta["feasible_joint"]].groupby(
            "scenario"
        )["delta_LCOE_joint_feasible_USD_per_MWh"].min()
        self.assertTrue((mins.abs() < 1e-12).all())

    def test_near_optimal_regret_reoptimizes_and_defines_robust_windows(self):
        rows = []
        values = {
            "S1": {(10, 10.0): 100.0, (20, 10.0): 104.0, (30, 100.0): 120.0},
            "S2": {(10, 10.0): 104.0, (20, 10.0): 100.0, (30, 100.0): 110.0},
            "S3": {(10, 10.0): 102.0, (20, 10.0): 103.0, (30, 100.0): 100.0},
        }
        for scenario, architectures in values.items():
            for (npw, rj), best_lcoe in architectures.items():
                for rho, penalty in ((1000.0, 0.0), (5000.0, 5.0)):
                    rows.append(
                        {
                            "scenario": scenario,
                            "feasible_joint": True,
                            "LCOE_plant_USD_per_MWh": best_lcoe + penalty,
                            "Npw": npw,
                            "R_joint_nOhm": rj,
                            "rho_turn_uOhm_cm2": rho,
                            "Top_K": 20.0,
                            "coolant": "H2",
                        }
                    )
        regret = compute_scenario_architecture_regret(pd.DataFrame(rows))
        self.assertTrue(
            np.allclose(regret.groupby("scenario")["regret_fraction"].min(), 0.0)
        )
        robust = compute_robust_architecture_window(regret)
        summary = summarize_robust_windows(robust)
        counts = dict(
            zip(summary["regret_threshold_pct"], summary["robust_architecture_count"])
        )
        self.assertEqual(counts[2.5], 0)
        self.assertEqual(counts[5.0], 2)
        self.assertEqual(counts[10.0], 2)

    def test_energy_boundary_sensitivity_preserves_closure(self):
        base = pd.DataFrame(
            {
                "E_gross_year_MWh": [1000.0],
                "E_other_year_MWh": [300.0],
                "E_cryo_year_MWh": [100.0],
                "E_net_year_MWh": [600.0],
                "E_cryo_charge_MWh": [10.0],
                "E_cryo_discharge_MWh": [10.0],
                "E_cryo_excdis_year_MWh": [20.0],
                "E_dynamic_charge_MWh": [2.0],
                "E_dynamic_discharge_MWh": [2.0],
                "charge_discharge_profile_ratio": [1.0],
                "LCOE_plant_USD_per_MWh": [50.0],
                "r_other": [0.30],
            }
        )
        revised = recalculate_energy_boundary(
            base,
            other_aux_fraction=0.20,
        )
        self.assertAlmostEqual(revised.loc[0, "E_other_year_MWh"], 200.0)
        self.assertAlmostEqual(revised.loc[0, "E_cryo_year_MWh"], 100.0)
        self.assertAlmostEqual(revised.loc[0, "E_net_year_MWh"], 700.0)
        self.assertAlmostEqual(revised.loc[0, "energy_closure_error_MWh"], 0.0)
        self.assertAlmostEqual(
            revised.loc[0, "LCOE_plant_USD_per_MWh"], 30000.0 / 700.0
        )

    def test_no_fixed_500_W_charge_or_discharge_load(self):
        sources = (
            inspect.getsource(lcoe),
            inspect.getsource(heat_load),
        )
        forbidden_patterns = (
            r"charge_stage_power_cap_W",
            r"P_cryo_(?:charge|discharge)\w*\s*=\s*500(?:\.0)?",
            r"500\s*W\s*(?:charge|discharge)",
        )
        for source in sources:
            for pattern in forbidden_patterns:
                self.assertIsNone(re.search(pattern, source, flags=re.IGNORECASE))


if __name__ == "__main__":
    unittest.main()
