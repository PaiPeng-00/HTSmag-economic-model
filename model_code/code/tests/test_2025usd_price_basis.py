"""Regression tests for constant-2025-US$ normalization and robust frontier."""
import unittest

import numpy as np
import pandas as pd

from fusion_tem.economic.cost_boundary import COST_BOUNDARY_VERSION
from fusion_tem.economic.near_optimal import exact_robust_frontier_at_npw
from fusion_tem.economic.price_basis import (
    ANNUAL_COOLANT_REPLENISH_FRACTION,
    BACKGROUND_CAPITAL_USD_BY_SCENARIO,
    COOLANT_PRICE_2025_USD_PER_KG_BY_SCENARIO,
    CPI_2025_OVER_2020,
    CORE_VOM_USD_PER_MWH_TH_BY_SCENARIO,
    HTS_PRICE_2025_USD_PER_KAM_BY_SCENARIO,
    PCS_CAPITAL_COST_USD_PER_KWE,
    PCS_FOM_FRACTION_PER_YEAR,
    PCS_VOM_USD_PER_MWH_E,
    POWER_SUPPLY_PRICE_2025_USD_PER_A,
    PRICE_BASIS_YEAR,
    monetary_conversion_audit_rows,
    validate_price_source_mapping,
)


class PriceBasis2025Tests(unittest.TestCase):
    def test_01_boundary_and_price_basis(self):
        self.assertEqual(PRICE_BASIS_YEAR, 2025)
        self.assertEqual(
            COST_BOUNDARY_VERSION,
            "plant_v4_core_pcs_coolant_2025usd_direct_hts",
        )
        validate_price_source_mapping()

    def test_02_converted_monetary_constants(self):
        self.assertAlmostEqual(
            BACKGROUND_CAPITAL_USD_BY_SCENARIO["S1"],
            6.200978647686833e9,
        )
        self.assertAlmostEqual(
            BACKGROUND_CAPITAL_USD_BY_SCENARIO["S2"],
            4.133985765124555e9,
            places=5,
        )
        self.assertAlmostEqual(
            BACKGROUND_CAPITAL_USD_BY_SCENARIO["S3"],
            2.0669928825622774e9,
        )
        self.assertAlmostEqual(
            CORE_VOM_USD_PER_MWH_TH_BY_SCENARIO["S2"],
            3.777831234818526,
        )
        self.assertAlmostEqual(PCS_CAPITAL_COST_USD_PER_KWE, 961.5711628907198)
        self.assertAlmostEqual(PCS_VOM_USD_PER_MWH_E, 2.191142116194745)
        self.assertEqual(PCS_FOM_FRACTION_PER_YEAR, 0.025)
        self.assertEqual(ANNUAL_COOLANT_REPLENISH_FRACTION, 0.25)
        self.assertEqual(POWER_SUPPLY_PRICE_2025_USD_PER_A, 20.0)

    def test_03_price_only_scenarios_inherit_s2(self):
        self.assertEqual(
            HTS_PRICE_2025_USD_PER_KAM_BY_SCENARIO["S4"], 100.0
        )
        self.assertAlmostEqual(
            HTS_PRICE_2025_USD_PER_KAM_BY_SCENARIO["S5"],
            50.0,
        )
        self.assertAlmostEqual(
            HTS_PRICE_2025_USD_PER_KAM_BY_SCENARIO["S6"],
            10.0,
        )
        for scenario in ("S4", "S5", "S6"):
            self.assertEqual(
                COOLANT_PRICE_2025_USD_PER_KG_BY_SCENARIO[scenario],
                COOLANT_PRICE_2025_USD_PER_KG_BY_SCENARIO["S2"],
            )
            self.assertEqual(
                BACKGROUND_CAPITAL_USD_BY_SCENARIO[scenario],
                BACKGROUND_CAPITAL_USD_BY_SCENARIO["S2"],
            )
            self.assertEqual(
                CORE_VOM_USD_PER_MWH_TH_BY_SCENARIO[scenario],
                CORE_VOM_USD_PER_MWH_TH_BY_SCENARIO["S2"],
            )

    def test_04_audit_records_one_conversion(self):
        rows = monetary_conversion_audit_rows()
        model_rows = [row for row in rows if row["affected_code_field"] != "context only"]
        self.assertEqual(len(model_rows), 20)
        self.assertTrue(all(row["conversion_application_count"] == 1 for row in rows))
        self.assertTrue(all(row["mapping_verification"] == "PASS" for row in rows))
        self.assertTrue(all(row["dependent_manuscript_figure_locations"] for row in rows))
        hts_rows = [row for row in model_rows if row["parameter"].startswith("p_HTS")]
        self.assertEqual(len(hts_rows), 3)
        self.assertEqual(
            {row["value_2025_USD"] for row in hts_rows},
            {100.0, 50.0, 10.0},
        )
        for row in hts_rows:
            self.assertEqual(row["source_price_year"], 2025)
            self.assertEqual(row["CPI_2025_over_CPI_y"], 1.0)
            self.assertEqual(
                row["conversion_method"],
                "direct_scenario_assumption_2025usd",
            )
            self.assertFalse(row["price_year_is_proxy"])
            self.assertIn("range/basis only", row["basis_note"])

    def test_05_contextual_2020_values_use_annual_average_cpi_u(self):
        self.assertAlmostEqual(CPI_2025_OVER_2020, 321.943 / 258.811)
        rows = {
            row["parameter"]: row for row in monetary_conversion_audit_rows()
        }
        foak = rows["context Lindley ARC-like FOAK specific capital"]
        self.assertAlmostEqual(
            foak["value_2025_USD_numeric_min"], 36907.429785, places=5
        )
        self.assertAlmostEqual(
            foak["value_2025_USD_numeric_max"], 42915.616029, places=5
        )

    def test_06_exact_grid_frontier_not_interpolated(self):
        robust = pd.DataFrame(
            {
                "Npw": [200, 200, 200],
                "R_joint_nOhm": [8.577, 10.0, 11.659],
                "scenario_complete": [True, True, True],
                "regret_S1_fraction": [0.03, 0.049, 0.051],
                "regret_S2_fraction": [0.02, 0.04, 0.045],
                "regret_S3_fraction": [0.01, 0.03, 0.04],
                "worst_case_regret_fraction": [0.03, 0.049, 0.051],
            }
        )
        frontier = exact_robust_frontier_at_npw(robust).iloc[0]
        self.assertEqual(frontier["Rj_max_retained_nOhm"], 10.0)
        self.assertEqual(frontier["Rj_next_excluded_nOhm"], 11.659)
        self.assertEqual(frontier["binding_scenario"], "S1")
        self.assertAlmostEqual(frontier["max_regret"], 0.049)
        self.assertFalse(bool(frontier["interpolated"]))
        self.assertTrue(np.isclose(frontier["next_excluded_regret_S1"], 0.051))


if __name__ == "__main__":
    unittest.main()