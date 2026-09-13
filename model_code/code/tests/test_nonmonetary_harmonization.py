"""Tests for guarded nonmonetary byte harmonization."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from fusion_tem.economic.nonmonetary_invariance import (
    harmonize_nonmonetary_csv,
)


class NonmonetaryHarmonizationTests(unittest.TestCase):
    @staticmethod
    def _frame(physical: float, lcoe: float) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Top_K": [20.0],
                "coolant": ["He"],
                "Npw": [200],
                "rho_turn_uOhm_cm2": [10000.0],
                "R_joint_nOhm": [10.0],
                "scenario": ["S2"],
                "physical_power_W": [physical],
                "AF_ref_system": [1.0],
                "LCOE_plant_USD_per_MWh": [lcoe],
                "cost_boundary_version": ["test"],
            }
        )

    def test_small_ulp_drift_is_harmonized_without_touching_money(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old_path = root / "old.csv"
            new_path = root / "new.csv"
            output_path = root / "out.csv"
            old_value = 30629.86877099196
            new_value = np.nextafter(old_value, 0.0)
            self._frame(old_value, 100.0).to_csv(old_path, index=False)
            self._frame(new_value, 125.0).to_csv(new_path, index=False)
            report = harmonize_nonmonetary_csv(
                old_path, new_path, output_path
            )
            old = pd.read_csv(old_path, dtype=str, keep_default_na=False)
            new = pd.read_csv(new_path, dtype=str, keep_default_na=False)
            out = pd.read_csv(output_path, dtype=str, keep_default_na=False)
            self.assertEqual(
                out.loc[0, "physical_power_W"],
                old.loc[0, "physical_power_W"],
            )
            self.assertEqual(
                out.loc[0, "LCOE_plant_USD_per_MWh"],
                new.loc[0, "LCOE_plant_USD_per_MWh"],
            )
            self.assertGreaterEqual(report["nonmonetary_cells_replaced"], 1)
            self.assertLessEqual(report["maximum_observed_ulp"], 4)

    def test_large_physical_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old_path = root / "old.csv"
            new_path = root / "new.csv"
            output_path = root / "out.csv"
            self._frame(100.0, 100.0).to_csv(old_path, index=False)
            self._frame(100.01, 125.0).to_csv(new_path, index=False)
            with self.assertRaises(AssertionError):
                harmonize_nonmonetary_csv(
                    old_path, new_path, output_path
                )
            self.assertFalse(output_path.exists())


if __name__ == "__main__":
    unittest.main()