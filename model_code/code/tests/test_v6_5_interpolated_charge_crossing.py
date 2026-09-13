"""V6.5 contract tests for interpolated 99.9% charging-time persistence."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np


def _charge_module():
    script = Path(__file__).parents[1] / "scripts" / "2_charging" / "2.5_charge_time999_TF_system_all_Npw=1-200.py"
    spec = spec_from_file_location("v65_charge_crossing", script)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_charge_time_uses_linear_crossing_interpolation_and_two_decimal_persistence():
    module = _charge_module()
    time_s = np.array([97.0, 98.0]) * 3600.0
    # target = 0.999 A; the crossing is 97.00194 h, not the first 98-h sample.
    current = np.array([[0.99899612], [1.0]])
    result = module.calculate_time_to_999(time_s, current, npw=1, I_target_per_conductor=1.0)
    assert result == 97.00


def test_charge_time_preserves_a_crossing_at_the_first_sample():
    module = _charge_module()
    time_s = np.array([0.0, 3600.0])
    current = np.array([[0.999], [1.0]])
    assert module.calculate_time_to_999(time_s, current, npw=1, I_target_per_conductor=1.0) == 0.0
