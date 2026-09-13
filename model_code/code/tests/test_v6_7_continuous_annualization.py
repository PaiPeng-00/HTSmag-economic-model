"""Regression tests for the V6.7 continuous annual-time contract."""
from pathlib import Path

import numpy as np
import pytest

from fusion_tem.economic.annual_time import allocate_continuous_pulse_dwell


def test_continuous_allocation_exhausts_free_time_exactly():
    schedule = allocate_continuous_pulse_dwell(8123.456, 2.0, 0.5)
    assert schedule.equivalent_cycles == pytest.approx(3249.3824)
    assert schedule.pulse_hours + schedule.dwell_hours == pytest.approx(8123.456)
    assert schedule.pulse_hours / schedule.dwell_hours == pytest.approx(4.0)


def test_crossing_rounding_near_npw85_cannot_create_a_whole_cycle_jump():
    hours_per_year = 8760.0
    fixed_noncycle_hours = 100.0
    pulse_h, dwell_h = 2.0, 0.5
    charge_times = np.array([97.05, 97.00, 96.99])
    schedules = [
        allocate_continuous_pulse_dwell(
            hours_per_year - fixed_noncycle_hours - 2.0 * time_h,
            pulse_h,
            dwell_h,
        )
        for time_h in charge_times
    ]
    counts = np.array([item.equivalent_cycles for item in schedules])
    assert np.max(np.abs(np.diff(counts))) < 0.05
    assert np.all(np.diff(counts) >= 0.0)


@pytest.mark.parametrize(
    "args",
    [(-1.0, 2.0, 0.5), (1.0, -2.0, 0.5), (1.0, 0.0, 0.0), (np.nan, 2.0, 0.5)],
)
def test_invalid_continuous_schedule_inputs_fail_closed(args):
    with pytest.raises(ValueError):
        allocate_continuous_pulse_dwell(*args)


def test_v67_active_annual_ledgers_contain_no_integer_flooring():
    root = Path(__file__).parents[1]
    sources = [
        root / "src" / "fusion_tem" / "economic" / "lcoe.py",
        root / "scripts" / "8_economic" / "8.33_run_v6_1_a_availability.py",
        root / "scripts" / "2_charging" / "2.9_af_form_full_grid.py",
        root / "scripts" / "2_charging" / "2.9_af_time999_3&3.py",
    ]
    for source in sources:
        text = source.read_text(encoding="utf-8")
        assert "remaining //" not in text
        assert "np.floor(remaining / tcycle)" not in text
