"""Annual operating-time accounting shared by the V6.7 model.

V6.7 treats the available pulse/dwell time as a continuous annual ledger.
No integer-cycle flooring is applied: the complete free time is apportioned
between pulse and dwell in their prescribed within-cycle ratio.
"""
from __future__ import annotations

import math
from typing import NamedTuple


class ContinuousAnnualSchedule(NamedTuple):
    """Continuous equivalent cycles and their pulse/dwell durations."""

    equivalent_cycles: float
    pulse_hours: float
    dwell_hours: float


def allocate_continuous_pulse_dwell(
    free_hours: float,
    pulse_hours_per_cycle: float,
    dwell_hours_per_cycle: float,
) -> ContinuousAnnualSchedule:
    """Allocate all non-negative ``free_hours`` to pulse and dwell.

    The equivalent-cycle count may be non-integer. This is an annualized time
    accounting device; it does not assert that a fractional physical cycle is
    executed at the end of a particular calendar year.
    """
    values = (free_hours, pulse_hours_per_cycle, dwell_hours_per_cycle)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("annual time inputs must be finite")
    if free_hours < 0.0:
        raise ValueError("free_hours must be non-negative")
    if pulse_hours_per_cycle < 0.0 or dwell_hours_per_cycle < 0.0:
        raise ValueError("pulse and dwell durations must be non-negative")

    cycle_hours = pulse_hours_per_cycle + dwell_hours_per_cycle
    if cycle_hours <= 0.0:
        if free_hours == 0.0:
            return ContinuousAnnualSchedule(0.0, 0.0, 0.0)
        raise ValueError("pulse_hours_per_cycle + dwell_hours_per_cycle must be positive")

    equivalent_cycles = free_hours / cycle_hours
    pulse_hours = free_hours * pulse_hours_per_cycle / cycle_hours
    dwell_hours = free_hours * dwell_hours_per_cycle / cycle_hours
    return ContinuousAnnualSchedule(equivalent_cycles, pulse_hours, dwell_hours)
