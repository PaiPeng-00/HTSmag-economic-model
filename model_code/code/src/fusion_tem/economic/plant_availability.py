"""Plant-availability accounting for the V7 annual operating model.

The model separates plant availability from pulse duty.  Pulse and dwell are
both plant-available states.  Planned core maintenance is supplied as days per
full-power year (FPY), converted through the ARIES availability convention,
and combined with one annual-equivalent TF shutdown--restart cycle.  Generic
unplanned unavailability is applied only to the time remaining after planned
outages, avoiding double counting.
"""
from __future__ import annotations

import math
from typing import NamedTuple

from fusion_tem.economic.annual_time import allocate_continuous_pulse_dwell


class PlantAnnualSchedule(NamedTuple):
    core_scheduled_availability: float
    core_scheduled_outage_hours: float
    tf_planned_outage_hours: float
    planned_outage_hours: float
    planned_unavailability: float
    effective_unplanned_outage_hours: float
    available_hours: float
    plant_availability: float
    pulse_duty_factor: float
    gross_capacity_factor: float
    equivalent_cycles: float
    pulse_hours: float
    dwell_hours: float
    cooldown_warmup_hours: float
    charge_discharge_hours: float


def availability_from_days_per_fpy(days_per_fpy: float) -> float:
    """Convert maintenance days/FPY to availability using a 365.25-day FPY."""
    value = float(days_per_fpy)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("maintenance days per FPY must be finite and non-negative")
    return 365.25 / (365.25 + value)


def core_scheduled_outage_hours(
    major_days_per_fpy: float,
    minor_days_per_fpy: float,
    *,
    hours_per_year: float = 8760.0,
) -> tuple[float, float]:
    """Return combined core-scheduled availability and equivalent annual hours."""
    year = float(hours_per_year)
    if not math.isfinite(year) or year <= 0.0:
        raise ValueError("hours_per_year must be finite and positive")
    availability = (
        availability_from_days_per_fpy(major_days_per_fpy)
        * availability_from_days_per_fpy(minor_days_per_fpy)
    )
    return availability, year * (1.0 - availability)


def allocate_plant_annual_schedule(
    *,
    charge_hours: float,
    major_days_per_fpy: float,
    minor_days_per_fpy: float,
    tf_cycles_per_year: float,
    cooldown_hours_per_cycle: float,
    warmup_hours_per_cycle: float,
    discharge_to_charge_ratio: float,
    unplanned_unavailability: float,
    pulse_hours_per_cycle: float,
    dwell_hours_per_cycle: float,
    hours_per_year: float = 8760.0,
) -> PlantAnnualSchedule:
    """Allocate one annual ledger under the V7 strict plant-availability model."""
    values = (
        charge_hours,
        tf_cycles_per_year,
        cooldown_hours_per_cycle,
        warmup_hours_per_cycle,
        discharge_to_charge_ratio,
        unplanned_unavailability,
        pulse_hours_per_cycle,
        dwell_hours_per_cycle,
        hours_per_year,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("annual schedule inputs must be finite")
    if charge_hours < 0.0 or tf_cycles_per_year < 0.0:
        raise ValueError("charge hours and TF-cycle frequency must be non-negative")
    if cooldown_hours_per_cycle < 0.0 or warmup_hours_per_cycle < 0.0:
        raise ValueError("cooldown and warm-up durations must be non-negative")
    if discharge_to_charge_ratio < 0.0:
        raise ValueError("discharge-to-charge ratio must be non-negative")
    if not 0.0 <= unplanned_unavailability < 1.0:
        raise ValueError("unplanned unavailability must lie in [0, 1)")
    if hours_per_year <= 0.0:
        raise ValueError("hours_per_year must be positive")

    core_availability, core_outage = core_scheduled_outage_hours(
        major_days_per_fpy,
        minor_days_per_fpy,
        hours_per_year=hours_per_year,
    )
    cooldown_warmup = tf_cycles_per_year * (
        cooldown_hours_per_cycle + warmup_hours_per_cycle
    )
    charge_discharge = tf_cycles_per_year * charge_hours * (
        1.0 + discharge_to_charge_ratio
    )
    tf_planned = cooldown_warmup + charge_discharge
    planned = core_outage + tf_planned
    planned_base = max(0.0, hours_per_year - planned)
    effective_unplanned = planned_base * unplanned_unavailability
    available = planned_base * (1.0 - unplanned_unavailability)
    plant_availability = available / hours_per_year

    pulse_dwell = allocate_continuous_pulse_dwell(
        available,
        pulse_hours_per_cycle,
        dwell_hours_per_cycle,
    )
    pulse_duty = pulse_hours_per_cycle / (
        pulse_hours_per_cycle + dwell_hours_per_cycle
    )
    return PlantAnnualSchedule(
        core_scheduled_availability=core_availability,
        core_scheduled_outage_hours=core_outage,
        tf_planned_outage_hours=tf_planned,
        planned_outage_hours=planned,
        planned_unavailability=min(1.0, planned / hours_per_year),
        effective_unplanned_outage_hours=effective_unplanned,
        available_hours=available,
        plant_availability=plant_availability,
        pulse_duty_factor=pulse_duty,
        gross_capacity_factor=plant_availability * pulse_duty,
        equivalent_cycles=pulse_dwell.equivalent_cycles,
        pulse_hours=pulse_dwell.pulse_hours,
        dwell_hours=pulse_dwell.dwell_hours,
        cooldown_warmup_hours=cooldown_warmup,
        charge_discharge_hours=charge_discharge,
    )
