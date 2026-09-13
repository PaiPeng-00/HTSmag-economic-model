import unittest

from fusion_tem.economic.plant_availability import (
    allocate_plant_annual_schedule,
    core_scheduled_outage_hours,
)


class V7PlantAvailabilityTests(unittest.TestCase):
    def test_frozen_scenario_core_outages(self):
        expected = {
            16.93: 524.4681699528142,
            8.47: 338.0382243191884,
            4.23: 241.39126662489753,
        }
        for major, target in expected.items():
            _, hours = core_scheduled_outage_hours(major, 6.05)
            self.assertAlmostEqual(hours, target, places=9)

    def test_overlap_and_dwell_classification(self):
        schedule = allocate_plant_annual_schedule(
            charge_hours=96.4,
            major_days_per_fpy=16.93,
            minor_days_per_fpy=6.05,
            tf_cycles_per_year=1.0,
            cooldown_hours_per_cycle=288.0,
            warmup_hours_per_cycle=144.0,
            discharge_to_charge_ratio=1.0,
            unplanned_unavailability=0.07,
            pulse_hours_per_cycle=1.0,
            dwell_hours_per_cycle=1.0,
        )
        self.assertAlmostEqual(schedule.plant_availability, 0.8079886531899411)
        self.assertAlmostEqual(
            schedule.pulse_hours + schedule.dwell_hours,
            schedule.available_hours,
        )
        self.assertAlmostEqual(schedule.pulse_duty_factor, 0.5)
        self.assertAlmostEqual(
            schedule.gross_capacity_factor,
            schedule.plant_availability * schedule.pulse_duty_factor,
        )

    def test_80_percent_charging_boundaries(self):
        expected = {16.93: 134.0239795397214, 8.47: 227.2389523565343, 4.23: 275.5624312036797}
        planned_max = 8760.0 * (1.0 - 0.80 / 0.93)
        for major, target in expected.items():
            _, core_hours = core_scheduled_outage_hours(major, 6.05)
            charge_limit = (planned_max - core_hours - 432.0) / 2.0
            self.assertAlmostEqual(charge_limit, target, places=9)


if __name__ == "__main__":
    unittest.main()
