import math

from fusion_tem.cryo.heat_load import coil_internal_joint_heat


def test_internal_splice_heat_uses_one_equivalent_contact_per_location():
    for npw in (1, 5, 50, 200):
        expected = (1000.0 / (200.0 * npw)) * 500.0**2 * 10.0e-9
        value = coil_internal_joint_heat(
            Npw=npw,
            L_total_m=1000.0,
            Ip=500.0,
            L_single_tape=200.0,
            R_ss=10.0e-9,
        )
        assert math.isclose(value, expected, rel_tol=0.0, abs_tol=1.0e-15)
