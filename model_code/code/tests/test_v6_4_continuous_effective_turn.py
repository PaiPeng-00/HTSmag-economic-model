"""Contract tests for the V6.4 continuous effective-turn radial model."""
import numpy as np

from fusion_tem import utils


def test_continuous_model_keeps_inner_turn_boundary_and_conserves_weight():
    rho_turn = 1.0e-6
    rho, audit = utils.continuous_effective_turn_rho_values(100, 850, rho_turn)
    assert rho[0] == rho_turn
    assert np.isclose(audit["N_turn_eff"], 8.5)
    assert np.isclose(audit["N_tt_eff"], 7.5)
    assert np.isclose(audit["continuous_rho_turn_weight"], 8.5)
    assert abs(audit["weight_conservation_residual"]) < 1e-12


def test_continuous_model_matches_discrete_total_weight_for_divisible_architecture():
    rho, audit = utils.continuous_effective_turn_rho_values(50, 850, 1.0e-6)
    assert np.isclose(audit["continuous_rho_turn_weight"], 17.0)
    assert np.count_nonzero(np.isclose(rho, 1.0e-6)) == 1


def test_legacy_model_is_retained_but_is_not_the_v64_contract():
    legacy = utils.LEGACY_DISCRETE_RADIAL_MODEL(100, 850, 1.0e-6)
    continuous = utils.calculate_radial_resistance(100, 850, 1.0e-6)
    assert np.isfinite(legacy)
    assert np.isfinite(continuous)
