"""Cryoplant conversion models used by the E0-E1 experiments."""
from __future__ import annotations

from typing import Mapping

import numpy as np


CRYO_REFERENCE_TEMPERATURE_K = 4.5
GREEN_COEFFICIENT = 0.155
GREEN_EXPONENT = 0.23
DEFAULT_CRYO_EFFICIENCY_MODEL = "green_rated"
DEFAULT_CRYO_ETA_MAX = 0.30
DEFAULT_CRYO_RATED_MARGIN = 1.0
VALID_CRYO_EFFICIENCY_MODELS = {"legacy_ter_brake", "green_rated"}


def carnot_cop(T_hot: float, T_cold: float) -> float:
    """Return the ideal refrigerator COP."""
    T_hot = float(T_hot)
    T_cold = float(T_cold)
    if not (0.0 < T_cold < T_hot):
        raise ValueError(
            f"temperatures must satisfy 0 < T_cold < T_hot; got {T_cold}, {T_hot}"
        )
    return T_cold / (T_hot - T_cold)


def equivalent_4p5K_heat_W(
    loads_by_temperature_W: Mapping[float, object],
    *,
    T_hot: float = 300.0,
):
    """Convert simultaneous cold-end loads to a 4.5 K-equivalent load."""
    reference_cop = carnot_cop(T_hot, CRYO_REFERENCE_TEMPERATURE_K)
    total = None
    for temperature_K, load_W in loads_by_temperature_W.items():
        load = np.asarray(load_W, dtype=float)
        if not np.all(np.isfinite(load)) or np.any(load < 0.0):
            raise ValueError("cold-end heat loads must be finite and non-negative")
        term = load * reference_cop / carnot_cop(T_hot, float(temperature_K))
        total = term if total is None else total + term
    return 0.0 if total is None else total


def build_rated_context(
    simultaneous_loads: list[Mapping[float, object]],
    *,
    N_tf: int,
    efficiency_model: str = DEFAULT_CRYO_EFFICIENCY_MODEL,
    eta_max: float = DEFAULT_CRYO_ETA_MAX,
    rated_margin: float = DEFAULT_CRYO_RATED_MARGIN,
    T_hot: float = 300.0,
) -> dict:
    """Size one cryoplant from the maximum simultaneous full-system load."""
    if int(N_tf) <= 0:
        raise ValueError("N_tf must be positive")
    if efficiency_model not in VALID_CRYO_EFFICIENCY_MODELS:
        raise ValueError(f"unsupported cryogenic efficiency model: {efficiency_model!r}")
    eta_max = float(eta_max)
    rated_margin = float(rated_margin)
    if not (0.0 < eta_max <= 1.0) or rated_margin <= 0.0:
        raise ValueError("eta_max must be in (0, 1] and rated_margin must be positive")
    peak_4p5eq_W = (
        0.0
        if not simultaneous_loads
        else max(
            float(np.max(equivalent_4p5K_heat_W(loads, T_hot=T_hot)))
            for loads in simultaneous_loads
        )
    )
    rated_4p5eq_kW = rated_margin * peak_4p5eq_W / 1000.0
    eta_raw = (
        0.0
        if rated_4p5eq_kW == 0.0
        else GREEN_COEFFICIENT * rated_4p5eq_kW ** GREEN_EXPONENT
    )
    eta_rated = min(eta_raw, eta_max, 1.0) if eta_raw > 0.0 else 0.0
    return {
        "cryo_efficiency_model": efficiency_model,
        "cryo_reference_temperature_K": CRYO_REFERENCE_TEMPERATURE_K,
        "cryo_rated_4p5eq_kW": rated_4p5eq_kW,
        "cryo_eta_raw_fraction_carnot": eta_raw,
        "cryo_eta_cap_fraction_carnot": eta_max,
        "cryo_eta_rated_fraction_carnot": eta_rated,
        "cryo_eta_cap_active": bool(eta_raw > eta_max),
        "cryo_rated_margin": rated_margin,
        "cryo_part_load_factor": 1.0,
        "N_tf": int(N_tf),
    }


def electrical_power_W(
    loads_by_temperature_W: Mapping[float, object],
    rated_context: Mapping[str, object],
    *,
    T_hot: float = 300.0,
):
    """Convert cold loads with the design's one fixed rated Green efficiency."""
    loads = []
    ideal_power = None
    for temperature_K, load_W in loads_by_temperature_W.items():
        load = np.asarray(load_W, dtype=float)
        if not np.all(np.isfinite(load)) or np.any(load < 0.0):
            raise ValueError("cold-end heat loads must be finite and non-negative")
        loads.append(load)
        term = load / carnot_cop(T_hot, float(temperature_K))
        ideal_power = term if ideal_power is None else ideal_power + term
    if ideal_power is None or all(np.all(load == 0.0) for load in loads):
        return 0.0 if ideal_power is None else np.zeros_like(ideal_power, dtype=float)
    eta = float(rated_context["cryo_eta_rated_fraction_carnot"])
    eta_cap = float(rated_context["cryo_eta_cap_fraction_carnot"])
    if not (0.0 < eta <= eta_cap <= 1.0):
        raise ValueError("rated cryogenic efficiency is outside its valid range")
    return ideal_power / eta
