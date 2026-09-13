"""Near-optimal conductor-architecture regret and robust-window analysis."""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


DEFAULT_SCENARIOS = ("S1", "S2", "S3")
DEFAULT_REGRET_THRESHOLDS = (0.025, 0.05, 0.10)
ARCHITECTURE_COLUMNS = ("Npw", "R_joint_nOhm")
REOPTIMIZED_COLUMNS = ("rho_turn_uOhm_cm2", "Top_K", "coolant")
LCOE_COLUMN = "LCOE_plant_USD_per_MWh"


def _require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = sorted(set(columns).difference(df.columns))
    if missing:
        raise KeyError(f"Missing near-optimal columns: {missing}")


def compute_scenario_architecture_regret(
    df: pd.DataFrame,
    *,
    scenarios: Iterable[str] = DEFAULT_SCENARIOS,
) -> pd.DataFrame:
    """Reoptimize nuisance variables and compute architecture regret.

    For each scenario and each conductor architecture (Npw, R_joint_nOhm), the
    jointly feasible minimum LCOE is selected over rho_turn, operating
    temperature, and coolant. Regret is measured against the scenario-wide
    jointly feasible minimum.
    """
    scenarios = tuple(scenarios)
    required = (
        "scenario",
        "feasible_joint",
        LCOE_COLUMN,
        *ARCHITECTURE_COLUMNS,
        *REOPTIMIZED_COLUMNS,
    )
    _require_columns(df, required)

    work = df.loc[df["scenario"].isin(scenarios)].copy()
    lcoe = pd.to_numeric(work[LCOE_COLUMN], errors="coerce")
    valid = work["feasible_joint"].astype(bool) & np.isfinite(lcoe)
    work = work.loc[valid].copy()
    work[LCOE_COLUMN] = lcoe.loc[valid]

    missing_scenarios = sorted(set(scenarios) - set(work["scenario"].unique()))
    if missing_scenarios:
        raise ValueError(
            "No jointly feasible rows for scenarios: " + ", ".join(missing_scenarios)
        )

    sort_columns = [
        "scenario",
        *ARCHITECTURE_COLUMNS,
        LCOE_COLUMN,
        *REOPTIMIZED_COLUMNS,
    ]
    best = (
        work.sort_values(sort_columns, kind="mergesort")
        .groupby(["scenario", *ARCHITECTURE_COLUMNS], as_index=False, sort=True)
        .first()
    )
    scenario_min = best.groupby("scenario")[LCOE_COLUMN].min()
    best["scenario_LCOE_min_USD_per_MWh"] = best["scenario"].map(scenario_min)
    best["delta_LCOE_architecture_USD_per_MWh"] = (
        best[LCOE_COLUMN] - best["scenario_LCOE_min_USD_per_MWh"]
    )
    best["regret_fraction"] = (
        best["delta_LCOE_architecture_USD_per_MWh"]
        / best["scenario_LCOE_min_USD_per_MWh"]
    )
    best["regret_pct"] = 100.0 * best["regret_fraction"]

    if not np.allclose(
        best.groupby("scenario")["regret_fraction"].min().to_numpy(),
        0.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise AssertionError("Scenario regret minima are not zero")
    return best


def compute_robust_architecture_window(
    regret_table: pd.DataFrame,
    *,
    scenarios: Iterable[str] = DEFAULT_SCENARIOS,
    thresholds: Iterable[float] = DEFAULT_REGRET_THRESHOLDS,
) -> pd.DataFrame:
    """Return worst-case regret and robust-window flags for each architecture."""
    scenarios = tuple(scenarios)
    thresholds = tuple(float(value) for value in thresholds)
    _require_columns(
        regret_table,
        ("scenario", "regret_fraction", *ARCHITECTURE_COLUMNS),
    )

    pivot = regret_table.pivot_table(
        index=list(ARCHITECTURE_COLUMNS),
        columns="scenario",
        values="regret_fraction",
        aggfunc="min",
    )
    for scenario in scenarios:
        if scenario not in pivot.columns:
            pivot[scenario] = np.nan
    pivot = pivot.loc[:, list(scenarios)]
    complete = pivot.notna().all(axis=1)

    out = pivot.rename(
        columns={scenario: f"regret_{scenario}_fraction" for scenario in scenarios}
    ).reset_index()
    regret_columns = [f"regret_{scenario}_fraction" for scenario in scenarios]
    out["scenario_complete"] = complete.to_numpy()
    out["worst_case_regret_fraction"] = out[regret_columns].max(
        axis=1, skipna=False
    )
    out["worst_case_regret_pct"] = 100.0 * out["worst_case_regret_fraction"]

    for threshold in thresholds:
        label = f"{100.0 * threshold:g}".replace(".", "p")
        out[f"robust_window_{label}pct"] = (
            out["scenario_complete"]
            & (out["worst_case_regret_fraction"] <= threshold)
        )
    return out.sort_values(
        ["worst_case_regret_fraction", *ARCHITECTURE_COLUMNS],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)


def summarize_robust_windows(
    robust_table: pd.DataFrame,
    *,
    thresholds: Iterable[float] = DEFAULT_REGRET_THRESHOLDS,
) -> pd.DataFrame:
    """Summarize robust counts and Npw/Rj bounds for each regret threshold."""
    rows = []
    for threshold in (float(value) for value in thresholds):
        label = f"{100.0 * threshold:g}".replace(".", "p")
        flag = f"robust_window_{label}pct"
        if flag not in robust_table.columns:
            raise KeyError(f"Missing robust-window column: {flag}")
        selected = robust_table.loc[robust_table[flag].astype(bool)]
        rows.append(
            {
                "regret_threshold_fraction": threshold,
                "regret_threshold_pct": 100.0 * threshold,
                "robust_architecture_count": int(len(selected)),
                "Npw_min": selected["Npw"].min() if not selected.empty else np.nan,
                "Npw_max": selected["Npw"].max() if not selected.empty else np.nan,
                "R_joint_nOhm_min": (
                    selected["R_joint_nOhm"].min() if not selected.empty else np.nan
                ),
                "R_joint_nOhm_max": (
                    selected["R_joint_nOhm"].max() if not selected.empty else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)

def exact_robust_frontier_at_npw(
    robust_table: pd.DataFrame,
    *,
    npw: int = 200,
    threshold: float = 0.05,
    scenarios: Iterable[str] = DEFAULT_SCENARIOS,
) -> pd.DataFrame:
    """Return the scanned 5% frontier and the next larger excluded Rj point.

    This is deliberately a grid threshold. No interpolation is performed.
    Regret values are fractions, so 0.05 means 5%.
    """
    scenarios = tuple(scenarios)
    regret_columns = [f"regret_{scenario}_fraction" for scenario in scenarios]
    _require_columns(
        robust_table,
        (
            *ARCHITECTURE_COLUMNS,
            "scenario_complete",
            "worst_case_regret_fraction",
            *regret_columns,
        ),
    )
    rows = robust_table.loc[
        pd.to_numeric(robust_table["Npw"], errors="coerce").eq(int(npw))
    ].copy()
    rows = rows.sort_values("R_joint_nOhm", kind="mergesort")
    if rows.empty:
        raise ValueError(f"No robust-architecture rows at Npw={npw}")
    retained = rows.loc[
        rows["scenario_complete"].astype(bool)
        & (
            pd.to_numeric(
                rows["worst_case_regret_fraction"], errors="coerce"
            )
            <= float(threshold) + 1e-15
        )
    ]
    if retained.empty:
        raise ValueError(
            f"No scanned Rj point at Npw={npw} satisfies {100*threshold:g}% regret"
        )
    kept = retained.iloc[-1]
    larger = rows.loc[rows["R_joint_nOhm"] > kept["R_joint_nOhm"]]
    if larger.empty:
        raise ValueError(
            f"No next larger scanned Rj point above the retained Npw={npw} frontier"
        )
    excluded = larger.iloc[0]
    if bool(excluded["scenario_complete"]) and (
        float(excluded["worst_case_regret_fraction"])
        <= float(threshold) + 1e-15
    ):
        raise AssertionError("The next larger Rj point is unexpectedly retained")

    kept_regrets = {scenario: float(kept[column]) for scenario, column in zip(scenarios, regret_columns)}
    excluded_regrets = {
        scenario: float(excluded[column])
        for scenario, column in zip(scenarios, regret_columns)
    }
    binding = max(scenarios, key=lambda scenario: kept_regrets[scenario])
    excluded_binding = max(
        scenarios, key=lambda scenario: excluded_regrets[scenario]
    )
    record: dict[str, object] = {
        "Npw": int(npw),
        "Rj_max_retained_nOhm": float(kept["R_joint_nOhm"]),
        "Rj_next_excluded_nOhm": float(excluded["R_joint_nOhm"]),
        **{f"regret_{scenario}": kept_regrets[scenario] for scenario in scenarios},
        "max_regret": float(kept["worst_case_regret_fraction"]),
        "binding_scenario": binding,
        **{
            f"next_excluded_regret_{scenario}": excluded_regrets[scenario]
            for scenario in scenarios
        },
        "next_excluded_max_regret": float(
            excluded["worst_case_regret_fraction"]
        ),
        "next_excluded_binding_scenario": excluded_binding,
        "threshold_fraction": float(threshold),
        "interpolated": False,
    }
    return pd.DataFrame([record])
