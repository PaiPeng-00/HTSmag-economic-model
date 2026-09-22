"""Derive Figure 4 LCOE-only statistics from the frozen V10 realization ledgers."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/model_data/current/full_model_dataset"
OUT = ROOT / "results/figure_inputs/current/main_figures/Fig4"
FILES = [
    (4.2, "He", "realizations_T4p2_He.parquet"),
    (10.0, "He", "realizations_T10p0_He.parquet"),
    (20.0, "He", "realizations_T20p0_He.parquet"),
    (20.0, "H2", "realizations_T20p0_H2.parquet"),
]
MARGINS = (0.01, 0.05, 0.10, 0.20)


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    by_temp: dict[float, dict] = {}
    scenario_minima = {scenario: float("inf") for scenario in ("S1", "S2", "S3")}
    inputs = []
    for temp, coolant, name in FILES:
        path = SOURCE / name
        inputs.append({"file": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        extra = [f"{column}_{scenario}" for scenario in ("S1", "S2", "S3")
                 for column in ("availability_pass", "economic_valid", "lcoe_anchor_USD_per_MWh")]
        table = pq.read_table(path, columns=["availability_pass_all", "economic_valid_all", "delta_max", *extra])
        for scenario in scenario_minima:
            scenario_available = table.column(f"availability_pass_{scenario}").to_numpy(zero_copy_only=False).astype(bool)
            scenario_valid = table.column(f"economic_valid_{scenario}").to_numpy(zero_copy_only=False).astype(bool)
            lcoe = table.column(f"lcoe_anchor_USD_per_MWh_{scenario}").to_numpy(zero_copy_only=False)
            scenario_minima[scenario] = min(scenario_minima[scenario],
                                            float(np.min(lcoe[scenario_available & scenario_valid])))
        availability = table.column("availability_pass_all").to_numpy(zero_copy_only=False).astype(bool)
        valid = table.column("economic_valid_all").to_numpy(zero_copy_only=False).astype(bool)
        delta = table.column("delta_max").to_numpy(zero_copy_only=False)
        finite = np.isfinite(delta)
        selected = delta[availability & valid & finite]
        if selected.size != int(np.count_nonzero(availability & valid)):
            raise ValueError(f"Valid LCOE does not match finite delta_max: {name}")
        entry = by_temp.setdefault(temp, {"availability": 0, "valid": [], "coolants": []})
        entry["availability"] += int(np.count_nonzero(availability))
        entry["valid"].append(selected)
        entry["coolants"].append(coolant)

    stats = []
    cdf = []
    full_tail = []
    pooled_valid = []
    pooled_availability = 0
    for temp in (4.2, 10.0, 20.0):
        entry = by_temp[temp]
        values = np.sort(np.concatenate(entry["valid"]))
        denominator = entry["availability"]
        pooled_valid.append(values)
        pooled_availability += denominator
        p1, p5, p10, median, p90, p95, p99 = np.percentile(
            values, [1, 5, 10, 50, 90, 95, 99], method="linear")
        row = {
            "temperature_K": temp,
            "coolants": "+".join(entry["coolants"]),
            "availability_qualified_n": denominator,
            "valid_all_scenarios_n": len(values),
            "invalid_lcoe_n": denominator - len(values),
            "minimum_percent": float(values[0] * 100),
            "P1_percent": float(p1 * 100),
            "P5_percent": float(p5 * 100),
            "P10_percent": float(p10 * 100),
            "median_percent": float(median * 100),
            "P90_percent": float(p90 * 100),
            "P95_percent": float(p95 * 100),
            "P99_percent": float(p99 * 100),
            "maximum_percent": float(values[-1] * 100),
        }
        for margin in MARGINS:
            row[f"retention_{int(margin * 100)}pct_percent"] = float(
                np.searchsorted(values, margin, side="right") / denominator * 100
            )
        stats.append(row)
        # Evaluate the exact empirical count on a 0.01%-point display grid.
        # The operation is not a smoothing or interpolation of the population.
        thresholds_percent = np.arange(2001, dtype=float) / 100.0
        counts = np.searchsorted(values, thresholds_percent / 100.0, side="right")
        cdf.extend({"temperature_K": temp,
                    "allowed_LCOE_increase_percent": float(threshold),
                    "economic_retention_percent": float(count / denominator * 100)}
                   for threshold, count in zip(thresholds_percent, counts))
        # Supplementary Fig. S7 shows the full valid-LCOE survival distribution,
        # including the finite extreme tail. No invalid-LCOE penalty is imputed.
        tail_thresholds_percent = np.geomspace(.3, float(values[-1] * 100), 1200)
        tail_counts = len(values) - np.searchsorted(values, tail_thresholds_percent / 100,
                                                    side="left")
        full_tail.extend({"temperature_K": temp,
                          "LCOE_increase_threshold_percent": float(threshold),
                          "valid_realizations_at_or_above_threshold_n": int(count),
                          "fraction_of_valid_realizations_at_or_above_percent": float(count / len(values) * 100)}
                         for threshold, count in zip(tail_thresholds_percent, tail_counts))

    pooled = np.concatenate(pooled_valid)
    pooled_retention = {
        f"{int(margin * 100)}pct": float(np.count_nonzero(pooled <= margin) / pooled_availability * 100)
        for margin in MARGINS
    }
    reference_rows = [{"scenario": scenario,
                       "minimum_LCOE_USD_per_MWh": minimum,
                       "LCOE_at_10pct_margin_USD_per_MWh": minimum * 1.10,
                       "LCOE_at_20pct_margin_USD_per_MWh": minimum * 1.20}
                      for scenario, minimum in scenario_minima.items()]
    write_csv(OUT / "panel_A_scenario_lcoe_reference.csv", list(reference_rows[0]), reference_rows)
    write_csv(OUT / "panel_B_temperature_penalty_statistics.csv", list(stats[0]), stats)
    write_csv(OUT / "panel_C_temperature_retention_ecdf.csv", list(cdf[0]), cdf)
    write_csv(OUT / "supplementary_figS7_full_penalty_survival.csv", list(full_tail[0]), full_tail)
    metadata = {
        "source": "V10 full_model_dataset complete realization ledgers",
        "inputs": inputs,
        "temperature_statistics": stats,
        "scenario_lcoe_references": reference_rows,
        "pooled_availability_qualified_n": pooled_availability,
        "pooled_valid_all_scenarios_n": len(pooled),
        "pooled_retention_percent": pooled_retention,
        "definitions": {
            "A": "scenario-specific minimum among availability-qualified and valid realizations, all four V10 ledgers",
            "B": "availability_pass_all and economic_valid_all and finite delta_max, per temperature; 20 K He+H2 pooled; main-figure thin interval P5-P95",
            "C": "same numerator at each threshold, denominator all availability_pass_all at that temperature including invalid LCOE",
            "quantile": "numpy percentile, linear interpolation",
            "ecdf": "exact unsmoothed empirical counts evaluated at a 0.01 percentage-point display grid",
            "S7": "full valid-LCOE survival counts evaluated on a logarithmic display grid, no smoothing",
        },
    }
    (OUT / "temperature_lcoe_statistics.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"scenario_lcoe_references": reference_rows,
                      "temperature_statistics": stats, "pooled_retention_percent": pooled_retention,
                      "ecdf_rows": len(cdf), "supplementary_full_tail_rows": len(full_tail)}, indent=2))


if __name__ == "__main__":
    main()
