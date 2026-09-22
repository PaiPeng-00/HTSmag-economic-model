"""Validate V10 data and recompute published statistics without a physics rerun."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
SCENARIOS = ("S1", "S2", "S3")
NAMES = ("realizations_T4p2_He.parquet", "realizations_T10p0_He.parquet",
         "realizations_T20p0_He.parquet", "realizations_T20p0_H2.parquet")
RAW = [f"{stem}_{s}" for s in SCENARIOS for stem in
       ("Aplant", "E_net_year_MWh", "lcoe_anchor_USD_per_MWh", "r_cryo_re_fraction")]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_manifest(folder):
    manifest = json.loads((folder / "MANIFEST.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        path = (folder / item["path"]).resolve()
        if not path.is_relative_to(folder.resolve()):
            raise ValueError("Manifest path escapes package")
        if not path.is_file() or sha(path) != item["sha256"]:
            raise ValueError(f"Checksum mismatch: {item['path']}")
    return len(manifest["files"])


def hydrate(data):
    """Only copy verified inputs, refusing to overwrite different local files."""
    copies = []
    for path in (data / "model_inputs").rglob("*.xlsx"):
        if path.name == "Data_S1.xlsx":
            continue
        copies.append((path, ROOT / "model/model_code/code/data/raw" /
                       path.relative_to(data / "model_inputs")))
    for path in (data / "figure_data").rglob("*"):
        if path.is_file():
            copies.append((path, ROOT / "results/figure_inputs/current" /
                           path.relative_to(data / "figure_data")))
    for source, target in copies:
        if target.exists() and sha(target) != sha(source):
            raise ValueError(f"Refuse to overwrite changed input: {target.relative_to(ROOT)}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
    return len(copies)


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "pipeline" / filename)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def compare(actual, expected_path, keys=None):
    expected = pd.read_csv(expected_path)
    actual = actual[expected.columns].replace({"": np.nan})
    if keys:
        expected = expected.sort_values(keys).reset_index(drop=True)
        actual = actual.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected.reset_index(drop=True),
                                  check_dtype=False, check_exact=False, rtol=2e-9, atol=2e-9)


def recompute(data, out):
    paths = [data / "full_model_dataset" / name for name in NAMES]
    minima = {s: np.inf for s in SCENARIOS}
    # Scenario reference domains are scenario-specific, not the common-availability set.
    for path in paths:
        frame = pd.read_parquet(path, columns=RAW)
        if len(frame) != 1_476_200:
            raise ValueError(f"Unexpected domain size: {path.name}")
        for s in SCENARIOS:
            lcoe = frame[f"lcoe_anchor_USD_per_MWh_{s}"].to_numpy()
            selected = ((frame[f"Aplant_{s}"].to_numpy() >= .8)
                        & (frame[f"E_net_year_MWh_{s}"].to_numpy() > 0)
                        & np.isfinite(lcoe) & (lcoe > 0))
            minima[s] = min(minima[s], float(lcoe[selected].min()))
    temperatures = {}
    pooled_pass = np.zeros(4, dtype=np.int64)
    margins = np.array([.01, .05, .10, .20])
    avail_total = 0
    n_total = 0
    refrigeration = {}
    for path in paths:
        f = pd.read_parquet(path)
        keys = ["Npw", "rho_turn_uOhm_cm2", "R_joint_nOhm"]
        if f.duplicated(keys).any() or tuple(f[k].nunique() for k in keys) != (200, 61, 121):
            raise ValueError(f"Incomplete or duplicate grid: {path.name}")
        avail = np.ones(len(f), dtype=bool)
        valid = np.ones(len(f), dtype=bool)
        deltas = []
        for s in SCENARIOS:
            lcoe = f[f"lcoe_anchor_USD_per_MWh_{s}"].to_numpy()
            a = f[f"Aplant_{s}"].to_numpy() >= .8
            v = (f[f"E_net_year_MWh_{s}"].to_numpy() > 0) & np.isfinite(lcoe) & (lcoe > 0)
            np.testing.assert_array_equal(a, f[f"availability_pass_{s}"].to_numpy())
            np.testing.assert_array_equal(v, f[f"economic_valid_{s}"].to_numpy())
            avail &= a
            valid &= v
            deltas.append(lcoe / minima[s] - 1)
        delta = np.max(deltas, axis=0)
        np.testing.assert_array_equal(avail, f.availability_pass_all.to_numpy())
        np.testing.assert_array_equal(valid, f.economic_valid_all.to_numpy())
        np.testing.assert_allclose(delta[valid], f.delta_max.to_numpy()[valid], rtol=2e-12, atol=2e-12)
        temp = float(f.Top_K.iloc[0])
        values = delta[avail & valid]
        entry = temperatures.setdefault(temp, {"n_avail": 0, "values": []})
        entry["n_avail"] += int(avail.sum())
        entry["values"].append(values)
        avail_total += int(avail.sum())
        n_total += len(f)
        pooled_pass += [np.count_nonzero(values <= margin) for margin in margins]
        for s in SCENARIOS:
            refrigeration.setdefault((temp, s), []).append(
                100 * f.loc[avail, f"r_cryo_re_fraction_{s}"].to_numpy())
        print(f"Validated raw values, stored flags and penalties: {path.name}", flush=True)
    table_s4 = pd.DataFrame({"margin_pct": margins * 100, "N_pass": pooled_pass,
                            "fraction_all_pct": 100 * pooled_pass / n_total,
                            "fraction_availability_pct": 100 * pooled_pass / avail_total})
    compare(table_s4, data / "reported_values/Table_S4_economic_robustness.csv")
    table_s4.to_csv(out / "Table_S4_economic_robustness.csv", index=False)
    stats = module("fig4_stats", "fig4_lcoe_v10_statistics.py")
    stats.SOURCE = data / "full_model_dataset"
    stats.OUT = out / "Fig4"
    stats.main()
    for name in ("panel_A_scenario_lcoe_reference.csv", "panel_B_temperature_penalty_statistics.csv",
                 "panel_C_temperature_retention_ecdf.csv", "supplementary_figS7_full_penalty_survival.csv"):
        compare(pd.read_csv(stats.OUT / name), data / "figure_data/main_figures/Fig4" / name)
    builders = module("figure_inputs", "build_v10_candidate_figure_inputs.py")
    ref = builders.refrigeration_statistics(paths)
    compare(ref, data / "figure_data/main_figures/Fig3/panel_D_refrigeration_statistics.csv",
            ["scenario", "Top_K"])
    ref.to_csv(out / "Fig3D_refrigeration.csv", index=False)
    print("Rebuilding Fig. 5 A/B conservative boundaries across all 61 contact resistivities...", flush=True)
    heat, boundaries, _ = builders.fig5_from_ledgers(paths, minima)
    compare(heat, data / "figure_data/main_figures/Fig5/panel_A_conservative_heatmaps.csv",
            ["Npw", "Top_K", "R_joint_nOhm"])
    compare(boundaries.loc[boundaries.epsilon.eq(.10)],
            data / "figure_data/main_figures/Fig5/panel_B_conservative_boundaries.csv", ["Npw", "Top_K"])
    compare(boundaries, data / "reported_values/all_margins_conservative_boundaries.csv",
            ["Npw", "Top_K", "epsilon"])
    heat.to_csv(out / "Fig5A.csv", index=False)
    boundaries.to_csv(out / "Fig5B_all_margins.csv", index=False)
    endpoints = boundaries.loc[boundaries.Npw.eq(200) & boundaries.epsilon.eq(.10)]
    headline = {str(t): {"refrigeration_min_percent": float(min(np.min(np.concatenate(v)) for (tt, s), v in refrigeration.items() if tt == t)),
                         "refrigeration_max_percent": float(max(np.max(np.concatenate(v)) for (tt, s), v in refrigeration.items() if tt == t))}
                for t in temperatures}
    return {"status": "PASS", "physical_full_grid_rerun": "NOT RUN",
            "n_realizations": n_total, "n_scenario_evaluations": n_total * 3,
            "n_availability_qualified": avail_total, "scenario_minimum_LCOE": minima,
            "headline_refrigeration": headline,
            "high_current_joint_limits": endpoints[["Top_K", "Rj_tol_nOhm"]].to_dict("records"),
            "checked": ["full data manifest", "unique full-factorial keys", "raw LCOE minima",
                        "raw availability and economic-validity flags", "raw cross-scenario penalty",
                        "Fig3D", "Fig4A/B/C", "FigS7", "Fig5A/B", "TableS4", "all-margin joint limits"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/reproduced")
    parser.add_argument("--hydrate-only", action="store_true")
    args = parser.parse_args()
    data = args.data.resolve()
    n = verify_manifest(data)
    if args.hydrate_only:
        print(json.dumps({"status": "PASS", "verified_files": n, "hydrated_files": hydrate(data)}))
        return
    out = args.output.resolve()
    if out.exists():
        raise SystemExit("Output already exists; select a new --output directory")
    if out.is_relative_to(data):
        raise SystemExit("Outputs must not be written inside the frozen data package")
    out.mkdir(parents=True)
    result = recompute(data, out)
    (out / "VALIDATION.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
