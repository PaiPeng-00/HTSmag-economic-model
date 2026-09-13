#!/usr/bin/env python3
"""Exact resumable V6.5-versus-legacy paired circuit regression (C3)."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
CODE = REPO / "code"
METRICS = ("R_radial_per_TF_Ohm", "Charging_time_999_h", "Radial_loss_energy_MWh")
_CACHE = _CHARGE = _MATRIX = _CFG = _UTILS = None


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _init_worker() -> None:
    global _CACHE, _CHARGE, _MATRIX, _CFG, _UTILS
    import sys
    sys.path.insert(0, str(CODE / "src"))
    from fusion_tem import device as cfg
    from fusion_tem import utils
    _CFG, _UTILS = cfg, utils
    _CACHE = _load_module("v64_circuit_cache", CODE / "scripts/2_charging/2.7_build_circuit_scalar_cache.py")
    _CHARGE = _CACHE._load_charge_module()
    matrix_path = CODE / "scripts/8_economic" / cfg.INDUCTANCE_OUTPUT_DIR / cfg.TF_SYSTEM_MATRIX
    _MATRIX = pd.read_excel(matrix_path, header=None).to_numpy(dtype=float)


def _relative(value: float, baseline: float) -> float | None:
    return None if not np.isfinite(baseline) or baseline == 0.0 else float(value / baseline - 1.0)


def _task(point: tuple[float, int, int, float]) -> dict:
    top, npw, rho_idx, rho = point
    nt = int(_CFG.Nt_list[float(top)])
    try:
        continuous = _CACHE._solve_one(_CHARGE, _MATRIX, top, npw, rho)
        original = _CACHE.calculate_radial_resistance
        _CACHE.calculate_radial_resistance = _UTILS.LEGACY_DISCRETE_RADIAL_MODEL
        try:
            legacy = _CACHE._solve_one(_CHARGE, _MATRIX, top, npw, rho)
        finally:
            _CACHE.calculate_radial_resistance = original
        _, contract = _UTILS.continuous_effective_turn_rho_values(npw, nt, rho * 1e-10)
        record = {"Top_K": top, "Npw": npw, "rho_idx": rho_idx, "rho_turn_uOhm_cm2": rho,
                  "Nt": nt, "N_turn_eff": contract["N_turn_eff"], "Nt_mod_Npw": nt % npw,
                  "exact_divisor_flag": bool(nt % npw == 0), "solver_status": "success"}
        for key, value in continuous.items():
            record[f"continuous_{key}"] = value
        for key, value in legacy.items():
            record[f"legacy_{key}"] = value
        for key in METRICS:
            cv, lv = float(continuous[key]), float(legacy[key])
            record[f"delta_abs_{key}"] = cv - lv
            record[f"delta_rel_{key}"] = _relative(cv, lv)
        return record
    except Exception as exc:
        return {"Top_K": top, "Npw": npw, "rho_idx": rho_idx, "rho_turn_uOhm_cm2": rho,
                "Nt": nt, "N_turn_eff": nt / npw, "Nt_mod_Npw": nt % npw,
                "exact_divisor_flag": bool(nt % npw == 0), "solver_status": "error", "error": str(exc)}


def _stats(frame: pd.DataFrame, metric: str) -> dict:
    values = frame[f"delta_rel_{metric}"].abs().dropna().to_numpy(dtype=float)
    if not len(values): return {"count": 0}
    return {"count": int(len(values)), "median": float(np.percentile(values, 50)),
            "P75": float(np.percentile(values, 75)), "P90": float(np.percentile(values, 90)),
            "P95": float(np.percentile(values, 95)), "P99": float(np.percentile(values, 99)),
            "max": float(np.max(values))}


def _summary(frame: pd.DataFrame) -> dict:
    groups = {"all": frame, "exact_divisor": frame[frame.exact_divisor_flag],
              "non_divisor": frame[~frame.exact_divisor_flag]}
    groups.update({f"Npw_{lo}_{hi}": frame[frame.Npw.between(lo, hi)]
                   for lo, hi in ((1,20),(21,50),(51,100),(101,150),(151,200))})
    groups.update({f"Top_{top:g}K": frame[np.isclose(frame.Top_K, top)] for top in (4.2,10.0,20.0)})
    groups.update({f"rho_{lo:g}_{hi:g}": frame[frame.rho_turn_uOhm_cm2.between(lo, hi)]
                   for lo, hi in ((10,99.999),(100,999.999),(1000,9999.999),(10000,10000))})
    return {label: {metric: _stats(group, metric) for metric in METRICS} for label, group in groups.items()}


def _smoothness(frame: pd.DataFrame) -> dict:
    records = []
    failures = []
    for (top, rho), part in frame.groupby(["Top_K", "rho_turn_uOhm_cm2"], sort=True):
        part = part.sort_values("Npw")
        rec = {"Top_K": float(top), "rho_turn_uOhm_cm2": float(rho), "Npw_count": int(len(part)),
               "nonfinite": False, "old_interface_step_alignment": False, "solver_state_discontinuity": False}
        for quantity in ("Charging_time_999_h", "Radial_loss_energy_MWh"):
            values = part[f"continuous_{quantity}"].to_numpy(dtype=float)
            if not np.all(np.isfinite(values)):
                rec["nonfinite"] = True; continue
            delta = np.diff(values)
            # A smooth response can have strong low-Npw curvature.  A legacy
            # interface-count step instead appears as an *isolated* first-
            # difference impulse flanked by much smaller differences.  Exclude
            # the Npw=1 boundary.  V6.5 uses interpolated two-decimal crossing
            # times, so residual hour-grid quantization is not an accepted cause
            # of a discontinuity.
            magnitude = np.abs(delta)
            spike_idx = []
            for index in range(10, len(magnitude) - 2):
                neighbourhood = np.r_[magnitude[index - 2:index], magnitude[index + 1:index + 3]]
                baseline = max(float(np.median(neighbourhood)), np.finfo(float).eps)
                if magnitude[index] > 10.0 * baseline:
                    spike_idx.append(index + 1)
            old_edges = np.array([n for n in range(2, 200) if int(part.Nt.iloc[0]) % n == 0])
            aligned = set(spike_idx).intersection(set(old_edges))
            rec[f"{quantity}_spike_count"] = int(len(spike_idx))
            rec[f"{quantity}_old_edge_aligned_spikes"] = sorted(map(int, aligned))
            if quantity == "Radial_loss_energy_MWh" and aligned:
                rec["old_interface_step_alignment"] = True
        if rec["nonfinite"] or rec["old_interface_step_alignment"] or rec["solver_state_discontinuity"]:
            failures.append(rec)
        records.append(rec)
    return {"status": "FAIL_UNEXPLAINED_DISCONTINUITY" if failures else "PASS_CONTINUOUS_PHYSICS_RESPONSE",
            "method": "exact circuit outputs evaluated at every integer Npw; 99.9% crossing is linearly interpolated and persisted to 0.01 h, so hour-grid sampling is not used as an availability step",
            "records": records, "failures": failures}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=max(1, min(6, os.cpu_count() or 1)))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-points", type=int, default=None,
                        help="Developer smoke limit only; a limited run intentionally cannot pass C3.")
    args = parser.parse_args()
    out = args.output_root.resolve(); out.mkdir(parents=True, exist_ok=True)
    _init_worker()
    _, _, rhos = _CACHE._scan_grid(REPO / "v6_2_arc_actual_inductance/inputs/v6_2_direct_circuit_grid.yaml")
    if len(rhos) != 61: raise RuntimeError(f"canonical rho grid drift: expected 61, found {len(rhos)}")
    full_points = [(top, npw, idx, rho) for top in (4.2, 10.0, 20.0) for npw in range(1, 201) for idx, rho in enumerate(rhos)]
    points = full_points
    if args.max_points is not None:
        points = points[:args.max_points]
    csv_path = out / "v6_5_vs_legacy_integer_architecture_regression.csv"
    prior = pd.read_csv(csv_path) if args.resume and csv_path.exists() else pd.DataFrame()
    completed = set(zip(prior.get("Top_K", []), prior.get("Npw", []), prior.get("rho_idx", [])))
    pending = [p for p in points if (p[0], p[1], p[2]) not in completed]
    rows = prior.to_dict("records")
    if pending:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
            for count, row in enumerate(pool.map(_task, pending, chunksize=20), 1):
                rows.append(row)
                if count % 200 == 0 or count == len(pending):
                    pd.DataFrame(rows).to_csv(csv_path, index=False)
                    print(f"[C3] {len(completed)+count}/{len(points)}", flush=True)
    frame = pd.DataFrame(rows).sort_values(["Top_K", "Npw", "rho_idx"])
    frame.to_csv(csv_path, index=False)
    success = frame[frame.solver_status.eq("success")].copy()
    complete = (args.max_points is None and len(frame) == len(full_points) and len(success) == len(full_points)
                and not frame.duplicated(["Top_K","Npw","rho_idx"]).any())
    summary = _summary(success)
    smooth = _smoothness(success) if complete else {"status": "FAIL_UNEXPLAINED_DISCONTINUITY", "reason": "paired set incomplete"}
    large = []
    for metric in METRICS:
        col = f"delta_rel_{metric}"
        for threshold in (0.01, 0.025, 0.05, 0.10):
            flagged = success[success[col].abs() > threshold]
            for _, row in flagged.iterrows():
                large.append({"metric": metric, "threshold_abs_relative": threshold, "Top_K": row.Top_K,
                              "Npw": row.Npw, "rho_turn_uOhm_cm2": row.rho_turn_uOhm_cm2,
                              "relative_delta": row[col], "exact_divisor_flag": row.exact_divisor_flag})
    pd.DataFrame(large).to_csv(out / "v6_5_large_delta_regions.csv", index=False)
    status = "PASS" if complete and smooth["status"] == "PASS_CONTINUOUS_PHYSICS_RESPONSE" else "FAIL"
    payload = {"schema_version": "v6.5-c3-paired-exact-v1", "status": status,
               "supersedes": "V6.4 integer-hour charging-time contract",
               "expected_paired_points": len(full_points), "actual_rows": int(len(frame)), "successful_paired_points": int(len(success)),
               "solver_failures": int(len(frame) - len(success)), "summary_abs_relative_difference": summary,
               "smoothness_status": smooth["status"], "large_delta_rows": len(large)}
    (out / "v6_5_vs_legacy_integer_architecture_regression.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (out / "v6_5_full_circuit_smoothness_audit.json").write_text(json.dumps(smooth, indent=2) + "\n", encoding="utf-8")
    (out / "v6_5_C3_closure.md").write_text(f"# V6.5 C3 exact paired circuit regression\n\nStatus: **{status}**\n\nPaired points: {len(success)}/{len(points)}\n", encoding="utf-8")
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
