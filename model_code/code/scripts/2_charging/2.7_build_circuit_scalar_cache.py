"""Build a compact, exact circuit cache for a scan grid.

The cache contains one row per physical circuit design ``(Top, Npw, rho_turn)``.
It deliberately stores scalar diagnostics only: the main economic scan must not
fall back to a nearest-neighbour Data S2 radial-loss trace.

The radial quantities are on a *per-TF-magnet* cold-end basis, matching
``calculate_charge_cryo_electrical_energy``.  ``P_R`` returned by the 18-by-18
circuit solver is summed over its 18 branches and divided by ``cfg.Ntf``.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from fusion_tem import device as cfg
from fusion_tem.utils import calculate_radial_resistance


def _load_charge_module():
    module_path = Path(__file__).with_name("2.5_charge_time999_TF_system_all_Npw=1-200.py")
    spec = importlib.util.spec_from_file_location("arc_charge_time999", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load circuit solver: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _scan_grid(config_path: Path) -> tuple[list[float], list[int], list[float]]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    temperatures = sorted({float(pair[0]) for pair in raw["temp_coolant_pairs"]})
    npw_values = [int(value) for value in raw["npw_values"]]
    rho_spec = raw["rho_turn_uohm_cm2_values"]
    if isinstance(rho_spec, dict):
        logspace = rho_spec["logspace"]
        values = np.logspace(
            np.log10(float(logspace["start"])),
            np.log10(float(logspace["stop"])),
            int(logspace["num"]),
        )
        values = np.concatenate([values, np.asarray(logspace.get("include", []), dtype=float)])
        rho_values = sorted(set(float(value) for value in values))
    else:
        rho_values = [float(value) for value in rho_spec]
    return temperatures, npw_values, rho_values


def _event_slice(time_h: np.ndarray, power_w: np.ndarray, event_end_h: float) -> tuple[np.ndarray, np.ndarray]:
    """Clip a solved trace to 99.9% charge, including an exact end point."""
    if not np.isfinite(event_end_h) or event_end_h <= 0.0:
        raise ValueError("99.9% charge time is non-finite or non-positive")
    if event_end_h > float(time_h[-1]):
        raise ValueError("99.9% charge time exceeds solver time horizon")
    keep = time_h < event_end_h
    event_time = np.concatenate([time_h[keep], [event_end_h]])
    event_power = np.concatenate([
        power_w[keep],
        [float(np.interp(event_end_h, time_h, power_w))],
    ])
    if event_time.size < 2:
        raise ValueError("charge-event trace has fewer than two samples")
    return event_time, event_power


def _solve_one(
    charge_module,
    base_l_matrix: np.ndarray,
    top_k: float,
    npw: int,
    rho_turn_uohm_cm2: float,
) -> dict:
    ip = float(cfg.Ip_list[top_k])
    ntape_coil = int(cfg.Nt_list[top_k])
    l_matrix = base_l_matrix * (ntape_coil * cfg.NP / npw) ** 2
    r_per_tf = calculate_radial_resistance(
        Npw=npw,
        Ntape_coil=ntape_coil,
        rho_turn=float(rho_turn_uohm_cm2) * 1e-10,
    ) * cfg.NP
    # Cover 12 slowest e-folding times of the post-ramp L^-1 R relaxation.
    relaxation = np.linalg.solve(l_matrix, np.diag([r_per_tf] * cfg.Ntf))
    rates = np.linalg.eigvals(relaxation)
    positive_rates = np.real(rates[np.real(rates) > 0.0])
    if positive_rates.size != cfg.Ntf:
        raise RuntimeError("non-positive circuit relaxation rate")
    slow_tau_h = 1.0 / float(np.min(positive_rates)) / 3600.0
    steady_hours = max(float(cfg.STEADY_HOURS), 12.0 * slow_tau_h)
    time_s, i_l, _, p_radial_branches, _ = charge_module.simulate_charging_system(
        l_matrix,
        [r_per_tf] * cfg.Ntf,
        npw=npw,
        I_target_local=ip,
        Ntape_coil_local=ntape_coil,
        steady_hours=steady_hours,
    )
    if time_s is None or i_l is None or p_radial_branches is None:
        raise RuntimeError("circuit solver returned no solution")
    charge_999_h = charge_module.calculate_time_to_999(
        time_s, i_l, npw=npw, I_target_per_conductor=ip,
    )
    time_h = np.asarray(time_s, dtype=float) / 3600.0
    # Each branch represents one TF magnet.  The scan heat model expects the
    # cold-end loss of one TF, not the sum of the 18-magnet system.
    radial_power_per_tf_w = np.sum(np.asarray(p_radial_branches, dtype=float), axis=1) / float(cfg.Ntf)
    if charge_999_h is None:
        raise RuntimeError(f"99.9% crossing absent after adaptive {steady_hours:.3g} h horizon")
    event_time_h, event_power_w = _event_slice(time_h, radial_power_per_tf_w, float(charge_999_h))
    energy_mwh = float(np.trapz(event_power_w, event_time_h) / 1e6)
    return {
        "Charging_time_999_h": float(charge_999_h),
        "Radial_loss_at_charge_W": float(np.interp(cfg.CHARGE_HOURS, time_h, radial_power_per_tf_w)),
        "Radial_loss_peak_W": float(np.max(event_power_w)),
        "Radial_loss_energy_MWh": energy_mwh,
        "Radial_loss_average_W": energy_mwh * 1e6 / float(charge_999_h),
        "R_radial_per_TF_Ohm": float(r_per_tf),
        "circuit_time_grid_points": int(time_h.size),
        "circuit_steady_horizon_h": float(steady_hours),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scan-config",
        type=Path,
        default=REPO_ROOT / "configs" / "scan_full_grid.yaml",
        help="V6/new-grid scan configuration used to define Top, Npw and rho_turn.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=cfg.OUTPUTS_TABLES_DIR / "circuit_scalar_cache.csv",
        help="CSV output. Full circuit time traces are never written.",
    )
    parser.add_argument("--top", type=float, action="append", default=None,
                        help="Optional temperature selector; repeat for multiple values.")
    parser.add_argument("--npw-min", type=int, default=None,
                        help="Optional inclusive Npw lower bound for an isolated benchmark block.")
    parser.add_argument("--npw-max", type=int, default=None,
                        help="Optional inclusive Npw upper bound for an isolated benchmark block.")
    parser.add_argument(
        "--device",
        required=True,
        help=(
            "Frozen FUSION_DEVICE identity. Set the identically named environment "
            "variable before starting Python, because fusion_tem.device is imported at "
            "module-load time."
        ),
    )
    args = parser.parse_args()

    requested_device = str(args.device).lower()
    if cfg.DEVICE != requested_device:
        raise RuntimeError(
            "device identity mismatch: set FUSION_DEVICE before invoking this script; "
            f"requested {requested_device!r}, loaded {cfg.DEVICE!r}"
        )

    temperatures, npw_values, rho_values = _scan_grid(args.scan_config)
    if args.top:
        requested = {round(float(value), 9) for value in args.top}
        available = {round(float(value), 9) for value in temperatures}
        unknown = requested.difference(available)
        if unknown:
            raise ValueError(f"requested --top values outside scan config: {sorted(unknown)}")
        temperatures = [value for value in temperatures if round(float(value), 9) in requested]
    if args.npw_min is not None:
        npw_values = [value for value in npw_values if value >= args.npw_min]
    if args.npw_max is not None:
        npw_values = [value for value in npw_values if value <= args.npw_max]
    if not temperatures or not npw_values or not rho_values:
        raise ValueError("filtered circuit-cache grid is empty")
    charge_module = _load_charge_module()
    matrix_path = REPO_ROOT / "scripts" / "8_economic" / cfg.INDUCTANCE_OUTPUT_DIR / cfg.TF_SYSTEM_MATRIX
    l_matrix = pd.read_excel(matrix_path, header=None).to_numpy(dtype=float)
    matrix_sha256 = _sha256(matrix_path)
    if l_matrix.shape != (cfg.Ntf, cfg.Ntf):
        raise ValueError(f"unexpected inductance-matrix shape {l_matrix.shape}; expected {(cfg.Ntf, cfg.Ntf)}")

    rows: list[dict] = []
    total = len(temperatures) * len(npw_values) * len(rho_values)
    for count, (top_k, npw, rho) in enumerate(
        ((t, n, r) for t in temperatures for n in npw_values for r in rho_values), start=1
    ):
        row = {
            "Top_K": top_k,
            "Npw": npw,
            "rho_turn_uOhm_cm2": rho,
            "device": cfg.DEVICE,
            "TF_system_matrix_file": matrix_path.name,
            "TF_system_matrix_sha256": matrix_sha256,
            "circuit_solver_error": "",
        }
        try:
            row.update(_solve_one(charge_module, l_matrix, top_k, npw, rho))
            row.update({
                "circuit_solver_status": "success",
                "Radial_loss_power_basis": "per_TF=sum_18_branch_P_R_over_Ntf",
                "circuit_solver_source": "2.5_charge_time999_TF_system_all_Npw=1-200.py",
            })
        except Exception as exc:
            row.update({
                "Charging_time_999_h": np.nan,
                "Radial_loss_at_charge_W": np.nan,
                "Radial_loss_peak_W": np.nan,
                "Radial_loss_energy_MWh": np.nan,
                "Radial_loss_average_W": np.nan,
                "R_radial_per_TF_Ohm": np.nan,
                "circuit_time_grid_points": np.nan,
                "circuit_steady_horizon_h": np.nan,
                "circuit_solver_status": "error",
                "circuit_solver_error": str(exc),
                "Radial_loss_power_basis": "per_TF=sum_18_branch_P_R_over_Ntf",
                "circuit_solver_source": "2.5_charge_time999_TF_system_all_Npw=1-200.py",
            })
        rows.append(row)
        if count % 25 == 0 or count == total:
            print(f"[circuit-cache] {count}/{total}")

    out = args.output.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    (out.with_suffix(out.suffix + ".manifest.json")).write_text(
        json.dumps({
            "device": cfg.DEVICE,
            "matrix_file": str(matrix_path),
            "matrix_sha256": matrix_sha256,
            "matrix_shape": list(l_matrix.shape),
            "rows": len(rows),
        }, indent=2),
        encoding="utf-8",
    )
    failures = sum(row["circuit_solver_status"] != "success" for row in rows)
    print(f"[circuit-cache] wrote {out} ({len(rows)} rows; failures={failures})")
    if failures:
        raise RuntimeError("circuit cache contains failed grid points")


if __name__ == "__main__":
    main()
