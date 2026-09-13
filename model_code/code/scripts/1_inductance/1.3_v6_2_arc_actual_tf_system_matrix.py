"""V6.2: direct ARC 18-coil single-turn TF-system mutual-inductance matrix.

This program deliberately does not use the SPARC matrix or a scalar geometry
factor.  It constructs the ARC D-shaped TF turn at the winding-pack centreline
from ``arc_16pancake_nuc600.yaml``, rotates it to the 18 actual TF azimuths,
and evaluates the Neumann line integral.  The self term is a finite-conductor
regularised integral whose equivalent radius follows the configured winding
pack area per turn, ``sqrt(WID * (R2/N_TOTAL_TAPE) / pi)``.

The output is an evidence artefact for V6.2, not a replacement for a legacy
matrix.  Downstream configuration is changed only after the comparison and
admission audit are reviewed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


MU0_OVER_4PI = 1.0e-7


def d_shape_centreline(l1: float, r1: float, r2: float, r10: float, n_segments: int) -> np.ndarray:
    """Return the actual ARC winding-pack-centre D-turn in global (R,Z).

    The layout is transcribed from ``ARC_field_mangiarotti_Nc16.m``: the
    inboard pack spans R10 to R10+R2, small arcs are centred at
    R10+R2+R1 and Z=+/-L1/2, and the outboard arc shares that R centre.
    The turn follows the pack centreline (R2/2) in global coordinates.
    """
    vertical_n = max(16, round(n_segments * 0.30))
    small_n = max(12, round(n_segments * 0.12))
    outer_n = max(32, n_segments - vertical_n - 2 * small_n)
    r_mid = r1 + r2 / 2.0
    r_center = r10 + r2 + r1
    r_inboard = r10 + r2 / 2.0
    r_outboard = l1 / 2.0 + r_mid
    z_half = l1 / 2.0
    z_vert = np.linspace(-z_half, z_half, vertical_n, endpoint=False)
    r_vert = np.full_like(z_vert, r_inboard)
    theta_top = np.linspace(np.pi, np.pi / 2.0, small_n, endpoint=False)
    r_top = r_center + r_mid * np.cos(theta_top)
    z_top = z_half + r_mid * np.sin(theta_top)
    theta_outer = np.linspace(np.pi / 2.0, -np.pi / 2.0, outer_n, endpoint=False)
    r_outer = r_center + r_outboard * np.cos(theta_outer)
    z_outer = r_outboard * np.sin(theta_outer)
    theta_bottom = np.linspace(-np.pi / 2.0, -np.pi, small_n, endpoint=False)
    r_bottom = r_center + r_mid * np.cos(theta_bottom)
    z_bottom = -z_half + r_mid * np.sin(theta_bottom)
    return np.column_stack((
        np.concatenate((r_vert, r_top, r_outer, r_bottom)),
        np.concatenate((z_vert, z_top, z_outer, z_bottom)),
        np.zeros(vertical_n + 2 * small_n + outer_n),
    ))

def rotate_about_vertical(loop_rz: np.ndarray, phi: float) -> np.ndarray:
    """Rotate an (R,Z,0) poloidal loop around the vertical Z axis."""
    r, z = loop_rz[:, 0], loop_rz[:, 1]
    return np.column_stack((r * np.cos(phi), r * np.sin(phi), z))


def segments(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nxt = np.roll(points, -1, axis=0)
    dl = nxt - points
    mid = 0.5 * (nxt + points)
    return dl, mid, np.linalg.norm(dl, axis=1)


def mutual_neumann(loop_a: np.ndarray, loop_b: np.ndarray, a_eff_m: float = 0.0) -> float:
    """Finite-segment Neumann mutual inductance for two distinct loops."""
    dl_a, mid_a, _ = segments(loop_a)
    dl_b, mid_b, _ = segments(loop_b)
    total = 0.0
    chunk = 64
    for start in range(0, len(mid_a), chunk):
        m_a = mid_a[start : start + chunk]
        d_a = dl_a[start : start + chunk]
        delta = m_a[:, None, :] - mid_b[None, :, :]
        distance = np.sqrt(np.einsum("ijk,ijk->ij", delta, delta) + a_eff_m**2)
        dot = d_a @ dl_b.T
        total += float(np.sum(dot / distance))
    return MU0_OVER_4PI * total


def self_neumann_regularised(loop: np.ndarray, a_eff_m: float) -> float:
    """Finite-conductor self inductance with exact same-segment replacement."""
    dl, mid, length = segments(loop)
    total = 0.0
    chunk = 64
    n = len(mid)
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        m_i = mid[start:stop]
        d_i = dl[start:stop]
        delta = m_i[:, None, :] - mid[None, :, :]
        distance = np.sqrt(np.einsum("ijk,ijk->ij", delta, delta) + a_eff_m**2)
        dot = d_i @ dl.T
        block = dot / distance
        for local_i, global_i in enumerate(range(start, stop)):
            ell = length[global_i]
            # Integral over the same straight segment, replacing midpoint l^2/a.
            exact = 2.0 * (ell * np.arcsinh(ell / a_eff_m) - np.sqrt(ell**2 + a_eff_m**2) + a_eff_m)
            block[local_i, global_i] = exact
        total += float(np.sum(block))
    return MU0_OVER_4PI * total


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments", type=int, default=960, help="segments for mutual terms")
    parser.add_argument("--self-segments", type=int, default=4800, help="segments for self term")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.segments < 180 or args.self_segments < args.segments:
        raise ValueError("use at least 180 mutual and at least as many self segments")

    repo = Path(__file__).resolve().parents[3]
    config_path = repo / "configs" / "devices" / "arc_16pancake_nuc600.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    out_dir = args.output_dir or repo / "v6_2_arc_actual_inductance" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_tf = int(cfg["Ntf"])
    l1 = float(cfg["L1"])
    r1 = float(cfg["R1"])
    r2 = float(cfg["R2"])
    wid = float(cfg["WID"])
    n_total_tape = int(cfg["N_TOTAL_TAPE"])
    r10 = 0.70  # frozen ARC inboard WP edge; ARC_field_mangiarotti_Nc16.m
    a_eff = float(np.sqrt(wid * (r2 / n_total_tape) / np.pi))

    base = d_shape_centreline(l1, r1, r2, r10, args.segments)
    loops = [rotate_about_vertical(base, 2.0 * np.pi * i / n_tf) for i in range(n_tf)]
    first_row = np.empty(n_tf)
    for separation in range(1, n_tf // 2 + 1):
        first_row[separation] = mutual_neumann(loops[0], loops[separation])
    first_row[0] = self_neumann_regularised(
        d_shape_centreline(l1, r1, r2, r10, args.self_segments), a_eff
    )
    for separation in range(n_tf // 2 + 1, n_tf):
        first_row[separation] = first_row[n_tf - separation]
    matrix = np.array([np.roll(first_row, i) for i in range(n_tf)])
    matrix = 0.5 * (matrix + matrix.T)

    csv_path = out_dir / "TF_system_L_matrix_ARC_actual_geometry_v6_2.csv"
    xlsx_path = out_dir / "TF_system_L_matrix_ARC_actual_geometry_v6_2.xlsx"
    pd.DataFrame(matrix).to_csv(csv_path, header=False, index=False)
    pd.DataFrame(matrix).to_excel(xlsx_path, header=False, index=False)
    meta = {
        "experiment_id": "V6.2",
        "status": "DIRECT_ARC_GEOMETRY_MATRIX",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "geometry": {
            "Ntf": n_tf,
            "L1_m": l1,
            "R1_m": r1,
            "R2_m": r2,
            "R10_m": r10,
            "centreline_inboard_R_m": r10 + r2 / 2.0,
            "arc_centre_R_m": r10 + r2 + r1,
            "WID_m": wid,
            "N_TOTAL_TAPE": n_total_tape,
            "a_eff_m": a_eff,
            "a_eff_definition": "sqrt(WID*(R2/N_TOTAL_TAPE)/pi)",
            "path": "ARC D-shaped centreline rotated to 18 actual toroidal positions",
        },
        "numerics": {
            "formula": "M_ij=mu0/(4pi) integral integral dl_i dot dl_j / |r_i-r_j|",
            "mutual_segments": args.segments,
            "self_segments": args.self_segments,
            "self_regularisation": "finite a_eff with exact same-straight-segment integral",
        },
        "matrix_checks": {
            "shape": list(matrix.shape),
            "symmetry_max_abs_H": float(np.max(np.abs(matrix - matrix.T))),
            "eigenvalue_min_H": float(np.linalg.eigvalsh(matrix)[0]),
            "eigenvalue_max_H": float(np.linalg.eigvalsh(matrix)[-1]),
            "self_inductance_H": float(matrix[0, 0]),
        },
        "artifacts": {"csv": csv_path.name, "csv_sha256": sha256(csv_path), "xlsx": xlsx_path.name},
    }
    (out_dir / "TF_system_L_matrix_ARC_actual_geometry_v6_2.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(json.dumps(meta["matrix_checks"], indent=2))
    print(csv_path)


if __name__ == "__main__":
    main()
