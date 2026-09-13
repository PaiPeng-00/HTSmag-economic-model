"""Compare new Fig. 5 A1 reruns against the existing Nc16 sweep outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve()
WORKSPACE = SCRIPT.parents[4]
OLD_DIR = WORKSPACE / "ARC-model" / "results" / "Nc16"
NEW_DIR = WORKSPACE / "ARC-model" / "results" / "Nc16_fig5"
OUT = NEW_DIR / "A1_repeat_qc.csv"

T_CASES = [4.2, 10.0, 20.0]


def t_token(T: float) -> str:
    return f"{T:g}"


def read_csv(path: Path) -> np.ndarray:
    data = np.loadtxt(path, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 3:
        raise ValueError(f"{path} has {data.shape[1]} columns; expected at least 3")
    return data[:, :3]


def compare_series(old: np.ndarray, new: np.ndarray, col: int) -> tuple[float, float]:
    old_t = old[:, 0]
    old_y = old[:, col]
    new_t = new[:, 0]
    new_y = new[:, col]
    interp_old = np.interp(new_t, old_t, old_y)
    abs_err = float(np.max(np.abs(new_y - interp_old)))
    denom = max(float(np.max(np.abs(interp_old))), 1e-12)
    rel_err = abs_err / denom
    return abs_err, rel_err


def main() -> None:
    rows = []
    missing = []
    for T in T_CASES:
        old_path = OLD_DIR / f"ARC16_sweep_T{t_token(T)}_Npw5.csv"
        new_path = NEW_DIR / f"ARC16_fig5_A1_T{t_token(T)}_Npw5_rho5000.csv"
        if not old_path.exists() or not new_path.exists():
            missing.append((T, old_path.exists(), new_path.exists()))
            continue
        old = read_csv(old_path)
        new = read_csv(new_path)
        mag_abs, mag_rel = compare_series(old, new, 1)
        radial_abs, radial_rel = compare_series(old, new, 2)
        rows.append(
            {
                "Top_K": T,
                "old_csv": str(old_path),
                "new_csv": str(new_path),
                "old_rows": len(old),
                "new_rows": len(new),
                "old_mag_peak_W": float(np.max(old[:, 1])),
                "new_mag_peak_W": float(np.max(new[:, 1])),
                "old_radial_peak_W": float(np.max(old[:, 2])),
                "new_radial_peak_W": float(np.max(new[:, 2])),
                "mag_max_abs_diff_W": mag_abs,
                "mag_max_rel_diff": mag_rel,
                "radial_max_abs_diff_W": radial_abs,
                "radial_max_rel_diff": radial_rel,
                "flag_gt_1pct": bool(max(mag_rel, radial_rel) > 0.01),
            }
        )

    if missing:
        print("[warning] missing A1 comparison inputs:")
        for T, has_old, has_new in missing:
            print(f"  Top={T:g} K old_exists={has_old} new_exists={has_new}")

    if rows:
        NEW_DIR.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows)
        df.to_csv(OUT, index=False)
        print(df.to_string(index=False))
        print(f"wrote: {OUT}")
        if df["flag_gt_1pct"].any():
            raise SystemExit("A1 repeat differs by more than 1%; audit required.")
    else:
        raise SystemExit("No A1 comparison rows were available.")


if __name__ == "__main__":
    main()
