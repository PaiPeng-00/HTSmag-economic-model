#!/usr/bin/env python
"""Build the circuit cache and complete manuscript-domain numerical grid."""
import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CODE = ROOT / "code"
DEVICE = "arc_16pancake_nuc600_v6_2"


def run(script: Path, arguments: list[str], env: dict[str, str]) -> None:
    print(f"[full-grid] {script.relative_to(CODE)} {' '.join(arguments)}")
    completed = subprocess.run(
        [sys.executable, str(script), *arguments],
        cwd=CODE,
        env=env,
    )
    if completed.returncode:
        raise SystemExit(f"[full-grid] exit {completed.returncode}: {script}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate the domain and required inputs without solving.",
    )
    args = parser.parse_args()

    required = [
        ROOT / "configs/devices/arc_16pancake_nuc600_v6_2.yaml",
        CODE / "configs/scan_full_grid.yaml",
        CODE / "configs/2025_USD_conversion_table.csv",
        CODE / "data/raw/heat_input/data_s2_radial_loss.xlsx",
        CODE / "data/raw/heat_input/data_s3_magnetization_loss.xlsx",
        CODE / "data/raw/inductance/TF_system_L_matrix_ARC_actual_geometry_v6_2.xlsx",
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise SystemExit("[full-grid] missing inputs:\n" + "\n".join(map(str, missing)))

    import yaml

    config = yaml.safe_load((CODE / "configs/scan_full_grid.yaml").read_text(encoding="utf-8"))
    npw = config["npw_values"]["range"]
    rho = config["rho_turn_uohm_cm2_values"]["logspace"]
    rj = config["r_joint_nohm_values"]["logspace"]
    actual = {
        "npw": (npw["start"], npw["stop"], npw.get("step", 1)),
        "rho": (rho["start"], rho["stop"], rho["num"]),
        "rj": (rj["start"], rj["stop"], rj["num"]),
        "scenarios": config["scenarios"],
    }
    expected = {
        "npw": (1, 200, 1),
        "rho": (10.0, 10000.0, 61),
        "rj": (1.0, 100.0, 121),
        "scenarios": ["S1", "S2", "S3"],
    }
    if actual != expected:
        raise SystemExit(f"[full-grid] domain mismatch: {actual!r} != {expected!r}")
    os.environ["FUSION_DEVICE"] = DEVICE
    sys.path.insert(0, str(CODE / "src"))
    scan_path = CODE / "scripts/8_economic/scan_full_grid.py"
    scan_spec = importlib.util.spec_from_file_location("full_grid_preflight", scan_path)
    if scan_spec is None or scan_spec.loader is None:
        raise SystemExit(f"[full-grid] cannot import {scan_path}")
    scan_module = importlib.util.module_from_spec(scan_spec)
    scan_spec.loader.exec_module(scan_module)
    runtime_counts = (
        len(scan_module.NPW_RANGE),
        len(scan_module.RHO_TURN_UOHM_CM2),
        len(scan_module.R_JOINT_NOHM),
        len(scan_module.SCENARIOS),
    )
    if runtime_counts != (200, 61, 121, 3):
        raise SystemExit(f"[full-grid] runtime-domain mismatch: {runtime_counts}")
    print("[full-grid] preflight PASS: imports and Npw=200, rho=61, Rj=121, S1-S3")
    if args.preflight_only:
        return 0

    env = dict(
        os.environ,
        FUSION_DEVICE=DEVICE,
        PYTHONIOENCODING="utf-8",
        # The complete CSV is reduced in chunks by the governed V10 ledger
        # builder.  Loading all 17.7 million rows into one DataFrame here is
        # unnecessary and can exceed workstation memory.
        SCAN_DEFER_FINALIZE="1",
    )
    env["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (str(CODE / "src"), env.get("PYTHONPATH", ""))
        if value
    )
    cache = CODE / "outputs/tables/circuit_scalar_cache.csv"
    run(
        CODE / "scripts/2_charging/2.7_build_circuit_scalar_cache.py",
        ["--device", DEVICE, "--output", str(cache)],
        env,
    )
    run(CODE / "scripts/8_economic/scan_full_grid.py", [], env)
    print("[full-grid] complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
