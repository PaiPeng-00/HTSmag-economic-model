#!/usr/bin/env python
"""Run the reproducible device-driven analysis pipeline.

Examples
--------
python scripts/run_scan.py
python scripts/run_scan.py --steps scan,figures
python scripts/run_scan.py --device sparc --steps scan
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CODE = REPO / "code"

STEP_SCRIPTS = {
    "charge": ["scripts/2_charging/2.5_charge_time999_TF_system_all_Npw=1-200.py"],
    "scan": ["scripts/8_economic/scan_full_grid.py"],
    "figures": [
        "scripts/8_economic/8.1_plot_from_scan_full_grid.py",
        "scripts/9_figures/9.0_stitch_econimic_svgs.py",
    ],
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        default="arc_16pancake_nuc600",
        help="Device YAML stem; defaults to the manuscript baseline.",
    )
    parser.add_argument(
        "--steps",
        default="charge,scan,figures",
        help="Comma-separated pipeline steps: charge,scan,figures.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Generate the reduced diagnostic figure set.",
    )
    args = parser.parse_args()

    device_yaml = REPO / "configs" / "devices" / f"{args.device}.yaml"
    if not device_yaml.is_file():
        available = sorted(p.stem for p in (REPO / "configs" / "devices").glob("*.yaml"))
        raise SystemExit(f"[run_scan] Device configuration not found: {device_yaml}\nAvailable: {available}")

    env = dict(
        os.environ,
        FUSION_DEVICE=args.device,
        PYTHONIOENCODING="utf-8",
    )
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(CODE / "src"), env.get("PYTHONPATH")])
    )
    if args.fast:
        env["FAST_FIG"] = "1"

    steps = [step.strip() for step in args.steps.split(",") if step.strip()]
    unknown = [step for step in steps if step not in STEP_SCRIPTS]
    if unknown:
        raise SystemExit(f"[run_scan] Unknown steps {unknown}; choose from {list(STEP_SCRIPTS)}")

    print(f"[run_scan] device={args.device} steps={','.join(steps)}")
    for step in steps:
        for script in STEP_SCRIPTS[step]:
            script_path = CODE / script
            if not script_path.is_file():
                raise SystemExit(f"[run_scan] Pipeline script not found: {script_path}")
            print(f"\n[run_scan] === {step}: {script} ===")
            result = subprocess.run([sys.executable, str(script_path)], cwd=CODE, env=env)
            if result.returncode != 0:
                raise SystemExit(
                    f"[run_scan] Step {step!r} failed in {script} "
                    f"with exit code {result.returncode}"
                )

    print("\n[run_scan] Complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
