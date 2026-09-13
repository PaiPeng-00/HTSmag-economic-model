#!/usr/bin/env python
"""Compare configured devices at a fixed steady-state design point."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
CODE = ROOT / "code"
DEVICES_DIR = ROOT / "configs" / "devices"
OUT = ROOT / "results" / "cases"
BASELINE_DEVICE = "arc_16pancake_nuc600"

FIELDS = [
    "device",
    "T_K",
    "NP",
    "WID_mm",
    "nuclear_density",
    "Ip_A",
    "Nt",
    "V_magnet_m3",
    "nuclear_W",
    "joint_W",
    "r_cryo_re_pct",
    "gross_MWe",
    "net_GWh_yr",
    "lcoe_C0",
    "lcoe_plant",
]

WORKER = r"""
import json
import sys
from fusion_tem.economic import lcoe as L
from fusion_tem import device as cfg

npw = int(sys.argv[1])
r_joint = float(sys.argv[2])
scenario = sys.argv[3]
parameters = L.define_parameters()
project_years = parameters["tech_scenario_to_years"][scenario]
rows = []
for temperature in (4.2, 10.0, 20.0):
    result, heat = L.compute_case(
        scenario, project_years, temperature, "He", npw, r_joint, parameters
    )
    rows.append({
        "device": cfg.DEVICE,
        "T_K": temperature,
        "NP": cfg.NP,
        "WID_mm": cfg.WID * 1000,
        "nuclear_density": cfg.NUCLEAR_POWER_DENSITY,
        "Ip_A": result["Ip"],
        "Nt": cfg.Nt_list[temperature],
        "V_magnet_m3": round(cfg.V_magnet, 3),
        "nuclear_W": round(heat["nuclear"], 1),
        "joint_W": round(heat["pancake_joint"], 3),
        "r_cryo_re_pct": round(result["r_parasitic_pct"], 4),
        "gross_MWe": round(result["gross_power_output_MWe"], 2),
        "net_GWh_yr": round(result["net_energy_annual_MWh"] / 1000, 2),
        "lcoe_C0": round(result["lcoe_C0_$/MWh"], 3),
        "lcoe_plant": round(result["lcoe_plant_$/MWh"], 3),
    })
print("RESULT_LINE|" + json.dumps(rows))
"""


def run_device(device: str, npw: int, r_joint: float, scenario: str):
    env = dict(os.environ, FUSION_DEVICE=device, PYTHONIOENCODING="utf-8")
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(CODE / "src"), env.get("PYTHONPATH")])
    )
    completed = subprocess.run(
        [sys.executable, "-c", WORKER, str(npw), str(r_joint), scenario],
        capture_output=True,
        text=True,
        cwd=CODE,
        env=env,
    )
    for line in completed.stdout.splitlines():
        if line.startswith("RESULT_LINE|"):
            return json.loads(line.split("|", 1)[1])
    sys.stderr.write(f"[{device}] failed\n{completed.stderr[-1000:]}\n")
    return None


def write_csv(path: Path, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npw", type=int, default=20)
    parser.add_argument("--rjoint", type=float, default=1e-8)
    parser.add_argument("--scenario", default="S1")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    devices = sorted(path.stem for path in DEVICES_DIR.glob("*.yaml"))
    if BASELINE_DEVICE in devices:
        devices = [BASELINE_DEVICE] + [d for d in devices if d != BASELINE_DEVICE]

    all_rows = []
    successful = []
    for device in devices:
        rows = run_device(device, args.npw, args.rjoint, args.scenario)
        if rows is None:
            continue
        write_csv(OUT / f"{device}.csv", rows)
        all_rows.extend(rows)
        successful.append(device)
        print(f"[run_case_comparison] wrote {OUT / (device + '.csv')}")

    write_csv(OUT / "all_cases.csv", all_rows)
    summary = {
        "baseline_device": BASELINE_DEVICE,
        "scenario": args.scenario,
        "Npw": args.npw,
        "R_joint_Ohm": args.rjoint,
        "successful_devices": successful,
        "failed_devices": [d for d in devices if d not in successful],
        "reported_cost_metric": "lcoe_plant",
    }
    (OUT / "manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[run_case_comparison] wrote {OUT / 'all_cases.csv'}")
    return 0 if successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
