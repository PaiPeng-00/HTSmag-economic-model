#!/usr/bin/env python3
"""V6-F1: audited 20 K H2 physical grid, isolated from V5/B1 outputs."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "v6_hts_temperature_tolerance_arc16pancake_nuc600"
DRIVER = Path(__file__).with_name("8.13_run_v6_arc16pancake_nuc600_grid.py")
RAW = OUT / "v6f_f1_s2_20K_H2_physical_grid.csv"
AUDIT = OUT / "v6f_f1_h2_grid_audit.json"
UPSTREAM = (OUT / "v6b_performance_sensitivity_audit.json", OUT / "v6e_cross_scenario_audit.json")
RHO = np.array([10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000,5000,10000], dtype=float)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_driver():
    spec = importlib.util.spec_from_file_location("v6f_driver", DRIVER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    if RAW.exists() or AUDIT.exists():
        raise FileExistsError("refusing to overwrite V6-F1 H2 grid artifacts")
    for path in UPSTREAM:
        if not path.exists() or json.loads(path.read_text(encoding="utf-8")).get("status") != "PASS":
            raise AssertionError(f"V6-F is blocked: upstream audit not PASS: {path.name}")
    started = datetime.now(timezone.utc)
    driver = load_driver()
    scan = driver.load_scan(output_csv=RAW, npw=np.arange(1, 201, dtype=int), rjoint=np.geomspace(1.0, 100.0, 121), rho=RHO, temperatures=[(20.0, "H2")])
    scan.SCENARIOS = ["S2"]
    scan.main()
    expected_rows = 200 * 121 * len(RHO)
    rows = sum(1 for _ in RAW.open("rb")) - 1
    if rows != expected_rows:
        raise AssertionError(f"unexpected V6-F1 raw row count {rows} != {expected_rows}")
    audit = {"experiment_id": "V6-F1-S2-20K-H2-physical-grid", "status": "PASS", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
             "device": "arc_16pancake_nuc600", "scenario": "S2", "Top_K": 20.0, "coolant": "H2", "grid": {"Npw": "1..200", "R_joint_nOhm": "121-point geomspace [1,100]", "rho_turn_uOhm_cm2": RHO.tolist()},
             "upstream_audits": {path.name: sha256(path) for path in UPSTREAM}, "raw": {"path": str(RAW.relative_to(ROOT)), "rows": rows, "sha256": sha256(RAW)}, "started_utc": started.isoformat()}
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
