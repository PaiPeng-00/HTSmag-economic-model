#!/usr/bin/env python3
"""V6-B: audit status and construct the only admissible feasibility denominator."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "v6_hts_temperature_tolerance_arc16pancake_nuc600"
INPUT = OUT / "v6_b1_anchor_economics_arc16pancake_nuc600.csv"
INPUT_AUDIT = OUT / "v6_b1_anchor_economics_audit.json"
FEASIBLE = OUT / "v6b_feasible_designs.csv"
GROUPS = OUT / "v6b_status_feasibility_by_group.csv"
REASONS = OUT / "v6b_status_invalid_reason_counts.csv"
AUDIT = OUT / "v6b_status_feasibility_audit.json"
EXPECTED_ROWS = 4_573_800
CHUNK_SIZE = 50_000
GROUP_KEYS = ["scenario", "Top_K", "coolant", "status"]
KEYS = ["scenario", "Top_K", "coolant", "Npw", "rho_turn_uOhm_cm2", "R_joint_nOhm"]
METRICS = ["status", "invalid_reason", "AF_ref", "r_cryo_re_fraction", "net_energy_annual_MWh", "lcoe_anchor_USD_per_MWh", "anchor_economic_valid"]
FEASIBLE_COLUMNS = KEYS + ["status", "AF_ref", "r_cryo_re_fraction", "net_energy_annual_MWh", "lcoe_anchor_USD_per_MWh"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(("true", "1", "yes"))


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if not INPUT.exists() or not INPUT_AUDIT.exists():
        raise FileNotFoundError("V6 B1-anchor input or its PASS audit is missing")
    targets = (FEASIBLE, FEASIBLE.with_suffix(".csv.partial"), GROUPS, REASONS, AUDIT)
    if any(path.exists() for path in targets):
        raise FileExistsError("refusing to overwrite a V6-B feasibility output")

    upstream = json.loads(INPUT_AUDIT.read_text(encoding="utf-8"))
    input_sha = sha256(INPUT)
    if upstream.get("status") != "PASS" or upstream.get("output", {}).get("sha256") != input_sha:
        raise AssertionError("V6 B1-anchor input is not the audited PASS artifact")

    header = pd.read_csv(INPUT, nrows=0).columns.tolist()
    required = set(GROUP_KEYS + METRICS + KEYS)
    missing = sorted(required.difference(header))
    if missing:
        raise AssertionError(f"missing V6-B fields: {missing}")

    started = datetime.now(timezone.utc)
    totals = Counter()
    statuses = Counter()
    reasons = Counter()
    groups: dict[tuple, Counter] = defaultdict(Counter)
    rows = 0
    feasible_rows = 0
    temp = FEASIBLE.with_suffix(".csv.partial")

    usecols = list(dict.fromkeys(GROUP_KEYS + METRICS + KEYS))
    for number, chunk in enumerate(pd.read_csv(INPUT, usecols=usecols, chunksize=CHUNK_SIZE, dtype=str, keep_default_na=False)):
        af = pd.to_numeric(chunk["AF_ref"], errors="coerce")
        rcryo = pd.to_numeric(chunk["r_cryo_re_fraction"], errors="coerce")
        net = pd.to_numeric(chunk["net_energy_annual_MWh"], errors="coerce")
        lcoe = pd.to_numeric(chunk["lcoe_anchor_USD_per_MWh"], errors="coerce")
        anchor_valid = bool_series(chunk["anchor_economic_valid"])
        status = chunk["status"].astype(str).str.strip().str.lower()
        numeric = np.isfinite(af) & np.isfinite(rcryo) & np.isfinite(net) & np.isfinite(lcoe) & anchor_valid
        af_gate = af >= 0.99
        rcryo_gate = rcryo <= 0.50
        net_gate = net > 0.0
        lcoe_gate = lcoe > 0.0
        thresholds = numeric & af_gate & rcryo_gate & net_gate & lcoe_gate
        success = status.eq("success")
        eligible = success & thresholds

        rows += len(chunk)
        feasible_rows += int(eligible.sum())
        totals.update({
            "numeric_valid": int(numeric.sum()), "af_gate_pass": int(af_gate.sum()),
            "rcryo_gate_pass": int(rcryo_gate.sum()), "net_gate_pass": int(net_gate.sum()),
            "lcoe_gate_pass": int(lcoe_gate.sum()), "thresholds_pass": int(thresholds.sum()),
            "status_success": int(success.sum()), "eligible": int(eligible.sum()),
            "warning_thresholds_pass": int((status.eq("warning") & thresholds).sum()),
            "error_thresholds_pass": int((status.eq("error") & thresholds).sum()),
            "warning_eligible": int((status.eq("warning") & eligible).sum()),
            "error_eligible": int((status.eq("error") & eligible).sum()),
        })
        statuses.update(status.value_counts(dropna=False).to_dict())
        reasons.update((status + "|" + chunk["invalid_reason"].replace("", "<blank>")).value_counts(dropna=False).to_dict())

        working = chunk.assign(
            numeric_valid=numeric, af_gate_pass=af_gate, rcryo_gate_pass=rcryo_gate,
            net_gate_pass=net_gate, lcoe_gate_pass=lcoe_gate, thresholds_pass=thresholds,
            status_success=success, eligible=eligible,
        )
        for key, subset in working.groupby(GROUP_KEYS, sort=False, dropna=False):
            entry = groups[key]
            entry["row_count"] += len(subset)
            for field in ("numeric_valid", "af_gate_pass", "rcryo_gate_pass", "net_gate_pass", "lcoe_gate_pass", "thresholds_pass", "status_success", "eligible"):
                entry[field] += int(subset[field].sum())

        selected = chunk.loc[eligible, FEASIBLE_COLUMNS]
        if len(selected):
            selected.to_csv(temp, mode="w" if feasible_rows == len(selected) else "a", header=feasible_rows == len(selected), index=False, encoding="utf-8", lineterminator="\n")
        if rows % 250_000 == 0:
            print(f"[Progress] audited {rows:,}/{EXPECTED_ROWS:,}; eligible {feasible_rows:,}", flush=True)

    if rows != EXPECTED_ROWS:
        raise AssertionError(f"V6-B input row count {rows} != {EXPECTED_ROWS}")
    if feasible_rows == 0:
        raise AssertionError("V6-B found no feasible designs")
    if totals["warning_eligible"] or totals["error_eligible"]:
        raise AssertionError("warning/error rows entered the V6-B denominator")
    os.replace(temp, FEASIBLE)

    group_rows = []
    for key in sorted(groups):
        row = dict(zip(GROUP_KEYS, key))
        row.update(groups[key])
        group_rows.append(row)
    write_rows(GROUPS, group_rows)
    reason_rows = []
    for key, count in sorted(reasons.items(), key=lambda item: (-item[1], item[0])):
        status_value, reason = key.split("|", 1)
        reason_rows.append({"status": status_value, "invalid_reason": reason, "row_count": count})
    write_rows(REASONS, reason_rows)

    passed = (
        rows == EXPECTED_ROWS and totals["eligible"] == feasible_rows
        and totals["warning_eligible"] == 0 and totals["error_eligible"] == 0
    )
    audit = {
        "experiment_id": "V6-B-status-feasibility-audit",
        "status": "PASS" if passed else "FAIL",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input": {"path": str(INPUT.relative_to(ROOT)), "rows": rows, "sha256": input_sha},
        "upstream_anchor_audit": {"path": str(INPUT_AUDIT.relative_to(ROOT)), "status": upstream["status"]},
        "feasibility_definition": {
            "status": "success only; warning and error are explicitly excluded",
            "numeric_valid": "finite AF_ref, r_cryo_re_fraction, net_energy_annual_MWh, lcoe_anchor_USD_per_MWh, and anchor_economic_valid=true",
            "thresholds": "AF_ref >= 0.99; r_cryo_re_fraction <= 0.50; net_energy_annual_MWh > 0; lcoe_anchor_USD_per_MWh > 0",
        },
        "status_counts": dict(statuses), "gate_counts": dict(totals),
        "outputs": {
            "feasible_designs": {"path": str(FEASIBLE.relative_to(ROOT)), "rows": feasible_rows, "sha256": sha256(FEASIBLE)},
            "by_group": {"path": str(GROUPS.relative_to(ROOT)), "rows": len(group_rows), "sha256": sha256(GROUPS)},
            "invalid_reason_counts": {"path": str(REASONS.relative_to(ROOT)), "rows": len(reason_rows), "sha256": sha256(REASONS)},
        },
        "started_utc": started.isoformat(),
    }
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
    if not passed:
        raise SystemExit("V6-B status/feasibility audit failed")


if __name__ == "__main__":
    main()
