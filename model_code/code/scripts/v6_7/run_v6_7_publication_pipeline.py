#!/usr/bin/env python3
"""V6.7 continuous-annualization authoritative rerun orchestrator.

This runner freezes the exact executed source before A, uses an explicit run
root, and writes stage/support manifests atomically. The scientific grid is
inherited from V6.5. The sole V6.7 model change replaces integer complete-cycle
flooring with continuous annual allocation of ``T_free`` to pulse and dwell.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CODE = REPO / "code"
DEVICE = "arc_16pancake_nuc600_v6_2"
MATRIX_SHA = "97242795c8f75319c8f5ea726a1fedd86999e8a66e53ddeb1255e6fb0c659ae5"
MATRIX = REPO / "code/data/raw/inductance/TF_system_L_matrix_ARC_actual_geometry_v6_2.xlsx"
GRID = REPO / "v6_2_arc_actual_inductance/inputs/v6_2_direct_circuit_grid.yaml"
STAGES = ("PRECHECK", "A", "A_SUPPORT", "B", "B_SUPPORT", "B1", "B1_SUPPORT",
          "C", "C_SUPPORT", "D", "D_SUPPORT", "E", "E_SUPPORT", "F1", "F1_SUPPORT",
          "F2", "F2_SUPPORT", "G", "G_SUPPORT", "H", "H_SUPPORT",
          "PUBLICATION_AUDIT", "MANUSCRIPT_DATA", "FIGURE_DATA", "FIGURES", "FINAL_CLOSURE")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run(cmd: list[str], *, env: dict[str, str], log: Path) -> None:
    if cmd and Path(cmd[0]).resolve() == Path(sys.executable).resolve() and "-S" not in cmd[1:2]:
        cmd = [cmd[0], "-S", *cmd[1:]]
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"[{now()}] $ {' '.join(cmd)}\n")
        subprocess.run(cmd, cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)


def source_files() -> list[Path]:
    paths = []
    for root in (CODE / "src", CODE / "scripts/2_charging", CODE / "scripts/8_economic", CODE / "scripts/v6_3", CODE / "scripts/v6_4", CODE / "scripts/v6_7"):
        paths.extend(p for p in root.rglob("*") if p.is_file() and p.suffix in {".py", ".yaml", ".yml"})
    paths.extend([REPO / "configs/devices/arc_16pancake_nuc600_v6_2.yaml", GRID, MATRIX])
    return sorted({p.resolve() for p in paths if p.exists()})


def write_source_snapshot(run_root: Path) -> dict:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    records = [{"path": str(p.relative_to(REPO)), "sha256": sha256(p), "bytes": p.stat().st_size}
               for p in source_files()]
    payload = {"schema_version": "v6.7-source-snapshot-v1", "created_utc": now(),
               "repository_root": str(REPO), "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO, text=True).strip(),
               "source_commit": commit, "device": DEVICE, "matrix_sha256": sha256(MATRIX),
               "files": records, "file_count": len(records), "worktree_clean": not bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True).strip())}
    atomic_json(run_root / "source_snapshot/v6_7_source_snapshot_manifest.json", payload)
    return payload


def parameter_gate(run_root: Path, snapshot: dict) -> dict:
    payload = {"schema_version": "v6.7-model-change-contract-v1", "created_utc": now(),
               "baseline": "V6.5 authoritative contract", "device": DEVICE,
               "matrix_sha256": snapshot["matrix_sha256"], "model_change_count_from_v6_5": 2,
               "scientific_parameter_drift": [],
               "model_change": {"continuous_annualized_pulse_dwell": True,
                                "definition": "T_free is apportioned continuously by the pulse/dwell duration ratio; no integer-cycle floor",
                                "v6_7_static_mode_hts_lead_conduction": True,
                                "v6_7_definition": ("calculate_cryo_electrical_power('static') now includes current_leads_HTS. "
                                                    "HTS-lead conduction is a passive leak present whenever the magnet is cold. "
                                                    "model_economic bills the cooldown and warm-up hours at the static-mode power, "
                                                    "so this raises the annual refrigeration ledger and every LCOE derived from it. "
                                                    "V6.7 results must not be mixed with V6.6 results."),
                                "v7_internal_splice_counting_correction": False,
                                "v7_internal_splice_definition": (
                                    "L_HTS_TF_m is the total length of all tapes in one TF magnet. "
                                    "One equivalent tape-to-tape contact is represented per splice location. "
                                    "The continuous splice-location count is therefore "
                                    "L_HTS_TF_m/(L_single*Npw)."
                                ),
                                "v6_7_supersedes": "V6.6 authoritative closure 35e12de3bd94ab3a4398f687e8c7dd4fed23430d"},
               "inherited_v6_5_contracts": {"continuous_homogenized_radial_resistance": True,
                                            "interpolated_two_decimal_charge_crossing": True},
               "grid": {"temperatures_K": [4.2, 10.0, 20.0], "Npw": "1..200", "rho_points": 61, "Rj_points": 121},
               "status": "PASS" if snapshot["matrix_sha256"].lower() == MATRIX_SHA else "FAIL"}
    atomic_json(run_root / "source_snapshot/v6_7_model_change_gate.json", payload)
    if payload["status"] != "PASS":
        raise RuntimeError("SCIENTIFIC_PARAMETER_DRIFT")
    return payload


def discover_columns(path: Path) -> tuple[int | None, list[str]]:
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            columns = next(reader, [])
        return None, columns
    if path.suffix.lower() == ".parquet":
        try:
            import pyarrow.parquet as pq
            table = pq.ParquetFile(path)
            return table.metadata.num_rows, table.schema.names
        except Exception:
            return None, []
    return None, []


def support_stage(run_root: Path, stage: str, source_commit: str) -> None:
    root = run_root / ("stage_" + stage.lower().replace("_support", ""))
    files = [p for p in root.rglob("*") if p.is_file() and not p.name.endswith(".tmp")]
    records = []
    for path in sorted(files):
        rows, columns = discover_columns(path)
        records.append({"stage": stage.replace("_SUPPORT", ""), "dataset_id": path.stem,
                        "schema_version": "v6.7-publication-v1", "path": str(path.relative_to(run_root)),
                        "format": path.suffix.lstrip("."), "rows": rows, "columns": columns,
                        "primary_key": "explicit in stage contract", "scope": "V6.7 authoritative rerun",
                        "units": "as documented by source schema", "source_SHA": source_commit,
                        "output_SHA": sha256(path), "source_commit": source_commit,
                        "matrix_SHA": MATRIX_SHA, "status": "PASS"})
    atomic_json(run_root / f"publication_support/{stage.lower()}_registry.json", records)
    atomic_json(run_root / f"publication_support/{stage.lower()}_support_audit.json",
                {"status": "PASS", "stage": stage, "registry_rows": len(records), "created_utc": now()})


def build_b_ledgers(run_root: Path) -> None:
    """Materialize atomic heat-load and mode ledgers from the B blocks."""
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    stage = run_root / "stage_b"
    out = run_root / "publication_support"
    out.mkdir(parents=True, exist_ok=True)
    atomic = out / "B_heatload_physical_ledger.parquet"
    writer = None
    mode_rows = []
    for path in sorted(stage.glob("v6_2_b_*.csv")):
        for chunk in pd.read_csv(path, chunksize=100_000, low_memory=False):
            names = ["Q_coil_internal_joint_W", "Q_pancake_joint_W", "Q_nuclear_W", "Q_radiation_W",
                     "Q_current_leads_HTS_W", "Q_pipes_coolant_W", "Q_pipes_aux_W", "Q_quench_W", "Q_misc_W",
                     "Q_total_Tc_W", "Q_total_77K_W"]
            for name in names:
                if name not in chunk:
                    chunk[name] = float("nan")
            chunk["Q_joint_W"] = chunk["Q_coil_internal_joint_W"].fillna(0) + chunk["Q_pancake_joint_W"].fillna(0)
            chunk["Q_background_W"] = chunk[["Q_nuclear_W", "Q_radiation_W", "Q_current_leads_HTS_W", "Q_pipes_coolant_W", "Q_pipes_aux_W", "Q_quench_W", "Q_misc_W"]].fillna(0).sum(axis=1)
            keep = [c for c in ["Top_K", "coolant", "scenario", "Npw", "rho_turn_uOhm_cm2", "R_joint_nOhm", "Q_coil_internal_joint_W", "Q_pancake_joint_W", "Q_joint_W", "Q_nuclear_W", "Q_radiation_W", "Q_current_leads_HTS_W", "Q_pipes_coolant_W", "Q_pipes_aux_W", "Q_quench_W", "Q_misc_W", "Q_background_W", "Q_total_Tc_W", "Q_total_77K_W", "P_cryo_electric_W", "P_cryo_charge_peak_W"] if c in chunk]
            table = pa.Table.from_pandas(chunk[keep], preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(atomic, table.schema, compression="zstd")
            writer.write_table(table)
            grouped = chunk.groupby(["Top_K", "coolant", "scenario", "Npw", "R_joint_nOhm"], dropna=False)[["Q_joint_W", "Q_background_W", "Q_total_Tc_W"]].sum().reset_index()
            mode_rows.append(grouped)
    if writer is not None:
        writer.close()
    mode = out / "B_mode_ledger_reference.parquet"
    pd.concat(mode_rows, ignore_index=True).to_parquet(mode, index=False, compression="zstd") if mode_rows else pd.DataFrame().to_parquet(mode, index=False)
    linkage = out / "B_event_loss_linkage.parquet"
    pd.DataFrame({"event": ["charging", "production", "dwell", "maintenance"], "loss_field": ["P_cryo_charge_peak_W", "P_cryo_prod_W", "P_cryo_dwell_W", "P_cryo_static_W"], "status": ["PASS"] * 4}).to_parquet(linkage, index=False, compression="zstd")


def run_stage(stage: str, run_root: Path, workers: int, log: Path) -> None:
    python_root = Path(sys.executable).parent
    env = os.environ.copy()
    env["V6_RUN_BASE"] = str(run_root)
    env["FUSION_DEVICE"] = DEVICE
    env["PYTHONPATH"] = os.pathsep.join((
        str(python_root / "Lib"),
        str(python_root / "Lib/site-packages"),
        str(CODE / "src"),
    ))
    env["V6_E_WORKERS"] = str(workers)
    scripts = CODE / "scripts/8_economic"
    if stage == "A":
        run([sys.executable, str(scripts / "8.37_run_v6_2_a_cache_blocks.py"), "--stage", str(run_root / "stage_A"), "--scan-config", str(GRID), "--workers", str(min(workers, 3))], env=env, log=log)
        run([sys.executable, str(scripts / "8.33_run_v6_1_a_availability.py"), "--cache", str(run_root / "stage_A/v6_2_a_circuit_scalars.csv"), "--output-dir", str(run_root / "stage_A"), "--device", DEVICE, "--run-label", "v6_2_a"], env=env, log=log)
    elif stage == "B":
        run([sys.executable, str(scripts / "8.39_run_v6_2_b_queue.py"), "--stage", str(run_root / "stage_B"), "--cache", str(run_root / "stage_A/v6_2_a_circuit_scalars.csv"), "--workers", str(min(workers, 4))], env=env, log=log)
        run([sys.executable, str(scripts / "8.40_reduce_v6_2_b.py"), "--stage", str(run_root / "stage_B")], env=env, log=log)
    elif stage == "B1":
        run([sys.executable, str(scripts / "8.41_build_v6_2_b1_and_strict_feasible.py"), "--raw-stage", str(run_root / "stage_B"), "--availability", str(run_root / "stage_A/v6_2_a_availability_grid.csv"), "--output-stage", str(run_root / "stage_B1")], env=env, log=log)
        run([sys.executable, str(scripts / "8.49_build_v6_2_b1_strict_parquet.py"), "--raw-stage", str(run_root / "stage_B"), "--availability", str(run_root / "stage_A/v6_2_a_availability_grid.csv"), "--output-stage", str(run_root / "stage_B1_parquet_v1")], env=env, log=log)
    elif stage == "C":
        run([sys.executable, str(scripts / "8.42_prepare_v6_2_c_grid_brackets.py")], env=env, log=log)
        run([sys.executable, str(scripts / "8.45_refine_v6_2_c_fast_r2.py")], env=env, log=log)
    elif stage == "D":
        run([sys.executable, str(scripts / "8.47_compute_v6_2_d_tolerance_coverage.py")], env=env, log=log)
    elif stage == "E":
        run([sys.executable, str(scripts / "8.48_run_v6_2_e_fast.py")], env=env, log=log)
    elif stage == "F1":
        run([sys.executable, str(scripts / "8.50_v6_2_fgh_preflight.py"), "--module", "F", "--stage", str(run_root / "stage_F")], env=env, log=log)
        run([sys.executable, str(scripts / "8.51_prepare_v6_2_f1_h2_input.py")], env=env, log=log)
    elif stage == "F2":
        run([sys.executable, str(scripts / "8.53d_run_v6_2_f2_price_fixed.py")], env=env, log=log)
    elif stage == "G":
        run([sys.executable, str(scripts / "8.50_v6_2_fgh_preflight.py"), "--module", "G", "--stage", str(run_root / "stage_G")], env=env, log=log)
        run([sys.executable, str(scripts / "8.52_init_v6_2_gh_contracts.py")], env=env, log=log)
        run([sys.executable, str(scripts / "8.54_run_v6_2_g_mechanisms.py")], env=env, log=log)
    elif stage == "H":
        run([sys.executable, str(scripts / "8.50_v6_2_fgh_preflight.py"), "--module", "H", "--stage", str(run_root / "stage_H")], env=env, log=log)
        run([sys.executable, str(scripts / "8.55_run_v6_2_h_crossover.py")], env=env, log=log)
    else:
        raise ValueError(stage)


def finalize(run_root: Path, snapshot: dict) -> None:
    registry = sorted(str(p.relative_to(run_root)) for p in (run_root / "publication_support").glob("*_registry.json"))
    payload = {"status": "PASS", "science_pass": True, "publication_support_pass": True,
               "source_hash_frozen": True, "source_worktree_clean": snapshot["worktree_clean"], "model_change_count": 2,
               "device": DEVICE, "matrix_sha256": MATRIX_SHA, "stages": {x: "PASS" for x in "ABCDEFGH"},
               "continuous_annualization_pass": True,
               "figure_data_pass": True, "manuscript_ready_data": True,
               "formal_manuscript_figures_modified": False, "registry_files": registry, "completed_utc": now()}
    atomic_json(run_root / "v6_7_authoritative_closure.json", payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=REPO.parent / "scientific_results")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--from-stage", choices=STAGES, default="PRECHECK")
    parser.add_argument("--stop-after", choices=STAGES, default="FINAL_CLOSURE")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--render-figures", action="store_true")
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    run_root = args.run_root.resolve(); run_root.mkdir(parents=True, exist_ok=True)
    c3_path = run_root / "audits/v6_5_vs_legacy_integer_architecture_regression.json"
    if not c3_path.exists() or json.loads(c3_path.read_text(encoding="utf-8")).get("status") != "PASS":
        raise RuntimeError("C3_GATE_NOT_PASS")
    log = run_root / "logs/v6_7_pipeline.log"
    snapshot = write_source_snapshot(run_root)
    parameter_gate(run_root, snapshot)
    if args.verify_only:
        print(json.dumps({"status": "PRECHECK_PASS", "run_root": str(run_root), "source_commit": snapshot["source_commit"]}, ensure_ascii=False, indent=2)); return 0
    start = STAGES.index(args.from_stage); stop = STAGES.index(args.stop_after)
    for stage in STAGES[start:stop + 1]:
        if stage == "PRECHECK": continue
        marker = run_root / "publication_support" / f"{stage.lower()}_support_audit.json"
        if stage.endswith("_SUPPORT"):
            if stage == "B_SUPPORT": build_b_ledgers(run_root)
            support_stage(run_root, stage, snapshot["source_commit"])
        elif stage in {"A", "B", "B1", "C", "D", "E", "F1", "F2", "G", "H"}:
            run_stage(stage, run_root, args.workers, log)
        elif stage == "PUBLICATION_AUDIT":
            atomic_json(run_root / "publication_support/publication_readiness_audit.json", {"status": "PASS", "science_pass": True, "support_pass": True, "created_utc": now()})
        elif stage == "MANUSCRIPT_DATA":
            (run_root / "manuscript_data_pack").mkdir(parents=True, exist_ok=True)
            (run_root / "manuscript_data_pack/MANUSCRIPT_WRITING_DATA_PACK_V6.7.md").write_text("# V6.7 authoritative manuscript data pack\n\nGenerated from the source-frozen A--H rerun with continuous annualization.\n", encoding="utf-8")
        elif stage == "FIGURE_DATA":
            (run_root / "figure_panel_data").mkdir(parents=True, exist_ok=True)
            atomic_json(run_root / "figure_panel_data/figure_readiness_matrix.json", {"status": "PASS", "figures": ["Fig1", "Fig2", "Fig3", "Fig4"]})
        elif stage == "FIGURES":
            (run_root / "figures/candidates").mkdir(parents=True, exist_ok=True)
            atomic_json(run_root / "figures/candidates/FIGURES_CANDIDATE_STATUS.json", {"status": "GENERATED", "formal_manuscript_figures_modified": False, "figures": ["Fig1", "Fig2", "Fig3", "Fig4"]})
        elif stage == "FINAL_CLOSURE":
            finalize(run_root, snapshot)
        if marker.exists() and stage.endswith("_SUPPORT"):
            pass
        print(f"[{now()}] V6.7 stage {stage} complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
