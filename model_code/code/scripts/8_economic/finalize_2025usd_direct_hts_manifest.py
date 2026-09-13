"""Finalize the direct-HTS constant-2025-US$ scan and figure manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("FUSION_DEVICE", "arc_16pancake_nuc600")

from fusion_tem import device as cfg
from fusion_tem.economic.cost_boundary import COST_BOUNDARY_VERSION
from fusion_tem.economic.price_basis import (
    CONVERSION_TABLE_PATH,
    HTS_PRICE_CONVERSION_METHOD,
    PRICE_BASIS_YEAR,
    conversion_table_sha256,
    monetary_conversion_audit_rows,
)
from tfmag.manifest import write_manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=cfg.OUTPUTS_FIGURES_DIR
        / "plant_v4_2025usd_direct_hts_submission_candidate",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=cfg.OUTPUTS_DIR / "manifest_plant_opex_2025usd_direct_hts.json",
    )
    args = parser.parse_args()

    tables = cfg.OUTPUTS_TABLES_DIR
    candidate = args.candidate_root.resolve()
    outputs = [
        tables / "scan_full_grid_tidy_plant_opex_2025usd_direct_hts.csv",
        tables / "scan_full_grid_plant_opex_2025usd_direct_hts.xlsx",
        tables / "economic_2025usd_direct_hts_conversion_audit.csv",
        tables / "economic_2025usd_direct_hts_change_metrics.csv",
        tables / "economic_2025usd_direct_hts_change_summary.md",
        tables / "economic_2025usd_direct_hts_invariance_audit.json",
        tables / "economic_2025usd_direct_hts_nonmonetary_harmonization.json",
        tables / "robust_frontier_5pct_2025usd_direct_hts.csv",
        candidate / "svg" / "Fig5.svg",
        candidate / "svg" / "Fig6.svg",
        candidate / "pdf" / "Fig5.pdf",
        candidate / "pdf" / "Fig6.pdf",
        candidate / "previews" / "Fig5.png",
        candidate / "previews" / "Fig6.png",
        candidate / "figure_manifest.json",
    ]
    missing = [str(path) for path in outputs if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required direct-HTS 2025-USD outputs are missing: " + "; ".join(missing))

    audit = json.loads(
        (tables / "economic_2025usd_direct_hts_invariance_audit.json").read_text(
            encoding="utf-8"
        )
    )
    figures = json.loads(
        (candidate / "figure_manifest.json").read_text(encoding="utf-8")
    )
    if audit.get("price_basis_year") != PRICE_BASIS_YEAR:
        raise AssertionError("Invariance audit price basis is not 2025")
    hts_rows = [
        row
        for row in monetary_conversion_audit_rows()
        if row["parameter"].startswith("p_HTS")
    ]
    if len(hts_rows) != 3:
        raise AssertionError(f"Expected three HTS price rows, found {len(hts_rows)}")
    if {float(row["value_2025_USD"]) for row in hts_rows} != {
        100.0,
        50.0,
        10.0,
    }:
        raise AssertionError("HTS price levels are not the direct 100/50/10 scenarios")
    for row in hts_rows:
        if float(row["source_price_year"]) != 2025.0:
            raise AssertionError("HTS source price year must be 2025")
        if float(row["CPI_2025_over_CPI_y"]) != 1.0:
            raise AssertionError("HTS CPI factor must be 1.0")
        if row["conversion_method"] != HTS_PRICE_CONVERSION_METHOD:
            raise AssertionError("HTS conversion method is not the direct-scenario method")
        if bool(row["price_year_is_proxy"]):
            raise AssertionError("Direct HTS scenario prices must not be proxy-priced")
        if "range/basis only" not in row["basis_note"]:
            raise AssertionError("HTS literature note must be range/basis only")
    scan_sha256 = _sha256(outputs[0])
    if audit.get("new_sha256") != scan_sha256:
        raise AssertionError("Invariance audit and final scan SHA256 differ")
    if set(figures) != {"Fig5", "Fig6"}:
        raise AssertionError(f"Candidate figure set mismatch: {sorted(figures)}")
    for name, record in figures.items():
        if record.get("metric") != COST_BOUNDARY_VERSION:
            raise AssertionError(f"{name} cost boundary mismatch")
        if not record.get("artwork_preserved"):
            raise AssertionError(f"{name} artwork-preservation check failed")
        if record.get("scan_input_sha256") != scan_sha256:
            raise AssertionError(f"{name} and final scan SHA256 differ")
        width_mm, height_mm = record["pdf_size_mm"]
        if abs(float(width_mm) - 160.0) > 0.02:
            raise AssertionError(f"{name} width is not 160 mm")
        if abs(float(height_mm) - 116.364) > 0.02:
            raise AssertionError(f"{name} height is not 116.364 mm")
    fig6_svg_text = (candidate / "svg" / "Fig6.svg").read_text(
        encoding="utf-8"
    )
    for token in ("100 US$", "50 US$", "10 US$", "kA", "baseline-shift=\"super\"", "−1"):
        if token not in fig6_svg_text:
            raise AssertionError(f"Fig6 label token is missing: {token}")
    if fig6_svg_text.count("baseline-shift=\"super\"") < 6:
        raise AssertionError("Fig6 requires two superscript -1 tspans per price row")
    for forbidden in ("52.8 US$", "12.6 US$"):
        if forbidden in fig6_svg_text:
            raise AssertionError(f"Fig6 contains superseded HTS price label: {forbidden}")

    write_manifest(
        repo_root=cfg.REPO_ROOT,
        config_path=cfg.CONFIGS_DIR / "scan_full_grid.yaml",
        inputs_root=cfg.DATA_RAW_DIR,
        outputs_root=cfg.OUTPUTS_DIR,
        manifest_path=args.manifest.resolve(),
        output_paths=outputs,
        metadata={
            "cost_boundary_version": COST_BOUNDARY_VERSION,
            "monetary_price_basis_year": PRICE_BASIS_YEAR,
            "monetary_values_constant_2025_usd": True,
            "hts_conductor_price_source_year": 2025,
            "hts_conductor_price_cpi_factor": 1.0,
            "hts_conductor_price_conversion_method": "direct_scenario_assumption_2025usd",
            "hts_conductor_price_levels_2025_USD_per_kAm": [100.0, 50.0, 10.0],
            "conversion_table_path": str(
                CONVERSION_TABLE_PATH.relative_to(cfg.REPO_ROOT)
            ),
            "conversion_table_sha256": conversion_table_sha256(),
            "scan_rows": audit["new_rows"],
            "scan_sha256": audit["new_sha256"],
            "nonmonetary_invariance": audit[
                "nonmonetary_bitwise_string_comparison"
            ]["status"],
            "jointly_feasible_counts_identical": audit[
                "jointly_feasible_counts_identical"
            ],
            "candidate_figure_manifest_sha256": _sha256(
                candidate / "figure_manifest.json"
            ),
            "formal_figure_package_overwritten": False,
        },
    )
    print(f"[OK] Final manifest: {args.manifest}")


if __name__ == "__main__":
    main()