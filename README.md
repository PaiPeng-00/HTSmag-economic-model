# HTS magnet economics — manuscript V10

Companion code for **Fusion economics are robust to performance degradation in high-temperature superconducting magnets**.
This release is `v3.0.0`. The corresponding GitHub release URL is
<https://github.com/PaiPeng-00/HTSmag-economic-model/releases/tag/v3.0.0>.
The frozen companion dataset uses the reserved Zenodo DOI
<https://doi.org/10.5281/zenodo.22733646>; the DOI will resolve after the Zenodo draft is published.
Code is MIT licensed; data are licensed under CC BY 4.0.

## Quick numerical reproduction

Use Python 3.11 in a clean environment (validated with 3.11.7). Dependencies are pinned to the tested versions. Commands below run from this repository root:

```sh
python -m venv .venv
# Activate .venv with your operating system's usual command.
python -m pip install -r requirements.txt
python reproduce.py --data ../zenodo_data --output outputs/reproduced
```

The command verifies SHA-256 checksums and recalculates scenario minima, availability and validity flags, cross-scenario LCOE penalties, Fig. 3D, Fig. 4A–C, Fig. S7, Fig. 5A/B, Table S4 and the conservative joint limits at all four LCOE margins. Numerical agreement is checked against the archived final tables. It does not solve the physics grid. Several GB of RAM are needed; the joint-boundary reduction can take several minutes.

## Model input preparation and full rerun

```sh
python reproduce.py --data ../zenodo_data --hydrate-only
python model/model_code/scripts/run_full_grid.py --preflight-only
python tests/smoke_model.py
python tests/tiny_grid_regression.py --data ../zenodo_data
python model/model_code/scripts/run_full_grid.py
```

The last command is the expensive **full physics/economics run**, not part of the quick check. It first solves the scalar circuit cache, then evaluates 200 tape counts × 61 contact resistivities × 121 joint resistances × 4 temperature/coolant cases × 3 scenarios = 17,714,400 scenario records. Allow substantial runtime and tens of GB of disk space. Do not run multiple jobs into the same output directory. A complete independent run was executed on 2026-09-20 in eight disjoint Npw partitions, with no numerical-model or grid changes. See `VERIFICATION.json` for the completed validation scope and status.

The raw scan is written beneath `model/model_code/code/outputs/arc_16pancake_nuc600_v6_2/tables/`. Convert a newly generated scan to V10 ledgers and final affected figure inputs:

```sh
python pipeline/build_v10_candidate_ledgers.py --raw model/model_code/code/outputs/arc_16pancake_nuc600_v6_2/tables/scan_full_grid_tidy_plant_opex_2025usd_direct_hts.csv --output outputs/rebuilt_v10
python pipeline/build_v10_candidate_figure_inputs.py --candidate outputs/rebuilt_v10 --raw model/model_code/code/outputs/arc_16pancake_nuc600_v6_2/tables/scan_full_grid_tidy_plant_opex_2025usd_direct_hts.csv --output outputs/rebuilt_figure_inputs
python summarize_new_ledgers.py --source outputs/rebuilt_v10/full_model_dataset --output outputs/rebuilt_figure_inputs/main_figures/Fig4_latest
python pipeline/build_v10_supplementary_inputs.py --raw model/model_code/code/outputs/arc_16pancake_nuc600_v6_2/tables/scan_full_grid_tidy_plant_opex_2025usd_direct_hts.csv --output outputs/rebuilt_supplementary
python pipeline/build_v10_circuit_figure_inputs.py --cache model/model_code/code/outputs/tables/circuit_scalar_cache.csv --output outputs/rebuilt_circuit_figures
```

These scripts reproduce the capital-anchor LCOE ledger, Fig. 3B/D, Fig. 5A/B/C and the current Fig. 4/S7 tables. The supplementary reducer also re-solves the eleven Fig. S3 cases and selects Fig. S4/S5 from the new raw scan. Magnetization histories use the same linear time interpolation as the model. Nuclear heat uses the V10 structural-volume definition (3111.5253 W per TF magnet). The obsolete archived population-matrix dependency has been removed from the release copy. The standalone `summarize_new_ledgers.py` generates Fig. 4/S7 tables without requiring a packaged manifest; `reproduce.py` additionally validates a frozen release against its reference data.

The three required XLSX input files are supplied in the data package. They include FEM-derived inductance and loss inputs. Reproducing the upstream COMSOL calculations from geometry is a separate licensed-solver workflow, not a dependency of this Python model or a claim of this release.

## Figure sources and exclusions

All final numerical panel data are in `zenodo_data/figure_data`; see `FIGURE_DATA_MAP.md`. Native renderers for Fig. 4, Fig. 5 and Figs. S3–S5/S7 are included. After hydration:

```sh
python figures/publication_figures/figures/Fig3/plot/render_fig4_lcoe_temperature_v10.py
python figures/publication_figures/figures/Fig3/plot/render_figs7_lcoe_full_tail_v10.py
python figures/publication_figures/figures/Fig4/plot/render_fig5_temperature_v8.py --language EN
python figures/publication_figures/programs/render_figs3_v10.py --data ../zenodo_data/figure_data --output outputs/FigS3
python figures/publication_figures/programs/build_combined_figs4.py --data-4p2 ../zenodo_data/figure_data/supplementary_figures/FigS4/heatload_components_4p2K.parquet --data-10 ../zenodo_data/figure_data/supplementary_figures/FigS4/heatload_components_10K.parquet --output-dir outputs/FigS4
python figures/publication_figures/programs/render_figs5_v10.py --data ../zenodo_data/figure_data --output outputs/FigS5
python tests/verify_supplementary.py --data ../zenodo_data --output outputs/supplementary_check.json
```

Some retained internal filenames contain older version labels; their source hashes identify the actual V10 implementation. The displayed Fig. 4 is internally asset `Fig3`, and displayed Fig. 5 is asset `Fig4`. Fig. 4B uses a logarithmic axis, median points, P10–P90 thick intervals and full finite-range thin intervals, without a 10% reference line. The approved supplementary style requires Arial; install that font for the native supplementary renderers. Font changes do not change numerical data. The supplementary check re-solves all Fig. S3 cases and checks the S4/S5 contracts; adding `--raw path/to/new_scan.csv` also compares every S4/S5 value against that independently generated scan.

No manuscript PDF/Word files, author artwork, rendered figures, circuit traces, temporary raw scans, caches, audit screenshots, earlier releases, or COMSOL binaries are included. Figures 1, 2A, S2 and S6 contain explanatory artwork rather than additional numerical results. The archive supports numerical reproduction; it does not promise pixel-identical reproduction of excluded author artwork.

## Statistical contract

Each complete five-parameter realization has equal weight. Availability requires `Aplant >= 0.80` in every scenario. LCOE is valid only with positive net export and finite positive LCOE in every scenario. Scenario-specific minima are determined within each scenario's own availability-qualified, valid population. The penalty is the maximum of the three relative increases. Invalid LCOE is excluded from finite-penalty distributions but stays in the availability-qualified denominator of design fractions. At 20 K, individual helium and hydrogen realizations are pooled, not averaged as two curves. Percentiles use linear interpolation. Joint limits use the first contiguous passing interval from the low-resistance endpoint, interpolate a valid crossing linearly in resistance and penalty, and take the conservative minimum across scenarios and all 61 contact resistivities. A still-passing 100 nΩ endpoint is right-censored, not a measured threshold above 100 nΩ.

## Provenance and release status

`SOURCE_PROVENANCE.json` records source and release-copy hashes, including the small portability changes. `MANIFEST.json` covers the files in each package. `RELEASE_METADATA.json` freezes the public version, repository release URL, and companion dataset DOI. `environment-tested.txt` records versions used for local validation, while `requirements.txt` specifies installable dependencies. Existing project licenses and public author metadata are retained. This package preparation does not upload files, create a remote tag, publish the Zenodo draft, or register the reserved DOI.
