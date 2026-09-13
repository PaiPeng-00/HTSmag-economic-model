# Manuscript numbers and their source files

Paths are relative to `scientific_results_v7_splice_equivalent_20260906/experiments/`.
Manuscript precision: derived results ≥ 1 to one decimal place, < 1 to two significant
figures, unless a threshold crossing needs more. The files keep full precision.

| Manuscript value | Where it appears | File | Row / column |
|---|---|---|---|
| Economic retention 82.9% (10%); 4.7%, 57.9%, 91.1% (1, 5, 20%) | Abstract, Intro, Results 3, Discussion, Fig. 1, Fig. 4D, Table S4 | `economic_retention_v8/economic_retention_1_5_10_20.csv` | `f_econ_given_avail_pct` |
| Common availability population = 83.1% of the grid | Note S6 | `full_realization_robustness_matrix_v7/Y0_Y3_population_matrix.csv`; `economic_retention_v8/economic_retention_1_5_10_20.csv` | `N_availability / N_all` |
| Refrigeration fraction 0.6–121.5%, median 2.0%, P10–P90 0.94–8.7% | Abstract, Intro, Results 2, Discussion, Fig. 1, Note S6 | `fig3d_common_availability_v8/YD_overall_summary.csv` | `min_pct`…`max_pct` |
| Median refrigeration 6.6→1.6% (S1), 4.8→1.2% (S2), 4.4→1.1% (S3); Fig. 3D | Results 2, Results 4, Discussion | `fig3d_common_availability_v8/YD_nine_group_summary.csv` | `median_pct` by scenario and `Top_K` (20 K pools He and H2) |
| 10%-retained magnets: refrigeration 0.6–8.6%, median 1.6%, P10–P90 0.90–4.1% | Results 3, Note S6 | `economic_retention_v8/refrigeration_before_after_10pct.csv` | row `availability_and_10pct` |
| Scenario-specific minimum LCOE 1122, 322, 79 US$/MWh (SI 1122.1, 322.0, 78.8); 10% margins | Results 3, Fig. 4B, Note S6 | `economic_retention_v8/scenario_10pct_margins.csv` | `minimum_lcoe_USD_MWh`, `margin_USD_MWh` |
| Temperature-balanced retention 79.1% | Note S6 | `temperature_balance_v8/temperature_balanced_retention.csv` | `temperature_balanced_pct` at `margin_pct` 10 |
| Conservative tolerance 74.4, ≥100, ≥100 nΩ (50 tapes); 3.6, 14.6, 37.1 nΩ (200 tapes); ratio 10.3 | Abstract, Intro, Fig. 1, Results 4, Fig. 5A/B, Discussion, Note S7 | `fig5_interpolated_v8/Fig5B_conservative_boundaries.csv` | `Rj_tol_nOhm`, `upper_censored`, `controlling_scenario` |
| Individual-resistivity spread < 0.2% (200 tapes); 74.4–78.6 nΩ (50 tapes, 4.2 K) | Note S7 | `fig5_interpolated_v8/individual_rho_interpolated_boundaries.csv` | `Rj_tol_nOhm` at `epsilon` 0.1 |
| Conservative limit defined from 48 tapes per turn at 20 K | Results 4 | `fig5_interpolated_v8/Fig5B_conservative_boundaries.csv` | first finite `Rj_tol_nOhm` |
| Conservative threshold sensitivity at 200 tapes and He: no qualifying interval at 1% for 4.2/20 K; 1.2 nΩ/no qualifying interval at 5%; 3.6/37.1 nΩ and ratio 10.3 at 10%; 7.8/98.1 nΩ and ratio 12.5 at 20% | Note S7 | `fig5_interpolated_v8/all_margins_conservative_boundaries.csv` | `Rj_tol_nOhm` at `Npw=200`, `Top_K=4.2/20`, `epsilon=0.01/0.05/0.10/0.20`; each row already minimizes over S1–S3 and all 61 `rho_turn` values |
| Fig. 5A heat maps and hatching | Fig. 5A | `fig5_interpolated_v8/Fig5A_conservative_heatmaps.csv` | `delta_max_pct`, `all_valid_economics` |
| Lowest S2 LCOE 323.1, 322.0, 329.6 US$/MWh (He; 2.4% spread) | Results 4, Discussion | `results4_temperature_price_v8/S2_temperature_minima_common_availability.csv` | `S2_minimum_USD_MWh` (He rows) |
| Preferred temperature 10 K at 50 and 100, 20 K at 10 US$/kA·m | Results 4, Fig. 5C, Discussion, Note S7 | `fig5_interpolated_v8/Fig5C_price_sensitivity.csv` | `preferred_temperature_K` |

Not produced here: charging time, plant-availability maxima, heat-load and charging-loss
values (Results 1–2, Figs. 2–3A–C, Figs. S3–S6) come from the upstream model outputs in
Data S1. The YT (paired temperature) and YR (base-state tolerance ratio) groups are
supporting analyses and are not quoted in the current manuscript. The YJ native-grid
endpoints (3.548/36.869 nΩ) were superseded by the interpolated tolerance above.
