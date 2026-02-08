# TF Magnet Parameter Scan & Analysis

This repository scans TF magnet design parameters, computes electromagnetic / thermal / economic metrics, and generates figures based on a unified results table.

## Quick Start

1) Put input data into `data/raw/` (see below).
2) Edit scan parameters in `configs/scan_full_grid.yaml`.
3) Run the full parameter scan:

```powershell
python analysis_scan_full_grid.py
```

Outputs are saved to:

- `outputs/tables/scan_full_grid_tidy.csv`
- `outputs/tables/scan_full_grid.xlsx`
- `outputs/manifest.json`

## Installation / Environment

- **Python**: 3.9 or newer.
- **Clone** the repository and run all scripts from the repo root.

Optional but recommended: use a virtual environment (venv or conda), then install dependencies:

```powershell
# Using pip (from repo root)
pip install -r requirements.txt
```

If you use YAML scan config (`configs/scan_full_grid.yaml`), install PyYAML as well:

```powershell
pip install pyyaml
```

Main dependencies (see `requirements.txt`): `numpy`, `pandas`, `matplotlib`, `scipy`, `openpyxl`, and optionally `seaborn`, `numba`, `tabulate`, `cairosvg` for full plotting and export support.

## Directory Layout

```
configs/            # YAML scan configuration
data/
  raw/              # raw inputs (xlsx, matrices, constant tables, etc.)
  processed/        # optional normalized inputs
outputs/
  figures/          # svg/pdf/png figures
  tables/           # csv/xlsx tables (including cooldown summaries)
  logs/             # run logs
src/tfmag/          # shared utilities (paths, manifest)
```

## Script Naming & Grouping

- `analysis_*`: analysis / computation / data-processing / scan scripts (e.g. `analysis_scan_full_grid.py`, `analysis_heatload_2bars.py`)
- `plot_*`: pure plotting scripts (e.g. `plot_scan_full_grid_outputs.py`)
- `stitch_*`: export/figure stitching scripts (e.g. `stitch_charging_tf_3temps.py`, `stitch_economic_svgs.py`)

All scripts are intended to be run from the repo root; outputs stay in existing subdirectories under `outputs/`.

### Root-Level Scripts Overview (currently present)

Below is a brief overview of scripts in the repo root that are directly used in the main workflow (only listing files that actually exist):

- **Core scanning & data generation**
  - `analysis_scan_full_grid.py`: main parameter scan script; produces the unified scan results table (`scan_full_grid_tidy.csv/xlsx`), which is the basis for all subsequent plots and economic analyses.
  - `analysis_calculate_inductance.py`: computes *single‑case* inductance matrices (per selected Npw) and saves them under `data/raw/inductance/` for inspection / debugging.
  - `analysis_economic_unified_counter.py`: unified economic analysis based on the scan table; generates LCOE, CAPEX/OPEX and other economic plots.
  - `analysis_availability_factor_full_grid.py`: computes / analyzes availability factor based on the scan table.
  - `analysis_heatload_2bars.py`: computes and analyzes heat loads at 2 bar, usually as a sub‑study for thermal conditions.
  - `analysis_charge_time999.py`: scans 99.9% charging time of the TF system over different Npw.

- **Plotting scripts (`plot_*`)**
  - `plot_scan_full_grid_outputs.py`: reads `scan_full_grid_tidy.csv` and generates heatmaps / contour plots for ΔLCOE, parasitic power, charging time, etc.
  - `plot_library.py`: plotting utility library; encapsulates common helpers (heatmaps, contours, colorbars, auto contour levels, etc.) used by other plotting scripts.

- **Figure stitching scripts (`stitch_*`)**
  - `stitch_charging_tf_3temps.py`: stitches TF charging figures at different temperatures (time series, I(t), P_loss(t), etc.) into well‑laid‑out composites.
  - `stitch_economic_svgs.py`: stitches economic SVG figures (ΔLCOE heatmaps, payback time plots, etc.) into large panels for papers/reports.
  - `stitch_heatload_charging_timeseries.py`: stitches heat‑load time‑series plots during charging.
  - `stitch_heatload_relative_grid.py`: stitches grids of relative heat load / parasitic power (usually derived from the scan table).
  - `snitch_inductance_tf_system.py`: legacy‑named script for processing/organizing TF system inductance matrix figures or intermediate artifacts (functionally similar to inductance figure stitching / post‑processing).

- **Models & base utilities**
  - `model_cryogenic.py`: cryogenic system model, including refrigeration power, COP, and heat‑load breakdown across temperature levels.
  - `model_economic.py`: economic model, including LCOE, CAPEX/OPEX, annualized cost and revenue, etc.
  - `material_properties.py`: material properties and interpolators for HTS tapes, structural materials, coolants, etc.
  - `config.py`: global configuration (scan parameters, default paths, unit conventions, etc.).
  - `utils.py`: common utility functions for I/O, unit conversion, path helpers, and some plotting/post‑processing helpers.
  - `src/tfmag/manifest.py`, `src/tfmag/paths.py`: path and manifest helpers, providing unified access to data and output locations from different scripts.

Other ad‑hoc scripts (e.g. `test.py`) are only for local testing or one‑off experiments and are typically not used in the main paper workflow.

## Key Inputs (`data/raw`)

Please organize the following according to the existing data structure:

- Inductance‑related inputs (e.g. `data/raw/inductance/`)
- Electromagnetic loss data at different temperatures:
  - `data/raw/heat_input/mag loss.xlsx`
  - `data/raw/heat_input/radial loss.xlsx`
- Thermal‑condition data:
  - `data/raw/charge/`, `data/raw/operation/`, `data/raw/cooldown/`, `data/raw/quench/`

## Scan Configuration

Edit `configs/scan_full_grid.yaml` to set:

- `temp_coolant_pairs`
- `npw_values`
- `rho_turn_uohm_cm2_values`
- `r_joint_nohm_values`
- `scenarios`

The scan uses these parameters directly; the core scan logic in `analysis_scan_full_grid.py` remains unchanged.

## Core Data Generation: `analysis_scan_full_grid.py`

`analysis_scan_full_grid.py` is the **single source of truth for all plotting data**. It runs the full parameter‑grid scan and produces a unified table of all design‑point combinations and their computed results.

### Scan Dimensions

The script scans all combinations over:

- **Temperature–coolant pairs** (`temp_coolant_pairs`):
  - (4.2 K, He)
  - (10 K, He)
  - (20 K, He)
  - (20 K, H₂)
- **Number of parallel stacks** (`npw_values`): 1–200 (from `NPW_SCAN_VALUES` in `config.py`)
- **Turn‑to‑turn resistivity** (`rho_turn_uOhm_cm2_values`): logarithmic spacing, 10–10000 μΩ·cm² (31 points by default)
- **Inter‑coil joint resistance** (`r_joint_nohm_values`): from `R_JOINT_SCAN_VALUES` in `config.py` (log‑spaced, nΩ)
- **Technology scenarios** (`scenarios`): S1, S2, S3 (extendable)

### Outputs

The script writes two equivalent outputs (same content, different formats):

1. **`outputs/tables/scan_full_grid_tidy.csv`** (primary output)
   - Long, tidy table: each row is one parameter combination plus all computed metrics
   - Contains all metrics listed below
   - **Only data source** for all downstream plotting scripts
  
2. **`outputs/tables/scan_full_grid.xlsx`** (optional)
   - Excel version for reviewers
   - Content identical to the CSV

### Metrics

For each combination the script computes:

#### 1. Charging‑time metrics
- `Charging_time_999_h` / `t_charge_999_h`: time (hours) to reach 99.9% of target current

#### 2. Electromagnetic loss metrics
- `Mag_loss_at_charge_W`: magnetization‑loss power (W) at the end of the charging ramp
- `Radial_loss_at_charge_W`: radial‑loss power (W) at the end of the charging ramp
- `Mag_loss_energy_MWh`: total magnetization‑loss energy (MWh) over charging
- `Radial_loss_energy_MWh`: total radial‑loss energy (MWh) over charging

#### 3. Thermal metrics
- `Total_heat_Tc_W`: total heat load (W) at operating temperature Top
- `Total_heat_77K_W`: total heat load (W) at 77 K
- `P_cryo_electric_W`: cryoplant electric power (W)
- `P_fusion_electric_W`: plant gross electric power (W)
- `COP_Tc` / `COP_77`: COP at operating temperature / 77 K
- `Parasitic_fraction`: parasitic‑power fraction
- `r_cryo_re_producton`: cryogenic power as a fraction of fusion power (%)
- Detailed heat‑source breakdown: `Coil_internal_joint_W`, `Pancake_joint_W`, `Nuclear_heat_W`, `Radiation_W`, etc.

#### 4. Electromagnetic metrics
- `R_radial_per_pancake_Ohm`: radial resistance of one pancake (Ω)
- `R_radial_per_TF_Ohm`: radial resistance of one TF magnet (Ω)
- `R_radial_system_Ohm`: effective radial resistance of the TF system (Ω)

#### 5. Economic metrics
- `LCOE_magnet_only_USD_per_MWh`: LCOE of the magnet system only (USD/MWh)
- `LCOE_magnet_only_CNY_per_kWh`: same LCOE in CNY/kWh
- `CAPEX_mag_direct_USD`: direct magnet CAPEX (USD)
- `CAPEX_mag_installed_USD`: installed magnet CAPEX (USD)
- `Annualized_CAPEX_mag_USD_per_year`: annualized magnet CAPEX (USD/year)
- `Annual_OPEX_mag_USD_per_year`: annual magnet OPEX (USD/year)
- `Tape_cost_USD`: HTS tape cost (USD)
- `Coolant_fill_cost_USD`: coolant fill cost (USD)
- `Power_supply_cost_USD`: power‑supply cost (USD)
- `Net_annual_profit_USD`: net annual profit (USD)
- `Simple_payback_years`: simple payback time (years)
- `Lifetime_revenue_USD`: lifetime revenue (USD)
- `AF`: availability factor = annual production hours / total yearly hours
- `r_cryo_re`: cryogenic power as a fraction of plant power (%)
- Cryo‑power breakdown: `P_cryo_prod_W`, `P_cryo_dwell_W`, `P_cryo_static_W`, `P_cryo_coolwarm_W`, `P_cryo_excdec_W`
- Annual energies: `E_cryo_year_MWh`, `E_gross_year_MWh`, `E_net_year_MWh`

#### 6. Derived metrics
- `AF_ref`: relative availability = AF / AF_max (within each technology scenario)
- `delta_LCOE_min_USD_per_MWh`: LCOE increment above the minimum feasible LCOE in that scenario (USD/MWh)
- `feasible_charge_time`: feasibility flag for charging time (charging time ≤ 2×`CHARGE_HOURS`)
- `feasible_r_cryo_re`: feasibility flag for parasitic power (`r_cryo_re` ≤ threshold)
- `feasible`: overall feasibility flag (both constraints satisfied)
- `feasibility_flags`: textual description of violated constraints

### Data Usage

**All plotting scripts read from `scan_full_grid_tidy.csv`, including:**

- `plot_scan_full_grid_outputs.py`: heatmaps from the scan table
- `analysis_economic_unified_counter.py`: economic plots (also depends on the scan table)
- `stitch_economic_svgs.py`: stitched economic figures

### Configuration

Scan parameters are configured in `configs/scan_full_grid.yaml`:

- **`temp_coolant_pairs`**: list of temperature–coolant pairs
  - Default (`config.py`): `[(4.2, "He"), (10.0, "He"), (20.0, "He"), (20.0, "H2")]`
  - 4 combinations: 4.2K He, 10K He, 20K He, 20K H₂
 
- **`npw_values`**: array of parallel‑stack counts
  - Default (`config.py`): `np.arange(10, 201, 10)` = `[10, 20, 30, 40, ..., 200]` (20 values)
  - YAML example: `[1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]` (21 values)
  - Typical range: 1–200
 
- **`rho_turn_uOhm_cm2_values`**: array of turn‑to‑turn resistivity values (supports log‑space specification)
  - Default (if not set in YAML): `np.logspace(np.log10(10), np.log10(10000), 31)` = 31 log‑spaced values from 10–10000 μΩ·cm²
  - YAML example (explicit list): `[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 5000, 10000]`
  - YAML example (log‑space spec):
    ```yaml
    rho_turn_uOhm_cm2_values:
      logspace:
        start: 10
        stop: 10000
        num: 31
        include: [5000]  # optional extra values
    ```
  - Typical range: 10–10000 μΩ·cm²
  - Note: if 5000.0 is missing, the script automatically adds it.
  
- **`r_joint_nohm_values`**: array of inter‑coil joint resistances (log‑spaced)
  - Default (`config.py`): 31 log‑spaced values from 1–100 nΩ
    ```
    [1.000, 1.166, 1.359, 1.585, 1.848, 2.154, 2.512, 2.929, 3.415, 3.981,
     4.642, 5.412, 6.310, 7.356, 8.577, 10.000, 11.659, 13.594, 15.849, 18.478,
     21.544, 25.119, 29.286, 34.145, 39.811, 46.416, 54.117, 63.096, 73.564, 85.770,
     100.000]
    ```
  - Range: 1–100 nΩ (31 log‑spaced points)
 
- **`scenarios`**: list of technology scenarios
  - Default (`config.py`): `['S1', 'S2', 'S3', 'S4', 'S5', 'S6']` (all available scenarios)
  - YAML example: `['S1', 'S2', 'S3']` (commonly just the first three)
  - Scenario meanings:
    - **S1, S2, S3**: baseline scenarios (HTS tape price 50 $/kA‑m with different operating assumptions)
    - **S4**: extended scenario
    - **S5**: HTS tape price sensitivity (high price: 100 $/kA‑m)
    - **S6**: HTS tape price sensitivity (low price: 10 $/kA‑m)

If the config file is missing, the script falls back to defaults in `config.py`.

### Resume Capability

The scan supports resume:
- If outputs already exist, previously computed combinations are detected and skipped.
- Only new parameter combinations are computed, which saves a lot of time.

### Performance Notes

- Uses `@lru_cache` to cache repeated computations (charging time, resistance, heat loads, etc.)
- Pre‑loads charging‑time DataFrames and EM‑loss data (each temperature is read only once)
- Writes CSV in batches (flush every 500 rows)

## Plotting Workflow

### Plots Based on the Scan Table

Plots should use the tidy scan table whenever possible:

- `plot_scan_full_grid_outputs.py` reads `outputs/tables/scan_full_grid_tidy.csv`
- Economic plots: `analysis_economic_unified_counter.py`
- Stitched figures: `stitch_economic_svgs.py`

### Inductance Matrix Heatmaps

Inductance matrix heatmaps visualize the TF system inductance matrix, showing mutual inductances between coils.

**Plotting script:**
- `snitch_inductance_tf_system.py`: reads `TF_system_L_matrix.xlsx` and generates / stitches inductance heatmaps.

**Inputs:**
- Inductance matrix data file: `data/raw/inductance/TF_system_L_matrix.xlsx`

**Output location:**
- `outputs/figures/inductance/`

**Output filename pattern:**
- `Npw={npw}_Top={temp}K_heatmap.svg`
- Examples: `Npw=1_Top=4.2K_heatmap.svg`, `Npw=20_Top=10K_heatmap.svg`, `Npw=100_Top=20K_heatmap.svg`

**Scan parameters:**
- **Temperature**: from `Ip_list` in `config.py`, default `[4.2, 10.0, 20.0]` K
- **Parallel‑stack count (Npw)**: default `[1, 20, 100]` (can be customized via `NPW_LIST` in the script)

**Features:**
- Supports inductance‑matrix visualization for multiple temperatures (matrix itself may be temperature‑independent, but tape count per coil varies with Top, affecting scaling).
- Uses log scale for inductance values (lg(Inductance [H])) to improve readability.
- Optionally annotates heatmaps with raw inductance values (up to 4 significant digits, non‑scientific notation, avoiding label clutter).
- Uses YlGnBu color map.

**Stitching:**
After generation, the script automatically calls `stitch_inductance_heatmaps()` to assemble selected subplots into a single figure:

- **Layout**: 2 columns × 3 rows
- **Left column**: Npw = 1 at 4.2K, 10K, 20K (from top to bottom) – labels **(a), (b), (c)**
- **Right column**: Npw = 20 at 4.2K, 10K, 20K (from top to bottom) – labels **(d), (e), (f)**
- **Output**: `outputs/figures/inductance/inductance_matrix_grid_Npw1&20.svg`
- **Labels**: panel label at upper‑left corner of each subplot (a–f)

**Related scripts:**
- `analysis_calculate_inductance.py`: optional; generates per‑case inductance matrices for inspection / debugging
- `snitch_inductance_tf_system.py`: reads `TF_system_L_matrix.xlsx` and generates / stitches inductance heatmaps

Figure format is globally set to `svg` in `config.py`.


## Contour‑Level Algorithm Notes

This section documents the contour‑level generation used in heatmaps (e.g. ΔLCOE, parasitic power), ensuring levels are evenly distributed in visual space and numerically “nice” (integers, 1‑2‑5×10^k, etc.).

### Overview

Contour levels are computed in two stages using a **visually uniform + numerically aligned** strategy:

1. **Visual‑space uniformity**

   - Map data values into normalized (visual) space via the chosen `norm`
   - Select target positions uniformly in this space
   - Map back to data space to get initial raw contour levels
2. **Numeric alignment**

   - Snap to a “nice” candidate set
   - Ensure labels are easy to read (integers, 1‑2‑5×10^k style)
   - Preserve approximately uniform spacing in visual space

### Key Functions

#### 1. `auto_contour_levels()` – automatic contour levels

**Location:** `plot_library.py` (around line 596)

**Purpose:** Automatically generate 4–6 contour levels for a single subplot, evenly spaced in visual (norm) space.

**Arguments:**

- `Z_sub`: 2D numpy array of subplot data
- `norm`: Matplotlib normalization object that supports `.inverse` (e.g. `SymLogNorm`, `LogNorm`)
- `n_min`: minimum number of contour levels (default 4)
- `n_max`: maximum number of contour levels (default 6)
- `q`: quantile range to avoid extremes (default 0.03 → use 3–97% quantiles)
- `margin`: margin in visual space (default 0.03 → 3%)
- `include_zero`: if data crosses zero, replace the level closest to 0 with exactly 0.0 (default True)
- `verbose`: whether to print verbose debug info (default False)

**Steps:**

1. Flatten data, drop NaN/inf, get valid range (z_min, z_max).
2. Map valid values into norm (visual) space, use quantiles (3–97%) to avoid extreme compression.
3. Choose contour‑count between `n_min` and `n_max` based on span in visual space.
4. Take evenly spaced points in visual space (excluding endpoints, honoring margin).
5. Map back to data space to get raw contour levels.
6. Call `snap_levels_to_nice()` to align to nice values.

**Returns:** sorted array of contour levels (excluding zmin/zmax).

**Example:**

```python
contour_levels = auto_contour_levels(
    Z_sub=Z,  # 2D data array
    norm=norm,  # SymLogNorm or LogNorm object
    n_min=4,
    n_max=6,
    q=0.03,
    margin=0.03,
    include_zero=True,
    verbose=False
)
```

---

#### 2. `snap_levels_to_nice()` – snap contour levels to nice values

**Location:** `plot_library.py` (around line 1189)

**Purpose:** Select numerically clean values from a “nice candidate set” so that contour labels are easy to read.

**Arguments:**

- `levels`: raw levels from `auto_contour_levels`
- `z_min`: data minimum
- `z_max`: data maximum
- `n_min`: minimum number of levels (default 4)
- `prefer_integers`: prefer integers when possible (default True)
- `norm`: Matplotlib normalization (optional; used for visual‑space mapping)
- `verbose`: print debug info (default False)

**Candidate‑set construction:**

1. **Local grid**: pick a suitable step (20, 10, 5, 2, 1, 0.5, 0.2) based on data range and build a regular grid.
2. **Coarse 1‑2‑5**: build a `{1, 2, 5} × 10^k` grid across several decades.
3. **Extended candidates** (if needed):
   - add denser `{1, 1.5, 2, 3, 5, 7} × 10^k` values,
   - finally add original raw levels as a fallback.

**Selection algorithm:**

- Use a **backtracking search** over candidates to select `n` levels.
- Aim for uniform spacing in visual space (via `norm`).
- Enforce a minimum spacing `t_gap_min = t_span / (n * 4.0)` to avoid clutter.
- Use an **upper anchor** so the topmost level lands near a nice value close to `z_max`.

**Returns:** sorted array of “nice” contour levels.

**Example:**

```python
nice_levels = snap_levels_to_nice(
    levels=raw_levels,  # raw levels from auto_contour_levels
    z_min=z_min,
    z_max=z_max,
    n_min=4,
    prefer_integers=True,
    norm=norm,
    verbose=False
)
```

---

#### 3. `filter_contour_levels_to_target_count()` – trim levels to a target count

**Location:** `plot_library.py` (around line 1407)

**Purpose:** Adjust the number of contour levels to 4–5, useful for multi‑subplot layouts where a consistent number of levels is desired.

**Arguments:**

- `levels`: sorted array of current levels
- `z_min`: data minimum
- `z_max`: data maximum
- `master_levels`: optional set of “global” levels to pull from
- `target_min_count`: target minimum (default 4)
- `target_max_count`: target maximum (default 5)
- `verbose`: print debug info

**Strategy:**

1. **Too many levels:**

   - Keep min and max.
   - Evenly subsample remaining levels from the interior.
   - Example: reduce from 10 to 5; keep min & max, select 3 from the middle 8.
2. **Too few levels:**

   - Try to add levels from `master_levels`.
   - Prefer values near the middle of the range.

**Returns:** filtered levels array.

**Example:**

```python
filtered_levels = filter_contour_levels_to_target_count(
    levels=contour_levels,
    z_min=z_min,
    z_max=z_max,
    master_levels=global_levels,  # optional
    target_min_count=4,
    target_max_count=5,
    verbose=False
)
```

---

#### 4. `determine_contour_color_by_background()` – color for a single contour line

**Location:** `plot_library.py` (around line 1546)

**Purpose:** Dynamically choose black or white for a single contour line based on background luminance along the path.

**Arguments:**

- `contour_path`: array of contour points, shape `(N, 2)` with `(x, y)` coordinates
- `X`: data‑grid X coordinates, shape `(ny, nx)`
- `Y`: data‑grid Y coordinates, shape `(ny, nx)`
- `Z`: data‑grid values, shape `(ny, nx)` (for background colors)
- `cmap`: colormap name
- `norm`: Matplotlib normalization
- `sample_points`: number of sample points along the path (default 20)
- `luminance_threshold`: threshold (0–1); above → black, below → white (default 0.5)

**Steps:**

1. Uniformly sample points along the contour path (default 20).
2. For each sample:
   - Use bilinear interpolation to obtain Z at that point (with log‑axis handling as needed).
   - Normalize via `norm` into \([0,1]\).
   - Map through `cmap` to get RGBA.
   - Compute luminance \(L = 0.299R + 0.587G + 0.114B\).
3. Average luminance across samples.
4. If `avg_luminance >= luminance_threshold`, return `'black'`; otherwise `'white'`.

**Returns:** `'black'` or `'white'`.

**Example:**

```python
color = determine_contour_color_by_background(
    contour_path=path_points,  # numpy array with shape (N, 2)
    X=X, Y=Y, Z=Z,
    cmap='viridis',
    norm=norm,
    sample_points=20,
    luminance_threshold=0.5
)
```

---

#### 5. `get_contour_colors_by_background()` – colors for all contour levels

**Location:** `plot_library.py` (around line 1688)

**Purpose:** Assign black/white colors to every contour level in a `QuadContourSet` based on background brightness.

**Arguments:**

- `contour_set`: Matplotlib `QuadContourSet`
- `X`: data‑grid X coordinates
- `Y`: data‑grid Y coordinates
- `Z`: data values
- `cmap`: colormap name
- `norm`: normalization
- `sample_points`: samples per contour path (default 20)
- `luminance_threshold`: brightness threshold (default 0.5)

**Steps:**

1. Loop over all contour levels in `contour_set`.
2. For each:
   - Gather all path segments (`contour_set.allsegs[i]`).
   - Merge segments into a single path.
   - Call `determine_contour_color_by_background()` to get the color.
3. Return a mapping `{level: color}`.

**Returns:** dict mapping level → `'black'` or `'white'`.

**Example:**

```python
# (1) First draw temporary contours to obtain path information
temp_CS = ax.contour(X, Y, Z, levels=levels, colors="black", linewidths=0.01)
# (2) Determine colors for each contour line
level_colors = get_contour_colors_by_background(temp_CS, X, Y, Z, cmap, norm)
# (3) Clear the temporary contours
for collection in temp_CS.collections:
    collection.remove()

# (4) Draw contours one by one using the chosen colors
for level in levels:
    color = level_colors.get(level, 'black')
    CS_level = ax.contour(X, Y, Z, levels=[level], colors=color, linewidths=2)
    ax.clabel(CS_level, inline=True, fontsize=12, fmt="%.1f", colors=color)
```

---

#### 6. `format_contour_label()` – human‑friendly contour labels

**Location:** `plot_library.py` (around line 2208, inside `plot_delta_lcoe_heatmap_single`)

**Purpose:** Format contour labels with adaptive precision, deduplicate them, and strip trailing zeros.

**Strategy:**

1. **Precision based on value range:**

   - `max_val >= 10`: integers or 1 decimal place (e.g. `10`, `10.5`)
   - `max_val >= 1`: 1–2 decimals (e.g. `1.5`, `1.25`)
   - `max_val >= 0.1`: 2–3 decimals (e.g. `0.15`, `0.125`)
   - `max_val < 0.1`: 3–4 decimals (e.g. `0.015`, `0.0125`)
2. **Strip trailing zeros:**

   - `1.0` → `1`
   - `1.50` → `1.5`
   - `1.5` → `1.5`
3. **Deduplicate:**

   - If formatting produces duplicates, keep the value that is numerically “nicer” (closer to an integer).

**Example:**

```python
def format_contour_label(x, all_levels):
    """Format contour labels."""
    # ... choose precision based on value range ...
    formatted = f"{x:.1f}"  # or another precision
    return remove_trailing_zeros(formatted)  # strip trailing zeros

# Use in clabel
CS.clabel(inline=True, fmt=format_contour_label_final, ...)
```

---

### End‑to‑End Example

Below is the full contour‑level workflow inside `plot_delta_lcoe_heatmap_single()`:

```python
# 1. Automatically generate contour levels (if enabled)
if draw_contours and auto_generate_levels and contour_levels is None:
    contour_levels = auto_contour_levels(
        Z_sub=Z,
        norm=norm,
        n_min=4,
        n_max=6,
        q=0.03,
        margin=0.03,
        include_zero=True,
        verbose=False
    )

# 2. Optionally trim contour levels to the target count
if len(contour_levels) > 5:
    contour_levels = filter_contour_levels_to_target_count(
        levels=contour_levels,
        z_min=z_min,
        z_max=z_max,
        target_min_count=4,
        target_max_count=5
    )

# 3. Deduplicate and format labels
# ... format_contour_label logic ...

# 4. Determine contour colors based on background brightness
temp_CS = ax.contour(X, Y, Z, levels=filtered_levels, colors="black", linewidths=0.01)
level_colors = get_contour_colors_by_background(temp_CS, X, Y, Z, cmap, norm)
# Clear temporary contours
for collection in temp_CS.collections:
    collection.remove()

# 5. Draw contours one by one using the chosen colors
for level in filtered_levels:
    color = level_colors.get(level, 'black')
    CS_level = ax.contour(X, Y, Z, levels=[level], colors=color, linewidths=2)
    labels_level = ax.clabel(CS_level, inline=True, fontsize=12, 
                             fmt=format_contour_label_final, colors=color)
```

---

### Design Principles

1. **Visual‑space uniformity**: levels are spaced uniformly in normalized visual space so plots look balanced over different data ranges.
2. **Numeric alignment**: `snap_levels_to_nice()` aligns levels to clean values (integers, 1‑2‑5×10^k) for readability.
3. **Dynamic color**: contour colors adapt to background brightness (black vs white) for clarity.
4. **Smart label formatting**: precision is adapted to value range, trailing zeros are stripped, duplicates are avoided.
5. **Robustness**: quantile‑based ranges avoid extreme‑value compression; backtracking search ensures usable levels are always found.

---

### Related Files

- **Core implementation**: `plot_library.py`
- **Call examples**:
  - `plot_delta_lcoe_heatmap_single()` – ΔLCOE heatmaps
  - `plot_parasitic_heatmap_single()` – parasitic‑power heatmaps
- **Config parameters**: `LCOE_CONTOUR_METHOD`, `CONTOUR_MIN_SPACING_USD`, etc. in `config.py`

## Manifest

Each full scan writes `outputs/manifest.json` containing:

- Code version (e.g. git hash, if available)
- Config file hash
- Input data hashes
- Output file hashes

To refresh the manifest manually, re‑run `analysis_scan_full_grid.py`.

## Python Dependencies

Typical dependencies:

- Python 3.9+
- numpy, pandas, matplotlib, scipy
- openpyxl (Excel I/O)
- pyyaml (scan config parsing)

## Notes

- All paths are relative to the repo root.
- Tables go to `outputs/tables/`, figures go to `outputs/figures/`.
