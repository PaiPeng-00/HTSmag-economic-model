"""
Full‑grid scan script that generates the main Supplementary Data tables.

Scan dimensions (conceptual default):
- Top_K and coolant: [(4.2, 'He'), (10, 'He'), (20, 'He'), (20, 'H2')]
- Npw: 1..200
- rho_turn_uOhm_cm2: logspace(10, 10000, 31)  # units: μΩ·cm²
- R_joint_nOhm: [10, 100]
- scenario: ['S1', 'S2', 'S3']

Outputs:
- scan_full_grid_tidy.csv  (long table; one row per parameter combination)
- scan_full_grid.xlsx      (optional; same content, convenient for reviewers)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from datetime import datetime
from functools import lru_cache

# Import project modules
sys.path.insert(0, str(Path(__file__).parent))
import config as cfg
from src.tfmag.paths import ensure_base_dirs
from model_cryogenic   import calculate_base_heat_loads
from utils import calculate_radial_resistance
from model_economic import compute_case, define_parameters
from src.tfmag.manifest import write_manifest

# Import charging‑time calculation module
import importlib.util
spec = importlib.util.spec_from_file_location("charge_time_module", 
    Path(__file__).parent / "2.5_charge_time999_TF_system_all_Npw=1-200.py")
charge_time_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(charge_time_module)
_L_MATRIX_CACHE = None

def _load_tf_system_l_matrix() -> np.ndarray:
    global _L_MATRIX_CACHE
    if _L_MATRIX_CACHE is not None:
        return _L_MATRIX_CACHE

    inductance_dir = Path(__file__).parent / cfg.INDUCTANCE_OUTPUT_DIR
    matrix_path = inductance_dir / cfg.TF_SYSTEM_MATRIX
    matrix = None
    if matrix_path.exists():
        try:
            matrix = pd.read_excel(matrix_path, header=None).values
        except Exception:
            matrix = None

    if matrix is None or matrix.shape != (cfg.Ntf, cfg.Ntf):
        matrix = (
            np.diag(np.full(cfg.Ntf, 5e-3))
            + np.diag(np.full(cfg.Ntf - 1, 0.5e-3), k=1)
            + np.diag(np.full(cfg.Ntf - 1, 0.5e-3), k=-1)
        )

    _L_MATRIX_CACHE = matrix
    return _L_MATRIX_CACHE

def _fallback_calculate_charge_time_999(
    Npw: int,
    rho_turn_uOhm_cm2: float,
    temperature: float
) -> float:
    try:
        l_matrix = _load_tf_system_l_matrix()
        Ip_case = cfg.Ip_list.get(temperature, cfg.I_TARGET)
        Ntape_coil_case = cfg.Nt_list.get(temperature, cfg.N_TOTAL_TAPE)
        l_matrix_current = l_matrix * (Ntape_coil_case * cfg.NP / Npw) ** 2
        rho_turn_ohm_m2 = rho_turn_uOhm_cm2 * 1e-10
        r_radial = calculate_radial_resistance(
            Npw=Npw,
            Ntape_coil=Ntape_coil_case,
            rho_turn=rho_turn_ohm_m2
        ) * cfg.NP
        r_values = [r_radial] * cfg.Ntf
        t, i_l, _, _, _ = charge_time_module.simulate_charging_system(
            l_matrix_current,
            r_values,
            npw=Npw,
            I_target_local=Ip_case,
            Ntape_coil_local=Ntape_coil_case
        )
        time_999_h = charge_time_module.calculate_time_to_999(
            t,
            i_l,
            npw=Npw,
            I_target_per_conductor=Ip_case
        )
        return float(time_999_h) if time_999_h is not None else cfg.charge_hours_max
    except Exception:
        return cfg.charge_hours_max

if hasattr(charge_time_module, "calculate_charge_time_999"):
    calculate_charge_time_999 = charge_time_module.calculate_charge_time_999
else:
    calculate_charge_time_999 = _fallback_calculate_charge_time_999

# =============================================================================
# Config params (aligned with economic analysis, section 8.0)
# =============================================================================
def _load_scan_config(config_path: Path) -> dict:
    if not config_path.exists():
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("Missing PyYAML; install with `pip install pyyaml` to read configs.") from exc
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


SCAN_CONFIG_PATH = cfg.CONFIGS_DIR / "scan_full_grid.yaml"
SCAN_CFG = _load_scan_config(SCAN_CONFIG_PATH)

# Use configs/scan_full_grid.yaml to override config.py
TEMP_COOLANT_PAIRS = SCAN_CFG.get("temp_coolant_pairs", cfg.SCAN_TEMP_COOLANT_PAIRS)
NPW_RANGE = np.array(SCAN_CFG.get("npw_values", cfg.NPW_SCAN_VALUES))
# R_JOINT_NOHM: joint‑resistance scan array in nOhm
R_JOINT_NOHM = np.array(
    SCAN_CFG.get(
        "r_joint_nohm_values",
        list(cfg.R_JOINT_SCAN_VALUES),  # default: log‑spaced nOhm array from config
    ),
    dtype=float,
)

rho_cfg = SCAN_CFG.get("rho_turn_uohm_cm2_values")
if isinstance(rho_cfg, dict):
    spec = rho_cfg.get("logspace")
    if spec:
        start = float(spec.get("start", 10))
        stop = float(spec.get("stop", 10000))
        num = int(spec.get("num", 31))
        RHO_TURN_UOHM_CM2 = np.logspace(np.log10(start), np.log10(stop), num)
        include_vals = [float(v) for v in spec.get("include", [])]
        if include_vals:
            RHO_TURN_UOHM_CM2 = np.sort(np.concatenate([RHO_TURN_UOHM_CM2, include_vals]))
    else:
        RHO_TURN_UOHM_CM2 = np.array([], dtype=float)
elif rho_cfg is None:
    RHO_TURN_UOHM_CM2 = np.logspace(np.log10(10), np.log10(10000), 31)
else:
    RHO_TURN_UOHM_CM2 = np.array(rho_cfg, dtype=float)
if 5000.0 not in RHO_TURN_UOHM_CM2:
    RHO_TURN_UOHM_CM2 = np.sort(np.concatenate([RHO_TURN_UOHM_CM2, [5000.0]]))

SCENARIOS = SCAN_CFG.get(
    "scenarios",
    list(cfg.SCENARIO_DEFINITIONS.keys()) if hasattr(cfg, 'SCENARIO_DEFINITIONS') else ['S1', 'S2', 'S3', 'S4', 'S5', 'S6'],
)

# Paths
PATHS = ensure_base_dirs()
OUTPUT_CSV = PATHS.outputs_tables / "scan_full_grid_tidy.csv"
OUTPUT_XLSX = PATHS.outputs_tables / "scan_full_grid.xlsx"

# Flush interval
FLUSH_INTERVAL = 500  # Flush every 500 rows

# =============================================================================
# Helper functions: input data
# =============================================================================

def load_charging_time999_df(Top: float, Ip: float, Ntape_coil: int) -> pd.DataFrame:
    """
    Load the pre‑computed 99.9% charging‑time grid for a given temperature
    and current/tape configuration.

    Returns:
        DataFrame indexed by rho_turn_uOhm_cm2 with columns Npw and values
        equal to charging time (hours).
    """
    base_path = Path(__file__).parent
    base_dir = base_path / cfg.TF_SYSTEM_CHARGING_SIM_OUTPUT_DIR / f"Temp_{Top}K_Ip_{Ip}A_Ntape_coil_{Ntape_coil}_charge999"
    file_path = base_dir / cfg.TF_SYSTEM_CHARGING_SIM_OUTPUT_FILE_Npw1_200
    
    if not file_path.exists():
        return pd.DataFrame()
    
    try:
        df = pd.read_excel(file_path, index_col=0)
        # Ensure index and columns are numeric
        df.index = pd.to_numeric(df.index, errors='coerce')
        df.columns = pd.to_numeric(df.columns, errors='coerce')
        return df
    except Exception as e:
        print(f"  [Warning] Failed to read charging time file: {e}")
        return pd.DataFrame()


def load_electromagnetic_losses_for_temp(Top: float) -> Dict[Tuple[int, float], Dict[str, pd.Series]]:
    """
    Load all electromagnetic loss data (magnetisation and radial losses) for a
    given operating temperature Top.

    Data are stored in a single Excel file with separate sheets per temperature
    (e.g. "4.2K", "10K", "20K"). Each sheet contains multiple (Npw, rho_turn)
    cases.

    Args:
        Top: operating temperature (K); used to pick the Excel sheet name.

    Returns:
        dict keyed by (Npw, rho_turn_uOhm_cm2) with values
        {'mag_loss': Series, 'radial_loss': Series}, where the index is time
        in hours and the value is power in W.
    """
    base_path = Path(__file__).parent
    # New layout: use HEAT_DATA_DIR directly instead of per‑temperature subdirs
    heat_data_dir = base_path / cfg.HEAT_DATA_DIR
    
    result = {}
    
    # Sheet name determined by temperature (e.g. "4.2K", "10K", "20K")
    sheet_name = f"{Top}K"
    
    # Read magnetisation loss (from a single multi‑sheet file)
    mag_loss_file = heat_data_dir / "mag loss.xlsx"
    if mag_loss_file.exists():
        try:
            # Try to read the requested sheet; if missing, fall back to the first sheet
            try:
                df_mag = pd.read_excel(mag_loss_file, sheet_name=sheet_name, header=0, skiprows=lambda x: x < 5)
            except (ValueError, KeyError):
                # If sheet is missing, read the first sheet as a fallback
                df_mag = pd.read_excel(mag_loss_file, sheet_name=0, header=0, skiprows=lambda x: x < 5)
                print(f"  [Warning] Sheet '{sheet_name}' not found in {mag_loss_file}, using first sheet")
            
            if len(df_mag.columns) >= 4:
                rhot_col = df_mag.columns[0]
                npw_col = df_mag.columns[1]
                time_col = df_mag.columns[2]
                value_col = df_mag.columns[3]
                
                # Group by (Npw, rho_turn_uOhm_cm2)
                for (npw, rhot), group in df_mag.groupby([npw_col, rhot_col]):
                    group = group.copy()
                    group['Time (h)'] = group[time_col] / 3600
                    series = group.set_index('Time (h)')[value_col]
                    key = (int(npw), float(rhot))
                    if key not in result:
                        result[key] = {}
                    result[key]['mag_loss'] = series
        except Exception as e:
            print(f"  [Warning] Failed to read magnetic loss from sheet '{sheet_name}': {e}")
    
    # Read radial loss (from the same style of multi‑sheet file)
    radial_loss_file = heat_data_dir / "radial loss.xlsx"
    if radial_loss_file.exists():
        try:
            # Try to read requested sheet; if missing, fall back to first sheet
            try:
                df_radial = pd.read_excel(radial_loss_file, sheet_name=sheet_name, header=0, skiprows=lambda x: x < 5)
            except (ValueError, KeyError):
                # If sheet is missing, use first sheet
                df_radial = pd.read_excel(radial_loss_file, sheet_name=0, header=0, skiprows=lambda x: x < 5)
                print(f"  [Warning] Sheet '{sheet_name}' not found in {radial_loss_file}, using first sheet")
            
            if len(df_radial.columns) >= 4:
                rhot_col = df_radial.columns[0]
                npw_col = df_radial.columns[1]
                time_col = df_radial.columns[2]
                value_col = df_radial.columns[3]
                
                # Group by (Npw, rho_turn_uOhm_cm2)
                for (npw, rhot), group in df_radial.groupby([npw_col, rhot_col]):
                    group = group.copy()
                    group['Time (h)'] = group[time_col] / 3600
                    series = group.set_index('Time (h)')[value_col]
                    key = (int(npw), float(rhot))
                    if key not in result:
                        result[key] = {}
                    result[key]['radial_loss'] = series
        except Exception as e:
            print(f"  [Warning] Failed to read radial loss from sheet '{sheet_name}': {e}")
    
    return result


def interpolate_loss_at_time(loss_series: pd.Series, target_time_h: float) -> float:
    """
    Interpolate loss power at a given time.

    Args:
        loss_series: time‑series (index in hours, values in W).
        target_time_h: time at which to interpolate (hours).

    Returns:
        Power in W at the requested time; last available value if the target
        time is beyond the data range; NaN if the series is empty.
    """
    if loss_series.empty:
        return np.nan
    
    if target_time_h <= loss_series.index.max():
        return np.interp(target_time_h, loss_series.index, loss_series.values)
    else:
        # If target time exceeds available data, return last value
        return loss_series.iloc[-1] if len(loss_series) > 0 else np.nan


def get_charging_time_999(
    Top: float,
    Ip: float,
    Ntape_coil: int,
    Npw: int,
    rho_turn_uOhm_cm2: float,
    time999_df: Optional[pd.DataFrame] = None
) -> float:
    """
    Get the 99.9% charging time.

    Preference order:
      1) use pre‑computed Excel grid (time999_df) if available,
      2) otherwise call calculate_charge_time_999 to compute on the fly.

    Args:
        Top: operating temperature (K)
        Ip: current (A)
        Ntape_coil: number of tapes per coil
        Npw: parallel strand count
        rho_turn_uOhm_cm2: turn‑to‑turn resistivity (μΩ·cm²)
        time999_df: optional pre‑loaded DataFrame from Excel

    Returns:
        Charging time in hours.
    """
    # Try DataFrame first
    if time999_df is not None and not time999_df.empty:
        try:
            # Find the closest available rho_turn and Npw in the grid
            available_rhot = time999_df.index.values
            available_npw = time999_df.columns.values
            
            if len(available_rhot) > 0 and len(available_npw) > 0:
                closest_rhot = available_rhot[np.argmin(np.abs(available_rhot - rho_turn_uOhm_cm2))]
                closest_npw = available_npw[np.argmin(np.abs(available_npw - Npw))]
                
                value = time999_df.at[closest_rhot, closest_npw]
                if pd.notna(value):
                    return float(value)
        except Exception:
            pass
    
    # If DataFrame lookup failed, call numerical function
    try:
        charging_time_999_h = _cached_charge_time_999(
            Top=Top,
            Npw=Npw,
            rho_turn_uOhm_cm2=rho_turn_uOhm_cm2,
        )
        if charging_time_999_h is not None:
            return charging_time_999_h
    except Exception as e:
        print(f"    [Warning] Charging time calculation failed: {e}")
    
    # If everything fails, return a conservative upper bound
    return cfg.charge_hours_max


def load_existing_results(csv_path: Path) -> pd.DataFrame:
    """
    Load an existing results CSV for resume capability.

    Returns:
        DataFrame, or an empty DataFrame if the file does not exist or fails
        to load.
    """
    if not csv_path.exists():
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        print(f"[Warning] Failed to load existing results: {e}")
        return pd.DataFrame()


def is_computed(
    existing_df: pd.DataFrame,
    Top: float,
    coolant: str,
    Npw: int,
    rho_turn_uOhm_cm2: float,
    R_joint_nOhm: int,
    scenario: str
) -> bool:
    """
    Check whether a given parameter combination has already been computed.
    """
    if existing_df.empty:
        return False
    
    mask = (
        (existing_df['Top_K'] == Top) &
        (existing_df['coolant'] == coolant) &
        (existing_df['Npw'] == Npw) &
        (np.isclose(existing_df['rho_turn_uOhm_cm2'], rho_turn_uOhm_cm2)) &
        (existing_df['R_joint_nOhm'] == R_joint_nOhm) &
        (existing_df['scenario'] == scenario)
    )
    return mask.any()


def _make_key(
    Top: float,
    coolant: str,
    Npw: int,
    rho_turn_uOhm_cm2: float,
    R_joint_nOhm: int,
    scenario: str,
) -> Tuple[float, str, int, float, int, str]:
    return (
        float(Top),
        str(coolant),
        int(Npw),
        float(rho_turn_uOhm_cm2),
        int(R_joint_nOhm),
        str(scenario),
    )


def load_existing_results_set(csv_path: Path) -> Tuple[pd.DataFrame, set]:
    """
    Load an existing results CSV and build a set of key tuples for fast
    membership tests (used for resume).
    """
    if not csv_path.exists():
        return pd.DataFrame(), set()
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[Warning] Failed to load existing results: {e}")
        return pd.DataFrame(), set()

    if df.empty:
        return df, set()

    required_cols = {"Top_K", "coolant", "Npw", "rho_turn_uOhm_cm2", "R_joint_nOhm", "scenario"}
    if not required_cols.issubset(df.columns):
        print("[Warning] Existing results missing required columns; resume disabled.")
        return df, set()

    keys = set(
        _make_key(
            row.Top_K,
            row.coolant,
            row.Npw,
            row.rho_turn_uOhm_cm2,
            row.R_joint_nOhm,
            row.scenario,
        )
        for row in df.itertuples(index=False)
    )
    return df, keys


def is_computed_key(existing_keys: set, key: Tuple[float, str, int, float, int, str]) -> bool:
    return key in existing_keys


@lru_cache(maxsize=None)
def _cached_charge_time_999(Top: float, Npw: int, rho_turn_uOhm_cm2: float) -> float:
    return float(
        calculate_charge_time_999(
            Npw=int(Npw),
            rho_turn_uOhm_cm2=float(rho_turn_uOhm_cm2),
            temperature=float(Top),
        )
    )


@lru_cache(maxsize=None)
def _cached_radial_resistance(Npw: int, Ntape_coil: int, rho_turn_ohm_m2: float) -> float:
    return float(
        calculate_radial_resistance(
            Npw=int(Npw),
            Ntape_coil=int(Ntape_coil),
            rho_turn=float(rho_turn_ohm_m2),
        )
    )




@lru_cache(maxsize=None)
def _cached_heat_loads(Top: float, Ip: float, Npw: int, R_joint_ohm: float) -> dict:
    return calculate_base_heat_loads(
        Ip=float(Ip),
        Npw=int(Npw),
        R_p2p_joint=float(R_joint_ohm),
        Top=float(Top),
    )


# =============================================================================
# Main driver
# =============================================================================
def main():
    """Run the full parameter‑grid scan."""
    
    # 1) Scan dimensions are already defined at module level
    
    # 2) Initialise output paths
    OUTPUT_CSV = PATHS.outputs_tables / "scan_full_grid_tidy.csv"
    OUTPUT_XLSX = PATHS.outputs_tables / "scan_full_grid.xlsx"
    
    # 3) Load existing results for resume
    existing_df, existing_keys = load_existing_results_set(OUTPUT_CSV)
    if not existing_df.empty:
        print(f"[Info] Found existing results: {len(existing_df)} rows. Will skip computed combinations.")
    written_count = len(existing_df)
    
    # 4) Initialise economic parameters once
    econ_params = define_parameters()
    
    # 5) Pre‑load per‑temperature charging‑time DataFrames and EM‑loss data
    #    (each temperature is read only once)
    time999_cache = {}   # key: (Top, Ip, Ntape_coil) -> DataFrame
    em_losses_cache = {} # key: Top -> Dict[(Npw, rho_turn_uOhm_cm2) -> Series]
    
    print("\n[Info] Pre-loading charging time and EM loss data for each temperature...")
    for Top, coolant in TEMP_COOLANT_PAIRS:
        # Current and tape count at this temperature
        Ip = cfg.Ip_list.get(Top, 300.0)
        Ntape_coil = cfg.Nt_list.get(Top, 1400)
        
        # Load charging‑time DataFrame
        time999_df = load_charging_time999_df(Top, Ip, Ntape_coil)
        time999_cache[(Top, Ip, Ntape_coil)] = time999_df
        if not time999_df.empty:
            print(
                f"  [OK] Loaded charging‑time data for {Top}K: "
                f"{len(time999_df)} rho_turn x {len(time999_df.columns)} Npw"
            )
        
        # Load EM‑loss data (each temperature read only once)
        em_losses_cache[Top] = load_electromagnetic_losses_for_temp(Top)
        if em_losses_cache[Top]:
            print(
                f"  [OK] Loaded EM‑loss data for {Top}K: "
                f"{len(em_losses_cache[Top])} (Npw, rho_turn) combinations"
            )
    
    # 6) Total number of combinations (for progress reporting only)
    total_combinations = len(TEMP_COOLANT_PAIRS) * len(NPW_RANGE) * len(RHO_TURN_UOHM_CM2) * len(R_JOINT_NOHM) * len(SCENARIOS)
    print(f"\n[Info] Total combinations to compute: {total_combinations:,}")
    
    # 7) Start scanning
    results = []  # in‑memory buffer of dict rows
    computed_count = 0
    skipped_count = 0
    
    print("\n[Info] Starting full grid scan...")
    
    for Top, coolant in TEMP_COOLANT_PAIRS:
        # Current and tape count at this temperature
        Ip = cfg.Ip_list.get(Top, 300.0)
        Ntape_coil = cfg.Nt_list.get(Top, 1400)
        
        # Retrieve pre‑loaded data
        time999_df = time999_cache.get((Top, Ip, Ntape_coil), pd.DataFrame())
        em_losses_dict = em_losses_cache.get(Top, {})
        
        for scenario in SCENARIOS:
            scenario_year = econ_params['tech_scenario_to_years'][scenario]
            
            for Npw in NPW_RANGE:
                for rho_turn_uOhm_cm2 in RHO_TURN_UOHM_CM2:
                    for R_joint_nOhm in R_JOINT_NOHM:
                        # Skip if this combination is already present (resume)
                        key = _make_key(Top, coolant, Npw, rho_turn_uOhm_cm2, R_joint_nOhm, scenario)
                        if is_computed_key(existing_keys, key):
                            skipped_count += 1
                            if skipped_count % 1000 == 0:
                                print(f"  [Info] Skipped {skipped_count:,} combinations...")
                            continue
                        
                        computed_count += 1
                        if computed_count % 100 == 0:
                            print(f"  [Progress] Computed {computed_count:,}/{total_combinations:,} ({100*computed_count/total_combinations:.1f}%)")

                        # Round R_joint_nOhm to 3 decimal places for storage
                        R_joint_nOhm = round(R_joint_nOhm, 3)
                        # Base result dictionary
                        result = {
                            "Top_K": Top,
                            "coolant": coolant,
                            "scenario": scenario,
                            "Npw": Npw,
                            "rho_turn_uOhm_cm2": rho_turn_uOhm_cm2,
                            "R_joint_nOhm": R_joint_nOhm,
                            "status": "success",
                            "invalid_reason": None,
                        }
                        
                        # ========== 0. Charging‑time metrics ==========
                        try:
                            charging_time_999_h = get_charging_time_999(
                                Top=Top,
                                Ip=Ip,
                                Ntape_coil=Ntape_coil,
                                Npw=Npw,
                                rho_turn_uOhm_cm2=rho_turn_uOhm_cm2,
                                time999_df=time999_df
                            )
                            result["Charging_time_999_h"] = charging_time_999_h
                        except Exception as e:
                            result["Charging_time_999_h"] = cfg.charge_hours_max
                            result["status"] = "warning"
                            result["invalid_reason"] = f"Charging time calculation failed: {e}"
                        
                        # ========== 0.5. Electromagnetic losses at charge ==========
                        # Look up EM‑loss data from cache; keys are (Npw, rho_turn_uOhm_cm2).
                        # If there is no exact match, use the closest combination.
                        mag_loss_at_charge = np.nan
                        radial_loss_at_charge = np.nan
                        mag_loss_energy_MWh = np.nan  # optional: energy integral
                        radial_loss_energy_MWh = np.nan
                        
                        if em_losses_dict:
                            # Find closest (Npw, rho_turn_uOhm_cm2) combination
                            closest_key = None
                            min_distance = float('inf')
                            for key in em_losses_dict.keys():
                                key_npw, key_rhot = key
                                # Normalised distance: absolute difference in Npw, relative difference in rho_turn
                                distance = abs(key_npw - Npw) / max(Npw, 1) + abs(key_rhot - rho_turn_uOhm_cm2) / max(rho_turn_uOhm_cm2, 1e-6)
                                if distance < min_distance:
                                    min_distance = distance
                                    closest_key = key
                            
                            # Always use the closest value, even if the distance is large (no tolerance cut‑off)
                            if closest_key:
                                em_data = em_losses_dict[closest_key]
                                
                                # Interpolate magnetisation loss at the fixed ramp‑end time CHARGE_HOURS
                                if 'mag_loss' in em_data:
                                    mag_loss_series = em_data['mag_loss']
                                    # Instantaneous power at ramp end (e.g. 96 h)
                                    mag_loss_at_charge = interpolate_loss_at_time(mag_loss_series, cfg.CHARGE_HOURS)
                                    # Energy integral from 0 to t999
                                    if not mag_loss_series.empty:
                                        mask = mag_loss_series.index <= charging_time_999_h
                                        if mask.any():
                                            times = mag_loss_series.index[mask].values
                                            powers = mag_loss_series.values[mask]
                                            # Trapezoidal integration: energy = ∫P dt (W·h -> MWh)
                                            if len(times) > 1:
                                                energy_Wh = np.trapz(powers, times)
                                                mag_loss_energy_MWh = energy_Wh / 1e6
                                
                                # Interpolate radial loss at CHARGE_HOURS
                                if 'radial_loss' in em_data:
                                    radial_loss_series = em_data['radial_loss']
                                    # Instantaneous power at ramp end
                                    radial_loss_at_charge = interpolate_loss_at_time(radial_loss_series, cfg.CHARGE_HOURS)
                                    # Energy integral from 0 to t999
                                    if not radial_loss_series.empty:
                                        mask = radial_loss_series.index <= charging_time_999_h
                                        if mask.any():
                                            times = radial_loss_series.index[mask].values
                                            powers = radial_loss_series.values[mask]
                                            if len(times) > 1:
                                                energy_Wh = np.trapz(powers, times)
                                                radial_loss_energy_MWh = energy_Wh / 1e6
                        
                        result.update({
                            "Mag_loss_at_charge_W": mag_loss_at_charge,
                            "Radial_loss_at_charge_W": radial_loss_at_charge,
                            "Mag_loss_energy_MWh": mag_loss_energy_MWh,
                            "Radial_loss_energy_MWh": radial_loss_energy_MWh,
                        })
                        
                        # ========== 1. Thermal metrics ==========
                        try:
                            heat_loads = _cached_heat_loads(
                                Top=Top,
                                Ip=Ip,
                                Npw=Npw,
                                R_joint_ohm=R_joint_nOhm*1e-9,
                            )
                            
                            heat_key = f"heat @ {Top}K"
                            cop_key = f"COP @ {Top}K"
                            
                            total_heat_tc = heat_loads[heat_key]
                            total_heat_77 = heat_loads["heat @ 77K"]
                            p_fusion = heat_loads["P_fusion_electric_W"]
                            
                            result.update({
                                "P_cryo_electric_W": heat_loads["P_cryo_electric_W"],
                                "P_fusion_electric_W": p_fusion,
                                "COP_Tc": heat_loads[cop_key],
                                "COP_77": heat_loads["COP @ 77K"],
                                "r_cryo_re_producton": heat_loads["para_cost(%)"],
                                # Top‑temperature heat‑source breakdown (Q_ prefix)
                                "Q_coil_internal_joint_W": heat_loads["coil_internal_joint"],
                                "Q_pancake_joint_W": heat_loads["pancake_joint"],
                                "Q_nuclear_W": heat_loads["nuclear"],
                                "Q_radiation_W": heat_loads["radiation"],
                                "Q_current_leads_HTS_W": heat_loads["current_leads_HTS"],
                                "Q_pipes_coolant_W": heat_loads["pipes_coolant"],
                                "Q_pipes_aux_W": heat_loads["pipes_aux"],
                                "Q_quench_W": heat_loads["quench"],
                                "Q_misc_W": heat_loads["misc"],
                                "Q_total_Tc_W": total_heat_tc,
                                # 77 K level heat‑source breakdown
                                "Q_current_leads_Cu_conduction_77K_W": heat_loads["current_leads_Cu_conduction"],
                                "Q_current_leads_Cu_joule_77K_W": heat_loads["current_leads_Cu_joule"],
                                "Q_total_77K_W": total_heat_77,
                            })
                        except Exception as e:
                            result["status"] = "error"
                            result["invalid_reason"] = f"Heat load calculation failed: {e}"
                            result.update({
                                "P_cryo_electric_W": np.nan,
                                "P_fusion_electric_W": np.nan,
                                "COP_Tc": np.nan,
                                "COP_77": np.nan,
                                "r_cryo_re_producton": np.nan,
                                "Q_coil_internal_joint_W": np.nan,
                                "Q_pancake_joint_W": np.nan,
                                "Q_nuclear_W": np.nan,
                                "Q_radiation_W": np.nan,
                                "Q_current_leads_HTS_W": np.nan,
                                "Q_pipes_coolant_W": np.nan,
                                "Q_pipes_aux_W": np.nan,
                                "Q_quench_W": np.nan,
                                "Q_misc_W": np.nan,
                                "Q_total_Tc_W": np.nan,
                                "Q_current_leads_Cu_conduction_77K_W": np.nan,
                                "Q_current_leads_Cu_joule_77K_W": np.nan,
                                "Q_total_77K_W": np.nan,
                            })
                        
                        # ========== 2. Electromagnetic metrics ==========
                        try:
                            # Radial resistance per pancake
                            R_radial_per_pancake = _cached_radial_resistance(
                                Npw=Npw,
                                Ntape_coil=Ntape_coil,
                                rho_turn_ohm_m2=rho_turn_uOhm_cm2*1e-10,
                            )
                            # Radial resistance per TF magnet = sum over pancakes
                            R_radial_per_TF = R_radial_per_pancake * cfg.NP
                            # System‑level radial resistance = parallel of all TF magnets
                            R_radial_system = R_radial_per_TF / cfg.Ntf
                            
                            result.update({
                                "R_radial_per_pancake_Ohm": R_radial_per_pancake,
                                "R_radial_per_TF_Ohm": R_radial_per_TF,
                                "R_radial_system_Ohm": R_radial_system,
                            })
                        except Exception as e:
                            if result["status"] == "success":
                                result["status"] = "warning"
                            if result["invalid_reason"]:
                                result["invalid_reason"] += f"; EM calculation failed: {e}"
                            else:
                                result["invalid_reason"] = f"EM calculation failed: {e}"
                            result.update({
                                "R_radial_per_pancake_Ohm": np.nan,
                                "R_radial_per_TF_Ohm": np.nan,
                                "R_radial_system_Ohm": np.nan,
                            })
                        
                        # ========== 3. Economic metrics ==========
                        original_charge_hours_max = cfg.charge_hours_max
                        try:
                            # Temporarily set charge_hours_max to the actual charging time
                            cfg.charge_hours_max = charging_time_999_h
                            
                            economic_result, _ = compute_case(
                                tech_scenario=scenario,
                                year=scenario_year,
                                temperature_K=Top,
                                coolant=coolant,
                                Npw=Npw,
                                R_p2p_joint=R_joint_nOhm*1e-9,
                                p=econ_params
                            )
                            
                            if economic_result:
                                # Availability factor AF = annual production hours / total yearly hours
                                annual_hours_prod = economic_result.get("annual_hours_prod", 0)
                                AF = annual_hours_prod / cfg.HOURS_PER_YEAR if cfg.HOURS_PER_YEAR > 0 else np.nan
                                
                                # Parasitic power fraction (%)
                                r_parasitic_pct = economic_result.get("r_parasitic_pct", np.nan)
                                
                                result.update({
                                    "LCOE_magnet_only_USD_per_MWh": economic_result.get("lcoe_magnet_only_$/MWh", np.nan),
                                    "LCOE_magnet_only_CNY_per_kWh": economic_result.get("lcoe_magnet_only_$/MWh", np.nan) * cfg.USD_TO_CNY_EXCHANGE_RATE / 1000 if not np.isnan(economic_result.get("lcoe_magnet_only_$/MWh", np.nan)) else np.nan,
                                    "CAPEX_mag_direct_USD": economic_result.get("capex_mag_direct_$", np.nan),
                                    "CAPEX_mag_installed_USD": economic_result.get("capex_mag_installed_$", np.nan),
                                    "Annualized_CAPEX_mag_USD_per_year": economic_result.get("annualized_capex_mag_$/year", np.nan),
                                    "Annual_OPEX_mag_USD_per_year": economic_result.get("annual_opex_mag_$/year", np.nan),
                                    "Tape_cost_USD": economic_result.get("tape_cost_$", np.nan),
                                    "Coolant_fill_cost_USD": economic_result.get("coolant_fill_cost_$", np.nan),
                                    "Power_supply_cost_USD": economic_result.get("power_supply_cost_$", np.nan),
                                    "Net_annual_profit_USD": economic_result.get("net_annual_profit_$", np.nan),
                                    "Simple_payback_years": economic_result.get("simple_payback_years", np.nan),
                                    "Lifetime_revenue_USD": economic_result.get("lifetime_revenue_$", np.nan),
                                    "AF": AF,
                                    "r_cryo_re": r_parasitic_pct,
                                    "P_cryo_prod_W": economic_result.get("cryo_power_prod_W", np.nan),
                                    "P_cryo_dwell_W": economic_result.get("cryo_power_dwell_W", np.nan),
                                    "P_cryo_static_W": economic_result.get("cryo_power_static_W", np.nan),
                                    "P_cryo_coolwarm_W": economic_result.get("cryo_power_coolwarm_W", np.nan),
                                    "P_cryo_excdec_W": economic_result.get("cryo_power_excdec_W", np.nan),
                                    "E_cryo_year_MWh": economic_result.get("cryo_energy_annual_MWh", np.nan),
                                    "E_gross_year_MWh": economic_result.get("gross_energy_annual_MWh", np.nan),
                                    "E_net_year_MWh": economic_result.get("net_energy_annual_MWh", np.nan),
                                })
                            else:
                                if result["status"] == "success":
                                    result["status"] = "warning"
                                if result["invalid_reason"]:
                                    result["invalid_reason"] += "; Economic calculation result is empty"
                                else:
                                    result["invalid_reason"] = "Economic calculation result is empty"
                                result.update({
                                    "LCOE_magnet_only_USD_per_MWh": np.nan,
                                    "LCOE_magnet_only_CNY_per_kWh": np.nan,
                                    "CAPEX_mag_direct_USD": np.nan,
                                    "CAPEX_mag_installed_USD": np.nan,
                                    "Annualized_CAPEX_mag_USD_per_year": np.nan,
                                    "Annual_OPEX_mag_USD_per_year": np.nan,
                                    "Tape_cost_USD": np.nan,
                                    "Coolant_fill_cost_USD": np.nan,
                                    "Power_supply_cost_USD": np.nan,
                                    "Net_annual_profit_USD": np.nan,
                                    "Simple_payback_years": np.nan,
                                    "Lifetime_revenue_USD": np.nan,
                                    "AF": np.nan,
                                    "r_cryo_re": np.nan,
                                })
                        except Exception as e:
                            if result["status"] == "success":
                                result["status"] = "error"
                            if result["invalid_reason"]:
                                result["invalid_reason"] += f"; Economic calculation failed: {e}"
                            else:
                                result["invalid_reason"] = f"Economic calculation failed: {e}"
                            result.update({
                                "LCOE_magnet_only_USD_per_MWh": np.nan,
                                "LCOE_magnet_only_CNY_per_kWh": np.nan,
                                "CAPEX_mag_direct_USD": np.nan,
                                "CAPEX_mag_installed_USD": np.nan,
                                "Annualized_CAPEX_mag_USD_per_year": np.nan,
                                "Annual_OPEX_mag_USD_per_year": np.nan,
                                "Tape_cost_USD": np.nan,
                                "Coolant_fill_cost_USD": np.nan,
                                "Power_supply_cost_USD": np.nan,
                                "Net_annual_profit_USD": np.nan,
                                "Simple_payback_years": np.nan,
                                "Lifetime_revenue_USD": np.nan,
                                    "AF": np.nan,
                                    "r_cryo_re": np.nan,
                            })
                        finally:
                            # Restore original values
                            cfg.charge_hours_max = original_charge_hours_max
                        
                        # ========== 4. Feasibility checks & invalid_reason ==========
                        CHARGE_HOURS = cfg.charge_hours_max
                        
                        # Collect feasibility violations into invalid_reasons
                        invalid_reasons = []
                        
                        # Condition 1: Charging_time_999_h > 2 * CHARGE_HOURS
                        if not np.isnan(result.get("Charging_time_999_h", np.nan)):
                            if result["Charging_time_999_h"] > 2 * CHARGE_HOURS:
                                invalid_reasons.append(f"Charging_time_999_h ({result['Charging_time_999_h']:.2f}h) > 2*CHARGE_HOURS ({2*CHARGE_HOURS:.2f}h)")
                        
                        # Condition 2: r_cryo_re > 50 %
                        if not np.isnan(result.get("r_cryo_re", np.nan)):
                            if result["r_cryo_re"] > 50.0:
                                invalid_reasons.append(f"r_cryo_re ({result['r_cryo_re']:.2f}%) > 50%")
                        
                        # Update invalid_reason and possibly downgrade status to "warning"
                        if invalid_reasons:
                            if result["invalid_reason"]:
                                result["invalid_reason"] += "; " + "; ".join(invalid_reasons)
                            else:
                                result["invalid_reason"] = "; ".join(invalid_reasons)
                            # If status is 'success', downgrade to 'warning'
                            if result["status"] == "success":
                                result["status"] = "warning"
                        
                        # Append to in‑memory buffer and mark as computed
                        results.append(result)
                        existing_keys.add(key)
                        
                        # Flush every FLUSH_INTERVAL rows
                        if len(results) >= FLUSH_INTERVAL:
                            batch_df = pd.DataFrame(results)
                            write_header = not OUTPUT_CSV.exists()
                            batch_df.to_csv(OUTPUT_CSV, mode="a", header=write_header, index=False)
                            written_count += len(results)
                            print(f"  [Flush] Saved {len(results)} new results. Total: {written_count} rows")
                            results = []  # Reset buffer
    
    # 8) Flush any remaining buffered rows
    if results:
        batch_df = pd.DataFrame(results)
        write_header = not OUTPUT_CSV.exists()
        batch_df.to_csv(OUTPUT_CSV, mode="a", header=write_header, index=False)
        written_count += len(results)
        print(f"  [Flush] Saved final {len(results)} results. Total: {written_count} rows")
    
    # 9) Load final CSV, add derived columns and write CSV+Excel
    print("\n[Info] Saving final results to Excel...")
    final_df = pd.read_csv(OUTPUT_CSV) if OUTPUT_CSV.exists() else pd.DataFrame()
    if not final_df.empty:
        if "feasible_charge_time" not in final_df.columns:
            final_df["feasible_charge_time"] = (
                final_df["Charging_time_999_h"].notna()
                & (final_df["Charging_time_999_h"] <= 2 * cfg.charge_hours_max)
            )
        if "feasible_r_cryo_re" not in final_df.columns:
            final_df["feasible_r_cryo_re"] = (
                final_df["r_cryo_re"].notna()
                & (final_df["r_cryo_re"] <= cfg.DELTA_LCOE_MASK_PARASITIC_PCT)
            )
        if "feasible" not in final_df.columns:
            final_df["feasible"] = final_df["feasible_charge_time"] & final_df["feasible_r_cryo_re"]
        if "feasibility_flags" not in final_df.columns:
            def _flags(row: pd.Series) -> str:
                flags = []
                if not bool(row.get("feasible_charge_time", True)):
                    flags.append("charge_time")
                if not bool(row.get("feasible_r_cryo_re", True)):
                    flags.append("r_cryo_re")
                if str(row.get("status", "")).lower() == "error":
                    flags.append("error")
                return ";".join(flags)
            final_df["feasibility_flags"] = final_df.apply(_flags, axis=1)

        # Compute relative AF: AF_ref = AF / AF_max(scenario)
        if "AF_ref" not in final_df.columns and "AF" in final_df.columns:
            final_df["AF_ref"] = np.nan
            max_af = (
                final_df.loc[final_df["AF"].notna()]
                .groupby("scenario")["AF"]
                .transform("max")
            )
            final_df.loc[final_df["AF"].notna(), "AF_ref"] = final_df.loc[
                final_df["AF"].notna(), "AF"
            ] / max_af

        if "delta_LCOE_min_USD_per_MWh" not in final_df.columns and "LCOE_magnet_only_USD_per_MWh" in final_df.columns:
            final_df["delta_LCOE_min_USD_per_MWh"] = np.nan
            group_cols = ["scenario"]
            feasible_mask = final_df["feasible"] & final_df["LCOE_magnet_only_USD_per_MWh"].notna()
            min_lcoe = (
                final_df.loc[feasible_mask]
                .groupby(group_cols)["LCOE_magnet_only_USD_per_MWh"]
                .transform("min")
            )
            final_df.loc[feasible_mask, "delta_LCOE_min_USD_per_MWh"] = (
                final_df.loc[feasible_mask, "LCOE_magnet_only_USD_per_MWh"] - min_lcoe
            )
    
    if not final_df.empty:
        # Write back CSV (with feasibility and ΔLCOE fields added)
        final_df.to_csv(OUTPUT_CSV, index=False)
        # Save Excel version
        final_df.to_excel(OUTPUT_XLSX, index=False, engine='openpyxl')
        print(f"[OK] Results saved:")
        print(f"  - CSV: {OUTPUT_CSV} ({len(final_df):,} rows)")
        print(f"  - Excel: {OUTPUT_XLSX} ({len(final_df):,} rows)")
        
        # Print summary statistics
        print(f"\n[Info] Scan summary:")
        print(f"  - Total combinations: {total_combinations:,}")
        print(f"  - Computed: {computed_count:,}")
        print(f"  - Skipped: {skipped_count:,}")
        print(f"  - Status distribution:")
        if 'status' in final_df.columns:
            print(final_df['status'].value_counts().to_string())
    else:
        print("[Warning] No results to save!")

    # 10) Generate manifest.json
    try:
        write_manifest(
            repo_root=cfg.REPO_ROOT,
            config_path=SCAN_CONFIG_PATH,
            inputs_root=cfg.DATA_RAW_DIR,
            outputs_root=cfg.OUTPUTS_DIR,
            manifest_path=cfg.OUTPUTS_DIR / "manifest.json",
        )
        print(f"[OK] Manifest saved: {cfg.OUTPUTS_DIR / 'manifest.json'}")
    except Exception as e:
        print(f"[Warning] Failed to write manifest.json: {e}")


if __name__ == "__main__":
    main()
