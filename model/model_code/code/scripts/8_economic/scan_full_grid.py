"""
完整网格扫描脚本：生成 Supplementary Data

扫描维度：
- Top_K 和 coolant: [(4.2, 'He'), (10, 'He'), (20, 'He'), (20, 'H2')]
- Npw: 1..200
- rho_turn_uOhm_cm2: logspace(10, 10000, 31)  # 单位 μΩ·cm²
- R_joint_nOhm: [10, 100]
- scenario: ['S1', 'S2', 'S3']

输出：
- scan_full_grid_tidy_plant_opex_2025usd_direct_hts.csv（长表：一行一个组合）
- scan_full_grid_plant_opex_2025usd_direct_hts.xlsx（可选：同内容，便于审稿人打开）
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from datetime import datetime
from functools import lru_cache

# 导入项目模块
from fusion_tem import device as cfg
from tfmag.paths import ensure_base_dirs
from fusion_tem.cryo.heat_load import calculate_base_heat_loads
from fusion_tem.cryo import heat_load as hlm
from fusion_tem.utils import calculate_radial_resistance
from fusion_tem.economic.lcoe import compute_case, define_parameters, crf
from fusion_tem.economic.cost_boundary import (
    ANNUAL_COOLANT_REPLENISH_FRACTION,
    BACKGROUND_CAPITAL_USD_BY_SCENARIO,
    CORE_VOM_USD_PER_MWH_TH_BY_SCENARIO,
    COST_BOUNDARY_VERSION,
    FUSION_POWER_MWTH,
    GROSS_ELECTRIC_POWER_MWE,
    PCS_CAPITAL_COST_USD_PER_KWE,
    PCS_FOM_FRACTION_PER_YEAR,
    PCS_VOM_USD_PER_MWH_E,
    THERMAL_POWER_AFTER_BLANKET_MWTH,
)
from fusion_tem.economic.price_basis import (
    CONVERSION_TABLE_PATH,
    COOLANT_PRICE_2025_USD_PER_KG_BY_SCENARIO,
    HTS_PRICE_2025_USD_PER_KAM_BY_SCENARIO,
    POWER_SUPPLY_PRICE_2025_USD_PER_A,
    PRICE_BASIS_YEAR,
    conversion_table_sha256,
)
from fusion_tem.economic.feasibility import (
    add_joint_lcoe_reference,
    apply_system_feasibility,
)
from tfmag.manifest import write_manifest
from tfmag.xlsx import stream_csv_to_xlsx

# 导入充电时间计算函数
import importlib.util
spec = importlib.util.spec_from_file_location("charge_time_module",
    cfg.REPO_ROOT / "scripts" / "2_charging" / "2.5_charge_time999_TF_system_all_Npw=1-200.py")
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

    if matrix is None:
        raise FileNotFoundError(
            f"TF-system inductance matrix is required; no synthetic fallback is permitted: {matrix_path}"
        )
    if matrix.shape != (cfg.Ntf, cfg.Ntf):
        raise ValueError(
            f"invalid TF-system inductance matrix shape {matrix.shape}; expected {(cfg.Ntf, cfg.Ntf)}"
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
        # 温度必须在器件 yaml 的 Ip_list / Nt_list 里显式定义。
        # 原先静默回退到 cfg.I_TARGET(500 A) / cfg.N_TOTAL_TAPE, 加新温度时会拿到错值。
        if temperature not in cfg.Ip_list or temperature not in cfg.Nt_list:
            raise KeyError(
                f"器件配置缺少 Top={temperature} K 的工作点: "
                f"Ip_list={sorted(cfg.Ip_list)}, Nt_list={sorted(cfg.Nt_list)}"
            )
        Ip_case = cfg.Ip_list[temperature]
        Ntape_coil_case = cfg.Nt_list[temperature]
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
        # 未收敛 -> 明确不可行。不能返回 cfg.charge_hours_max, 它同时是筛选阈值
        # (MAX_CHARGE_HOURS), 而判据用严格 ">", 会让失败恰好"刚好合格"。见 §14.45。
        return float(time_999_h) if time_999_h is not None else float("inf")
    except Exception:
        return float("inf")

if hasattr(charge_time_module, "calculate_charge_time_999"):
    calculate_charge_time_999 = charge_time_module.calculate_charge_time_999
else:
    calculate_charge_time_999 = _fallback_calculate_charge_time_999

# =============================================================================
# Config params (aligned with 8.0)
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
npw_cfg = SCAN_CFG.get("npw_values", cfg.NPW_SCAN_VALUES)
if isinstance(npw_cfg, dict) and "range" in npw_cfg:
    npw_spec = npw_cfg["range"]
    NPW_RANGE = np.arange(
        int(npw_spec["start"]),
        int(npw_spec["stop"]) + 1,
        int(npw_spec.get("step", 1)),
        dtype=int,
    )
else:
    NPW_RANGE = np.array(npw_cfg, dtype=int)
# R_JOINT_NOHM：以 nOhm 为单位的接头电阻扫描数组
r_joint_cfg = SCAN_CFG.get("r_joint_nohm_values", list(cfg.R_JOINT_SCAN_VALUES))
if isinstance(r_joint_cfg, dict) and "logspace" in r_joint_cfg:
    rj_spec = r_joint_cfg["logspace"]
    R_JOINT_NOHM = np.geomspace(
        float(rj_spec["start"]), float(rj_spec["stop"]), int(rj_spec["num"])
    )
else:
    R_JOINT_NOHM = np.array(r_joint_cfg, dtype=float)

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
SCENARIOS = SCAN_CFG.get(
    "scenarios",
    list(cfg.SCENARIO_DEFINITIONS.keys()) if hasattr(cfg, 'SCENARIO_DEFINITIONS') else ['S1', 'S2', 'S3', 'S4', 'S5', 'S6'],
)

# Paths
PATHS = ensure_base_dirs()
_output_csv_name = os.environ.get(
    "SCAN_OUTPUT_CSV_NAME", "scan_full_grid_tidy_plant_opex_2025usd_direct_hts.csv"
)
_output_xlsx_name = os.environ.get(
    "SCAN_OUTPUT_XLSX_NAME", "scan_full_grid_plant_opex_2025usd_direct_hts.xlsx"
)
if Path(_output_csv_name).name != _output_csv_name or not _output_csv_name.lower().endswith(".csv"):
    raise ValueError("SCAN_OUTPUT_CSV_NAME must be a plain .csv filename")
if Path(_output_xlsx_name).name != _output_xlsx_name or not _output_xlsx_name.lower().endswith(".xlsx"):
    raise ValueError("SCAN_OUTPUT_XLSX_NAME must be a plain .xlsx filename")
OUTPUT_CSV = PATHS.outputs_tables / _output_csv_name
OUTPUT_XLSX = PATHS.outputs_tables / _output_xlsx_name
_manifest_name = os.environ.get(
    "SCAN_MANIFEST_NAME", "manifest_plant_opex_2025usd_direct_hts.json"
)
if Path(_manifest_name).name != _manifest_name or not _manifest_name.lower().endswith(".json"):
    raise ValueError("SCAN_MANIFEST_NAME must be a plain .json filename")
OUTPUT_MANIFEST = cfg.OUTPUTS_DIR / "manifests" / _manifest_name
OUTPUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)

def _resolve_circuit_cache_path() -> Path:
    value = os.environ.get("CIRCUIT_SCALAR_CACHE_CSV", SCAN_CFG.get("circuit_scalar_cache_csv", "outputs/tables/circuit_scalar_cache.csv"))
    path = Path(value)
    return path if path.is_absolute() else cfg.REPO_ROOT / path


CIRCUIT_SCALAR_CACHE = _resolve_circuit_cache_path()

# Optional compact production schema.  It is deliberately applied only at
# serialization, after every physical/economic field has been evaluated.
_COMPACT_COLUMNS_ENV = os.environ.get("SCAN_COMPACT_COLUMNS", "").strip()
COMPACT_OUTPUT_COLUMNS = tuple(
    column.strip() for column in _COMPACT_COLUMNS_ENV.split(",") if column.strip()
)

def _serialize_scan_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if not COMPACT_OUTPUT_COLUMNS:
        return frame
    missing = sorted(set(COMPACT_OUTPUT_COLUMNS).difference(frame.columns))
    if missing:
        raise KeyError(f"Compact output requested unavailable columns: {missing}")
    return frame.loc[:, list(COMPACT_OUTPUT_COLUMNS)]

# Flush interval
FLUSH_INTERVAL = int(os.environ.get("SCAN_FLUSH_INTERVAL", "500"))
PROGRESS_INTERVAL = int(os.environ.get("SCAN_PROGRESS_INTERVAL", "100"))
if FLUSH_INTERVAL < 1 or PROGRESS_INTERVAL < 1:
    raise ValueError("SCAN_FLUSH_INTERVAL and SCAN_PROGRESS_INTERVAL must be positive")

# =============================================================================
# 辅助函数：读取数据
# =============================================================================

def load_charging_time999_df(Top: float, Ip: float, Ntape_coil: int) -> pd.DataFrame:
    """
    读取充电99.9%时间数据。
    
    Returns:
        DataFrame，索引为rho_turn_uOhm_cm2，列为Npw，值为充电时间(小时)
    """
    base_path = Path(__file__).parent
    base_dir = base_path / cfg.TF_SYSTEM_CHARGING_SIM_OUTPUT_DIR / f"Temp_{Top}K_Ip_{Ip}A_Ntape_coil_{Ntape_coil}_charge999"
    file_path = base_dir / cfg.TF_SYSTEM_CHARGING_SIM_OUTPUT_FILE_Npw1_200
    
    if not file_path.exists():
        return pd.DataFrame()
    
    try:
        df = pd.read_excel(file_path, index_col=0)
        # 确保索引和列都是数值类型
        df.index = pd.to_numeric(df.index, errors='coerce')
        df.columns = pd.to_numeric(df.columns, errors='coerce')
        return df
    except Exception as e:
        print(f"  [Warning] Failed to read charging time file: {e}")
        return pd.DataFrame()


def load_magnetization_losses_for_temp(Top: float) -> Dict[Tuple[int, float], pd.Series]:
    """Load Data S3 only; direct circuit scalars replace Data S2."""
    mag_loss_file = Path(__file__).parent / cfg.HEAT_DATA_DIR / cfg.MAG_LOSS_FILE
    if not mag_loss_file.exists():
        return {}
    sheet_name = f"{Top}K"
    try:
        try:
            df = pd.read_excel(mag_loss_file, sheet_name=sheet_name, header=0, skiprows=lambda x: x < 5)
        except (ValueError, KeyError):
            df = pd.read_excel(mag_loss_file, sheet_name=0, header=0, skiprows=lambda x: x < 5)
        if len(df.columns) < 4:
            return {}
        rho_col, npw_col, time_col, value_col = df.columns[:4]
        return {(int(npw), float(rho)): group.assign(**{"Time (h)": group[time_col] / 3600.0}).set_index("Time (h)")[value_col]
                for (npw, rho), group in df.groupby([npw_col, rho_col])}
    except Exception as exc:
        print(f"  [Warning] Failed to read magnetic loss from {mag_loss_file}: {exc}")
        return {}


def _circuit_key(Top: float, Npw: int, rho: float) -> Tuple[float, int, float]:
    return (round(float(Top), 9), int(Npw), round(float(rho), 9))


def load_circuit_scalar_cache(csv_path: Path) -> Dict[Tuple[float, int, float], dict]:
    required = {"Top_K", "Npw", "rho_turn_uOhm_cm2", "device", "TF_system_matrix_file", "TF_system_matrix_sha256", "Charging_time_999_h", "Radial_loss_at_charge_W", "Radial_loss_peak_W", "Radial_loss_energy_MWh", "Radial_loss_average_W", "circuit_solver_status", "Radial_loss_power_basis"}
    if not csv_path.exists():
        raise FileNotFoundError(f"Direct circuit scalar cache missing: {csv_path}; run 2.7_build_circuit_scalar_cache.py")
    df = pd.read_csv(csv_path)
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Circuit cache missing columns: {sorted(missing)}")
    cache = {}
    for row in df.to_dict(orient="records"):
        key = _circuit_key(row["Top_K"], row["Npw"], row["rho_turn_uOhm_cm2"])
        values = [row[name] for name in ("Charging_time_999_h", "Radial_loss_at_charge_W", "Radial_loss_peak_W", "Radial_loss_energy_MWh", "Radial_loss_average_W")]
        if str(row["device"]).lower() != cfg.DEVICE:
            raise ValueError(f"Circuit-cache device mismatch for {key}: {row['device']!r} != {cfg.DEVICE!r}")
        if not str(row["TF_system_matrix_file"]).strip() or not str(row["TF_system_matrix_sha256"]).strip():
            raise ValueError(f"Circuit-cache matrix provenance missing for {key}")
        if key in cache or row["circuit_solver_status"] != "success" or not np.all(np.isfinite(np.asarray(values, float))) or float(values[0]) <= 0.0 or any(float(v) < 0.0 for v in values[1:]):
            raise ValueError(f"Invalid circuit-cache row for {key}")
        if row["Radial_loss_power_basis"] != "per_TF=sum_18_branch_P_R_over_Ntf":
            raise ValueError(f"Unexpected radial-loss basis for {key}")
        cache[key] = row
    return cache


def interpolate_loss_at_time(loss_series: pd.Series, target_time_h: float) -> float:
    """
    在指定时间点插值获取损失功率。
    
    Args:
        loss_series: 时间序列（索引为时间(小时)，值为功率(W)）
        target_time_h: 目标时间(小时)
    
    Returns:
        插值得到的功率(W)，如果无法插值则返回NaN
    """
    if loss_series.empty:
        return np.nan
    
    if target_time_h <= loss_series.index.max():
        return np.interp(target_time_h, loss_series.index, loss_series.values)
    else:
        # 如果目标时间超出范围，返回最后一个值
        return loss_series.iloc[-1] if len(loss_series) > 0 else np.nan



def align_loss_series_to_event(
    magnetization_loss: pd.Series,
    radial_loss: pd.Series,
    event_end_h: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align Data S2/S3 series and include the exact event endpoint."""
    if event_end_h <= 0:
        raise ValueError("event_end_h must be positive")
    if magnetization_loss is None or radial_loss is None:
        raise ValueError("both Data S2 and Data S3 series are required")
    if magnetization_loss.empty or radial_loss.empty:
        raise ValueError("Data S2/S3 series cannot be empty")

    mag = magnetization_loss.sort_index()
    radial = radial_loss.sort_index()
    times = np.unique(np.concatenate([
        np.array([0.0, float(event_end_h)]),
        mag.index.to_numpy(dtype=float),
        radial.index.to_numpy(dtype=float),
    ]))
    times = times[(times >= 0.0) & (times <= float(event_end_h))]
    if times.size < 2:
        raise ValueError("aligned loss series must contain at least two time points")
    mag_values = np.interp(
        times,
        mag.index.to_numpy(dtype=float),
        mag.to_numpy(dtype=float),
        left=float(mag.iloc[0]),
        right=float(mag.iloc[-1]),
    )
    radial_values = np.interp(
        times,
        radial.index.to_numpy(dtype=float),
        radial.to_numpy(dtype=float),
        left=float(radial.iloc[0]),
        right=float(radial.iloc[-1]),
    )
    return times, mag_values, radial_values


def get_charging_time_999(
    Top: float,
    Ip: float,
    Ntape_coil: int,
    Npw: int,
    rho_turn_uOhm_cm2: float,
    time999_df: Optional[pd.DataFrame] = None
) -> float:
    """
    获取充电时间99.9%。
    
    优先从预计算的Excel读取，如果缺失则调用calculate_charge_time_999补算。
    
    Args:
        Top: 运行温度 (K)
        Ip: 电流 (A)
        Ntape_coil: 带材根数
        Npw: 并绕根数
        rho_turn_uOhm_cm2: 匝间电阻率 (μΩ·cm²)
        time999_df: 预加载的充电时间DataFrame（可选）
    
    Returns:
        充电时间(小时)
    """
    # 从 2.5 的预算表精确查值。
    # 2026-07-26 前这里是"线性最近邻、无距离上限"——扫描网格若不被充电表覆盖,
    # 会静默取用任意远的邻点(差 10 倍也照取)。现要求精确命中(容差见下),
    # 命中不了就抛错, 由调用方决定是补算还是终止。见工作文档 §14.45。
    if time999_df is not None and not time999_df.empty:
        available_rhot = np.asarray(time999_df.index.values, dtype=float)
        available_npw = np.asarray(time999_df.columns.values, dtype=float)
        if len(available_rhot) and len(available_npw):
            i = int(np.argmin(np.abs(available_rhot - rho_turn_uOhm_cm2)))
            j = int(np.argmin(np.abs(available_npw - Npw)))
            d_rhot = abs(available_rhot[i] - rho_turn_uOhm_cm2)
            d_npw = abs(available_npw[j] - Npw)
            # 容差: rho 用相对 1e-6(表内索引按 round 存储), Npw 必须整数级精确
            if d_rhot <= max(1e-6 * max(rho_turn_uOhm_cm2, 1.0), 1e-6) and d_npw < 0.5:
                value = time999_df.at[time999_df.index[i], time999_df.columns[j]]
                if pd.notna(value):
                    return float(value)
            else:
                raise KeyError(
                    f"充电时间表未覆盖扫描点 (Top={Top}, Npw={Npw}, "
                    f"rho_turn={rho_turn_uOhm_cm2}): 最近可用点相差 "
                    f"Npw {d_npw:g}, rho {d_rhot:g}。"
                    "请先用同一套网格重跑 2.5, 不要依赖最近邻近似。"
                )

    # 表内该格为 NaN(2.5 未收敛)时才补算
    charging_time_999_h = _cached_charge_time_999(
        Top=Top,
        Npw=Npw,
        rho_turn_uOhm_cm2=rho_turn_uOhm_cm2,
    )
    if charging_time_999_h is not None:
        return charging_time_999_h
    # 补算也失败 -> 明确不可行, 不能返回等于阈值的 charge_hours_max(见 §1.4)
    return float("inf")


def load_existing_results(csv_path: Path) -> pd.DataFrame:
    """
    加载已存在的结果文件，用于resume功能。
    
    Returns:
        DataFrame，如果文件不存在则返回空DataFrame
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
    检查某个组合是否已经计算过。
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
    R_joint_nOhm: float,
    scenario: str,
) -> Tuple[float, str, int, float, float, str]:
    return (
        float(Top),
        str(coolant),
        int(Npw),
        float(rho_turn_uOhm_cm2),
        round(float(R_joint_nOhm), 3),
        str(scenario),
    )


def load_existing_results_set(csv_path: Path) -> Tuple[pd.DataFrame, set]:
    """
    ?????????????? set ????????
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

    completion_cols = {
        "cost_boundary_version",
        "monetary_price_basis_year",
        "monetary_values_constant_2025_usd",
        "hts_conductor_price_source_year",
        "hts_conductor_price_cpi_factor",
        "hts_conductor_price_conversion_method",
        "C0_background_USD",
        "CAPEX_mag_direct_USD",
        "CAPEX_mag_installed_USD",
        "CAPEX_PCS_reference_USD",
        "Annualized_CAPEX_mag_USD_per_year",
        "Annualized_C0_USD_per_year",
        "Annualized_CAPEX_total_modeled_USD_per_year",
        "E_fusion_th_year_MWh",
        "OPEX_core_VOM_USD_per_year",
        "OPEX_PCS_VOM_USD_per_year",
        "OPEX_PCS_FOM_USD_per_year",
        "OPEX_coolant_VOM_USD_per_year",
        "OPEX_total_modeled_USD_per_year",
        "LCOE_plant_USD_per_MWh",
        "LCOE_magnet_only_USD_per_MWh",
        "LCOE_fullplant_legacy_USD_per_MWh",
        "E_cryo_charge_MWh",
        "E_cryo_discharge_MWh",
        "P_cryo_charge_avg_W",
        "P_cryo_discharge_avg_W",
        "E_dynamic_charge_MWh",
        "E_dynamic_discharge_MWh",
        "charge_discharge_profile_ratio",
        "charge_time_event_h",
        "discharge_time_event_h",
        "transient_energy_fraction_of_Ecryo",
        "E_cryo_year_MWh",
        "E_gross_year_MWh",
        "E_other_year_MWh",
        "E_net_year_MWh",
    }
    missing_completion_cols = sorted(completion_cols.difference(df.columns))
    if missing_completion_cols:
        raise RuntimeError(
            "Existing scan uses a pre-P0 schema; choose a new "
            "SCAN_OUTPUT_CSV_NAME instead of mixing rows. Missing: "
            + ", ".join(missing_completion_cols)
        )
    version_mismatch = df["cost_boundary_version"].astype(str).ne(COST_BOUNDARY_VERSION)
    if version_mismatch.any():
        observed = sorted(df.loc[version_mismatch, "cost_boundary_version"].astype(str).unique())
        raise RuntimeError(
            "Existing scan has a different cost boundary version; refusing resume. "
            f"Expected {COST_BOUNDARY_VERSION!r}, observed {observed!r}."
        )
    incomplete_mask = df[list(completion_cols)].isna().any(axis=1)
    if incomplete_mask.any():
        print(
            f"[Info] Removing {int(incomplete_mask.sum()):,} incomplete resume rows "
            "so their annual energy fields can be recomputed."
        )
        df = df.loc[~incomplete_mask].copy()
        df.to_csv(csv_path, index=False)

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


def is_computed_key(existing_keys: set, key: Tuple[float, str, int, float, float, str]) -> bool:
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


COST_EXPORT_FIELDS = (
    "C0_background_USD",
    "CAPEX_mag_direct_USD",
    "CAPEX_mag_installed_USD",
    "CAPEX_PCS_reference_USD",
    "Annualized_CAPEX_mag_USD_per_year",
    "Annualized_C0_USD_per_year",
    "Annualized_CAPEX_total_modeled_USD_per_year",
    "E_fusion_th_year_MWh",
    "E_gross_year_MWh",
    "OPEX_core_VOM_USD_per_year",
    "OPEX_PCS_VOM_USD_per_year",
    "OPEX_PCS_FOM_USD_per_year",
    "OPEX_coolant_VOM_USD_per_year",
    "OPEX_total_modeled_USD_per_year",
    "LCOE_plant_USD_per_MWh",
    "LCOE_magnet_only_USD_per_MWh",
    "LCOE_fullplant_legacy_USD_per_MWh",
    "economic_invalid_reason",
    "monetary_price_basis_year",
    "monetary_values_constant_2025_usd",
    "hts_conductor_price_source_year",
    "hts_conductor_price_cpi_factor",
    "hts_conductor_price_conversion_method",
)


def _economic_export_fields(economic_result: Optional[dict] = None) -> dict:
    """Return the stable plant-v4/direct-2025-USD schema plus legacy aliases."""
    source = economic_result or {}
    out = {
        name: source.get(name, None if name == "economic_invalid_reason" else np.nan)
        for name in COST_EXPORT_FIELDS
    }
    out["cost_boundary_version"] = source.get(
        "cost_boundary_version", COST_BOUNDARY_VERSION
    )
    # Deprecated aliases retain their pre-v2 mathematical meaning.
    out.update({
        "C0_nonmagnet_USD": source.get("C0_background_USD", np.nan),
        "CAPEX_mag_USD": source.get("CAPEX_mag_installed_USD", np.nan),
        "Annualized_CAPEX_mag_USD_per_year": source.get(
            "Annualized_CAPEX_mag_USD_per_year", np.nan
        ),
        "Annual_OPEX_mag_USD_per_year": source.get(
            "OPEX_coolant_VOM_USD_per_year", np.nan
        ),
        "LCOE_fullplant_USD_per_MWh": source.get(
            "LCOE_fullplant_legacy_USD_per_MWh", np.nan
        ),
    })
    return out


# =============================================================================
# 主函数
# =============================================================================
def main():
    """运行完整网格扫描"""
    
    # 1. 不再定义扫描维度，使用模块开头最开始定义的
    
    # 2. 初始化输出路径
    # OUTPUT_CSV and OUTPUT_XLSX are module-level and can be redirected with
    # SCAN_OUTPUT_CSV_NAME to preserve earlier datasets.
    
    # 3. 加载已存在的结果（resume功能）
    existing_df, existing_keys = load_existing_results_set(OUTPUT_CSV)
    if not existing_df.empty:
        print(f"[Info] Found existing results: {len(existing_df)} rows. Will skip computed combinations.")
    written_count = len(existing_df)
    
    # 4. 初始化经济学参数（只需要一次）
    econ_params = define_parameters()
    
    # 5. 预加载每个温度的充电时间DataFrame和电磁损失数据（每个温度只读一次）
    circuit_scalar_cache = load_circuit_scalar_cache(CIRCUIT_SCALAR_CACHE)
    magnetization_losses_cache = {}
    transient_metric_cache = {}
    required_keys = {_circuit_key(Top, Npw, rho) for Top, _ in TEMP_COOLANT_PAIRS for Npw in NPW_RANGE for rho in RHO_TURN_UOHM_CM2}
    missing_keys = sorted(required_keys.difference(circuit_scalar_cache))
    if missing_keys:
        raise RuntimeError(f"Circuit cache lacks {len(missing_keys)} exact scan points; examples={missing_keys[:5]}")
    print(f"[OK] Direct circuit cache: {CIRCUIT_SCALAR_CACHE} ({len(circuit_scalar_cache)} designs; exact coverage PASS)")
    for Top, _ in TEMP_COOLANT_PAIRS:
        if Top not in magnetization_losses_cache:
            magnetization_losses_cache[Top] = load_magnetization_losses_for_temp(Top)

    # 6. 计算总组合数
    total_combinations = len(TEMP_COOLANT_PAIRS) * len(NPW_RANGE) * len(RHO_TURN_UOHM_CM2) * len(R_JOINT_NOHM) * len(SCENARIOS)
    print(f"\n[Info] Total combinations to compute: {total_combinations:,}")
    
    # 7. 开始扫描
    results = []  # Reset buffer
    computed_count = 0
    skipped_count = 0
    
    print("\n[Info] Starting full grid scan...")
    
    for Top, coolant in TEMP_COOLANT_PAIRS:
        # 获取电流和带材根数
        Ip = cfg.Ip_list.get(Top, 300.0)
        Ntape_coil = cfg.Nt_list.get(Top, 1400)
        
        # 获取预加载的数据
        magnetization_losses_dict = magnetization_losses_cache.get(Top, {})
        
        for scenario in SCENARIOS:
            scenario_year = econ_params['tech_scenario_to_years'][scenario]
            
            for Npw in NPW_RANGE:
                for rho_turn_uOhm_cm2 in RHO_TURN_UOHM_CM2:
                    for R_joint_nOhm in R_JOINT_NOHM:
                        # 检查是否已计算
                        key = _make_key(Top, coolant, Npw, rho_turn_uOhm_cm2, R_joint_nOhm, scenario)
                        if is_computed_key(existing_keys, key):
                            skipped_count += 1
                            if skipped_count % 1000 == 0:
                                print(f"  [Info] Skipped {skipped_count:,} combinations...")
                            continue
                        
                        computed_count += 1
                        if computed_count % PROGRESS_INTERVAL == 0:
                            print(f"  [Progress] Computed {computed_count:,}/{total_combinations:,} ({100*computed_count/total_combinations:.1f}%)")

                        # INSERT_YOUR_CODE
                        # 保证R_joint_nOhm保留小数点后三位存储
                        R_joint_nOhm = round(R_joint_nOhm, 3)
                        # 初始化结果字典
                        result = {
                            "Top_K": Top,
                            "coolant": coolant,
                            "scenario": scenario,
                            "Npw": Npw,
                            "rho_turn_uOhm_cm2": rho_turn_uOhm_cm2,
                            "R_joint_nOhm": R_joint_nOhm,
                            "status": "success",
                            "invalid_reason": None,
                            "C0_nonmagnet_USD": econ_params["C0_nonmagnet_USD_by_scenario"][scenario],
                            "project_lifetime_years": scenario_year,
                            "discount_rate": econ_params["discount_rate"],
                            "CRF": crf(econ_params["discount_rate"], scenario_year),
                            "P_cryo_charge_avg_W": np.nan,
                            "P_cryo_discharge_avg_W": np.nan,
                            "E_cryo_charge_MWh": np.nan,
                            "E_cryo_discharge_MWh": np.nan,
                            "E_dynamic_charge_MWh": np.nan,
                            "E_dynamic_discharge_MWh": np.nan,
                            "charge_discharge_profile_ratio": np.nan,
                            "charge_time_event_h": np.nan,
                            "discharge_time_event_h": np.nan,
                            "transient_energy_fraction_of_Ecryo": np.nan,
                            "r_cryo_re_fraction": np.nan,
                            "energy_closure_error_MWh": np.nan,
                        }
                        result.update(_economic_export_fields())
                        result["C0_background_USD"] = econ_params[
                            "background_capital_USD_by_scenario"
                        ][scenario]
                        
                        # ========== 0. Direct circuit scalar cache ==========
                        circuit_row = circuit_scalar_cache[_circuit_key(Top, Npw, rho_turn_uOhm_cm2)]
                        charging_time_999_h = float(circuit_row["Charging_time_999_h"])
                        radial_loss_at_charge = float(circuit_row["Radial_loss_at_charge_W"])
                        radial_loss_peak_w = float(circuit_row["Radial_loss_peak_W"])
                        radial_loss_energy_MWh = float(circuit_row["Radial_loss_energy_MWh"])
                        radial_loss_average_w = float(circuit_row["Radial_loss_average_W"])
                        result.update({"Charging_time_999_h": charging_time_999_h, "Radial_loss_at_charge_W": radial_loss_at_charge, "Radial_loss_peak_W": radial_loss_peak_w, "Radial_loss_energy_MWh": radial_loss_energy_MWh, "Radial_loss_average_W": radial_loss_average_w, "Circuit_cache_source": str(CIRCUIT_SCALAR_CACHE), "TF_system_matrix_file": circuit_row["TF_system_matrix_file"], "TF_system_matrix_sha256": circuit_row["TF_system_matrix_sha256"], "Circuit_solver_status": circuit_row["circuit_solver_status"], "Radial_loss_power_basis": circuit_row["Radial_loss_power_basis"]})
                        mag_loss_at_charge = np.nan
                        mag_loss_energy_MWh = np.nan
                        mag_loss_values = radial_loss_values = loss_time_h = None
                        closest_key = None
                        if magnetization_losses_dict:
                            closest_key = min(magnetization_losses_dict, key=lambda key: abs(key[0] - Npw) / max(Npw, 1) + abs(key[1] - rho_turn_uOhm_cm2) / max(rho_turn_uOhm_cm2, 1e-6))
                            mag_series = magnetization_losses_dict[closest_key].sort_index()
                            mag_loss_at_charge = interpolate_loss_at_time(mag_series, cfg.CHARGE_HOURS)
                            loss_time_h = np.unique(np.concatenate([np.array([0.0, charging_time_999_h]), mag_series.index.to_numpy(float)]))
                            loss_time_h = loss_time_h[(loss_time_h >= 0.0) & (loss_time_h <= charging_time_999_h)]
                            if loss_time_h.size >= 2:
                                mag_loss_values = np.interp(loss_time_h, mag_series.index.to_numpy(float), mag_series.to_numpy(float), left=float(mag_series.iloc[0]), right=float(mag_series.iloc[-1]))
                                # Scalar cache coupling: energy-conserving radial power, no stored Data S2 trace.
                                radial_loss_values = np.full_like(loss_time_h, radial_loss_average_w, dtype=float)
                                mag_loss_energy_MWh = float(np.trapz(mag_loss_values, loss_time_h) / 1e6)
                            else:
                                loss_time_h = None
                        result.update({"EM_source_Npw": closest_key[0] if closest_key else np.nan, "EM_source_rho_turn_uOhm_cm2": closest_key[1] if closest_key else np.nan, "EM_coupling_method": "direct_circuit_scalar_cache_radial__magnetization_nearest_neighbor", "Mag_loss_at_charge_W": mag_loss_at_charge, "Mag_loss_energy_MWh": mag_loss_energy_MWh})

                        # ========== 1. 热学指标 ==========
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
                            result.update({
                                "P_cryo_electric_W": heat_loads["P_cryo_electric_W"],
                                "COP_Tc": heat_loads[cop_key],
                                "COP_77": heat_loads["COP @ 77K"],
                                # Top温区热源（使用Q_前缀）
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
                                # 77K温区热源（分解）
                                "Q_current_leads_Cu_conduction_77K_W": heat_loads["current_leads_Cu_conduction"],
                                "Q_current_leads_Cu_joule_77K_W": heat_loads["current_leads_Cu_joule"],
                                "Q_total_77K_W": total_heat_77,
                            })


                            transient_key = (
                                float(Top),
                                int(Npw),
                                float(rho_turn_uOhm_cm2),
                                float(R_joint_nOhm),
                                float(charging_time_999_h),
                                closest_key,
                            )
                            if transient_key in transient_metric_cache:
                                charge_metrics = transient_metric_cache[transient_key]
                            elif loss_time_h is not None:
                                charge_metrics = hlm.calculate_charge_cryo_electrical_energy(
                                    base_heat_loads=heat_loads,
                                    Top=Top,
                                    time_h=loss_time_h,
                                    magnetization_loss_W=mag_loss_values,
                                    radial_loss_W=radial_loss_values,
                                    N_tf=cfg.Ntf,
                                    efficiency_model=econ_params["cryo_efficiency_model"],
                                    eta_max=econ_params["cryo_eta_max"],
                                    rated_margin=econ_params["cryo_rated_margin"],
                                )
                                transient_metric_cache[transient_key] = charge_metrics
                            else:
                                charge_metrics = None
                                if result["status"] == "success":
                                    result["status"] = "warning"
                                msg = "Data S2/S3 transient series unavailable; charge energy uses design base load"
                                result["invalid_reason"] = (
                                    f"{result['invalid_reason']}; {msg}"
                                    if result["invalid_reason"] else msg
                                )

                            result.update({
                                "Q_charge_peak_Tc_W_per_TF": (
                                    charge_metrics["peak_heat_Tc_W_per_TF"]
                                    if charge_metrics else np.nan
                                ),
                                "Q_charge_77K_W_per_TF": (
                                    charge_metrics["heat_77K_W_per_TF"]
                                    if charge_metrics else np.nan
                                ),
                                "P_cryo_charge_peak_W": (
                                    charge_metrics["peak_power_W"]
                                    if charge_metrics else np.nan
                                ),
                            })
                        except Exception as e:
                            result["status"] = "error"
                            result["invalid_reason"] = f"Heat load calculation failed: {e}"
                            result.update({
                                "P_cryo_electric_W": np.nan,
                                "COP_Tc": np.nan,
                                "COP_77": np.nan,
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
                        
                        # ========== 2. 电磁指标 ==========
                        try:
                            # 计算径向电阻（单个饼状线圈）
                            R_radial_per_pancake = _cached_radial_resistance(
                                Npw=Npw,
                                Ntape_coil=Ntape_coil,
                                rho_turn_ohm_m2=rho_turn_uOhm_cm2*1e-10,
                            )
                            # 单个TF磁体的径向电阻 = 所有饼状线圈的串联
                            R_radial_per_TF = R_radial_per_pancake * cfg.NP
                            # TF系统的径向电阻 = 所有TF磁体的并联
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
                        
                        # ========== 3. 经济学指标 ==========
                        try:
                            # 临时设置真实的充电时间
                            economic_result, _ = compute_case(
                                tech_scenario=scenario,
                                year=scenario_year,
                                temperature_K=Top,
                                coolant=coolant,
                                Npw=Npw,
                                R_p2p_joint=R_joint_nOhm*1e-9,
                                p=econ_params,
                                charge_time_h=charging_time_999_h,
                                charge_cryo_energy_event_MWh=(
                                    charge_metrics["event_energy_MWh"]
                                    if charge_metrics else None
                                ),
                                charge_cryo_average_power_W=(
                                    charge_metrics["average_power_W"]
                                    if charge_metrics else None
                                ),
                                base_heat_loads_override=heat_loads,
                                rho_turn_uOhm_cm2=rho_turn_uOhm_cm2,
                                cryo_rated_context=(charge_metrics["rated_context"] if charge_metrics else None),
                                cryo_efficiency_model=econ_params["cryo_efficiency_model"],
                                cryo_eta_max=econ_params["cryo_eta_max"],
                            )
                            
                            if economic_result:
                                # 获取寄生功耗比例（百分比）
                                r_parasitic_pct = economic_result.get("r_parasitic_pct", np.nan)
                                
                                result.update({
                                    **_economic_export_fields(economic_result),
                                    "project_lifetime_years": economic_result.get("project_years", scenario_year),
                                    "discount_rate": economic_result.get("discount_rate", result["discount_rate"]),
                                    "CRF": economic_result.get("crf", result["CRF"]),
                                    "LCOE_C0_USD_per_MWh": economic_result.get("lcoe_C0_$/MWh", np.nan),
                                    "Tape_cost_USD": economic_result.get("tape_cost_$", np.nan),
                                    "HTS_price_2025USD_per_kAm": economic_result.get("input_hts_price_2025USD_per_kAm", np.nan),
                                    "Coolant_price_2025USD_per_kg": economic_result.get("input_coolant_price_2025USD_per_kg", np.nan),
                                    "Coolant_fill_cost_USD": economic_result.get("coolant_fill_cost_$", np.nan),
                                    "Power_supply_cost_USD": economic_result.get("power_supply_cost_$", np.nan),
                                    "Aplant": economic_result.get("plant_availability", np.nan),
                                    "pulse_duty_factor": economic_result.get("pulse_duty_factor", np.nan),
                                    "CF_gross": economic_result.get("gross_capacity_factor", np.nan),
                                    # Legacy percent field retained for figure compatibility.
                                    "r_cryo_re": r_parasitic_pct,
                                    "r_cryo_re_fraction": r_parasitic_pct / 100.0,
                                    "P_cryo_prod_W": economic_result.get("cryo_power_prod_W", np.nan),
                                    "P_cryo_dwell_W": economic_result.get("cryo_power_dwell_W", np.nan),
                                    "P_cryo_static_W": economic_result.get("cryo_power_static_W", np.nan),
                                    "P_cryo_coolwarm_W": economic_result.get("cryo_power_coolwarm_W", np.nan),
                                    "P_cryo_excdec_W": economic_result.get("cryo_power_excdec_W", np.nan),
                                    "P_cryo_charge_average_W": economic_result.get("cryo_power_charge_average_W", np.nan),
                                    "P_cryo_charge_avg_W": economic_result.get("cryo_power_charge_average_W", np.nan),
                                    "P_cryo_discharge_avg_W": economic_result.get("cryo_power_discharge_average_W", np.nan),
                                    "P_cryo_charge_base_W": economic_result.get("cryo_power_charge_base_W", np.nan),
                                    "E_cryo_charge_event_MWh": economic_result.get("cryo_energy_charge_event_MWh", np.nan),
                                    "E_cryo_discharge_event_MWh": economic_result.get("cryo_energy_discharge_event_MWh", np.nan),
                                    "E_cryo_charge_MWh": economic_result.get("cryo_energy_charge_annual_MWh", np.nan),
                                    "E_cryo_discharge_MWh": economic_result.get("cryo_energy_discharge_annual_MWh", np.nan),
                                    "E_dynamic_charge_MWh": economic_result.get("cryo_energy_dynamic_charge_annual_MWh", np.nan),
                                    "E_dynamic_discharge_MWh": economic_result.get("cryo_energy_dynamic_discharge_annual_MWh", np.nan),
                                    "charge_discharge_profile_ratio": economic_result.get("charge_discharge_profile_ratio", np.nan),
                                    "charge_time_event_h": economic_result.get("charge_time_event_h", np.nan),
                                    "discharge_time_event_h": economic_result.get("discharge_time_event_h", np.nan),
                                    "transient_energy_fraction_of_Ecryo": economic_result.get("transient_energy_fraction_of_Ecryo", np.nan),
                                    "E_cryo_excdis_year_MWh": economic_result.get("cryo_energy_excdec_annual_MWh", np.nan),
                                    "r_other": economic_result.get("other_aux_fraction", np.nan),
                                    "E_other_year_MWh": economic_result.get("other_aux_energy_annual_MWh", np.nan),
                                    "E_cryo_year_MWh": economic_result.get("cryo_energy_annual_MWh", np.nan),
                                    "E_gross_year_MWh": economic_result.get("gross_energy_annual_MWh", np.nan),
                                    "E_net_year_MWh": economic_result.get("net_energy_annual_MWh", np.nan),
                                    "cryo_efficiency_model": economic_result.get("cryo_efficiency_model"),
                                    "cryo_reference_temperature_K": economic_result.get("cryo_reference_temperature_K", np.nan),
                                    "cryo_rated_4p5eq_kW": economic_result.get("cryo_rated_4p5eq_kW", np.nan),
                                    "cryo_eta_raw_fraction_carnot": economic_result.get("cryo_eta_raw_fraction_carnot", np.nan),
                                    "cryo_eta_cap_fraction_carnot": economic_result.get("cryo_eta_cap_fraction_carnot", np.nan),
                                    "cryo_eta_rated_fraction_carnot": economic_result.get("cryo_eta_rated_fraction_carnot", np.nan),
                                    "cryo_eta_cap_active": economic_result.get("cryo_eta_cap_active"),
                                    "cryo_rated_margin": economic_result.get("cryo_rated_margin", np.nan),
                                    "cryo_part_load_factor": economic_result.get("cryo_part_load_factor", np.nan),
                                    "non_tf_aux_fraction": economic_result.get("non_tf_aux_fraction", np.nan),
                                    "E_cryo_TF_annual_MWh": economic_result.get("E_cryo_TF_annual_MWh", np.nan),
                                    "gross_energy_annual_MWh": economic_result.get("gross_energy_annual_MWh", np.nan),
                                    "net_energy_annual_MWh": economic_result.get("net_energy_annual_MWh", np.nan),
                                    "r_cryo_fraction": economic_result.get("r_cryo_fraction", np.nan),
                                    "r_cryo_pct": economic_result.get("r_cryo_pct", np.nan),
                                })
                                result["energy_closure_error_MWh"] = (
                                    result["E_gross_year_MWh"]
                                    - result["E_other_year_MWh"]
                                    - result["E_cryo_year_MWh"]
                                    - result["E_net_year_MWh"]
                                )
                                economic_invalid_reason = economic_result.get(
                                    "economic_invalid_reason"
                                )
                                if economic_invalid_reason:
                                    result["invalid_reason"] = (
                                        f"{result['invalid_reason']}; {economic_invalid_reason}"
                                        if result["invalid_reason"]
                                        else economic_invalid_reason
                                    )
                                    result["status"] = "error"
                            else:
                                if result["status"] == "success":
                                    result["status"] = "warning"
                                if result["invalid_reason"]:
                                    result["invalid_reason"] += "; Economic calculation result is empty"
                                else:
                                    result["invalid_reason"] = "Economic calculation result is empty"
                                result.update({
                                    "C0_nonmagnet_USD": econ_params["C0_nonmagnet_USD_by_scenario"][scenario],
                                    "project_lifetime_years": scenario_year,
                                    "discount_rate": econ_params["discount_rate"],
                                    "CRF": crf(econ_params["discount_rate"], scenario_year),
                                    "LCOE_fullplant_USD_per_MWh": np.nan,
                                    "LCOE_C0_USD_per_MWh": np.nan,
                                    "CAPEX_mag_USD": np.nan,
                                    "Annualized_CAPEX_mag_USD_per_year": np.nan,
                                    "Annual_OPEX_mag_USD_per_year": np.nan,
                                    "Tape_cost_USD": np.nan,
                                    "Coolant_fill_cost_USD": np.nan,
                                    "Power_supply_cost_USD": np.nan,
                                    "Aplant": np.nan,
                                    "pulse_duty_factor": np.nan,
                                    "CF_gross": np.nan,
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
                                "C0_nonmagnet_USD": econ_params["C0_nonmagnet_USD_by_scenario"][scenario],
                                "project_lifetime_years": scenario_year,
                                "discount_rate": econ_params["discount_rate"],
                                "CRF": crf(econ_params["discount_rate"], scenario_year),
                                "LCOE_fullplant_USD_per_MWh": np.nan,
                                "LCOE_C0_USD_per_MWh": np.nan,
                                "CAPEX_mag_USD": np.nan,
                                "Annualized_CAPEX_mag_USD_per_year": np.nan,
                                "Annual_OPEX_mag_USD_per_year": np.nan,
                                "Tape_cost_USD": np.nan,
                                "Coolant_fill_cost_USD": np.nan,
                                "Power_supply_cost_USD": np.nan,
                                    "Aplant": np.nan,
                                    "pulse_duty_factor": np.nan,
                                    "CF_gross": np.nan,
                                    "r_cryo_re": np.nan,
                            })
                        # ========== 4. 检查有效性并填充invalid_reason ==========
                        # 最大允许充电时间：直接采用配置中的120 h，不再放宽为2倍。
                        MAX_CHARGE_HOURS = cfg.charge_hours_max
                        
                        # 检查条件并更新invalid_reason
                        invalid_reasons = []
                        
                        # 条件1: Charging_time_999_h > MAX_CHARGE_HOURS
                        if not np.isnan(result.get("Charging_time_999_h", np.nan)):
                            if result["Charging_time_999_h"] > MAX_CHARGE_HOURS:
                                invalid_reasons.append(f"Charging_time_999_h ({result['Charging_time_999_h']:.2f}h) > MAX_CHARGE_HOURS ({MAX_CHARGE_HOURS:.2f}h)")
                        
                        # 条件2: r_cryo_re > 50%
                        if not np.isnan(result.get("r_cryo_re", np.nan)):
                            if result["r_cryo_re"] > 50.0:
                                invalid_reasons.append(f"r_cryo_re ({result['r_cryo_re']:.2f}%) > 50%")
                        
                        # 更新invalid_reason
                        if invalid_reasons:
                            if result["invalid_reason"]:
                                result["invalid_reason"] += "; " + "; ".join(invalid_reasons)
                            else:
                                result["invalid_reason"] = "; ".join(invalid_reasons)
                            # 如果状态是success，改为warning
                            if result["status"] == "success":
                                result["status"] = "warning"
                        
                        # 添加到结果列表
                        results.append(result)
                        existing_keys.add(key)
                        
                        # 每500行flush一次
                        if len(results) >= FLUSH_INTERVAL:
                            batch_df = pd.DataFrame(results)
                            write_header = not OUTPUT_CSV.exists()
                            _serialize_scan_frame(batch_df).to_csv(OUTPUT_CSV, mode="a", header=write_header, index=False)
                            written_count += len(results)
                            print(f"  [Flush] Saved {len(results)} new results. Total: {written_count} rows")
                            results = []  # Reset buffer
    
    # 8. 保存剩余结果
    if results:
        batch_df = pd.DataFrame(results)
        write_header = not OUTPUT_CSV.exists()
        _serialize_scan_frame(batch_df).to_csv(OUTPUT_CSV, mode="a", header=write_header, index=False)
        written_count += len(results)
        print(f"  [Flush] Saved final {len(results)} results. Total: {written_count} rows")
    
    # 9. 读取最终结果并保存为Excel
    if os.environ.get("SCAN_DEFER_FINALIZE", "0") == "1":
        print("[Info] Deferred finalization: V6.1-B reducer will apply full-domain feasibility.")
        return

    print("\n[Info] Saving final results to Excel...")
    final_df = (
        pd.read_csv(OUTPUT_CSV, low_memory=False)
        if OUTPUT_CSV.exists()
        else pd.DataFrame()
    )
    if not final_df.empty:
        primary_key_columns = [
            "Top_K", "coolant", "Npw", "rho_turn_uOhm_cm2",
            "R_joint_nOhm", "scenario",
        ]
        duplicate_count = int(
            final_df.duplicated(primary_key_columns, keep="first").sum()
        )
        if duplicate_count:
            raise RuntimeError(
                f"Scan primary key is not unique: {duplicate_count:,} "
                "extra rows. Concurrent writers are not supported."
            )
        if len(final_df) != total_combinations:
            raise RuntimeError(
                f"Scan row-count mismatch: {len(final_df):,} rows for "
                f"{total_combinations:,} configured combinations."
            )
        # Retain legacy diagnostics, but do not use them as the manuscript's
        # system-level feasibility boundary.
        final_df["feasible_charge_time"] = (
            final_df["Charging_time_999_h"].notna()
            & (final_df["Charging_time_999_h"] <= cfg.charge_hours_max)
        )
        final_df, af_max = apply_system_feasibility(final_df)
        final_df, scenario_min = add_joint_lcoe_reference(final_df)
        print(f"[Info] Scenario-level Aplant maxima: {af_max.to_dict()}")
        print(f"[Info] Joint-feasible LCOE references: {scenario_min.to_dict()}")
    if not final_df.empty:
        # 写回 CSV（补充可行性与ΔLCOE等字段）
        final_df.to_csv(OUTPUT_CSV, index=False)
        # 保存为Excel
        if os.environ.get("WRITE_SCAN_XLSX", "0") == "1":
            xlsx_rows, xlsx_columns = stream_csv_to_xlsx(
                OUTPUT_CSV, OUTPUT_XLSX, sheet_name="scan"
            )
            if (xlsx_rows, xlsx_columns) != (len(final_df) + 1, len(final_df.columns)):
                raise RuntimeError(
                    "XLSX dimension mismatch after streaming export: "
                    f"{xlsx_rows}x{xlsx_columns}"
                )
        else:
            print("[Info] Skipped XLSX export (set WRITE_SCAN_XLSX=1 to enable).")
        print(f"[OK] Results saved:")
        print(f"  - CSV: {OUTPUT_CSV} ({len(final_df):,} rows)")
        if os.environ.get("WRITE_SCAN_XLSX", "0") == "1":
            print(f"  - Excel: {OUTPUT_XLSX} ({len(final_df):,} rows)")
        
        # 打印统计信息
        print(f"\n[Info] Scan summary:")
        print(f"  - Total combinations: {total_combinations:,}")
        print(f"  - Computed: {computed_count:,}")
        print(f"  - Skipped: {skipped_count:,}")
        print(f"  - Status distribution:")
        if 'status' in final_df.columns:
            print(final_df['status'].value_counts().to_string())
    else:
        print("[Warning] No results to save!")

    # 10. Generate the versioned 2025-USD manifest without overwriting plant-v2.
    try:
        write_manifest(
            repo_root=cfg.REPO_ROOT,
            config_path=SCAN_CONFIG_PATH,
            inputs_root=cfg.DATA_RAW_DIR,
            outputs_root=cfg.OUTPUTS_DIR,
            manifest_path=OUTPUT_MANIFEST,
            output_paths=[OUTPUT_CSV] + ([OUTPUT_XLSX] if OUTPUT_XLSX.exists() else []),
            metadata={
                "cost_boundary_version": COST_BOUNDARY_VERSION,
                "parameter_register_version": "Tier1_Tier2_parameter_reference_register_V2",
                "cryo_efficiency_model": econ_params["cryo_efficiency_model"],
                "green_coefficient": hlm.cryo_eff.GREEN_COEFFICIENT,
                "green_exponent": hlm.cryo_eff.GREEN_EXPONENT,
                "cryo_eta_max": econ_params["cryo_eta_max"],
                "cryo_rated_margin": econ_params["cryo_rated_margin"],
                "cryo_part_load_factor": 1.0,
                "non_tf_aux_fraction": econ_params["non_tf_aux_fraction"],
                "monetary_price_basis_year": PRICE_BASIS_YEAR,
                "monetary_values_constant_2025_usd": True,
                "conversion_table_path": str(CONVERSION_TABLE_PATH.relative_to(cfg.REPO_ROOT)),
                "conversion_table_sha256": conversion_table_sha256(),
                "hts_price_2025_USD_per_kAm_by_scenario": dict(HTS_PRICE_2025_USD_PER_KAM_BY_SCENARIO),
                "hts_conductor_price_source_year": 2025,
                "hts_conductor_price_cpi_factor": 1.0,
                "hts_conductor_price_conversion_method": "direct_scenario_assumption_2025usd",
                "coolant_price_2025_USD_per_kg_by_scenario": {
                    key: dict(value)
                    for key, value in COOLANT_PRICE_2025_USD_PER_KG_BY_SCENARIO.items()
                },
                "power_supply_price_2025_USD_per_A": POWER_SUPPLY_PRICE_2025_USD_PER_A,
                "background_capital_USD_by_scenario": dict(
                    BACKGROUND_CAPITAL_USD_BY_SCENARIO
                ),
                "core_vom_USD_per_MWh_th_by_scenario": dict(
                    CORE_VOM_USD_PER_MWH_TH_BY_SCENARIO
                ),
                "pcs_capital_cost_USD_per_kWe": PCS_CAPITAL_COST_USD_PER_KWE,
                "pcs_fom_fraction_per_year": PCS_FOM_FRACTION_PER_YEAR,
                "pcs_vom_USD_per_MWh_e": PCS_VOM_USD_PER_MWH_E,
                "annual_coolant_replenish_fraction": (
                    ANNUAL_COOLANT_REPLENISH_FRACTION
                ),
                "fusion_power_MWth": FUSION_POWER_MWTH,
                "thermal_power_after_blanket_MWth": (
                    THERMAL_POWER_AFTER_BLANKET_MWTH
                ),
                "gross_electric_power_MWe": GROSS_ELECTRIC_POWER_MWE,
                "source_csv": str(OUTPUT_CSV.relative_to(cfg.REPO_ROOT)),
                "source_xlsx": (
                    str(OUTPUT_XLSX.relative_to(cfg.REPO_ROOT))
                    if OUTPUT_XLSX.exists()
                    else None
                ),
            },
        )
        print(f"[OK] Manifest saved: {OUTPUT_MANIFEST}")
    except Exception as e:
        print(f"[Warning] Failed to write {OUTPUT_MANIFEST.name}: {e}")


if __name__ == "__main__":
    main()
