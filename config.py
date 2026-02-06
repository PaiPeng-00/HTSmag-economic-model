# ==============================================================================
# === Global configuration file (config.py) ===
# ==============================================================================
# Place this file at the project root; all other scripts import parameters from here.
from pathlib import Path
import numpy as np

# ----------------------------------------------------------------------
# Repository paths
# ----------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
DATA_RAW_DIR = REPO_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = REPO_ROOT / "data" / "processed"
OUTPUTS_DIR = REPO_ROOT / "outputs"
OUTPUTS_FIGURES_DIR = OUTPUTS_DIR / "figures"
OUTPUTS_TABLES_DIR = OUTPUTS_DIR / "tables"
OUTPUTS_LOGS_DIR = OUTPUTS_DIR / "logs"
CONFIGS_DIR = REPO_ROOT / "configs"

# ------------------------------------------------------------------------------
# 1. Magnet geometry & physics
# ------------------------------------------------------------------------------
# These parameters define the basic geometry of the superconducting magnet.
# They are used in both inductance and resistance calculations.

# Number of pancakes
NP = 16

# Straight‑section length of D‑shaped coil (m)
L1 = 3.0

# Inner small‑arc radius of D‑shaped coil (m)
R1 = 0.4
R1_BASE = R1

# Total radial build (m).
# This is the total radial width of all turns, used to compute inter‑turn spacing.
R2 = 0.3

# Axial thickness of a single pancake (m)
WID = 4e-3

# Axial spacing between pancakes (m)
DIST = 10e-3

# Average length of a single D‑shaped turn (m)
L_TAPE_PER_TURN = (L1/2 + (R1+R2/2)*np.pi/2 + (R1+L1/2+R2/2)*np.pi/2)*2
# Total nominal number of tapes/turns for the system.
# This is repartitioned across physical turns according to Npw.
N_TOTAL_TAPE = 2000  # number of HTS tapes per TF at 20 K
L_tot = L_TAPE_PER_TURN * N_TOTAL_TAPE * NP  # total tape length per TF magnet (m)

# Number of TF magnets
Ntf = 18
S_large = (R1+R2)**2*np.pi/2 + (R1+R2+L1/2)**2*np.pi/2

S_small = (R2)**2*np.pi/2 + (R2+L1/2)**2*np.pi/2

S_magnet = S_large - S_small +R2*L1
H_magnet = (DIST+WID)*NP
V_magnet = S_magnet * H_magnet  # magnet volume (m^3)


# Cryostat surface area (for radiation and conduction)
R_cryostat_bottom = (L1 / 2 + R1 + R2) + R2   # outer radius of cryostat bottom
r_cryostat_bottom = (R1 + R2) + R2           # inner radius of cryostat bottom
# Perimeter of cryostat bottom
C_cryostat_bottom = L1+ np.pi*(R_cryostat_bottom+r_cryostat_bottom) 

A_cryostat_bottom = L1 * (2 * R2 + R1) + (np.pi * R_cryostat_bottom**2 + np.pi * r_cryostat_bottom**2) / 2

H_cryostat = H_magnet * 1.5  # cryostat height, taken as 1.5× magnet height

# Lateral area of cryostat side wall
A_cryostat_side = C_cryostat_bottom * H_cryostat

# Total cryostat surface area
A_cryostat = A_cryostat_bottom * 2 + A_cryostat_side


# ------------------------------------------------------------------------------
# 2. Conductor & material parameters
# ------------------------------------------------------------------------------
# Nominal tape length per pancake (m), used to estimate the number of su‑su joints.
LEN_PER_SINGEL_REBCO = 200.0
# Total tape length per pancake (m)
L_TAPE_PER_PANCAKE = L_TAPE_PER_TURN * N_TOTAL_TAPE

# Intra‑turn (tape‑to‑tape) resistivity (Ω·m²), i.e. contact resistivity
# between parallel tapes within a turn (typically low).
RHO_TAPE = 50e-10
# Total number of cooling pipes in the magnet
N_COOLING_PIPES_PER_PANCAKE = 8
N_COOLING_PIPES = NP * N_COOLING_PIPES_PER_PANCAKE  # 16 pancakes × 8 pipes per pancake

RRR = 100  # copper residual‑resistivity ratio
# ------------------------------------------------------------------------------
# 3. Simulation control parameters
# ------------------------------------------------------------------------------
# These parameters control numerical resolution, ranges and process behaviour.
# -------------------------------------------------------------------------
# Temperature case definitions (for charging comparisons)
# Each case only changes: Ntape_coil (number of parallel tape roots used per coil)
#                         Ip (target current per conductor or per-design)
# Geometry (NP, L1, R1
#, etc.) is unchanged.
# Key is the float temperature in K (use same keys as Ip_list, L_HTS_TAPE_LIST).
# -------------------------------------------------------------------------
# Target operating current Ip at different temperatures
Ip_list = {4.2: 350.0, 10.0: 300.0, 20.0: 210.0}

Nt_list = {4.2: 1200 , 10.0: 1400 , 20.0: 2000  }

# Total tape length for one TF magnet
L_HTS_TF_m = {T: Nt * L_TAPE_PER_TURN * NP for T, Nt in Nt_list.items()}
# Total tape length for the full TF system (km)
L_HTS_SYSTEM_km = {T: L_HTS_TF_m[T] * Ntf *1e-3 for T in L_HTS_TF_m.keys()}

Ic0_cm = 400  # 77 K, 0 T critical current (A/cm) for 4 mm tape; scale as 400 A/cm * 0.4 cm = 160 A
Ic0 = Ic0_cm * WID * 100  # actual Ic (A) for 4‑mm‑wide tape
# kAm requirement for the economic model
kAm_HTS_TAPE_LIST = {T: Ic0 * L for T, L in L_HTS_SYSTEM_km.items()}
TEMPERATURE_CASES = {
    4.2: {
        'label': '4.2K',
        # Number of tapes per coil (may be adjusted per design)
        'Ntape_coil': Nt_list.get(4.2, 1200),
        # Ip_list is defined above; we duplicate here for convenience/overrides
        'Ip': Ip_list.get(4.2, 350.0)
    },
    10.0: {
        'label': '10K',
        'Ntape_coil': Nt_list.get(10.0, 1400),
        'Ip': Ip_list.get(10.0, 300.0)
    },
    20.0: {
        'label': '20K',
        'Ntape_coil': Nt_list.get(20.0, 2000),
        'Ip': Ip_list.get(20.0, 200.0)
    }
}

# --- Inductance calculation parameters ---
# Number of discretisation points per turn
POINTS_PER_TURN = 200
# Grouping number for mutual‑inductance calculation;
# `ng` is determined dynamically from `Npw`.
def get_ng_for_npw(npw) -> int:
    """
    Determine the grouping number `ng` from the number of parallel tapes `Npw`.
    """
    if npw <= 10:
        return 10
    elif npw <= 50:
        return 2
    elif npw <= 100:
        return 1
    else:
        # Default for very large Npw
        return 1

# --- Charging‑simulation parameters ---
# Target current (A) per single conductor
I_TARGET = 500.0
# Charging duration (hours)
CHARGE_HOURS = 96 
# Steady‑state duration (hours)
STEADY_HOURS = 2000 * CHARGE_HOURS

# ------------------------------------------------------------------------------
# 4. Parameter‑scan lists
# ------------------------------------------------------------------------------
# Lists used for parametric studies.

# Experiment 1: scan values of Npw
NPW_LIST_EXP1 = [ 100, 50, 20, 10, 5,2,1]
# Experiment 1: fixed turn‑to‑turn resistivity (Ω·m²)
RHO_TURN_EXP1_FIXED = 50e-10

# Experiment 2: scan values of turn‑to‑turn resistivity (Ω·m²)
RHO_TURN_LIST_EXP2 = [50e-10, 100e-10, 1500e-10, 5000e-10]
# Experiment 2: fixed Npw
NPW_EXP2_FIXED = 15

# Fixed turn‑to‑turn resistivity (µΩ·cm²) used when plotting LCOE.
RHO_TURN_LCOE_UOHM_CM2 = 5000

# 5. Heat‑load analysis parameters

# Coolant enthalpy differences (kJ/kg); values are already in kJ/kg.
ENTHALPY_HE_KJ_KG = 106.13 - 100.18
ENTHALPY_H2_KJ_KG = 24.142 - 15.036

T_HIGH = 300

P_fusion_W = 140e6  # 140 MW thermal fusion output
eat_conv = 0.35     # thermal‑to‑electric conversion efficiency
# ------------------------------------------------------------------------------
# 6. I/O paths
# ------------------------------------------------------------------------------
# Define base directories for all saved results.

# Inductance results for a single TF
INDUCTANCE_OUTPUT_DIR = DATA_RAW_DIR / "inductance"

# Inductance results for the TF system
TF_SYSTEM_INDUCTANCE_OUTPUT_DIR = OUTPUTS_FIGURES_DIR / "inductance"
TF_SYSTEM_MATRIX = "TF_system_L_matrix.xlsx"
# Charging-simulation results
CHARGING_SIM_OUTPUT_DIR = OUTPUTS_TABLES_DIR / "charging"
CHARGING_SIM_OUTPUT_FILE = 'charging_time_summary.xlsx'
CHARGING_SIM_OUTPUT_PLOT_FILE = "fixed_combinations_heatmap.svg"

# TF-system charging-simulation outputs
TF_SYSTEM_CHARGING_SIM_OUTPUT_DIR = OUTPUTS_TABLES_DIR / "charging_tf"
TF_SYSTEM_CHARGING_SIM_OUTPUT_FILE = 'charging_time999_summary_TF_system.xlsx'
TF_SYSTEM_CHARGING_SIM_OUTPUT_PLOT_FILE = 'charging_time999_heatmap_contour_TF_system.svg'
TF_SYSTEM_CHARGING_SIM_OUTPUT_FILE_Npw1_200 = 'charging_time999_summary_TF_system_Npw=1_to_200.xlsx'
TF_SYSTEM_CHARGING_SIM_OUTPUT_PLOT_FILE_Npw1_200 = 'charging_time999_heatmap_contour_TF_system_Npw=1-200.svg'
# Directory for COMSOL ODE exports
COMSOL_ODE_OUTPUT_DIR = OUTPUTS_TABLES_DIR / "comsol"
# ------------------------------------------------------------------------------
# 6. Heat-load analysis parameters (static and dynamic sources)
# ------------------------------------------------------------------------------

# --- Static pipe conduction (W) ---
N_cool_pipe = 2                      # number of cooling pipes
d_in_cool = 0.02                     # cooling-pipe inner diameter (m)
d_out_cool = 0.022                   # cooling-pipe outer diameter (m)
L_cool = 1.0                         # cooling-pipe length (m)

N_aux_pipe = 2                       # number of auxiliary pipes
d_in_aux = 0.01                      # auxiliary-pipe inner diameter (m)
d_out_aux = 0.012                    # auxiliary-pipe outer diameter (m)
L_aux = 1.0                          # auxiliary-pipe length (m)

# --- Cryostat radiation / MLI parameters ---
eps = 0.03                           # emissivity for radiation
N_LAYER_CRYOSTAT = 30                # number of MLI layers around cryostat

# --- Nuclear heating and joint parameters ---

NUCLEAR_POWER_DENSITY = 600.0        # nuclear power density (W/m^3)
VOLUME_M3 = V_magnet                 # volume affected by nuclear heating (m^3)
JOINT_WIDTH = 12.0                   # joint width (mm)
JOINT_LENGTH = 200.0                 # joint length (mm)
JOINT_PRESSURE = 75.0                # joint pressure (MPa)
R_SU_JOINT = 10e-9                   # Su-Su joint resistance (Ω)

# REBCO current-lead heat-load parameters
carrying_factor = 0.7                # current-sharing factor
Time_transfer = 5.0                  # energy-transfer time to stainless steel (s)
Temp_transfer = 100.0                # max transfer-temperature limit (K)
T_HOT_HTS_LEAD = 77.0                # hot-end temperature of REBCO lead (K)
L_LEAD_HTS = 0.5                     # REBCO lead length (m)

# --- Copper-lead parameters ---
design_factor = 5e6                  # I*L/A design factor ≈ 5e6 A/m
INTK_CU30 = 42853.56                 # integral thermal conductivity of Cu at ~30 K (W/m)
LEN_LEAD_CU = 1.0                    # copper length of the lead (m)
N_LEAD = 2                           # number of leads

# --- Other static sources ---
Q_OTHER_SOURCES = 10.0               # lumped other static heat sources (W)


ceff_77to20 = 5                      # factor converting 77 K heat to 20 K equivalent

# --- Time-integration control (h) ---

TOTAL_SIMULATION_TIME_H = 192        # total simulated time (h)
TIME_STEP_H = 1                      # time step (h)

# --- Coolant parameters ---
P0_HE_BAR = 10.0  # initial He pressure (bar)
P0_H2_BAR = 10.0  # initial H2 pressure (bar)
P_REF = 10.0      # reference pressure (bar)

# Maximum operating temperature
MAX_TEMP_RUN = 21

# --- Specific values used for time-series plots ---
NPW_FOR_TIMESERIES_PLOT = [10, 100]       # Npw values for which to plot time series
RHOT_FOR_TIMESERIES_PLOT = [1500, 5000]   # rhot values for which to plot time series

# --- Heat-load I/O paths ---
HEAT_DATA_DIR = DATA_RAW_DIR / "heat_input"  # root folder for heat-load input data
# Subdirectories for different experiments
HEAT_DATA_SUBDIR_NPW = 'variable_Npw'
HEAT_DATA_SUBDIR_RHOT = 'variable_rhot'

HEAT_LOAD_OUTPUT_DIR = OUTPUTS_TABLES_DIR / "heatload"   # output directory for heat-load analysis

LEGEND_IMAGE_FILE = "heat_load_legend.svg"  # standalone legend image
MAG_LOSS_FILE = 'mag_loss.xlsx'            # magnetisation-loss data
RADIAL_LOSS_FILE = 'radial_loss.xlsx'      # radial-loss data
# Preprocessed heat-loss input/output
PREPROCESSED_DATA_DIR = DATA_PROCESSED_DIR / "heat_input"
PREPROCESSED_FILENAME = 'preprocessed_loss_data.xlsx'
# ------------------------------------------------------------------------------
# 7. Global plotting output format
#  Define global output settings for all figures.
# ------------------------------------------------------------------------------
PLOT_FORMAT = 'svg'
# Unified image resolution for all plt.savefig calls.
# Keep high resolution; control export format via file extension (SVG/PDF).
PLOT_DPI = 1800
PLOT_TRANSPARENT = True

# 8. Cooldown analysis parameters
COOLDOWN_DATA_DIR = DATA_RAW_DIR / "cooldown"
COOLDOWN_PROFILES_ALL_FILE = 'cooldown_profiles.xlsx' 
COOLDOWN_OUTPUT_DIR = OUTPUTS_TABLES_DIR / "cooldown"

# 9. Charging performance analysis parameters
CHARGE_DATA_DIR = DATA_RAW_DIR / "charge"
CHARGE_PROFILE_ALL_FILE = 'charge_profile.xlsx'
CHARGE_SUMMARY_FILE = 'charge_performance_summary.xlsx'
CHARGE_OUTPUT_DIR = OUTPUTS_TABLES_DIR / "charge"

# 10. Operation performance analysis parameters
OPERATION_DATA_DIR = DATA_RAW_DIR / "operation"
OPERATION_PROFILE_ALL_FILE = 'operation_profile.xlsx'
OPERATION_SUMMARY_FILE = 'operation_performance_summary.xlsx'
OPERATION_OUTPUT_DIR = OUTPUTS_TABLES_DIR / "operation"

# 11. Quench analysis parameters
QUENCH_DATA_DIR = DATA_RAW_DIR / "quench"

QUENCH_OUTPUT_DIR = OUTPUTS_TABLES_DIR / "quench"
QUENCH_SUMMARY_FILE = 'quench_summary_all.xlsx'

# Separate data files for two quench experiments
QUENCH_FILE_VARY_QM = 'quench_profile_varible=qm_max_step.xlsx'
QUENCH_FILE_VARY_PQ = 'quench_profile_varible=Pquench_max_step.xlsx'

# Time-window definition for quench events (s)
QUENCH_START_S = 1000.0
QUENCH_END_S = 2000.0
SIMULATION_END_S = 3000.0

# Steady-state temperature criteria
STABLE_TEMP_START_S = 100.0  # average from t >= 100 s
STABLE_TEMP_RECOVERY_FACTOR = 1.05  # within 105% of steady-state temperature


#12. Scaling analysis
SCALING_DATA_DIR = DATA_RAW_DIR / "scaling_input"
SCALING_OUTPUT_DIR = OUTPUTS_TABLES_DIR / "scaling"
SCALING_SUMMARY_FILE = 'scaling_summary_all.xlsx'

# 14. Economic benefit / LCOE analysis
ECONOMIC_DATA_DIR = DATA_RAW_DIR / "economic_input"
ECONOMIC_OUTPUT_DIR = OUTPUTS_TABLES_DIR / "economic"
ECONOMIC_FIGURES_DIR = OUTPUTS_FIGURES_DIR / "economic"

PARASITIC_RATIO_OUTPUT_DIR = 'parasitic_ratio'
# Heatmap subdirectories: USD, CNY and intermediate Excel data
PARASITIC_RATIO_USD_DIR = 'USD'
PARASITIC_RATIO_CNY_DIR = 'CNY'
PARASITIC_RATIO_DATA_DIR = 'data'
# Available colormaps for heatmaps
color_schemes = [
    "viridis", "plasma", "inferno", "magma", "cividis",
    "Blues", "YlGnBu", "PuBuGn", "mako", "rocket"
]
# Active subset of colormaps used for main heatmaps.
# "YlGnBu_trunc_0p8" means YlGnBu truncated to [0.0, 0.8] in the original map
# (compressing the darkest end) while keeping the scheme consistent.
color_schemes = ["cividis", "Blues", "YlGnBu", "YlGnBu_trunc_0p8"]


def resolve_cmap(cmap):
    """
    Resolve a config colormap spec into a matplotlib Colormap object.
    - If a string is passed: support built-in/Seaborn colormap names as well
      as project-specific truncated variants.
    - If a Colormap is passed: return it unchanged.
    """
    try:
        import matplotlib as _mpl  # local import to avoid hard top-level dependency
        if isinstance(cmap, _mpl.colors.Colormap):
            return cmap
    except Exception:
        # If matplotlib is unavailable, just return the original spec
        return cmap

    if not isinstance(cmap, str):
        return cmap

    # --- Custom truncated colormap: YlGnBu over [0.0, 0.8] ---
    if cmap == "YlGnBu_trunc_0p8":
        try:
            import numpy as _np
            import matplotlib.pyplot as _plt
            from matplotlib.colors import ListedColormap as _ListedColormap

            _cmap_full = _plt.get_cmap("YlGnBu")
            _colors = _cmap_full(_np.linspace(0.0, 0.8, 256))
            return _ListedColormap(_colors, name="YlGnBu_trunc_0p8")
        except Exception:
            # Fall back to the original YlGnBu if anything fails
            return "YlGnBu"

    # Default: let the caller handle it (plt.get_cmap / seaborn fallback)
    return cmap

# Active colormap for parasitic-ratio plots
cmap_parasitic_ratio = "cividis"
# LCOE contour-selection method: 'adaptive' (data-driven) or 'fixed' (pre-set levels)
LCOE_CONTOUR_METHOD = 'fixed'
# Fixed LCOE contour levels used when LCOE_CONTOUR_METHOD='fixed'.
# Seeded from user-specified values and extended to cover a broad range
# with minimum spacing >= 0.1.
LCOE_CONTOUR_FIXED_LEVELS = np.array([
    -40, -20, -10, -5, -3, -2, -1, -0.5, -0.3, -0.2, -0.1, -0.05,
    0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
    1, 1.2, 1.4, 1.6,
    2, 2.5, 3, 5, 10, 12.5, 13, 14, 15, 16, 17, 18, 19, 20,
    25, 26, 28, 30, 32, 34, 36, 38, 40,
    65, 70, 80, 90, 100, 120, 140, 160, 180, 200
])
# Fixed LCOE contour levels in CNY/kWh (for CNY heatmaps)
LCOE_CONTOUR_FIXED_LEVELS_CNY = np.array([
    0, 0.001, 0.002, 0.003, 0.004, 0.004, 0.005,
    0.01, 0.016, 0.02, 0.03, 0.04, 0.05,
    0.09, 0.1, 0.12, 0.13, 0.14, 0.16,
    0.2, 0.3, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7,
    1, 1.2, 1.4, 1.6
])

# Global configuration for ΔLCOE
# Points with >50% parasitic fraction are considered infeasible and masked.
DELTA_LCOE_MASK_PARASITIC_PCT = 50.0
# Legacy: percentile used when clipping |ΔLCOE| to avoid extreme outliers.
DELTA_LCOE_ABS_PCTL = 99.5
# Exchange rate USD→CNY (for LCOE unit conversion)
USD_TO_CNY_EXCHANGE_RATE = 7.2

# Output folder for cost-analysis results
COST_OUTPUT_DIR = 'cost_output'

ECONOMIC_SUMMARY_FILE = f'cost&heat_summary_all.xlsx'
ECONOMIC_SUMMARY_FIG = 'cost_summary_all.svg'
HOURS_PER_YEAR = 8760  # hours per year
# Annual operating schedule

pulse_hours = 2.0             # production-pulse duration (h)
dwell_hours = 10.0 / 60.0     # dwell/charge duration (h); default 10 min
maintenance_hours_per_year = 24 * 30 * 2  # default: 2 months = 1440 h
nuclear_decay_factor = 0.1   # decay factor for nuclear heating during dwell

# Reference point for cost analysis
Npw_TARGET = 50              # target number of parallel tapes
R_p2p_joint_TARGET = 10e-9   # target joint resistance between pancakes (Ω)
charge_hours_max = 24*5      # target max charge time (5 days)

# Power-supply cost
power_supply_price_perA = 1e4/500.0  # per-amp price (e.g. 500 A supply ≈ 10 k$)

# Effective annual production hours
production_hours_per_year = (HOURS_PER_YEAR-maintenance_hours_per_year)*pulse_hours/(pulse_hours+dwell_hours )

# --- Operating-scenario definitions (shared by economic and AF scripts) ---
SCENARIO_DEFINITIONS = {
    "S1": {
        "label": "S1",
        "tmaint_h": 1440.0,
        "nmaint": 1.0,
        "tcool_h": 12 * 24.0,
        "twarm_h": 6 * 24.0,
        "kdis": 0.8,
        "tau_pulse_h": 1,
        "tau_dwell_h": 1,
    },
    "S2": {
        "label": "S2",
        "tmaint_h": 720.0,
        "nmaint": 1.0,
        "tcool_h": 12 * 24.0,
        "twarm_h": 6 * 24.0,
        "kdis": 0.8,
        "tau_pulse_h": 1,
        "tau_dwell_h": 0.25,
    },
    "S3": {
        "label": "S3",
        "tmaint_h": 720.0,
        "nmaint": 0.5,
        "tcool_h": 12 * 24.0,
        "twarm_h": 6 * 24.0,
        "kdis": 0.8,
        "tau_pulse_h": 1.0,
        "tau_dwell_h": 0.063,
    },
    "S4": {
        "label": "S4",
        "tmaint_h": 1440.0,   # same as S1
        "nmaint": 1.0,        # same as S1
        "tcool_h": 12 * 24.0, # same as S1
        "twarm_h": 6 * 24.0,  # same as S1
        "kdis": 0.8,          # same as S1
        "tau_pulse_h": 1,     # same as S1
        "tau_dwell_h": 1,     # same as S1
    },
    "S5": {
        "label": "S5",
        "tmaint_h": 720.0,    # same as S2
        "nmaint": 1.0,        # same as S2
        "tcool_h": 12 * 24.0, # same as S2
        "twarm_h": 6 * 24.0,  # same as S2
        "kdis": 0.8,          # same as S2
        "tau_pulse_h": 1,     # same as S2
        "tau_dwell_h": 0.25,  # same as S2
    },
    "S6": {
        "label": "S6",
        "tmaint_h": 720.0,    # same as S2
        "nmaint": 1.0,        # same as S2
        "tcool_h": 12 * 24.0, # same as S2
        "twarm_h": 6 * 24.0,  # same as S2
        "kdis": 0.8,          # same as S2
        "tau_pulse_h": 1,     # same as S2
        "tau_dwell_h": 0.25,  # same as S2
    },
}

N_facility_list = range(1,51)  # number of facilities at one site sharing a cryoplant (1–50)
N_cooldown_per_year = 0        # number of cooldowns per year (0: remain cold during maintenance)
E_cooldown_Wh = 0              # energy per cooldown event (Wh)

# =============================================================================
# 15. Economic-analysis parameter-scan configuration
# =============================================================================
# R_joint values for plotting: log-spaced array matching the log-scale axis,
# spanning 1–100 nOhm.
R_JOINT_SCAN_VALUES = np.array([
    1.000, 1.166, 1.359, 1.585, 1.848, 2.154,
    2.512, 2.929, 3.415, 3.981, 4.642, 5.412,
    6.310, 7.356, 8.577, 10.000, 11.659, 13.594,
    15.849, 18.478, 21.544, 25.119, 29.286, 34.145,
    39.811, 46.416, 54.117, 63.096, 73.564, 85.770,
    100.000
])  # nOhm
# Npw scan values, e.g. [1, 10, 20, 30, ..., 200]
# NPW_SCAN_VALUES = np.concatenate([[1], np.arange(10, 201, 10)])
NPW_SCAN_VALUES = np.concatenate([np.arange(1, 21, 1), np.arange(10, 201, 10)])

# Temperature–coolant pairs used in the parameter scan
SCAN_TEMP_COOLANT_PAIRS = [(4.2, "He"), (10.0, "He"), (20.0, "He"), (20.0, "H2")]

# Operating conditions used in economic analysis
ECONOMIC_OPERATING_CONDITIONS = [(4.2, "He"), (10.0, "He"), (20.0, "He"), (20.0, "H2")]

# Global reference LCOE point (for global relative-LCOE plots)
GLOBAL_REFERENCE_SCENARIO = 'S3'
GLOBAL_REFERENCE_TEMP = 20.0
GLOBAL_REFERENCE_COOLANT = 'H2'

# SymLogNorm parameters used for ΔLCOE heatmaps
DELTA_LCOE_LINTHRESH_USD = 0.1   # linear threshold in USD/MWh
DELTA_LCOE_LINSCALE_USD = 0.5    # linear scale factor in USD/MWh
DELTA_LCOE_LINTHRESH_CNY = 0.001 # linear threshold in CNY/kWh
DELTA_LCOE_LINSCALE_CNY = 0.5    # linear scale factor in CNY/kWh

# Contour-level selection parameters
CONTOUR_TARGET_COUNT = 5         # target number of contour levels per subplot
CONTOUR_MIN_SPACING_USD = 0.1   # minimum contour spacing in USD/MWh
CONTOUR_MIN_SPACING_CNY = 0.001 # minimum contour spacing in CNY/kWh

# Stroke (halo) settings for contours and labels
USE_STROKE = True                      # enable stroke (black contour + white halo)
CONTOUR_STROKE_LINEWIDTH = 4           # stroke linewidth for regular contours
CONTOUR_STROKE_LINEWIDTH_ZERO = 5      # stroke linewidth for zero contour (thicker)
CONTOUR_STROKE_LINEWIDTH_AF = 2.0      # stroke linewidth for AF heatmaps
CONTOUR_STROKE_FOREGROUND = 'white'    # stroke color
CONTOUR_STROKE_ALPHA = 0.9             # stroke alpha
LABEL_STROKE_LINEWIDTH = 4             # label stroke linewidth
LABEL_STROKE_FOREGROUND = 'white'      # label stroke color
LABEL_STROKE_ALPHA = 0.8               # label stroke alpha

# Heatmap transparency
HEATMAP_ALPHA = 0.8  # 0.0–1.0

# Heatmap font sizes
HEATMAP_FONT_SIZE = 18             # main text (contour labels, axis labels, etc.)
HEATMAP_FONT_SIZE_TICK = 20        # tick-label font size
HEATMAP_FONT_SIZE_TITLE = 24       # title font size
HEATMAP_FONT_SIZE_COLORBAR = 18    # colorbar label font size
HEATMAP_FONT_SIZE_COLORBAR_TITLE = 24  # colorbar title font size

# Heatmap figure sizes (width, height)
HEATMAP_FIGSIZE_SINGLE = (5, 5)          # single heatmap
HEATMAP_FIGSIZE_GRID = (15, 5)           # heatmap grid
HEATMAP_FIGSIZE_COLORBAR = (0.5, 6)      # colorbar-only figure
HEATMAP_FIGSIZE_COLORBAR_LARGE = (3.0, 6.0)  # large colorbar figure

# Heatmap contour linewidths
HEATMAP_CONTOUR_LINEWIDTH = 2           # regular contours
HEATMAP_CONTOUR_LINEWIDTH_ZERO = 3      # zero-contour linewidth (thicker)
HEATMAP_CONTOUR_LINEWIDTH_AF = 2        # AF heatmap contours
HEATMAP_CONTOUR_LINEWIDTH_CRYO = 2      # cryogenic heatmap contours

# Heatmap contour-label formats
HEATMAP_CONTOUR_FMT_PARASITIC = "%.1f"  # parasitic-power contour labels
HEATMAP_CONTOUR_FMT_LCOE = "%.2f"       # LCOE contour labels
HEATMAP_CONTOUR_FMT_AF = "%.2f"         # AF heatmap contour labels
HEATMAP_CONTOUR_FMT_CRYO = "%.2f"       # cryogenic contour labels






