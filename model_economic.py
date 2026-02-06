# economic_model.py
import pandas as pd
import numpy as np
import itertools
from pathlib import Path
from tabulate import tabulate
import config as cfg
from typing import Any, Dict

# Import the refactored cryogenic / heat‑load model
import model_cryogenic as hlm 

# =============================================================================
# 1. Define scenarios and parameters
# =============================================================================
def crf(r: float, N: float) -> float:
    """Capital Recovery Factor."""
    if r <= 0:
        return 1.0 / N
    return r * (1 + r) ** N / ((1 + r) ** N - 1)

def define_parameters():
    """Define all parameters required by the economic model."""
    # 1) Coolant densities at different operating conditions (kg/m^3)
    coolant_density_kg_m3 = {
        'He': {4.2: 151.15, 10.0: 61.273, 20.0: 24.244},
        'H2': {20.0: 72.405}
    }
    
    # 2) Convert all coolant prices to $/kg.
    # Helium price conversion: reference density is liquid He at 4.2 K,
    # 151.15 kg/m^3 = 0.15115 kg/L.
    # Price ($/kg) = price ($/L) / density (kg/L).
    price_He_2025_per_kg = 50 / 0.15115
    price_He_2035_baseline_per_kg = 80 / 0.15115
    price_He_2035_optimistic_per_kg = 100 / 0.15115
    print(f"s1 He: {int(price_He_2025_per_kg)}; s2 He: {int(price_He_2035_baseline_per_kg)}; s3 He: {int(price_He_2035_optimistic_per_kg)}")

    # All coolant prices are in $/kg
    coolant_price_per_kg_by_scenario = {
        'S1': {'He': price_He_2025_per_kg, 'H2': 9},
        # S2: conservative helium price, S3: optimistic, S4–S6 share other
        # parameters with S1/S2 but differ in HTS tape price (see below).
        'S2': {'He': price_He_2035_baseline_per_kg, 'H2': 6.0},
        'S3': {'He': price_He_2035_optimistic_per_kg, 'H2': 3},
        'S4': {'He': price_He_2025_per_kg, 'H2': 9},
        'S5': {'He': price_He_2035_baseline_per_kg, 'H2': 6.0},
        'S6': {'He': price_He_2035_baseline_per_kg, 'H2': 6.0},
    }
    params = {
        # --- Basic scenario settings ---
        # Project lifetime (years) is tied to the technology scenario:
        # S1 → 20y, S2 → 30y, S3 → 50y, S4 → 50y, S5 → 30y (same as S2),
        # S6 → 30y (same as S2).
        'tech_scenario_to_years': {
            'S1': 20,
            'S2': 30,
            'S3': 50,
            'S4': 50,  # S4 uses S1's project lifetime
            'S5': 30,  # S5 uses S2's project lifetime
            'S6': 30,  # S6 uses S2's project lifetime
        },
        # Technology scenarios; S4/S5/S6 are HTS tape cost sensitivity cases.
        'tech_scenarios': ['S1', 'S2', 'S3', 'S4', 'S5', 'S6'],
        'temperatures_K': [4.2, 10, 20],  # compare 4.2 K, 10 K, 20 K
        'coolants': ['He', 'H2'],         # compare He and H2

        # --- Magnet / coolant parameters ---
        # Operating current depends on temperature.
        'current_by_temperature': cfg.Ip_list,
        # Coolant density at 4.2 K, 10 K, 20 K (see `coolant_density_kg_m3`)
        'coolant_density_kg_m3': coolant_density_kg_m3,
        'coolant_price_per_kg_by_scenario': coolant_price_per_kg_by_scenario,
        # --- Economic parameters ---
        'power_price_per_MWh_by_scenario': {
            # Electricity price (all scenarios currently share the same value)
            'S1': 127.0, 'S2': 127.0, 'S3': 127.0,
            'S4': 127.0, 'S5': 127.0, 'S6': 127.0,
        },

        # HTS tape price in $/kAm: compare 2025, 2035 baseline, and 2035 optimistic.
        # S4: same as S1 except tape price uses S3 (sensitivity).
        # S5: same as S2 except tape price uses S1 (sensitivity).
        # S6: same as S2 except tape price uses S3 (sensitivity).
        'hts_price_per_kAm_by_scenario': {
            'S1': 100.0, 'S2': 50.0, 'S3': 10.0,
            'S4': 10.0,  # uses S3 tape price
            'S5': 100.0, # uses S1 tape price
            'S6': 10.0,  # uses S3 tape price
        },

        # --- Physical and fixed‑cost parameters ---
        'fusion_output_MWth': 140,          # thermal fusion output (MWth)
        'other_recirc_fraction': 0.1,       # other recirculating electrical fraction
        # Cryogenic loop volume: 1.5 m^3 per TF, annual replenish fraction below.
        'cryo_loop_volume_m3': 1.5 * cfg.Ntf,
        'annual_coolant_replenish_fraction': 0.25,
        # Power‑supply unit price (per A); e.g. a 500 A supply ≈ 25 k$ (2025).
        'power_supply_unit_price': cfg.power_supply_price_perA,

        # HTS tape requirement at each temperature (kAm)
        'hts_kAm_by_temperature': cfg.kAm_HTS_TAPE_LIST,

        # OPEX components
        # 1) Power‑conversion system (PCS) parameters [see paper cites 166, 185–187]
        'pcs_params': {
            'efficiency': 0.35,              # thermal‑to‑electric efficiency
            'capital_cost_per_kWe': 750,     # PCS CAPEX per kWe
            'fom_fraction_of_capex': 0.025,  # fixed O&M fraction of CAPEX
            'vom_per_MWh_e': 1.74,           # variable O&M per MWh_e
        },
        # 2) Core component replacement cost (VOM_th), scenario‑dependent
        'core_vom_th_per_MWh_th': {
            # O&M cost per MWh_th of fusion heat; S4 shares S1, S5/S6 share S2.
            'S1': 5.0, 'S2': 3.0, 'S3': 1.0,
            'S4': 5.0, 'S5': 3.0, 'S6': 3.0,
        } ,
        # --- Operating‑time / scenario parameters (see documentation) ---
        'time_scenarios': {
            key: {k: v for k, v in scenario.items() if k != 'label'}
            for key, scenario in cfg.SCENARIO_DEFINITIONS.items()
        },
        # Approximate upper bound of cryogenic power (W) during charge/discharge
        # (excitation/de‑excitation) phases, total system, fixed at 500 W.
        'charge_stage_power_cap_W': 500.0,

        # --- Financial / scaling parameters for ΔLCOE (magnet‑only) ---
        "discount_rate": 0.08,                 # r
        "magnet_installed_multiplier": 1.0,    # installed‑cost multiplier (magnet CAPEX only)
        "contingency_fraction": 0.0,           # FOAK contingency (CAPEX × (1 + contingency))
        }
    return params

# =============================================================================
# 2. Core calculation module
# =============================================================================
def compute_case(tech_scenario: str, year, temperature_K: float, coolant: str,
                 Npw, R_p2p_joint: float, p: dict) -> dict:

    """
    Perform a full techno‑economic calculation for a single parameter point.

    Returns a tuple (economic_result_dict, base_heat_loads_dict).
    """
    pcs_params = p['pcs_params']
    Ip = p['current_by_temperature'][temperature_K]  # operating current per tape

    # --- Step 1: use the physics model to compute parasitic power ---
    # 1a. Compute base heat loads for the given magnet parameters.
    base_heat_loads = hlm.calculate_base_heat_loads(
    Ip=Ip, Npw=Npw, R_p2p_joint=R_p2p_joint, Top=temperature_K)
    # 1b. Cryogenic electrical power (W) in three operating modes
    P_cryo_prod_W = hlm.calculate_cryo_electrical_power('prod', base_heat_loads, temperature_K)
    P_cryo_dwell_W = hlm.calculate_cryo_electrical_power('dwell', base_heat_loads, temperature_K)
    P_cryo_static_W = hlm.calculate_cryo_electrical_power('static', base_heat_loads, temperature_K)

    
    # --- Step 2: annual energy balance (time/heat decomposed by scenario) ---
    # 2a. Read the time parameters for this technology scenario
    ts = p['time_scenarios'][tech_scenario]
    tmaint_h  = float(ts['tmaint_h'])
    nmaint    = float(ts['nmaint'])
    tcool_h   = float(ts['tcool_h'])
    twarm_h   = float(ts['twarm_h'])
    kdis      = float(ts['kdis'])
    tau_pulse = float(ts['tau_pulse_h'])
    tau_dwell = float(ts['tau_dwell_h'])

    # Magnet charge time t_charge should ideally use the 99.9% charge‑time
    # table; here we approximate by `cfg.charge_hours_max` while keeping the
    # external interface compatible. If a more detailed coupling to
    # `charge_time999` tables is added later, this is the place to wire it in.
    tcharge_h = cfg.charge_hours_max

    # 2b. Time decomposition (maintenance / cool‑down + warm‑up /
    #     excitation + de‑excitation / production / dwell)
    H_maint    = nmaint * tmaint_h
    H_coolwarm = nmaint * (tcool_h + twarm_h)
    H_excdec   = nmaint * (tcharge_h * (1.0 + kdis))

    remaining  = cfg.HOURS_PER_YEAR - H_maint - H_coolwarm - H_excdec
    if remaining < 0: 
        remaining = 0.0
    tcycle     = tau_pulse + tau_dwell
    n_cycles   = int(remaining // tcycle) if tcycle > 0 else 0
    H_prod     = n_cycles * tau_pulse
    H_dwell    = n_cycles * tau_dwell

    # 2c. Annual cryogenic electricity consumption (MWh).
    # Assumptions:
    # - Maintenance: no electrical consumption.
    # - Cool‑down / warm‑up: use the static‑mode cryogenic load.
    # - Excitation / de‑excitation: fixed 500 W upper bound, representing
    #   additional power for control / eddy current etc.
    P_cryo_maint_W = 0
    P_cryo_coolwarm_W = P_cryo_static_W
    P_cryo_excdec_W   = p.get('charge_stage_power_cap_W', 500.0)

    E_cryo_year_MWh = (
        (P_cryo_prod_W     * H_prod)     +
        (P_cryo_dwell_W    * H_dwell)    +
        (P_cryo_maint_W    * H_maint)    +
        (P_cryo_coolwarm_W * H_coolwarm) +
        (P_cryo_excdec_W   * H_excdec)
    ) / 1e6
        
    # 2d. Annual gross and net electricity generation (MWh)
    gross_mw_e = p['fusion_output_MWth'] * pcs_params['efficiency']
    gross_mwh_per_year = gross_mw_e * H_prod  # electricity only during production
    other_recirc_mwh = (gross_mw_e * cfg.HOURS_PER_YEAR) * p['other_recirc_fraction']
    net_mwh_per_year = gross_mwh_per_year - E_cryo_year_MWh - other_recirc_mwh
    
    if net_mwh_per_year <= 0:
        fallback_result = {
            # --- Input scenario description ---
            "tech_scenario": tech_scenario, 
            "project_years": year, 
            "temperature_K": temperature_K, 
            "coolant": coolant,
            "Npw": Npw,
            "R_p2p_joint": R_p2p_joint,
            # --- Parasitic power ---
            "cryo_power_prod_W": P_cryo_prod_W,
            "cryo_power_dwell_W": P_cryo_dwell_W,
            "cryo_power_static_W": P_cryo_static_W,
            # --- Final economic indicator (parasitic fraction only) ---
            "r_parasitic_pct": (E_cryo_year_MWh / gross_mwh_per_year) * 100 if gross_mwh_per_year > 0 else np.inf
        }
        return fallback_result, base_heat_loads
    
    r_parasitic_pct = (E_cryo_year_MWh / gross_mwh_per_year) * 100 if gross_mwh_per_year > 0 else np.inf

    # --- Step 3: magnet‑related capital costs (CAPEX) ---

    # 3a. HTS tape cost
    hts_kAm = p["hts_kAm_by_temperature"][temperature_K]
    hts_unit_price = p["hts_price_per_kAm_by_scenario"][tech_scenario]
    tape_cost = hts_kAm * hts_unit_price

    # 3b. Coolant: initial fill CAPEX + annual replenishment OPEX
    # Coolant density at this operating point (kg/m^3)
    density = p["coolant_density_kg_m3"][coolant][temperature_K]
    # Total mass required to fill the cryogenic loop (kg)
    required_mass_kg = p["cryo_loop_volume_m3"] * density
    # Price for this coolant in this scenario ($/kg)
    price_per_kg = p["coolant_price_per_kg_by_scenario"][tech_scenario][coolant]
    # Single‑fill cost and lifetime coolant cost
    coolant_fill_cost = required_mass_kg * price_per_kg
    coolant_replenish_cost_per_year = (
        required_mass_kg * p["annual_coolant_replenish_fraction"] * price_per_kg
    )
    coolant_lifetime_cost = coolant_fill_cost + coolant_replenish_cost_per_year * year

    # 3c. Power‑supply cost, estimated from maximum current
    # Per‑TF supply cost ≈ (Ip / carrying_factor) * unit_price * Npw
    power_supply_max_current = Ip / cfg.carrying_factor 
    power_supply_cost = power_supply_max_current * cfg.power_supply_price_perA * Npw 

    # 3d. Direct CAPEX for the magnet system
    capex_mag_direct = tape_cost + coolant_fill_cost + power_supply_cost
    # Optional: installed‑cost multiplier and FOAK contingency (currently zero)
    installed_multiplier = float(p.get("magnet_installed_multiplier", 1.0))
    contingency_fraction = float(p.get("contingency_fraction", 0.0))
    capex_mag_installed = capex_mag_direct * installed_multiplier * (1.0 + contingency_fraction)


    # 3f. PCS capital investment; currently only tracked, not varied by magnet
    pcs_capital_cost = pcs_params['capital_cost_per_kWe'] * gross_mw_e * 1000
    # total_capex = tape_cost + coolant_fill_cost + power_supply_cost + pcs_capital_cost

    # --- Step 4: economic indicators ---

    r = float(p.get("discount_rate", 0.08))
    crf_val = crf(r, float(year))

    annualized_capex_mag = capex_mag_installed * crf_val
    annual_opex_mag = coolant_replenish_cost_per_year  # currently only coolant replenishment

    lcoe_magnet_only = (annualized_capex_mag + annual_opex_mag) / net_mwh_per_year

    annual_revenue = net_mwh_per_year * p["power_price_per_MWh_by_scenario"][tech_scenario]
    
    # Total annual OPEX (core + PCS)
    vom_core_cost = (p['fusion_output_MWth'] * H_prod) * p['core_vom_th_per_MWh_th'][tech_scenario]
    vom_pcs_cost = gross_mwh_per_year * pcs_params['vom_per_MWh_e']
    fom_pcs_cost = pcs_capital_cost * pcs_params['fom_fraction_of_capex']
    annual_opex = vom_core_cost + vom_pcs_cost + fom_pcs_cost
    
    net_annual_profit = annual_revenue - annual_opex
    simple_payback_years = capex_mag_installed / net_annual_profit if net_annual_profit > 0 else np.inf
    lifetime_revenue = (net_annual_profit * year) - capex_mag_installed

    economic_result = {
        # --- Input description ---
        "tech_scenario": tech_scenario, 
        "project_years": year, 
        "temperature_K": temperature_K, 
        'Ip': Ip,
        "coolant": coolant,
        "Npw": Npw,
        "R_p2p_joint": R_p2p_joint,
        # --- Economic input parameters ---
        "input_power_price_$/MWh": p["power_price_per_MWh_by_scenario"][tech_scenario],
        "input_hts_price_$/kAm": hts_unit_price,
        # --- Coolant cost parameters ---
        "input_coolant_price_$/kg": price_per_kg,
        "coolant_density_kg/m3": density,
        "coolant_required_mass_kg": required_mass_kg,
        "coolant_fill_cost_$": coolant_fill_cost,
        "coolant_lifetime_cost_$": coolant_lifetime_cost,
        # --- Parasitic power and timing ---
        "cryo_power_prod_W": P_cryo_prod_W,
        "cryo_power_dwell_W": P_cryo_dwell_W,
        "cryo_power_static_W": P_cryo_static_W,
        "annual_hours_prod": H_prod,
        "annual_hours_dwell": H_dwell,
        "annual_hours_maint": H_maint,
        
        "annual_hours_coolwarm": H_coolwarm,
        "annual_hours_excdec": H_excdec,
        "cryo_power_coolwarm_W": P_cryo_coolwarm_W,
        "cryo_power_excdec_W": P_cryo_excdec_W,

        "cryo_energy_annual_MWh": E_cryo_year_MWh,
        "r_parasitic_pct": r_parasitic_pct,
        # --- Energy balance ---
        "gross_power_output_MWe": gross_mw_e,
        "gross_energy_annual_MWh": gross_mwh_per_year,
        "net_energy_annual_MWh": net_mwh_per_year,
        # --- Cost breakdown (magnet system only) ---
        "input_hts_price_$/kAm": hts_unit_price,
        "tape_cost_$": tape_cost,

        "input_coolant_price_$/kg": price_per_kg,
        "coolant_density_kg/m3": density,
        "coolant_required_mass_kg": required_mass_kg,
        "coolant_fill_cost_$": coolant_fill_cost,
        "coolant_replenish_cost_$/year": coolant_replenish_cost_per_year,
        "coolant_lifetime_cost_$": coolant_lifetime_cost,

        "power_supply_cost_$": power_supply_cost,

        "magnet_installed_multiplier": installed_multiplier,
        "contingency_fraction": contingency_fraction,
        "capex_mag_direct_$": capex_mag_direct,
        "capex_mag_installed_$": capex_mag_installed,
        # --- LCOE (magnet‑attributed only) ---
        "discount_rate": r,
        "crf": crf_val,
        "annualized_capex_mag_$/year": annualized_capex_mag,
        "annual_opex_mag_$/year": annual_opex_mag,
        "lcoe_magnet_only_$/MWh": lcoe_magnet_only,
        # --- Economic indicators ---
        "annual_revenue_$": annual_revenue,
        "annual_opex_$": annual_opex,
        "net_annual_profit_$": net_annual_profit,
        "simple_payback_years": simple_payback_years,
        "lifetime_revenue_$": lifetime_revenue,
          }
    return economic_result, base_heat_loads  # base_heat_loads propagated for convenience


# =============================================================================
# 3. Driver to run all cases and assemble a table
# =============================================================================
def run_all_cases(p: dict) -> pd.DataFrame:
    """
    Loop over all parameter combinations and run the economic model.

    Note:
        Project lifetime is coupled to the technology scenario
        (S1→20y, S2→30y, S3→50y, etc.; see `tech_scenario_to_years`).
    """
    rows = []
    # Use the scenario‑specific lifetime instead of scanning over all lifetimes.
    for tech in p["tech_scenarios"]:
        year = p["tech_scenario_to_years"][tech]
        param_combinations = itertools.product(
            [tech], [year],  # each scenario uses its corresponding project lifetime
            p["temperatures_K"], p["coolants"],
            p["Npw"], p["R_p2p_joint"])

        for tech_scenario, year_val, temp, cool, Npw, R_p2p_joint in param_combinations:
            if cool == "H2" and temp != 20: continue
            result, base_heat_loads = compute_case(tech_scenario, year_val, temp, cool, Npw, R_p2p_joint, p)
            if result: rows.append(result)
    df = pd.DataFrame(rows)
    df = df.sort_values(by=["tech_scenario", "project_years", "temperature_K", "coolant"]).reset_index(drop=True)
    return df

# =============================================================================
# 4. A→B ΔLCOE (magnet‑attributed)
# =============================================================================
def delta_lcoe_magnet_only(
    caseA: Dict[str, Any],
    caseB: Dict[str, Any],
    p: dict,
) -> Dict[str, float]:
    """
    Compute ΔLCOE and incremental LCOE attributable to the magnet.

    Parameters
    ----------
    caseA, caseB : dict
        Must contain the `compute_case` input keys:
          `tech_scenario`, `year`, `temperature_K`, `coolant`,
          `Npw`, `R_p2p_joint`.
    """
    resA, _ = compute_case(
        caseA["tech_scenario"],
        caseA["year"],
        caseA["temperature_K"],
        caseA["coolant"],
        caseA["Npw"],
        caseA["R_p2p_joint"],
        p,
    )
    resB, _ = compute_case(
        caseB["tech_scenario"],
        caseB["year"],
        caseB["temperature_K"],
        caseB["coolant"],
        caseB["Npw"],
        caseB["R_p2p_joint"],
        p,
    )

    lcoeA = float(resA.get("lcoe_magnet_only_$/MWh", np.nan))
    lcoeB = float(resB.get("lcoe_magnet_only_$/MWh", np.nan))

    # Incremental LCOE (marginal cost / marginal net energy)
    ann_cost_A = float(resA.get("annualized_capex_mag_$/year", np.nan)) + float(resA.get("annual_opex_mag_$/year", np.nan))
    ann_cost_B = float(resB.get("annualized_capex_mag_$/year", np.nan)) + float(resB.get("annual_opex_mag_$/year", np.nan))
    netE_A = float(resA.get("net_energy_annual_MWh", np.nan))
    netE_B = float(resB.get("net_energy_annual_MWh", np.nan))

    dE = netE_B - netE_A
    inc_lcoe = (ann_cost_B - ann_cost_A) / dE if dE != 0 else np.inf

    return {
        "LCOE_A_$/MWh": lcoeA,
        "LCOE_B_$/MWh": lcoeB,
        "delta_LCOE_B_minus_A_$/MWh": lcoeB - lcoeA,
        "incremental_LCOE_$/MWh": inc_lcoe,
        "delta_net_energy_MWh_per_year": dE,
        "delta_annualized_cost_$/year": ann_cost_B - ann_cost_A,
    }
