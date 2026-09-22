# economic_model.py
import pandas as pd
import numpy as np
import itertools
import os
from fusion_tem import device as cfg
from typing import Any, Dict

# 导入您重构后的物理模型
from fusion_tem.cryo import heat_load as hlm
from fusion_tem.economic.price_basis import (
    COOLANT_PRICE_2025_USD_PER_KG_BY_SCENARIO,
    HTS_PRICE_2025_USD_PER_KAM_BY_SCENARIO,
    HTS_PRICE_CONVERSION_METHOD,
    POWER_SUPPLY_PRICE_2025_USD_PER_A,
    PRICE_BASIS_YEAR,
)

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
    calculate_modeled_plant_costs,
    validate_price_only_scenarios,
)
from fusion_tem.economic.plant_availability import allocate_plant_annual_schedule

# =============================================================================
# 1. 定义情景和参数
# =============================================================================
def crf(r: float, N: float) -> float:
    """Capital Recovery Factor."""
    if r <= 0:
        return 1.0 / N
    return r * (1 + r) ** N / ((1 + r) ** N - 1)

def define_parameters():
    """ 定义所有计算所需的参数 """
    #  1. 定义不同工况下的制冷剂密度 (kg/m^3)
    coolant_density_kg_m3 = {
        'He': {4.2: 151.15, 10.0: 61.273, 20.0: 24.244},
        'H2': {20.0: 72.405}
    }
    
    # 2. All monetary inputs are normalized exactly once to constant 2025 US$.
    coolant_price_per_kg_by_scenario = {
        scenario: dict(prices)
        for scenario, prices in COOLANT_PRICE_2025_USD_PER_KG_BY_SCENARIO.items()
    }
    # --- 冷却剂回路体积: ARC 几何驱动, 以 SPARC TFMC 为标定锚点 ---
    # Michael et al. 2024 (IEEE TAS 34(2), 0600113): TFMC(2.9m×1.9m, 16饼) 绕组包+plena
    # 内部 SHe = 0.34 m³; 外部(cryomodules+transfer lines) +0.12 m³ -> 总回路 0.46 m³ (=1.35×内部),
    # 充注量 "not more than 20 kg (<200-L LHe equiv.)". 取外部管线裕量 1.5 (论文实测 1.35).
    # 比例系数 β = 0.34 / V_WP(TFMC)。TFMC 与 ARC 必须采用同一修正后的绕组包定义；
    # 50%冷结构裕量不代表额外氦体积，因此回路只按 cfg.V_WP 缩放。
    TFMC_HE_INTERNAL_M3 = 0.34       # TFMC 绕组包+plena 内部 SHe 体积 (Michael 2024)
    tfmc_np = 16
    tfmc_l1_m = 3.0
    tfmc_r1_m = 0.4
    tfmc_r2_m = 0.3
    tfmc_wid_m = 0.004
    tfmc_dist_m = 0.010
    tfmc_s_outer_m2 = (np.pi / 2) * (
        (tfmc_r1_m + tfmc_r2_m) ** 2
        + (tfmc_r1_m + tfmc_r2_m + tfmc_l1_m / 2) ** 2
    )
    tfmc_s_inner_m2 = (np.pi / 2) * (
        tfmc_r1_m**2 + (tfmc_r1_m + tfmc_l1_m / 2) ** 2
    )
    TFMC_V_WP_M3 = (
        tfmc_s_outer_m2 - tfmc_s_inner_m2 + tfmc_r2_m * tfmc_l1_m
    ) * tfmc_np * (tfmc_wid_m + tfmc_dist_m)
    EXT_PIPING_FACTOR   = 1.5        # 外部管线裕量 (论文实测 0.46/0.34=1.35)
    cryo_loop_volume_m3 = (EXT_PIPING_FACTOR
                           * (TFMC_HE_INTERNAL_M3 / TFMC_V_WP_M3)
                           * cfg.V_WP * cfg.Ntf)

    params = {
        # --- 基本情景设置 ---
        # 运行年限已耦合到技术场景：S1→20年，S2→30年，S3→50年；S4-S6均为30年（同S2）
        'tech_scenario_to_years': {
            'S1': 20,
            'S2': 30,
            'S3': 50,
            'S4': 30,  # S4-S6均使用S2的运行年限
            'S5': 30,  # S5使用S2的运行年限
            'S6': 30   # S6使用S2的运行年限
        },
        'tech_scenarios': ['S1', 'S2', 'S3', 'S4', 'S5', 'S6'], #技术场景，S4/S5/S6为带材价格敏感性分析
        'temperatures_K': [4.2, 10, 20], #温度，对比4.2K，10K，20K
        'coolants': ['He', 'H2'], #制冷剂，对比He和H2

        # --- 磁体参数 ---
        #不同温度有不同的电流
        'current_by_temperature':cfg.Ip_list,
        #制冷剂密度，对比4.2K，10K，20K
        'coolant_density_kg_m3': coolant_density_kg_m3,
        'coolant_price_per_kg_by_scenario': coolant_price_per_kg_by_scenario,
        # Constant-2025-US$ HTS prices; S4-S6 inherit S2 except this price.
        'hts_price_per_kAm_by_scenario': dict(
            HTS_PRICE_2025_USD_PER_KAM_BY_SCENARIO
        ),

        # --- 物理和固定成本参数 ---
        'fusion_power_MWth': FUSION_POWER_MWTH,
        'thermal_power_after_blanket_MWth': THERMAL_POWER_AFTER_BLANKET_MWTH,
        'gross_electric_power_MWe': GROSS_ELECTRIC_POWER_MWE,
        # Legacy alias retained for older callers; this is the 708 MW(th) value.
        'fusion_output_MWth': THERMAL_POWER_AFTER_BLANKET_MWTH,
        'C0_nonmagnet_USD': cfg.C0_nonmagnet_USD,
        'C0_nonmagnet_USD_by_scenario': dict(BACKGROUND_CAPITAL_USD_BY_SCENARIO),
        'background_capital_USD_by_scenario': dict(BACKGROUND_CAPITAL_USD_BY_SCENARIO),
        'core_vom_USD_per_MWh_th_by_scenario': dict(CORE_VOM_USD_PER_MWH_TH_BY_SCENARIO),
        'cost_boundary_version': COST_BOUNDARY_VERSION,
        'monetary_price_basis_year': PRICE_BASIS_YEAR,
        'monetary_values_constant_2025_usd': True,
        'hts_conductor_price_source_year': 2025,
        'hts_conductor_price_cpi_factor': 1.0,
        'hts_conductor_price_conversion_method': HTS_PRICE_CONVERSION_METHOD,
        # Independent non-TF auxiliary assumption; excludes every TF cryogenic load.
        'non_tf_aux_fraction': float(os.environ.get('NON_TF_AUX_FRACTION', '0.32')),
        'cryo_efficiency_model': os.environ.get('CRYO_EFFICIENCY_MODEL', 'green_rated'),
        'cryo_eta_max': float(os.environ.get('COP_ETA_MAX', '0.30')),
        'cryo_rated_margin': float(os.environ.get('CRYO_RATED_MARGIN', '1.0')),

        'cryo_loop_volume_m3': cryo_loop_volume_m3, #ARC几何驱动, TFMC标定(见上); was SPARC遗留 1.5*Ntf=27m3
        'annual_coolant_replenish_fraction': ANNUAL_COOLANT_REPLENISH_FRACTION,
        # 500 A / 10,000 US$ reference device, equivalent to 20 US$/A.
        'power_supply_unit_price': POWER_SUPPLY_PRICE_2025_USD_PER_A,
        
        'hts_kAm_by_temperature': cfg.kAm_HTS_TAPE_LIST, #不同温度下的HTS带材需求，单位：kAm
        
        # Thermal-to-electric conversion efficiency used for gross generation.
        'pcs_params': {
            'efficiency': cfg.eat_conv,
            'capital_cost_USD_per_kWe': PCS_CAPITAL_COST_USD_PER_KWE,
            'fom_fraction_per_year': PCS_FOM_FRACTION_PER_YEAR,
            'vom_USD_per_MWh_e': PCS_VOM_USD_PER_MWH_E,
        },
        # --- 运行时间/场景参数（以文档为准） ---
        'time_scenarios': {
            key: {k: v for k, v in scenario.items() if k != 'label'}
            for key, scenario in cfg.SCENARIO_DEFINITIONS.items()
        },
        # --- 全厂 LCOE 用到的金融/放大参数 ---
        "discount_rate": 0.08,
        "discount_rate_by_scenario": {key: 0.08 for key in cfg.SCENARIO_DEFINITIONS},
        }
    for scenario, expected in BACKGROUND_CAPITAL_USD_BY_SCENARIO.items():
        configured = float(cfg.SCENARIO_DEFINITIONS[scenario]['C0_nonmagnet_USD'])
        if configured != expected:
            raise ValueError(
                f"{scenario} C0 mismatch: device={configured}, cost_boundary={expected}"
            )
    validate_price_only_scenarios(
        time_scenarios=params['time_scenarios'],
        project_lifetime_years=params['tech_scenario_to_years'],
        coolant_prices=params['coolant_price_per_kg_by_scenario'],
        hts_prices=params['hts_price_per_kAm_by_scenario'],
        discount_rate_by_scenario=params['discount_rate_by_scenario'],
    )
    return params

# =============================================================================
# 2. 核心计算模块
# =============================================================================
def compute_case(tech_scenario: str, year, temperature_K: float, coolant: str,
                 Npw, R_p2p_joint: float, p: dict, *,
                 charge_time_h: float | None = None,
                 charge_cryo_energy_event_MWh: float | None = None,
                 charge_cryo_average_power_W: float | None = None,
                 base_heat_loads_override: dict | None = None,
                 rho_turn_uOhm_cm2: float | None = None,
                 cryo_rated_context: dict | None = None,
                 cryo_efficiency_model: str | None = None,
                 cryo_eta_max: float | None = None) -> tuple[dict, dict]:

    """ 对单个情景进行完整的技术经济计算 """
    pcs_params = p['pcs_params']
    Ip = p['current_by_temperature'][temperature_K] #单根带材的运行电流
    if charge_time_h is None:
        device_name = os.environ.get("FUSION_DEVICE", "unknown")
        raise ValueError(
            "missing charge_time_h for design "
            f"device={device_name}, scenario={tech_scenario}, Top={temperature_K}, "
            f"coolant={coolant}, Npw={Npw}, rho_turn={rho_turn_uOhm_cm2}, "
            f"R_joint={R_p2p_joint}"
        )
    efficiency_model = (
        p.get("cryo_efficiency_model", "green_rated")
        if cryo_efficiency_model is None else cryo_efficiency_model
    )
    eta_max = float(p.get("cryo_eta_max", 0.30) if cryo_eta_max is None else cryo_eta_max)
    rated_margin = float(p.get("cryo_rated_margin", 1.0))

    # --- Step 1: 调用物理模型计算寄生功率 ---
    # 1a. 计算一次基础热负荷,确定了磁体的参数：Ip=400A, Npw=10, R_p2p_joint=3e-9, Top=20K
    base_heat_loads = (
        base_heat_loads_override
        if base_heat_loads_override is not None
        else hlm.calculate_base_heat_loads(
            Ip=Ip, Npw=Npw, R_p2p_joint=R_p2p_joint, Top=temperature_K
        )
    )
    if cryo_rated_context is None:
        cryo_rated_context = hlm.build_cryo_rated_context(
            base_heat_loads,
            temperature_K,
            N_tf=cfg.Ntf,
            efficiency_model=efficiency_model,
            eta_max=eta_max,
            rated_margin=rated_margin,
        )
    # All modes use the same design-rated efficiency.
    power_kwargs = {
        "N_tf": cfg.Ntf,
        "rated_context": cryo_rated_context,
        "efficiency_model": efficiency_model,
        "eta_max": eta_max,
        "rated_margin": rated_margin,
    }
    P_cryo_prod_W = hlm.calculate_cryo_electrical_power('prod', base_heat_loads, temperature_K, **power_kwargs)
    P_cryo_dwell_W = hlm.calculate_cryo_electrical_power('dwell', base_heat_loads, temperature_K, **power_kwargs)
    P_cryo_static_W = hlm.calculate_cryo_electrical_power('static', base_heat_loads, temperature_K, **power_kwargs)

    
    # --- Step 2: 计算年度能量平衡（按场景分解时间与热负荷） ---
    # 2a. 读取该技术场景的时间参数（文档 s1,s2,s3）
    ts = p['time_scenarios'][tech_scenario]
    major_days_per_fpy = float(ts['major_scheduled_days_per_fpy'])
    minor_days_per_fpy = float(ts['minor_scheduled_days_per_fpy'])
    unplanned_unavailability = float(ts['unplanned_unavailability'])
    tf_cycles_per_year = float(ts['tf_cycles_per_year'])
    tcool_h   = float(ts['tcool_h'])
    twarm_h   = float(ts['twarm_h'])
    tau_pulse = float(ts['tau_pulse_h'])
    tau_dwell = float(ts['tau_dwell_h'])

    # 励磁时间 tcharge 使用充电 99.9% 时间（若有电磁表格），此处用 tau_pulse 的上限近似或外部对接；
    # 为保持与现版接口一致，先用 tau_pulse 的量级近似，也允许使用更保守的 5 天（可在 config 或外部耦合时覆盖）。
    # 如果后续对接 2.5_charge_time999 的插值结果，可在这里传入。
    # 这里采用保守：tcharge_h = tau_pulse（小时级）与文档 5 天不同，但不会改变接口；若需固定 5 天，请直接改为 5*24.0。
    tcharge_h = float(charge_time_h)

    # 2b. Strict plant-availability ledger. Pulse and dwell are both
    # plant-available states. Generic unplanned unavailability is applied to
    # the remainder after planned outages, so planned and unplanned time are
    # not double counted.
    annual_schedule = allocate_plant_annual_schedule(
        charge_hours=tcharge_h,
        major_days_per_fpy=major_days_per_fpy,
        minor_days_per_fpy=minor_days_per_fpy,
        tf_cycles_per_year=tf_cycles_per_year,
        cooldown_hours_per_cycle=tcool_h,
        warmup_hours_per_cycle=twarm_h,
        discharge_to_charge_ratio=1.0,
        unplanned_unavailability=unplanned_unavailability,
        pulse_hours_per_cycle=tau_pulse,
        dwell_hours_per_cycle=tau_dwell,
        hours_per_year=float(cfg.HOURS_PER_YEAR),
    )
    n_cycles   = annual_schedule.equivalent_cycles
    H_prod     = annual_schedule.pulse_hours
    H_dwell    = annual_schedule.dwell_hours
    H_core_scheduled = annual_schedule.core_scheduled_outage_hours
    H_coolwarm = annual_schedule.cooldown_warmup_hours
    H_excdec = annual_schedule.charge_discharge_hours
    H_tf_planned = annual_schedule.tf_planned_outage_hours
    H_planned = annual_schedule.planned_outage_hours
    H_unplanned = annual_schedule.effective_unplanned_outage_hours
    H_available = annual_schedule.available_hours

    # 2c. Annual TF cryogenic electricity (MWh).
    # Charge uses the time-integrated Data S2/S3 result when supplied. Calls
    # without those series use the design-specific base charge load, never a
    # fixed electrical power. Discharge mirrors the complete charge-stage
    # cryoplant power profile over the same design-specific duration.
    P_cryo_maint_W = 0.0
    P_cryo_coolwarm_W = P_cryo_static_W
    P_charge_base_W = hlm.calculate_cryo_electrical_power(
        'charge', base_heat_loads, temperature_K, **power_kwargs
    )
    E_charge_base_event_MWh = P_charge_base_W * tcharge_h / 1e6
    if charge_cryo_energy_event_MWh is None:
        E_charge_event_MWh = E_charge_base_event_MWh
        E_charge_dynamic_event_MWh = 0.0
        charge_energy_source = "design_base_without_em_timeseries"
        P_charge_average_W = P_charge_base_W
    else:
        E_charge_event_MWh = float(charge_cryo_energy_event_MWh)
        if E_charge_event_MWh < 0:
            raise ValueError("charge_cryo_energy_event_MWh cannot be negative")
        charge_energy_source = "data_s2_s3_transient_cop_integration"
        E_charge_dynamic_event_MWh = E_charge_event_MWh - E_charge_base_event_MWh
        if E_charge_dynamic_event_MWh < -1e-9:
            raise ValueError(
                "integrated charge event energy is below the design base load"
            )
        E_charge_dynamic_event_MWh = max(E_charge_dynamic_event_MWh, 0.0)
        P_charge_average_W = (
            float(charge_cryo_average_power_W)
            if charge_cryo_average_power_W is not None
            else (E_charge_event_MWh * 1e6 / tcharge_h if tcharge_h > 0 else 0.0)
        )

    # Definition: discharge has the same duration and the same complete
    # cryoplant electrical-load profile as charge. This mirrors the base and
    # dynamic terms separately and leaves no free discharge scaling factor.
    tdis_h = tcharge_h
    E_discharge_event_MWh = E_charge_event_MWh
    E_discharge_base_event_MWh = E_charge_base_event_MWh
    E_discharge_dynamic_event_MWh = E_charge_dynamic_event_MWh
    P_discharge_average_W = P_charge_average_W

    E_charge_annual_MWh = tf_cycles_per_year * E_charge_event_MWh
    E_discharge_annual_MWh = tf_cycles_per_year * E_discharge_event_MWh
    E_dynamic_charge_annual_MWh = tf_cycles_per_year * E_charge_dynamic_event_MWh
    E_dynamic_discharge_annual_MWh = tf_cycles_per_year * E_discharge_dynamic_event_MWh
    E_excdec_year_MWh = E_charge_annual_MWh + E_discharge_annual_MWh
    P_cryo_excdec_W = E_excdec_year_MWh * 1e6 / H_excdec if H_excdec > 0 else 0.0

    E_cryo_year_MWh = (
        (P_cryo_prod_W * H_prod)
        + (P_cryo_dwell_W * H_dwell)
        + (P_cryo_maint_W * H_core_scheduled)
        + (P_cryo_coolwarm_W * H_coolwarm)
    ) / 1e6 + E_excdec_year_MWh
    transient_energy_fraction_of_Ecryo = (
        E_excdec_year_MWh / E_cryo_year_MWh if E_cryo_year_MWh > 0 else np.nan
    )

    # 2c. 年度发电量 (MWh)
    gross_mw_e = float(p['gross_electric_power_MWe'])
    gross_from_thermal = (
        float(p['thermal_power_after_blanket_MWth']) * float(pcs_params['efficiency'])
    )
    if not np.isclose(gross_mw_e, gross_from_thermal, rtol=0.0, atol=1e-12):
        raise ValueError(
            f"gross electric power mismatch: explicit={gross_mw_e}, "
            f"thermal*efficiency={gross_from_thermal}"
        )
    gross_mwh_per_year = gross_mw_e * H_prod # 发电仅在生产阶段
    # Independent non-TF auxiliary energy, mutually exclusive with TF cryogenics.
    r_other = float(p['non_tf_aux_fraction'])
    other_recirc_mwh = gross_mwh_per_year * r_other
    net_mwh_per_year = gross_mwh_per_year - E_cryo_year_MWh - other_recirc_mwh
    
    r_parasitic_pct = (E_cryo_year_MWh / gross_mwh_per_year) * 100 if gross_mwh_per_year > 0 else np.inf

    # --- Step 3: 计算磁体相关资本成本 (CAPEX) ---

    # 3a. 带材成本
    hts_kAm = p["hts_kAm_by_temperature"][temperature_K]
    hts_unit_price = p["hts_price_per_kAm_by_scenario"][tech_scenario]
    tape_cost = hts_kAm * hts_unit_price

    # 3b. 制冷剂：首次填充CAPEX + 年度补充OPEX
    # 获取当前工况的密度 (kg/m^3)
    density = p["coolant_density_kg_m3"][coolant][temperature_K]
    # 计算充满回路所需的总质量 (kg)
    required_mass_kg = p["cryo_loop_volume_m3"] * density
    # 获取当前情景的价格 ($/kg)
    price_per_kg = p["coolant_price_per_kg_by_scenario"][tech_scenario][coolant]
    # 计算单次填充成本和全生命周期成本
    coolant_fill_cost = required_mass_kg * price_per_kg
    coolant_replenish_cost_per_year = (
        required_mass_kg * p["annual_coolant_replenish_fraction"] * price_per_kg
    )
    coolant_lifetime_cost = coolant_fill_cost + coolant_replenish_cost_per_year * year

    # 3c. 电源成本：18 个 TF 磁体采用单一串联电源，额定电流为
    # Iop/carrying_factor = Ip*Npw/carrying_factor。该项只代理随设计变化的
    # 电流容量，不解析串联系统的电压和储能相关设备成本。
    power_supply_rated_current_A = Ip * Npw / cfg.carrying_factor
    power_supply_cost = power_supply_rated_current_A * float(p["power_supply_unit_price"])

    # 3d. Magnet CAPEX: no installation multiplier is currently modeled.
    capex_mag_direct = tape_cost + coolant_fill_cost + power_supply_cost
    capex_mag_installed = capex_mag_direct
    capex_mag = capex_mag_direct  # legacy alias

    # --- Step 4: modeled plant-v2 cost boundary and legacy diagnostics ---
    r = float(p.get("discount_rate_by_scenario", {}).get(
        tech_scenario, p.get("discount_rate", 0.08)
    ))
    crf_val = crf(r, float(year))
    cost_result = calculate_modeled_plant_costs(
        scenario=tech_scenario,
        annual_hours_prod=H_prod,
        E_net_year_MWh=net_mwh_per_year,
        CAPEX_mag_direct_USD=capex_mag_direct,
        CAPEX_mag_installed_USD=capex_mag_installed,
        coolant_fill_cost_USD=coolant_fill_cost,
        CRF=crf_val,
        fusion_power_MWth=float(p['fusion_power_MWth']),
        gross_electric_power_MWe=gross_mw_e,
    )
    if not np.isclose(
        coolant_replenish_cost_per_year,
        cost_result['OPEX_coolant_VOM_USD_per_year'],
        rtol=0.0,
        atol=1e-9,
    ):
        raise AssertionError("coolant replenishment calculation drifted")

    C0_nonmagnet_USD = cost_result['C0_background_USD']
    annualized_capex_mag = cost_result['Annualized_CAPEX_mag_USD_per_year']
    annual_opex_mag = cost_result['OPEX_coolant_VOM_USD_per_year']
    lcoe_magnet_component = cost_result['LCOE_magnet_only_USD_per_MWh']
    lcoe_fullplant = cost_result['LCOE_fullplant_legacy_USD_per_MWh']
    lcoe_plant = cost_result['LCOE_plant_USD_per_MWh']
    lcoe_C0 = (
        cost_result['Annualized_C0_USD_per_year'] / net_mwh_per_year
        if np.isfinite(net_mwh_per_year) and net_mwh_per_year > 0.0
        else np.nan
    )

    economic_result = {
        # --- 输入情景 ---
        "tech_scenario": tech_scenario, 
        "project_years": year, 
        "temperature_K": temperature_K, 
        'Ip': Ip,
        "coolant": coolant,
        "Npw": Npw,
        "R_p2p_joint": R_p2p_joint,
        # --- 经济输入参数 ---
        "input_hts_price_$/kAm": hts_unit_price,
        "input_hts_price_2025USD_per_kAm": hts_unit_price,
        #-----制冷剂的成本参数
        "input_coolant_price_$/kg": price_per_kg,
        "input_coolant_price_2025USD_per_kg": price_per_kg,
        "coolant_density_kg/m3": density,
        "coolant_required_mass_kg": required_mass_kg,
        "coolant_fill_cost_$": coolant_fill_cost,
        "coolant_lifetime_cost_$": coolant_lifetime_cost,
        # --- 寄生功耗计算 ---
        "cryo_power_prod_W": P_cryo_prod_W,
        "cryo_power_dwell_W": P_cryo_dwell_W,
        "cryo_power_static_W": P_cryo_static_W,
        "annual_hours_prod": H_prod,
        "annual_hours_dwell": H_dwell,
        "annual_hours_core_scheduled_outage": H_core_scheduled,
        "annual_hours_tf_planned_outage": H_tf_planned,
        "annual_hours_planned_outage": H_planned,
        "annual_hours_unplanned_outage_effective": H_unplanned,
        "annual_hours_available": H_available,
        "annual_hours_coolwarm": H_coolwarm,
        "annual_hours_excdec": H_excdec,
        "core_scheduled_availability": annual_schedule.core_scheduled_availability,
        "planned_unavailability": annual_schedule.planned_unavailability,
        "unplanned_unavailability": unplanned_unavailability,
        "plant_availability": annual_schedule.plant_availability,
        "pulse_duty_factor": annual_schedule.pulse_duty_factor,
        "gross_capacity_factor": annual_schedule.gross_capacity_factor,
        "charge_time_event_h": tcharge_h,
        "discharge_time_event_h": tdis_h,
        "cryo_power_coolwarm_W": P_cryo_coolwarm_W,
        "cryo_power_excdec_W": P_cryo_excdec_W,
        "cryo_power_charge_average_W": P_charge_average_W,
        "cryo_power_discharge_average_W": P_discharge_average_W,
        "cryo_power_charge_base_W": P_charge_base_W,
        "cryo_charge_energy_source": charge_energy_source,
        "cryo_energy_charge_event_MWh": E_charge_event_MWh,
        "cryo_energy_charge_base_event_MWh": E_charge_base_event_MWh,
        "cryo_energy_charge_dynamic_event_MWh": E_charge_dynamic_event_MWh,
        "cryo_energy_discharge_event_MWh": E_discharge_event_MWh,
        "cryo_energy_discharge_base_event_MWh": E_discharge_base_event_MWh,
        "cryo_energy_discharge_dynamic_event_MWh": E_discharge_dynamic_event_MWh,
        "cryo_energy_charge_annual_MWh": E_charge_annual_MWh,
        "cryo_energy_discharge_annual_MWh": E_discharge_annual_MWh,
        "cryo_energy_dynamic_charge_annual_MWh": E_dynamic_charge_annual_MWh,
        "cryo_energy_dynamic_discharge_annual_MWh": E_dynamic_discharge_annual_MWh,
        "charge_discharge_profile_ratio": 1.0,
        "cryo_energy_excdec_annual_MWh": E_excdec_year_MWh,

        "cryo_energy_annual_MWh": E_cryo_year_MWh,
        "transient_energy_fraction_of_Ecryo": transient_energy_fraction_of_Ecryo,
        "r_parasitic_pct": r_parasitic_pct,
        **cryo_rated_context,
        "non_tf_aux_fraction": r_other,
        "E_cryo_TF_annual_MWh": E_cryo_year_MWh,
        "r_cryo_fraction": E_cryo_year_MWh / gross_mwh_per_year if gross_mwh_per_year > 0 else np.inf,
        "r_cryo_pct": r_parasitic_pct,
        # --- 能量平衡 ---
        "gross_power_output_MWe": gross_mw_e,
        "gross_energy_annual_MWh": gross_mwh_per_year,
        "other_aux_fraction": r_other,
        "other_aux_energy_annual_MWh": other_recirc_mwh,
        "net_energy_annual_MWh": net_mwh_per_year,
        # --- 成本分解（磁体系统）---
        "input_hts_price_$/kAm": hts_unit_price,
        "input_hts_price_2025USD_per_kAm": hts_unit_price,
        "tape_cost_$": tape_cost,

        "input_coolant_price_$/kg": price_per_kg,

        "input_coolant_price_2025USD_per_kg": price_per_kg,
        "coolant_density_kg/m3": density,
        "coolant_required_mass_kg": required_mass_kg,
        "coolant_fill_cost_$": coolant_fill_cost,
        "coolant_replenish_cost_$/year": coolant_replenish_cost_per_year,
        "coolant_lifetime_cost_$": coolant_lifetime_cost,

        "power_supply_cost_$": power_supply_cost,

        "capex_mag_$": capex_mag,
        # --- LCOE（磁体归因）---
        "discount_rate": r,
        "crf": crf_val,
        "C0_nonmagnet_$": C0_nonmagnet_USD,
        "annualized_capex_mag_$/year": annualized_capex_mag,
        "annual_opex_mag_$/year": annual_opex_mag,
        "lcoe_C0_$/MWh": lcoe_C0,
        # Legacy diagnostics retain their historical mathematical definitions.
        "lcoe_magnet_only_$/MWh": lcoe_magnet_component,
        "lcoe_fullplant_$/MWh": lcoe_fullplant,
        "lcoe_plant_$/MWh": lcoe_plant,
        **cost_result,
          }
    return economic_result,  base_heat_loads # ---热负荷---

# =============================================================================
# 3. 主运行和输出模块 (与之前版本相同)
# =============================================================================
def run_all_cases(p: dict) -> pd.DataFrame:
    """
    遍历所有情景组合并运行计算。
    注意：运行年限已耦合到技术场景（S1→20年，S2→30年，S3→50年）
    """
    rows = []
    # 使用场景对应的年限，而不是遍历所有年限组合
    for tech in p["tech_scenarios"]:
        year = p["tech_scenario_to_years"][tech]
        param_combinations = itertools.product(
            [tech], [year],  # 每个场景使用其对应的年限
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
# 4. A->B 全厂差分LCOE
# =============================================================================
def delta_lcoe_fullplant(
    caseA: Dict[str, Any],
    caseB: Dict[str, Any],
    p: dict,
) -> Dict[str, float]:
    """
    计算全厂口径的 ΔLCOE 与增量LCOE。

    Parameters
    ----------
    caseA, caseB : dict
        必须包含 compute_case 的入参键：
          tech_scenario, year, temperature_K, coolant, Npw, R_p2p_joint
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

    lcoeA = float(resA.get("lcoe_plant_$/MWh", np.nan))
    lcoeB = float(resB.get("lcoe_plant_$/MWh", np.nan))

    # 增量LCOE（边际成本/边际电量）
    ann_cost_A = lcoeA * float(resA.get("net_energy_annual_MWh", np.nan))
    ann_cost_B = lcoeB * float(resB.get("net_energy_annual_MWh", np.nan))
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
