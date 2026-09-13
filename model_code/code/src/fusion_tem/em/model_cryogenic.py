# heat_load_model.py
#
# DEAD MODULE. Nothing in this package imports it, and no pipeline stage runs it.
# The live cryogenic heat-load model is fusion_tem/cryo/heat_load.py, whose
# _mode_system_heat_loads defines four operating modes (prod, dwell, charge,
# static) against this file's three. Read that module, not this one, when
# checking what the published results actually computed. Kept only because older
# standalone scripts outside the package still reference a copy of it.
import os
import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
import pandas as pd
from fusion_tem import device as cfg
from fusion_tem import materials as mat  # 导入新的材料属性文件
colors = {4.2:"tab:blue", 10:"tab:orange", 20:"tab:green", 77:"tab:red"}

# REBCO 引线热负荷计算
def lead_REBCO_heat(Top, I_operation):
    """
    根据给定的磁体运行温度和运行电流，计算稳定器截面积和传导热。
    """
    t_f = cfg.Time_transfer
    T_start = 77.0
    T_f = cfg.Temp_transfer

    T_hot = cfg.T_HIGH
    L = cfg.L_LEAD_HTS
    I_max = I_operation / cfg.carrying_factor
    
    E_value = (float(I_max)**2 * float(t_f)) / 3.0
    integrand_G = lambda T: mat.d_ss304 * mat.Cp_ss304(T) / mat.rho_ss304(T)
    G_value, _ = quad(integrand_G, T_start, T_f)
    
    if G_value == 0:
        return 0, 0
    A_m2 = np.sqrt(E_value / G_value)

    k_integral, _ = quad(mat.k_ss304, Top, T_hot)
    Q_watts = A_m2 * (1/L) * k_integral

    return  Q_watts, A_m2

# 铜引线热负荷计算
def lead_Cu_heat(I_operation, L=cfg.LEN_LEAD_CU, A=None, design_factor=cfg.design_factor, RRR=cfg.RRR):
    """
    I_operation   : 工作电流 [A]
    L   : 引线长度 [m]
    A   : 截面积 [m^2] (若不给则按设计因子计算)
    design_factor : I*L/A ≈ 5e6 A/m (经验设计值)
    RRR : 铜纯度
    返回：一根77K 铜电流引线的传导热，焦耳热，总热，截面积
    """
    # 按照最大电流计算截面积
    I_max = I_operation / cfg.carrying_factor
    if A is None:
        A = I_max*L/design_factor

    # --- 传导热 ---
    def integrand_k(T):
        return mat.thermal_conductivity_Cu(T, RRR)
    k_int, _ = quad(integrand_k, 77, 300)
    Q_cond = A/L * k_int   # W

    # --- 电阻 & Joule 热 ---
    def integrand_rho(T):
        return mat.resistivity_Cu(T, RRR)
    rho_avg, _ = quad(integrand_rho, 77, 300)
    rho_avg /= (300-77)
    R = rho_avg * L / A
    Q_joule = 0.5 * I_operation **2 * R  # 冷端只承担一半 Joule 热

    Q_total = Q_cond + Q_joule
    return Q_cond, Q_joule, Q_total, A


# ---------------------------
# 1. 线圈内部超导-超导接头焦耳热
# ---------------------------
def coil_internal_joint_heat(Npw, L_total_m,Ip, L_single_tape=cfg.LEN_PER_SINGEL_REBCO,  R_ss=cfg.R_SU_JOINT):
    # One equivalent tape-to-tape splice contact is represented at each
    # location along the effective winding path. The effective path length is
    # L_total_m / Npw, so the continuous splice-location count scales as 1/Npw.
    N_joint = L_total_m / (L_single_tape * Npw)
    Q = (Ip**2) * R_ss * N_joint
    return Q

# ---------------------------
# 2. 饼对饼接头焦耳热
# ---------------------------
def pancake_joint_heat(Npw,  R_p2p_joint, Ip, Np=cfg.NP):
    
    I_total = float(Ip * Npw)
    Q = (I_total**2) * R_p2p_joint * (Np + 1)
    if Q<0:
        print(type(I_total), type(Ip),type(R_p2p_joint), type(Np))
        print(I_total,R_p2p_joint,Np,Q)
    return Q

# ---------------------------
# 3. 核子加热
# ---------------------------
def nuclear_heating():
    return cfg.V_magnet * cfg.NUCLEAR_POWER_DENSITY

# ---------------------------
# 4. 恒温器辐射热
# ---------------------------
def radiation_heat(A_surf, eps, T_hot, Top):
    N_layer = cfg.N_LAYER_CRYOSTAT
    sigma = 5.67e-8
    eps_eff = eps / (2 * (N_layer + 1))
    return  sigma * A_surf * eps_eff* (T_hot**4 - Top**4) 

# ---------------------------
# 5. 电流引线热
# ---------------------------
def current_lead_heat(Ip, Npw, Top, N_lead=cfg.N_LEAD, L_cu=cfg.LEN_LEAD_CU, design_factor=cfg.design_factor, RRR=cfg.RRR):
    I_operation = Ip * Npw # 单根导体的工作电流 * 并绕数
    Qlead_cond_77, Qlead_joule_77, Qlead_total_77, A_Cu = lead_Cu_heat(I_operation=I_operation, L=L_cu,
                                   design_factor=cfg.design_factor, RRR=cfg.RRR)
    Q_lead_hts, A_REBCO = lead_REBCO_heat(Top, I_operation=I_operation)
    return Qlead_cond_77 * N_lead , Qlead_joule_77 * N_lead, Q_lead_hts * N_lead

# ---------------------------
# 6. 管道导热
# ---------------------------
def conduction_heat(A, L, T_hot, Top, k_func=mat.k_ss304):
    integral, _ = quad(k_func, Top, T_hot)
    return A / L * integral

# ---------------------------
# 7. 其他管线传导热
def pipe_heat(N_pipe, d_in, d_out, L, T_hot, Top, k_func=mat.k_ss304):
    """ 计算一类管道的总导热量 """
    A = np.pi * (d_out**2 - d_in**2) / 4.0   # 截面积 [m2]
    Q_single = conduction_heat(A, L, T_hot, Top, k_func)
    return N_pipe * Q_single

# ---------------------------
# 8. 失超热
# ---------------------------
def quench_heat():
    return 0.0

# ---------------------------
# 9. 支撑装置 + 残余气体
# ---------------------------
def misc_heat():
    return cfg.Q_OTHER_SOURCES

# 10.

def achieve_COP(Th,Tl,Wheat):
    """ 计算制冷机的性能系数 (COP) """
    if Th==Tl:
        eat_Carnot = 1
    else:
        eat_Carnot = Tl/(Th-Tl)
    # 经验公式，表示实际COP与理想卡诺COP的比率，随热负荷变化
    
    eta_achieved = 10**(-0.92237+0.07763*np.log10(1+Wheat))   
    return eat_Carnot*eta_achieved

def calculate_base_heat_loads(Ip,Npw,R_p2p_joint, Top=20):
    """
    计算生产模式下，单个装置所有可能的热负荷分量。
    这是后续计算的基础。
    """
    L_HTS_TF_m = cfg.L_HTS_TF_m[Top]
    Q1 = coil_internal_joint_heat(Npw =Npw, L_total_m=L_HTS_TF_m, Ip=Ip, R_ss=cfg.R_SU_JOINT)
    Q2 = pancake_joint_heat(Npw =Npw, R_p2p_joint=R_p2p_joint, Ip=Ip)
    Q3 = nuclear_heating()
    Q4 = radiation_heat(A_surf=cfg.A_cryostat,  eps=cfg.eps, T_hot=cfg.T_HIGH, Top=Top)
    Q_lead_cond_77, Q_lead_joule_77, Q_lead_hts = current_lead_heat(Ip=Ip, Npw=Npw, Top=Top) 
    #电流引线分为两段，77K 铜导线传导热和焦耳热，20K HTS传导热
    Q5 = Q_lead_hts#HTS 引线传导热，在运行温区，已经计入Q5

    # 制冷管道
    Q7 = pipe_heat(N_pipe=cfg.N_cool_pipe, d_in=cfg.d_in_cool, d_out=cfg.d_out_cool, L=cfg.L_cool, T_hot=cfg.T_HIGH, Top=Top)
    # 其他辅助管道
    Q8 = pipe_heat(N_pipe=cfg.N_aux_pipe, d_in=cfg.d_in_aux, d_out=cfg.d_out_aux, L=cfg.L_aux, T_hot=cfg.T_HIGH, Top=Top)
    
    Q9 = quench_heat()
    Q10 = misc_heat()

    Q_Tc = Q1 + Q2 + Q3 + Q4 + Q5 + Q7 + Q8 + Q9 +Q10

    Q_total_Tc = Q_Tc*cfg.Ntf
    Q_total_77 = Q_lead_cond_77*cfg.Ntf + Q_lead_joule_77*cfg.Ntf
    COP_77 = achieve_COP(cfg.T_HIGH,77,Q_total_77)
    COP_Tc = achieve_COP(cfg.T_HIGH,Top,Q_total_Tc)
    heat_loads = [Q1, Q2, Q3, Q4, Q5, Q7, Q8, Q9, Q10, Q_total_Tc, Q_total_77, COP_77, COP_Tc]
    for i in range(len(heat_loads)):
        heat_loads[i] = heat_loads[i]*cfg.Ntf
        if heat_loads[i] < 0:
            print(f"i={i}",Q2,R_p2p_joint, Ip**2*R_p2p_joint)

    P_cryo_electric_W = (Q_total_Tc/COP_Tc+Q_total_77/COP_77)
    return {
        "coil_internal_joint": Q1,
        "pancake_joint": Q2,
        "nuclear": Q3,
        "radiation": Q4,
        "current_leads_HTS": Q5,
        "pipes_coolant": Q7,       
        "pipes_aux": Q8,
        "quench": Q9,
        "misc": Q10,
        "current_leads_Cu_conduction": Q_lead_cond_77,
        "current_leads_Cu_joule": Q_lead_joule_77,
        "heat @ 77K":Q_total_77,
        "COP @ 77K":COP_77,
        f"heat @ {Top}K": Q_total_Tc,    
        f"COP @ {Top}K":COP_Tc,
        "P_cryo_electric_W":P_cryo_electric_W,
    }
def calculate_cryo_electrical_power(mode: str, base_heat_loads: dict, Top: float, N_tf = 18):
    """
    根据运行模式（prod, dwell, maint），计算低温系统所需的总电功率。

    Args:
        mode (str): 'prod', 'dwell', or 'maint'.
        base_heat_loads (dict): 由 calculate_base_heat_loads() 计算的基础热负荷字典。
        Top (float): 低温运行温度 (K)。
        N_tf (int): TF线圈总数。

    Returns:
        float: 该模式下低温系统的总耗电功率 (W)。
    """
    Q_Tc_components = 0
    Q_77_components = 0
    
    # 假设核加热在dwell模式下的衰减因子
    nuclear_decay_factor = 0.1

    # 根据模式累加不同的热负荷
    if mode == "prod":
        Q_Tc_components = sum([
            base_heat_loads["coil_internal_joint"], base_heat_loads["pancake_joint"],
            base_heat_loads["nuclear"], base_heat_loads["radiation"],
            base_heat_loads["current_leads_HTS"], base_heat_loads["pipes_coolant"], base_heat_loads["pipes_aux"],
            base_heat_loads["misc"], base_heat_loads["quench"]
        ])
        Q_77_components = base_heat_loads["current_leads_Cu_conduction"] + base_heat_loads["current_leads_Cu_joule"]

    elif mode == "dwell":
        Q_Tc_components = sum([
            base_heat_loads["coil_internal_joint"], base_heat_loads["pancake_joint"],
            base_heat_loads["nuclear"] * nuclear_decay_factor, base_heat_loads["radiation"],
            base_heat_loads["current_leads_HTS"], base_heat_loads["pipes_coolant"], base_heat_loads["pipes_aux"],
            base_heat_loads["misc"], base_heat_loads["quench"]
        ])
        Q_77_components = base_heat_loads["current_leads_Cu_conduction"] + base_heat_loads["current_leads_Cu_joule"]

    elif mode == "static":
        # 无电流、无聚变反应的被动冷态。model_economic 用本模式的功率给降温与升温阶段计费
        # （P_cryo_coolwarm_W = P_cryo_static_W），因此该分支直接进入年度账本。
        # V6.7 模型改动：补入 current_leads_HTS。HTS 引线传导是被动热漏，磁体只要处于冷态
        # 就存在，与是否通流无关。V6.6 及以前遗漏该项，与 Note S3 的 Table S6（静态列勾选
        # HTS-lead conduction）以及同文件 scale_base_heat_loads 的 maint 分支都不一致。
        # 本改动抬高降温升温段的制冷电耗，年度账本、制冷负担分布与 LCOE 全部随之变化，
        # V6.7 结果不可与 V6.6 结果混用。
        Q_Tc_components = sum([
            base_heat_loads["radiation"], base_heat_loads["current_leads_HTS"],
            base_heat_loads["pipes_coolant"], base_heat_loads["pipes_aux"], base_heat_loads["misc"], base_heat_loads["quench"]
        ])
        # 引线无电流，只有从77K到300K的静态热传导
        Q_77_components = base_heat_loads["current_leads_Cu_conduction"]
    
    # 计算总热负荷 (乘以TF线圈总数)
    Q_total_Tc = Q_Tc_components * N_tf
    Q_total_77 = Q_77_components * N_tf

    # 将热负荷通过COP转换为电功率
    T_hot = cfg.T_HIGH # 假设环境温度为300K

    P_cryo_Tc_W = Q_total_Tc / achieve_COP(T_hot, Top, Q_total_Tc) if Q_total_Tc > 0 else 0
    P_cryo_77_W = Q_total_77 / achieve_COP(T_hot, 77, Q_total_77) if Q_total_77 > 0 else 0
    
    return P_cryo_Tc_W + P_cryo_77_W

def plot_scaling_effect(
    Npw, R_p2p_joint, Top=20, 
    N_facility_list=range(1,51),   # 默认研究 1~50 台规模化
    save_dir=None
):
    """
    绘制规模化建设对寄生功率成本的影响
    """
    para_costs = []
    
    # 逐个规模计算
    for Nf in N_facility_list:
        results = calculate_base_heat_loads(
            Ip=400,
            Npw=Npw,
            L_tot=cfg.L_tot,
            L_single_tape=cfg.LEN_PER_SINGEL_REBCO,
            R_ss=cfg.R_SU_JOINT,
           
            R_p2p_joint=R_p2p_joint,
            V_coil=cfg.V_magnet,
            A_cryostat=cfg.A_cryostat,
            eps=cfg.eps,
            N_cool_pipe=cfg.N_cool_pipe,
            N_aux_pipe=cfg.N_aux_pipe,
            Top=Top,
        )
        
        # 注意：规模化时热负荷要乘 Nf
        Q_total_Tc = results[f"heat @ {Top}K"] * Nf
        Q_total_77 = results["heat under 77K"] * Nf
        
        COP_77 = achieve_COP(cfg.T_HIGH,77,Q_total_77)
        COP_Tc = achieve_COP(cfg.T_HIGH,Top,Q_total_Tc)
        
        P_cryo_electric_W = (Q_total_Tc/COP_Tc + Q_total_77/COP_77)
        P_gross_electric_W = cfg.fusion_output_MWth * 1e6 * cfg.eat_conv * Nf
        
        para_cost = P_cryo_electric_W / P_gross_electric_W * 100
        para_costs.append(para_cost)
    
    # 画图
    plt.figure(figsize=(8,6))
    plt.plot(N_facility_list, para_costs, "o-", color=colors[Top], markersize=5, label=f"Top={Top} K")
    plt.xlabel("Number of facilities built at site")
    plt.ylabel("Parasitic power cost (%)")
    plt.grid(True, ls="--", alpha=0.5)
    plt.title(f"Scaling effect (Npw={Npw}, R_p2p_joint={R_p2p_joint:.1e}, Top={Top} K)")
    if save_dir is not None:
        plt.savefig(save_dir)
    plt.show()
    
    return N_facility_list, para_costs

def annual_parasitic_fraction_three_stage(
    # --- device / steady-state baseline (production) parameters ---
    Ip=400,
    Npw=20,
    R_p2p_joint=None,
    Top=20,           # K, 冷端在生产态的目标温度
    # --- time scheduling parameters ---
    pulse_hours=2.0,           # 每个生产脉冲时长 (h)
    dwell_hours=10.0/60.0,     # 每个间隔/充电时长 (h) ; 默认 10 min
    maintenance_hours_per_year=24*30*2,  # 默认: 每年 2 个月 = 1440 h

    # --- 冷站共享选项 ---
    shared_cryo=False,         # True: 多台共享大型冷站（合并热负荷再算 COP）
    N_facilities=1,            # 同地点建多少台（用于规模化评估）
    # --- 一次性冷却/回暖能量（Wh/次） ---
    N_cooldown_per_year=0, E_cooldown_Wh=0.0,
    N_warmup_per_year=0, E_warmup_Wh=0.0,
    # --- 温度环境 ---
    T_hot=cfg.T_HIGH
):
    """
    返回字典，包含年度寄生占比、各阶段小时与耗能、以及中间量。
    依赖外部函数:
      - calculate_base_heat_loads(...): returns heat loads at the operating and 77 K stages
      - achieve_COP(Th, Tl, Wheat)
    """ 
    # 1) 计算生产态单台基准稳态热负荷（W）
    base = calculate_base_heat_loads(Ip=Ip, Npw=Npw,R_p2p_joint=R_p2p_joint,Top=Top)
    Q4_base_per_device_W = base[f"heat @ {Top}K"]
    Q77_base_per_device_W = base["heat @ 77K"]
    P_gross_elec_per_device_W = cfg.fusion_output_MWth * 1e6 * cfg.eat_conv

    # 2) 年度时间分配
    remaining_hours = cfg.HOURS_PER_YEAR - maintenance_hours_per_year
    cycle_hours = pulse_hours + dwell_hours
    if cycle_hours <= 0:
        raise ValueError("pulse_hours + dwell_hours must be > 0")
    n_cycles = remaining_hours / cycle_hours
    # 如果 remaining_hours < 0，则意味着维护占全年 > 8760h，不合理
    if remaining_hours <= 0:
        raise ValueError("maintenance_hours_per_year >= 8760 -> no runtime left")

    H_prod = n_cycles * pulse_hours
    H_dwell = n_cycles * dwell_hours
    H_maint = maintenance_hours_per_year

    # 3) 每个阶段的热负荷（整个TF系统） - 使用缩放因子
    Q4_prod_per_device, Q77_prod_per_device = scale_base_heat_loads(base, mode="prod")
    Q4_dwell_per_device, Q77_dwell_per_device   = scale_base_heat_loads(base, mode="dwell")
    Q4_maint_per_device, Q77_maint_per_device = scale_base_heat_loads(base, mode="maint")


    # 4) 把台数并入：如果共享冷站 -> 先合并热负荷再算 COP；否则逐台算功率相加
    def cryo_power_for_loads(Q4_total_W, Q77_total_W, Top_local):
        """给定总Q4与Q77求低温压缩机电功 W（合并计）"""
        P = 0.0
        if Q4_total_W > 0:
            COP4 = achieve_COP(cfg.T_HIGH, Top_local, Q4_total_W)  # 注意：achieve_COP 接受 Wheat=总热功
            P += Q4_total_W / COP4
        if Q77_total_W > 0:
            COP77 = achieve_COP(cfg.T_HIGH, 77.0, Q77_total_W)
            P += Q77_total_W / COP77
        return P  # W

    # 4a) if shared_cryo: 合并台数后计算每个阶段的冷机功率（W）
    if shared_cryo:
        Q4_prod_total = Q4_prod_per_device * N_facilities
        Q77_prod_total = Q77_prod_per_device * N_facilities
        P_cryo_prod_total_W = cryo_power_for_loads(Q4_prod_total, Q77_prod_total, Top)

        Q4_dwell_total = Q4_dwell_per_device * N_facilities
        Q77_dwell_total = Q77_dwell_per_device * N_facilities
        P_cryo_dwell_total_W = cryo_power_for_loads(Q4_dwell_total, Q77_dwell_total, Top)

        Q4_maint_total = Q4_maint_per_device * N_facilities
        Q77_maint_total = Q77_maint_per_device * N_facilities
        P_cryo_maint_total_W = cryo_power_for_loads(Q4_maint_total, Q77_maint_total, Top)
    else:
        # 逐台算（COP 以单台热负荷计算），然后乘以台数
        P_cryo_prod_per_device_W = cryo_power_for_loads(Q4_prod_per_device, Q77_prod_per_device, Top)
        P_cryo_dwell_per_device_W = cryo_power_for_loads(Q4_dwell_per_device, Q77_dwell_per_device, Top)
        P_cryo_maint_per_device_W = cryo_power_for_loads(Q4_maint_per_device, Q77_maint_per_device, Top)

        P_cryo_prod_total_W = P_cryo_prod_per_device_W * N_facilities
        P_cryo_dwell_total_W = P_cryo_dwell_per_device_W * N_facilities
        P_cryo_maint_total_W = P_cryo_maint_per_device_W * N_facilities

    # 5) 年度压缩机能量 (Wh)
    E_cryo_prod_Wh = P_cryo_prod_total_W * H_prod
    E_cryo_dwell_Wh = P_cryo_dwell_total_W * H_dwell
    E_cryo_maint_Wh = P_cryo_maint_total_W * H_maint

    # 一次性冷却/回暖能量
    E_cooldown_Wh = N_cooldown_per_year * E_cooldown_Wh
    E_warmup_Wh = N_warmup_per_year * E_warmup_Wh

    E_cryo_year_Wh = E_cryo_prod_Wh + E_cryo_dwell_Wh + E_cryo_maint_Wh + E_cooldown_Wh + E_warmup_Wh

    # 6) 年度毛发电能量（仅生产阶段产生）
    E_out_year_Wh = P_gross_elec_per_device_W * N_facilities * H_prod

    # 7) 寄生占比（按能量）
    F_parasitic_pct = 100.0 * E_cryo_year_Wh / E_out_year_Wh if E_out_year_Wh > 0 else np.inf

    return {
        "H_prod_h": H_prod,
        "H_dwell_h": H_dwell,
        "H_maint_h": H_maint,
        "P_cryo_prod_total_W": P_cryo_prod_total_W,
        "P_cryo_dwell_total_W": P_cryo_dwell_total_W,
        "P_cryo_maint_total_W": P_cryo_maint_total_W,
        "E_cryo_prod_Wh": E_cryo_prod_Wh,
        "E_cryo_dwell_Wh": E_cryo_dwell_Wh,
        "E_cryo_maint_Wh": E_cryo_maint_Wh,
        "E_cryo_year_Wh": E_cryo_year_Wh,
        "E_out_year_Wh": E_out_year_Wh,
        "F_parasitic_pct": F_parasitic_pct,
        "n_cycles_per_year": n_cycles,
        "Q4_base_per_device_W": Q4_base_per_device_W,
        "Q77_base_per_device_W": Q77_base_per_device_W,
        "P_gross_elec_per_device_W": P_gross_elec_per_device_W
    }

def scale_base_heat_loads(results_base, mode="prod", Top=20):
    """
    根据运行阶段（prod/dwell/maint）返回缩放后的 (Q_Tc, Q_77)，单个TF线圈的值
    """
    Q1 = results_base["coil_internal_joint"]
    Q2 = results_base["pancake_joint"]
    Q3 = results_base["nuclear"]
    Q4 = results_base["radiation"]
    Q5 = results_base["current_leads_HTS"]
    Q7_coolant = results_base["pipes_coolant"]
    Q8_aux = results_base["pipes_aux"]
    Q9_quench = results_base["quench"]
    Q10_misc = results_base["misc"]
    Q_lead_cond_77 = results_base["current_leads_Cu_conduction"]
    Q_lead_joule_77 = results_base["current_leads_Cu_joule"]
    
    # 管道总热负荷
    Q_pipes = Q7_coolant + Q8_aux

    if mode == "prod":
        # 生产态，原始值,整个TF系统（一个device）的值
        Q_Tc = Q1 + Q2 + Q3 + Q4 + Q5 + Q_pipes + Q9_quench + Q10_misc
        Q77 = Q_lead_cond_77 + Q_lead_joule_77
        return Q_Tc*cfg.Ntf, Q77*cfg.Ntf

    elif mode == "dwell":
        # dwell：核热减到10%，其余保持
        Q_Tc = Q1 + Q2 + cfg.nuclear_decay_factor*Q3 + Q4 + Q5 + Q_pipes + Q9_quench + Q10_misc
        Q77 = Q_lead_cond_77 + Q_lead_joule_77
        return Q_Tc*cfg.Ntf, Q77*cfg.Ntf    

    elif mode == "maint":
        # maintenance：只剩下静态热漏（Q4+Q5+Q_pipes+Q10_misc）
        Q_Tc = Q4 + Q5 + Q_pipes + Q10_misc 
        Q77 = Q_lead_cond_77 
        return Q_Tc*cfg.Ntf, Q77*cfg.Ntf   # 维护时通常不通电流，所以77K部分(lead)只剩下传导热，没有焦耳热

def plot_scaling_multiT(
    Npw, R_p2p_joint, 
    Top_list=[4.2,10,20,77], 
    N_facility_list=range(1,51),
    save_dir=None
):
    """
    在一张图上比较不同冷端温度下规模化的寄生功率
    """
    plt.figure(figsize=(8,6))
    
    for Top in Top_list:
        para_costs = []
        for Nf in N_facility_list:
            results = calculate_base_heat_loads(
                Ip=400,
                Npw=Npw,
                L_tot=cfg.L_tot,
                L_single_tape=cfg.LEN_PER_SINGEL_REBCO,
                R_ss=cfg.R_SU_JOINT,
                R_p2p_joint=R_p2p_joint,
                V_coil=cfg.V_magnet,
                A_cryostat=cfg.A_cryostat,
                eps=cfg.eps,
                N_cool_pipe=cfg.N_cool_pipe,
                N_aux_pipe=cfg.N_aux_pipe,
                Top=Top,
            )
            
            # 规模化时热负荷乘 Nf
            Q_total_Tc = results[f"heat @ {Top}K"] * Nf
            Q_total_77 = results["heat @ 77K"] * Nf

            COP_77 = achieve_COP(cfg.T_HIGH,77,Q_total_77)
            COP_Tc = achieve_COP(cfg.T_HIGH,Top,Q_total_Tc)

            P_cryo_electric_W = (Q_total_Tc/COP_Tc + Q_total_77/COP_77)
            P_gross_electric_W = cfg.fusion_output_MWth * 1e6 * cfg.eat_conv * Nf

            para_cost = P_cryo_electric_W / P_gross_electric_W * 100
            para_costs.append(para_cost)
        
        # 画曲线
        plt.plot(N_facility_list, para_costs, marker="o", label=f"Top={Top} K", color=colors[Top])
    
    plt.xlabel("Number of facilities built at site")
    #plt.xscale("log")
    #plt.yscale("log")
    plt.ylabel("Parasitic power cost (%)")
    plt.grid(True, ls="--", alpha=0.5)
    plt.legend()
    plt.title(f"Scaling effect (Npw={Npw}, R_p2p_joint={R_p2p_joint*1e9:.2f} nOhm)")
    plt.tight_layout()
    plt.savefig(save_dir)
    plt.show()

# ===========================
# 示例运行
# ===========================

if __name__ == "__main__":
    # 需要先定义 cfg.NP, cfg.V_magnet, cfg.R_p2p_joint_max 等并保证 calculate_total_heat 与 achieve_COP 在作用域内
    #先算一次 生产态基准热负荷
    
    Npw = 10 #并绕根数，20
    R_p2p_joint = cfg.R_p2p_joint_TARGET
    save_dir = cfg.OUTPUTS_FIGURES_DIR / "scaling"
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    Top_list = [4.2,10,20]
    Ip_list = [1500, 1000, 400]
    for Top, Ip in zip(Top_list, Ip_list):
        print("-"*100)
        print(f"Top = {Top} K")
        base = calculate_base_heat_loads(
            Ip=Ip, Npw=Npw, 
            R_p2p_joint=R_p2p_joint, Top=Top,
        )
        base = calculate_base_heat_loads(
            Ip=Ip, Npw=Npw, 
            R_p2p_joint=R_p2p_joint, Top=Top)
    
        res = annual_parasitic_fraction_three_stage(
            Ip=Ip, Npw=Npw, R_p2p_joint=R_p2p_joint, Top=Top,
            
            pulse_hours=cfg.pulse_hours, dwell_hours=cfg.dwell_hours,
            maintenance_hours_per_year=cfg.maintenance_hours_per_year,  # 每年 2 个月
            shared_cryo=True, N_facilities=1,
            N_cooldown_per_year=cfg.N_cooldown_per_year, E_cooldown_Wh=cfg.E_cooldown_Wh
        )

        print(f"年度寄生占比 = {res['F_parasitic_pct']:.2f}%")
        print("各阶段小时数（h）:", res['H_prod_h'], res['H_dwell_h'], res['H_maint_h'])

    
