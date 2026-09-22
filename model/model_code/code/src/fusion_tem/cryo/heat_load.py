# heat_load_model.py
import os
import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
import pandas as pd
from fusion_tem import device as cfg
from fusion_tem import materials as mat  # 导入新的材料属性文件
from fusion_tem.cryo import efficiency as cryo_eff
colors = {4.2:"tab:blue", 10:"tab:orange", 20:"tab:green", 77:"tab:red"}

# REBCO 引线热负荷计算
def lead_REBCO_heat(Top, I_operation):
    """
    根据给定的磁体运行温度和运行电流，计算稳定器截面积和传导热。
    """
    # 二元电流引线: 电阻(铜)段跨 300 K -> 77 K 中间温度(见 lead_Cu_heat),
    # HTS/稳定体段只跨 77 K -> Top。参考 Fry et al. 2024, IEEE TAS 34(2):0600518 §III.A。
    # 2026-07-26 修正 4 处(见工作文档 §14.44):
    #   (1) k 积分上限由 T_HIGH=300 K 改为 T_HOT_HTS_LEAD=77 K —— 原先把 77-300 K
    #       又积了一遍, 与 lead_Cu_heat 重复计数, 使传导热高估约 12.7 倍;
    #   (2) 绝热定尺积分上限 Temp_transfer 由 100 K 改为 150 K(Fry §III.D 判据);
    #   (3) 绝热定尺积分下限由硬编码 77 K 改为 Top(正文式 S19);
    #   (4) 初始电流用实际工作电流 I_operation, 不再乘 1/carrying_factor 的设计余量。
    # 对 Fry 的 50 kA 基准(A≈2856 mm², Q≈5 W/CL @20 K): 修正前 A=3.87x、Q=12.65x;
    # 修正后 A=1.24x、Q=0.43x, 与三维实测分析同量级。
    t_f = cfg.Time_transfer
    T_f = cfg.Temp_transfer
    T_intercept = cfg.T_HOT_HTS_LEAD
    L = cfg.L_LEAD_HTS

    E_value = (float(I_operation) ** 2 * float(t_f)) / 3.0
    integrand_G = lambda T: mat.d_ss304 * mat.Cp_ss304(T) / mat.rho_ss304(T)
    G_value, _ = quad(integrand_G, Top, T_f)

    if G_value == 0:
        return 0, 0
    A_m2 = np.sqrt(E_value / G_value)

    k_integral, _ = quad(mat.k_ss304, Top, T_intercept)
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
    return cfg.V_mag * cfg.NUCLEAR_POWER_DENSITY

# ---------------------------
# 4. 恒温器辐射热
# ---------------------------
def radiation_heat(A_surf, eps, T_hot, Top):
    N_layer = cfg.N_LAYER_CRYOSTAT
    sigma = 5.67e-8
    # Ideal identical-shield approximation used in the SI.
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

def _mode_system_heat_loads(mode: str, base_heat_loads: dict, Top: float, N_tf: int) -> tuple[float, float]:
    """Return simultaneous full-system loads at Top and 77 K."""
    if mode == "prod":
        tc_keys = ["coil_internal_joint", "pancake_joint", "nuclear", "radiation",
                   "current_leads_HTS", "pipes_coolant", "pipes_aux", "misc", "quench"]
        q_tc = sum(base_heat_loads[key] for key in tc_keys)
        q_77 = base_heat_loads["current_leads_Cu_conduction"] + base_heat_loads["current_leads_Cu_joule"]
    elif mode == "dwell":
        tc_keys = ["coil_internal_joint", "pancake_joint", "radiation", "current_leads_HTS",
                   "pipes_coolant", "pipes_aux", "misc", "quench"]
        q_tc = sum(base_heat_loads[key] for key in tc_keys) + 0.1 * base_heat_loads["nuclear"]
        q_77 = base_heat_loads["current_leads_Cu_conduction"] + base_heat_loads["current_leads_Cu_joule"]
    elif mode == "charge":
        tc_keys = ["coil_internal_joint", "pancake_joint", "radiation", "current_leads_HTS",
                   "pipes_coolant", "pipes_aux", "misc", "quench"]
        q_tc = sum(base_heat_loads[key] for key in tc_keys)
        q_77 = base_heat_loads["current_leads_Cu_conduction"] + base_heat_loads["current_leads_Cu_joule"]
    elif mode == "static":
        # V6.7 model change: current_leads_HTS added. HTS-lead conduction is a passive
        # leak present whenever the magnet is cold, independent of current flow, so it
        # belongs in the passive state exactly as it does in prod, dwell and charge.
        # V6.6 and earlier omitted it, which contradicted Note S3 Table S6, whose Static
        # column already marks HTS-lead conduction as included.
        # This mode is not confined to a nominally idle state. lcoe.py bills the cooldown
        # and warm-up hours at the static power (P_cryo_coolwarm_W = P_cryo_static_W), and
        # build_cryo_rated_context sizes the cryoplant from all four modes simultaneously,
        # so the change moves both the annual ledger and the rated-capacity COP context.
        # V6.7 results must not be mixed with V6.6 results.
        tc_keys = ["radiation", "current_leads_HTS", "pipes_coolant", "pipes_aux", "misc", "quench"]
        q_tc = sum(base_heat_loads[key] for key in tc_keys)
        q_77 = base_heat_loads["current_leads_Cu_conduction"]
    else:
        raise ValueError(f"unknown cryogenic operating mode: {mode!r}")
    q_tc_system = float(q_tc) * int(N_tf)
    q_77_system = float(q_77) * int(N_tf)
    if not np.isfinite(q_tc_system) or not np.isfinite(q_77_system) or q_tc_system < 0 or q_77_system < 0:
        raise ValueError("mode heat loads must be finite and non-negative")
    return q_tc_system, q_77_system


def build_cryo_rated_context(
    base_heat_loads: dict,
    Top: float,
    *,
    N_tf: int | None = None,
    charge_tc_system_W=None,
    charge_77_system_W=None,
    efficiency_model: str = cryo_eff.DEFAULT_CRYO_EFFICIENCY_MODEL,
    eta_max: float = cryo_eff.DEFAULT_CRYO_ETA_MAX,
    rated_margin: float = cryo_eff.DEFAULT_CRYO_RATED_MARGIN,
) -> dict:
    N_tf = cfg.Ntf if N_tf is None else int(N_tf)
    simultaneous = []
    for mode in ("prod", "dwell", "static", "charge"):
        q_tc, q_77 = _mode_system_heat_loads(mode, base_heat_loads, Top, N_tf)
        simultaneous.append({Top: q_tc, 77.0: q_77})
    if charge_tc_system_W is not None or charge_77_system_W is not None:
        if charge_tc_system_W is None or charge_77_system_W is None:
            raise ValueError("charge loads for both temperature zones are required")
        simultaneous.append({Top: charge_tc_system_W, 77.0: charge_77_system_W})
    return cryo_eff.build_rated_context(
        simultaneous, N_tf=N_tf, efficiency_model=efficiency_model,
        eta_max=eta_max, rated_margin=rated_margin, T_hot=cfg.T_HIGH,
    )

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
    # 2026-07-26 删除一段死代码: 原先构造 heat_loads 列表并对每项再乘一次 cfg.Ntf,
    # 使 Q_total_Tc 变成 Ntf² 倍。该列表既不被返回也不被使用(下方 return 用的是原始
    # 变量), 因此删除不改变任何结果; 留着反而是个雷 —— 一旦有人改成返回它就会出错。
    # 原循环里附带的负值打印仅用于调试, 一并移除。
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
def calculate_cryo_electrical_power(
    mode: str,
    base_heat_loads: dict,
    Top: float,
    N_tf=None,
    *,
    rated_context: dict | None = None,
    efficiency_model: str = cryo_eff.DEFAULT_CRYO_EFFICIENCY_MODEL,
    eta_max: float = cryo_eff.DEFAULT_CRYO_ETA_MAX,
    rated_margin: float = cryo_eff.DEFAULT_CRYO_RATED_MARGIN,
):
    """Return full-system cryoplant electric power for one operating mode."""
    N_tf = cfg.Ntf if N_tf is None else int(N_tf)
    q_tc_system, q_77_system = _mode_system_heat_loads(mode, base_heat_loads, Top, N_tf)
    if efficiency_model == "legacy_ter_brake":
        p_tc = q_tc_system / achieve_COP(cfg.T_HIGH, Top, q_tc_system) if q_tc_system > 0 else 0.0
        p_77 = q_77_system / achieve_COP(cfg.T_HIGH, 77, q_77_system) if q_77_system > 0 else 0.0
        return float(p_tc + p_77)
    if rated_context is None:
        rated_context = build_cryo_rated_context(
            base_heat_loads, Top, N_tf=N_tf, efficiency_model=efficiency_model,
            eta_max=eta_max, rated_margin=rated_margin,
        )
    return float(cryo_eff.electrical_power_W(
        {Top: q_tc_system, 77.0: q_77_system}, rated_context, T_hot=cfg.T_HIGH
    ))
def calculate_charge_cryo_electrical_energy(
    base_heat_loads: dict,
    Top: float,
    time_h,
    magnetization_loss_W,
    radial_loss_W,
    N_tf=cfg.Ntf,
    *,
    efficiency_model: str = cryo_eff.DEFAULT_CRYO_EFFICIENCY_MODEL,
    eta_max: float = cryo_eff.DEFAULT_CRYO_ETA_MAX,
    rated_margin: float = cryo_eff.DEFAULT_CRYO_RATED_MARGIN,
):
    """Integrate full-TF-system cryoplant electricity over one charge event.

    Data S2/S3 losses are cold-end loads for one TF magnet. All cold-end
    components are multiplied by N_tf before the load-dependent COP model
    is evaluated. Returned event energy is in MWh(e).
    """
    time_h = np.asarray(time_h, dtype=float)
    magnetization_loss_W = np.asarray(magnetization_loss_W, dtype=float)
    radial_loss_W = np.asarray(radial_loss_W, dtype=float)

    if time_h.ndim != 1 or time_h.size < 2:
        raise ValueError("time_h must contain at least two one-dimensional points")
    if magnetization_loss_W.shape != time_h.shape or radial_loss_W.shape != time_h.shape:
        raise ValueError("loss arrays must have the same shape as time_h")
    if not np.all(np.isfinite(time_h)) or not np.all(np.isfinite(magnetization_loss_W)) or not np.all(np.isfinite(radial_loss_W)):
        raise ValueError("time and loss arrays must be finite")
    if np.any(np.diff(time_h) <= 0):
        raise ValueError("time_h must be strictly increasing")
    if np.any(magnetization_loss_W < 0) or np.any(radial_loss_W < 0):
        raise ValueError("electromagnetic loss arrays cannot be negative")

    q_base_tc_per_tf = sum([
        base_heat_loads["coil_internal_joint"], base_heat_loads["pancake_joint"],
        base_heat_loads["radiation"], base_heat_loads["current_leads_HTS"],
        base_heat_loads["pipes_coolant"], base_heat_loads["pipes_aux"],
        base_heat_loads["misc"], base_heat_loads["quench"],
    ])
    q_77_per_tf = (
        base_heat_loads["current_leads_Cu_conduction"]
        + base_heat_loads["current_leads_Cu_joule"]
    )

    q_tc_per_tf = q_base_tc_per_tf + magnetization_loss_W + radial_loss_W
    q_tc_system = q_tc_per_tf * float(N_tf)
    q_77_system = np.full_like(time_h, q_77_per_tf * float(N_tf), dtype=float)

    q_base_tc_system = q_base_tc_per_tf * float(N_tf)
    q_base_77_system = q_77_per_tf * float(N_tf)
    rated_context = build_cryo_rated_context(
        base_heat_loads,
        Top,
        N_tf=N_tf,
        charge_tc_system_W=q_tc_system,
        charge_77_system_W=q_77_system,
        efficiency_model=efficiency_model,
        eta_max=eta_max,
        rated_margin=rated_margin,
    )
    if efficiency_model == "legacy_ter_brake":
        p_base_W = (
            q_base_tc_system / achieve_COP(cfg.T_HIGH, Top, q_base_tc_system)
            + q_base_77_system / achieve_COP(cfg.T_HIGH, 77, q_base_77_system)
        )
        cop_tc = achieve_COP(cfg.T_HIGH, Top, q_tc_system)
        cop_77 = achieve_COP(cfg.T_HIGH, 77, q_77_system)
        p_cryo_W = q_tc_system / cop_tc + q_77_system / cop_77
    else:
        p_base_W = float(cryo_eff.electrical_power_W(
            {Top: q_base_tc_system, 77.0: q_base_77_system}, rated_context, T_hot=cfg.T_HIGH
        ))
        p_cryo_W = cryo_eff.electrical_power_W(
            {Top: q_tc_system, 77.0: q_77_system}, rated_context, T_hot=cfg.T_HIGH
        )

    event_energy_MWh = float(np.trapz(p_cryo_W, time_h) / 1e6)
    duration_h = float(time_h[-1] - time_h[0])

    base_event_energy_MWh = float(p_base_W * duration_h / 1e6)
    dynamic_event_energy_MWh = float(event_energy_MWh - base_event_energy_MWh)
    if dynamic_event_energy_MWh < -1e-12:
        raise ValueError("integrated charge energy is below the charge-mode base energy")
    dynamic_event_energy_MWh = max(dynamic_event_energy_MWh, 0.0)

    return {
        "event_energy_MWh": event_energy_MWh,
        "base_event_energy_MWh": base_event_energy_MWh,
        "dynamic_event_energy_MWh": dynamic_event_energy_MWh,
        "average_power_W": event_energy_MWh * 1e6 / duration_h if duration_h > 0 else np.nan,
        "base_power_W": float(p_base_W),
        "peak_power_W": float(np.max(p_cryo_W)),
        "peak_heat_Tc_W_per_TF": float(np.max(q_tc_per_tf)),
        "heat_77K_W_per_TF": float(q_77_per_tf),
        "rated_context": rated_context,
    }



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
            V_coil=cfg.V_mag,
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
                V_coil=cfg.V_mag,
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
    # 需要先定义 cfg.NP, cfg.V_mag, cfg.R_p2p_joint_max 等并保证 calculate_total_heat 与 achieve_COP 在作用域内
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

    
