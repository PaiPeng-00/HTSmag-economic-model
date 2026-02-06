# heat_load_model.py
import os
import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
import pandas as pd
import config as cfg
import material_properties as mat  # import material-property helpers
colors = {4.2:"tab:blue", 10:"tab:orange", 20:"tab:green", 77:"tab:red"}

# REBCO current-lead heat-load calculation
def lead_REBCO_heat(Top, I_operation):
    """
    Compute stabiliser cross-section and conduction heat for a REBCO lead
    at a given operating temperature and current.
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

# Copper current-lead heat-load calculation
def lead_Cu_heat(I_operation, L=cfg.LEN_LEAD_CU, A=None, design_factor=cfg.design_factor, RRR=cfg.RRR):
    """
    Parameters
    ----------
    I_operation : float
        Operating current [A].
    L : float
        Lead length [m].
    A : float or None
        Cross-sectional area [m^2]. If None, computed from the design factor.
    design_factor : float
        Empirical design factor, I*L/A ≈ 5e6 A/m.
    RRR : float
        Residual-resistivity ratio of copper.

    Returns
    -------
    (Q_cond, Q_joule, Q_total, A)
        Conduction heat at 77 K, Joule heat at the cold end, total heat, and area.
    """
    # Compute cross-sectional area from maximum current
    I_max = I_operation / cfg.carrying_factor
    if A is None:
        A = I_max*L/design_factor

    # --- Conduction heat ---
    def integrand_k(T):
        return mat.thermal_conductivity_Cu(T, RRR)
    k_int, _ = quad(integrand_k, 77, 300)
    Q_cond = A/L * k_int   # W

    # --- Electrical resistance & Joule heat ---
    def integrand_rho(T):
        return mat.resistivity_Cu(T, RRR)
    rho_avg, _ = quad(integrand_rho, 77, 300)
    rho_avg /= (300-77)
    R = rho_avg * L / A
    Q_joule = 0.5 * I_operation**2 * R  # assume half of Joule heat loads the cold end

    Q_total = Q_cond + Q_joule
    return Q_cond, Q_joule, Q_total, A


# ---------------------------
# 1. Internal su-su joint Joule heat
# ---------------------------
def coil_internal_joint_heat(Npw, L_total_m,Ip, L_single_tape=cfg.LEN_PER_SINGEL_REBCO,  R_ss=cfg.R_SU_JOINT):
    N_joint = L_total_m / (L_single_tape * Npw)
    Q = (Ip**2) * R_ss * N_joint
    return Q

# ---------------------------
# 2. Pancake-to-pancake joint Joule heat
# ---------------------------
def pancake_joint_heat(Npw,  R_p2p_joint, Ip, Np=cfg.NP):
    
    I_total = float(Ip * Npw)
    Q = (I_total**2) * R_p2p_joint * (Np + 1)
    if Q<0:
        print(type(I_total), type(Ip),type(R_p2p_joint), type(Np))
        print(I_total,R_p2p_joint,Np,Q)
    return Q

# ---------------------------
# 3. Nuclear heating
# ---------------------------
def nuclear_heating():
    return cfg.V_magnet * cfg.NUCLEAR_POWER_DENSITY

# ---------------------------
# 4. Cryostat radiation
# ---------------------------
def radiation_heat(A_surf, eps, T_hot, Top):
    N_layer = cfg.N_LAYER_CRYOSTAT
    sigma = 5.67e-8
    eps_eff = 2* eps /(N_layer+1)
    return  sigma * A_surf * eps_eff* (T_hot**4 - Top**4) 

# ---------------------------
# 5. Current-lead heat
# ---------------------------
def current_lead_heat(Ip, Npw, Top, N_lead=cfg.N_LEAD, L_cu=cfg.LEN_LEAD_CU, design_factor=cfg.design_factor, RRR=cfg.RRR):
    I_operation = Ip * Npw  # total operating current per lead
    Qlead_cond_77, Qlead_joule_77, Qlead_total_77, A_Cu = lead_Cu_heat(I_operation=I_operation, L=L_cu,
                                   design_factor=cfg.design_factor, RRR=cfg.RRR)
    Q_lead_hts, A_REBCO = lead_REBCO_heat(Top, I_operation=I_operation)
    return Qlead_cond_77 * N_lead , Qlead_joule_77 * N_lead, Q_lead_hts * N_lead

# ---------------------------
# 6. Generic pipe/wall conduction
# ---------------------------
def conduction_heat(A, L, T_hot, Top, k_func=mat.k_ss304):
    integral, _ = quad(k_func, Top, T_hot)
    return A / L * integral

# ---------------------------
# 7. Conduction through generic piping
def pipe_heat(N_pipe, d_in, d_out, L, T_hot, Top, k_func=mat.k_ss304):
    """Compute the total conductive heat for a given pipe family."""
    A = np.pi * (d_out**2 - d_in**2) / 4.0   # cross-sectional area [m^2]
    Q_single = conduction_heat(A, L, T_hot, Top, k_func)
    return N_pipe * Q_single

# ---------------------------
# 8. Quench heat (placeholder)
# ---------------------------
def quench_heat():
    return 0.0

# ---------------------------
# 9. Structural supports + residual gas
# ---------------------------
def misc_heat():
    return cfg.Q_OTHER_SOURCES

# 10.

def achieve_COP(Th, Tl, Wheat):
    """Compute the effective COP of a cryoplant."""
    if Th == Tl:
        eat_Carnot = 1
    else:
        eat_Carnot = Tl / (Th - Tl)
    # Empirical relation: achieved COP as a fraction of Carnot COP, as a
    # function of heat load Wheat.
    eta_achieved = 10**(-0.92237 + 0.07763 * np.log10(1 + Wheat))
    return eat_Carnot * eta_achieved

def calculate_base_heat_loads(Ip,Npw,R_p2p_joint, Top=20):
    """
    Compute all steady-state heat-load components for one device in
    production mode. This forms the basis for subsequent analyses.
    """
    L_HTS_TF_m = cfg.L_HTS_TF_m[Top]
    Q1 = coil_internal_joint_heat(Npw =Npw, L_total_m=L_HTS_TF_m, Ip=Ip, R_ss=cfg.R_SU_JOINT)
    Q2 = pancake_joint_heat(Npw =Npw, R_p2p_joint=R_p2p_joint, Ip=Ip)
    Q3 = nuclear_heating()
    Q4 = radiation_heat(A_surf=cfg.A_cryostat,  eps=cfg.eps, T_hot=cfg.T_HIGH, Top=Top)
    Q_lead_cond_77, Q_lead_joule_77, Q_lead_hts = current_lead_heat(Ip=Ip, Npw=Npw, Top=Top)
    # Current leads consist of:
    # - copper section between 77 K and room temperature (conduction + Joule)
    # - HTS section between Top and 77 K (conduction only, accounted in Q5)
    Q5 = Q_lead_hts

    # Cooling pipes
    Q7 = pipe_heat(N_pipe=cfg.N_cool_pipe, d_in=cfg.d_in_cool, d_out=cfg.d_out_cool, L=cfg.L_cool, T_hot=cfg.T_HIGH, Top=Top)
    # Auxiliary pipes
    Q8 = pipe_heat(N_pipe=cfg.N_aux_pipe, d_in=cfg.d_in_aux, d_out=cfg.d_out_aux, L=cfg.L_aux, T_hot=cfg.T_HIGH, Top=Top)
    
    Q9 = quench_heat()
    Q10 = misc_heat()

    Q_Tc = Q1 + Q2 + Q3 + Q4 + Q5 + Q7 + Q8 + Q9 + Q10

    Q_total_Tc = Q_Tc * cfg.Ntf
    Q_total_77 = (Q_lead_cond_77 + Q_lead_joule_77) * cfg.Ntf
    COP_77 = achieve_COP(cfg.T_HIGH, 77, Q_total_77)
    COP_Tc = achieve_COP(cfg.T_HIGH, Top, Q_total_Tc)
    heat_loads = [Q1, Q2, Q3, Q4, Q5, Q7, Q8, Q9, Q10, Q_total_Tc, Q_total_77, COP_77, COP_Tc]
    for i in range(len(heat_loads)):
        heat_loads[i] = heat_loads[i] * cfg.Ntf
        if heat_loads[i] < 0:
            print(f"i={i}", Q2, R_p2p_joint, Ip**2 * R_p2p_joint)

    P_cryo_electric_W = (Q_total_Tc / COP_Tc + Q_total_77 / COP_77)
    P_fusion_electric_W = cfg.P_fusion_W * cfg.eat_conv

    para_cost = P_cryo_electric_W / P_fusion_electric_W
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
        "P_fusion_electric_W":P_fusion_electric_W,
        "para_cost(%)":para_cost*100
    }
def calculate_cryo_electrical_power(mode: str, base_heat_loads: dict, Top: float, N_tf = 18):
    """
    Compute total cryogenic electrical power for a given operating mode.

    Args:
        mode: 'prod', 'dwell', or 'static'.
        base_heat_loads: dictionary from calculate_base_heat_loads().
        Top: cold-end temperature (K).
        N_tf: number of TF coils.

    Returns:
        Total cryogenic electrical power (W).
    """
    Q_Tc_components = 0
    Q_77_components = 0
    
    # Nuclear-heating decay factor in dwell mode
    nuclear_decay_factor = 0.1

    # Accumulate heat-load components by mode
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
        # During maintenance: no transport current and no fusion heating.
        Q_Tc_components = sum([
            base_heat_loads["radiation"], base_heat_loads["pipes_coolant"], base_heat_loads["pipes_aux"], base_heat_loads["misc"], base_heat_loads["quench"]
        ])
        # Leads have no current: only static conduction from 77 K to 300 K.
        Q_77_components = base_heat_loads["current_leads_Cu_conduction"]
    
    # Total heat load, scaled by number of TF coils
    Q_total_Tc = Q_Tc_components * N_tf
    Q_total_77 = Q_77_components * N_tf

    # Convert heat load to electrical power via COP
    T_hot = cfg.T_HIGH  # assume ambient/hot temperature is 300 K

    P_cryo_Tc_W = Q_total_Tc / achieve_COP(T_hot, Top, Q_total_Tc) if Q_total_Tc > 0 else 0
    P_cryo_77_W = Q_total_77 / achieve_COP(T_hot, 77, Q_total_77) if Q_total_77 > 0 else 0
    
    return P_cryo_Tc_W + P_cryo_77_W

def plot_scaling_effect(
    Npw, R_p2p_joint, Top=20, 
    N_facility_list=range(1,51),   # default: study 1–50 facilities
    save_dir=None
):
    """
    Plot how parasitic power fraction changes with the number of facilities
    built at a single site (scaling effect with shared cryoplant).
    """
    para_costs = []
    
    # Evaluate parasitic fraction for each facility count
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
        
        # Scale heat loads with Nf when multiple facilities share the site
        Q_total_Tc = results[f"heat @ {Top}K"] * Nf
        Q_total_77 = results["heat under 77K"] * Nf
        
        COP_77 = achieve_COP(cfg.T_HIGH,77,Q_total_77)
        COP_Tc = achieve_COP(cfg.T_HIGH,Top,Q_total_Tc)
        
        P_cryo_electric_W = (Q_total_Tc/COP_Tc + Q_total_77/COP_77)
        P_fusion_electric_W = cfg.P_fusion_W*cfg.eat_conv * Nf  # fusion power scales with Nf
        
        para_cost = P_cryo_electric_W / P_fusion_electric_W * 100
        para_costs.append(para_cost)
    
    # Plot scaling curve
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
    Top=20,           # K, target cold-end temperature in production
    # --- time scheduling parameters ---
    pulse_hours=2.0,           # production-pulse duration (h)
    dwell_hours=10.0/60.0,     # dwell/charging duration (h); default 10 min
    maintenance_hours_per_year=24*30*2,  # default: 2 months ≈ 1440 h

    # --- shared cryoplant options ---
    shared_cryo=False,         # True: multiple units share a large cryoplant
    N_facilities=1,            # number of units built at the same site
    # --- one-off cooldown/warmup energies (Wh per event) ---
    N_cooldown_per_year=0, E_cooldown_Wh=0.0,
    N_warmup_per_year=0, E_warmup_Wh=0.0,
    # --- thermal environment ---
    T_hot=cfg.T_HIGH
):
    """
    Return a dictionary with annual parasitic fraction, hours and energy
    spent in each phase, and intermediate quantities.
    """
    # 1) Baseline steady-state heat loads per device (production mode)
    base = calculate_base_heat_loads(Ip=Ip, Npw=Npw,R_p2p_joint=R_p2p_joint,Top=Top)
    Q4_base_per_device_W = base[f"heat @ {Top}K"]
    Q77_base_per_device_W = base["heat @ 77K"]
    P_fusion_elec_per_device_W = base["P_fusion_electric_W"]  # net electric output

    # 2) Annual time allocation
    remaining_hours = cfg.HOURS_PER_YEAR - maintenance_hours_per_year
    cycle_hours = pulse_hours + dwell_hours
    if cycle_hours <= 0:
        raise ValueError("pulse_hours + dwell_hours must be > 0")
    n_cycles = remaining_hours / cycle_hours
    # If remaining_hours < 0, maintenance exceeds 8760 h/year -> invalid
    if remaining_hours <= 0:
        raise ValueError("maintenance_hours_per_year >= 8760 -> no runtime left")

    H_prod = n_cycles * pulse_hours
    H_dwell = n_cycles * dwell_hours
    H_maint = maintenance_hours_per_year

    # 3) Heat loads per phase for the full TF system (using scaling helper)
    Q4_prod_per_device, Q77_prod_per_device = scale_base_heat_loads(base, mode="prod")
    Q4_dwell_per_device, Q77_dwell_per_device   = scale_base_heat_loads(base, mode="dwell")
    Q4_maint_per_device, Q77_maint_per_device = scale_base_heat_loads(base, mode="maint")


    # 4) Include facility count:
    #    if shared_cryo: combine loads then compute COP;
    #    else: compute per-device power then scale.
    def cryo_power_for_loads(Q4_total_W, Q77_total_W, Top_local):
        """Given total Q4 and Q77, compute cryoplant electrical power (W)."""
        P = 0.0
        if Q4_total_W > 0:
            COP4 = achieve_COP(cfg.T_HIGH, Top_local, Q4_total_W)  # Note: achieve_COP expects Wheat = total heat load
            P += Q4_total_W / COP4
        if Q77_total_W > 0:
            COP77 = achieve_COP(cfg.T_HIGH, 77.0, Q77_total_W)
            P += Q77_total_W / COP77
        return P  # W

    # 4a) if shared_cryo: combine loads over all units then compute COP
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
        # Per-device COP/power, then scale up by N_facilities
        P_cryo_prod_per_device_W = cryo_power_for_loads(Q4_prod_per_device, Q77_prod_per_device, Top)
        P_cryo_dwell_per_device_W = cryo_power_for_loads(Q4_dwell_per_device, Q77_dwell_per_device, Top)
        P_cryo_maint_per_device_W = cryo_power_for_loads(Q4_maint_per_device, Q77_maint_per_device, Top)

        P_cryo_prod_total_W = P_cryo_prod_per_device_W * N_facilities
        P_cryo_dwell_total_W = P_cryo_dwell_per_device_W * N_facilities
        P_cryo_maint_total_W = P_cryo_maint_per_device_W * N_facilities

    # 5) Annual compressor energy (Wh)
    E_cryo_prod_Wh = P_cryo_prod_total_W * H_prod
    E_cryo_dwell_Wh = P_cryo_dwell_total_W * H_dwell
    E_cryo_maint_Wh = P_cryo_maint_total_W * H_maint

    # One-off cooldown/warmup energies
    E_cooldown_Wh = N_cooldown_per_year * E_cooldown_Wh
    E_warmup_Wh = N_warmup_per_year * E_warmup_Wh

    E_cryo_year_Wh = E_cryo_prod_Wh + E_cryo_dwell_Wh + E_cryo_maint_Wh + E_cooldown_Wh + E_warmup_Wh

    # 6) Annual net electric generation (production only)
    E_out_year_Wh = P_fusion_elec_per_device_W * N_facilities * H_prod

    # 7) Parasitic fraction (by energy)
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
        "P_fusion_elec_per_device_W": P_fusion_elec_per_device_W
    }

def scale_base_heat_loads(results_base, mode="prod", Top=20):
    """
    For a given operating phase (prod/dwell/maint), return scaled
    (Q_Tc, Q_77) for the full TF system.
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
    
    # Total piping heat
    Q_pipes = Q7_coolant + Q8_aux

    if mode == "prod":
        # Production: original values for the full TF system (one device)
        Q_Tc = Q1 + Q2 + Q3 + Q4 + Q5 + Q_pipes + Q9_quench + Q10_misc
        Q77 = Q_lead_cond_77 + Q_lead_joule_77
        return Q_Tc*cfg.Ntf, Q77*cfg.Ntf

    elif mode == "dwell":
        # Dwell: nuclear heating reduced to 10%, others unchanged
        Q_Tc = Q1 + Q2 + cfg.nuclear_decay_factor*Q3 + Q4 + Q5 + Q_pipes + Q9_quench + Q10_misc
        Q77 = Q_lead_cond_77 + Q_lead_joule_77
        return Q_Tc*cfg.Ntf, Q77*cfg.Ntf    

    elif mode == "maint":
        # Maintenance: only static heat leaks (Q4+Q5+Q_pipes+Q10_misc)
        Q_Tc = Q4 + Q5 + Q_pipes + Q10_misc 
        Q77 = Q_lead_cond_77 
        # In maintenance there is usually no current in the leads, so the 77 K
        # section only has conduction, no Joule heat.
        return Q_Tc*cfg.Ntf, Q77*cfg.Ntf

def plot_scaling_multiT(
    Npw, R_p2p_joint, 
    Top_list=[4.2,10,20,77], 
    N_facility_list=range(1,51),
    save_dir=None
):
    """
    Compare parasitic power cost versus facility count for multiple
    cold-end temperatures on a single figure.
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
            
            # Scale heat loads with Nf
            Q_total_Tc = results[f"heat @ {Top}K"] * Nf
            Q_total_77 = results["heat @ 77K"] * Nf

            COP_77 = achieve_COP(cfg.T_HIGH,77,Q_total_77)
            COP_Tc = achieve_COP(cfg.T_HIGH,Top,Q_total_Tc)

            P_cryo_electric_W = (Q_total_Tc/COP_Tc + Q_total_77/COP_77)
            P_fusion_electric_W = cfg.P_fusion_W*cfg.eat_conv * Nf

            para_cost = P_cryo_electric_W / P_fusion_electric_W * 100
            para_costs.append(para_cost)
        
        # Plot curve for this Top
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
# Example usage
# ===========================

if __name__ == "__main__":
    # Example usage / sanity checks

    Npw = 10  # example number of parallel tapes
    R_p2p_joint = cfg.R_p2p_joint_TARGET
    save_dir = cfg.OUTPUTS_FIGURES_DIR / "scaling"
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    Top_list = [4.2,10,20]
    Ip_list = [1500, 1000, 400]
    for Top, Ip in zip(Top_list, Ip_list):
        print("-" * 100)
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
            maintenance_hours_per_year=cfg.maintenance_hours_per_year,
            shared_cryo=True, N_facilities=1,
            N_cooldown_per_year=cfg.N_cooldown_per_year, E_cooldown_Wh=cfg.E_cooldown_Wh
        )

        print(f"Annual parasitic fraction = {res['F_parasitic_pct']:.2f}%")
        print("Stage hours (h):", res['H_prod_h'], res['H_dwell_h'], res['H_maint_h'])

    
