"""
Hybrid_ADP_policy.py
====================
Ultimate Hybrid Policy: 2-Stage Stochastic Programming + Approximate Dynamic Programming (VFA).
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pyomo.environ import (
    ConcreteModel, Param, Var, Objective, Constraint, RangeSet,
    SolverFactory, NonNegativeReals, minimize, Binary, value,
)

from Data.v2_SystemCharacteristics import get_fixed_data
from Data.PriceProcessRestaurant import price_model
from Data.OccupancyProcessRestaurant import next_occupancy_levels

# =============================================================================
# 0. HYPERPARAMETERS E PESI VFA
# =============================================================================
HORIZON       = 2    # Riduciamo la miopia: l'SP guarda solo 1 step avanti, poi VFA
GEN_SCENARIOS = 15   # Meno scenari grezzi
N_SCENARIOS   = 1    # K-Means=1 significa usare la PREVISIONE MEDIA ESATTA (Expected Value)

VFA_WEIGHTS = {
    0: {'T1': -20.6314, 'T2': -18.7053, 'H': 30.5908, 'price_t': 157.7467, 'price_previous': -42.691, 'Occ1': 2.3523, 'Occ2': -0.6226, 'vent_counter': 0.0, 'low_override_r1': 0.0, 'low_override_r2': 0.0, 'intercept': 71.9595},
    1: {'T1': -23.1232, 'T2': -22.4761, 'H': 34.6573, 'price_t': 128.5207, 'price_previous': 17.8191, 'Occ1': -1.8911, 'Occ2': -2.7245, 'vent_counter': 18.705, 'low_override_r1': 3.9332, 'low_override_r2': 6.1482, 'intercept': 38.8828},
    2: {'T1': -19.8738, 'T2': -18.6219, 'H': 32.629, 'price_t': 91.2015, 'price_previous': 45.8344, 'Occ1': 1.3279, 'Occ2': -3.7556, 'vent_counter': 26.2862, 'low_override_r1': 17.0778, 'low_override_r2': 23.885, 'intercept': 28.0218},
    3: {'T1': -14.4715, 'T2': -11.9419, 'H': 29.8808, 'price_t': 74.6288, 'price_previous': 47.6527, 'Occ1': 3.1659, 'Occ2': 2.1649, 'vent_counter': 10.9305, 'low_override_r1': 22.0882, 'low_override_r2': 21.6948, 'intercept': 17.8444},
    4: {'T1': -11.2325, 'T2': -11.7933, 'H': 23.5832, 'price_t': 63.8428, 'price_previous': 39.8127, 'Occ1': 5.2452, 'Occ2': 1.0054, 'vent_counter': -0.9738, 'low_override_r1': 21.565, 'low_override_r2': 21.047, 'intercept': 17.1136},
    5: {'T1': -11.8721, 'T2': -11.3412, 'H': 21.5031, 'price_t': 52.4831, 'price_previous': 34.8803, 'Occ1': 0.8439, 'Occ2': 2.1349, 'vent_counter': -0.8831, 'low_override_r1': 17.231, 'low_override_r2': 15.9669, 'intercept': 9.9187},
    6: {'T1': -12.294, 'T2': -10.8586, 'H': 18.6272, 'price_t': 44.8077, 'price_previous': 32.453, 'Occ1': -1.5675, 'Occ2': 0.6263, 'vent_counter': 0.5772, 'low_override_r1': 12.7988, 'low_override_r2': 14.8465, 'intercept': 1.1687},
    7: {'T1': -12.5534, 'T2': -10.4603, 'H': 15.6056, 'price_t': 34.8084, 'price_previous': 26.9222, 'Occ1': -0.7614, 'Occ2': 3.8736, 'vent_counter': 0.0728, 'low_override_r1': 10.5216, 'low_override_r2': 13.2053, 'intercept': -8.9961},
    8: {'T1': -8.378, 'T2': -9.2264, 'H': 11.4391, 'price_t': 25.7716, 'price_previous': 18.8258, 'Occ1': 0.2459, 'Occ2': 0.4687, 'vent_counter': 0.0443, 'low_override_r1': 11.6758, 'low_override_r2': 15.3764, 'intercept': -11.2201},
    9: {'T1': -2.1289, 'T2': -1.5791, 'H': 5.7552, 'price_t': 9.8121, 'price_previous': 7.2285, 'Occ1': 0.3075, 'Occ2': 0.9364, 'vent_counter': -0.7448, 'low_override_r1': 13.8564, 'low_override_r2': 14.3856, 'intercept': -7.6084},
}

DATA = get_fixed_data()

# =============================================================================
# 1. SCENARIO GENERATION E CLUSTERING
# =============================================================================
def generate_scenarios(price_now, price_prev, occ_r1_now, occ_r2_now, horizon, n_scenarios):
    price_dict, occ_dict = {}, {}
    for s in range(n_scenarios):
        p_cur,  p_prev  = price_now,  price_prev
        o1_cur, o2_cur  = occ_r1_now, occ_r2_now
        for t in range(horizon):
            p_next           = price_model(p_cur, p_prev)
            o1_next, o2_next = next_occupancy_levels(o1_cur, o2_cur)
            price_dict[t, s] = p_next
            occ_dict[1, t, s] = o1_next
            occ_dict[2, t, s] = o2_next
            p_prev, p_cur   = p_cur,  p_next
            o1_cur, o2_cur  = o1_next, o2_next
    return price_dict, occ_dict

def cluster_scenarios(price_dict, occ_dict, n_clusters, horizon, scenarios_to_generate):
    X = np.array([
        [price_dict[t, s] for t in range(horizon)] +
        [occ_dict[1, t, s] for t in range(horizon)] +
        [occ_dict[2, t, s] for t in range(horizon)]
        for s in range(scenarios_to_generate)
    ])
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, n_init=5, random_state=42)
    km.fit(X_scaled)
    labels        = km.labels_
    cluster_sizes = np.bincount(labels)
    probabilities = cluster_sizes / scenarios_to_generate

    centroids = scaler.inverse_transform(km.cluster_centers_)
    price_dict_clus, occ_dict_clus = {}, {}
    
    for s in range(n_clusters):
        for t in range(horizon):
            price_dict_clus[t, s]   = float(centroids[s, t])
            occ_dict_clus[1, t, s]  = float(max(0.0, centroids[s, horizon + t]))
            occ_dict_clus[2, t, s]  = float(max(0.0, centroids[s, 2*horizon + t]))

    return price_dict_clus, occ_dict_clus, probabilities

# =============================================================================
# 2. PYOMO MILP MODEL  (Hybrid: 2-stage SP + VFA)
# =============================================================================
def build_hybrid_model(state, price_dict_clus, occ_dict_clus, horizon, n_clus, probabilities):
    d = DATA    
    _num_timeslots = int(d['num_timeslots'])
    m = ConcreteModel()

    # ── Sets
    m.R = RangeSet(1, 2)
    m.T = RangeSet(0, horizon - 1)      # Orizzonte operativo (0, 1, 2)
    m.T_ext = RangeSet(0, horizon)      # Orizzonte esteso per i parametri futuri (0, 1, 2, 3)
    m.S = RangeSet(0, n_clus - 1)
    
    # ── State
    v_prev = 1 if state['vent_counter'] > 0 else 0
    m.Tinit    = Param(m.R, initialize={1: state['T1'], 2: state['T2']})
    m.Hinit    = Param(initialize=state['H'])
    m.VentInit = Param(initialize=v_prev)
    
    # ── Scenario params (Indicizzati con T_ext per non andare fuori range)
    m.O      = Param(m.R, m.T_ext, m.S, initialize=occ_dict_clus)
    m.prices = Param(m.T_ext, m.S,  initialize=price_dict_clus)
    m.pi     = Param(m.S,   initialize={s: float(probabilities[s]) for s in range(n_clus)})
    m.Tout   = Param(m.T, initialize={
        t: d['outdoor_temperature'][(state['current_time'] + t) % _num_timeslots]
        for t in range(horizon)
    })

    # ── Physical constants
    m.Pr     = Param(initialize=d['heating_max_power'])
    m.Pvent  = Param(initialize=d['ventilation_power'])
    m.Zexch  = Param(initialize=d['heat_exchange_coeff'])
    m.Zconv  = Param(initialize=d['heating_efficiency_coeff'])
    m.Zloss  = Param(initialize=d['thermal_loss_coeff'])
    m.Zcool  = Param(initialize=d['heat_vent_coeff'])
    m.Zocc   = Param(initialize=d['heat_occupancy_coeff'])
    m.Hocc   = Param(initialize=d['humidity_occupancy_coeff'])
    m.Hvent  = Param(initialize=d['humidity_vent_coeff'])
    m.Tmin   = Param(initialize=d['temp_min_comfort_threshold'])
    m.Tok    = Param(initialize=d['temp_OK_threshold'])
    m.Thigh  = Param(initialize=d['temp_max_comfort_threshold'])
    m.Hhigh  = Param(initialize=d['humidity_threshold'])
    m.L      = Param(initialize=horizon)
    m.M_temp = Param(initialize=100)
    m.M_hum  = Param(initialize=100)
    m.U_vent = Param(initialize=3)
  

    # ── Decision variables
    m.Vent   = Var(m.T, m.S,  domain=Binary)
    m.Vstart = Var(m.T, m.S,  domain=Binary)   
    m.Heat   = Var(m.R, m.T, m.S, domain=NonNegativeReals, bounds=(0, d['heating_max_power']))

    m.y_low  = Var(m.R, m.T, m.S, domain=Binary)
    m.y_ok   = Var(m.R, m.T, m.S, domain=Binary)
    m.y_high = Var(m.R, m.T, m.S, domain=Binary)
    m.u      = Var(m.R, m.T, m.S, domain=Binary)

    m.T_in = Var(m.R, m.T, m.S, domain=NonNegativeReals)
    m.Hum  = Var(m.T, m.S,  domain=NonNegativeReals)
    m.VC   = Var(m.T, m.S,  domain=NonNegativeReals) 

    # ── Dynamics
    def temp_dynamics(m, r, t, s):
        if t == 0: return m.T_in[r, t, s] == m.Tinit[r]
        tp = t - 1
        r_other = 2 if r == 1 else 1
        return m.T_in[r, t, s] == (
            m.T_in[r, tp, s]
            + m.Zexch * (m.T_in[r_other, tp, s] - m.T_in[r, tp, s])
            + m.Zloss * (m.Tout[tp] - m.T_in[r, tp, s])
            + m.Zconv * m.Heat[r, tp, s]
            - m.Zcool * m.Vent[tp, s]
            + m.Zocc  * m.O[r, tp, s]
        )
    m.TempDyn = Constraint(m.R, m.T, m.S, rule=temp_dynamics)

    def hum_dynamics(m, t, s):
        if t == 0: return m.Hum[t, s] == m.Hinit
        tp = t - 1
        return m.Hum[t, s] == (
            m.Hum[tp, s]
            - m.Hvent * m.Vent[tp, s]
            + m.Hocc  * (m.O[1, tp, s] + m.O[2, tp, s])
        )
    m.HumDyn = Constraint(m.T, m.S, rule=hum_dynamics)
    
    # ── Ventilation Counter
    def vc_dyn_1(m, t, s): return m.VC[t, s] <= 10 * m.Vent[t, s]
    m.C_vc1 = Constraint(m.T, m.S, rule=vc_dyn_1)

    def vc_dyn_2(m, t, s):
        prev = state['vent_counter'] if t == 0 else m.VC[t-1, s]
        return m.VC[t, s] <= prev + 1
    m.C_vc2 = Constraint(m.T, m.S, rule=vc_dyn_2)

    def vc_dyn_3(m, t, s):
        prev = state['vent_counter'] if t == 0 else m.VC[t-1, s]
        return m.VC[t, s] >= prev + 1 - 10 * (1 - m.Vent[t, s])
    m.C_vc3 = Constraint(m.T, m.S, rule=vc_dyn_3)

    # ── Overrules
    m.CThigh1 = Constraint(m.R, m.T, m.S, rule=lambda m, r, t, s: m.T_in[r, t, s] >= m.Thigh - m.M_temp * (1 - m.y_high[r, t, s]))
    m.CThigh2 = Constraint(m.R, m.T, m.S, rule=lambda m, r, t, s: m.T_in[r, t, s] <= m.Thigh + m.M_temp * m.y_high[r, t, s])
    m.CHeatOff = Constraint(m.R, m.T, m.S, rule=lambda m, r, t, s: m.Heat[r, t, s] <= m.Pr * (1 - m.y_high[r, t, s]))

    m.CTlow1 = Constraint(m.R, m.T, m.S, rule=lambda m, r, t, s: m.T_in[r, t, s] <= m.Tmin + m.M_temp * (1 - m.y_low[r, t, s]))
    m.CTlow2 = Constraint(m.R, m.T, m.S, rule=lambda m, r, t, s: m.T_in[r, t, s] >= m.Tmin - m.M_temp * m.y_low[r, t, s])

    m.CTok1 = Constraint(m.R, m.T, m.S, rule=lambda m, r, t, s: m.T_in[r, t, s] >= m.Tok - m.M_temp * (1 - m.y_ok[r, t, s]))
    m.CTok2 = Constraint(m.R, m.T, m.S, rule=lambda m, r, t, s: m.T_in[r, t, s] <= m.Tok + m.M_temp * m.y_ok[r, t, s])

    m.CU1 = Constraint(m.R, m.T, m.S, rule=lambda m, r, t, s: m.u[r, t, s] >= m.y_low[r, t, s])
    def c_u2(m, r, t, s):
        u_prev = state[f'low_override_r{r}'] if t == 0 else m.u[r, t - 1, s]
        return m.u[r, t, s] <= u_prev + m.y_low[r, t, s]
    m.CU2 = Constraint(m.R, m.T, m.S, rule=c_u2)
    m.CHeatMax = Constraint(m.R, m.T, m.S, rule=lambda m, r, t, s: m.Heat[r, t, s] >= m.Pr * m.u[r, t, s])

    def c_u3(m, r, t, s):
        u_prev = state.get(f'low_override_r{r}', 0) if t == 0 else m.u[r, t - 1, s]
        return m.u[r, t, s] >= u_prev - m.y_ok[r, t, s]
    m.CU3 = Constraint(m.R, m.T, m.S, rule=c_u3)
    m.CU4 = Constraint(m.R, m.T, m.S, rule=lambda m, r, t, s: m.u[r, t, s] <= 1 - m.y_ok[r, t, s])

    # ── Ventilation Inerzia
    def c_vstart1(m, t, s):
        v_p = m.VentInit if t == 0 else m.Vent[t - 1, s]
        return m.Vstart[t, s] >= m.Vent[t, s] - v_p
    m.CVstart1 = Constraint(m.T, m.S, rule=c_vstart1)
    m.CVstart2 = Constraint(m.T, m.S, rule=lambda m, t, s: m.Vstart[t, s] <= m.Vent[t, s])
    def c_vstart3(m, t, s):
        v_p = m.VentInit if t == 0 else m.Vent[t - 1, s]
        return m.Vstart[t, s] <= 1 - v_p
    m.CVstart3 = Constraint(m.T, m.S, rule=c_vstart3)

    def min_uptime(m, t, s):
        end_idx = min(t + m.U_vent - 1, m.L - 1)
        min_val = min(m.U_vent, m.L - t)
        return sum(m.Vent[tau, s] for tau in range(t, end_idx + 1)) >= min_val * m.Vstart[t, s]
    m.MinVentOn = Constraint(m.T, m.S, rule=min_uptime)

    m.CVentHum = Constraint(m.T, m.S, rule=lambda m, t, s: m.Hum[t, s] <= m.Hhigh + m.M_hum * m.Vent[t, s])

    # ── Non-anticipativity
    def heat_na(m, r, s1, s2):
        if s1 >= s2: return Constraint.Skip
        return m.Heat[r, 0, s1] == m.Heat[r, 0, s2]
    m.HeatNA = Constraint(m.R, m.S, m.S, rule=heat_na)

    def vent_na(m, s1, s2):
        if s1 >= s2: return Constraint.Skip
        return m.Vent[0, s1] == m.Vent[0, s2]
    m.VentNA = Constraint(m.S, m.S, rule=vent_na)

    # ── OBIETTIVO IBRIDO
    def hybrid_objective(m):
        operational_cost = sum(
            m.pi[s] * sum(
                m.prices[t, s] * (sum(m.Heat[r, t, s] for r in m.R) + m.Vent[t, s] * m.Pvent)
                for t in m.T
            ) for s in m.S
        )

        t_end = horizon - 1
        real_time_end = int(state['current_time']) + t_end
        future_value = 0.0

        if real_time_end < 9:
            w = VFA_WEIGHTS[real_time_end + 1]
            future_value = sum(
                m.pi[s] * (
                    w['intercept'] +   
                    w['T1'] * ((m.T_in[1, t_end, s] - 22.0) / 8.0) +  
                    w['T2'] * ((m.T_in[2, t_end, s] - 22.0) / 8.0) + 
                    w['H']  * ((m.Hum[t_end, s] - 40.0) / 40.0) +
                    w['vent_counter'] * (m.VC[t_end, s] / 3.0) + 
                    w['low_override_r1'] * m.u[1, t_end, s] + 
                    w['low_override_r2'] * m.u[2, t_end, s] +
                    w['price_t'] * (m.prices[horizon, s] / 10.0) +
                    w['price_previous'] * (m.prices[t_end, s] / 10.0) + 
                    w['Occ1'] * ((m.O[1, horizon, s] - 20.0) / 30.0) +
                    w['Occ2'] * ((m.O[2, horizon, s] - 10.0) / 20.0)
                ) for s in m.S
            )

        return operational_cost + future_value

    m.obj = Objective(rule=hybrid_objective, sense=minimize)
    return m

# =============================================================================
# 3. POLICY EXECUTION WRAPPER
# =============================================================================
def select_action(state):
    t = state['current_time']
    remaining = DATA['num_timeslots'] - t
    horizon   = min(HORIZON, remaining)

    if horizon <= 0:
        return {'HeatPowerRoom1': 0.0, 'HeatPowerRoom2': 0.0, 'VentilationON': 0}

    gen_horizon = horizon + 1
    price_dict, occ_dict = generate_scenarios(
        state["price_t"], state["price_previous"],
        state["Occ1"], state["Occ2"],
        horizon=gen_horizon, n_scenarios=GEN_SCENARIOS
    )
    price_dict_clus, occ_dict_clus, probabilities = cluster_scenarios(
        price_dict, occ_dict, N_SCENARIOS, gen_horizon, GEN_SCENARIOS
    )

    model = build_hybrid_model(state, price_dict_clus, occ_dict_clus, horizon, len(probabilities), probabilities)
    solver = SolverFactory('gurobi_direct')
    
    solver.options['TimeLimit'] = 12.0
    solver.options['MIPGap'] = 0.02
    
    try:
        solver.solve(model, tee=False)
        p1 = float(value(model.Heat[1, 0, 0]))
        p2 = float(value(model.Heat[2, 0, 0]))
        v  = int(round(float(value(model.Vent[0, 0]))))
    except:
        return {'HeatPowerRoom1': 0.0, 'HeatPowerRoom2': 0.0, 'VentilationON': 0}

    p1 = min(max(p1, 0.0), DATA['heating_max_power'])
    p2 = min(max(p2, 0.0), DATA['heating_max_power'])

    return {'HeatPowerRoom1': p1, 'HeatPowerRoom2': p2, 'VentilationON': v}

# Alias compatibile con le istruzioni dell'Assignment
def PolicyRestaurant(state):
    return select_action(state)