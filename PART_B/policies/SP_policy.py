"""
SP_policy.py
============
2-Stage Stochastic Programming policy for the restaurant HVAC system.

Compatible with Environment.py:
    from policies.SP_policy import select_action

The grader-compatible wrapper `select_action(state)` is defined at the
bottom of this file. All helper functions (scenario generation, clustering,
model building) are self-contained here.

Design choices
--------------
  Lookahead horizon  : HORIZON = 4 steps
  Raw MC scenarios   : GEN_SCENARIOS = 50
  Clustered scenarios: N_SCENARIOS   = 10   (K-Means centroids)
  Non-anticipativity : enforced only at t = 0 (here-and-now, 2-stage)
  Solver             : Gurobi, TimeLimit = 12 s, MIPGap = 2 %
"""

from unittest import result

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from pyomo.environ import (
    ConcreteModel, Set, Param, Var, Objective, Constraint,
    SolverFactory, NonNegativeReals, minimize, Binary, value,
)

# ── Module-level RNG (re-seeded once per import) ───────────────────────────────
#_rng = np.random.default_rng(seed=42)

# ── Hyper-parameters ───────────────────────────────────────────────────────────
HORIZON       = 4    # lookahead steps  (must be >= 3 due to vent-inertia)
GEN_SCENARIOS = 50  # Monte-Carlo draws before clustering
N_SCENARIOS   = 10   # K-Means clusters (representative scenarios)

# =============================================================================
# 1. SYSTEM PARAMETERS
# =============================================================================

def get_fixed_data():
    """
    Returns the fixed system characteristics.
    THIS FUNCTION SHOULD NOT BE CHANGED.
    """
    num_timeslots = 10
    return {
        'num_timeslots'               : num_timeslots,
        'initial_temperature'         : 21.0,
        'previous_initial_temperature': 21.0,
        'initial_humidity'            : 40.0,
        'heating_max_power'           : 3.0,    # Pr  (kW)
        'heat_exchange_coeff'         : 0.6,    # ζ_exch
        'heating_efficiency_coeff'    : 1.0,    # ζ_conv
        'thermal_loss_coeff'          : 0.1,    # ζ_loss
        'heat_vent_coeff'             : 0.7,    # ζ_cool
        'heat_occupancy_coeff'        : 0.02,   # ζ_occ
        'temp_min_comfort_threshold'  : 18.0,   # T_low
        'temp_OK_threshold'           : 22.0,   # T_OK
        'temp_max_comfort_threshold'  : 26.0,   # T_high
        'humidity_threshold'          : 70.0,   # H_high
        'vent_min_up_time'            : 3,      # minimum consecutive ON hours
        'ventilation_power'           : 2.0,    # P_vent (kW)
        'humidity_occupancy_coeff'    : 0.18,   # η_occ
        'humidity_vent_coeff'         : 15.0,   # η_vent
        'outdoor_temperature'         : [
            3 * np.sin(2 * np.pi * t / num_timeslots - np.pi / 2)
            for t in range(num_timeslots)
        ],
    }


DATA = get_fixed_data()


# =============================================================================
# 2. STOCHASTIC PROCESS MODELS
# =============================================================================

def price_model(current_price, previous_price, rng=None):
    """
    One-step-ahead electricity price sample (AR(2)-like with mean reversion).

    Parameters
    ----------
    current_price  : float  – price at t
    previous_price : float  – price at t-1
    rng            : numpy Generator (optional); uses np.random if None

    Returns
    -------
    float – sampled price at t+1
    """
    mean_price         = 4.0
    reversion_strength = 0.12
    price_cap          = 12.0
    price_floor        = 0.0

    mean_reversion = reversion_strength * (mean_price - current_price)
    noise = (rng.normal(0, 0.5) if rng is not None
             else np.random.normal(0, 0.5))

    next_price = (current_price
                  + 0.6 * (current_price - previous_price)
                  + mean_reversion
                  + noise)

    if next_price < 0:
        rand_val = rng.random() if rng is not None else np.random.rand()
        if rand_val > 0.2:
            next_price = (rng.uniform(0, mean_price * 0.3) if rng is not None
                          else np.random.uniform(0, mean_price * 0.3))

    return float(np.clip(next_price, price_floor, price_cap))


def next_occupancy_levels(r1_current, r2_current, rng=None):
    """
    One-step-ahead occupancy sample for both rooms (coupled mean-reverting).

    Returns
    -------
    (r1_next, r2_next) : (float, float)
    """
    mean_r1, mean_r2 = 35.0, 25.0
    rev      = 0.25
    coupling = 0.1

    noise_r1 = (rng.normal(0, 3.0) if rng is not None
                else np.random.normal(0, 3.0))
    noise_r2 = (rng.normal(0, 2.5) if rng is not None
                else np.random.normal(0, 2.5))

    r1_next = (r1_current
               + rev      * (mean_r1 - r1_current)
               + coupling * (r2_current - r1_current)
               + noise_r1)

    r2_next = (r2_current
               + rev      * (mean_r2 - r2_current)
               + coupling * (r1_current - r2_current)
               + noise_r2)

    return float(np.clip(r1_next, 20, 50)), float(np.clip(r2_next, 10, 30))


# =============================================================================
# 3. SCENARIO GENERATION
# =============================================================================

def generate_scenarios(price_now, price_prev,
                       occ_r1_now, occ_r2_now,
                       horizon, n_scenarios, rng):
    """
    Draw *n_scenarios* independent Monte-Carlo sample paths over *horizon*
    steps, starting from the current observed state.

    Returns
    -------
    price_dict   : {(t, s): float}        – price at lookahead step t, scenario s
    occ_dict     : {(r, t, s): float}     – occupancy of room r
    hum_occ_dict : {(t, s): float}        – total occupancy (both rooms)
    """
    price_dict   = {}
    occ_dict     = {}
    hum_occ_dict = {}

    for s in range(n_scenarios):
        p_cur,  p_prev  = price_now,  price_prev
        o1_cur, o2_cur  = occ_r1_now, occ_r2_now

        for t in range(horizon):
            p_next           = price_model(p_cur, p_prev, rng)
            o1_next, o2_next = next_occupancy_levels(o1_cur, o2_cur, rng)

            price_dict[t, s]   = p_next
            occ_dict[1, t, s]  = o1_next
            occ_dict[2, t, s]  = o2_next
            hum_occ_dict[t, s] = o1_next + o2_next

            p_prev, p_cur   = p_cur,  p_next
            o1_cur, o2_cur  = o1_next, o2_next

    return price_dict, occ_dict, hum_occ_dict


# =============================================================================
# 4. SCENARIO CLUSTERING  (K-Means → weighted centroids)
# =============================================================================

def cluster_scenarios(price_dict, occ_dict, n_clusters, horizon, scenarios_to_generate):
    """
    Reduce *scenarios_to_generate* Monte-Carlo paths to *n_clusters*
    representative centroids via K-Means, returning Pyomo-ready dicts.

    Parameters
    ----------
    price_dict            : {(t, s): float}
    occ_dict              : {(r, t, s): float}
    n_clusters            : int   – number of clusters (K)
    horizon               : int   – lookahead horizon
    scenarios_to_generate : int   – total raw scenarios (= len of s-axis)

    Returns
    -------
    price_dict_clus   : {(t, s): float}        – clustered price scenarios
    occ_dict_clus     : {(r, t, s): float}     – clustered occupancy
    hum_occ_dict_clus : {(t, s): float}        – clustered total occupancy
    probabilities     : np.ndarray shape (n_clusters,)  – cluster weights
    X                 : np.ndarray – raw feature matrix (for diagnostics)
    labels            : np.ndarray – cluster labels for each raw scenario
    centroids         : np.ndarray – centroid matrix in original scale
    """
    # Build feature matrix: each row = one scenario trajectory
    # columns = [price_t0, ..., price_tH-1, occ1_t0, ..., occ2_tH-1]
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

    # Unpack centroids back to original scale
    centroids = scaler.inverse_transform(km.cluster_centers_)

    price_dict_clus   = {}
    occ_dict_clus     = {}
    hum_occ_dict_clus = {}

    for s in range(n_clusters):
        for t in range(horizon):
            price_dict_clus[t, s]   = float(np.clip(centroids[s, t],                0, 12))
            occ_dict_clus[1, t, s]  = float(np.clip(centroids[s, horizon + t],   20, 50))
            occ_dict_clus[2, t, s]  = float(np.clip(centroids[s, 2*horizon + t], 10, 30))
            hum_occ_dict_clus[t, s] = occ_dict_clus[1, t, s] + occ_dict_clus[2, t, s]

    return price_dict_clus, occ_dict_clus, hum_occ_dict_clus, probabilities, X, labels, centroids


# =============================================================================
# 5. PYOMO MILP MODEL  (2-stage SP)
# =============================================================================

def build_sp_model(current_state, price_dict, occ_dict, hum_occ_dict,
                   horizon, n_scenarios, probabilities):
    """
    Build the 2-stage stochastic MILP for the SP policy.

    Parameters
    ----------
    current_state : dict – keys: T_in_r1, T_in_r2, humidity,
                                 vent_prev, vent_on_count
    price_dict    : {(t, s): float}
    occ_dict      : {(r, t, s): float}
    hum_occ_dict  : {(t, s): float}
    horizon       : int
    n_scenarios   : int
    probabilities : array-like, shape (n_scenarios,)

    Returns
    -------
    Pyomo ConcreteModel (unsolved)
    """

    d      = DATA
    M      = 200.0   # big-M (larger than any realistic T or H range)

    T_init = {1: current_state['T_in_r1'], 2: current_state['T_in_r2']}
    H_init = current_state['humidity']
    v_prev = int(current_state['vent_prev'])
    v_on_h = int(current_state.get('vent_on_count', 0))

    m = ConcreteModel()

    # ── Sets ──────────────────────────────────────────────────────────────────
    m.R   = Set(initialize=[1, 2])
    m.T   = Set(initialize=list(range(horizon)))
    m.S   = Set(initialize=list(range(n_scenarios)))
    m.RTS = m.R * m.T * m.S
    m.TS  = m.T * m.S

    # ── Parameters ────────────────────────────────────────────────────────────
    m.Pr    = Param(initialize=d['heating_max_power'])
    m.Zexch = Param(initialize=d['heat_exchange_coeff'])
    m.Zconv = Param(initialize=d['heating_efficiency_coeff'])
    m.Zloss = Param(initialize=d['thermal_loss_coeff'])
    m.Zcool = Param(initialize=d['heat_vent_coeff'])
    m.Zocc  = Param(initialize=d['heat_occupancy_coeff'])
    m.Tmin  = Param(initialize=d['temp_min_comfort_threshold'])
    m.Tok   = Param(initialize=d['temp_OK_threshold'])
    m.Thigh = Param(initialize=d['temp_max_comfort_threshold'])
    m.Hhigh = Param(initialize=d['humidity_threshold'])
    m.Pvent = Param(initialize=d['ventilation_power'])
    m.Hocc  = Param(initialize=d['humidity_occupancy_coeff'])
    m.Hvent = Param(initialize=d['humidity_vent_coeff'])
    m.Tout  = Param(m.T, initialize={t: d['outdoor_temperature'][t]
                                     for t in range(horizon)})
    m.pi    = Param(m.S, initialize={s: probabilities[s]
                                     for s in range(n_scenarios)})

    m.O      = Param(m.RTS, initialize=occ_dict)
    m.prices = Param(m.TS,  initialize=price_dict)
    m.HumOcc = Param(m.TS,  initialize=hum_occ_dict)

    # ── Decision variables ────────────────────────────────────────────────────
    m.Heat = Var(m.RTS, domain=NonNegativeReals, bounds=(0, d['heating_max_power']))
    m.Vent = Var(m.TS,  domain=Binary)
    m.Uon  = Var(m.TS,  domain=Binary)   # ventilation start-up indicator
    m.Uoff = Var(m.TS,  domain=Binary)   # ventilation shut-down indicator

    # ── Auxiliary binaries for overrule controllers ───────────────────────────
    m.u = Var(m.RTS, domain=Binary)   # 1 → low-temp overrule active
    m.w = Var(m.RTS, domain=Binary)   # 1 → temperature recovered to T_OK
    m.y = Var(m.RTS, domain=Binary)   # 1 → high-temp overrule active

    # ── State variables ───────────────────────────────────────────────────────
    m.T_in = Var(m.RTS, domain=NonNegativeReals)
    m.Hum  = Var(m.TS,  domain=NonNegativeReals)

    # ── Temperature dynamics ──────────────────────────────────────────────────
    def temp_dynamics(m, r, t, s):
        if t == 0:
            return m.T_in[r, t, s] == T_init[r]
        tp      = t - 1
        other_r = 2 if r == 1 else 1
        return m.T_in[r, t, s] == (
            m.T_in[r, tp, s]
            + m.Zexch * (m.T_in[other_r, tp, s] - m.T_in[r, tp, s])
            + m.Zloss * (m.Tout[tp]              - m.T_in[r, tp, s])
            + m.Zconv * m.Heat[r, tp, s]
            - m.Zcool * m.Vent[tp, s]
            + m.Zocc  * m.O[r, tp, s]
        )
    m.TempDyn = Constraint(m.RTS, rule=temp_dynamics)

    # ── Humidity dynamics ─────────────────────────────────────────────────────
    def hum_dynamics(m, t, s):
        if t == 0:
            return m.Hum[t, s] == H_init
        tp = t - 1
        return m.Hum[t, s] == (
            m.Hum[tp, s]
            - m.Hvent * m.Vent[tp, s]
            + m.Hocc  * m.HumOcc[tp, s]
        )
    m.HumDyn = Constraint(m.TS, rule=hum_dynamics)

    # ── Overrule controller: LOW temperature ──────────────────────────────────
    def u_activation(m, r, t, s):
        return m.T_in[r, t, s] >= m.Tmin - M * m.u[r, t, s]
    m.UActivation = Constraint(m.RTS, rule=u_activation)

    def w_deactivation(m, r, t, s):
        return m.T_in[r, t, s] >= m.Tok - M * (1 - m.w[r, t, s])
    m.WDeactivation = Constraint(m.RTS, rule=w_deactivation)

    def u_persistence(m, r, t, s):
        if t == 0:
            init_u = 1 if current_state.get(f'low_override_r{r}', 0) else 0
            m.u[r, t, s].fix(init_u)
            m.w[r, t, s].fix(0)
            return Constraint.Skip
        return m.u[r, t, s] >= m.u[r, t - 1, s] - m.w[r, t, s]
    m.UPersistence = Constraint(m.RTS, rule=u_persistence)

    def heat_max_when_overrule(m, r, t, s):
        return m.Heat[r, t, s] >= m.Pr * m.u[r, t, s]
    m.HeatMaxOverrule = Constraint(m.RTS, rule=heat_max_when_overrule)

    # ── Overrule controller: HIGH temperature ─────────────────────────────────
    def y_activation(m, r, t, s):
        return m.T_in[r, t, s] <= m.Thigh + M * m.y[r, t, s]
    m.YActivation = Constraint(m.RTS, rule=y_activation)

    def heat_off_when_overrule(m, r, t, s):
        return m.Heat[r, t, s] <= m.Pr * (1 - m.y[r, t, s])
    m.HeatOffOverrule = Constraint(m.RTS, rule=heat_off_when_overrule)

    # ── Humidity overrule: force ventilation ON when humid ────────────────────
    def vent_humidity_overrule(m, t, s):
        return m.Hum[t, s] <= m.Hhigh + M * m.Vent[t, s]
    m.VentHumOverrule = Constraint(m.TS, rule=vent_humidity_overrule)

    # ── Ventilation inertia: 3-hour minimum ON time ───────────────────────────
    def on_off_exclusivity(m, t, s):
        return m.Uon[t, s] + m.Uoff[t, s] <= 1
    m.OnOffExcl = Constraint(m.TS, rule=on_off_exclusivity)

    def uoff_bound(m, t, s):
        return m.Uoff[t, s] <= 1 - m.Vent[t, s]
    m.UoffBound = Constraint(m.TS, rule=uoff_bound)

    def uon_bound(m, t, s):
        return m.Uon[t, s] <= 1 - m.Vent[t, s]
    m.UonBound = Constraint(m.TS, rule=uon_bound)

    def min_uptime(m, t, s):
        L         = 3
        remaining = [k for k in range(t, t + L) if k <= m.T.last()]
        v_p       = v_prev if t == 0 else m.Vent[t - 1, s]
        return (sum(m.Vent[k, s] for k in remaining)
                >= len(remaining) * (m.Vent[t, s] - v_p))
    m.MinUptime = Constraint(m.TS, rule=min_uptime)

    # Carry over inertia commitment from previous real-system hours
    if v_prev == 1 and v_on_h < 3:
        forced_on = 3 - v_on_h
        for t in range(min(forced_on, horizon)):
            for s in range(n_scenarios):
                m.Vent[t, s].fix(1)

    # ── Non-anticipativity at t = 0 (here-and-now decisions) ─────────────────
    def heat_na(m, r, s1, s2):
        if s1 >= s2:
            return Constraint.Skip
        return m.Heat[r, 0, s1] == m.Heat[r, 0, s2]
    m.HeatNA = Constraint(m.R, m.S, m.S, rule=heat_na)

    def vent_na(m, s1, s2):
        if s1 >= s2:
            return Constraint.Skip
        return m.Vent[0, s1] == m.Vent[0, s2]
    m.VentNA = Constraint(m.S, m.S, rule=vent_na)

    # ── Objective: minimise expected electricity cost ─────────────────────────
    def objective(m):
        return sum(
            m.pi[s] * sum(
                m.prices[t, s] * (
                    sum(m.Heat[r, t, s] for r in m.R)
                    + m.Vent[t, s] * m.Pvent
                )
                for t in m.T
            )
            for s in m.S
        )
    m.obj = Objective(rule=objective, sense=minimize)

    return m


# =============================================================================
# 6. APPLY DYNAMICS  (advance the real system state by one step)
# =============================================================================
"""
def apply_dynamics(state, decisions, data):
    
    Advance the real system state by one timestep using the true dynamics.

    Parameters
    ----------
    state     : dict – current state (keys defined in Environment.py)
    decisions : tuple (heat_r1, heat_r2, vent)
    data      : dict – fixed system data from get_fixed_data()

    Returns
    -------
    next_state : dict – state advanced by one step.
        'price_now' is set to None (environment fills the new observed price).
        'price_prev' is set to state['price_t'] so the AR model has its lag.
    real_cost  : float – actual electricity cost incurred this step
    
    d    = data
    t    = state['current_time']
    p1   = decisions[0]
    p2   = decisions[1]
    v    = decisions[2]
    o1   = state['Occ1']
    o2   = state['Occ2']
    T1   = state['T1']
    T2   = state['T2']
    H    = state['H']
    Tout = d['outdoor_temperature'][t]

    # ── Temperature dynamics ──────────────────────────────────────────────────
    T1_next = (T1
               + d['heat_exchange_coeff']      * (T2   - T1)
               + d['thermal_loss_coeff']       * (Tout - T1)
               + d['heating_efficiency_coeff'] * p1
               - d['heat_vent_coeff']          * v
               + d['heat_occupancy_coeff']     * o1)

    T2_next = (T2
               + d['heat_exchange_coeff']      * (T1   - T2)
               + d['thermal_loss_coeff']       * (Tout - T2)
               + d['heating_efficiency_coeff'] * p2
               - d['heat_vent_coeff']          * v
               + d['heat_occupancy_coeff']     * o2)

    # ── Humidity dynamics ─────────────────────────────────────────────────────
    H_next = max(0.0,
                 H
                 - d['humidity_vent_coeff']      * v
                 + d['humidity_occupancy_coeff'] * (o1 + o2))

    # ── Ventilation inertia counter ───────────────────────────────────────────
    vc_prev = state.get('vent_counter', 0)
    vc_prev_status = 1 if vc_prev > 0 else 0   # treat counter > 0 as vent was ON

    if v == 1 and vc_prev_status == 0:
        vent_on_count_next = 1
    elif v == 1:
        vent_on_count_next = min(vc_prev + 1, 3)
    else:
        vent_on_count_next = 0

    # ── Real cost incurred ────────────────────────────────────────────────────
    real_cost = state['price_t'] * (p1 + p2 + v * d['ventilation_power'])

    # ── Overrule flags (passed through for continuity) ────────────────────────
    low_override_r1 = state.get('low_override_r1', 0)
    low_override_r2 = state.get('low_override_r2', 0)

    next_state = {
        'T1'            : T1_next,
        'T2'            : T2_next,
        'H'             : H_next,
        'Occ1'          : None,          # filled by environment
        'Occ2'          : None,          # filled by environment
        'price_t'       : None,          # filled by environment
        'price_previous': state['price_t'],   # lag for price model
        'vent_counter'  : vent_on_count_next,
        'low_override_r1': low_override_r1,
        'low_override_r2': low_override_r2,
        'current_time'  : t + 1,
    }

    return next_state, real_cost
"""

# =============================================================================
# 7. SP POLICY FUNCTION
# =============================================================================

def SP_policy(state):
    """
    2-stage stochastic programming policy.

    Reads the current state dict (Environment.py format), generates and
    clusters scenarios, solves the stochastic MILP, and returns the
    here-and-now decisions for this timestep.

    Parameters
    ----------
    state : dict  – keys as used by Environment.py:
        T1, T2, H, Occ1, Occ2, price_t, price_previous,
        vent_counter, low_override_r1, low_override_r2, current_time

    Returns
    -------
    dict with keys 'heat_r1', 'heat_r2', 'vent'
    """
    def debug_print_state(state):
        print("Debug incoming state:")
        for k, v in state.items():
            print(f"  {k}: {v}")
    
    #print(debug_print_state(state))

    # Make scenario generation reproducible per (day, timestep)
    #rng = np.random.default_rng(seed=int(state['current_time']) + 100)

    rng = np.random.default_rng(seed=42 + state['current_time'])

    t         = state['current_time']
    remaining = DATA['num_timeslots'] - t
    horizon   = min(HORIZON, remaining)

    if horizon <= 0:
        return {'heat_r1': 0.0, 'heat_r2': 0.0, 'vent': 0}

    # Handle missing previous price at t = 0
    p_prev = state.get('price_previous') or 4.0

    # Ventilation status: vent_counter > 0 means vent was ON last step
    vc       = state.get('vent_counter', 0)
    v_status = 1 if vc > 0 else 0

    # ── Generate raw Monte-Carlo scenarios ────────────────────────────────────
    price_dict, occ_dict, hum_occ_dict = generate_scenarios(
        price_now   = state['price_t'],
        price_prev  = p_prev,
        occ_r1_now  = state['Occ1'],
        occ_r2_now  = state['Occ2'],
        horizon     = horizon,
        n_scenarios = GEN_SCENARIOS,
        rng         = rng,
    )

    # ── Cluster to N_SCENARIOS representative centroids ───────────────────────
    n_clus = min(N_SCENARIOS, GEN_SCENARIOS)
    price_dict_clus, occ_dict_clus, hum_occ_dict_clus, probabilities, _, _, _ = \
        cluster_scenarios(
            price_dict, occ_dict,
            n_clusters            = n_clus,
            horizon               = horizon,
            scenarios_to_generate = GEN_SCENARIOS,
        )

    # ── Assemble current state for the MILP ──────────────────────────────────
    milp_state = {
        'T_in_r1'        : state['T1'],
        'T_in_r2'        : state['T2'],
        'humidity'       : state['H'],
        'vent_prev'      : v_status,
        'vent_on_count'  : vc,
        'low_override_r1': state.get('low_override_r1', 0),
        'low_override_r2': state.get('low_override_r2', 0),
    }

    # ── Build and solve ───────────────────────────────────────────────────────
    model  = build_sp_model(milp_state,
                            price_dict_clus, occ_dict_clus, hum_occ_dict_clus,
                            horizon, n_clus, probabilities)
    solver = SolverFactory('gurobi_direct')
    solver.options['TimeLimit'] = 12    # wall-clock cap (s)
    solver.options['MIPGap']    = 0.02  # 2 % optimality gap
    result = solver.solve(model, tee=False)

    # Guard against infeasible/failed solves
    from pyomo.opt import TerminationCondition
    if result.solver.termination_condition in (
            TerminationCondition.infeasible,
            TerminationCondition.unknown,
            TerminationCondition.error):
        # Fallback: return a safe do-nothing action
        return {'HeatPowerRoom1': 0.0, 'HeatPowerRoom2': 0.0, 'VentilationON': 0}

    # ── Extract here-and-now decisions (non-anticipativity → s=0 is canonical)
    s0 = 0
    p1 = float(value(model.Heat[1, 0, s0]))
    p2 = float(value(model.Heat[2, 0, s0]))
    v  = int(round(float(value(model.Vent[0, s0]))))

    # Safety clip to feasible bounds
    pr_max = DATA['heating_max_power']
    p1 = float(np.clip(p1, 0.0, pr_max))
    p2 = float(np.clip(p2, 0.0, pr_max))
    v  = int(np.clip(v,  0,   1))

    #return {'heat_r1': p1, 'heat_r2': p2, 'vent': v}
    return {'HeatPowerRoom1': p1, 'HeatPowerRoom2': p2, 'VentilationON': v}


# =============================================================================
# 8. GRADER-COMPATIBLE WRAPPER  (matches Environment.py calling convention)
# =============================================================================

def select_action(state):
    """
    Wrapper expected by Environment.py.

    Parameters
    ----------
    state : dict  – Environment.py state dict

    Returns
    -------
    dict  – {'heat_r1': float, 'heat_r2': float, 'vent': int}
    """
    return SP_policy(state)
