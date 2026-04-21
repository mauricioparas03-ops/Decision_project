"""
lookahead_policy.py
===================
Deterministic rolling-horizon (lookahead) policy for the restaurant HVAC system.

Compatible with Environment.py:
    from policies.lookahead_policy import select_action

Design choices
--------------
  Lookahead horizon : HORIZON = 4 steps  (minimum 3 due to vent inertia)
  Scenarios         : 1 single sampled path (deterministic lookahead)
  Solver            : Gurobi, TimeLimit = 10 s, MIPGap = 2 %
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np

from pyomo.environ import (
    ConcreteModel, Set, Param, Var, Objective, Constraint,
    SolverFactory, NonNegativeReals, minimize, Binary, value,
)
from pyomo.opt import TerminationCondition

# ── Hyper-parameters ───────────────────────────────────────────────────────────
HORIZON = 4   # lookahead steps (must be >= 3 due to vent-inertia constraint)

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

def price_model(current_price, previous_price, rng):
    """
    One-step-ahead electricity price sample (AR(2)-like with mean reversion).
    """
    mean_price         = 4.0
    reversion_strength = 0.12
    price_cap          = 12.0
    price_floor        = 0.0

    mean_reversion = reversion_strength * (mean_price - current_price)
    noise          = rng.normal(0, 0.5)

    next_price = (current_price
                  + 0.6 * (current_price - previous_price)
                  + mean_reversion
                  + noise)

    if next_price < 0:
        if rng.random() > 0.2:
            next_price = rng.uniform(0, mean_price * 0.3)

    return float(np.clip(next_price, price_floor, price_cap))


def next_occupancy_levels(r1_current, r2_current, rng):
    """
    One-step-ahead occupancy sample for both rooms (coupled mean-reverting).
    """
    mean_r1, mean_r2 = 35.0, 25.0
    rev      = 0.25
    coupling = 0.1

    noise_r1 = rng.normal(0, 3.0)
    noise_r2 = rng.normal(0, 2.5)

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
# 3. SINGLE-SCENARIO PATH GENERATION
# =============================================================================

def generate_single_scenario(price_now, price_prev, occ_r1_now, occ_r2_now,
                              horizon, rng):
    """
    Sample one deterministic-lookahead path over *horizon* steps.

    Returns
    -------
    price_dict   : {t: float}        – price at lookahead step t
    occ_dict     : {(r, t): float}   – occupancy of room r at step t
    hum_occ_dict : {t: float}        – total occupancy at step t
    """
    price_dict   = {}
    occ_dict     = {}
    hum_occ_dict = {}

    p_cur,  p_prev  = price_now,  price_prev
    o1_cur, o2_cur  = occ_r1_now, occ_r2_now

    for t in range(horizon):
        p_next           = price_model(p_cur, p_prev, rng)
        o1_next, o2_next = next_occupancy_levels(o1_cur, o2_cur, rng)

        price_dict[t]      = p_next
        occ_dict[1, t]     = o1_next
        occ_dict[2, t]     = o2_next
        hum_occ_dict[t]    = o1_next + o2_next

        p_prev, p_cur   = p_cur,  p_next
        o1_cur, o2_cur  = o1_next, o2_next

    return price_dict, occ_dict, hum_occ_dict


# =============================================================================
# 4. PYOMO MILP MODEL  (deterministic lookahead)
# =============================================================================

def build_lookahead_model(current_state, price_dict, occ_dict, hum_occ_dict,
                          horizon):
    """
    Build the deterministic MILP for the lookahead policy.

    Parameters
    ----------
    current_state : dict – keys: T_in_r1, T_in_r2, humidity,
                                 vent_prev, vent_on_count,
                                 low_override_r1, low_override_r2
    price_dict    : {t: float}
    occ_dict      : {(r, t): float}
    hum_occ_dict  : {t: float}
    horizon       : int

    Returns
    -------
    Pyomo ConcreteModel (unsolved)
    """
    d      = DATA
    M      = 200.0

    T_init = {1: current_state['T_in_r1'], 2: current_state['T_in_r2']}
    H_init = current_state['humidity']
    v_prev = int(current_state['vent_prev'])
    v_on_h = int(current_state.get('vent_on_count', 0))

    m = ConcreteModel()

    # ── Sets ──────────────────────────────────────────────────────────────────
    m.R  = Set(initialize=[1, 2])
    m.T  = Set(initialize=list(range(horizon)))
    m.RT = m.R * m.T

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

    m.O      = Param(m.RT, initialize=occ_dict)
    m.prices = Param(m.T,  initialize=price_dict)
    m.HumOcc = Param(m.T,  initialize=hum_occ_dict)

    # ── Decision variables ────────────────────────────────────────────────────
    m.Heat = Var(m.RT, domain=NonNegativeReals, bounds=(0, d['heating_max_power']))
    m.Vent = Var(m.T,  domain=Binary)
    m.Uon  = Var(m.T,  domain=Binary)
    m.Uoff = Var(m.T,  domain=Binary)

    # ── Auxiliary binaries for overrule controllers ───────────────────────────
    m.u = Var(m.RT, domain=Binary)   # 1 → low-temp overrule active
    m.w = Var(m.RT, domain=Binary)   # 1 → temperature recovered to T_OK
    m.y = Var(m.RT, domain=Binary)   # 1 → high-temp overrule active
    m.y_low = Var(m.RT, domain=Binary) # 1 -> T is below T_low


    # ── State variables ───────────────────────────────────────────────────────
    m.T_in = Var(m.RT, domain=NonNegativeReals)
    m.Hum  = Var(m.T,  domain=NonNegativeReals)

    # ── Temperature dynamics ──────────────────────────────────────────────────
    def temp_dynamics(m, r, t):
        if t == 0:
            return m.T_in[r, t] == T_init[r]
        tp      = t - 1
        other_r = 2 if r == 1 else 1
        return m.T_in[r, t] == (
            m.T_in[r, tp]
            + m.Zexch * (m.T_in[other_r, tp] - m.T_in[r, tp])
            + m.Zloss * (m.Tout[tp]           - m.T_in[r, tp])
            + m.Zconv * m.Heat[r, tp]
            - m.Zcool * m.Vent[tp]
            + m.Zocc  * m.O[r, tp]
        )
    m.TempDyn = Constraint(m.RT, rule=temp_dynamics)

    # ── Humidity dynamics ─────────────────────────────────────────────────────
    def hum_dynamics(m, t):
        if t == 0:
            return m.Hum[t] == H_init
        tp = t - 1
        return m.Hum[t] == (
            m.Hum[tp]
            - m.Hvent * m.Vent[tp]
            + m.Hocc  * m.HumOcc[tp]
        )
    m.HumDyn = Constraint(m.T, rule=hum_dynamics)

    # ── Overrule controller: LOW temperature ──────────────────────────────────
    #EQN 8&9
    # 1. Strict Detection of Low Temperature (y_low) 
    def y_low_lower(m, r, t):
        # Forces y_low=1 when T < Tmin
        return m.T_in[r, t] >= m.Tmin - M * m.y_low[r, t]
    m.YLowLower = Constraint(m.RT, rule=y_low_lower)

    def y_low_upper(m, r, t):
        # Forces y_low=0 when T >= Tmin
        return m.T_in[r, t] <= m.Tmin + M * (1 - m.y_low[r, t])
    m.YLowUpper = Constraint(m.RT, rule=y_low_upper)

    #EQN 10&11
    # 2. Strict Detection of Recovery Temperature (w)
    def w_deactivation_lower(m, r, t):
        # Forces w=1 when T >= Tok
        return m.T_in[r, t] >= m.Tok - M * (1 - m.w[r, t])
    m.WDeactivationLower = Constraint(m.RT, rule=w_deactivation_lower)

    def w_deactivation_upper(m, r, t):
        # Forces w=0 when T < Tok
        return m.T_in[r, t] <= m.Tok + M * m.w[r, t]
    m.WDeactivationUpper = Constraint(m.RT, rule=w_deactivation_upper)

    def u_persistence(m, r, t):
        if t == 0:
            return Constraint.Skip  # Let OverruleInit handle this
        return m.u[r, t] >= m.u[r, t - 1] - m.w[r, t]
    m.UPersistence = Constraint(m.RT, rule=u_persistence)

    def heat_max_when_overrule(m, r, t):
        return m.Heat[r, t] >= m.Pr * m.u[r, t]
    m.HeatMaxOverrule = Constraint(m.RT, rule=heat_max_when_overrule)

    # ── Overrule controller: HIGH temperature ─────────────────────────────────
    def y_activation_lower(m, r, t):
        # Forces y=1 when T > Thigh
        return m.T_in[r, t] <= m.Thigh + M * m.y[r, t]
    m.YActivationLower = Constraint(m.RT, rule=y_activation_lower)

    def y_activation_upper(m, r, t):
        # Forces y=0 when T <= Thigh
        return m.T_in[r, t] >= m.Thigh - M * (1 - m.y[r, t])
    m.YActivationUpper = Constraint(m.RT, rule=y_activation_upper)

    def heat_off_when_overrule(m, r, t):
        return m.Heat[r, t] <= m.Pr * (1 - m.y[r, t])
    m.HeatOffOverrule = Constraint(m.RT, rule=heat_off_when_overrule)

    # ── Humidity overrule: force ventilation ON when humid ────────────────────
    def vent_humidity_overrule(m, t):
        return m.Hum[t] <= m.Hhigh + M * m.Vent[t]
    m.VentHumOverrule = Constraint(m.T, rule=vent_humidity_overrule)

    # __ State update for overrule controller _________________________________
    #EQN 12,13,15,16
    def temp_lower_than_tmin_1(m, r, t):
        # Must turn ON if it gets too cold
        return m.u[r, t] >= m.y_low[r, t]
    m.UStateLogic1 = Constraint(m.RT, rule=temp_lower_than_tmin_1)

    def temp_higher_than_tok(m, r, t):
        # Cannot turn ON unless it is too cold
        if t == 0: return Constraint.Skip
        return m.u[r, t] <= m.u[r, t - 1] + m.y_low[r, t]
    m.UStateLogic3 = Constraint(m.RT, rule=temp_higher_than_tok)

    def temp_higher_than_tok_2(m, r, t):
        # Must turn OFF if it reaches Tok
        return m.u[r, t] <= 1 - m.w[r, t]
    m.UStateLogic4 = Constraint(m.RT, rule=temp_higher_than_tok_2)

    def overrule_init(m, r, t):
        # At t=0, fix u to match the real system's overrule state
        if t == 0:
            init_u = 1 if current_state.get(f'low_override_r{r}', 0) else 0
            return m.u[r, t] == init_u
        return Constraint.Skip
    m.OverruleInit = Constraint(m.RT, rule=overrule_init)

    # ── Ventilation inertia: 3-hour minimum ON time ───────────────────────────
    def on_off_exclusivity(m, t):
        return m.Uon[t] + m.Uoff[t] <= 1
    m.OnOffExcl = Constraint(m.T, rule=on_off_exclusivity)

    def uoff_bound(m, t):
        return m.Uoff[t] <= 1 - m.Vent[t]
    m.UoffBound = Constraint(m.T, rule=uoff_bound)

    def uon_bound(m, t):
        return m.Uon[t] <= 1 - m.Vent[t]
    m.UonBound = Constraint(m.T, rule=uon_bound)

    def min_uptime(m, t):
        L         = 3
        remaining = [k for k in range(t, t + L) if k <= m.T.last()]
        v_p       = v_prev if t == 0 else m.Vent[t - 1]
        return (sum(m.Vent[k] for k in remaining)
                >= len(remaining) * (m.Vent[t] - v_p))
    m.MinUptime = Constraint(m.T, rule=min_uptime)

    # Carry over inertia from previous real-system hours
    if v_prev == 1 and v_on_h < 3:
        forced_on = 3 - v_on_h
        for t in range(min(forced_on, horizon)):
            m.Vent[t].fix(1)

    # ── Objective: minimise electricity cost over horizon ─────────────────────
    def objective(m):
        return sum(
            m.prices[t] * (sum(m.Heat[r, t] for r in m.R) + m.Vent[t] * m.Pvent)
            for t in m.T
        )
    m.obj = Objective(rule=objective, sense=minimize)

    return m


# =============================================================================
# 5. LOOKAHEAD POLICY FUNCTION
# =============================================================================

def lookahead_policy(state):
    """
    Deterministic rolling-horizon lookahead policy.

    Reads the current state dict (Environment.py format), generates one
    sampled price/occupancy path, solves the deterministic MILP, and
    returns the here-and-now decisions.

    Parameters
    ----------
    state : dict  – keys as used by Environment.py:
        T1, T2, H, Occ1, Occ2, price_t, price_previous,
        vent_counter, low_override_r1, low_override_r2, current_time

    Returns
    -------
    dict with keys 'HeatPowerRoom1', 'HeatPowerRoom2', 'VentilationON'
    """
    def debug_print_state(state):
        print("Debug incoming state:")
        for k, v in state.items():
            print(f"  {k}: {v}")
    
    #print(debug_print_state(state))

    t         = state['current_time']
    remaining = DATA['num_timeslots'] - t
    horizon   = min(HORIZON, remaining)

    if horizon <= 0:
        return {'HeatPowerRoom1': 0.0, 'HeatPowerRoom2': 0.0, 'VentilationON': 0}

    # Reproducible per-timestep RNG
    rng    = np.random.default_rng(seed=42 + t)
    p_prev = state.get('price_previous') or 4.0

    # Ventilation status derived from counter
    vc       = state.get('vent_counter', 0)
    v_status = 1 if vc > 0 else 0

    # ── Generate one sample path ──────────────────────────────────────────────
    price_dict, occ_dict, hum_occ_dict = generate_single_scenario(
        price_now   = state['price_t'],
        price_prev  = p_prev,
        occ_r1_now  = state['Occ1'],
        occ_r2_now  = state['Occ2'],
        horizon     = horizon,
        rng         = rng,
    )

    # ── Assemble current state for the MILP ──────────────────────────────────
    milp_state = {
        'T_in_r1'        : state['T1'],
        'T_in_r2'        : state['T2'],
        'humidity'        : state['H'],
        'vent_prev'       : v_status,
        'vent_on_count'   : vc,
        'low_override_r1' : state.get('low_override_r1', 0),
        'low_override_r2' : state.get('low_override_r2', 0),
    }

    # ── Build and solve ───────────────────────────────────────────────────────
    model  = build_lookahead_model(milp_state, price_dict, occ_dict,
                                   hum_occ_dict, horizon)
    solver = SolverFactory('gurobi_direct')
    solver.options['TimeLimit'] = 10
    solver.options['MIPGap']    = 0.02
    solver.options['Seed']      = 42
    solver.options['Threads']   = 1
    result = solver.solve(model, tee=False)

    # ── Guard against infeasible / failed solves ──────────────────────────────
    if result.solver.termination_condition in (
            TerminationCondition.infeasible,
            TerminationCondition.unknown,
            TerminationCondition.error):
        return {'HeatPowerRoom1': 0.0, 'HeatPowerRoom2': 0.0, 'VentilationON': 0}

    # ── Extract here-and-now decisions ────────────────────────────────────────
    p1 = float(value(model.Heat[1, 0]))
    p2 = float(value(model.Heat[2, 0]))
    v  = int(round(float(value(model.Vent[0]))))

    # Safety clip to feasible bounds
    pr_max = DATA['heating_max_power']
    p1 = float(np.clip(p1, 0.0, pr_max))
    p2 = float(np.clip(p2, 0.0, pr_max))
    v  = int(np.clip(v,  0,   1))

    return {'HeatPowerRoom1': p1, 'HeatPowerRoom2': p2, 'VentilationON': v}


# =============================================================================
# 6. GRADER-COMPATIBLE WRAPPER
# =============================================================================

def select_action(state):
    """
    Wrapper expected by Environment.py.

    Parameters
    ----------
    state : dict  – Environment.py state dict

    Returns
    -------
    dict  – {'HeatPowerRoom1': float, 'HeatPowerRoom2': float, 'VentilationON': int}
    """
    return lookahead_policy(state)
