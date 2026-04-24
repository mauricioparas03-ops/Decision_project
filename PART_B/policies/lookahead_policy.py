"""
lookahead_policy.py
===================
Deterministic rolling-horizon (lookahead) policy for the restaurant HVAC system.

Compatible with Environment.py:
    from policies.lookahead_policy import select_action

Design choices
--------------
  Lookahead horizon : HORIZON = 4 steps  (minimum 3 due to vent inertia)
    Scenarios         : N_SCENARIOS sampled paths, averaged into expected trajectories
  Solver            : Gurobi, TimeLimit = 10 s, MIPGap = 2 %
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from pyomo.environ import (
    ConcreteModel, Set, Param, Var, Objective, Constraint,
    SolverFactory, NonNegativeReals, minimize, Binary, value,
)
from pyomo.opt import TerminationCondition
from Data.v2_SystemCharacteristics import get_fixed_data
from Data.OccupancyProcessRestaurant import next_occupancy_levels
from Data.PriceProcessRestaurant import price_model

# ── Hyper-parameters ───────────────────────────────────────────────────────────
HORIZON = 4   # lookahead steps (must be >= 3 due to vent-inertia constraint)
N_SCENARIOS = 1 # special case with just 1 scenario

_CLUSTERED_CACHE = None


def _load_clustered_cache():
    """
    Load clustered scenarios from CSV once and cache them.

    Returns
    -------
    tuple | None
        (price_matrix, occ1_matrix, occ2_matrix, probabilities) or None if files are missing.
    """
    global _CLUSTERED_CACHE
    if _CLUSTERED_CACHE is not None:
        return _CLUSTERED_CACHE

    data_dir = Path(__file__).resolve().parents[1] / "Data"
    ts_path = data_dir / "clustered_scenarios_timeseries.csv"
    prob_path = data_dir / "clustered_scenarios_probabilities.csv"

    if not ts_path.exists():
        _CLUSTERED_CACHE = None
        return None

    ts_df = pd.read_csv(ts_path)
    ts_df = ts_df.sort_values(["hour", "cluster_id"]).copy()

    price_matrix = (
        ts_df.pivot(index="hour", columns="cluster_id", values="price")
        .sort_index(axis=0)
        .sort_index(axis=1)
        .to_numpy(dtype=float)
    )
    occ1_matrix = (
        ts_df.pivot(index="hour", columns="cluster_id", values="occ1")
        .sort_index(axis=0)
        .sort_index(axis=1)
        .to_numpy(dtype=float)
    )
    occ2_matrix = (
        ts_df.pivot(index="hour", columns="cluster_id", values="occ2")
        .sort_index(axis=0)
        .sort_index(axis=1)
        .to_numpy(dtype=float)
    )

    if prob_path.exists():
        probabilities = (
            pd.read_csv(prob_path)
            .sort_values("cluster_id")["probability"]
            .to_numpy(dtype=float)
        )
    elif "probability" in ts_df.columns:
        probabilities = (
            ts_df[["cluster_id", "probability"]]
            .drop_duplicates(subset=["cluster_id"])
            .sort_values("cluster_id")["probability"]
            .to_numpy(dtype=float)
        )
    else:
        _CLUSTERED_CACHE = None
        return None

    if probabilities.sum() <= 0:
        _CLUSTERED_CACHE = None
        return None

    probabilities = probabilities / probabilities.sum()
    _CLUSTERED_CACHE = (price_matrix, occ1_matrix, occ2_matrix, probabilities)
    return _CLUSTERED_CACHE


def _expected_trajectories_from_clustered(state, horizon):
    """
    Build expected trajectories from pre-clustered scenarios.
    Falls back to Monte Carlo generation when clustered CSVs are unavailable.
    """
    clustered = _load_clustered_cache()
    # if clustered is None:
    #     return generate_expected_trajectories(
    #         price_now=state['price_t'],
    #         price_prev=state['price_previous'],
    #         occ_r1_now=state['Occ1'],
    #         occ_r2_now=state['Occ2'],
    #         horizon=horizon,
    #     )

    price_matrix, occ1_matrix, occ2_matrix, probabilities = clustered
    n_hours = price_matrix.shape[0]
    start_hour = int(state['current_time'])

    expected_prices = np.zeros(horizon)
    expected_occ1 = np.zeros(horizon)
    expected_occ2 = np.zeros(horizon)

    for t in range(horizon):
        idx = (start_hour + t) % n_hours
        expected_prices[t] = float(np.dot(price_matrix[idx], probabilities))
        expected_occ1[t] = float(np.dot(occ1_matrix[idx], probabilities))
        expected_occ2[t] = float(np.dot(occ2_matrix[idx], probabilities))

    price_dict = {t: float(expected_prices[t]) for t in range(horizon)}
    occ_dict = {(1, t): float(expected_occ1[t]) for t in range(horizon)}
    occ_dict.update({(2, t): float(expected_occ2[t]) for t in range(horizon)})
    return price_dict, occ_dict

# =============================================================================
# 1. SYSTEM PARAMETERS
# =============================================================================
DATA = get_fixed_data()


# =============================================================================
# 2. STOCHASTIC PROCESS MODELS
# =============================================================================

# def price_model(current_price, previous_price, rng):
#     """
#     One-step-ahead electricity price sample (AR(2)-like with mean reversion).
#     """
#     mean_price         = 4.0
#     reversion_strength = 0.12
#     price_cap          = 12.0
#     price_floor        = 0.0

#     mean_reversion = reversion_strength * (mean_price - current_price)
#     noise          = rng.normal(0, 0.5)

#     next_price = (current_price
#                   + 0.6 * (current_price - previous_price)
#                   + mean_reversion
#                   + noise)

#     if next_price < 0:
#         if rng.random() > 0.2:
#             next_price = rng.uniform(0, mean_price * 0.3)

#     return float(np.clip(next_price, price_floor, price_cap))


# def next_occupancy_levels(r1_current, r2_current, rng):
#     """
#     One-step-ahead occupancy sample for both rooms (coupled mean-reverting).
#     """
#     mean_r1, mean_r2 = 35.0, 25.0
#     rev      = 0.25
#     coupling = 0.1

#     noise_r1 = rng.normal(0, 3.0)
#     noise_r2 = rng.normal(0, 2.5)

#     r1_next = (r1_current
#                + rev      * (mean_r1 - r1_current)
#                + coupling * (r2_current - r1_current)
#                + noise_r1)

#     r2_next = (r2_current
#                + rev      * (mean_r2 - r2_current)
#                + coupling * (r1_current - r2_current)
#                + noise_r2)

#     return float(np.clip(r1_next, 20, 50)), float(np.clip(r2_next, 10, 30))


# =============================================================================
# 3. MULTI-SCENARIO PATH GENERATION
# =============================================================================

# def generate_expected_trajectories(price_now, price_prev, occ_r1_now, occ_r2_now,
#                                    horizon, n_scenarios=N_SCENARIOS):
#     """
#     Sample multiple lookahead paths and average them step by step.

#     Returns
#     -------
#     price_dict   : {t: float}        – price at lookahead step t
#     occ_dict     : {(r, t): float}   – occupancy of room r at step t
#     """
#     price_sum = np.zeros(horizon)
#     occ1_sum = np.zeros(horizon)
#     occ2_sum = np.zeros(horizon)

#     n_scenarios = max(1, int(n_scenarios))

#     for _ in range(n_scenarios):
#         p_cur = price_now
#         p_prev = price_prev
#         o1_cur = occ_r1_now
#         o2_cur = occ_r2_now

#         for t in range(horizon):
#             p_next = price_model(p_cur, p_prev)
#             o1_next, o2_next = next_occupancy_levels(o1_cur, o2_cur)

#             price_sum[t] += p_next
#             occ1_sum[t] += o1_next
#             occ2_sum[t] += o2_next

#             p_prev, p_cur = p_cur, p_next
#             o1_cur, o2_cur = o1_next, o2_next

#     expected_prices_trajectory = price_sum / n_scenarios
#     expected_occ1_trajectory = occ1_sum / n_scenarios
#     expected_occ2_trajectory = occ2_sum / n_scenarios

#     price_dict = {t: float(expected_prices_trajectory[t]) for t in range(horizon)}
#     occ_dict = {(1, t): float(expected_occ1_trajectory[t]) for t in range(horizon)}
#     occ_dict.update({(2, t): float(expected_occ2_trajectory[t]) for t in range(horizon)})

#     return price_dict, occ_dict


# =============================================================================
# 4. PYOMO MILP MODEL  (deterministic lookahead)
# =============================================================================

def build_lookahead_model(current_state, price_dict, occ_dict,
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
    _num_timeslots = int(d['num_timeslots'])

    m = ConcreteModel()

    # ── Sets ──────────────────────────────────────────────────────────────────
    m.R  = Set(initialize=[1, 2])
    m.T  = Set(initialize=list(range(horizon)))
    m.RT = m.R * m.T

    # Current real state as model parameters
    m.Tinit = Param(m.R, initialize={1: current_state['T1'], 2: current_state['T2']})
    m.Hinit = Param(initialize=current_state['H'])
    m.VentInit = Param(initialize=1 if current_state['vent_counter'] > 0 else 0)

    # PARAMETERS (Exogenous Data)
    m.O      = Param(m.RT, initialize=occ_dict)
    m.prices = Param(m.T,  initialize=price_dict)

    m.Tout   = Param(m.T, initialize={t: d['outdoor_temperature'][t % _num_timeslots] for t in m.T}) 

    # Physical constants and thresholds
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
    m.L = Param(initialize=horizon)
    m.M_temp = Param(initialize=100)
    m.M_hum = Param(initialize=100)
    m.U_vent = Param(initialize=3) # Minimum ventilation up-time

    # ── Decision variables ────────────────────────────────────────────────────
    m.Vent = Var(m.T, domain=Binary)    # Ventilation ON/OFF
    m.s    = Var(m.T, domain=Binary)    # Ventilation startup signal
    m.Heat = Var(m.RT, domain=NonNegativeReals, bounds=(0, m.Pr))

    # Sensor variables for overrule logic (Big-M)
    m.y_low  = Var(m.RT, domain=Binary) # 1 if T < Tmin
    m.y_ok   = Var(m.RT, domain=Binary) # 1 if T > Tok
    m.y_high = Var(m.RT, domain=Binary) # 1 if T > Thigh
    # State variable (Memory) for heating maintenance
    m.u = Var(m.RT, domain=Binary)
    # ── State variables ───────────────────────────────────────────────────────
    m.T_in = Var(m.RT, domain=NonNegativeReals)
    m.Hum  = Var(m.T,  domain=NonNegativeReals)

    # ── Temperature dynamics ──────────────────────────────────────────────────
    def temp_dynamics(m, r, t):
        if t == 0:
            return m.T_in[r, t] == m.Tinit[r]

        t_prev = t - 1
        occ = m.O[r, t_prev] if r == 1 else m.O[2, t_prev]
        r_other = 2 if r == 1 else 1

        return m.T_in[r, t] == (
            m.T_in[r, t_prev]
            + m.Zexch * (m.T_in[r_other, t_prev] - m.T_in[r, t_prev])
            + m.Zloss * (m.Tout[t_prev] - m.T_in[r, t_prev])
            + m.Zconv * m.Heat[r, t_prev]
            - m.Zcool * m.Vent[t_prev]
            + m.Zocc * occ
        )
    m.TempDyn = Constraint(m.RT, rule=temp_dynamics)

    # ── Humidity dynamics ─────────────────────────────────────────────────────
    def hum_dynamics(m, t):
        if t == 0:
            return m.Hum[t] == m.Hinit

        else:
            t_prev = t - 1
            return m.Hum[t] == (
                m.Hum[t_prev]
                - m.Hvent * m.Vent[t_prev]
                + m.Hocc * (m.O[1, t_prev] + m.O[2, t_prev])
        )
    m.HumDyn = Constraint(m.T, rule=hum_dynamics)

    # ── Overrule controller: LOW temperature ──────────────────────────────────
    # High Temperature Logic (Forced heating shutdown)
    m.c_thigh1 = Constraint(m.RT, rule=lambda m,r,t: m.T_in[r,t] >= m.Thigh - m.M_temp*(1 - m.y_high[r,t]))
    m.c_thigh2 = Constraint(m.RT, rule=lambda m,r,t: m.T_in[r,t] <= m.Thigh + m.M_temp*m.y_high[r,t])
    m.c_heat_off = Constraint(m.RT, rule=lambda m,r,t: m.Heat[r,t] <= m.Pr*(1 - m.y_high[r,t]))

    # Low Temperature Logic (Overrule Activation)
    m.c_tlow1 = Constraint(m.RT, rule=lambda m,r,t: m.T_in[r,t] <= m.Tmin + m.M_temp*(1 - m.y_low[r,t]))
    m.c_tlow2 = Constraint(m.RT, rule=lambda m,r,t: m.T_in[r,t] >= m.Tmin - m.M_temp*m.y_low[r,t])

    # Temperature OK Logic (Overrule Deactivation)
    m.c_tok1 = Constraint(m.RT, rule=lambda m,r,t: m.T_in[r,t] >= m.Tok - m.M_temp*(1 - m.y_ok[r,t]))
    m.c_tok2 = Constraint(m.RT, rule=lambda m,r,t: m.T_in[r,t] <= m.Tok + m.M_temp*m.y_ok[r,t])


    # Overrule Memory Management (u)
    m.c_u1 = Constraint(m.RT, rule=lambda m, r, t: m.u[r, t] >= m.y_low[r, t])

    def u_rule2(m, r, t):
        if t == 0:
            u_prev = current_state[f"low_override_r{r}"]
        else:
            u_prev = m.u[r,t-1]
        return m.u[r,t] <= u_prev + m.y_low[r,t]
    m.c_u2 = Constraint(m.RT, rule=u_rule2)

    m.c_heat_max = Constraint(m.RT, rule=lambda m,r,t: m.Heat[r,t] >= m.Pr * m.u[r,t])

    def u_rule3(m, r, t):
        if t == 0:
            u_prev = current_state[f"low_override_r{r}"]
        else:
            u_prev = m.u[r, t-1]
        return m.u[r, t] >= u_prev - m.y_ok[r,t]
    m.c_u3 = Constraint(m.RT, rule=u_rule3)
    m.c_u4 = Constraint(m.RT, rule=lambda m,r,t: m.u[r,t] <= 1 - m.y_ok[r,t])
    # -----------------------------------------------------------------------------
    # 7) Ventilation Constraints (Startup, Min-Up Time, Humidity)
    # -----------------------------------------------------------------------------
    def s_rule1(m, t):
        v_prev = m.VentInit if t == 0 else m.Vent[t-1]
        return m.s[t] >= m.Vent[t] - v_prev
    m.c_s1 = Constraint(m.T, rule=s_rule1)
    m.c_s2 = Constraint(m.T, rule=lambda m, t: m.s[t] <= m.Vent[t])

    def s_rule3(m, t):
        v_prev = m.VentInit if t == 0 else m.Vent[t-1]
        return m.s[t] <= 1 - v_prev
    m.c_s3 = Constraint(m.T, rule=s_rule3)

    def min_up_time_ventilation_rule(m, t):
        end_idx = min(t + m.U_vent - 1, m.L - 1)
        sum_vent = sum(m.Vent[tau] for tau in range(t, end_idx + 1))
        min_val = min(m.U_vent, m.L - t)
        return sum_vent >= min_val * m.s[t]
    m.min_vent_on = Constraint(m.T, rule=min_up_time_ventilation_rule)

    m.c_hum = Constraint(m.T, rule=lambda m, t: m.Hum[t] <= m.Hhigh + m.M_hum * m.Vent[t])


    # ── Objective: minimise electricity cost over horizon ─────────────────────
    def objective(model):
        heat_cost = sum(model.prices[t] * model.Heat[r, t] for r in model.R for t in model.T)
        vent_cost = sum(model.prices[t] * model.Vent[t] * model.Pvent for t in model.T)
        return heat_cost + vent_cost

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
    # def debug_print_state(state):
    #     print("Debug incoming state:")
    #     for k, v in state.items():
    #         print(f"  {k}: {v}")
    
    #print(debug_print_state(state))

    t         = state['current_time']
    remaining = DATA['num_timeslots'] - t
    horizon   = min(HORIZON, remaining)

    # if horizon <= 0:
    #     return {'HeatPowerRoom1': 0.0, 'HeatPowerRoom2': 0.0, 'VentilationON': 0}


    # Ventilation status derived from counter
    # vc       = state['vent_counter']
    # v_status = 1 if vc > 0 else 0

    # ── Build expected trajectories from clustered scenarios (or fallback) ───
    price_dict, occ_dict = _expected_trajectories_from_clustered(state, horizon)

    # ── Assemble current state for the MILP ──────────────────────────────────
    # milp_state = {
    #     'T_in_r1'        : state['T1'],
    #     'T_in_r2'        : state['T2'],
    #     'humidity'        : state['H'],
    #     'vent_prev'       : v_status,
    #     'vent_on_count'   : vc,
    #     'low_override_r1' : state.get('low_override_r1', 0),
    #     'low_override_r2' : state.get('low_override_r2', 0),
    # }

    # ── Build and solve ───────────────────────────────────────────────────────
    model  = build_lookahead_model(state, price_dict, occ_dict, horizon)
    solver = SolverFactory('gurobi')
    result = solver.solve(model, tee=False)

    if result.solver.termination_condition not in (
        TerminationCondition.optimal,
        TerminationCondition.feasible,
    ):
        warnings.warn(
            f"Gurobi failed at time {state['current_time']} with termination condition "
            f"{result.solver.termination_condition}. Falling back to zero action.",
            RuntimeWarning,
        )
        return {'HeatPowerRoom1': 0.0, 'HeatPowerRoom2': 0.0, 'VentilationON': 0}

    # ── Guard against infeasible / failed solves ──────────────────────────────
    # if result.solver.termination_condition in (
    #         TerminationCondition.infeasible,
    #         TerminationCondition.unknown,
    #         TerminationCondition.error):
    #     return {'HeatPowerRoom1': 0.0, 'HeatPowerRoom2': 0.0, 'VentilationON': 0}

    # ── Extract here-and-now decisions ────────────────────────────────────────
    p1 = float(value(model.Heat[1, 0]))
    p2 = float(value(model.Heat[2, 0]))
    v  = int(round(float(value(model.Vent[0]))))

    # # Safety clip to feasible bounds
    # pr_max = DATA['heating_max_power']
    # p1 = float(np.clip(p1, 0.0, pr_max))
    # p2 = float(np.clip(p2, 0.0, pr_max))
    # v  = int(np.clip(v,  0,   1))

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
