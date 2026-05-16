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
N_SCENARIOS = 10 # number of scenarios for Monte Carlo simulation

# =============================================================================
# 1. SYSTEM PARAMETERS
# =============================================================================
DATA = get_fixed_data()


# =============================================================================
# 3. MULTI-SCENARIO PATH GENERATION
# =============================================================================

def generate_expected_trajectories(price_now, price_prev, occ_r1_now, occ_r2_now,
                                   horizon, n_scenarios=N_SCENARIOS):
    """
    Sample multiple lookahead paths and average them step by step.

    Returns
    -------
    price_dict   : {t: float}        – price at lookahead step t
    occ_dict     : {(r, t): float}   – occupancy of room r at step t
    """
    price_sum = np.zeros(horizon)
    occ1_sum = np.zeros(horizon)
    occ2_sum = np.zeros(horizon)

    for _ in range(n_scenarios):
        p_cur = price_now
        p_prev = price_prev
        o1_cur = occ_r1_now
        o2_cur = occ_r2_now

        for t in range(horizon):
            p_next = price_model(p_cur, p_prev)
            o1_next, o2_next = next_occupancy_levels(o1_cur, o2_cur)

            price_sum[t] += p_next
            occ1_sum[t] += o1_next
            occ2_sum[t] += o2_next

            p_prev, p_cur = p_cur, p_next
            o1_cur, o2_cur = o1_next, o2_next

    expected_prices_trajectory = price_sum / n_scenarios
    expected_occ1_trajectory = occ1_sum / n_scenarios
    expected_occ2_trajectory = occ2_sum / n_scenarios

    price_dict = {t: float(expected_prices_trajectory[t]) for t in range(horizon)}
    occ_dict = {(1, t): float(expected_occ1_trajectory[t]) for t in range(horizon)}
    occ_dict.update({(2, t): float(expected_occ2_trajectory[t]) for t in range(horizon)})

    return price_dict, occ_dict


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

    #Controllare che ho un vent che dura sempre almeno 3 ore

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
    price_dict, occ_dict = generate_expected_trajectories(state["price_t"], state["price_previous"], state["Occ1"], state["Occ2"],
                                   horizon, n_scenarios=N_SCENARIOS)

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
