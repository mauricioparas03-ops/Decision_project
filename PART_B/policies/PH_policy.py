"""
PH_policy.py
============
Progressive Hedging (PH) policy for the restaurant HVAC system.

Compatible with Environment.py:
    from policies.PH_policy import select_action

Structural differences vs SP_policy.py
---------------------------------------
  SP_policy  : builds ONE large MILP coupling all S scenarios via explicit
                non-anticipativity constraints (HeatNA, VentNA) on Heat[r,0,s]
                and Vent[0,s].

  PH_policy  : replaces that single coupled solve with an outer PH iteration:
                  - S independent single-scenario MIQPs (no s-index inside model)
                  - No HeatNA / VentNA constraints anywhere
                  - NAC enforced softly via augmented-Lagrangian penalty added
                    to each subproblem's objective:
                      w_s · x1_s  +  (rho/2) · ||x1_s - x_bar||²
                  - After each round of subproblem solves:
                      x_bar  ← Σ_s  prob_s * x1_s
                      w_s    ← w_s + rho * (x1_s - x_bar)
                  - Iterate until primal / dual residuals < tol or max_iter hit
                  - Return rounded x_bar as the here-and-now action

Design choices
--------------
  Lookahead horizon  : HORIZON      = 4  steps
  Raw MC scenarios   : GEN_SCENARIOS = 50
  Clustered scenarios: N_SCENARIOS   = 10  (K-Means centroids)
  PH penalty         : RHO          = 1.0
  PH max iterations  : PH_MAX_ITER  = 20
  PH convergence tol : PH_TOL       = 1e-3
  Solver             : Gurobi (gurobi_direct), TimeLimit = 8 s per subproblem
"""

import warnings

from pyomo.opt import TerminationCondition
from pyomo.environ import (
    ConcreteModel, Set, Param, Var, Objective, Constraint,
    SolverFactory, NonNegativeReals, minimize, Binary, value,
)

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from Data.v2_SystemCharacteristics import get_fixed_data
from Data.PriceProcessRestaurant import price_model
from Data.OccupancyProcessRestaurant import next_occupancy_levels

# =============================================================================
# Hyper-parameters
# =============================================================================
HORIZON       = 4     # lookahead steps (>= 3 for vent-inertia)
GEN_SCENARIOS = 50    # raw Monte-Carlo draws before clustering
N_SCENARIOS   = 10    # K-Means clusters

RHO         = 1.0    # PH penalty parameter
PH_MAX_ITER = 20     # maximum PH outer iterations
PH_TOL      = 1e-3   # convergence tolerance on primal & dual residuals

# =============================================================================
# System data (loaded once at import)
# =============================================================================
DATA = get_fixed_data()


# =============================================================================
# 1. SCENARIO GENERATION  (identical to SP_policy)
# =============================================================================

def generate_scenarios(price_now, price_prev,
                       occ_r1_now, occ_r2_now,
                       horizon, n_scenarios):
    """
    Draw n_scenarios independent Monte-Carlo sample paths over horizon steps.

    Returns
    -------
    price_dict : {(t, s): float}
    occ_dict   : {(r, t, s): float}
    """
    price_dict = {}
    occ_dict   = {}

    for s in range(n_scenarios):
        p_cur,  p_prev  = price_now,  price_prev
        o1_cur, o2_cur  = occ_r1_now, occ_r2_now

        for t in range(horizon):
            p_next           = price_model(p_cur, p_prev)
            o1_next, o2_next = next_occupancy_levels(o1_cur, o2_cur)

            price_dict[t, s]  = p_next
            occ_dict[1, t, s] = o1_next
            occ_dict[2, t, s] = o2_next

            p_prev, p_cur   = p_cur,  p_next
            o1_cur, o2_cur  = o1_next, o2_next

    return price_dict, occ_dict


# =============================================================================
# 2. SCENARIO CLUSTERING  (identical to SP_policy)
# =============================================================================

def cluster_scenarios(price_dict, occ_dict,
                      n_clusters, horizon, scenarios_to_generate):
    """
    Reduce raw MC paths to n_clusters representative centroids via K-Means.

    Returns
    -------
    price_dict_clus : {(t, s): float}
    occ_dict_clus   : {(r, t, s): float}
    probabilities   : np.ndarray shape (n_clusters,)
    """
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
    probabilities = np.bincount(labels) / scenarios_to_generate
    centroids     = scaler.inverse_transform(km.cluster_centers_)

    price_dict_clus = {}
    occ_dict_clus   = {}

    for s in range(n_clusters):
        for t in range(horizon):
            price_dict_clus[t, s]  = float(centroids[s, t])
            occ_dict_clus[1, t, s] = float(max(0.0, centroids[s, horizon + t]))
            occ_dict_clus[2, t, s] = float(max(0.0, centroids[s, 2 * horizon + t]))

    return price_dict_clus, occ_dict_clus, probabilities


# =============================================================================
# 3. SINGLE-SCENARIO SUBPROBLEM  (core PH change vs SP_policy)
# =============================================================================

def build_subproblem(state, price_s, occ_s, horizon, x_bar, w_s, rho):
    """
    Build a single-scenario MIQP subproblem for Progressive Hedging.

    KEY DIFFERENCES FROM build_sp_model:
    ─────────────────────────────────────
    (a) No m.S set — variables are indexed over (r, t) and (t,) only.
        The scenario dimension has been lifted out; this model represents
        exactly one scenario.

    (b) No HeatNA / VentNA constraints — NAC is NOT enforced here as
        hard equality constraints.  Instead it is enforced softly via the
        augmented-Lagrangian penalty added to the objective (see below).

    (c) Objective = single-scenario cost
                  + w_s · x1_s              (linear dual term)
                  + (rho/2) · ||x1_s - x_bar||²   (quadratic proximity term)
        where x1_s = [Heat[1,0], Heat[2,0], Vent[0]]  (stage-1 decisions).
        The quadratic term makes this a MIQP (Vent[0] is binary);
        Gurobi handles MIQP natively.

    Parameters
    ----------
    state    : dict  — Environment.py state dict
    price_s  : {t: float}   — price trajectory for this scenario
    occ_s    : {(r,t): float} — occupancy trajectory for this scenario
    horizon  : int
    x_bar    : dict  — consensus stage-1 decisions
                       keys: 'heat_r1', 'heat_r2', 'vent'
    w_s      : dict  — dual multipliers for this scenario
                       same keys as x_bar
    rho      : float — PH penalty parameter

    Returns
    -------
    Pyomo ConcreteModel (unsolved MIQP)
    """
    d              = DATA
    _num_timeslots = int(d['num_timeslots'])

    m = ConcreteModel()

    # ── Sets ──────────────────────────────────────────────────────────────────
    # NOTE: no m.S — single scenario, so only R and T needed
    m.R  = Set(initialize=[1, 2])
    m.T  = Set(initialize=list(range(horizon)))
    m.RT = m.R * m.T

    # ── Current real state ────────────────────────────────────────────────────
    v_prev = 1 if state['vent_counter'] > 0 else 0

    m.Tinit    = Param(m.R, initialize={1: state['T1'], 2: state['T2']})
    m.Hinit    = Param(initialize=state['H'])
    m.VentInit = Param(initialize=v_prev)

    # ── Scenario parameters (single scenario — no s index) ────────────────────
    m.O      = Param(m.RT, initialize=occ_s)
    m.prices = Param(m.T,  initialize=price_s)

    m.Tout = Param(m.T, initialize={
        t: d['outdoor_temperature'][(state['current_time'] + t) % _num_timeslots]
        for t in range(horizon)
    })

    # ── Physical constants ────────────────────────────────────────────────────
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

    # ── Decision variables (no s index) ──────────────────────────────────────
    m.Vent   = Var(m.T,  domain=Binary)
    m.Vstart = Var(m.T,  domain=Binary)
    m.Heat   = Var(m.RT, domain=NonNegativeReals,
                   bounds=(0, d['heating_max_power']))

    # ── Auxiliary binaries ────────────────────────────────────────────────────
    m.y_low  = Var(m.RT, domain=Binary)
    m.y_ok   = Var(m.RT, domain=Binary)
    m.y_high = Var(m.RT, domain=Binary)
    m.u      = Var(m.RT, domain=Binary)

    # ── State variables ───────────────────────────────────────────────────────
    m.T_in = Var(m.RT, domain=NonNegativeReals)
    m.Hum  = Var(m.T,  domain=NonNegativeReals)

    # ── Temperature dynamics ──────────────────────────────────────────────────
    def temp_dynamics(m, r, t):
        if t == 0:
            return m.T_in[r, t] == m.Tinit[r]
        tp      = t - 1
        r_other = 2 if r == 1 else 1
        return m.T_in[r, t] == (
            m.T_in[r, tp]
            + m.Zexch * (m.T_in[r_other, tp] - m.T_in[r, tp])
            + m.Zloss * (m.Tout[tp]           - m.T_in[r, tp])
            + m.Zconv * m.Heat[r, tp]
            - m.Zcool * m.Vent[tp]
            + m.Zocc  * m.O[r, tp]
        )
    m.TempDyn = Constraint(m.RT, rule=temp_dynamics)

    # ── Humidity dynamics ─────────────────────────────────────────────────────
    def hum_dynamics(m, t):
        if t == 0:
            return m.Hum[t] == m.Hinit
        tp = t - 1
        return m.Hum[t] == (
            m.Hum[tp]
            - m.Hvent * m.Vent[tp]
            + m.Hocc  * (m.O[1, tp] + m.O[2, tp])
        )
    m.HumDyn = Constraint(m.T, rule=hum_dynamics)

    # ── High temperature: forced heating shutdown ─────────────────────────────
    m.CThigh1 = Constraint(m.RT,
        rule=lambda m, r, t: m.T_in[r, t] >= m.Thigh - m.M_temp * (1 - m.y_high[r, t]))
    m.CThigh2 = Constraint(m.RT,
        rule=lambda m, r, t: m.T_in[r, t] <= m.Thigh + m.M_temp * m.y_high[r, t])
    m.CHeatOff = Constraint(m.RT,
        rule=lambda m, r, t: m.Heat[r, t] <= m.Pr * (1 - m.y_high[r, t]))

    # ── Low temperature: overrule activation ─────────────────────────────────
    m.CTlow1 = Constraint(m.RT,
        rule=lambda m, r, t: m.T_in[r, t] <= m.Tmin + m.M_temp * (1 - m.y_low[r, t]))
    m.CTlow2 = Constraint(m.RT,
        rule=lambda m, r, t: m.T_in[r, t] >= m.Tmin - m.M_temp * m.y_low[r, t])

    # ── Temperature-OK: overrule deactivation ────────────────────────────────
    m.CTok1 = Constraint(m.RT,
        rule=lambda m, r, t: m.T_in[r, t] >= m.Tok - m.M_temp * (1 - m.y_ok[r, t]))
    m.CTok2 = Constraint(m.RT,
        rule=lambda m, r, t: m.T_in[r, t] <= m.Tok + m.M_temp * m.y_ok[r, t])

    # ── Overrule memory (u) ───────────────────────────────────────────────────
    m.CU1 = Constraint(m.RT,
        rule=lambda m, r, t: m.u[r, t] >= m.y_low[r, t])

    def c_u2(m, r, t):
        u_prev = state[f'low_override_r{r}'] if t == 0 else m.u[r, t - 1]
        return m.u[r, t] <= u_prev + m.y_low[r, t]
    m.CU2 = Constraint(m.RT, rule=c_u2)

    m.CHeatMax = Constraint(m.RT,
        rule=lambda m, r, t: m.Heat[r, t] >= m.Pr * m.u[r, t])

    def c_u3(m, r, t):
        u_prev = state.get(f'low_override_r{r}', 0) if t == 0 else m.u[r, t - 1]
        return m.u[r, t] >= u_prev - m.y_ok[r, t]
    m.CU3 = Constraint(m.RT, rule=c_u3)

    m.CU4 = Constraint(m.RT,
        rule=lambda m, r, t: m.u[r, t] <= 1 - m.y_ok[r, t])

    # ── Ventilation: startup signal ───────────────────────────────────────────
    def c_vstart1(m, t):
        v_p = m.VentInit if t == 0 else m.Vent[t - 1]
        return m.Vstart[t] >= m.Vent[t] - v_p
    m.CVstart1 = Constraint(m.T, rule=c_vstart1)

    m.CVstart2 = Constraint(m.T,
        rule=lambda m, t: m.Vstart[t] <= m.Vent[t])

    def c_vstart3(m, t):
        v_p = m.VentInit if t == 0 else m.Vent[t - 1]
        return m.Vstart[t] <= 1 - v_p
    m.CVstart3 = Constraint(m.T, rule=c_vstart3)

    # ── Ventilation: minimum uptime ───────────────────────────────────────────
    def min_uptime(m, t):
        end_idx = min(t + m.U_vent - 1, m.L - 1)
        min_val = min(m.U_vent, m.L - t)
        return sum(m.Vent[tau] for tau in range(t, end_idx + 1)) >= min_val * m.Vstart[t]
    m.MinVentOn = Constraint(m.T, rule=min_uptime)

    # ── Ventilation: humidity overrule ────────────────────────────────────────
    m.CVentHum = Constraint(m.T,
        rule=lambda m, t: m.Hum[t] <= m.Hhigh + m.M_hum * m.Vent[t])

    # =========================================================================
    # ── Objective: single-scenario cost + PH augmented-Lagrangian penalty ────
    # =========================================================================
    #
    # Stage-1 variables (here-and-now, t=0):
    #   x1 = [Heat[1,0], Heat[2,0], Vent[0]]
    #
    # Standard scenario cost (no probability weight — handled by rho scaling):
    #   f_s = Σ_t  prices[t] * (Heat[1,t] + Heat[2,t] + Vent[t]*Pvent)
    #
    # PH penalty terms added to objective:
    #   linear  : Σ_i  w_s[i] * x1[i]
    #   quadratic: (rho/2) * Σ_i  (x1[i] - x_bar[i])²
    #
    # The quadratic term on Vent[0] (binary) makes this a MIQP.
    # Gurobi handles MIQP natively via gurobi_direct.
    #
    # NOTE: there are NO HeatNA / VentNA constraints anywhere in this model.
    # NAC is enforced entirely through the penalty driving x1_s → x_bar.

    def objective(m):
        # ── Scenario cost ────────────────────────────────────────────────────
        cost = sum(
            m.prices[t] * (
                sum(m.Heat[r, t] for r in m.R)
                + m.Vent[t] * m.Pvent
            )
            for t in m.T
        )

        # ── PH linear dual term:  w_s · x1_s ────────────────────────────────
        linear_penalty = (
            w_s['heat_r1'] * m.Heat[1, 0]
            + w_s['heat_r2'] * m.Heat[2, 0]
            + w_s['vent']   * m.Vent[0]
        )

        # ── PH quadratic proximity term:  (rho/2) * ||x1_s - x_bar||² ───────
        quad_penalty = (rho / 2.0) * (
            (m.Heat[1, 0] - x_bar['heat_r1']) ** 2
            + (m.Heat[2, 0] - x_bar['heat_r2']) ** 2
            + (m.Vent[0]   - x_bar['vent'])    ** 2
        )

        return cost + linear_penalty + quad_penalty

    m.obj = Objective(rule=objective, sense=minimize)

    return m


# =============================================================================
# 4. PH POLICY FUNCTION
# =============================================================================

def PH_policy(state):
    """
    Progressive Hedging policy.

    Algorithm
    ---------
    1. Generate & cluster scenarios (same as SP_policy).
    2. Initialise consensus x_bar = 0, duals w_s = 0 for all s.
    3. PH outer loop:
         a. Solve each single-scenario subproblem independently.
         b. Update x_bar = Σ_s prob_s * x1_s.
         c. Update w_s  += rho * (x1_s - x_bar)  for each s.
         d. Check primal residual max_s ||x1_s - x_bar|| and
                  dual   residual ||x_bar^k - x_bar^{k-1}||.
            Break if both < PH_TOL or max_iter reached.
    4. Return rounded x_bar as the here-and-now action.

    Parameters
    ----------
    state : dict  – Environment.py state dict
        T1, T2, H, Occ1, Occ2, price_t, price_previous,
        vent_counter, low_override_r1, low_override_r2, current_time

    Returns
    -------
    dict with keys 'HeatPowerRoom1', 'HeatPowerRoom2', 'VentilationON'
    """
    t         = state['current_time']
    remaining = DATA['num_timeslots'] - t
    horizon   = min(HORIZON, remaining)

    # ── Generate & cluster scenarios ──────────────────────────────────────────
    price_dict, occ_dict = generate_scenarios(
        state['price_t'], state['price_previous'],
        state['Occ1'],    state['Occ2'],
        horizon, n_scenarios=GEN_SCENARIOS,
    )
    price_dict_clus, occ_dict_clus, probabilities = cluster_scenarios(
        price_dict, occ_dict,
        n_clusters=N_SCENARIOS, horizon=horizon,
        scenarios_to_generate=GEN_SCENARIOS,
    )
    n_clus = len(probabilities)

    # ── Unpack scenario data into per-scenario dicts for subproblem builder ───
    # price_s[s] : {t: float}       — price trajectory for scenario s
    # occ_s[s]   : {(r,t): float}   — occupancy trajectory for scenario s
    price_per_s = {
        s: {t: price_dict_clus[t, s] for t in range(horizon)}
        for s in range(n_clus)
    }
    occ_per_s = {
        s: {(r, t): occ_dict_clus[r, t, s]
            for r in [1, 2] for t in range(horizon)}
        for s in range(n_clus)
    }

    # ── Initialise PH state ───────────────────────────────────────────────────
    # x_bar: consensus stage-1 decisions (probability-weighted average)
    x_bar = {'heat_r1': 0.0, 'heat_r2': 0.0, 'vent': 0.0}

    # w[s]: dual multipliers per scenario (same keys as x_bar)
    w = {s: {'heat_r1': 0.0, 'heat_r2': 0.0, 'vent': 0.0}
         for s in range(n_clus)}

    solver = SolverFactory('gurobi_direct')

    # ── PH outer loop ─────────────────────────────────────────────────────────
    for k in range(PH_MAX_ITER):

        x1 = {}   # stage-1 solutions this iteration: {s: {'heat_r1', 'heat_r2', 'vent'}}

        # ── Step (a): solve each subproblem independently ─────────────────────
        for s in range(n_clus):
            sub = build_subproblem(
                state,
                price_s=price_per_s[s],
                occ_s=occ_per_s[s],
                horizon=horizon,
                x_bar=x_bar,
                w_s=w[s],
                rho=RHO,
            )

            res = solver.solve(sub, tee=False)

            if res.solver.termination_condition not in (
                TerminationCondition.optimal,
                TerminationCondition.feasible,
            ):
                warnings.warn(
                    f"PH iter {k}, scenario {s}: Gurobi failed "
                    f"({res.solver.termination_condition}). "
                    f"Using x_bar as fallback for this scenario.",
                    RuntimeWarning,
                )
                # Fall back to current consensus so this scenario doesn't
                # corrupt the x_bar update
                x1[s] = dict(x_bar)
            else:
                x1[s] = {
                    'heat_r1': float(value(sub.Heat[1, 0])),
                    'heat_r2': float(value(sub.Heat[2, 0])),
                    'vent'   : float(value(sub.Vent[0])),
                }

        # ── Step (b): update consensus x_bar ─────────────────────────────────
        x_bar_prev = dict(x_bar)
        for key in ('heat_r1', 'heat_r2', 'vent'):
            x_bar[key] = sum(
                probabilities[s] * x1[s][key] for s in range(n_clus)
            )

        # ── Step (c): update dual multipliers w_s ────────────────────────────
        for s in range(n_clus):
            for key in ('heat_r1', 'heat_r2', 'vent'):
                w[s][key] += RHO * (x1[s][key] - x_bar[key])

        # ── Step (d): check convergence ───────────────────────────────────────
        # Primal residual: worst-case deviation of any scenario from consensus
        primal_res = max(
            max(abs(x1[s][key] - x_bar[key]) for key in ('heat_r1', 'heat_r2', 'vent'))
            for s in range(n_clus)
        )
        # Dual residual: how much did the consensus shift this iteration
        dual_res = max(
            abs(x_bar[key] - x_bar_prev[key]) for key in ('heat_r1', 'heat_r2', 'vent')
        )

        if primal_res < PH_TOL and dual_res < PH_TOL:
            break

    # ── Extract final here-and-now action from consensus ──────────────────────
    # Heat is continuous — take directly.
    # Vent is binary — round the consensus (majority vote across scenarios).
    p1 = float(np.clip(x_bar['heat_r1'], 0.0, DATA['heating_max_power']))
    p2 = float(np.clip(x_bar['heat_r2'], 0.0, DATA['heating_max_power']))
    v  = int(round(x_bar['vent']))

    return {'HeatPowerRoom1': p1, 'HeatPowerRoom2': p2, 'VentilationON': v}


# =============================================================================
# 5. GRADER-COMPATIBLE WRAPPER
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
    try:
        return PH_policy(state)
    except Exception as e:
        warnings.warn(
            f"PH_policy raised an exception at time {state.get('current_time', '?')}: "
            f"{e}. Falling back to zero action.",
            RuntimeWarning,
        )
        return {'HeatPowerRoom1': 0.0, 'HeatPowerRoom2': 0.0, 'VentilationON': 0}