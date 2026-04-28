"""
Hybrid_policy.py
============
Hybrid Stochastic Programming policy with heuristic functions for the restaurant HVAC system.

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
import warnings
from pathlib import Path

from pyomo.opt import TerminationCondition

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from pyomo.environ import (
    ConcreteModel, Set, Param, Var, Objective, Constraint,
    SolverFactory, NonNegativeReals, minimize, Binary, value,
)

from Data.v2_SystemCharacteristics import get_fixed_data
from Data.PriceProcessRestaurant import price_model
from Data.OccupancyProcessRestaurant import next_occupancy_levels

# ── Module-level RNG (re-seeded once per import) ───────────────────────────────
#_rng = np.random.default_rng(seed=42)

# ── Hyper-parameters ───────────────────────────────────────────────────────────
HORIZON       = 4    # lookahead steps  (must be >= 3 due to vent-inertia)
GEN_SCENARIOS = 50  # Monte-Carlo draws before clustering
N_SCENARIOS   = 10   # K-Means clusters (representative scenarios)

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
    if "hour" not in ts_df.columns or "cluster_id" not in ts_df.columns:
        _CLUSTERED_CACHE = None
        return None

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

    if probabilities.size == 0 or probabilities.sum() <= 0:
        _CLUSTERED_CACHE = None
        return None

    probabilities = probabilities / probabilities.sum()
    _CLUSTERED_CACHE = (price_matrix, occ1_matrix, occ2_matrix, probabilities)
    return _CLUSTERED_CACHE


def _clustered_scenarios_from_csv(horizon):
    """
    Build clustered scenario dicts from the precomputed CSVs.

    Falls back to None if the CSVs are missing or incompatible.
    """
    clustered = _load_clustered_cache()
    if clustered is None:
        return None

    price_matrix, occ1_matrix, occ2_matrix, probabilities = clustered
    if horizon > price_matrix.shape[0]:
        horizon = price_matrix.shape[0]

    n_clusters = price_matrix.shape[1]
    price_dict_clus = {}
    occ_dict_clus = {}

    for s in range(n_clusters):
        for t in range(horizon):
            price_dict_clus[t, s] = float(price_matrix[t, s])
            occ_dict_clus[1, t, s] = float(occ1_matrix[t, s])
            occ_dict_clus[2, t, s] = float(occ2_matrix[t, s])

    return price_dict_clus, occ_dict_clus, probabilities

# =============================================================================
# 1. SYSTEM PARAMETERS
# =============================================================================

DATA = get_fixed_data()

# =============================================================================
# 3. SCENARIO GENERATION
# =============================================================================

def generate_scenarios(price_now, price_prev,
                       occ_r1_now, occ_r2_now,
                       horizon, n_scenarios):
    """
    Draw *n_scenarios* independent Monte-Carlo sample paths over *horizon*
    steps, starting from the current observed state.

    Returns
    -------
    price_dict   : {(t, s): float}        – price at lookahead step t, scenario s
    occ_dict     : {(r, t, s): float}     – occupancy of room r
    """
    price_dict   = {}
    occ_dict     = {}

    for s in range(n_scenarios):
        p_cur,  p_prev  = price_now,  price_prev
        o1_cur, o2_cur  = occ_r1_now, occ_r2_now

        for t in range(horizon):
            p_next           = price_model(p_cur, p_prev)
            o1_next, o2_next = next_occupancy_levels(o1_cur, o2_cur)

            price_dict[t, s]   = p_next
            occ_dict[1, t, s]  = o1_next
            occ_dict[2, t, s]  = o2_next

            p_prev, p_cur   = p_cur,  p_next
            o1_cur, o2_cur  = o1_next, o2_next

    return price_dict, occ_dict


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

    for s in range(n_clusters):
        for t in range(horizon):
            price_dict_clus[t, s]   = float(centroids[s, t])
            occ_dict_clus[1, t, s]  = float(max(0.0, centroids[s, horizon + t]))
            occ_dict_clus[2, t, s]  = float(max(0.0, centroids[s, 2*horizon + t]))

    return price_dict_clus, occ_dict_clus, probabilities


# =============================================================================
# 5. PYOMO MILP MODEL  (2-stage SP)
# =============================================================================

def build_sp_model(state, price_dict_clus, occ_dict_clus, horizon, n_clus, probabilities, heating_bonus):
    """
    Build the 2-stage stochastic MILP for the SP policy.

    Parameters
    ----------
    state : dict – keys: T_in_r1, T_in_r2, humidity,
                         vent_prev, vent_on_count
    price_dict_clus    : {(t, s): float}
    occ_dict_clus      : {(r, t, s): float}
    horizon       : int
    n_clus   : int
    probabilities : array-like, shape (n_clus,)

    Returns
    -------
    Pyomo ConcreteModel (unsolved)
    """

    d      = DATA    
    _num_timeslots = int(d['num_timeslots'])

    m = ConcreteModel()

    # ── Sets ──────────────────────────────────────────────────────────────────
    m.R   = Set(initialize=[1, 2])
    m.T   = Set(initialize=list(range(horizon)))
    m.S   = Set(initialize=list(range(n_clus)))
    m.RTS = m.R * m.T * m.S
    m.TS  = m.T * m.S
    # ── Current real state ────────────────────────────────────────────────────
    v_prev = 1 if state['vent_counter'] > 0 else 0
    vc     = int(state['vent_counter'])

    m.Tinit    = Param(m.R, initialize={1: state['T1'],
                                        2: state['T2']})
    m.Hinit    = Param(initialize=state['H'])
    m.VentInit = Param(initialize=v_prev)
     # ── Scenario parameters ───────────────────────────────────────────────────
    m.O      = Param(m.RTS, initialize=occ_dict_clus)
    m.prices = Param(m.TS,  initialize=price_dict_clus)
    m.pi     = Param(m.S,   initialize={s: float(probabilities[s])
                                        for s in range(n_clus)})
    m.Tout   = Param(m.T,   initialize={t: d['outdoor_temperature'][t % _num_timeslots]
                                        for t in range(horizon)})

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

    # ── Decision variables ────────────────────────────────────────────────────
    m.Vent   = Var(m.TS,  domain=Binary)
    m.Vstart = Var(m.TS,  domain=Binary)   # startup signal for ventilation
    m.Heat   = Var(m.RTS, domain=NonNegativeReals, bounds=(0, d['heating_max_power']))

    # ── Auxiliary binaries for overrule controllers ───────────────────────────
    m.y_low  = Var(m.RTS, domain=Binary)   # 1 if T < Tmin
    m.y_ok   = Var(m.RTS, domain=Binary)   # 1 if T > Tok
    m.y_high = Var(m.RTS, domain=Binary)   # 1 if T > Thigh
    m.u      = Var(m.RTS, domain=Binary)   # overrule memory

    # ── State variables ───────────────────────────────────────────────────────
    m.T_in = Var(m.RTS, domain=NonNegativeReals)
    m.Hum  = Var(m.TS,  domain=NonNegativeReals)

    # ── Temperature dynamics ──────────────────────────────────────────────────
    def temp_dynamics(m, r, t, s):
        if t == 0:
            return m.T_in[r, t, s] == m.Tinit[r]

        tp      = t - 1
        r_other = 2 if r == 1 else 1
        return m.T_in[r, t, s] == (
            m.T_in[r, tp, s]
            + m.Zexch * (m.T_in[r_other, tp, s] - m.T_in[r, tp, s])
            + m.Zloss * (m.Tout[tp]               - m.T_in[r, tp, s])
            + m.Zconv * m.Heat[r, tp, s]
            - m.Zcool * m.Vent[tp, s]
            + m.Zocc  * m.O[r, tp, s]
        )
    m.TempDyn = Constraint(m.RTS, rule=temp_dynamics)

    # ── Humidity dynamics ─────────────────────────────────────────────────────
    def hum_dynamics(m, t, s):
        if t == 0:
            return m.Hum[t, s] == m.Hinit
        tp = t - 1
        return m.Hum[t, s] == (
            m.Hum[tp, s]
            - m.Hvent * m.Vent[tp, s]
            + m.Hocc  * (m.O[1, tp, s] + m.O[2, tp, s])
        )
    m.HumDyn = Constraint(m.TS, rule=hum_dynamics)
    # ── Overrule controller: LOW temperature ──────────────────────────────────

    # ── 1. High temperature: forced heating shutdown ──────────────────────────
    # y_high = 1  ⟺  T_in > Thigh
    m.CThigh1 = Constraint(m.RTS,
        rule=lambda m, r, t, s: m.T_in[r, t, s] >= m.Thigh - m.M_temp * (1 - m.y_high[r, t, s]))
    m.CThigh2 = Constraint(m.RTS,
        rule=lambda m, r, t, s: m.T_in[r, t, s] <= m.Thigh + m.M_temp * m.y_high[r, t, s])
    m.CHeatOff = Constraint(m.RTS,
        rule=lambda m, r, t, s: m.Heat[r, t, s] <= m.Pr * (1 - m.y_high[r, t, s]))

    # ── 2. Low temperature: overrule activation ───────────────────────────────
    # y_low = 1  ⟺  T_in < Tmin
    m.CTlow1 = Constraint(m.RTS,
        rule=lambda m, r, t, s: m.T_in[r, t, s] <= m.Tmin + m.M_temp * (1 - m.y_low[r, t, s]))
    m.CTlow2 = Constraint(m.RTS,
        rule=lambda m, r, t, s: m.T_in[r, t, s] >= m.Tmin - m.M_temp * m.y_low[r, t, s])

      # ── 3. Temperature-OK: overrule deactivation ──────────────────────────────
    # y_ok = 1  ⟺  T_in >= Tok
    m.CTok1 = Constraint(m.RTS,
        rule=lambda m, r, t, s: m.T_in[r, t, s] >= m.Tok - m.M_temp * (1 - m.y_ok[r, t, s]))
    m.CTok2 = Constraint(m.RTS,
        rule=lambda m, r, t, s: m.T_in[r, t, s] <= m.Tok + m.M_temp * m.y_ok[r, t, s])

     # ── 4. Overrule memory (u) ────────────────────────────────────────────────
    # u >= y_low
    m.CU1 = Constraint(m.RTS,
        rule=lambda m, r, t, s: m.u[r, t, s] >= m.y_low[r, t, s])

    # u <= u_prev + y_low  
    def c_u2(m, r, t, s):
        u_prev = state[f'low_override_r{r}'] if t == 0 else m.u[r, t - 1, s]
        return m.u[r, t, s] <= u_prev + m.y_low[r, t, s]
    m.CU2 = Constraint(m.RTS, rule=c_u2)

    # Heat >= Pr * u  
    m.CHeatMax = Constraint(m.RTS,
        rule=lambda m, r, t, s: m.Heat[r, t, s] >= m.Pr * m.u[r, t, s])

    # u >= u_prev - y_ok  
    def c_u3(m, r, t, s):
        u_prev = state.get(f'low_override_r{r}', 0) if t == 0 else m.u[r, t - 1, s]
        return m.u[r, t, s] >= u_prev - m.y_ok[r, t, s]
    m.CU3 = Constraint(m.RTS, rule=c_u3)

    # u <= 1 - y_ok  (si disattiva appena T >= Tok)
    m.CU4 = Constraint(m.RTS,
        rule=lambda m, r, t, s: m.u[r, t, s] <= 1 - m.y_ok[r, t, s])

    # =========================================================================
    # ── Ventilation constraints ───────────────────────────────────────────────
    # =========================================================================

    # ── Startup signal ──────────
    def c_vstart1(m, t, s):
        v_p = m.VentInit if t == 0 else m.Vent[t - 1, s]
        return m.Vstart[t, s] >= m.Vent[t, s] - v_p
    m.CVstart1 = Constraint(m.TS, rule=c_vstart1)

    m.CVstart2 = Constraint(m.TS,
        rule=lambda m, t, s: m.Vstart[t, s] <= m.Vent[t, s])

    def c_vstart3(m, t, s):
        v_p = m.VentInit if t == 0 else m.Vent[t - 1, s]
        return m.Vstart[t, s] <= 1 - v_p
    m.CVstart3 = Constraint(m.TS, rule=c_vstart3)

    # ── Minimum uptime:  ───────────────
    def min_uptime(m, t, s):
        end_idx = min(t + m.U_vent - 1, m.L - 1)
        min_val = min(m.U_vent, m.L - t)
        return sum(m.Vent[tau, s] for tau in range(t, end_idx + 1)) >= min_val * m.Vstart[t, s]
    m.MinVentOn = Constraint(m.TS, rule=min_uptime)

    # ── Humidity overrule:─────────────────────
    m.CVentHum = Constraint(m.TS,
        rule=lambda m, t, s: m.Hum[t, s] <= m.Hhigh + m.M_hum * m.Vent[t, s])

    # =========================================================================
    # ── Non-anticipativity at t = 0 (here-and-now decisions) ────────────────────────────
    # =========================================================================
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

    # =========================================================================
    # ── Objective: minimise expected electricity cost ────────────────────────
    # =========================================================================
    def objective(m):
        return sum(
            m.pi[s] * sum(
                m.prices[t, s] * (
                    sum(m.Heat[r, t, s] for r in m.R)
                    + m.Vent[t, s] * m.Pvent
                )
                for t in m.T
            )
            # Subtract heating bonus at t=0 only — the here-and-now decision.
            # This makes the optimizer prefer heating when price is cheap
            # and temperature has headroom, without changing any constraints.
            - heating_bonus[s] * (m.Heat[1, 0, s] + m.Heat[2, 0, s])
            for s in m.S
        )
    m.obj = Objective(rule=objective, sense=minimize)

    return m

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
    # def debug_print_state(state):
    #     print("Debug incoming state:")
    #     for k, v in state.items():
    #         print(f"  {k}: {v}")
    
    #print(debug_print_state(state))

    # Make scenario generation reproducible per (day, timestep)
    #rng = np.random.default_rng(seed=int(state['current_time']) + 100)

    # rng = np.random.default_rng(seed=42 + state['current_time'])

    t         = state['current_time']
    remaining = DATA['num_timeslots'] - t
    horizon   = min(HORIZON, remaining)

    # if horizon <= 0:
    #     return {'heat_r1': 0.0, 'heat_r2': 0.0, 'vent': 0}

    # # Handle missing previous price at t = 0
    # p_prev = state.get('price_previous') or 4.0

    # # Ventilation status: vent_counter > 0 means vent was ON last step
    # vc       = state.get('vent_counter', 0)
    # v_status = 1 if vc > 0 else 0

    # # ── Generate raw Monte-Carlo scenarios ────────────────────────────────────
    # price_dict, occ_dict = generate_scenarios(
    #     price_now   = state['price_t'],
    #     price_prev  = state['price_previous'],
    #     occ_r1_now  = state['Occ1'],
    #     occ_r2_now  = state['Occ2'],
    #     horizon     = horizon,
    #     n_scenarios = GEN_SCENARIOS
    # )

    clustered = _clustered_scenarios_from_csv(horizon)
    if clustered is None:
        raise RuntimeError("Clustered scenario CSVs are missing or incompatible.")

    price_dict_clus, occ_dict_clus, probabilities = clustered
    n_clus = len(probabilities)

    # ── Assemble current state for the MILP ──────────────────────────────────
    # milp_state = {
    #     'T_in_r1'        : state['T1'],
    #     'T_in_r2'        : state['T2'],
    #     'humidity'       : state['H'],
    #     'vent_prev'      : v_status,
    #     'vent_on_count'  : vc,
    #     'low_override_r1': state.get('low_override_r1', 0),
    #     'low_override_r2': state.get('low_override_r2', 0),
    # }

    # ── Heating heuristic: opportunistic preheating bonus ─────────────────────
    # Fires when price is cheap AND temperature has headroom below the ceiling.
    # Applied only at t=0 (the decision we actually execute).
    LAMBDA_HEAT     = 0.5   # tune this — increase if heuristic has no effect
    PRICE_THRESHOLD = 0.15  # below this price, bonus activates (check your price scale)
    T_MAX           = float(value(DATA['temp_max_comfort_threshold']) 
                            if hasattr(DATA['temp_max_comfort_threshold'], '__float__') 
                            else DATA['temp_max_comfort_threshold'])

    temp_headroom = max(0.0, T_MAX - max(state['T1'], state['T2']))  # headroom in both rooms

    heating_bonus = {
        s: LAMBDA_HEAT
           * max(0.0, PRICE_THRESHOLD - price_dict_clus[0, s])  # cheap price bonus
           * temp_headroom                                        # only if room to heat
        for s in range(n_clus)
    }

    # ── Build and solve ───────────────────────────────────────────────────────
    model  = build_sp_model(state,
                            price_dict_clus, occ_dict_clus,
                            horizon, n_clus, probabilities,
                            heating_bonus)
    solver = SolverFactory('gurobi_direct')
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

    # ── Extract here-and-now decisions (non-anticipativity → s=0 is canonical)
    s0 = 0
    p1 = float(value(model.Heat[1, 0, s0]))
    p2 = float(value(model.Heat[2, 0, s0]))
    v  = int(round(float(value(model.Vent[0, s0]))))



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
