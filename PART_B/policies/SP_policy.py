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
HORIZON       = 2    # lookahead steps  (must be >= 3 due to vent-inertia)
GEN_SCENARIOS = 50  # Monte-Carlo draws before clustering
N_CLUSTERS    = 10   # K-Means clusters (representative scenarios)

# =============================================================================
# 1. SYSTEM PARAMETERS
# =============================================================================

DATA = get_fixed_data()

# =============================================================================
# 3. SCENARIO GENERATION
# =============================================================================

def generate_scenarios(price_now, price_prev,
                       occ_r1_now, occ_r2_now,
                        n_scenarios):
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
        p_next           = price_model(p_cur, p_prev)
        o1_next, o2_next = next_occupancy_levels(o1_cur, o2_cur)

        price_dict[s]   = p_next
        occ_dict[1,s]  = o1_next
        occ_dict[2,s]  = o2_next

        p_prev, p_cur   = p_cur,  p_next
        o1_cur, o2_cur  = o1_next, o2_next

    return price_dict, occ_dict


# =============================================================================
# 4. SCENARIO CLUSTERING  (K-Means → weighted centroids)
# =============================================================================

def cluster_scenarios(price_dict, occ_dict, n_clusters, scenarios_to_generate):
    """
    Reduce *scenarios_to_generate* Monte-Carlo paths to *n_clusters*
    representative centroids via K-Means, returning Pyomo-ready dicts.

    Parameters
    ----------
    price_dict            : {(t, s): float}
    occ_dict              : {(r, t, s): float}
    n_clusters            : int   – number of clusters (K)
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
        [price_dict[s]] +
        [occ_dict[1,s]] +
        [occ_dict[2,s]]
        for s in range(scenarios_to_generate)
    ])

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    km.fit(X_scaled)

    labels        = km.labels_
    cluster_sizes = np.bincount(labels)
    probabilities = cluster_sizes / scenarios_to_generate

    # Unpack centroids back to original scale
    centroids = scaler.inverse_transform(km.cluster_centers_)

    price_dict_clus   = {}
    occ_dict_clus     = {}

    for s in range(n_clusters):
        # centroids[s] is a vector: [price, occ1, occ2]
        price_dict_clus[s]   = float(centroids[s, 0])
        occ_dict_clus[1, s]  = float(max(0.0, centroids[s, 1]))
        occ_dict_clus[2, s]  = float(max(0.0, centroids[s, 2]))

    return price_dict_clus, occ_dict_clus, probabilities

# =============================================================================
# 5. PYOMO MILP MODEL  (2-stage SP)
# =============================================================================

def build_sp_model(state, price_dict_clus, occ_dict_clus, n_clus, probabilities):
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
    eps = 10e-6

    m = ConcreteModel()

    # ── Sets ──────────────────────────────────────────────────────────────────
    m.R   = Set(initialize=[1, 2])
    m.S   = Set(initialize=list(range(n_clus)))
    m.T = Set(initialize=[state["current_time"], state["current_time"] + 1])
    m.RTS = m.R * m.T * m.S
    m.TS  = m.T * m.S
    m.RS = m.R * m.S
    # ── Current real state ────────────────────────────────────────────────────
    v_prev = 1 if state['vent_counter'] > 0 else 0
    m.Tinit    = Param(m.R, initialize={1: state['T1'],
                                        2: state['T2']})
    m.Hinit    = Param(initialize=state['H'])
    m.VentInit = Param(initialize=v_prev)
     # ── Scenario parameters ───────────────────────────────────────────────────
# Inizializzazione dinamica dell'occupazione
    def occ_init_rule(m, r, t, s):
        if t == 0:
            # Al tempo presente usiamo l'occupazione REALE dello state
            return float(state['Occ1'] if r == 1 else state['Occ2'])
        else:
            # Al tempo futuro usiamo il dato del cluster (tornando indietro all'indice relativo)
            return occ_dict_clus[r, t, s]
            
    m.O = Param(m.RTS, rule=occ_init_rule)


    # Inizializzazione dinamica dei prezzi
    def price_init_rule(m, t, s):
        if t == 0:
            # Prezzo REALE attuale
            return float(state['price_t'])
        else:
            # Prezzo futuro del cluster
            return price_dict_clus[t, s]
            
    m.prices = Param(m.TS, rule=price_init_rule)
    m.pi     = Param(m.S,   initialize={s: float(probabilities[s])
                                        for s in range(n_clus)})
    m.Tout = Param(m.T, initialize={t: d['outdoor_temperature'][t] for t in m.T})
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
    m.L = Param(initialize=len(m.T))  # number of lookahead steps (for vent inertia)
    m.M_temp = Param(initialize=100)
    m.M_hum  = Param(initialize=100)
    m.U_vent = Param(initialize=d['vent_min_up_time'])

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
# ── Temperature dynamics ──────────────────────────────────────────────────
    def temp_dynamics(m, r, t, s):
        # Se t è il tempo iniziale corrente, la temperatura è già decisa (è m.Tinit).
        # Non c'è una dinamica da calcolare in entrata, quindi saltiamo il vincolo.
        if t == state["current_time"]:  
            return Constraint.Skip 
            
        r_other = 2 if r == 1 else 1
        t_prev = t - 1  # Guardiamo all'ora precedente che ha causato lo stato attuale t
        
        if t_prev == state["current_time"]:  
            # Se l'ora precedente era il presente reale, i dati storici sono certi e nel m.Tinit
            T_past           = m.Tinit[r]
            T_OtherRoom_past = m.Tinit[r_other]
            occ_term         = state["Occ1"] if r == 1 else state["Occ2"]
        else:
            # Per i passi successivi nel futuro, usiamo le variabili calcolate dal modello a t-1
            T_past           = m.T_in[r, t_prev, s]
            T_OtherRoom_past = m.T_in[r_other, t_prev, s]
            # Nota: usa l'indice corretto del dizionario dei cluster [r, s]
            occ_term         = occ_dict_clus[r, s] 

        # Le decisioni e il meteo che cambiano la temperatura tra t_prev e t
        heat = m.Heat[r, t_prev, s]
        vent = m.Vent[t_prev, s]
        Tout = m.Tout[t_prev]

        # Equazione dinamica (Bilancio energetico)
        return m.T_in[r, t, s] == (
            T_past
            + m.Zexch * (T_OtherRoom_past - T_past)
            + m.Zloss * (Tout - T_past)  # Se Tout > T_past il sistema si scalda, coerente con il tuo +
            + m.Zconv * heat
            - m.Zcool * vent
            + m.Zocc * occ_term
        )
        
    m.TempDyn = Constraint(m.RTS, rule=temp_dynamics)

    # ── Humidity dynamics ─────────────────────────────────────────────────────
# ── Humidity dynamics ─────────────────────────────────────────────────────
    def hum_dynamics(m, t, s):
        # Se t è il tempo iniziale corrente, l'umidità è già decisa (è m.Hinit).
        # Non c'è una dinamica in entrata da calcolare, quindi saltiamo il vincolo.
        if t == state["current_time"]:
            return Constraint.Skip

        t_prev = t - 1  # L'ora precedente che determina lo stato attuale t

        if t_prev == state["current_time"]:
            # Se l'ora precedente era il presente reale, usiamo i dati storici certi dello state
            H_past   = m.Hinit
            occ_term = state["Occ1"] + state["Occ2"]
        else:
            # Per i passi successivi nel futuro, usiamo le variabili e i parametri dello scenario s
            H_past   = m.Hum[t_prev, s]
            # Assicurati che m.O sia indicizzato correttamente (es. m.O[stanza, scenario])
            # Se m.O ha anche l'indice temporale nel tuo modello, usa m.O[1, t_prev, s]
            occ_term = occ_dict_clus[1, s] + occ_dict_clus[2, s]

        # La decisione di ventilazione presa a t_prev che influisce sul tempo t
        vent = m.Vent[t_prev, s]

        # Equazione dinamica dell'umidità
        return m.Hum[t, s] == (
            H_past
            - m.Hvent * vent
            + m.Hocc  * occ_term
        )

    m.HumDyn = Constraint(m.TS, rule=hum_dynamics)
    # ── Overrule controller: LOW temperature ──────────────────────────────────

    # ── 1. High temperature: forced heating shutdown ──────────────────────────
    # y_high = 1  ⟺  T_in > Thigh
    m.CThigh1 = Constraint(m.RTS,
        rule=lambda m, r, t, s: m.T_in[r, t, s] >= eps + m.Thigh - m.M_temp * (1 - m.y_high[r, t, s]))
    m.CThigh2 = Constraint(m.RTS,
        rule=lambda m, r, t, s: m.T_in[r, t, s] <= m.Thigh + m.M_temp * m.y_high[r, t, s])
    m.CHeatOff = Constraint(m.RTS,
        rule=lambda m, r, t, s: m.Heat[r, t, s] <= m.Pr * (1 - m.y_high[r, t, s]))

    # ── 2. Low temperature: overrule activation ───────────────────────────────
    # y_low = 1  ⟺  T_in < Tmin
    m.CTlow1 = Constraint(m.RTS,
        rule=lambda m, r, t, s: m.T_in[r, t, s] <= m.Tmin + m.M_temp * (1 - m.y_low[r, t, s]))
    m.CTlow2 = Constraint(m.RTS,
        rule=lambda m, r, t, s: m.T_in[r, t, s] >= eps + m.Tmin - m.M_temp * m.y_low[r, t, s])

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
        u_prev = state[f'low_override_r{r}']if t == state["current_time"] else m.u[r, t - 1, s]
        return m.u[r, t, s] <= u_prev + m.y_low[r, t, s]
    m.CU2 = Constraint(m.RTS, rule=c_u2)

    # Heat >= Pr * u  
    m.CHeatMax = Constraint(m.RTS,
        rule=lambda m, r, t, s: m.Heat[r, t, s] >= m.Pr * m.u[r, t, s])

    # u >= u_prev - y_ok  
    def c_u3(m, r, t, s):
        u_prev = state[f'low_override_r{r}'] if t == state["current_time"] else m.u[r, t - 1, s]
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
        v_p = m.VentInit if t == state["current_time"] else m.Vent[t - 1, s]
        return m.Vstart[t, s] >= m.Vent[t, s] - v_p
    m.CVstart1 = Constraint(m.TS, rule=c_vstart1)

    m.CVstart2 = Constraint(m.TS,
        rule=lambda m, t, s: m.Vstart[t, s] <= m.Vent[t, s])

    def c_vstart3(m, t, s):
        v_p = m.VentInit if t == state["current_time"] else m.Vent[t - 1, s]
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
        return m.Heat[r, state["current_time"], s1] == m.Heat[r, state["current_time"], s2]
    m.HeatNA = Constraint(m.R, m.S, m.S, rule=heat_na)

    def vent_na(m, s1, s2):
        if s1 >= s2:
            return Constraint.Skip
        return m.Vent[state["current_time"], s1] == m.Vent[state["current_time"], s2]
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
    price_dict, occ_dict = generate_scenarios(state["price_t"], state["price_previous"],
                       state["Occ1"], state["Occ2"],
                       n_scenarios=GEN_SCENARIOS)

    price_dict_clus, occ_dict_clus, probabilities = cluster_scenarios(price_dict, occ_dict, N_CLUSTERS, scenarios_to_generate = GEN_SCENARIOS)

    # Build full price dict indexed by (t, s): at t = now use observed price, at t+1 use clustered price
    t0 = state["current_time"]
    t1 = t0 + 1
    price_matrix = {}
    for s in range(len(probabilities)):
        price_matrix[(t0, s)] = float(state['price_t'])
        # price_dict_clus currently maps s -> clustered future price
        price_matrix[(t1, s)] = float(price_dict_clus[s])

    # ── Build and solve ───────────────────────────────────────────────────────
    model  = build_sp_model(state,
                            price_matrix, occ_dict_clus,
                            N_CLUSTERS, probabilities)
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
    p1 = float(value(model.Heat[1, state["current_time"], s0]))
    p2 = float(value(model.Heat[2, state["current_time"], s0]))
    v  = int(round(float(value(model.Vent[state["current_time"], s0]))))



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
