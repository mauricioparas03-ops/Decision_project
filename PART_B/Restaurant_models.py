Restaurant_models.py

"""
This module contains the data and model-building code for the restaurant SP policy.
The main function is build_sp_model(), which constructs the Pyomo model for the SP policy.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyomo.environ import (
    ConcreteModel, Set, Param, Var, Objective, Constraint, Block,
    SolverFactory, NonNegativeReals, minimize, Binary, value
)
from pyomo.opt import TerminationCondition

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class Restaurant_models:
    ## - Fixed data characteristics -----------------------------
    def get_fixed_data():
        """
        Returns the fixed system characteristics.
        THIS FUNCTION SHOULD NOT BE CHANGED.
        """
        num_timeslots = 10
        return {
            'num_timeslots'              : num_timeslots,
            'initial_temperature'        : 21.0,
            'previous_initial_temperature': 21.0,
            'initial_humidity'           : 40.0,
            'heating_max_power'          : 3.0,       # Pr  (kW)
            'heat_exchange_coeff'        : 0.6,        # ζ_exch
            'heating_efficiency_coeff'   : 1.0,        # ζ_conv
            'thermal_loss_coeff'         : 0.1,        # ζ_loss
            'heat_vent_coeff'            : 0.7,        # ζ_cool
            'heat_occupancy_coeff'       : 0.02,       # ζ_occ
            'temp_min_comfort_threshold' : 18.0,       # T_low
            'temp_OK_threshold'          : 22.0,       # T_OK
            'temp_max_comfort_threshold' : 26.0,       # T_high
            'humidity_threshold'         : 70.0,       # H_high
            'vent_min_up_time'           : 3,          # 3 hour ventilation time
            'ventilation_power'          : 2.0,        # P_vent (kW)
            'humidity_occupancy_coeff'   : 0.18,       # η_occ
            'humidity_vent_coeff'        : 15.0,       # η_vent
            'outdoor_temperature'        : [
                3 * np.sin(2 * np.pi * t / num_timeslots - np.pi / 2)
                for t in range(num_timeslots)
            ],
        }
    

    #----------------------------------------------------------------------------------------------------------
    # Look ahead functions: 

    # ── Electricity price process  ──────────────────────────────────────────────
    def price_model(current_price, previous_price, rng=None):
        """
        One-step-ahead price sample.

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
        noise          = (rng.normal(0, 0.5) if rng is not None
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


    # ── Occupancy process (coupled mean-reverting, one room per step) ─────────────
    def next_occupancy_levels(r1_current, r2_current, rng=None):
        """
        One-step-ahead occupancy sample for both rooms.

        Returns
        -------
        (r1_next, r2_next) : (float, float)
        """
        mean_r1, mean_r2 = 35.0, 25.0
        rev              = 0.25
        coupling         = 0.1

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
    
    #----------------------------------------------------------------------------------------------------------
    # Scenario Generation:

    def generate_scenarios(price_now, price_prev,
                        occ_r1_now, occ_r2_now,
                        horizon, n_scenarios, rng):
        """
        Draw *n_scenarios* independent sample paths over *horizon* steps,
        starting from the current observed state.

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
            p_cur,  p_prev  = price_now,   price_prev
            o1_cur, o2_cur  = occ_r1_now,  occ_r2_now

            for t in range(horizon):
                p_next          = Restaurant_models.price_model(p_cur, p_prev, rng)
                o1_next, o2_next = Restaurant_models.next_occupancy_levels(o1_cur, o2_cur, rng)

                price_dict[t, s]   = p_next
                occ_dict[1, t, s]  = o1_next
                occ_dict[2, t, s]  = o2_next
                hum_occ_dict[t, s] = o1_next + o2_next

                p_prev, p_cur   = p_cur,  p_next
                o1_cur, o2_cur  = o1_next, o2_next

        return price_dict, occ_dict, hum_occ_dict


    # ── Scenario Clustering ───────────────────────────────────────────────

    def cluster_scenarios(price_dict, occ_dict, n_clusters, horizon, scenarios_to_generate):
        
        scaler = StandardScaler()
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        #horizon = DATA['num_timeslots']

        X = np.array([
                [price_dict[t, s] for t in range(horizon)] +
                [occ_dict[1, t, s] for t in range(horizon)] +
                [occ_dict[2, t, s] for t in range(horizon)]
                for s in range(scenarios_to_generate)
            ])

        X_scaled = scaler.fit_transform(X)

        # K means clustering

        km = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
        km.fit(X_scaled)

        labels        = km.labels_
        cluster_sizes = np.bincount(labels)
        probabilities = cluster_sizes / scenarios_to_generate

        # Unpack vector into dictonaries that Pyomo can use
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
    
    # Building the SP model --------------------------------------------------------------------------------

    def build_sp_model(current_state, price_dict, occ_dict, hum_occ_dict,
                   horizon, n_scenarios, probabilities):
        """
        Build the stochastic MILP for the SP policy.

        Parameters
        ----------
        current_state : dict with keys
            T_in_r1      – current temperature room 1  (°C)
            T_in_r2      – current temperature room 2  (°C)
            humidity     – current humidity             (%)
            vent_prev    – ventilation status at t-1    (0 or 1)
            vent_on_count– consecutive hours vent has been ON (for inertia)
        price_dict    : {(t,s): float}
        occ_dict      : {(r,t,s): float}
        hum_occ_dict  : {(t,s): float}
        horizon       : int
        n_scenarios   : int

        Returns
        -------
        Pyomo ConcreteModel (unsolved)
        """
        d  = Restaurant_models.get_fixed_data()
        M  = 200.0          # big-M constant (larger than any realistic T or H range)

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
        m.pi = Param(m.S, initialize={s: probabilities[s] for s in range(n_scenarios)})

        m.O      = Param(m.RTS, initialize=occ_dict)
        m.prices = Param(m.TS,  initialize=price_dict)
        m.HumOcc = Param(m.TS,  initialize=hum_occ_dict)

        # ── Decision variables ────────────────────────────────────────────────────
        m.Heat = Var(m.RTS, domain=NonNegativeReals, bounds=(0, d['heating_max_power']))
        m.Vent = Var(m.TS,  domain=Binary)
        m.Uon  = Var(m.TS,  domain=Binary)   # ventilation start-up indicator
        m.Uoff = Var(m.TS,  domain=Binary)   # ventilation shut-down indicator

        # ── Auxiliary binary variables for overrule controllers ───────────────────
        m.u = Var(m.RTS, domain=Binary)   # 1 → low-temp overrule active
        m.w = Var(m.RTS, domain=Binary)   # 1 → temperature has recovered to T_OK
        m.y = Var(m.RTS, domain=Binary)   # 1 → high-temp overrule active

        # ── State variables ───────────────────────────────────────────────────────
        m.T_in = Var(m.RTS, domain=NonNegativeReals)
        m.Hum  = Var(m.TS,  domain=NonNegativeReals)

        # ── Temperature dynamics ──────────────────────────────────────────────────
        def temp_dynamics(m, r, t, s):
            if t == 0:
                return m.T_in[r, t, s] == T_init[r]
            tp = t - 1
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
        # u[r,t,s] = 1  iff  overrule is active (T < T_low or still recovering)
        # w[r,t,s] = 1  iff  T has reached T_OK this step (deactivates overrule)

        def u_activation(m, r, t, s):
            # T_in >= T_low - M*u  →  u forced to 1 when T < T_low
            return m.T_in[r, t, s] >= m.Tmin - M * m.u[r, t, s]
        m.UActivation = Constraint(m.RTS, rule=u_activation)

        def w_deactivation(m, r, t, s):
            # T_in >= T_OK - M*(1-w)  →  w forced to 1 when T >= T_OK
            return m.T_in[r, t, s] >= m.Tok - M * (1 - m.w[r, t, s])
        m.WDeactivation = Constraint(m.RTS, rule=w_deactivation)

        def u_persistence(m, r, t, s):
            # Overrule persists: u[t] >= u[t-1] - w[t]
            if t == 0:
                m.u[r, t, s].fix(0)
                m.w[r, t, s].fix(0)
                return Constraint.Skip
            return m.u[r, t, s] >= m.u[r, t - 1, s] - m.w[r, t, s]
        m.UPersistence = Constraint(m.RTS, rule=u_persistence)

        def heat_max_when_overrule(m, r, t, s):
            # u = 1  →  Heat = Pr
            return m.Heat[r, t, s] >= m.Pr * m.u[r, t, s]
        m.HeatMaxOverrule = Constraint(m.RTS, rule=heat_max_when_overrule)

        # ── Overrule controller: HIGH temperature ─────────────────────────────────
        def y_activation(m, r, t, s):
            # T_in <= T_high + M*y  →  y forced to 1 when T > T_high
            return m.T_in[r, t, s] <= m.Thigh + M * m.y[r, t, s]
        m.YActivation = Constraint(m.RTS, rule=y_activation)

        def heat_off_when_overrule(m, r, t, s):
            # y = 1  →  Heat = 0
            return m.Heat[r, t, s] <= m.Pr * (1 - m.y[r, t, s])
        m.HeatOffOverrule = Constraint(m.RTS, rule=heat_off_when_overrule)

        # ── Humidity overrule: force ventilation ON ───────────────────────────────
        def vent_humidity_overrule(m, t, s):
            # Hum <= H_high + M*v  →  v forced to 1 when Hum > H_high
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
        # All scenarios must agree on the first-stage action.
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