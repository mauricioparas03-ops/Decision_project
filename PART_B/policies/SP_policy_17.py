"""Multi-stage stochastic policy (multiSP) with clearer English comments.

This module builds a scenario tree from the current state, constructs a Pyomo
mixed-integer model over the tree, solves it with Gurobi, and returns the
first-stage control decisions.

Only comments and docstrings were updated to be concise and in English; the
algorithmic logic and variable names are unchanged to preserve behavior.
"""

import numpy as np
import warnings
import time
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pyomo.environ import (
    ConcreteModel, Set, Param, Var, Objective, Constraint,
    SolverFactory, NonNegativeReals, minimize, Binary, value,
)
from pyomo.opt import TerminationCondition

from Data.v2_SystemCharacteristics import get_fixed_data
from Data.PriceProcessRestaurant import price_model
from Data.OccupancyProcessRestaurant import next_occupancy_levels
from sklearn.preprocessing import StandardScaler

# Hyper-parameters
HORIZON_MULTI = 4
N_CLUSTERS = 4
BRANCHING_FACTOR = 100

# Load fixed system data
DATA = get_fixed_data()


def build_scenario_tree(state, L, S, K):
    """Construct a scenario tree from the provided state.

    Parameters
    - state: dict containing the current price/occupancy and other state items
    - L: planning horizon (number of stages)
    - S: number of Monte Carlo samples per node
    - K: number of clusters (branching) per stage

    Returns
    - tree: list of lists; tree[t] is the list of nodes at stage t. Each node
      is a dict with keys: id, price, price_prev, occupancy1, occupancy2,
      probability, parent_id, children
    """
    global_id_counter = 1

    root = {
            "id": global_id_counter,
            "price": state['price_t'],
            "price_prev": state['price_previous'],
            "occupancy1": state['Occ1'],
            "occupancy2": state['Occ2'],
            "probability": 1.0,
            "parent_id": None, 
            "children": []
        }
    tree = [[root]]  
    global_id_counter += 1

    for tau in range(1, L):
        new_nodes = []
        for node in tree[tau - 1]:
            # --- Generate S samples of next state ---
            samples_prices= []
            sample_occ1 = []
            sample_occ2 = []
            for _ in range(S):
                price_next = price_model(node['price'], node['price_prev'])
                occ1_next, occ2_next = next_occupancy_levels(node['occupancy1'], node['occupancy2'])
                samples_prices.append(price_next)
                sample_occ1.append(occ1_next)
                sample_occ2.append(occ2_next)

            # --- Cluster samples into K clusters  ---
            X = np.column_stack([samples_prices, sample_occ1, sample_occ2])  
            n_samples = X.shape[0]
            K_eff = min(K, n_samples)

            if K_eff <= 0:
                # no samples generated, skip
                continue

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)


            if K_eff == 1:
                # single cluster: centroid is the mean, all labels 0
                labels = np.zeros(n_samples, dtype=int)
                # Mediod = sample closest to the mean
                mean = X_scaled.mean(axis=0)
                medoid_indices = [np.argmin(np.linalg.norm(X_scaled - mean, axis=1))]
            else:
                kmeans = KMeans(n_clusters=K_eff, random_state=0, n_init=10).fit(X_scaled)
                labels = kmeans.labels_
                centroids = kmeans.cluster_centers_

                medoid_indices = []
                for k in range(K_eff):
                    cluster_mask = labels == k
                    cluster_points = X_scaled[cluster_mask]
                    dist = np.linalg.norm(cluster_points - centroids[k], axis=1)
                     # index back into original X
                    original_indices = np.where(cluster_mask)[0]
                    medoid_idx = original_indices[np.argmin(dist)]
                    medoid_indices.append(medoid_idx)

            centers = X[medoid_indices]  # centroids in original scale

            # --- Create new nodes for each cluster center ---
            for k in range(K_eff):
                # Conditional probability: p(cluster k | parent)
                conditional_prob = np.sum(labels == k) / n_samples
                joint_prob = node['probability'] * conditional_prob 
                new_node = {
                    "id": global_id_counter,           
                    "price": centers[k][0],
                    "price_prev": node['price'],
                    "occupancy1": centers[k][1],
                    "occupancy2": centers[k][2],
                    "probability": joint_prob,
                    "parent_id": node['id'],          
                    "children": []
                }

                node['children'].append(new_node['id'])  
                new_nodes.append(new_node)               
                global_id_counter += 1
                
        tree.append(new_nodes)
    return tree


def get_descendant_chains(node_id, depth, nodes_map):
    """Return all descendant chains starting at node_id up to the given depth.

    Each chain is a list of node ids representing a contiguous path downward
    in the scenario tree. If a node has no children, the returned chain is the
    single-node list [node_id].
    """

    if depth <= 1:
        return [[node_id]]

    node = nodes_map[node_id]
    if not node["children"]:
        return [[node_id]]

    all_chains = []
    for child_id in node["children"]:
        child_chains = get_descendant_chains(child_id, depth - 1, nodes_map)
        for cc in child_chains:
            all_chains.append([node_id] + cc)
    return all_chains


def build_multisp_model(state, tree, horizon):
    """Construct the Pyomo model for the multi-stage stochastic control problem.

    The model is indexed on non-root scenario nodes. Variables named with a
    trailing '0' correspond to the root / first-stage state and decision.
    """

    d = DATA

    # Initial temperatures and humidity from the provided state
    T_init = {1: state["T1"], 2: state["T2"]}
    H_init = state["H"]
    occ1_root = state["Occ1"]
    occ2_root = state["Occ2"]

    vent_counter = int(state["vent_counter"])
    v_prev = 1 if vent_counter > 0 else 0

    # Determine whether a low-temperature override is active for each room
    low_override = {}
    for r, T_init_r, T_ok_threshold in [
        (1, state["T1"], DATA["temp_OK_threshold"]),
        (2, state["T2"], DATA["temp_OK_threshold"]),
    ]:
        ov = state[f"low_override_r{r}"]
        if T_init_r >= T_ok_threshold:
            ov = 0
        low_override[r] = ov
    for r in [1, 2]:
        T_r = state[f"T{r}"]
        if T_r > DATA["temp_max_comfort_threshold"]:
            low_override[r] = 0

    eps = 10e-7

    # Flatten the tree into a node id -> node dict for fast lookup. Exclude root
    root_id = tree[0][0]["id"]
    nodes_map = {
        node["id"]: node
        for stage in tree
        for node in stage
        if node["id"] != root_id
    }

    all_node_ids = list(nodes_map.keys())
    leaf_ids = [nid for nid, node in nodes_map.items() if not node["children"]]
    non_root_ids = [nid for nid in all_node_ids if nid != root_id]
    decision_ids = non_root_ids

    m = ConcreteModel()

    # Index sets
    m.R = Set(initialize=[1, 2])
    m.N = Set(initialize=non_root_ids)
    m.RN = Set(initialize=m.R * m.N)

    # Parameters (copied from DATA)
    m.Pr = Param(initialize=d["heating_max_power"])
    m.Pvent = Param(initialize=d["ventilation_power"])
    m.Zexch = Param(initialize=d["heat_exchange_coeff"])
    m.Zconv = Param(initialize=d["heating_efficiency_coeff"])
    m.Zloss = Param(initialize=d["thermal_loss_coeff"])
    m.Zcool = Param(initialize=d["heat_vent_coeff"])
    m.Zocc = Param(initialize=d["heat_occupancy_coeff"])
    m.Hocc = Param(initialize=d["humidity_occupancy_coeff"])
    m.Hvent = Param(initialize=d["humidity_vent_coeff"])
    m.Tmin = Param(initialize=d["temp_min_comfort_threshold"])
    m.Tok = Param(initialize=d["temp_OK_threshold"])
    m.Thigh = Param(initialize=d["temp_max_comfort_threshold"])
    m.Hhigh = Param(initialize=d["humidity_threshold"])
    m.M_temp = Param(initialize=100.0)
    m.M_hum = Param(initialize=100.0)
    m.U_vent = Param(initialize=d["vent_min_up_time"])

    # Outdoor temperature over the horizon (time-indexed)
    m.Tout = Param(
        range(state["current_time"], state["current_time"] + horizon),
        initialize={
            t: d["outdoor_temperature"][min(t, len(d["outdoor_temperature"]) - 1)]
            for t in range(state["current_time"], state["current_time"] + horizon)
        },
    )

    # Node-dependent parameters
    price_by_node = {nid: node["price"] for nid, node in nodes_map.items()}
    occ1_by_node = {nid: node["occupancy1"] for nid, node in nodes_map.items()}
    occ2_by_node = {nid: node["occupancy2"] for nid, node in nodes_map.items()}

    m.prices = Param(m.N, initialize=price_by_node)
    m.O1 = Param(m.N, initialize=occ1_by_node)
    m.O2 = Param(m.N, initialize=occ2_by_node)

    # Decision variables for the root (first-stage)
    m.Heat0 = Var(m.R, domain=NonNegativeReals, bounds=(0, d["heating_max_power"]))
    m.Vent0 = Var(domain=Binary)
    m.Vstart0 = Var(domain=Binary)
    m.y_low0 = Var(m.R, domain=Binary)
    m.y_ok0 = Var(m.R, domain=Binary)
    m.y_high0 = Var(m.R, domain=Binary)
    m.u0 = Var(m.R, domain=Binary)
    m.T_in0 = Var(m.R, domain=NonNegativeReals)
    m.Hum0 = Var(domain=NonNegativeReals)

    # Decision variables for non-root nodes
    m.Heat = Var(m.RN, domain=NonNegativeReals, bounds=(0, d["heating_max_power"]))
    m.Vent = Var(m.N, domain=Binary)
    m.Vstart = Var(m.N, domain=Binary)
    m.y_low = Var(m.RN, domain=Binary)
    m.y_ok = Var(m.RN, domain=Binary)
    m.y_high = Var(m.RN, domain=Binary)
    m.u = Var(m.RN, domain=Binary)

    # State variables for non-root nodes
    m.T_in = Var(m.RN, domain=NonNegativeReals)
    m.Hum = Var(m.N, domain=NonNegativeReals)

    # Map each node id to its stage index for use in rules
    node_stage = {}
    for stage_idx, stage_nodes in enumerate(tree):
        for node in stage_nodes:
            node_stage[node["id"]] = stage_idx

    # Root state equalities
    m.c_real = Constraint(expr=m.T_in0[1] == T_init[1])
    m.c_real2 = Constraint(expr=m.T_in0[2] == T_init[2])
    m.c_hum = Constraint(expr=m.Hum0 == H_init)

    # Big-M constraints for root comfort categories
    m.c_0thigh1 = Constraint(
        m.R, rule=lambda m, r: m.T_in0[r] >= eps + m.Thigh - m.M_temp * (1 - m.y_high0[r])
    )
    m.c_0thigh2 = Constraint(
        m.R, rule=lambda m, r: m.T_in0[r] <= m.Thigh + m.M_temp * m.y_high0[r]
    )
    m.c_0heat_off = Constraint(m.R, rule=lambda m, r: m.Heat0[r] <= m.Pr * (1 - m.y_high0[r]))

    m.c_0tlow1 = Constraint(
        m.R, rule=lambda m, r: m.T_in0[r] <= m.Tmin + m.M_temp * (1 - m.y_low0[r])
    )
    m.c_0tlow2 = Constraint(
        m.R, rule=lambda m, r: m.T_in0[r] >= m.Tmin + eps - m.M_temp * m.y_low0[r]
    )

    m.c_0tok1 = Constraint(
        m.R, rule=lambda m, r: m.T_in0[r] >= m.Tok - m.M_temp * (1 - m.y_ok0[r])
    )
    m.c_0tok2 = Constraint(m.R, rule=lambda m, r: m.T_in0[r] <= m.Tok + m.M_temp * m.y_ok0[r])

    # Overrule-memory (u0) constraints at root
    m.c_0u1 = Constraint(m.R, rule=lambda m, r: m.u0[r] >= m.y_low0[r])

    def u_memory_rule0(m, r):
        return m.u0[r] <= low_override[r] + m.y_low0[r]

    m.c_0u2 = Constraint(m.R, rule=u_memory_rule0)
    m.c_0u3 = Constraint(m.R, rule=lambda m, r: m.Heat0[r] >= m.Pr * m.u0[r])

    def u_memory_rule02(m, r):
        return m.u0[r] >= low_override[r] - m.y_ok0[r]

    m.c_0u4 = Constraint(m.R, rule=u_memory_rule02)
    m.c_0u5 = Constraint(m.R, rule=lambda m, r: m.u0[r] <= 1 - m.y_ok0[r])

    # Minimum uptime link for root ventilation
    m.min_uptime_root = Constraint(expr=m.Vstart0 >= m.Vent0 - v_prev)
    m.min_uptime_root_2 = Constraint(expr=m.Vstart0 <= m.Vent0)
    m.min_uptime_root_3 = Constraint(expr=m.Vstart0 <= 1 - v_prev)

    # Build chains that enforce minimum up-time across descendant nodes
    root_uptime_depth = min(value(m.U_vent), horizon)
    m.RootChains = []
    if root_uptime_depth == 1:
        m.RootChains = [[root_id]]
    elif len(tree) > 1:
        for child in tree[1]:
            child_chains = get_descendant_chains(child["id"], root_uptime_depth - 1, nodes_map)
            for chain in child_chains:
                m.RootChains.append([root_id] + chain)

    if m.RootChains:
        m.RootChainSet = Set(initialize=list(range(len(m.RootChains))))

        def min_uptime_root_rule(m, chain_idx):
            chain = m.RootChains[chain_idx]
            return m.Vent0 + sum(m.Vent[k] for k in chain[1:]) >= len(chain) * m.Vstart0

        m.MinVentOnRoot = Constraint(m.RootChainSet, rule=min_uptime_root_rule)

    # Humidity forcing constraint at root
    m.c_hum_limit0 = Constraint(rule=lambda m: m.Hum0 <= m.Hhigh + m.M_hum * m.Vent0)

    # Thermal dynamics: compute child node temperature from parent state and decisions
    def thermal_dynamics_rule(m, r, n):
        p_id = nodes_map[n]["parent_id"]
        tau_n = node_stage[n]
        r_other = 2 if r == 1 else 1
        t = state["current_time"] + node_stage[p_id]

        if tau_n == 1:
            occ_term = occ1_root if r == 1 else occ2_root
            T_parent = T_init[r]
            T_other_parent = T_init[r_other]
            heat_parent = m.Heat0[r]
            vent_parent = m.Vent0
        else:
            occ_term = m.O1[p_id] if r == 1 else m.O2[p_id]
            T_parent = m.T_in[r, p_id]
            T_other_parent = m.T_in[r_other, p_id]
            heat_parent = m.Heat[r, p_id]
            vent_parent = m.Vent[p_id]

        return m.T_in[r, n] == (
            T_parent
            + m.Zexch * (T_other_parent - T_parent)
            + m.Zloss * (m.Tout[t] - T_parent)
            + m.Zconv * heat_parent
            - m.Zcool * vent_parent
            + m.Zocc * occ_term
        )

    m.Temp_Dynamics = Constraint(m.R, m.N, rule=thermal_dynamics_rule)

    # Humidity dynamics: child humidity as function of parent humidity and ventilation
    def humidity_dynamics_rule(m, n):
        p_id = nodes_map[n]["parent_id"]
        tau_n = node_stage[n]
        if tau_n == 1:
            occ_term = occ1_root + occ2_root
            H_parent = H_init
            vent_parent = m.Vent0
        else:
            occ_term = m.O1[p_id] + m.O2[p_id]
            H_parent = m.Hum[p_id]
            vent_parent = m.Vent[p_id]

        return m.Hum[n] == (H_parent - m.Hvent * vent_parent + m.Hocc * occ_term)

    m.Hum_Dynamics = Constraint(m.N, rule=humidity_dynamics_rule)

    # Big-M logic for comfort categories and overrule memory (applies to all nodes)
    m.c_thigh1 = Constraint(
        m.RN, rule=lambda m, r, n: m.T_in[r, n] >= eps + m.Thigh - m.M_temp * (1 - m.y_high[r, n])
    )
    m.c_thigh2 = Constraint(m.RN, rule=lambda m, r, n: m.T_in[r, n] <= m.Thigh + m.M_temp * m.y_high[r, n])
    m.c_heat_off = Constraint(m.RN, rule=lambda m, r, n: m.Heat[r, n] <= m.Pr * (1 - m.y_high[r, n]))

    m.c_tlow1 = Constraint(m.RN, rule=lambda m, r, n: m.T_in[r, n] <= m.Tmin + m.M_temp * (1 - m.y_low[r, n]))
    m.c_tlow2 = Constraint(
        m.RN, rule=lambda m, r, n: m.T_in[r, n] >= m.Tmin + eps - m.M_temp * m.y_low[r, n]
    )

    m.c_tok1 = Constraint(m.RN, rule=lambda m, r, n: m.T_in[r, n] >= m.Tok - m.M_temp * (1 - m.y_ok[r, n]))
    m.c_tok2 = Constraint(m.RN, rule=lambda m, r, n: m.T_in[r, n] <= m.Tok + m.M_temp * m.y_ok[r, n])

    # Overrule-memory (u variable) propagation across the tree
    def u_memory_rule(m, r, n):
        tau_n = node_stage[n]
        if tau_n == 1:
            return m.u[r, n] <= m.u0[r] + m.y_low[r, n]
        p_id = nodes_map[n]["parent_id"]
        return m.u[r, n] <= m.u[r, p_id] + m.y_low[r, n]

    m.c_u1 = Constraint(m.RN, rule=lambda m, r, n: m.u[r, n] >= m.y_low[r, n])
    m.c_u2 = Constraint(m.RN, rule=u_memory_rule)
    m.c_u3 = Constraint(m.RN, rule=lambda m, r, n: m.Heat[r, n] >= m.Pr * m.u[r, n])

    def u_memory_rule2(m, r, n):
        tau_n = node_stage[n]
        if tau_n == 1:
            return m.u[r, n] >= m.u0[r] - m.y_ok[r, n]
        p_id = nodes_map[n]["parent_id"]
        return m.u[r, n] >= m.u[r, p_id] - m.y_ok[r, n]

    m.c_u4 = Constraint(m.RN, rule=u_memory_rule2)
    m.c_u5 = Constraint(m.RN, rule=lambda m, r, n: m.u[r, n] <= 1 - m.y_ok[r, n])

    # Ventilation-start logic and minimum uptime chains for non-root nodes
    def vent_start_rule(m, n):
        tau_n = node_stage[n]
        if tau_n == 1:
            return m.Vstart[n] >= m.Vent[n] - m.Vent0
        p_id = nodes_map[n]["parent_id"]
        return m.Vstart[n] >= m.Vent[n] - m.Vent[p_id]

    m.c_vstart = Constraint(m.N, rule=vent_start_rule)

    m.c_vstart2 = Constraint(m.N, rule=lambda m, n: m.Vstart[n] <= m.Vent[n])

    def vent_start_rule_3(m, n):
        tau_n = node_stage[n]
        if tau_n == 1:
            return m.Vstart[n] <= 1 - m.Vent0
        p_id = nodes_map[n]["parent_id"]
        return m.Vstart[n] <= 1 - m.Vent[p_id]

    m.c_vstart3 = Constraint(m.N, rule=vent_start_rule_3)

    # Build descendant chains for minimum up-time enforcement for every decision node
    m.Chains = {}
    for nid in decision_ids:
        m.Chains[nid] = get_descendant_chains(nid, value(m.U_vent), nodes_map)

    def min_uptime_rule(m, nid, chain_idx):
        chain = m.Chains[nid][chain_idx]
        return sum(m.Vent[k] for k in chain) >= len(chain) * m.Vstart[nid]

    m.NodeChainSet = Set(initialize=[(nid, i) for nid in decision_ids for i in range(len(m.Chains[nid]))])
    m.MinVentOn = Constraint(m.NodeChainSet, rule=min_uptime_rule)

    # Humidity forcing constraint for non-root nodes
    m.c_hum_limit = Constraint(m.N, rule=lambda m, n: m.Hum[n] <= m.Hhigh + m.M_hum * m.Vent[n])

    # Objective: minimize expected cost (energy + ventilation cost) across nodes
    def obj_rule(m):
        expected_cost = sum(
            nodes_map[n]["probability"]
            * (
                m.prices[n] * sum(m.Heat[r, n] for r in m.R)
                + m.prices[n] * m.Pvent * m.Vent[n]
            )
            for n in m.N
        )

        # Add first-stage (root) cost
        first_stage_cost = state["price_t"] * (m.Heat0[1] + m.Heat0[2] + m.Pvent * m.Vent0)
        return expected_cost + first_stage_cost

    m.obj = Objective(rule=obj_rule, sense=minimize)

    return m


def multiSP_policy(state):
    """Top-level policy: build tree, create model, solve, and return actions.

    Returns a dict with keys: 'HeatPowerRoom1', 'HeatPowerRoom2', 'VentilationON'.
    In case the solver fails, returns a safe fallback of zero heating and no
    ventilation.
    """

    t = state["current_time"]
    remaining = DATA["num_timeslots"] - t
    horizon = min(HORIZON_MULTI, remaining)

    tree = build_scenario_tree(state, horizon, S=BRANCHING_FACTOR, K=N_CLUSTERS)
    #t1 = time.time()
    ##print(f"Scenario tree built in {t1 - t0:.2f} seconds.", flush=True)
    model = build_multisp_model(state, tree, horizon)

    solver = SolverFactory("gurobi")
    result = solver.solve(model, tee=False)

    # If the solver failed or the result is not feasible/optimal, warn and use fallback
    if result.solver.termination_condition not in (
        TerminationCondition.optimal,
        TerminationCondition.feasible,
    ):
        warnings.warn(
            f"Gurobi failed at time {state['current_time']} with termination condition "
            f"{result.solver.termination_condition}. Falling back to zero action.",
            RuntimeWarning,
        )
        return {"HeatPowerRoom1": 0.0, "HeatPowerRoom2": 0.0, "VentilationON": 0}

    # Extract first-stage decisions
    p1 = value(model.Heat0[1])
    p2 = value(model.Heat0[2])
    v = value(model.Vent0)
    return {"HeatPowerRoom1": p1, "HeatPowerRoom2": p2, "VentilationON": v}


def select_action(state):
    return multiSP_policy(state)

