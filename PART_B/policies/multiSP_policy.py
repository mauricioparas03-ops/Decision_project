"""
multiSP_policy.py
=================
Multi-stage stochastic programming policy for the restaurant HVAC system.

Compatible with Environment.py:
    from policies.multiSP_policy import select_action

Design choices
--------------
  Lookahead horizon : HORIZON_MULTI = 3 steps  (minimum 3 due to vent inertia)
  Raw MC scenarios  : GEN_SCENARIOS = 100
  Branches per node : N_BRANCHES    = 3
  Non-anticipativity: structural (one variable per node, no explicit NAC needed)
  Solver            : Gurobi, TimeLimit = 12 s, MIPGap = 2 %
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from pyomo.environ import (
    ConcreteModel, Set, Param, Var, Objective, Constraint,
    SolverFactory, NonNegativeReals, minimize, Binary, value,
)
from pyomo.opt import TerminationCondition

# ── Hyper-parameters ───────────────────────────────────────────────────────────
HORIZON_MULTI = 3    # lookahead steps (must be >= 3 due to vent-inertia)
GEN_SCENARIOS = 500  # raw Monte Carlo paths before tree clustering
N_BRANCHES    = 3    # branches per node in the scenario tree / not currently being used, replaced by BRANCHING_FACTORS
BRANCHING_FACTORS = [10,4,1] 

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
    """One-step-ahead electricity price sample (AR(2)-like with mean reversion)."""
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
    """One-step-ahead occupancy sample for both rooms (coupled mean-reverting)."""
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
    Draw n_scenarios independent Monte-Carlo paths over horizon steps.

    Returns
    -------
    price_dict   : {(t, s): float}
    occ_dict     : {(r, t, s): float}
    hum_occ_dict : {(t, s): float}
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



def generate_tree_scenarios(price_now, price_prev, occ_r1_now, occ_r2_now, 
                            branching_factors, rng):
    """
    Generates a tree where branching_factors[t] defines how many 
    children each node at time t produces.
    """
    horizon = len(branching_factors)
    
    # Pre-calculate all scenario paths (indices)
    import itertools
    ranges = [range(b) for b in branching_factors]
    scenario_indices = list(itertools.product(*ranges)) # e.g., (0,0,0), (0,0,1)...
    
    price_dict = {}
    occ_dict = {}
    hum_occ_dict = {}

    # This dictionary will cache the values for each node in the tree
    # Key: (time_step, path_prefix) -> Value: (price, occ1, occ2)
    tree_nodes = {}
    
    # Root state (Stage -1)
    tree_nodes[(-1, ())] = (price_now, price_prev, occ_r1_now, occ_r2_now)

    for t in range(horizon):
        # We only need to generate values for unique prefixes at this time step
        unique_prefixes = set(idx[:t] for idx in scenario_indices)
        
        for prefix in unique_prefixes:
            # Get parent values
            p_cur, p_prev, o1_cur, o2_cur = tree_nodes[(t-1, prefix)]
            
            # Generate N children for this specific parent
            num_children = branching_factors[t]
            for branch_idx in range(num_children):
                current_prefix = prefix + (branch_idx,)
                
                # Draw from your models
                p_next = price_model(p_cur, p_prev, rng)
                o1_next, o2_next = next_occupancy_levels(o1_cur, o2_cur, rng)
                
                # Store node value
                tree_nodes[(t, current_prefix)] = (p_next, p_cur, o1_next, o2_next)

    # Finally, map the tree nodes back to the (t, s) format your model expects
    for s_int, path in enumerate(scenario_indices):
        for t in range(horizon):
            node_val = tree_nodes[(t, path[:t+1])]
            price_dict[t, s_int] = node_val[0]
            occ_dict[1, t, s_int] = node_val[2]
            occ_dict[2, t, s_int] = node_val[3]
            hum_occ_dict[t, s_int] = node_val[2] + node_val[3]

    return price_dict, occ_dict, hum_occ_dict


# =============================================================================
# 4. SCENARIO TREE CONSTRUCTION  (recursive conditional K-Means)
# =============================================================================

def cluster_scenarios_tree(price_dict, occ_dict, n_branches, horizon,
                           scenarios_to_generate):
    """
    Build a scenario tree by recursive conditional K-Means clustering.

    At each stage t, clusters the t-th observation within each group of
    scenarios that shared the same node at t-1.

    Returns
    -------
    tree : dict  node_id -> {
        't'        : int,
        'parent'   : int or None,
        'children' : list[int],
        'scenarios': list[int],
        'centroid' : dict  {'price', 'occ1', 'occ2', 'hum_occ'} or None (root),
        'cond_prob': float,
        'path_prob': float,
    }
    """
    tree = {
        0: {
            't'        : 0,
            'parent'   : None,
            'children' : [],
            'scenarios': list(range(scenarios_to_generate)),
            'centroid' : None,
            'cond_prob': 1.0,
            'path_prob': 1.0,
        }
    }

    node_counter           = 1
    nodes_at_current_stage = [0]

    for t in range(horizon):
        nodes_at_next_stage = []

        for parent_id in nodes_at_current_stage:
            parent_node      = tree[parent_id]
            member_scenarios = parent_node['scenarios']
            n_parent         = len(member_scenarios)
            k                = min(n_branches, n_parent)

            X_t = np.array([
                [price_dict[t, s], occ_dict[1, t, s], occ_dict[2, t, s]]
                for s in member_scenarios
            ])

            scaler     = StandardScaler()
            X_t_scaled = scaler.fit_transform(X_t)

            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            km.fit(X_t_scaled)

            centroids_raw = scaler.inverse_transform(km.cluster_centers_)

            for cluster_label in range(k):
                child_scenarios = [
                    s for s, lbl in zip(member_scenarios, km.labels_)
                    if lbl == cluster_label
                ]

                cond_prob = len(child_scenarios) / n_parent
                path_prob = parent_node['path_prob'] * cond_prob

                c        = centroids_raw[cluster_label]
                centroid = {
                    'price'  : float(np.clip(c[0],  0, 12)),
                    'occ1'   : float(np.clip(c[1], 20, 50)),
                    'occ2'   : float(np.clip(c[2], 10, 30)),
                }
                centroid['hum_occ'] = centroid['occ1'] + centroid['occ2']

                child_id          = node_counter
                tree[child_id]    = {
                    't'        : t + 1,
                    'parent'   : parent_id,
                    'children' : [],
                    'scenarios': child_scenarios,
                    'centroid' : centroid,
                    'cond_prob': cond_prob,
                    'path_prob': path_prob,
                }
                tree[parent_id]['children'].append(child_id)
                nodes_at_next_stage.append(child_id)
                node_counter += 1

        nodes_at_current_stage = nodes_at_next_stage

    return tree


def cluster_scenarios_tree2(price_dict, occ_dict, branching_factors, horizon):
    """
    Builds a scenario tree by recursive conditional K-Means clustering.
    Supports variable branching factors (non-symmetric trees).
    """

    # --- 1. DYNAMIC SCENARIO EXTRACTION (Fixes the KeyError) ---
    # Extract unique scenario indices directly from the keys of price_dict
    all_scenarios = sorted(list(set(s for (t, s) in price_dict.keys())))

    # --- Root node: contains all scenarios, no centroid needed ---
    tree = {}
    tree[0] = {
        't'         : 0,
        'parent'    : None,
        'children'  : [],
        'scenarios' : all_scenarios,
        'centroid'  : None,
        'cond_prob' : 1.0,
        'path_prob' : 1.0,
    }

    node_counter = 1
    nodes_at_current_stage = [0]

    for t in range(horizon):
        nodes_at_next_stage = []
        
        # --- 2. APPLY VARIABLE BRANCHING ---
        # Get the specific number of branches for this stage t
        if isinstance(branching_factors, (list, tuple)):
            # If the horizon is longer than the provided list, default to 1 branch (deterministic tail)
            current_n_branches = branching_factors[t] if t < len(branching_factors) else 1
        else:
            current_n_branches = branching_factors # Fallback if passed an integer

        for parent_id in nodes_at_current_stage:

            parent_node      = tree[parent_id]
            member_scenarios = parent_node['scenarios']
            n_parent         = len(member_scenarios)

            # Gracefully handle nodes with fewer scenarios than current_n_branches
            k = min(current_n_branches, n_parent)

            # --- Feature matrix: only the t-th observation ---
            X_t = np.array([
                [price_dict[t, s],
                 occ_dict[1, t, s],
                 occ_dict[2, t, s]]
                for s in member_scenarios
            ])

            scaler = StandardScaler()
            X_t_scaled = scaler.fit_transform(X_t)

            km = KMeans(n_clusters=k, n_init=20, random_state=42)
            km.fit(X_t_scaled)

            centroids_raw = scaler.inverse_transform(km.cluster_centers_)

            # --- Create one child node per cluster ---
            for cluster_label in range(k):

                child_scenarios = [
                    s for s, lbl in zip(member_scenarios, km.labels_)
                    if lbl == cluster_label
                ]

                cond_prob = len(child_scenarios) / n_parent
                path_prob = parent_node['path_prob'] * cond_prob

                c = centroids_raw[cluster_label]
                centroid = {
                    'price' : float(np.clip(c[0],  0, 12)),
                    'occ1'  : float(np.clip(c[1], 20, 50)),
                    'occ2'  : float(np.clip(c[2], 10, 30)),
                }
                centroid['hum_occ'] = centroid['occ1'] + centroid['occ2']

                child_id = node_counter
                tree[child_id] = {
                    't'         : t + 1,
                    'parent'    : parent_id,
                    'children'  : [],
                    'scenarios' : child_scenarios,
                    'centroid'  : centroid,
                    'cond_prob' : cond_prob,
                    'path_prob' : path_prob,
                }
                tree[parent_id]['children'].append(child_id)
                nodes_at_next_stage.append(child_id)
                node_counter += 1

        nodes_at_current_stage = nodes_at_next_stage

    return tree


# =============================================================================
# 5. TREE TRAVERSAL HELPERS
# =============================================================================

def _path_to_root(tree, leaf_id):
    """
    Return list of internal node ids from leaf back to (but not including) root.
    Covers all decision nodes on this path, including stage-1 nodes.
    """
    path = []
    nid  = tree[leaf_id]['parent']
    while nid is not None and tree[nid]['parent'] is not None:
        path.append(nid)
        nid = tree[nid]['parent']
    # Include stage-1 node (its parent is root which has parent=None)
    if nid is not None and tree[nid]['parent'] is None and nid != 0:
        path.append(nid)
    return path


def _descendants_chain(tree, nid, L):
    """
    Return up to L successive nodes starting from nid, following first child.
    Used for the minimum-uptime window.
    """
    chain   = [nid]
    current = nid
    for _ in range(L - 1):
        children = tree[current]['children']
        if not children:
            break
        current = children[0]
        chain.append(current)
    return chain


# =============================================================================
# 6. PYOMO MILP MODEL  (multi-stage SP, node-indexed)
# =============================================================================

def build_multisp_model(current_state, tree, horizon):
    """
    Build the multi-stage stochastic MILP indexed on scenario-tree nodes.

    Parameters
    ----------
    current_state : dict – keys: T_in_r1, T_in_r2, humidity,
                                 vent_prev, vent_on_count,
                                 low_override_r1, low_override_r2
    tree          : dict  node_id -> {t, parent, children, centroid, path_prob}
    horizon       : int

    Returns
    -------
    Pyomo ConcreteModel (unsolved)
    """
    d = DATA
    M = 200.0

    T_init = {1: current_state['T_in_r1'], 2: current_state['T_in_r2']}
    H_init = current_state['humidity']
    v_prev = int(current_state['vent_prev'])
    v_on_h = int(current_state.get('vent_on_count', 0))

    # ── Node sets ─────────────────────────────────────────────────────────────
    all_nodes      = list(tree.keys())
    internal_nodes = [nid for nid, n in tree.items() if n['children'] and nid != 0]
    leaf_nodes     = [nid for nid, n in tree.items() if not n['children'] and nid != 0]
    internal_set   = set(internal_nodes)
    leaf_set       = set(leaf_nodes)

    # ── Exogenous data dicts (keyed by internal node id) ──────────────────────
    price_by_node  = {nid: tree[nid]['centroid']['price']   for nid in internal_nodes}
    occ1_by_node   = {nid: tree[nid]['centroid']['occ1']    for nid in internal_nodes}
    occ2_by_node   = {nid: tree[nid]['centroid']['occ2']    for nid in internal_nodes}
    humocc_by_node = {nid: tree[nid]['centroid']['hum_occ'] for nid in internal_nodes}
    path_prob_leaf = {nid: tree[nid]['path_prob']           for nid in leaf_nodes}

    m = ConcreteModel()

    # ── Sets ──────────────────────────────────────────────────────────────────
    m.R      = Set(initialize=[1, 2])
    m.N      = Set(initialize=all_nodes)
    m.N_int  = Set(initialize=internal_nodes)
    m.N_leaf = Set(initialize=leaf_nodes)
    m.RN_int = m.R * m.N_int

    # ── Physical parameters ───────────────────────────────────────────────────
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
    m.Tout  = Param(range(horizon),
                    initialize={t: d['outdoor_temperature'][t] for t in range(horizon)})

    m.prices = Param(m.N_int, initialize=price_by_node)
    m.HumOcc = Param(m.N_int, initialize=humocc_by_node)
    m.O1     = Param(m.N_int, initialize=occ1_by_node)
    m.O2     = Param(m.N_int, initialize=occ2_by_node)
    m.pi     = Param(m.N_leaf, initialize=path_prob_leaf)

    # ── Decision variables (internal nodes only) ──────────────────────────────
    m.Heat = Var(m.RN_int, domain=NonNegativeReals, bounds=(0, d['heating_max_power']))
    m.Vent = Var(m.N_int,  domain=Binary)
    m.Uon  = Var(m.N_int,  domain=Binary)
    m.Uoff = Var(m.N_int,  domain=Binary)

    m.u = Var(m.RN_int, domain=Binary)   # 1 → low-temp overrule active
    m.w = Var(m.RN_int, domain=Binary)   # 1 → temperature recovered to T_OK
    m.y = Var(m.RN_int, domain=Binary)   # 1 → high-temp overrule active
    m.y_low = Var(m.RN_int, domain=Binary) # 1 -> T is below T_low
    # ── State variables (all nodes) ───────────────────────────────────────────
    m.T_in = Var(m.R * m.N, domain=NonNegativeReals)
    m.Hum  = Var(m.N,        domain=NonNegativeReals)

    # ── Helper: occupancy lookup ──────────────────────────────────────────────
    def occ(r, nid):
        return m.O1[nid] if r == 1 else m.O2[nid]

    # ── Temperature dynamics ──────────────────────────────────────────────────
    def temp_dynamics(m, r, nid):
        if nid == 0:
            return m.T_in[r, nid] == T_init[r]

        pid     = tree[nid]['parent']
        other_r = 2 if r == 1 else 1

        if pid == 0:
            # State flows from root; decisions made AT nid (root has no vars)
            t_root = tree[0]['t']   # = 0
            if nid in leaf_set:
                # Horizon=1 edge case: leaf whose parent is root, no decision
                return m.T_in[r, nid] == (
                    m.T_in[r, 0]
                    + m.Zexch * (m.T_in[other_r, 0] - m.T_in[r, 0])
                    + m.Zloss * (m.Tout[t_root]      - m.T_in[r, 0])
                )
            return m.T_in[r, nid] == (
                m.T_in[r, 0]
                + m.Zexch * (m.T_in[other_r, 0] - m.T_in[r, 0])
                + m.Zloss * (m.Tout[t_root]      - m.T_in[r, 0])
                + m.Zconv * m.Heat[r, nid]
                - m.Zcool * m.Vent[nid]
                + m.Zocc  * occ(r, nid)
            )

        # General case: pid is a non-root internal node
        t_parent = tree[pid]['t']
        if nid in leaf_set:
            return m.T_in[r, nid] == (
                m.T_in[r, pid]
                + m.Zexch * (m.T_in[other_r, pid] - m.T_in[r, pid])
                + m.Zloss * (m.Tout[t_parent]      - m.T_in[r, pid])
                + m.Zconv * m.Heat[r, pid]
                - m.Zcool * m.Vent[pid]
                + m.Zocc  * occ(r, pid)
            )
        return m.T_in[r, nid] == (
            m.T_in[r, pid]
            + m.Zexch * (m.T_in[other_r, pid] - m.T_in[r, pid])
            + m.Zloss * (m.Tout[t_parent]      - m.T_in[r, pid])
            + m.Zconv * m.Heat[r, pid]
            - m.Zcool * m.Vent[pid]
            + m.Zocc  * occ(r, pid)
        )
    m.TempDyn = Constraint(m.R * m.N, rule=temp_dynamics)

    # ── Humidity dynamics ─────────────────────────────────────────────────────
    def hum_dynamics(m, nid):
        if nid == 0:
            return m.Hum[nid] == H_init
        pid = tree[nid]['parent']
        if pid == 0:
            if nid in leaf_set:
                return m.Hum[nid] == m.Hum[0]
            return m.Hum[nid] == (
                m.Hum[0]
                - m.Hvent * m.Vent[nid]
                + m.Hocc  * m.HumOcc[nid]
            )
        return m.Hum[nid] == (
            m.Hum[pid]
            - m.Hvent * m.Vent[pid]
            + m.Hocc  * m.HumOcc[pid]
        )
    m.HumDyn = Constraint(m.N, rule=hum_dynamics)

    # ── Overrule controller: LOW temperature ──────────────────────────────────
    #EQN 8&9
    # 1. Strict Detection of Low Temperature (y_low) 
    def y_low_lower(m, r, nid):
        # Forces y_low=1 when T < Tmin
        return m.T_in[r, nid] >= m.Tmin - M * m.y_low[r, nid]
    m.YLowLower = Constraint(m.RN_int, rule=y_low_lower)

    def y_low_upper(m, r, nid):
        # Forces y_low=0 when T >= Tmin
        return m.T_in[r, nid] <= m.Tmin + M * (1 - m.y_low[r, nid])
    m.YLowUpper = Constraint(m.RN_int, rule=y_low_upper)
    # def u_activation(m, r, nid):
    #   return m.T_in[r, nid] >= m.Tmin - M * m.u[r, nid]
    # m.UActivation = Constraint(m.RN_int, rule=u_activation)

    #EQN 10&11
    # 2. Strict Detection of Recovery Temperature (w)
    def w_deactivation_lower(m, r, nid):
        # Forces w=1 when T >= Tok
        return m.T_in[r, nid] >= m.Tok - M * (1 - m.w[r, nid])
    m.WDeactivationLower = Constraint(m.RN_int, rule=w_deactivation_lower)

    def w_deactivation_upper(m, r, nid):
        # Forces w=0 when T < Tok
        return m.T_in[r, nid] <= m.Tok + M * m.w[r, nid]
    m.WDeactivationUpper = Constraint(m.RN_int, rule=w_deactivation_upper)
    # def w_deactivation(m, r, nid):
    #     return m.T_in[r, nid] >= m.Tok - M * (1 - m.w[r, nid])
    # m.WDeactivation = Constraint(m.RN_int, rule=w_deactivation)

    def u_persistence(m, r, nid):
        pid = tree[nid]['parent']
        if pid is None or pid == 0:
            return Constraint.Skip
        return m.u[r, nid] >= m.u[r, pid] - m.w[r, nid]
    m.UPersistence = Constraint(m.RN_int, rule=u_persistence)

    def heat_max_when_overrule(m, r, nid):
        return m.Heat[r, nid] >= m.Pr * m.u[r, nid]
    m.HeatMaxOverrule = Constraint(m.RN_int, rule=heat_max_when_overrule)

    # ── Overrule controller: HIGH temperature ─────────────────────────────────
    def y_activation_lower(m, r, nid):
        # Forces y=1 when T > Thigh
        return m.T_in[r, nid] <= m.Thigh + M * m.y[r, nid]
    m.YActivationLower = Constraint(m.RN_int, rule=y_activation_lower)

    def y_activation_upper(m, r, nid):
        # Forces y=0 when T <= Thigh
        return m.T_in[r, nid] >= m.Thigh - M * (1 - m.y[r, nid])
    m.YActivationUpper = Constraint(m.RN_int, rule=y_activation_upper)
    # def y_activation(m, r, nid):
    #     return m.T_in[r, nid] <= m.Thigh + M * m.y[r, nid]
    # m.YActivation = Constraint(m.RN_int, rule=y_activation)

    def heat_off_when_overrule(m, r, nid):
        return m.Heat[r, nid] <= m.Pr * (1 - m.y[r, nid])
    m.HeatOffOverrule = Constraint(m.RN_int, rule=heat_off_when_overrule)

    # ── Humidity overrule: force ventilation ON when humid ────────────────────
    def vent_humidity_overrule(m, nid):
        return m.Hum[nid] <= m.Hhigh + M * m.Vent[nid]
    m.VentHumOverrule = Constraint(m.N_int, rule=vent_humidity_overrule)

    # __ State update for overrule controller _________________________________
    #EQN 12,13,15,16
    def temp_lower_than_tmin_1(m, r, nid):
        # Must turn ON if it gets too cold
        return m.u[r, nid] >= m.y_low[r, nid]
    m.UStateLogic1 = Constraint(m.RN_int, rule=temp_lower_than_tmin_1)

    def temp_higher_than_tok(m, r, nid):
        pid = tree[nid]['parent']
        # Cannot turn ON unless it is too cold
        if pid is None or pid == 0:
            return Constraint.Skip
        return m.u[r, nid] <= m.u[r, pid] + m.y_low[r, nid]
    m.UStateLogic3 = Constraint(m.RN_int, rule=temp_higher_than_tok)

    def temp_higher_than_tok_2(m, r, nid):
        # Must turn OFF if it reaches Tok
        return m.u[r, nid] <= 1 - m.w[r, nid]
    m.UStateLogic4 = Constraint(m.RN_int, rule=temp_higher_than_tok_2)

    def overrule_init(m, r, nid):
        # At t=0, fix u to match the real system's overrule state
        if nid == 0:
            init_u = 1 if current_state.get(f'low_override_r{r}', 0) else 0
            return m.u[r, nid] == init_u
        return Constraint.Skip
    m.OverruleInit = Constraint(m.RN_int, rule=overrule_init)

    # ── Ventilation inertia: 3-hour minimum ON time ───────────────────────────
    def on_off_exclusivity(m, nid):
        return m.Uon[nid] + m.Uoff[nid] <= 1
    m.OnOffExcl = Constraint(m.N_int, rule=on_off_exclusivity)

    def uoff_bound(m, nid):
        return m.Uoff[nid] <= 1 - m.Vent[nid]
    m.UoffBound = Constraint(m.N_int, rule=uoff_bound)

    def uon_bound(m, nid):
        return m.Uon[nid] <= 1 - m.Vent[nid]
    m.UonBound = Constraint(m.N_int, rule=uon_bound)

    def min_uptime(m, nid):
        L     = d['vent_min_up_time']
        chain = [k for k in _descendants_chain(tree, nid, L) if k in internal_set]
        pid   = tree[nid]['parent']
        v_p   = v_prev if (pid is None or pid == 0) else m.Vent[pid]
        return sum(m.Vent[k] for k in chain) >= len(chain) * (m.Vent[nid] - v_p)
    m.MinUptime = Constraint(m.N_int, rule=min_uptime)

    # Carry-over inertia from previous real hours
    if v_prev == 1 and v_on_h < 3:
        forced_on = 3 - v_on_h
        for nid in internal_nodes:
            if tree[nid]['t'] <= forced_on:
                m.Vent[nid].fix(1)

    # ── Objective: E[cost along each leaf path] ───────────────────────────────
    def objective(m):
        return sum(
            m.pi[leaf] * sum(
                m.prices[nid] * (
                    sum(m.Heat[r, nid] for r in m.R)
                    + m.Vent[nid] * m.Pvent
                )
                for nid in _path_to_root(tree, leaf)
            )
            for leaf in leaf_nodes
        )
    m.obj = Objective(rule=objective, sense=minimize)
    return m


# =============================================================================
# 7. MULTI-STAGE SP POLICY FUNCTION
# =============================================================================

def multi_SP_policy(state):
    """
    Multi-stage stochastic programming policy.

    Reads the current state dict (Environment.py format), builds a scenario
    tree, solves the node-indexed MILP, and returns the first-stage decision.

    Parameters
    ----------
    state : dict  – keys as used by Environment.py:
        T1, T2, H, Occ1, Occ2, price_t, price_previous,
        vent_counter, low_override_r1, low_override_r2, current_time

    Returns
    -------
    dict with keys 'HeatPowerRoom1', 'HeatPowerRoom2', 'VentilationON'
    """
    t         = state['current_time']
    remaining = DATA['num_timeslots'] - t
    horizon   = min(HORIZON_MULTI, remaining)

    if horizon <= 0:
        return {'HeatPowerRoom1': 0.0, 'HeatPowerRoom2': 0.0, 'VentilationON': 0}

    # Reproducible per-timestep RNG
    rng    = np.random.default_rng(seed=42 + t)
    p_prev = state.get('price_previous') or 4.0

    # Ventilation status derived from counter
    vc       = state.get('vent_counter', 0)
    v_status = 1 if vc > 0 else 0

    # ── Generate raw Monte-Carlo scenarios ────────────────────────────────────

    # price_dict, occ_dict, _ = generate_scenarios(
    #     price_now   = state['price_t'],
    #     price_prev  = p_prev,
    #     occ_r1_now  = state['Occ1'],
    #     occ_r2_now  = state['Occ2'],
    #     horizon =  horizon,
    #     n_scenarios= GEN_SCENARIOS,
    #     rng         = rng,
    # )

    # # ── Build scenario tree ───────────────────────────────────────────────────
    # tree = cluster_scenarios_tree2(
    #     price_dict, occ_dict,
    #     n_branches = N_BRANCHES,
    #     horizon             = horizon,
    #     scenarios_to_generate = GEN_SCENARIOS
    # )

    price_dict, occ_dict, _ = generate_tree_scenarios(
        price_now   = state['price_t'],
        price_prev  = p_prev,
        occ_r1_now  = state['Occ1'],
        occ_r2_now  = state['Occ2'],
        branching_factors= BRANCHING_FACTORS,
        rng         = rng,
    )

    # ── Build scenario tree ───────────────────────────────────────────────────
    tree = cluster_scenarios_tree2(
        price_dict, occ_dict,
        branching_factors = BRANCHING_FACTORS,
        horizon             = horizon,
    )

    # ── Guard: need at least one internal non-root node for decisions ─────────
    internal_nodes = [nid for nid, n in tree.items() if n['children'] and nid != 0]
    if not internal_nodes:
        return {'HeatPowerRoom1': 0.0, 'HeatPowerRoom2': 0.0, 'VentilationON': 0}

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
    model  = build_multisp_model(milp_state, tree, horizon)
    solver = SolverFactory('gurobi_direct')
    solver.options['TimeLimit'] = 12
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

    # ── Extract first-stage decision ──────────────────────────────────────────
    # First-stage nodes are children of root. In multi-stage SP, NAC is
    # structural — each node has its own variable, so branches may differ.
    # We take the first child of root as the representative decision.
    stage1_nodes  = tree[0]['children']
    internal_set  = set(internal_nodes)
    decision_node = next(nid for nid in stage1_nodes if nid in internal_set)

    p1 = float(value(model.Heat[1, decision_node]))
    p2 = float(value(model.Heat[2, decision_node]))
    v  = int(round(float(value(model.Vent[decision_node]))))

    pr_max = DATA['heating_max_power']
    p1 = float(np.clip(p1, 0.0, pr_max))
    p2 = float(np.clip(p2, 0.0, pr_max))
    v  = int(np.clip(v,  0,   1))

    return {'HeatPowerRoom1': p1, 'HeatPowerRoom2': p2, 'VentilationON': v}


# =============================================================================
# 8. GRADER-COMPATIBLE WRAPPER
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
    return multi_SP_policy(state)
