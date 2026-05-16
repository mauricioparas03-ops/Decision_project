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
import warnings
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from pyomo.environ import (
    ConcreteModel, Set, Param, Var, Objective, Constraint,
    SolverFactory, NonNegativeReals, minimize, Binary, value,
)
from pyomo.opt import TerminationCondition

from Data.v2_SystemCharacteristics import get_fixed_data


# ── Hyper-parameters ───────────────────────────────────────────────────────────
HORIZON_MULTI = 4    # lookahead steps (must be >= 3 due to vent-inertia)
GEN_SCENARIOS = 300  # raw Monte Carlo paths before tree clustering
N_CLUSTERS    = 3 
BRANCHING_FACTORS = [5,4,3,3] 

# =============================================================================
# 1. SYSTEM PARAMETERS
# =============================================================================

DATA = get_fixed_data()
#print(DATA['outdoor_temperature'])

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


# def generate_scenarios(price_now, price_prev, occ_r1_now, occ_r2_now, horizon, n_scenarios):
#     price_dict = {}
#     occ_dict = {}

#     for s in range(n_scenarios):
#         p_cur, p_prev = price_now, price_prev
#         o1_cur, o2_cur = occ_r1_now, occ_r2_now

#         for t in range(horizon):
#             p_next = price_model(p_cur, p_prev)
#             o1_next, o2_next = next_occupancy_levels(o1_cur, o2_cur)

#             price_dict[t, s] = float(p_next)
#             occ_dict[1, t, s] = float(o1_next)
#             occ_dict[2, t, s] = float(o2_next)

#             p_prev, p_cur = p_cur, p_next
#             o1_cur, o2_cur = o1_next, o2_next

#     return price_dict, occ_dict




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
        unique_prefixes = sorted({idx[:t] for idx in scenario_indices})
        
        for prefix in unique_prefixes:
            # Get parent values
            p_cur, p_prev, o1_cur, o2_cur = tree_nodes[(t-1, prefix)]
            
            # Generate N children for this specific parent
            num_children = branching_factors[t]
            for branch_idx in range(num_children):
                current_prefix = prefix + (branch_idx,)
                
                # Draw from the models
                p_next = price_model(p_cur, p_prev, rng)
                o1_next, o2_next = next_occupancy_levels(o1_cur, o2_cur, rng)
                
                # Store node value
                tree_nodes[(t, current_prefix)] = (p_next, p_cur, o1_next, o2_next)

    # Finally, map the tree nodes back to the (t, s) format the model expects
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
            random_seed = 42 + t

            km = KMeans(n_clusters=k, n_init=10, random_state=random_seed)
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


def cluster_scenarios_tree2(price_dict, occ_dict, N_CLUSTERS, horizon, rng):
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
        current_n_branches = N_CLUSTERS
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

            rng_seed = 42 + t

            km = KMeans(n_clusters=k, n_init=20, random_state=rng_seed)
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
                    'price' : float(c[0]),
                    'occ1'  : float(c[1]),
                    'occ2'  : float(c[2]),
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
    Return list of decision node ids from leaf back to (and including) stage-1 nodes.
    Excludes root (node 0). Used to sum costs along a path in the scenario tree.
    """
    path = []
    nid  = leaf_id
    while nid is not None and nid != 0:   #nid != 0 
        path.append(nid)
        nid = tree[nid]['parent']
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
    current_state : dict – keys: T1, T2, H,
                                 vent_counter,
                                 low_override_r1, low_override_r2
    tree          : dict  node_id -> {t, parent, children, centroid, path_prob}
    horizon       : int

    Returns
    -------
    Pyomo ConcreteModel (unsolved)
    """
    d = DATA

    T_init = {1: current_state['T1'], 2: current_state['T2']}
    H_init = current_state['H']
    # v_prev = int(current_state['vent_prev'])
    # v_on_h = int(current_state.get('vent_on_count', 0))
    vent_counter = int(current_state['vent_counter'])
    v_prev       = 1 if vent_counter > 0 else 0
    v_on_h       = vent_counter  
    low_override = {
        1: int(current_state['low_override_r1']),
        2: int(current_state['low_override_r2']),
    }

    # ── Node sets ─────────────────────────────────────────────────────────────
    all_nodes       = list(tree.keys())
    #decision_nodes  = [0] + [nid for nid in tree.keys() if nid != 0]  # ALL nodes (root + all non-root) - true here-and-now decisions from root
    decision_nodes = [nid for nid in tree.keys() if nid != 0]   # non-root only
    internal_nodes  = [nid for nid, n in tree.items() if n['children'] and nid != 0]  # Nodes with children
    leaf_nodes      = [nid for nid, n in tree.items() if not n['children'] and nid != 0]  # Leaf nodes
    decision_set    = set(decision_nodes)
    internal_set    = set(internal_nodes)
    leaf_set        = set(leaf_nodes)

    # ── Exogenous data dicts (keyed by decision node id) ──────────────────────
    # For root (node 0): use current state as centroid (here-and-now decision point)
    root_centroid = {
        'price': current_state['price_t'],
        'occ1': current_state['Occ1'],
        'occ2': current_state['Occ2'],
    }
    # price_by_node  = {0: root_centroid['price']}
    # occ1_by_node   = {0: root_centroid['occ1']}
    # occ2_by_node   = {0: root_centroid['occ2']}

    price_by_node  = {}
    occ1_by_node   = {}
    occ2_by_node   = {}

    # Add non-root nodes
    for nid in [n for n in tree.keys() if n != 0]:
        price_by_node[nid] = tree[nid]['centroid']['price']
        occ1_by_node[nid] = tree[nid]['centroid']['occ1']
        occ2_by_node[nid] = tree[nid]['centroid']['occ2']
    path_prob_leaf = {nid: tree[nid]['path_prob']           for nid in leaf_nodes}

    m = ConcreteModel()

    # ── Sets ──────────────────────────────────────────────────────────────────
    m.R      = Set(initialize=[1, 2])
    m.N      = Set(initialize=all_nodes)
    m.N_dec  = Set(initialize=decision_nodes)   # All nodes including root (here-and-now decisions)
    m.N_int  = Set(initialize=internal_nodes)   # Nodes with children
    m.N_leaf = Set(initialize=leaf_nodes)
    m.RN_dec = m.R * m.N_dec
    m.RN_int = m.R * m.N_int

    # ── Physical parameters ───────────────────────────────────────────────────
    # m.Pr    = Param(initialize=d['heating_max_power'])
    # m.Zexch = Param(initialize=d['heat_exchange_coeff'])
    # m.Zconv = Param(initialize=d['heating_efficiency_coeff'])
    # m.Zloss = Param(initialize=d['thermal_loss_coeff'])
    # m.Zcool = Param(initialize=d['heat_vent_coeff'])
    # m.Zocc  = Param(initialize=d['heat_occupancy_coeff'])
    # m.Tmin  = Param(initialize=d['temp_min_comfort_threshold'])
    # m.Tok   = Param(initialize=d['temp_OK_threshold'])
    # m.Thigh = Param(initialize=d['temp_max_comfort_threshold'])
    # m.Hhigh = Param(initialize=d['humidity_threshold'])
    # m.Pvent = Param(initialize=d['ventilation_power'])
    # m.Hocc  = Param(initialize=d['humidity_occupancy_coeff'])
    # m.Hvent = Param(initialize=d['humidity_vent_coeff'])
    # m.Tout  = Param(range(horizon),
    #                 initialize={t: d['outdoor_temperature'][t] for t in range(horizon)})
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
    m.M_temp = Param(initialize=100.0)
    m.M_hum  = Param(initialize=100.0)
    m.U_vent = Param(initialize=d['vent_min_up_time'])
    #m.Tout   = Param(range(horizon),
     #                initialize={t: d['outdoor_temperature'][t] for t in range(horizon)})
    _num_timeslots = int(d['num_timeslots'])

    m.Tout = Param(range(horizon),
                initialize={t: d['outdoor_temperature'][(current_state['current_time'] + t) % _num_timeslots]
                            for t in range(horizon)})

    m.prices = Param(m.N_dec, initialize=price_by_node)
    m.O1     = Param(m.N_dec, initialize=occ1_by_node)
    m.O2     = Param(m.N_dec, initialize=occ2_by_node)
    m.pi     = Param(m.N_leaf, initialize=path_prob_leaf)

    # ── Decision variables (decision nodes: all non-root nodes) ──────────────────
    m.Heat   = Var(m.RN_dec, domain=NonNegativeReals, bounds=(0, d['heating_max_power']))
    m.Vent   = Var(m.N_dec,  domain=Binary)
    m.Vstart = Var(m.N_dec,  domain=Binary)
    # Overrule indicator variables
    m.y_low  = Var(m.RN_dec, domain=Binary)  
    m.y_ok   = Var(m.RN_dec, domain=Binary)  
    m.y_high = Var(m.RN_dec, domain=Binary)   
    m.u      = Var(m.RN_dec, domain=Binary) 
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

        # if pid == 0:
        #     # State flows from root; decisions made AT nid (root has no vars)
        #     t_root = tree[0]['t']   # = 0
        #     if nid in leaf_set:
        #         # Horizon=1 edge case: leaf whose parent is root, no decision
        #         return m.T_in[r, nid] == (
        #             m.T_in[r, 0]
        #             + m.Zexch * (m.T_in[other_r, 0] - m.T_in[r, 0])
        #             + m.Zloss * (m.Tout[t_root]      - m.T_in[r, 0])
        #         )
        #     return m.T_in[r, nid] == (
        #         m.T_in[r, 0]
        #         + m.Zexch * (m.T_in[other_r, 0] - m.T_in[r, 0])
        #         + m.Zloss * (m.Tout[t_root]      - m.T_in[r, 0])
        #         + m.Zconv * m.Heat[r, nid]
        #         - m.Zcool * m.Vent[nid]
        #         + m.Zocc  * occ(r, nid)
        #     )

        # # General case: pid is a non-root internal node
        # t_parent = tree[pid]['t']
        # if nid in leaf_set:
        #     return m.T_in[r, nid] == (
        #         m.T_in[r, pid]
        #         + m.Zexch * (m.T_in[other_r, pid] - m.T_in[r, pid])
        #         + m.Zloss * (m.Tout[t_parent]      - m.T_in[r, pid])
        #         + m.Zconv * m.Heat[r, pid]
        #         - m.Zcool * m.Vent[pid]
        #         + m.Zocc  * occ(r, pid)
        #     )
        # return m.T_in[r, nid] == (
        #     m.T_in[r, pid]
        #     + m.Zexch * (m.T_in[other_r, pid] - m.T_in[r, pid])
        #     + m.Zloss * (m.Tout[t_parent]      - m.T_in[r, pid])
        #     + m.Zconv * m.Heat[r, pid]
        #     - m.Zcool * m.Vent[pid]
        #     + m.Zocc  * occ(r, pid)

        if pid == 0:
            t_idx = 0  # parent is root, which is at current_time → Tout[0]
            if nid in leaf_set:
                return m.T_in[r, nid] == (
                    m.T_in[r, 0]
                    + m.Zexch * (m.T_in[other_r, 0] - m.T_in[r, 0])
                    + m.Zloss * (m.Tout[t_idx]       - m.T_in[r, 0])
                )
            return m.T_in[r, nid] == (
                m.T_in[r, 0]
                + m.Zexch * (m.T_in[other_r, 0] - m.T_in[r, 0])
                + m.Zloss * (m.Tout[t_idx]       - m.T_in[r, 0])
                + m.Zconv * m.Heat[r, nid]
                - m.Zcool * m.Vent[nid]
                + m.Zocc  * occ(r, nid)
            )

        # General case
        t_idx = tree[pid]['t'] - 1  # parent's stage = correct Tout offset
        if nid in leaf_set:
            return m.T_in[r, nid] == (
                m.T_in[r, pid]
                + m.Zexch * (m.T_in[other_r, pid] - m.T_in[r, pid])
                + m.Zloss * (m.Tout[t_idx]         - m.T_in[r, pid])
                + m.Zconv * m.Heat[r, pid]
                - m.Zcool * m.Vent[pid]
                + m.Zocc  * occ(r, pid)
            )
        return m.T_in[r, nid] == (
            m.T_in[r, pid]
            + m.Zexch * (m.T_in[other_r, pid] - m.T_in[r, pid])
            + m.Zloss * (m.Tout[t_idx]         - m.T_in[r, pid])
            + m.Zconv * m.Heat[r, pid]
            - m.Zcool * m.Vent[pid]
            + m.Zocc  * occ(r, pid)
        )

    m.TempDyn = Constraint(m.R * m.N, rule=temp_dynamics)

    # ── Humidity dynamics ─────────────────────────────────────────────────────
    
    # def hum_dynamics(m, nid):
    #     if nid == 0:
    #         return m.Hum[nid] == H_init
    #     pid = tree[nid]['parent']
    #     if pid == 0:
    #         if nid in leaf_set:
    #             return m.Hum[nid] == m.Hum[0]
    #         return m.Hum[nid] == (
    #             m.Hum[0]
    #             - m.Hvent * m.Vent[nid]
    #             + m.Hocc * (m.O1[nid] + m.O2[nid])   # nid è in N_int, pid=0 non lo è
    #         )
    #     return m.Hum[nid] == (
    #         m.Hum[pid]
    #         - m.Hvent * m.Vent[pid]
    #         + m.Hocc * (m.O1[pid] + m.O2[pid])        # pid è in N_int, coerente con temp_dynamics
    # )
    def hum_dynamics(m, nid):
        if nid == 0:
            return m.Hum[nid] == H_init
        pid = tree[nid]['parent']
        if pid == 0:
            if nid in leaf_set:
                # leaf whose parent is root: no decisions at leaf, use root state only
                return m.Hum[nid] == m.Hum[0]
            # stage-1 non-leaf: decisions live at nid, applied from root state
            return m.Hum[nid] == (
                m.Hum[0]
                - m.Hvent * m.Vent[nid]
                + m.Hocc  * (m.O1[nid] + m.O2[nid])
            )
        # General case: transition from pid using pid's decisions
        if nid in leaf_set:
            # leaf: no decisions at nid, use parent's decisions
            return m.Hum[nid] == (
                m.Hum[pid]
                - m.Hvent * m.Vent[pid]
                + m.Hocc  * (m.O1[pid] + m.O2[pid])
            )
        return m.Hum[nid] == (
            m.Hum[pid]
            - m.Hvent * m.Vent[pid]
            + m.Hocc  * (m.O1[pid] + m.O2[pid])
        )

    m.HumDyn = Constraint(m.N, rule=hum_dynamics)

        # ── 1. High temperature: forced heating shutdown ──────────────────────────
    #   y_high = 1  ⟺  T_in > Thigh
    m.CThigh1 = Constraint(m.RN_dec,
        rule=lambda m, r, nid:
            m.T_in[r, nid] >= m.Thigh - m.M_temp * (1 - m.y_high[r, nid]))
    m.CThigh2 = Constraint(m.RN_dec,
        rule=lambda m, r, nid:
            m.T_in[r, nid] <= m.Thigh + m.M_temp * m.y_high[r, nid])
    m.CHeatOff = Constraint(m.RN_dec,
        rule=lambda m, r, nid:
             m.Heat[r, nid] <= m.Pr * (1 - m.y_high[r, nid]))
    
    # ── 2. Low temperature: overrule activation ───────────────────────────────
    #   y_low = 1  ⟺  T_in < Tmin
    m.CTlow1 = Constraint(m.RN_dec,
        rule=lambda m, r, nid:
            m.T_in[r, nid] <= m.Tmin + m.M_temp * (1 - m.y_low[r, nid]))
    m.CTlow2 = Constraint(m.RN_dec,
        rule=lambda m, r, nid:
            m.T_in[r, nid] >= m.Tmin - m.M_temp * m.y_low[r, nid])
 
    # ── 3. Temperature-OK: overrule deactivation ──────────────────────────────
    #   y_ok = 1  ⟺  T_in >= Tok
    m.CTok1 = Constraint(m.RN_dec,
        rule=lambda m, r, nid:
            m.T_in[r, nid] >= m.Tok - m.M_temp * (1 - m.y_ok[r, nid]))
    m.CTok2 = Constraint(m.RN_dec,
        rule=lambda m, r, nid:
            m.T_in[r, nid] <= m.Tok + m.M_temp * m.y_ok[r, nid])
 
    # ── 4. Overrule memory (u) propagated through tree parent pointers ────────
    #   CU1 : u >= y_low
    m.CU1 = Constraint(m.RN_dec,
        rule=lambda m, r, nid: m.u[r, nid] >= m.y_low[r, nid])
 
    #   CU2 : u <= u_prev + y_low
    def c_u2(m, r, nid):
        pid    = tree[nid]['parent']
        u_prev = low_override[r] if (pid is None or pid == 0) else m.u[r, pid]
        return m.u[r, nid] <= u_prev + m.y_low[r, nid]
    m.CU2 = Constraint(m.RN_dec, rule=c_u2)
 
    #   CU3 : u >= u_prev - y_ok   (persist until temperature recovers)
    def c_u3(m, r, nid):
        pid    = tree[nid]['parent']
        u_prev = low_override[r] if (pid is None or pid == 0) else m.u[r, pid]
        return m.u[r, nid] >= u_prev - m.y_ok[r, nid]
    m.CU3 = Constraint(m.RN_dec, rule=c_u3)
 
    #   CU4 : u <= 1 - y_ok        (deactivate as soon as T >= Tok)
    m.CU4 = Constraint(m.RN_dec,
        rule=lambda m, r, nid: m.u[r, nid] <= 1 - m.y_ok[r, nid])
 
    #   Full heating power required during overrule
    m.CHeatMax = Constraint(m.RN_dec,
        rule=lambda m, r, nid: m.Heat[r, nid] >= m.Pr * m.u[r, nid])
 
    # ─────────────────────────────────────────────────────────────────────────
    # HUMIDITY OVERRULE
    # ─────────────────────────────────────────────────────────────────────────
    m.CVentHum = Constraint(m.N_dec,
        rule=lambda m, nid: m.Hum[nid] <= m.Hhigh + m.M_hum * m.Vent[nid])
 
    # ─────────────────────────────────────────────────────────────────────────
    # VENTILATION INERTIA  (SP-style Vstart startup signal)
    # ─────────────────────────────────────────────────────────────────────────
 
    # CVstart1 : Vstart[nid] >= Vent[nid] - Vent[parent]
    def c_vstart1(m, nid):
        pid = tree[nid]['parent']
        v_p = v_prev if (pid is None or pid == 0) else m.Vent[pid]
        return m.Vstart[nid] >= m.Vent[nid] - v_p
    m.CVstart1 = Constraint(m.N_dec, rule=c_vstart1)
 
    # CVstart2 : Vstart[nid] <= Vent[nid]
    m.CVstart2 = Constraint(m.N_dec,
        rule=lambda m, nid: m.Vstart[nid] <= m.Vent[nid])
 
    # CVstart3 : Vstart[nid] <= 1 - Vent[parent]
    def c_vstart3(m, nid):
        pid = tree[nid]['parent']
        v_p = v_prev if (pid is None or pid == 0) else m.Vent[pid]
        return m.Vstart[nid] <= 1 - v_p
    m.CVstart3 = Constraint(m.N_dec, rule=c_vstart3)
 
    # MinVentOn : Σ_{k in descendant chain} Vent[k] >= |chain| * Vstart[nid]
    def min_uptime(m, nid):
        chain = [k for k in _descendants_chain(tree, nid, m.U_vent)
                 if k in decision_set]
        if not chain:
            return Constraint.Skip
        return sum(m.Vent[k] for k in chain) >= len(chain) * m.Vstart[nid]
    m.MinVentOn = Constraint(m.N_dec, rule=min_uptime)


    # ── Stage-1 Non-Anticipativity Constraints ────────────────────────────────
    # All children of root have not yet observed any new information,
    # so they must share the same here-and-now decision.
    stage1_nodes = tree[0]['children']

    def heat_nac(m, r, i, j):
        if i >= j:
            return Constraint.Skip
        return m.Heat[r, stage1_nodes[i]] == m.Heat[r, stage1_nodes[j]]
    m.HeatNAC = Constraint(m.R, 
                            range(len(stage1_nodes)), 
                            range(len(stage1_nodes)), 
                            rule=heat_nac)

    def vent_nac(m, i, j):
        if i >= j:
            return Constraint.Skip
        return m.Vent[stage1_nodes[i]] == m.Vent[stage1_nodes[j]]
    m.VentNAC = Constraint(range(len(stage1_nodes)), 
                            range(len(stage1_nodes)), 
                            rule=vent_nac)

 
    # ─────────────────────────────────────────────────────────────────────────
    # OBJECTIVE: E[cost along each leaf path]
    # ─────────────────────────────────────────────────────────────────────────
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
    # # ── Overrule controller: LOW temperature ──────────────────────────────────
    # def u_activation(m, r, nid):
    #     return m.T_in[r, nid] >= m.Tmin - M * m.u[r, nid]
    # m.UActivation = Constraint(m.RN_int, rule=u_activation)

    # def w_deactivation(m, r, nid):
    #     return m.T_in[r, nid] >= m.Tok - M * (1 - m.w[r, nid])
    # m.WDeactivation = Constraint(m.RN_int, rule=w_deactivation)

    # def u_persistence(m, r, nid):
    #     pid = tree[nid]['parent']
    #     if pid is None or pid == 0:
    #         # Initialise from real system overrule state
    #         init_u = 1 if current_state.get(f'low_override_r{r}', 0) else 0
    #         m.u[r, nid].fix(init_u)
    #         m.w[r, nid].fix(0)
    #         return Constraint.Skip
    #     return m.u[r, nid] >= m.u[r, pid] - m.w[r, nid]
    # m.UPersistence = Constraint(m.RN_int, rule=u_persistence)

    # def heat_max_when_overrule(m, r, nid):
    #     return m.Heat[r, nid] >= m.Pr * m.u[r, nid]
    # m.HeatMaxOverrule = Constraint(m.RN_int, rule=heat_max_when_overrule)

    # # ── Overrule controller: HIGH temperature ─────────────────────────────────
    # def y_activation(m, r, nid):
    #     return m.T_in[r, nid] <= m.Thigh + M * m.y[r, nid]
    # m.YActivation = Constraint(m.RN_int, rule=y_activation)

    # def heat_off_when_overrule(m, r, nid):
    #     return m.Heat[r, nid] <= m.Pr * (1 - m.y[r, nid])
    # m.HeatOffOverrule = Constraint(m.RN_int, rule=heat_off_when_overrule)

    # # ── Humidity overrule: force ventilation ON when humid ────────────────────
    # def vent_humidity_overrule(m, nid):
    #     return m.Hum[nid] <= m.Hhigh + M * m.Vent[nid]
    # m.VentHumOverrule = Constraint(m.N_int, rule=vent_humidity_overrule)

    # # ── Ventilation inertia: 3-hour minimum ON time ───────────────────────────
    # def on_off_exclusivity(m, nid):
    #     return m.Uon[nid] + m.Uoff[nid] <= 1
    # m.OnOffExcl = Constraint(m.N_int, rule=on_off_exclusivity)

    # def uoff_bound(m, nid):
    #     return m.Uoff[nid] <= 1 - m.Vent[nid]
    # m.UoffBound = Constraint(m.N_int, rule=uoff_bound)

    # def uon_bound(m, nid):
    #     return m.Uon[nid] <= 1 - m.Vent[nid]
    # m.UonBound = Constraint(m.N_int, rule=uon_bound)

    # def min_uptime(m, nid):
    #     L     = d['vent_min_up_time']
    #     chain = [k for k in _descendants_chain(tree, nid, L) if k in internal_set]
    #     pid   = tree[nid]['parent']
    #     v_p   = v_prev if (pid is None or pid == 0) else m.Vent[pid]
    #     return sum(m.Vent[k] for k in chain) >= len(chain) * (m.Vent[nid] - v_p)
    # m.MinUptime = Constraint(m.N_int, rule=min_uptime)

    # # Carry-over inertia from previous real hours
    # if v_prev == 1 and v_on_h < 3:
    #     forced_on = 3 - v_on_h
    #     for nid in internal_nodes:
    #         if tree[nid]['t'] <= forced_on:
    #             m.Vent[nid].fix(1)

    # # ── Objective: E[cost along each leaf path] ───────────────────────────────
    # def objective(m):
    #     return sum(
    #         m.pi[leaf] * sum(
    #             m.prices[nid] * (
    #                 sum(m.Heat[r, nid] for r in m.R)
    #                 + m.Vent[nid] * m.Pvent
    #             )
    #             for nid in _path_to_root(tree, leaf)
    #         )
    #         for leaf in leaf_nodes
    #     )
    # m.obj = Objective(rule=objective, sense=minimize)
    # return m

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

    # Reproducible per-timestep RNG
    rng    = np.random.default_rng(seed=42 + t)
    # Ventilation status derived from counter
    vc       = state.get('vent_counter', 0)
    v_status = 1 if vc > 0 else 0

    # # ── Generate raw Monte-Carlo scenarios ────────────────────────────────────

    # price_dict, occ_dict, _ = generate_tree_scenarios(
    #     price_now   = state['price_t'],
    #     price_prev  = state['price_previous'],
    #     occ_r1_now  = state['Occ1'],
    #     occ_r2_now  = state['Occ2'],
    #     branching_factors= BRANCHING_FACTORS,
    #     rng         = rng,
    # )

    price_dict, occ_dict, _ = generate_scenarios(
        price_now   = state['price_t'],
        price_prev  = state['price_previous'],
        occ_r1_now  = state['Occ1'],
        occ_r2_now  = state['Occ2'],
        horizon= HORIZON_MULTI,
        n_scenarios = GEN_SCENARIOS,
        rng         = rng,
    )

    #── Build scenario tree ───────────────────────────────────────────────────
    tree = cluster_scenarios_tree2(
        price_dict, occ_dict,
        N_CLUSTERS = N_CLUSTERS,
        horizon             = horizon,
        rng                 = rng
    )

    # tree = cluster_scenarios_tree(
    #     price_dict, occ_dict,
    #     n_branches = N_CLUSTERS,
    #     horizon             = horizon,
    #     scenarios_to_generate=  GEN_SCENARIOS
    # )

    # for nid, node in sorted(tree.items()):
    #     print(f"node={nid}, t={node['t']}, parent={node['parent']}, children={node['children']}")

    # ── Guard: need at least one internal non-root node for decisions ─────────
    internal_nodes = [nid for nid, n in tree.items() if n['children'] and nid != 0]
    # if not internal_nodes:
    #     return {'HeatPowerRoom1': 0.0, 'HeatPowerRoom2': 0.0, 'VentilationON': 0}

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
    model  = build_multisp_model(state, tree, horizon)
    solver = SolverFactory('gurobi_direct')
    solver.options['TimeLimit'] = 15
    solver.options['MIPGap']    = 0.02
    solver.options['Seed']      = 42
    solver.options['Threads']   = 1
    result = solver.solve(model, tee=False)



    # Debug print - remove after verification
    # if state['current_time'] == 0:  # only print at t=0 to avoid spam
    #     print(f"\nTout values (current_time={state['current_time']}):")
    #     for t in range(horizon):
    #         print(f"  Tout[{t}] = {value(model.Tout[t]):.3f}")
    #     print(f"  outdoor_temp raw: {DATA['outdoor_temperature']}")

    #     # ← ADD HERE
    #     stage1_nodes = tree[0]['children']
    #     print("Stage-1 decisions:")
    #     for nid in stage1_nodes:
    #         p1 = float(value(model.Heat[1, nid]))
    #         p2 = float(value(model.Heat[2, nid]))
    #         v  = int(round(float(value(model.Vent[nid]))))
    #         print(f"  node {nid}: Heat1={p1:.3f}, Heat2={p2:.3f}, Vent={v}")

    # if state['current_time'] == 0:
    #     stage1_nodes = tree[0]['children']
    #     print("Stage-1 centroids:")
    #     for nid in stage1_nodes:
    #         print(f"  node {nid}: price={tree[nid]['centroid']['price']:.3f}, "
    #             f"occ1={tree[nid]['centroid']['occ1']:.3f}, "
    #             f"occ2={tree[nid]['centroid']['occ2']:.3f}")

    # print(f"T1={state['T1']}, T2={state['T2']}")
    # print(f"Tmin={DATA['temp_min_comfort_threshold']}, Tok={DATA['temp_OK_threshold']}")


    # ── Guard against infeasible / failed solves ──────────────────────────────
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

    # ── Extract here-and-now decision from root ──────────────────────────────
    # True non-anticipativity: decision taken at root (before any scenario unfolds)
    # All scenarios must follow this same decision at node 0
    # decision_node = 0  # Root node

    # p1 = float(value(model.Heat[1, decision_node]))
    # p2 = float(value(model.Heat[2, decision_node]))
    # v  = int(round(float(value(model.Vent[decision_node]))))

    decision_node = tree[0]['children'][0]   # First stage-1 child (all share the same here-and-now decision)
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
