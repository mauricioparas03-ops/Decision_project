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
from Data.PriceProcessRestaurant import price_model
from Data.OccupancyProcessRestaurant import next_occupancy_levels

# ── Hyper-parameters ───────────────────────────────────────────────────────────
HORIZON           = 4              # lookahead steps (must be >= 3 due to vent-inertia)
GEN_SCENARIOS     = 1000           # raw Monte-Carlo paths before tree clustering (total paths)
N_SCENARIOS       = 3              # K-Means clusters retained per node (branches)
BRANCHING_FACTORS = [3, 3, 3, 3]  # per-stage branch counts; len must equal HORIZON
 
# =============================================================================
# SYSTEM DATA
# =============================================================================
DATA = get_fixed_data()
 
 
# =============================================================================
# STAGE 1A – FULL-HORIZON PATH GENERATION
# =============================================================================
 
def generate_scenarios(price_now, price_prev, occ1_now, occ2_now,
                       horizon, n_scenarios):
    """
    Draw *n_scenarios* full-horizon Monte-Carlo paths from the current state.
 
    Mirrors SP_policy.generate_scenarios() exactly: each path chains AR(1)
    price draws and occupancy draws step-by-step so that path coherence is
    fully preserved — the price at step t+1 is drawn from the price at step t
    on the SAME path, not from a cluster centroid.
 
    Parameters
    ----------
    price_now   : float – observed electricity price (current timestep)
    price_prev  : float – price one step earlier (AR model input)
    occ1_now    : float – observed occupancy room 1
    occ2_now    : float – observed occupancy room 2
    horizon     : int   – number of lookahead steps
    n_scenarios : int   – number of MC paths to draw (→ GEN_SCENARIOS)
 
    Returns
    -------
    paths : np.ndarray shape (n_scenarios, horizon, 3)
            axis 0 = scenario index
            axis 1 = lookahead step (0 = first step ahead)
            axis 2 = [price, occ1, occ2]
    """
    paths = np.empty((n_scenarios, horizon, 3))
 
    for s in range(n_scenarios):
        p_cur,  p_prev  = price_now,  price_prev
        o1_cur, o2_cur  = occ1_now,   occ2_now
 
        for t in range(horizon):
            p_next           = price_model(p_cur, p_prev)
            o1_next, o2_next = next_occupancy_levels(o1_cur, o2_cur)
 
            paths[s, t] = [p_next, o1_next, o2_next]
 
            # Chain: next step's AR(1) input uses this step's output
            p_prev, p_cur  = p_cur,  p_next
            o1_cur, o2_cur = o1_next, o2_next
 
    return paths
 
 
# =============================================================================
# STAGE 1B – CLUSTER A SUBSET OF PATHS AT ONE STAGE LEVEL
# =============================================================================
 
def cluster_scenarios(paths_subset, stage_idx, n_clusters):
    """
    Cluster *paths_subset* by their values at *stage_idx* into *n_clusters*
    groups via K-Means.
 
    Clustering is done on the stage_idx column only (price, occ1, occ2 at
    that step) so that paths are grouped by what actually happens at that
    stage — preserving the within-group path coherence for deeper stages.
 
    Parameters
    ----------
    paths_subset : np.ndarray shape (n, horizon, 3) – subset of full paths
    stage_idx    : int  – which lookahead step to cluster on (0-based)
    n_clusters   : int  – number of clusters (→ BRANCHING_FACTORS[stage-1])
 
    Returns
    -------
    labels     : np.ndarray (n,)           – cluster assignment per path
    centroids  : np.ndarray (n_clusters, 3) – mean [price,occ1,occ2] per cluster
                                              in original scale
    cond_prob  : np.ndarray (n_clusters,)  – fraction of paths in each cluster
    """
    n          = len(paths_subset)
    n_clusters = min(n_clusters, n)
 
    # Cluster on the values at this stage only
    X = paths_subset[:, stage_idx, :]       # shape (n, 3)
 
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
 
    km     = KMeans(n_clusters=n_clusters, n_init=5, random_state=42)
    labels = km.fit_predict(X_scaled)
 
    cond_prob = np.bincount(labels, minlength=n_clusters) / n
 
    centroids = scaler.inverse_transform(km.cluster_centers_)
    centroids[:, 1] = np.maximum(0.0, centroids[:, 1])   # occ1 >= 0
    centroids[:, 2] = np.maximum(0.0, centroids[:, 2])   # occ2 >= 0
 
    return labels, centroids, cond_prob
 
 
# =============================================================================
# STAGE 1C – BUILD THE FULL SCENARIO TREE (PATH-COHERENT)
# =============================================================================
 
def grow_scenario_tree(price_now, price_prev, occ1_now, occ2_now,
                       horizon=HORIZON,
                       gen_scenarios=GEN_SCENARIOS,
                       branching_factors=BRANCHING_FACTORS):
    """
    Grow a branching scenario tree by partitioning full coherent paths top-down.
 
    Algorithm
    ---------
    1. Generate GEN_SCENARIOS full horizon-length paths from the current state.
    2. Root node (stage=0) owns ALL paths.
    3. For each stage 1..horizon, for each existing node at stage-1:
         a. Take the subset of paths that belong to this node.
         b. Cluster those paths by their values at this stage →
            branching_factors[stage-1] clusters.
         c. Each cluster becomes a child node whose centroid is the mean
            [price, occ1, occ2] of the paths in that cluster at this stage.
         d. The child inherits only the paths in its cluster for deeper stages.
         e. Child probability = parent.prob × (cluster_size / parent_size).
 
    Key property: every node's centroid is derived from real chained AR(1)
    paths, not from resampling at cluster centroids. Path coherence is fully
    preserved within each subtree.
 
    Node dict format
    ----------------
    nodes[nid] = {
        'stage'     : int,
        'parent'    : int | None,
        'prob'      : float,        – joint probability of reaching this node
        'price'     : float,        – centroid price at this stage
        'occ1'      : float,        – centroid occ1 at this stage
        'occ2'      : float,        – centroid occ2 at this stage
        'price_prev': float,        – centroid price at previous stage
                                       (for Tout indexing; not used in AR chain)
    }
 
    Parameters
    ----------
    price_now         : float
    price_prev        : float
    occ1_now          : float
    occ2_now          : float
    horizon           : int   – lookahead steps        (default HORIZON)
    gen_scenarios     : int   – total MC paths drawn   (default GEN_SCENARIOS)
    branching_factors : list  – branches per stage     (default BRANCHING_FACTORS)
 
    Returns
    -------
    nodes    : dict  {nid: node_dict}
    children : dict  {nid: [child_nid, ...]}
    leaves   : list  [nid, ...]  – leaf node ids (stage == horizon)
    """
    # ── Step 1: generate all full paths from root ─────────────────────────────
    all_paths = generate_scenarios(
        price_now, price_prev, occ1_now, occ2_now,
        horizon=horizon, n_scenarios=gen_scenarios,
    )
    # all_paths: shape (gen_scenarios, horizon, 3)
 
    # ── Step 2: root node owns all path indices ───────────────────────────────
    nodes = {
        0: {
            'stage'     : 0,
            'parent'    : None,
            'prob'      : 1.0,
            'price'     : price_now,
            'occ1'      : occ1_now,
            'occ2'      : occ2_now,
            'price_prev': price_prev,
        }
    }
    children   = {0: []}
    # Track which path indices belong to each node
    node_paths = {0: np.arange(gen_scenarios)}
    next_id    = 1
 
    # ── Step 3: grow stage by stage, partitioning paths top-down ─────────────
    for stage in range(1, horizon + 1):
        stage_idx = stage - 1          # 0-based index into paths axis-1
        b         = branching_factors[stage - 1]
        frontier  = [nid for nid, nd in nodes.items()
                     if nd['stage'] == stage - 1]
 
        for pid in frontier:
            parent_idx = node_paths[pid]          # path indices at this node
            pnode      = nodes[pid]
 
            if len(parent_idx) == 0:
                children[pid] = []
                continue
 
            # Subset of paths belonging to this parent node
            subset = all_paths[parent_idx]        # shape (n_parent, horizon, 3)
 
            # Cluster by values at this stage
            labels, centroids, cond_prob = cluster_scenarios(
                subset, stage_idx=stage_idx, n_clusters=b
            )
 
            children[pid] = []
            for k in range(len(centroids)):
                # Path indices (in all_paths) that fall in cluster k
                cluster_mask    = (labels == k)
                cluster_path_idx = parent_idx[cluster_mask]
 
                if len(cluster_path_idx) == 0:
                    continue
 
                joint_prob = pnode['prob'] * cond_prob[k]
                p_next, o1_next, o2_next = centroids[k]
 
                nid = next_id
                next_id += 1
 
                nodes[nid] = {
                    'stage'     : stage,
                    'parent'    : pid,
                    'prob'      : joint_prob,
                    'price'     : float(p_next),
                    'occ1'      : float(o1_next),
                    'occ2'      : float(o2_next),
                    'price_prev': pnode['price'],
                }
                children[pid].append(nid)
                children[nid]   = []
                node_paths[nid] = cluster_path_idx
 
    # ── Collect leaves ────────────────────────────────────────────────────────
    leaves = [nid for nid, nd in nodes.items() if nd['stage'] == horizon]
 
    return nodes, children, leaves
 
 
# =============================================================================
# DIAGNOSTIC UTILITIES
# =============================================================================
 
def print_tree_summary(nodes, children, leaves):
    """Print a compact summary of the scenario tree (for debugging)."""
    horizon     = max(nd['stage'] for nd in nodes.values())
    total       = len(nodes)
    total_leaves = 1
    for b in BRANCHING_FACTORS:
        total_leaves *= b
    print(f"Scenario tree: horizon={horizon}, total nodes={total}, leaves={len(leaves)}")
    print(f"  Expected leaves = {' × '.join(str(b) for b in BRANCHING_FACTORS)} = {total_leaves}")
    print(f"  Prob mass at leaves = {sum(nodes[l]['prob'] for l in leaves):.4f}  (should ≈ 1.0)")
    print()
    for stage in range(horizon + 1):
        stage_nodes = [nid for nid, nd in nodes.items() if nd['stage'] == stage]
        print(f"  Stage {stage}: {len(stage_nodes)} nodes  "
              f"(ids {min(stage_nodes)}..{max(stage_nodes)})")

# # ── Hyper-parameters ───────────────────────────────────────────────────────────
# HORIZON = 4    # lookahead steps (must be >= 3 due to vent-inertia)
# GEN_SCENARIOS = 100  # raw Monte Carlo paths before tree clustering
# N_SCENARIOS    = 3 
# BRANCHING_FACTORS = [3,3,3,3] 

# # =============================================================================
# # SYSTEM PARAMETERS
# # =============================================================================

# DATA = get_fixed_data()

# # =============================================================================
# # STAGE 1A – FORWARD SAMPLING FROM A SINGLE NODE
# # =============================================================================

# def generate_scenarios(price_now, price_prev, occ1_now, occ2_now, n_scenarios):
#     """
#     Draw *n_scenarios* one-step-ahead realisations from a single tree node.
 
#     Mirrors SP_policy.generate_scenarios() in name and spirit, but advances
#     only ONE step forward (the tree builder calls it once per node, not once
#     per full horizon).
 
#     Parameters
#     ----------
#     price_now  : float  – electricity price at this node
#     price_prev : float  – price one step earlier (AR model input)
#     occ1_now   : float  – occupancy room 1 at this node
#     occ2_now   : float  – occupancy room 2 at this node
#     n_scenarios : int   – number of Monte-Carlo draws (→ GEN_SCENARIOS)
 
#     Returns
#     -------
#     samples : np.ndarray shape (n_scenarios, 3)
#               columns: [price_next, occ1_next, occ2_next]
#     """
#     samples = np.empty((n_scenarios, 3))
#     for i in range(n_scenarios):
#         p_next           = price_model(price_now, price_prev)
#         o1_next, o2_next = next_occupancy_levels(occ1_now, occ2_now)
#         samples[i]       = [p_next, o1_next, o2_next]
#     return samples

# # =============================================================================
# # STAGE 1B – CLUSTER ONE NODE'S CHILDREN → N_SCENARIOS CENTROIDS
# # =============================================================================
 
# def cluster_scenarios(samples, n_clusters):
#     """
#     Reduce *samples* (shape n_scenarios × 3) to *n_clusters* representative
#     centroids via K-Means, returning centroid values and conditional
#     probabilities.
 
#     Mirrors SP_policy.cluster_scenarios() in name and approach (StandardScaler
#     + KMeans), adapted for the single-node, one-step-ahead case.
 
#     Parameters
#     ----------
#     samples    : np.ndarray  shape (n_scenarios, 3)  – [price, occ1, occ2]
#     n_clusters : int  – number of clusters to retain (→ N_SCENARIOS or
#                         BRANCHING_FACTORS[stage-1] for per-stage control)
 
#     Returns
#     -------
#     centroids : np.ndarray shape (n_clusters, 3)  – in original scale
#     cond_prob : np.ndarray shape (n_clusters,)    – conditional probability
#                 of each cluster given the parent node was reached
#     """
#     n          = len(samples)
#     n_clusters = min(n_clusters, n)   # can't have more clusters than samples
 
#     scaler   = StandardScaler()
#     X_scaled = scaler.fit_transform(samples)
 
#     km = KMeans(n_clusters=n_clusters, n_init=5, random_state=42)
#     km.fit(X_scaled)
 
#     labels    = km.labels_
#     cond_prob = np.bincount(labels, minlength=n_clusters) / n
 
#     centroids = scaler.inverse_transform(km.cluster_centers_)
#     centroids[:, 1] = np.maximum(0.0, centroids[:, 1])   # occ1 >= 0
#     centroids[:, 2] = np.maximum(0.0, centroids[:, 2])   # occ2 >= 0
 
#     return centroids, cond_prob

# # =============================================================================
# # STAGE 1C – BUILD THE FULL SCENARIO TREE
# # =============================================================================
 
# def grow_scenario_tree(price_now, price_prev, occ1_now, occ2_now,
#                        horizon=HORIZON,
#                        gen_scenarios=GEN_SCENARIOS,
#                        branching_factors=BRANCHING_FACTORS):
#     """
#     Grow a branching scenario tree forward from the current observed state.
 
#     Algorithm
#     ---------
#     1. Create root node (stage=0) with the current exogenous state.
#     2. For each stage 1..horizon:
#          For every leaf node at stage-1:
#            a. Sample gen_scenarios one-step realisations (generate_scenarios).
#            b. Cluster to branching_factors[stage-1] centroids (cluster_scenarios).
#            c. Create child nodes; each child's joint prob =
#               parent.prob × conditional_prob_of_this_cluster.
#     3. Return the completed node dict.
 
#     Node dict format
#     ----------------
#     nodes[nid] = {
#         'stage'     : int,         # 0 = root; 1..horizon = decision stages
#         'parent'    : int | None,  # None only for root
#         'prob'      : float,       # joint probability P(path to this node)
#         'price'     : float,       # electricity price AT this node
#         'occ1'      : float,       # occupancy room 1 AT this node
#         'occ2'      : float,       # occupancy room 2 AT this node
#         'price_prev': float,       # price one step before (AR model input)
#     }
 
#     Parameters
#     ----------
#     price_now         : float
#     price_prev        : float
#     occ1_now          : float
#     occ2_now          : float
#     horizon           : int    – stages to grow          (default HORIZON)
#     gen_scenarios     : int    – MC draws per node       (default GEN_SCENARIOS)
#     branching_factors : list   – branches per stage      (default BRANCHING_FACTORS)
#                                  branching_factors[stage-1] used at each stage
 
#     Returns
#     -------
#     nodes    : dict  {nid: node_dict}
#     children : dict  {nid: [child_nid, ...]}   – adjacency list
#     leaves   : list  [nid, ...]                – leaf node ids (stage==horizon)
#     """
#     # ── Root node (stage 0: current observed state, no decision yet) ──────────
#     nodes = {
#         0: {
#             'stage'     : 0,
#             'parent'    : None,
#             'prob'      : 1.0,
#             'price'     : price_now,
#             'occ1'      : occ1_now,
#             'occ2'      : occ2_now,
#             'price_prev': price_prev,
#         }
#     }
#     children = {0: []}
#     next_id  = 1
 
#     # ── Grow stage by stage ───────────────────────────────────────────────────
#     for stage in range(1, horizon + 1):
#         b        = branching_factors[stage - 1]   # branches at this stage
#         frontier = [nid for nid, nd in nodes.items() if nd['stage'] == stage - 1]
 
#         for pid in frontier:
#             pnode = nodes[pid]
 
#             # 1. Sample MC scenarios from this parent's exogenous state
#             samples = generate_scenarios(
#                 price_now   = pnode['price'],
#                 price_prev  = pnode['price_prev'],
#                 occ1_now    = pnode['occ1'],
#                 occ2_now    = pnode['occ2'],
#                 n_scenarios = gen_scenarios,
#             )
 
#             # 2. Cluster to b representative centroids
#             centroids, cond_prob = cluster_scenarios(samples, n_clusters=b)
 
#             # 3. Create child nodes
#             children[pid] = []
#             for k in range(len(centroids)):
#                 p_next, o1_next, o2_next = centroids[k]
#                 joint_prob = pnode['prob'] * cond_prob[k]
 
#                 nid = next_id
#                 next_id += 1
 
#                 nodes[nid] = {
#                     'stage'     : stage,
#                     'parent'    : pid,
#                     'prob'      : joint_prob,
#                     'price'     : float(p_next),
#                     'occ1'      : float(o1_next),
#                     'occ2'      : float(o2_next),
#                     'price_prev': pnode['price'],
#                 }
#                 children[pid].append(nid)
#                 children[nid] = []
 
#     # ── Collect leaves (stage == horizon) ────────────────────────────────────
#     leaves = [nid for nid, nd in nodes.items() if nd['stage'] == horizon]

#     # for stage in range(1, horizon + 1):
#     #     stage_nodes = [nid for nid, nd in nodes.items() if nd['stage'] == stage]
#     #     prices = [nodes[nid]['price'] for nid in stage_nodes]
#     #     print(f"Stage {stage}: price range [{min(prices):.2f}, {max(prices):.2f}] "
#     #         f"std={np.std(prices):.3f}")
 
#     return nodes, children, leaves

# # =============================================================================
# # DIAGNOSTIC UTILITIES
# # =============================================================================
 
# def print_tree_summary(nodes, children, leaves):
#     """Print a compact summary of the scenario tree (for debugging)."""
#     horizon     = max(nd['stage'] for nd in nodes.values())
#     total       = len(nodes)
#     total_leaves = 1
#     for b in BRANCHING_FACTORS:
#         total_leaves *= b
#     print(f"Scenario tree: horizon={horizon}, total nodes={total}, leaves={len(leaves)}")
#     print(f"  Expected leaves = {' × '.join(str(b) for b in BRANCHING_FACTORS)} = {total_leaves}")
#     print(f"  Prob mass at leaves = {sum(nodes[l]['prob'] for l in leaves):.4f}  (should ≈ 1.0)")
#     print()
#     for stage in range(horizon + 1):
#         stage_nodes = [nid for nid, nd in nodes.items() if nd['stage'] == stage]
#         print(f"  Stage {stage}: {len(stage_nodes)} nodes  "
#               f"(ids {min(stage_nodes)}..{max(stage_nodes)})")
 
# =============================================================================
# STAGE 2 – PYOMO MILP OVER THE SCENARIO TREE
# =============================================================================
 
# This will replace build_sp_model in multiSP_policy.py
 
def build_sp_model(state, nodes, children, leaves):
    """
    Build the multi-stage stochastic MILP indexed over the scenario tree.
 
    Root node (nid=0) = current observed state → decision executed HERE.
    Stage-1..HORIZON nodes = sampled futures → recourse decisions.
 
    Non-anticipativity is structural: each node appears once, so all paths
    through a node share its decision automatically. No NAC constraints needed.
 
    Dynamics propagate parent → child:
        T_in[r, nid]  = f(T_in[r, parent], Heat[r, parent], Vent[parent], ...)
        Hum[nid]      = f(Hum[parent], Vent[parent], ...)
    Stage-1 nodes use Tinit/Hinit as the parent state (real observed values).
 
    Objective
    ---------
        min  Σ_{nid ∈ N_dec}  π[nid] · price[nid] · cost[nid]
 
    Root has π[0]=1.0 and uses the real observed price.
    Each stage>=1 node's π[nid] is its joint probability along the tree path.
 
    Parameters
    ----------
    state    : dict   – Environment.py state dict
    nodes    : dict   – {nid: node_dict} from grow_scenario_tree()
    children : dict   – {nid: [child_nid, ...]}
    leaves   : list   – leaf nid list
 
    Returns
    -------
    Pyomo ConcreteModel (unsolved)
    """
    d              = DATA
    _num_timeslots = int(d['num_timeslots'])
 
    # ── Node sets ─────────────────────────────────────────────────────────────
    # Root (nid=0): decisions only — T_in/Hum pinned to Tinit/Hinit (not vars)
    # Stage 1..HORIZON nodes: decisions + state variables T_in, Hum
    all_nodes  = list(nodes.keys())
    scen_nodes = [nid for nid, nd in nodes.items() if nd['stage'] >= 1]
    int_nodes  = [nid for nid in scen_nodes if nid not in set(leaves)]
    leaf_set   = set(leaves)
 
    # ── Helper: path from a node back to root ────────────────────────────────
    def path_to_root(nid):
        path = []
        cur  = nid
        while cur is not None:
            path.append(cur)
            cur = nodes[cur]['parent']
        return path
 
    m = ConcreteModel()
 
    # ── Sets ──────────────────────────────────────────────────────────────────
    m.R      = Set(initialize=[1, 2])
    m.N      = Set(initialize=all_nodes)    # all nodes incl. root
    m.N_scen = Set(initialize=scen_nodes)   # stage >= 1: have T_in/Hum vars
    m.N_leaf = Set(initialize=leaves)
    m.RN     = m.R * m.N                   # decisions at all nodes
    m.RN_scen= m.R * m.N_scen              # state vars at scenario nodes only
 
    # ── Current real state ────────────────────────────────────────────────────
    v_prev = 1 if state['vent_counter'] > 0 else 0
    vc     = int(state['vent_counter'])
 
    # ── State parameters (fixed initial conditions — never optimised over) ───
    m.Tinit    = Param(m.R, initialize={1: state['T1'], 2: state['T2']})
    m.Hinit    = Param(initialize=state['H'])
    m.VentInit = Param(initialize=v_prev)
    m.VCinit   = Param(initialize=vc)
 
    # ── Exogenous params ──────────────────────────────────────────────────────
    price_init = {nid: nd['price'] for nid, nd in nodes.items()}
    occ1_init  = {nid: nd['occ1']  for nid, nd in nodes.items()}
    occ2_init  = {nid: nd['occ2']  for nid, nd in nodes.items()}
    prob_init  = {nid: nd['prob']  for nid, nd in nodes.items()}
 
    # Root anchored to real observed values
    price_init[0] = state['price_t']
    occ1_init[0]  = state['Occ1']
    occ2_init[0]  = state['Occ2']
    prob_init[0]  = 1.0
 
    m.price = Param(m.N, initialize=price_init)
    m.O1    = Param(m.N, initialize=occ1_init)
    m.O2    = Param(m.N, initialize=occ2_init)
    m.pi    = Param(m.N, initialize=prob_init)
 
    # Tout offset: root uses current_time (stage=0), stage-1 uses +1, etc.
    m.Tout = Param(m.N, initialize={
        nid: d['outdoor_temperature'][
            (state['current_time'] + nd['stage']) % _num_timeslots
        ]
        for nid, nd in nodes.items()
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
    m.U_vent = Param(initialize=3)
    m.M_temp = Param(initialize=100)
    m.M_hum  = Param(initialize=100)
 
    # ── Decision variables: all nodes including root ──────────────────────────
    m.Heat   = Var(m.RN,   domain=NonNegativeReals,
                   bounds=(0, d['heating_max_power']))
    m.Vent   = Var(m.N,    domain=Binary)
    m.Vstart = Var(m.N,    domain=Binary)
 
    # ── Auxiliary binaries: scenario nodes only (overrule acts on T_in) ───────
    m.y_low  = Var(m.RN_scen, domain=Binary)
    m.y_ok   = Var(m.RN_scen, domain=Binary)
    m.y_high = Var(m.RN_scen, domain=Binary)
    m.u      = Var(m.RN_scen, domain=Binary)
 
    # ── State variables: scenario nodes only (root pinned to Tinit/Hinit) ─────
    m.T_in = Var(m.RN_scen, domain=NonNegativeReals)
    m.Hum  = Var(m.N_scen,  domain=NonNegativeReals)
 
    # =========================================================================
    # ── Temperature dynamics ──────────────────────────────────────────────────
    # Mirrors 2-stage exactly:
    #   stage-1 node: T_in[r,nid] = Tinit[r] + f(Heat[r,0], Vent[0], Tout[nid])
    #   stage-k node: T_in[r,nid] = T_in[r,pid] + f(Heat[r,pid], Vent[pid], ...)
    # The root's T_in is NOT a variable — it is always Tinit (fixed parameter).
    # =========================================================================
    def temp_dynamics(m, r, nid):
        nd      = nodes[nid]
        pid     = nd['parent']   # always exists since nid is in N_scen
        r_other = 2 if r == 1 else 1
 
        if nd['stage'] == 1:
            # Parent is root: use fixed Tinit as the starting temperature,
            # apply the ROOT's decision (Heat[r,0], Vent[0])
            T_par       = m.Tinit[r]
            T_par_other = m.Tinit[r_other]
            H_par_dec   = m.Heat[r, 0]
            V_par_dec   = m.Vent[0]
            Tout_par    = m.Tout[0]   # outdoor temp at root stage
            Occ_par     = m.O1[0] if r == 1 else m.O2[0]
        else:
            # Parent is a scenario node: use its T_in as starting temperature,
            # apply the PARENT's decision
            T_par       = m.T_in[r, pid]
            T_par_other = m.T_in[r_other, pid]
            H_par_dec   = m.Heat[r, pid]
            V_par_dec   = m.Vent[pid]
            Tout_par    = m.Tout[pid]  # outdoor temp at parent stage
            Occ_par     = m.O1[pid] if r == 1 else m.O2[pid]
 
        return m.T_in[r, nid] == (
            T_par
            + m.Zexch * (T_par_other - T_par)
            + m.Zloss * (Tout_par    - T_par)
            + m.Zconv * H_par_dec
            - m.Zcool * V_par_dec
            + m.Zocc  * Occ_par
        )
    m.TempDyn = Constraint(m.RN_scen, rule=temp_dynamics)
 
    # =========================================================================
    # ── Humidity dynamics ─────────────────────────────────────────────────────
    # stage-1: Hum[nid] = Hinit + f(Vent[0])   — root's vent decision
    # stage-k: Hum[nid] = Hum[pid] + f(Vent[pid])
    # =========================================================================
    def hum_dynamics(m, nid):
        nd  = nodes[nid]
        pid = nd['parent']
 
        if nd['stage'] == 1:
            H_par     = m.Hinit
            V_par_dec = m.Vent[0]
            Occ1_par  = m.O1[0]
            Occ2_par  = m.O2[0]
        else:
            H_par     = m.Hum[pid]
            V_par_dec = m.Vent[pid]
            Occ1_par  = m.O1[pid]
            Occ2_par  = m.O2[pid]
 
        return m.Hum[nid] == (
            H_par
            - m.Hvent * V_par_dec
            + m.Hocc  * (Occ1_par + Occ2_par)
        )
    m.HumDyn = Constraint(m.N_scen, rule=hum_dynamics)
 
    # =========================================================================
    # ── Overrule controller: HIGH temperature ────────────────────────────────
    # Acts on scenario nodes only (T_in is defined there)
    # =========================================================================
    m.CThigh1 = Constraint(m.RN_scen,
        rule=lambda m, r, nid:
            m.T_in[r, nid] >= m.Thigh - m.M_temp * (1 - m.y_high[r, nid]))
    m.CThigh2 = Constraint(m.RN_scen,
        rule=lambda m, r, nid:
            m.T_in[r, nid] <= m.Thigh + m.M_temp * m.y_high[r, nid])
    m.CHeatOff = Constraint(m.RN_scen,
        rule=lambda m, r, nid:
            m.Heat[r, nid] <= m.Pr * (1 - m.y_high[r, nid]))
 
    # =========================================================================
    # ── Overrule controller: LOW temperature ─────────────────────────────────
    # =========================================================================
    m.CTlow1 = Constraint(m.RN_scen,
        rule=lambda m, r, nid:
            m.T_in[r, nid] <= m.Tmin + m.M_temp * (1 - m.y_low[r, nid]))
    m.CTlow2 = Constraint(m.RN_scen,
        rule=lambda m, r, nid:
            m.T_in[r, nid] >= m.Tmin - m.M_temp * m.y_low[r, nid])
 
    # =========================================================================
    # ── Overrule controller: temperature OK ───────────────────────────────────
    # =========================================================================
    m.CTok1 = Constraint(m.RN_scen,
        rule=lambda m, r, nid:
            m.T_in[r, nid] >= m.Tok - m.M_temp * (1 - m.y_ok[r, nid]))
    m.CTok2 = Constraint(m.RN_scen,
        rule=lambda m, r, nid:
            m.T_in[r, nid] <= m.Tok + m.M_temp * m.y_ok[r, nid])
 
    # =========================================================================
    # ── Overrule memory (u): scenario nodes only ──────────────────────────────
    # stage-1 nodes: u_prev = real low_override from state
    # stage-k nodes: u_prev = m.u[r, pid]
    # =========================================================================
    def get_u_prev(r, nid):
        nd = nodes[nid]
        if nd['stage'] == 1:
            return state[f'low_override_r{r}']
        return m.u[r, nd['parent']]
 
    m.CU1 = Constraint(m.RN_scen,
        rule=lambda m, r, nid: m.u[r, nid] >= m.y_low[r, nid])
 
    def c_u2(m, r, nid):
        return m.u[r, nid] <= get_u_prev(r, nid) + m.y_low[r, nid]
    m.CU2 = Constraint(m.RN_scen, rule=c_u2)
 
    m.CHeatMax = Constraint(m.RN_scen,
        rule=lambda m, r, nid: m.Heat[r, nid] >= m.Pr * m.u[r, nid])
 
    def c_u3(m, r, nid):
        return m.u[r, nid] >= get_u_prev(r, nid) - m.y_ok[r, nid]
    m.CU3 = Constraint(m.RN_scen, rule=c_u3)
 
    m.CU4 = Constraint(m.RN_scen,
        rule=lambda m, r, nid: m.u[r, nid] <= 1 - m.y_ok[r, nid])
 
    # =========================================================================
    # ── Ventilation: startup signal (all nodes) ───────────────────────────────
    # Root: v_prev = VentInit
    # Scenario nodes: v_prev = Vent[pid]
    # =========================================================================
    def get_v_prev(nid):
        nd = nodes[nid]
        if nd['stage'] == 0:
            return m.VentInit
        return m.Vent[nd['parent']]
 
    def c_vstart1(m, nid):
        return m.Vstart[nid] >= m.Vent[nid] - get_v_prev(nid)
    m.CVstart1 = Constraint(m.N, rule=c_vstart1)
 
    m.CVstart2 = Constraint(m.N,
        rule=lambda m, nid: m.Vstart[nid] <= m.Vent[nid])
 
    def c_vstart3(m, nid):
        return m.Vstart[nid] <= 1 - get_v_prev(nid)
    m.CVstart3 = Constraint(m.N, rule=c_vstart3)
 
    # =========================================================================
    # ── Ventilation: minimum uptime (all nodes) ───────────────────────────────
    # =========================================================================
    def descendants_up_to_depth(start_nid, max_depth):
        result   = []
        frontier = [(start_nid, 0)]
        while frontier:
            cur, depth = frontier.pop()
            for ch in children[cur]:
                if depth + 1 <= max_depth:
                    result.append(ch)
                    frontier.append((ch, depth + 1))
        return result
 
    all_node_set = set(all_nodes)
    def min_uptime(m, nid):
        desc = descendants_up_to_depth(nid, int(value(m.U_vent)) - 1)
        if not desc:
            return Constraint.Skip
        return (sum(m.Vent[d] for d in desc if d in all_node_set)
                >= (int(value(m.U_vent)) - 1) * m.Vstart[nid])
    m.MinVentOn = Constraint(m.N, rule=min_uptime)
 
    # =========================================================================
    # ── Ventilation: humidity overrule (scenario nodes) ───────────────────────
    # =========================================================================
    m.CVentHum = Constraint(m.N_scen,
        rule=lambda m, nid:
            m.Hum[nid] <= m.Hhigh + m.M_hum * m.Vent[nid])
 
    # =========================================================================
    # ── Objective ─────────────────────────────────────────────────────────────
    # Root (nid=0): cost of here-and-now decision at real price, weight=1.0
    # Scenario nodes: expected future cost weighted by joint probability
    # =========================================================================
    def objective(m):
        return sum(
            m.pi[nid] * m.price[nid] * (
                sum(m.Heat[r, nid] for r in m.R)
                + m.Vent[nid] * m.Pvent
            )
            for nid in m.N
        )
    m.obj = Objective(rule=objective, sense=minimize)
 
    return m
 


# =============================================================================
# STAGE 2 – PYOMO MILP OVER THE SCENARIO TREE
# =============================================================================
 
# This will replace build_sp_model in multiSP_policy.py
 
# def build_sp_model(state, nodes, children, leaves):
#     """
#     Build the multi-stage stochastic MILP indexed over the scenario tree.
 
#     Root node (nid=0) = current observed state → decision executed HERE.
#     Stage-1..HORIZON nodes = sampled futures → recourse decisions.
 
#     Non-anticipativity is structural: each node appears once, so all paths
#     through a node share its decision automatically. No NAC constraints needed.
 
#     Dynamics propagate parent → child:
#         T_in[r, nid]  = f(T_in[r, parent], Heat[r, parent], Vent[parent], ...)
#         Hum[nid]      = f(Hum[parent], Vent[parent], ...)
#     Stage-1 nodes use Tinit/Hinit as the parent state (real observed values).
 
#     Objective
#     ---------
#         min  Σ_{nid ∈ N_dec}  π[nid] · price[nid] · cost[nid]
 
#     Root has π[0]=1.0 and uses the real observed price.
#     Each stage>=1 node's π[nid] is its joint probability along the tree path.
 
#     Parameters
#     ----------
#     state    : dict   – Environment.py state dict
#     nodes    : dict   – {nid: node_dict} from grow_scenario_tree()
#     children : dict   – {nid: [child_nid, ...]}
#     leaves   : list   – leaf nid list
 
#     Returns
#     -------
#     Pyomo ConcreteModel (unsolved)
#     """
#     d              = DATA
#     _num_timeslots = int(d['num_timeslots'])
 
#     # ── Derived node sets ─────────────────────────────────────────────────────
#     # ALL nodes have decisions (root included)
#     dec_nodes = list(nodes.keys())                              # stage 0..HORIZON
#     int_nodes = [nid for nid in dec_nodes if nid not in set(leaves)]  # non-leaves
#     leaf_set  = set(leaves)
 
#     # ── Helper: path from a node back to root (inclusive) ────────────────────
#     def path_to_root(nid):
#         path = []
#         cur  = nid
#         while cur is not None:
#             path.append(cur)
#             cur = nodes[cur]['parent']
#         return path
 
#     m = ConcreteModel()
 
#     # ── Sets ──────────────────────────────────────────────────────────────────
#     m.R      = Set(initialize=[1, 2])
#     m.N      = Set(initialize=dec_nodes)        # all nodes (root + scenario tree)
#     m.N_int  = Set(initialize=int_nodes)        # non-leaf nodes
#     m.N_leaf = Set(initialize=leaves)
#     m.RN     = m.R * m.N
 
#     # ── Current real state ────────────────────────────────────────────────────
#     v_prev = 1 if state['vent_counter'] > 0 else 0
#     vc     = int(state['vent_counter'])
 
#     # ── State parameters (initial conditions carried into dynamics) ───────────
#     m.Tinit    = Param(m.R, initialize={1: state['T1'], 2: state['T2']})
#     m.Hinit    = Param(initialize=state['H'])
#     m.VentInit = Param(initialize=v_prev)   # vent status BEFORE this timestep
#     m.VCinit   = Param(initialize=vc)
 
#     # ── Exogenous params: root uses real observed values ──────────────────────
#     # Build price/occ/prob dicts; root (nid=0) anchored to real state
#     price_init = {nid: nd['price'] for nid, nd in nodes.items()}
#     occ1_init  = {nid: nd['occ1']  for nid, nd in nodes.items()}
#     occ2_init  = {nid: nd['occ2']  for nid, nd in nodes.items()}
#     prob_init  = {nid: nd['prob']  for nid, nd in nodes.items()}
 
#     # Override root with real observed values
#     price_init[0] = state['price_t']
#     occ1_init[0]  = state['Occ1']
#     occ2_init[0]  = state['Occ2']
#     prob_init[0]  = 1.0
 
#     m.price = Param(m.N, initialize=price_init)
#     m.O1    = Param(m.N, initialize=occ1_init)
#     m.O2    = Param(m.N, initialize=occ2_init)
#     m.pi    = Param(m.N, initialize=prob_init)
 
#     # ── Outdoor temperature: stage offset from current_time ───────────────────
#     m.Tout = Param(m.N, initialize={
#         nid: d['outdoor_temperature'][
#             (state['current_time'] + nd['stage']) % _num_timeslots
#         ]
#         for nid, nd in nodes.items()
#     })
 
#     # ── Physical constants ────────────────────────────────────────────────────
#     m.Pr     = Param(initialize=d['heating_max_power'])
#     m.Pvent  = Param(initialize=d['ventilation_power'])
#     m.Zexch  = Param(initialize=d['heat_exchange_coeff'])
#     m.Zconv  = Param(initialize=d['heating_efficiency_coeff'])
#     m.Zloss  = Param(initialize=d['thermal_loss_coeff'])
#     m.Zcool  = Param(initialize=d['heat_vent_coeff'])
#     m.Zocc   = Param(initialize=d['heat_occupancy_coeff'])
#     m.Hocc   = Param(initialize=d['humidity_occupancy_coeff'])
#     m.Hvent  = Param(initialize=d['humidity_vent_coeff'])
#     m.Tmin   = Param(initialize=d['temp_min_comfort_threshold'])
#     m.Tok    = Param(initialize=d['temp_OK_threshold'])
#     m.Thigh  = Param(initialize=d['temp_max_comfort_threshold'])
#     m.Hhigh  = Param(initialize=d['humidity_threshold'])
#     m.U_vent = Param(initialize=3)
#     m.M_temp = Param(initialize=100)
#     m.M_hum  = Param(initialize=100)
 
#     # ── Decision variables (all nodes including root) ─────────────────────────
#     m.Heat   = Var(m.RN,   domain=NonNegativeReals,
#                    bounds=(0, d['heating_max_power']))
#     m.Vent   = Var(m.N,    domain=Binary)
#     m.Vstart = Var(m.N,    domain=Binary)
 
#     # ── Auxiliary binaries ────────────────────────────────────────────────────
#     m.y_low  = Var(m.RN, domain=Binary)
#     m.y_ok   = Var(m.RN, domain=Binary)
#     m.y_high = Var(m.RN, domain=Binary)
#     m.u      = Var(m.RN, domain=Binary)
 
#     # ── State variables (resulting temperature/humidity after each decision) ──
#     m.T_in = Var(m.RN, domain=NonNegativeReals)
#     m.Hum  = Var(m.N,  domain=NonNegativeReals)
 
#     # =========================================================================
#     # ── Temperature dynamics ──────────────────────────────────────────────────
#     # T_in[r, nid] = temperature RESULTING from applying Heat/Vent at nid.
#     # Stage-1 nodes: parent state = Tinit (real observed temperature).
#     # Stage>=2 nodes: parent state = T_in[r, pid] (result of parent's decision).
#     # Root (stage 0): T_in[r, 0] results from applying Heat[r,0]/Vent[0] to Tinit.
#     # =========================================================================
#     def temp_dynamics(m, r, nid):
#         nd      = nodes[nid]
#         pid     = nd['parent']
#         r_other = 2 if r == 1 else 1
 
#         if pid is None:
#             # Root node: parent state is real observed Tinit
#             T_par      = m.Tinit[r]
#             T_par_other= m.Tinit[r_other]
#         else:
#             # All other nodes: parent state is the result of parent's decision
#             T_par      = m.T_in[r, pid]
#             T_par_other= m.T_in[r_other, pid]
 
#         return m.T_in[r, nid] == (
#             T_par
#             + m.Zexch * (T_par_other - T_par)
#             + m.Zloss * (m.Tout[nid]  - T_par)
#             + m.Zconv * m.Heat[r, nid]
#             - m.Zcool * m.Vent[nid]
#             + m.Zocc  * (m.O1[nid] if r == 1 else m.O2[nid])
#         )
#     m.TempDyn = Constraint(m.RN, rule=temp_dynamics)
 
#     # =========================================================================
#     # ── Humidity dynamics ─────────────────────────────────────────────────────
#     # Hum[nid] results from applying Vent[nid] to parent humidity.
#     # Root: parent humidity = Hinit.
#     # =========================================================================
#     def hum_dynamics(m, nid):
#         nd  = nodes[nid]
#         pid = nd['parent']
#         H_par = m.Hinit if pid is None else m.Hum[pid]
 
#         return m.Hum[nid] == (
#             H_par
#             - m.Hvent * m.Vent[nid]
#             + m.Hocc  * (m.O1[nid] + m.O2[nid])
#         )
#     m.HumDyn = Constraint(m.N, rule=hum_dynamics)
 
#     # =========================================================================
#     # ── Overrule controller: HIGH temperature ────────────────────────────────
#     # =========================================================================
#     m.CThigh1 = Constraint(m.RN,
#         rule=lambda m, r, nid:
#             m.T_in[r, nid] >= m.Thigh - m.M_temp * (1 - m.y_high[r, nid]))
#     m.CThigh2 = Constraint(m.RN,
#         rule=lambda m, r, nid:
#             m.T_in[r, nid] <= m.Thigh + m.M_temp * m.y_high[r, nid])
#     m.CHeatOff = Constraint(m.RN,
#         rule=lambda m, r, nid:
#             m.Heat[r, nid] <= m.Pr * (1 - m.y_high[r, nid]))
 
#     # =========================================================================
#     # ── Overrule controller: LOW temperature ─────────────────────────────────
#     # =========================================================================
#     m.CTlow1 = Constraint(m.RN,
#         rule=lambda m, r, nid:
#             m.T_in[r, nid] <= m.Tmin + m.M_temp * (1 - m.y_low[r, nid]))
#     m.CTlow2 = Constraint(m.RN,
#         rule=lambda m, r, nid:
#             m.T_in[r, nid] >= m.Tmin - m.M_temp * m.y_low[r, nid])
 
#     # =========================================================================
#     # ── Overrule controller: temperature OK ───────────────────────────────────
#     # =========================================================================
#     m.CTok1 = Constraint(m.RN,
#         rule=lambda m, r, nid:
#             m.T_in[r, nid] >= m.Tok - m.M_temp * (1 - m.y_ok[r, nid]))
#     m.CTok2 = Constraint(m.RN,
#         rule=lambda m, r, nid:
#             m.T_in[r, nid] <= m.Tok + m.M_temp * m.y_ok[r, nid])
 
#     # =========================================================================
#     # ── Overrule memory (u) ───────────────────────────────────────────────────
#     # Root node reads low_override directly from state.
#     # All other nodes read u_prev from their parent's u variable.
#     # =========================================================================
#     def get_u_prev(r, nid):
#         nd  = nodes[nid]
#         pid = nd['parent']
#         if pid is None:
#             # Root: u_prev is the real override state from environment
#             return state[f'low_override_r{r}']
#         else:
#             return m.u[r, pid]
 
#     m.CU1 = Constraint(m.RN,
#         rule=lambda m, r, nid: m.u[r, nid] >= m.y_low[r, nid])
 
#     def c_u2(m, r, nid):
#         return m.u[r, nid] <= get_u_prev(r, nid) + m.y_low[r, nid]
#     m.CU2 = Constraint(m.RN, rule=c_u2)
 
#     m.CHeatMax = Constraint(m.RN,
#         rule=lambda m, r, nid: m.Heat[r, nid] >= m.Pr * m.u[r, nid])
 
#     def c_u3(m, r, nid):
#         return m.u[r, nid] >= get_u_prev(r, nid) - m.y_ok[r, nid]
#     m.CU3 = Constraint(m.RN, rule=c_u3)
 
#     m.CU4 = Constraint(m.RN,
#         rule=lambda m, r, nid: m.u[r, nid] <= 1 - m.y_ok[r, nid])
 
#     # =========================================================================
#     # ── Ventilation: startup signal ───────────────────────────────────────────
#     # Root node: v_prev = VentInit (real vent status before this timestep).
#     # All other nodes: v_prev = Vent[pid].
#     # =========================================================================
#     def get_v_prev(nid):
#         nd  = nodes[nid]
#         pid = nd['parent']
#         return m.VentInit if pid is None else m.Vent[pid]
 
#     def c_vstart1(m, nid):
#         return m.Vstart[nid] >= m.Vent[nid] - get_v_prev(nid)
#     m.CVstart1 = Constraint(m.N, rule=c_vstart1)
 
#     m.CVstart2 = Constraint(m.N,
#         rule=lambda m, nid: m.Vstart[nid] <= m.Vent[nid])
 
#     def c_vstart3(m, nid):
#         return m.Vstart[nid] <= 1 - get_v_prev(nid)
#     m.CVstart3 = Constraint(m.N, rule=c_vstart3)
 
#     # =========================================================================
#     # ── Ventilation: minimum uptime ───────────────────────────────────────────
#     # =========================================================================
#     def descendants_up_to_depth(start_nid, max_depth):
#         result   = []
#         frontier = [(start_nid, 0)]
#         while frontier:
#             cur, depth = frontier.pop()
#             for ch in children[cur]:
#                 if depth + 1 <= max_depth:
#                     result.append(ch)
#                     frontier.append((ch, depth + 1))
#         return result
 
#     dec_set = set(dec_nodes)
#     def min_uptime(m, nid):
#         desc = descendants_up_to_depth(nid, int(value(m.U_vent)) - 1)
#         if not desc:
#             return Constraint.Skip
#         return (sum(m.Vent[d] for d in desc if d in dec_set)
#                 >= (int(value(m.U_vent)) - 1) * m.Vstart[nid])
#     m.MinVentOn = Constraint(m.N, rule=min_uptime)
 
#     # =========================================================================
#     # ── Ventilation: humidity overrule ────────────────────────────────────────
#     # =========================================================================
#     m.CVentHum = Constraint(m.N,
#         rule=lambda m, nid:
#             m.Hum[nid] <= m.Hhigh + m.M_hum * m.Vent[nid])
 
#     # =========================================================================
#     # ── Objective ─────────────────────────────────────────────────────────────
#     # Root (nid=0): π=1.0, real observed price — cost of the decision we execute.
#     # Stage>=1 nodes: π[nid] = joint probability, scenario price.
#     # =========================================================================
#     def objective(m):
#         return sum(
#             m.pi[nid] * m.price[nid] * (
#                 sum(m.Heat[r, nid] for r in m.R)
#                 + m.Vent[nid] * m.Pvent
#             )
#             for nid in m.N
#         )
#     m.obj = Objective(rule=objective, sense=minimize)
 
#     return m
 

 
# =============================================================================
# STAGE 3 – MULTI-STAGE SP POLICY FUNCTION
# =============================================================================
 
def multiSP_policy(state):
    """
    Multi-stage stochastic programming policy.
 
    Reads the current Environment.py state dict, builds the scenario tree,
    solves the multi-stage stochastic MILP, and returns the stage-1 (here-and-
    now) decisions — the decisions at the direct children of the root node.
    All scenarios that share the same stage-1 node automatically share the
    same decision (non-anticipativity via tree structure).
 
    Parameters
    ----------
    state : dict  – Environment.py keys:
        T1, T2, H, Occ1, Occ2, price_t, price_previous,
        vent_counter, low_override_r1, low_override_r2, current_time
 
    Returns
    -------
    dict with keys 'HeatPowerRoom1', 'HeatPowerRoom2', 'VentilationON'
    """
    t         = state['current_time']
    remaining = DATA['num_timeslots'] - t
    horizon   = min(HORIZON, remaining)
 
    # Only skip if the episode is already over (should never happen in normal
    # operation — Environment.py stops calling select_action at end of day).
    if remaining <= 0:
        return {'HeatPowerRoom1': 0.0, 'HeatPowerRoom2': 0.0, 'VentilationON': 0}
 
    # Trim branching factors to the actual horizon length
    branching = BRANCHING_FACTORS[:horizon]
 
    # ── Build scenario tree ───────────────────────────────────────────────────
    nodes, children, leaves = grow_scenario_tree(
        price_now         = state['price_t'],
        price_prev        = state['price_previous'],
        occ1_now          = state['Occ1'],
        occ2_now          = state['Occ2'],
        horizon           = horizon,
        branching_factors = branching,
    )

    # print(f"t={state['current_time']} | "
    #   f"T1={state['T1']:.2f} T2={state['T2']:.2f} | "
    #   f"H={state['H']:.2f} | "
    #   f"vent_counter={state['vent_counter']} | "
    #   f"override_r1={state['low_override_r1']} override_r2={state['low_override_r2']}")
 
    # ── Build and solve MILP ──────────────────────────────────────────────────
    model  = build_sp_model(state, nodes, children, leaves)
    solver = SolverFactory('gurobi_direct')
    solver.options['TimeLimit'] = 12
    solver.options['MIPGap']    = 0.02
    result = solver.solve(model, tee=False)
 
    if result.solver.termination_condition not in (
        TerminationCondition.optimal,
        TerminationCondition.feasible,
    ):
        warnings.warn(
            f"Gurobi failed at t={t} "
            f"({result.solver.termination_condition}). Falling back to zero.",
            RuntimeWarning,
        )
        return {'HeatPowerRoom1': 0.0, 'HeatPowerRoom2': 0.0, 'VentilationON': 0}
 
    # ── Extract here-and-now decision ─────────────────────────────────────────
    # The root node (nid=0) IS the here-and-now decision.
    # Heat[r, 0] and Vent[0] are the actions to execute this timestep.
    p1 = float(value(model.Heat[1, 0]))
    p2 = float(value(model.Heat[2, 0]))
    v  = int(round(float(value(model.Vent[0]))))

    # print(f"  → H1={p1:.2f} H2={p2:.2f} V={v} | "
    #   f"T_in1={value(model.T_in[1,0]):.2f} T_in2={value(model.T_in[2,0]):.2f} | "
    #   f"Hum={value(model.Hum[0]):.2f} | "
    #   f"u_r1={value(model.u[1,0])} u_r2={value(model.u[2,0])}")
 
    return {'HeatPowerRoom1': p1, 'HeatPowerRoom2': p2, 'VentilationON': v}
 
 
# =============================================================================
# STAGE 4 – GRADER-COMPATIBLE WRAPPER
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
    return multiSP_policy(state) 
 
# # =============================================================================
# # STAGE 2 – PYOMO MILP OVER THE SCENARIO TREE
# # =============================================================================
 
# def build_sp_model(state, nodes, children, leaves):
#     """
#     Build the multi-stage stochastic MILP indexed over the scenario tree.
 
#     Key difference from 2-stage SP
#     --------------------------------
#     Variables and constraints are indexed by node id (nid), not by (t, s).
#     Non-anticipativity is structural: every node appears exactly once, so
#     decisions at the same node are automatically identical across all paths
#     that pass through it.  No explicit NAC constraints are needed.
 
#     Dynamics propagate parent → child:
#         T_in[r, nid]  defined by T_in[r, parent(nid)] and Heat[r, parent(nid)]
#         Hum[nid]      defined by Hum[parent(nid)]      and Vent[parent(nid)]
 
#     Objective
#     ---------
#     Minimise expected electricity cost accumulated along every root-to-leaf
#     path, weighted by leaf probability:
 
#         min  Σ_{leaf ∈ leaves}  π_leaf ·
#                 Σ_{nid on path root→leaf, nid ≥ stage-1}
#                     price[nid] · (Heat[1,nid] + Heat[2,nid] + Vent[nid]·Pvent)
 
#     Because each internal node nid appears in exactly (# descendant leaves)
#     paths, its cost is weighted by the sum of those leaf probabilities, which
#     equals π(nid) itself.  We therefore equivalently write:
 
#         min  Σ_{nid, stage ≥ 1}  π[nid] · price[nid] · cost[nid]
 
#     which is what we implement (cheaper to build, same result).
 
#     Parameters
#     ----------
#     state    : dict   – Environment.py state dict
#     nodes    : dict   – {nid: node_dict} from grow_scenario_tree()
#     children : dict   – {nid: [child_nid, ...]}
#     leaves   : list   – leaf nid list
 
#     Returns
#     -------
#     Pyomo ConcreteModel (unsolved)
#     """
#     d              = DATA
#     _num_timeslots = int(d['num_timeslots'])
 
#     # ── Derived node sets ─────────────────────────────────────────────────────
#     # Decision nodes: all nodes at stage >= 1 (root has no decision variables)
#     dec_nodes  = [nid for nid, nd in nodes.items() if nd['stage'] >= 1]
#     # Internal nodes: non-leaf decision nodes (have children)
#     int_nodes  = [nid for nid in dec_nodes if nid not in leaves]
#     leaf_set   = set(leaves)
 
#     # ── Helper: path from a node back to root (inclusive, excluding root=0) ──
#     def path_to_root(nid):
#         """Return list of decision-stage ancestors [nid, ..., stage-1 node],
#         stopping before the root (stage 0) which carries no decision."""
#         path = []
#         cur  = nid
#         while cur is not None and nodes[cur]['stage'] >= 1:
#             path.append(cur)
#             cur = nodes[cur]['parent']
#         return path
 
#     m = ConcreteModel()
 
#     # ── Sets ──────────────────────────────────────────────────────────────────
#     m.R        = Set(initialize=[1, 2])
#     m.N        = Set(initialize=list(nodes.keys()))        # all nodes
#     m.N_dec    = Set(initialize=dec_nodes)                 # stage >= 1 (have decisions)
#     m.N_int    = Set(initialize=int_nodes)                 # internal dec nodes
#     m.N_leaf   = Set(initialize=leaves)                    # leaf nodes
#     m.RN_dec   = m.R * m.N_dec
 
#     # ── Current real state ────────────────────────────────────────────────────
#     v_prev = 1 if state['vent_counter'] > 0 else 0
#     vc     = int(state['vent_counter'])
 
#     # ── State parameters (initial conditions) ─────────────────────────────────
#     m.Tinit    = Param(m.R, initialize={1: state['T1'], 2: state['T2']})
#     m.Hinit    = Param(initialize=state['H'])
#     m.VentInit = Param(initialize=v_prev)
#     m.VCinit   = Param(initialize=vc)   # vent_counter: steps vent has been ON
 
#     # ── Exogenous scenario parameters (indexed over ALL nodes) ────────────────
#     m.price  = Param(m.N, initialize={nid: nd['price'] for nid, nd in nodes.items()})
#     m.O1     = Param(m.N, initialize={nid: nd['occ1']  for nid, nd in nodes.items()})
#     m.O2     = Param(m.N, initialize={nid: nd['occ2']  for nid, nd in nodes.items()})
#     m.pi     = Param(m.N, initialize={nid: nd['prob']  for nid, nd in nodes.items()})
 
#     # ── Outdoor temperature: indexed by stage (same as 2-stage t-index) ───────
#     m.Tout = Param(m.N, initialize={
#         nid: d['outdoor_temperature'][
#             (state['current_time'] + nd['stage']) % _num_timeslots
#         ]
#         for nid, nd in nodes.items()
#     })
 
#     # ── Physical constants (identical to 2-stage SP) ──────────────────────────
#     m.Pr     = Param(initialize=d['heating_max_power'])
#     m.Pvent  = Param(initialize=d['ventilation_power'])
#     m.Zexch  = Param(initialize=d['heat_exchange_coeff'])
#     m.Zconv  = Param(initialize=d['heating_efficiency_coeff'])
#     m.Zloss  = Param(initialize=d['thermal_loss_coeff'])
#     m.Zcool  = Param(initialize=d['heat_vent_coeff'])
#     m.Zocc   = Param(initialize=d['heat_occupancy_coeff'])
#     m.Hocc   = Param(initialize=d['humidity_occupancy_coeff'])
#     m.Hvent  = Param(initialize=d['humidity_vent_coeff'])
#     m.Tmin   = Param(initialize=d['temp_min_comfort_threshold'])
#     m.Tok    = Param(initialize=d['temp_OK_threshold'])
#     m.Thigh  = Param(initialize=d['temp_max_comfort_threshold'])
#     m.Hhigh  = Param(initialize=d['humidity_threshold'])
#     m.U_vent = Param(initialize=3)    # minimum ventilation uptime (steps)
#     m.M_temp = Param(initialize=100)
#     m.M_hum  = Param(initialize=100)
 
#     # ── Decision variables (one per decision node) ────────────────────────────
#     m.Heat   = Var(m.RN_dec, domain=NonNegativeReals,
#                    bounds=(0, d['heating_max_power']))
#     m.Vent   = Var(m.N_dec,  domain=Binary)
#     m.Vstart = Var(m.N_dec,  domain=Binary)   # 1 if ventilation starts at this node
 
#     # ── Auxiliary binaries for overrule controller ────────────────────────────
#     m.y_low  = Var(m.RN_dec, domain=Binary)   # 1 if T < Tmin
#     m.y_ok   = Var(m.RN_dec, domain=Binary)   # 1 if T >= Tok  (overrule deactivation)
#     m.y_high = Var(m.RN_dec, domain=Binary)   # 1 if T > Thigh (forced off)
#     m.u      = Var(m.RN_dec, domain=Binary)   # overrule memory (heating forced on)
 
#     # ── State variables ───────────────────────────────────────────────────────
#     m.T_in = Var(m.RN_dec, domain=NonNegativeReals)
#     m.Hum  = Var(m.N_dec,  domain=NonNegativeReals)
 
#     # =========================================================================
#     # ── Temperature dynamics ──────────────────────────────────────────────────
#     # =========================================================================
#     def temp_dynamics(m, r, nid):
#         nd    = nodes[nid]
#         pid   = nd['parent']
#         stage = nd['stage']
 
#         if stage == 1:
#             # Parent is root: use real initial temperature
#             return m.T_in[r, nid] == (
#                 m.Tinit[r]
#                 + m.Zexch * (m.Tinit[3 - r]       - m.Tinit[r])
#                 + m.Zloss * (m.Tout[nid]            - m.Tinit[r])
#                 + m.Zconv * m.Heat[r, nid]
#                 - m.Zcool * m.Vent[nid]
#                 + m.Zocc  * (m.O1[nid] if r == 1 else m.O2[nid])
#             )
#         else:
#             # Parent is a decision node: propagate from parent's state
#             r_other = 2 if r == 1 else 1
#             return m.T_in[r, nid] == (
#                 m.T_in[r, pid]
#                 + m.Zexch * (m.T_in[r_other, pid] - m.T_in[r, pid])
#                 + m.Zloss * (m.Tout[nid]            - m.T_in[r, pid])
#                 + m.Zconv * m.Heat[r, pid]
#                 - m.Zcool * m.Vent[pid]
#                 + m.Zocc  * (m.O1[nid] if r == 1 else m.O2[nid])
#             )
#     m.TempDyn = Constraint(m.RN_dec, rule=temp_dynamics)
 
#     # =========================================================================
#     # ── Humidity dynamics ─────────────────────────────────────────────────────
#     # =========================================================================
#     def hum_dynamics(m, nid):
#         nd    = nodes[nid]
#         pid   = nd['parent']
#         stage = nd['stage']
 
#         if stage == 1:
#             return m.Hum[nid] == (
#                 m.Hinit
#                 - m.Hvent * m.Vent[nid]
#                 + m.Hocc  * (m.O1[nid] + m.O2[nid])
#             )
#         else:
#             return m.Hum[nid] == (
#                 m.Hum[pid]
#                 - m.Hvent * m.Vent[pid]
#                 + m.Hocc  * (m.O1[nid] + m.O2[nid])
#             )
#     m.HumDyn = Constraint(m.N_dec, rule=hum_dynamics)
 
#     # =========================================================================
#     # ── Overrule controller: HIGH temperature (forced heating shutdown) ────────
#     # =========================================================================
#     m.CThigh1 = Constraint(m.RN_dec,
#         rule=lambda m, r, nid:
#             m.T_in[r, nid] >= m.Thigh - m.M_temp * (1 - m.y_high[r, nid]))
#     m.CThigh2 = Constraint(m.RN_dec,
#         rule=lambda m, r, nid:
#             m.T_in[r, nid] <= m.Thigh + m.M_temp * m.y_high[r, nid])
#     m.CHeatOff = Constraint(m.RN_dec,
#         rule=lambda m, r, nid:
#             m.Heat[r, nid] <= m.Pr * (1 - m.y_high[r, nid]))
 
#     # =========================================================================
#     # ── Overrule controller: LOW temperature (forced heating on) ──────────────
#     # =========================================================================
#     m.CTlow1 = Constraint(m.RN_dec,
#         rule=lambda m, r, nid:
#             m.T_in[r, nid] <= m.Tmin + m.M_temp * (1 - m.y_low[r, nid]))
#     m.CTlow2 = Constraint(m.RN_dec,
#         rule=lambda m, r, nid:
#             m.T_in[r, nid] >= m.Tmin - m.M_temp * m.y_low[r, nid])
 
#     # =========================================================================
#     # ── Overrule controller: temperature OK (overrule deactivation) ───────────
#     # =========================================================================
#     m.CTok1 = Constraint(m.RN_dec,
#         rule=lambda m, r, nid:
#             m.T_in[r, nid] >= m.Tok - m.M_temp * (1 - m.y_ok[r, nid]))
#     m.CTok2 = Constraint(m.RN_dec,
#         rule=lambda m, r, nid:
#             m.T_in[r, nid] <= m.Tok + m.M_temp * m.y_ok[r, nid])
 
#     # =========================================================================
#     # ── Overrule memory (u) ───────────────────────────────────────────────────
#     # u = 1 means heating is forced on (overrule active)
#     # =========================================================================
 
#     # u >= y_low  (overrule activates as soon as T < Tmin)
#     m.CU1 = Constraint(m.RN_dec,
#         rule=lambda m, r, nid: m.u[r, nid] >= m.y_low[r, nid])
 
#     # u <= u_prev + y_low  (overrule can only be ON if it was ON or T just went low)
#     def c_u2(m, r, nid):
#         nd  = nodes[nid]
#         pid = nd['parent']
#         u_prev = state[f'low_override_r{r}'] if nd['stage'] == 1 else m.u[r, pid]
#         return m.u[r, nid] <= u_prev + m.y_low[r, nid]
#     m.CU2 = Constraint(m.RN_dec, rule=c_u2)
 
#     # Heat >= Pr * u  (overrule forces full heating)
#     m.CHeatMax = Constraint(m.RN_dec,
#         rule=lambda m, r, nid: m.Heat[r, nid] >= m.Pr * m.u[r, nid])
 
#     # u >= u_prev - y_ok  (overrule stays ON unless T reaches Tok)
#     def c_u3(m, r, nid):
#         nd  = nodes[nid]
#         pid = nd['parent']
#         u_prev = state[f'low_override_r{r}'] if nd['stage'] == 1 else m.u[r, pid]
#         return m.u[r, nid] >= u_prev - m.y_ok[r, nid]
#     m.CU3 = Constraint(m.RN_dec, rule=c_u3)
 
#     # u <= 1 - y_ok  (overrule deactivates as soon as T >= Tok)
#     m.CU4 = Constraint(m.RN_dec,
#         rule=lambda m, r, nid: m.u[r, nid] <= 1 - m.y_ok[r, nid])
 
#     # =========================================================================
#     # ── Ventilation: startup signal ───────────────────────────────────────────
#     # =========================================================================
#     def c_vstart1(m, nid):
#         nd    = nodes[nid]
#         pid   = nd['parent']
#         v_prv = m.VentInit if nd['stage'] == 1 else m.Vent[pid]
#         return m.Vstart[nid] >= m.Vent[nid] - v_prv
#     m.CVstart1 = Constraint(m.N_dec, rule=c_vstart1)
 
#     m.CVstart2 = Constraint(m.N_dec,
#         rule=lambda m, nid: m.Vstart[nid] <= m.Vent[nid])
 
#     def c_vstart3(m, nid):
#         nd    = nodes[nid]
#         pid   = nd['parent']
#         v_prv = m.VentInit if nd['stage'] == 1 else m.Vent[pid]
#         return m.Vstart[nid] <= 1 - v_prv
#     m.CVstart3 = Constraint(m.N_dec, rule=c_vstart3)
 
#     # =========================================================================
#     # ── Ventilation: minimum uptime (>= U_vent consecutive steps after start) ─
#     # =========================================================================
#     # For each decision node nid where ventilation starts (Vstart=1), all
#     # nodes on the unique path from nid to depth U_vent-1 below must have Vent=1.
#     # We enforce this by walking DOWN the tree from nid, following any one
#     # child path (the constraint must hold on every path, so we apply it to
#     # all descendants up to U_vent steps deep).
 
#     def descendants_up_to_depth(start_nid, max_depth):
#         """BFS: all nodes within max_depth steps below start_nid (exclusive of start)."""
#         result = []
#         frontier = [(start_nid, 0)]
#         while frontier:
#             cur, depth = frontier.pop()
#             for ch in children[cur]:
#                 if depth + 1 <= max_depth:
#                     result.append(ch)
#                     frontier.append((ch, depth + 1))
#         return result
 
#     def min_uptime(m, nid):
#         """If ventilation starts at nid, it must stay on for U_vent steps."""
#         desc = descendants_up_to_depth(nid, int(value(m.U_vent)) - 1)
#         if not desc:
#             return Constraint.Skip
#         return sum(m.Vent[d] for d in desc if d in set(dec_nodes)) >= \
#                (int(value(m.U_vent)) - 1) * m.Vstart[nid]
#     m.MinVentOn = Constraint(m.N_dec, rule=min_uptime)
 
#     # =========================================================================
#     # ── Ventilation: humidity overrule ────────────────────────────────────────
#     # =========================================================================
#     m.CVentHum = Constraint(m.N_dec,
#         rule=lambda m, nid:
#             m.Hum[nid] <= m.Hhigh + m.M_hum * m.Vent[nid])
 
#     # =========================================================================
#     # ── Objective: minimise expected electricity cost ─────────────────────────
#     # =========================================================================
#     # Each decision node nid at stage >= 1 contributes its cost weighted by
#     # its joint probability π[nid].  This is equivalent to summing over all
#     # root-to-leaf paths weighted by leaf probability (since π[nid] equals the
#     # sum of probabilities of all leaves below nid, and each leaf path visits
#     # nid exactly once).
#     def objective(m):
#         return sum(
#             m.pi[nid] * m.price[nid] * (
#                 sum(m.Heat[r, nid] for r in m.R)
#                 + m.Vent[nid] * m.Pvent
#             )
#             for nid in m.N_dec
#         )
#     m.obj = Objective(rule=objective, sense=minimize)
 
#     return m

# # =============================================================================
# # STAGE 3 – MULTI-STAGE SP POLICY FUNCTION
# # =============================================================================
 
# def multiSP_policy(state):
#     """
#     Multi-stage stochastic programming policy.
 
#     Reads the current Environment.py state dict, builds the scenario tree,
#     solves the multi-stage stochastic MILP, and returns the stage-1 (here-and-
#     now) decisions — the decisions at the direct children of the root node.
#     All scenarios that share the same stage-1 node automatically share the
#     same decision (non-anticipativity via tree structure).
 
#     Parameters
#     ----------
#     state : dict  – Environment.py keys:
#         T1, T2, H, Occ1, Occ2, price_t, price_previous,
#         vent_counter, low_override_r1, low_override_r2, current_time
 
#     Returns
#     -------
#     dict with keys 'HeatPowerRoom1', 'HeatPowerRoom2', 'VentilationON'
#     """
#     t         = state['current_time']
#     remaining = DATA['num_timeslots'] - t
#     horizon   = min(HORIZON, remaining)
 
#     if remaining <= 0:
#         return {'HeatPowerRoom1': 0.0, 'HeatPowerRoom2': 0.0, 'VentilationON': 0}
 
#     # Trim branching factors to the actual horizon length
#     branching = BRANCHING_FACTORS[:horizon]
 
#     # ── Build scenario tree ───────────────────────────────────────────────────
#     nodes, children, leaves = grow_scenario_tree(
#         price_now         = state['price_t'],
#         price_prev        = state['price_previous'],
#         occ1_now          = state['Occ1'],
#         occ2_now          = state['Occ2'],
#         horizon           = horizon,
#         branching_factors = branching,
#     )
 
#     # ── Build and solve MILP ──────────────────────────────────────────────────
#     model  = build_sp_model(state, nodes, children, leaves)
#     solver = SolverFactory('gurobi_direct')
#     solver.options['TimeLimit'] = 12
#     solver.options['MIPGap']    = 0.02
#     result = solver.solve(model, tee=False)
 
#     if result.solver.termination_condition not in (
#         TerminationCondition.optimal,
#         TerminationCondition.feasible,
#     ):
#         warnings.warn(
#             f"Gurobi failed at t={t} "
#             f"({result.solver.termination_condition}). Falling back to zero.",
#             RuntimeWarning,
#         )
#         return {'HeatPowerRoom1': 0.0, 'HeatPowerRoom2': 0.0, 'VentilationON': 0}
 
#     # ── Extract here-and-now decision ─────────────────────────────────────────
#     # Stage-1 nodes are the direct children of root (nid=0).
#     # All of them share the same exogenous scenario branch at stage 1, but they
#     # ARE different nodes — however, the optimal solution may differ slightly
#     # across them. We take the decision from children[0][0] (the first stage-1
#     # node) as the canonical here-and-now action, consistent with the 2-stage
#     # SP's s=0 convention.
#     first_stage1_nid = children[0][0]
 
#     p1 = float(value(model.Heat[1, first_stage1_nid]))
#     p2 = float(value(model.Heat[2, first_stage1_nid]))
#     v  = int(round(float(value(model.Vent[first_stage1_nid]))))
 
#     return {'HeatPowerRoom1': p1, 'HeatPowerRoom2': p2, 'VentilationON': v}
 
 
# # =============================================================================
# # STAGE 4 – GRADER-COMPATIBLE WRAPPER
# # =============================================================================
 
# def select_action(state):
#     """
#     Wrapper expected by Environment.py.
 
#     Parameters
#     ----------
#     state : dict  – Environment.py state dict
 
#     Returns
#     -------
#     dict  – {'HeatPowerRoom1': float, 'HeatPowerRoom2': float, 'VentilationON': int}
#     """
#     return multiSP_policy(state)