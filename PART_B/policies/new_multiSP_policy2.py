"""
multiSP_policy.py
=================
Multi-Stage Stochastic Programming policy for the restaurant HVAC system.

Compatible with Environment.py:
    from policies.multiSP_policy import select_action

Structure
---------
  Stage 1 (THIS FILE SO FAR):
    - grow_scenario_tree()   : builds the full branching tree via MC sampling
    - _cluster_children()    : clusters MC children of one node → B centroids
    - Shared helpers reused from SP_policy logic (generate_scenarios /
      cluster_scenarios adapted for single-node forward sampling)

  Stage 2 (TODO): build_multistage_model()  – Pyomo MILP over the tree
  Stage 3 (TODO): multistage_SP_policy()    – solve + extract t=0 decision
  Stage 4 (TODO): select_action()           – grader-compatible wrapper

Design choices
--------------
  Lookahead horizon    : HORIZON           = 4         (configurable)
  Raw MC per node      : GEN_SCENARIOS     = 100       (paths sampled from each node)
  Clusters per node    : N_SCENARIOS       = 3         (K-Means centroids per node)
  Per-stage branching  : BRANCHING_FACTORS = [5,4,3,3] (can vary by stage)
  Solver               : Gurobi, gurobi_direct
"""

import warnings

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
BRANCHING_FACTORS = [5, 4, 3, 3]  # per-stage branch counts; len must equal HORIZON

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
    # ── Root overrule: fix root decisions to match apply_dynamics exactly ─────
    # apply_dynamics enforces overrule BEFORE computing dynamics, so the root
    # decision must respect the same rules — otherwise the MILP optimises over
    # a root action that the environment will silently override, causing a
    # mismatch between the planned and executed cost.
    # =========================================================================
    for r in [1, 2]:
        if state[f'low_override_r{r}'] == 1:
            # Environment forces full heating — fix root lower bound
            m.Heat[r, 0].setlb(d['heating_max_power'])
        if float(state[f'T{r}']) > d['temp_max_comfort_threshold']:
            # Environment forces heating off — fix root upper bound to zero
            m.Heat[r, 0].setub(0.0)
    if float(state['H']) > d['humidity_threshold'] or state['vent_counter'] in [1, 2]:
        # Environment forces ventilation on — fix root Vent to 1.
        # vent_counter in [1,2] means apply_dynamics sets v_eff=1 regardless
        # of our decision, so the MILP must plan with Vent[0]=1.
        m.Vent[0].fix(1)

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

    # ── Min uptime: root startup forces stage-1 nodes ON ──────────────────────
    # descendants_up_to_depth from root reaches stage-1 and stage-2 nodes,
    # but Vent at those nodes is only defined on N_scen (stage>=1). The
    # general MinVentOn constraint above handles this correctly via all_node_set
    # filtering — but we add an explicit constraint to be safe: if ventilation
    # starts at the root (Vstart[0]=1), ALL stage-1 nodes must have Vent=1
    # (they represent the first step after the root decision).
    stage1_nids_set = set(children[0])
    m.MinVentOnStage1 = Constraint(
        rule=lambda m: (
            sum(m.Vent[nid] for nid in stage1_nids_set)
            >= len(stage1_nids_set) * m.Vstart[0]
        )
    )

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

    print_tree_summary(nodes,children,leaves)


    # ── Build and solve MILP ──────────────────────────────────────────────────
    model  = build_sp_model(state, nodes, children, leaves)
    solver = SolverFactory('gurobi_direct')
    solver.options['TimeLimit'] = 12
    solver.options['MIPGap']    = 0.01
    result = solver.solve(model, tee=False)

    tc = result.solver.termination_condition

    if tc not in (TerminationCondition.optimal, TerminationCondition.feasible):
        # Solver did not find a proven feasible solution — try the incumbent.
        # Gurobi almost always finds a good incumbent within the first few
        # seconds of branch-and-bound, even when it can't close the MIP gap
        # within the time limit.  Reading the variables will succeed if an
        # incumbent exists; otherwise it raises and we fall back to zeros.
        try:
            p1 = float(value(model.Heat[1, 0]))
            p2 = float(value(model.Heat[2, 0]))
            v  = int(round(float(value(model.Vent[0]))))
            warnings.warn(
                f"Gurobi timeout at t={t} ({tc}) — using incumbent solution "
                f"(H1={p1:.2f}, H2={p2:.2f}, V={v}).",
                RuntimeWarning,
            )
            return {'HeatPowerRoom1': p1, 'HeatPowerRoom2': p2, 'VentilationON': v}
        except Exception:
            warnings.warn(
                f"Gurobi failed at t={t} ({tc}) — no incumbent available, "
                f"falling back to zero.",
                RuntimeWarning,
            )
            return {'HeatPowerRoom1': 0.0, 'HeatPowerRoom2': 0.0, 'VentilationON': 0}

    # ── Extract here-and-now decision ─────────────────────────────────────────
    # The root node (nid=0) IS the here-and-now decision.
    # Heat[r, 0] and Vent[0] are the actions to execute this timestep.
    p1 = float(value(model.Heat[1, 0]))
    p2 = float(value(model.Heat[2, 0]))
    v  = int(round(float(value(model.Vent[0]))))

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