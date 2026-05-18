"""
multiSP_policy.py
=================
Multi-Stage Stochastic Programming policy for the restaurant HVAC system.

Compatible with Environment.py:
    from policies.multiSP_policy import select_action

Design
------
The MILP is indexed by (t, s) — identical to SP_policy.py — where s now
refers to a FULL PATH (root-to-leaf) through the scenario tree.

The only structural differences vs SP_policy.py are:

  1. Scenario generation:  repeated branch-and-cluster (one KMeans per
     node per stage) instead of a single flat KMeans over full trajectories.

  2. Non-anticipativity:   explicit NAC constraints at EVERY stage t,
     not just t=0.  Two paths s, s' must share Heat[r,t,s] and Vent[t,s]
     if and only if they pass through the same tree node at stage t.

  3. Probabilities:        path probability = product of conditional branch
     probabilities along the path from root to leaf.

Everything else — constraints, variables, objective, solver call, output
format — is COPY-PASTE IDENTICAL to SP_policy.py.

Hyper-parameters
----------------
  HORIZON      : lookahead stages (default 3, i.e. stages t=0,1,2)
  MC_CHILDREN  : raw MC draws per node before clustering (default 50)
  B            : branches kept per node after clustering (default 3)
  Total paths  : B ** HORIZON   (e.g. 3**3 = 27 paths for defaults)
"""

import os
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

os.environ.setdefault('OMP_NUM_THREADS', '1')

# =============================================================================
# HYPER-PARAMETERS
# =============================================================================
HORIZON     = 3   # lookahead stages
MC_CHILDREN = 100  # raw MC draws per node
B           = 4   # clusters (branches) per node

# =============================================================================
# SYSTEM DATA
# =============================================================================
DATA = get_fixed_data()


# =============================================================================
# PART 1 — SCENARIO TREE CONSTRUCTION
# =============================================================================

def _cluster_one_node(samples):
    """
    Cluster *samples* (shape [MC_CHILDREN, 3]: price, occ1, occ2) into B
    centroids.  Returns (centroids array [B,3], conditional probs [B]).
    """
    n = len(samples)
    k = min(B, n)

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(samples)

    km = KMeans(n_clusters=k, n_init=5, random_state=42)
    km.fit(X_sc)

    labels      = km.labels_
    cond_probs  = np.bincount(labels, minlength=k) / n
    centroids   = scaler.inverse_transform(km.cluster_centers_)
    centroids   = np.clip(centroids, 0, None)   # occupancy cannot be negative

    return centroids, cond_probs   # shapes [k, 3], [k]


def _sample_one_step(price_cur, price_prev, occ1_cur, occ2_cur, n):
    """
    Draw *n* one-step-ahead realisations from a single node state.
    Returns array of shape [n, 3]: (price_next, occ1_next, occ2_next).
    """
    rows = []
    for _ in range(n):
        p_next           = price_model(price_cur, price_prev)
        o1_next, o2_next = next_occupancy_levels(occ1_cur, occ2_cur)
        rows.append([p_next, o1_next, o2_next])
    return np.array(rows)


def build_scenario_tree(price_now, price_prev, occ1_now, occ2_now, horizon):
    """
    Build a scenario tree via repeated branch-and-cluster.

    Returns
    -------
    nodes : list of dicts, one per node (including root node 0)
        Each dict has keys:
          'stage'      : int          (0 = root, 1..horizon = future)
          'parent'     : int or None  (None for root)
          'price'      : float        (exogenous value at this node)
          'occ1'       : float
          'occ2'       : float
          'cond_prob'  : float        (P(this node | parent))
          'path_prob'  : float        (joint prob of path root -> this node)
          'children'   : list[int]   (child node indices)

    paths : list of lists
        Each inner list = sequence of node indices [root=0, n1, n2, ..., leaf]
        for one full root-to-leaf path.  len(paths) == B**horizon.

    na_groups : list of lists (length horizon+1)
        na_groups[t] = list of frozensets; each frozenset is a group of
        path indices that share the same tree node at stage t and therefore
        must have identical decisions at t.
    """
    # ── Build nodes level by level ────────────────────────────────────────────
    root = {
        'stage'    : 0,
        'parent'   : None,
        'price'    : price_now,
        'occ1'     : occ1_now,
        'occ2'     : occ2_now,
        'price_prev': price_prev,
        'cond_prob': 1.0,
        'path_prob': 1.0,
        'children' : [],
    }
    nodes    = [root]           # nodes[0] = root
    frontier = [0]              # indices of nodes at the current stage

    for stage in range(1, horizon + 1):
        next_frontier = []
        for pid in frontier:
            p = nodes[pid]
            # Sample MC_CHILDREN one-step-ahead realisations from this node
            samples = _sample_one_step(
                p['price'], p.get('price_prev', p['price']),
                p['occ1'],  p['occ2'], MC_CHILDREN
            )
            centroids, cond_probs = _cluster_one_node(samples)

            for k in range(len(centroids)):
                cid = len(nodes)
                child = {
                    'stage'     : stage,
                    'parent'    : pid,
                    'price'     : float(centroids[k, 0]),
                    'occ1'      : float(centroids[k, 1]),
                    'occ2'      : float(centroids[k, 2]),
                    'price_prev': p['price'],
                    'cond_prob' : float(cond_probs[k]),
                    'path_prob' : p['path_prob'] * float(cond_probs[k]),
                    'children'  : [],
                }
                nodes.append(child)
                nodes[pid]['children'].append(cid)
                next_frontier.append(cid)

        frontier = next_frontier

    # ── Enumerate all root-to-leaf paths ─────────────────────────────────────
    # leaf nodes = nodes with no children
    leaf_ids = [i for i, n in enumerate(nodes) if len(n['children']) == 0]

    def path_to_root(nid):
        """Return list of node indices from root down to nid."""
        seq = []
        cur = nid
        while cur is not None:
            seq.append(cur)
            cur = nodes[cur]['parent']
        return list(reversed(seq))   # root first

    paths = [path_to_root(lid) for lid in leaf_ids]
    # paths[s] = [0, stage1_node, stage2_node, ..., leaf_node]  (length horizon+1)

    # ── Non-anticipativity groups ──────────────────────────────────────────────
    # na_groups[t] = list of sets; each set = paths sharing the same node at t
    na_groups = []
    for t in range(horizon + 1):
        node_to_paths = {}
        for s, path in enumerate(paths):
            nid = path[t]           # node index at stage t along path s
            node_to_paths.setdefault(nid, []).append(s)
        na_groups.append(list(node_to_paths.values()))

    return nodes, paths, na_groups


def validate_scenario_tree(nodes, paths, na_groups, horizon, B, plot=False):
    """
    Run sanity checks on a freshly built scenario tree.
    Call this immediately after build_scenario_tree() during development.
    Set plot=False when running in environment loops to avoid blocking.
    """
    import warnings
    passed = True

    # ── Check 1: path probabilities sum to 1 ─────────────────────────────
    leaf_probs = [nodes[path[-1]]['path_prob'] for path in paths]
    prob_sum   = sum(leaf_probs)
    min_prob   = min(leaf_probs)

    if abs(prob_sum - 1.0) > 0.01:
        warnings.warn(f"[Tree] Leaf probs sum to {prob_sum:.4f}, expected ~1.0")
        passed = False
    if min_prob < 1e-4:
        warnings.warn(f"[Tree] Near-zero path probability: {min_prob:.2e}")
        passed = False
    print(f"[Tree] Prob sum: {prob_sum:.4f} | Min path prob: {min_prob:.4f}")

    # ── Check 2: fan-out is B at every non-leaf node ──────────────────────
    non_leaves  = [n for n in nodes if n['children']]
    fan_outs    = [len(n['children']) for n in non_leaves]
    bad_fanouts = [f for f in fan_outs if f != B]

    if bad_fanouts:
        warnings.warn(f"[Tree] {len(bad_fanouts)} nodes have fan-out != B={B}: {set(bad_fanouts)}")
        passed = False
    print(f"[Tree] Fan-out check: {set(fan_outs)} (expected {{{B}}})")

    # ── Check 3: na_groups cover all paths at every stage ────────────────
    n_paths = len(paths)
    for t, groups in enumerate(na_groups):
        covered = sum(len(g) for g in groups)
        if covered != n_paths:
            warnings.warn(f"[Tree] na_groups[{t}] covers {covered} paths, expected {n_paths}")
            passed = False
    print(f"[Tree] NAC groups: {[len(g) for g in na_groups]} paths per stage")

    # ── Check 4: na_groups collapse correctly (all in one group at t=0) ──
    if len(na_groups[0]) != 1:
        warnings.warn(f"[Tree] na_groups[0] has {len(na_groups[0])} groups, expected 1 (all paths share root)")
        passed = False
    if not all(len(g) == 1 for g in na_groups[horizon]):
        warnings.warn(f"[Tree] na_groups[{horizon}] has non-singleton groups — leaves not fully branched")
        passed = False

    # ── Check 5: path price trajectories spread over time (optional plot) ─
    price_stds = []
    for t in range(horizon + 1):
        prices_at_t = [nodes[path[t]]['price'] for path in paths]
        price_stds.append(np.std(prices_at_t))

    if price_stds[-1] < price_stds[0] * 1.1:
        warnings.warn("[Tree] Price spread does not grow over horizon — tree may not capture uncertainty")
        passed = False
    print(f"[Tree] Price std by stage: {[f'{s:.3f}' for s in price_stds]}")

    if plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for path in paths:
            prices = [nodes[nid]['price'] for nid in path]
            occ1s  = [nodes[nid]['occ1']  for nid in path]
            occ2s  = [nodes[nid]['occ2']  for nid in path]
            axes[0].plot(prices, alpha=0.5)
            axes[1].plot(occ1s,  alpha=0.5)
            axes[2].plot(occ2s,  alpha=0.5)
        for ax, title in zip(axes, ['Price', 'Occ1', 'Occ2']):
            ax.set_title(title)
            ax.set_xlabel('Stage')
        plt.tight_layout()
        plt.savefig('tree_validation.png')   # save instead of show — safe for env loops
        plt.close()
        print("[Tree] Path plot saved to tree_validation.png")

    print(f"[Tree] Validation {'PASSED' if passed else 'FAILED'}")
    return passed


# =============================================================================
# PART 2 — BUILD PYOMO MILP
# =============================================================================

def build_multistage_model(state, nodes, paths, na_groups, horizon):
    """
    Build the multi-stage stochastic MILP.

    Variables and constraints are indexed by (t, s) where s = path index,
    IDENTICAL to SP_policy.py.  The only additions are NAC constraints at
    every stage t (not just t=0).

    Parameters
    ----------
    state     : Environment.py state dict
    nodes     : list of node dicts from build_scenario_tree()
    paths     : list of path lists (s -> [node_ids])
    na_groups : non-anticipativity groups per stage
    horizon   : int

    Returns
    -------
    Pyomo ConcreteModel (unsolved)
    """
    d  = DATA
    _num_timeslots = int(d['num_timeslots'])

    n_paths = len(paths)   # total number of full paths = B**horizon

    m = ConcreteModel()

    # ── Sets ──────────────────────────────────────────────────────────────────
    m.R   = Set(initialize=[1, 2])
    m.T   = Set(initialize=list(range(horizon)))
    m.S   = Set(initialize=list(range(n_paths)))
    m.RTS = m.R * m.T * m.S
    m.TS  = m.T * m.S

    # ── Initial state (identical to SP_policy) ────────────────────────────────
    # vent_counter > 0 means vent was recently on, but only [1,2] means forced ON
    # vent_counter >= U_vent means the inertia window has expired
    U_vent = int(DATA['U_vent']) if 'U_vent' in DATA else 3  # match your m.U_vent
    v_prev = 1 if 0 < state['vent_counter'] < U_vent else 0
    #v_prev = 1 if state['vent_counter'] > 0 else 0

    m.Tinit    = Param(m.R, initialize={1: state['T1'], 2: state['T2']})
    m.Hinit    = Param(initialize=state['H'])
    m.VentInit = Param(initialize=v_prev)

    # ── Exogenous parameters: price and occupancy per (t, s) ──────────────────
    # For path s at stage t, read from the node at paths[s][t].
    # Stage 0 node = root (current observation), but dynamics at t=0 pin
    # T_in[r,0,s] = Tinit, so price/occ at t=0 is not needed in the objective.
    # We index t over range(horizon), and paths[s][t+1] is the node at stage t+1
    # whose exogenous values are REALISED at t (i.e., drive the cost at t).
    #
    # Convention (same as 2-stage):
    #   prices[t, s]  = price that will be paid at decision step t
    #   O[r, t, s]    = occupancy at decision step t
    #
    # For multi-stage: at decision step t, the path has already branched to the
    # node paths[s][t+1], so we use that node's values.
    # At t = horizon-1 (last step), paths[s][horizon] = leaf node.

    price_init = {}
    occ_init   = {}
    for s, path in enumerate(paths):
        for t in range(horizon):
            node = nodes[path[t + 1]]   # node revealed at step t
            price_init[t, s]   = node['price']
            occ_init[1, t, s]  = node['occ1']
            occ_init[2, t, s]  = node['occ2']

    m.prices = Param(m.TS,  initialize=price_init)
    m.O      = Param(m.RTS, initialize=occ_init)

    # Path probabilities
    path_probs = {s: nodes[paths[s][-1]]['path_prob'] for s in range(n_paths)}
    m.pi = Param(m.S, initialize=path_probs)

    # Outdoor temperature (deterministic, same for all scenarios)
    m.Tout = Param(m.T, initialize={
        t: d['outdoor_temperature'][(state['current_time'] + t) % _num_timeslots]
        for t in range(horizon)
    })

    # ── Physical constants (IDENTICAL to SP_policy) ───────────────────────────
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

    # ── Decision variables (IDENTICAL to SP_policy) ───────────────────────────
    m.Vent   = Var(m.TS,  domain=Binary)
    m.Vstart = Var(m.TS,  domain=Binary)
    m.Heat   = Var(m.RTS, domain=NonNegativeReals, bounds=(0, d['heating_max_power']))

    m.y_low  = Var(m.RTS, domain=Binary)
    m.y_ok   = Var(m.RTS, domain=Binary)
    m.y_high = Var(m.RTS, domain=Binary)
    m.u      = Var(m.RTS, domain=Binary)

    m.T_in = Var(m.RTS, domain=NonNegativeReals)
    m.Hum  = Var(m.TS,  domain=NonNegativeReals)


    # Slack variables for comfort violations (added to objective)
    # At the top of build_multistage_model, add one parameter:
    COMFORT_BUFFER = 1.5   # °C above Tmin to start penalizing
    COMFORT_WEIGHT = 5.0   # tune this — should be ~mean(price * Pr) to make it meaningful

    m.slack_temp = Var(m.RTS, domain=NonNegativeReals)

    # slack >= (Tmin + buffer) - T_in  (active only when T_in is below buffer)
    m.CSlack = Constraint(m.RTS,
        rule=lambda m, r, t, s: m.slack_temp[r, t, s] >= 
            (m.Tmin + COMFORT_BUFFER) - m.T_in[r, t, s])

    # =========================================================================
    # CONSTRAINTS — copy-paste identical to SP_policy.py
    # =========================================================================

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

    # ── High temperature: forced heating shutdown ─────────────────────────────
    m.CThigh1 = Constraint(m.RTS,
        rule=lambda m, r, t, s: m.T_in[r, t, s] >= m.Thigh - m.M_temp * (1 - m.y_high[r, t, s]))
    m.CThigh2 = Constraint(m.RTS,
        rule=lambda m, r, t, s: m.T_in[r, t, s] <= m.Thigh + m.M_temp * m.y_high[r, t, s])
    m.CHeatOff = Constraint(m.RTS,
        rule=lambda m, r, t, s: m.Heat[r, t, s] <= m.Pr * (1 - m.y_high[r, t, s]))

    # ── Low temperature: overrule activation ─────────────────────────────────
    m.CTlow1 = Constraint(m.RTS,
        rule=lambda m, r, t, s: m.T_in[r, t, s] <= m.Tmin + m.M_temp * (1 - m.y_low[r, t, s]))
    m.CTlow2 = Constraint(m.RTS,
        rule=lambda m, r, t, s: m.T_in[r, t, s] >= m.Tmin - m.M_temp * m.y_low[r, t, s])

    # ── Temperature-OK: overrule deactivation ────────────────────────────────
    m.CTok1 = Constraint(m.RTS,
        rule=lambda m, r, t, s: m.T_in[r, t, s] >= m.Tok - m.M_temp * (1 - m.y_ok[r, t, s]))
    m.CTok2 = Constraint(m.RTS,
        rule=lambda m, r, t, s: m.T_in[r, t, s] <= m.Tok + m.M_temp * m.y_ok[r, t, s])

    # ── Overrule memory (u) ───────────────────────────────────────────────────
    m.CU1 = Constraint(m.RTS,
        rule=lambda m, r, t, s: m.u[r, t, s] >= m.y_low[r, t, s])

    def c_u2(m, r, t, s):
        u_prev = state[f'low_override_r{r}'] if t == 0 else m.u[r, t - 1, s]
        return m.u[r, t, s] <= u_prev + m.y_low[r, t, s]
    m.CU2 = Constraint(m.RTS, rule=c_u2)

    def c_u3(m, r, t, s):
        u_prev = state.get(f'low_override_r{r}', 0) if t == 0 else m.u[r, t - 1, s]
        return m.u[r, t, s] >= u_prev - m.y_ok[r, t, s]
    m.CU3 = Constraint(m.RTS, rule=c_u3)

    m.CU4 = Constraint(m.RTS,
        rule=lambda m, r, t, s: m.u[r, t, s] <= 1 - m.y_ok[r, t, s])

    m.CHeatMax = Constraint(m.RTS,
        rule=lambda m, r, t, s: m.Heat[r, t, s] >= m.Pr * m.u[r, t, s])

    # ── Ventilation startup signal ────────────────────────────────────────────
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

    # ── Minimum uptime ────────────────────────────────────────────────────────
    def min_uptime(m, t, s):
        end_idx = min(t + m.U_vent - 1, m.L - 1)
        min_val = min(m.U_vent, m.L - t)
        return sum(m.Vent[tau, s] for tau in range(t, end_idx + 1)) >= min_val * m.Vstart[t, s]
    m.MinVentOn = Constraint(m.TS, rule=min_uptime)

    # ── Humidity overrule ─────────────────────────────────────────────────────
    m.CVentHum = Constraint(m.TS,
        rule=lambda m, t, s: m.Hum[t, s] <= m.Hhigh + m.M_hum * m.Vent[t, s])

    # =========================================================================
    # NON-ANTICIPATIVITY CONSTRAINTS — at every stage t
    # =========================================================================
    # Two paths s1, s2 that share the same tree node at stage t must make
    # identical decisions at that stage.  na_groups[t] contains the groups.
    #
    # We enforce: for each group, pin every member to equal the first member.

    heat_na_rules = {}
    vent_na_rules = {}

    for t in range(horizon):
        for group in na_groups[t]:
            if len(group) < 2:
                continue
            s_ref = group[0]
            for s in group[1:]:
                for r in [1, 2]:
                    heat_na_rules[(r, t, s_ref, s)] = (r, t, s_ref, s)
                vent_na_rules[(t, s_ref, s)] = (t, s_ref, s)

    # Build NAC sets and constraints
    m.HeatNA_idx = Set(initialize=list(heat_na_rules.keys()))
    m.VentNA_idx = Set(initialize=list(vent_na_rules.keys()))

    def heat_nac(m, r, t, s_ref, s):
        return m.Heat[r, t, s_ref] == m.Heat[r, t, s]
    m.HeatNA = Constraint(m.HeatNA_idx, rule=heat_nac)

    def vent_nac(m, t, s_ref, s):
        return m.Vent[t, s_ref] == m.Vent[t, s]
    m.VentNA = Constraint(m.VentNA_idx, rule=vent_nac)

    # =========================================================================
    # OBJECTIVE — probability-weighted cost (IDENTICAL to SP_policy)
    # =========================================================================
    # def objective(m):
    #     return sum(
    #         m.pi[s] * sum(
    #             m.prices[t, s] * (
    #                 sum(m.Heat[r, t, s] for r in m.R)
    #                 + m.Vent[t, s] * m.Pvent
    #             )
    #             for t in m.T
    #         )
    #         for s in m.S
    #     )
    # m.obj = Objective(rule=objective, sense=minimize)
    def objective(m):
        energy_cost = sum(
            m.pi[s] * sum(
                m.prices[t, s] * (
                    sum(m.Heat[r, t, s] for r in m.R)
                    + m.Vent[t, s] * m.Pvent
                )
                for t in m.T
            )
            for s in m.S
        )
        # Penalise scenarios where T_in is below Tmin + buffer
        # slack[r,t,s] = max(0, (Tmin + buffer) - T_in[r,t,s])
        comfort_penalty = sum(
            m.pi[s] * sum(
                COMFORT_WEIGHT * m.slack_temp[r, t, s]
                for r in m.R for t in m.T
            )
            for s in m.S
        )
        return energy_cost + comfort_penalty
    m.obj = Objective(rule=objective, sense=minimize)

    return m


# =============================================================================
# PART 3 — POLICY FUNCTION
# =============================================================================

def multiSP_policy(state):
    """
    Multi-stage stochastic programming policy.

    Builds a scenario tree, solves the multi-stage MILP, and returns the
    here-and-now decisions (stage t=0, which are identical across all paths
    by non-anticipativity).
    """
    t         = state['current_time']
    remaining = DATA['num_timeslots'] - t
    horizon   = min(HORIZON, remaining)
    validate = True # set to True to run tree validation checks and plots (recommended during development, can be False in env loops)
    plot     = True # set to True to show tree validation plots (set to False in env loops to avoid blocking)

    if horizon <= 0:
        return {'HeatPowerRoom1': 0.0, 'HeatPowerRoom2': 0.0, 'VentilationON': 0}

    # ── Build scenario tree ───────────────────────────────────────────────────
    nodes, paths, na_groups = build_scenario_tree(
        price_now  = state['price_t'],
        price_prev = state['price_previous'],
        occ1_now   = state['Occ1'],
        occ2_now   = state['Occ2'],
        horizon    = horizon,
    )

    if validate:
        ok = validate_scenario_tree(nodes, paths, na_groups, horizon, B, plot=plot)
        if not ok:
            import warnings
            warnings.warn("[Policy] Tree validation failed — solution may be unreliable")

    # ── Build and solve MILP ──────────────────────────────────────────────────
    model = build_multistage_model(state, nodes, paths, na_groups, horizon)

    solver = SolverFactory('gurobi_direct')
    solver.options['TimeLimit'] = 10
    solver.options['MIPGap']    = 0.02
    solver.options['Seed']      = 42
    solver.options['Threads']   = 1

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
    # By NAC, Heat[r, 0, s] and Vent[0, s] are identical for all s.
    # Read from s=0 (first path), same convention as SP_policy.
    s0 = 0
    p1 = float(value(model.Heat[1, 0, s0]))
    p2 = float(value(model.Heat[2, 0, s0]))
    v  = int(round(float(value(model.Vent[0, s0]))))

    return {'HeatPowerRoom1': p1, 'HeatPowerRoom2': p2, 'VentilationON': v}


# =============================================================================
# GRADER-COMPATIBLE WRAPPER
# =============================================================================

def select_action(state):
    """Entry point expected by Environment.py."""
    return multiSP_policy(state)