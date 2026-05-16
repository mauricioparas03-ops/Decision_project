import numpy as np
import warnings
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
HORIZON_MULTI = 4    
N_CLUSTERS    = 3 
BRANCHING_FACTOR = 100

# ── System Parameters ─────────────────────────────────
DATA = get_fixed_data()

# ── Scenario tree construction ─────────────────────────────────────────────
def build_scenario_tree(state, L, S, K):
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
            samples = []
            for _ in range(S):
                price_next = price_model(node['price'], node['price_prev'])
                occ1_next, occ2_next = next_occupancy_levels(node['occupancy1'], node['occupancy2'])
                samples.append([price_next, occ1_next, occ2_next])

            # --- Cluster samples into K clusters ---
            X = np.asarray(samples)
            n_samples = X.shape[0]
            K_eff = min(K, n_samples)

            if K_eff <= 0:
                # no samples generated, skip
                continue

            if K_eff == 1:
                # single cluster: centroid is the mean, all labels 0
                labels = np.zeros(n_samples, dtype=int)
                centers = X.mean(axis=0, keepdims=True)
            else:
                kmeans = KMeans(n_clusters=K_eff, random_state=0, n_init=10).fit(X)
                labels = kmeans.labels_
                centers = kmeans.cluster_centers_

            # --- Create new nodes for each cluster center ---
            for k in range(K_eff):
                # Conditional probability: p(cluster k | parent)
                conditional_prob = np.sum(labels == k) / n_samples
                
                # Joint probability: p(path to this node)
                joint_prob = node['probability'] * conditional_prob
                new_node = {
                    "id": global_id_counter,           # ID univoco (es: 5, 6, 7...)
                    "price": centers[k][0],
                    "price_prev": node['price'],
                    "occupancy1": centers[k][1],
                    "occupancy2": centers[k][2],
                    "probability": joint_prob,
                    "parent_id": node['id'],          # Puntatore globale al padre
                    "children": []
                }

                node['children'].append(new_node['id'])
                new_nodes.append(new_node)
                global_id_counter += 1
                
        tree.append(new_nodes)
    return tree

def get_descendant_chains(node_id, depth, nodes_map):

    if depth <= 1:
        return [[node_id]]
    
    node = nodes_map[node_id]
    if not node['children']:
        return [[node_id]]
    
    all_chains = []
    for child_id in node['children']:
        child_chains = get_descendant_chains(child_id, depth - 1, nodes_map)
        for cc in child_chains:
            all_chains.append([node_id] + cc)
    return all_chains
    

def build_multisp_model(state, tree, horizon):
    """
    Build the multi-stage stochastic MILP indexed on scenario-tree nodes.

    Parameters
    ----------
    current_state : dict – keys: T1, T2, H,
                                 vent_counter,
                                 low_override_r1, low_override_r2
    tree          : list of lists of dicts, where tree[tau] is the list of nodes at stage tau, and each node is a dict with keys:
                                 id, price, price_prev, occupancy1, occupancy2, probability, parent_id, children
    horizon       : int

    Returns
    -------
    Pyomo ConcreteModel (unsolved)
    """
    d = DATA

    T_init = {1: state['T1'], 2: state['T2']}
    H_init = state['H']
    occ1_root = state['Occ1']
    occ2_root = state['Occ2']

    vent_counter = int(state['vent_counter'])
    v_prev       = 1 if vent_counter > 0 else 0
    low_override = {}
    for r, T_init_r, T_ok_threshold in [(1, state['T1'], DATA['temp_OK_threshold']),
                                        (2, state['T2'], DATA['temp_OK_threshold'])]:
        ov = state[f'low_override_r{r}']
        if T_init_r >= T_ok_threshold:
            ov = 0   # override già terminato
        low_override[r] = ov
    for r in [1, 2]:
        T_r = state[f'T{r}']
        if T_r > DATA['temp_max_comfort_threshold']:
            low_override[r] = 0 

    # ── Node sets ─────────────────────────────────────────────────────────────
    # Root è sempre il primo nodo del primo stage
    root_id = tree[0][0]['id']

    # 1. Creiamo un dizionario piatto per accesso rapido: id_globale -> dati_nodo
    # Questo risolve il problema "tree[stage][nid]" che non funzionerebbe
    nodes_map = {
        node['id']: node
        for stage in tree
        for node in stage
        if node['id'] != root_id
    }

    # 2. Definiamo i set usando gli ID globali
    all_node_ids = list(nodes_map.keys())

    # Foglie: nodi che non hanno figli
    leaf_ids = [nid for nid, node in nodes_map.items() if not node['children']]

    # Solo i nodi dopo la root
    non_root_ids = [nid for nid in all_node_ids if nid != root_id]

    decision_ids = non_root_ids

    # Set per Pyomo

    m = ConcreteModel()

    # ── Sets ──────────────────────────────────────────────────────────────────
    m.R      = Set(initialize=[1, 2])
    m.N = Set(initialize=non_root_ids)
    m.RN = Set(initialize = m.R* m.N)
    m.LeafNodes = Set(initialize=leaf_ids)

    
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
    m.Tout   = Param(range(state["current_time"], state["current_time"] + horizon),
                     initialize={t: d['outdoor_temperature'][t] for t in range(state["current_time"], state["current_time"] + horizon)})        
    price_by_node = {nid: node["price"] for nid, node in nodes_map.items()}
    occ1_by_node = {nid: node["occupancy1"] for nid, node in nodes_map.items()}
    occ2_by_node = {nid: node["occupancy2"] for nid, node in nodes_map.items()}
    path_prob_leaf = {nid: node["probability"] for nid, node in nodes_map.items() if not node["children"]}
        
    m.prices = Param(m.N, initialize=price_by_node)
    m.O1     = Param(m.N, initialize=occ1_by_node)
    m.O2     = Param(m.N, initialize=occ2_by_node)
    m.pi     = Param(m.LeafNodes, initialize=path_prob_leaf)

    # ── Decision variables (decision nodes: all non-root nodes) ──────────────────
    # root
    m.Heat0 = Var(m.R, domain=NonNegativeReals, bounds=(0, d['heating_max_power']))
    m.Vent0 = Var(domain=Binary)
    m.Vstart0 = Var(domain=Binary)
    # internal stages
    m.Heat  = Var(m.RN, domain=NonNegativeReals, bounds=(0, d['heating_max_power']))
    m.Vent  = Var(m.N,  domain=Binary)
    m.Vstart= Var(m.N,  domain=Binary)
    # Overrule indicator variables
    m.y_low  = Var(m.RN, domain=Binary)  
    m.y_ok   = Var(m.RN, domain=Binary)  
    m.y_high = Var(m.RN, domain=Binary)   
    m.u      = Var(m.RN, domain=Binary) 
    # ── State variables (all nodes) ───────────────────────────────────────────
    m.T_in = Var(m.RN, domain=NonNegativeReals)
    m.Hum  = Var(m.N,        domain=NonNegativeReals)

    node_stage = {}
    for stage_idx, stage_nodes in enumerate(tree):
        for node in stage_nodes:
            node_stage[node['id']] = stage_idx

    #constraint for root node
    remaining_uptime = DATA['vent_min_up_time'] - vent_counter
    if vent_counter > 0 and remaining_uptime > 0:
        m.Vent0.fix(1)
   
    if vent_counter > 0 and remaining_uptime > 0:
        for stage_idx in range(1, min(remaining_uptime, horizon)):
            for node in tree[stage_idx]:
                m.Vent[node['id']].fix(1)

    for r in [1, 2]:
        temp_now = state["T1"] if r == 1 else state["T2"]
        if low_override[r]: 
            m.Heat0[r].fix(m.Pr) 
        if temp_now >= m.Thigh: 
            m.Heat0[r].fix(0)

    if state['H'] > m.Hhigh:
        m.Vent0.fix(1)

    m.min_uptime_root = Constraint(expr=m.Vstart0 >= m.Vent0 - v_prev)
    m.min_uptime_root_2 = Constraint(expr=m.Vstart0 <= m.Vent0)
    m.min_uptime_root_3 = Constraint(expr=m.Vstart0 <= 1 - v_prev)
    

    # Dynamics: Child node state = f(Parent state, Parent decision)
    def thermal_dynamics_rule(m, r, n):
        p_id = nodes_map[n]['parent_id']
        tau_n = node_stage[n]          # stage del nodo corrente
        r_other = 2 if r == 1 else 1
        t = state["current_time"] + node_stage[p_id]

        if tau_n == 1:  # nodo subito dopo root
            occ_term = occ1_root if r == 1 else occ2_root
            T_parent = T_init[r]
            T_other_parent = T_init[r_other]
            heat_parent = m.Heat0[r]
            vent_parent = m.Vent0
        else:
            # per gli altri stage usa occupazione del parent (o del nodo, se preferisci)
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

    def humidity_dynamics_rule(m, n):        
        p_id = nodes_map[n]['parent_id']
        tau_n = node_stage[n]          # stage del nodo corrente
        if tau_n == 1:  # nodo subito dopo root
            occ_term = occ1_root + occ2_root
            H_parent = H_init
            vent_parent = m.Vent0
        else:
            # per gli altri stage usa occupazione del parent (o del nodo, se preferisci)
            occ_term = m.O1[p_id] + m.O2[p_id]
            H_parent = m.Hum[p_id]
            vent_parent = m.Vent[p_id]

        return m.Hum[n] == (
            H_parent
            - m.Hvent * vent_parent
            + m.Hocc * occ_term
        )
    m.Hum_Dynamics = Constraint(m.N, rule=humidity_dynamics_rule)

    # ── 3. Big-M Logic (Comfort & Overrule) ───────────────────────────────────
    # These apply to ALL nodes
    m.c_thigh1 = Constraint(m.RN, rule=lambda m,r,n: m.T_in[r,n] >= m.Thigh - m.M_temp*(1 - m.y_high[r,n]))
    m.c_thigh2 = Constraint(m.RN, rule=lambda m,r,n: m.T_in[r,n] <= m.Thigh + m.M_temp*m.y_high[r,n])
    m.c_heat_off = Constraint(m.RN, rule=lambda m,r,n: m.Heat[r,n] <= m.Pr*(1 - m.y_high[r,n]))

    m.c_tlow1 = Constraint(m.RN, rule=lambda m,r,n: m.T_in[r,n] <= m.Tmin + m.M_temp*(1 - m.y_low[r,n]))
    m.c_tlow2 = Constraint(m.RN, rule=lambda m,r,n: m.T_in[r,n] >= m.Tmin - m.M_temp*m.y_low[r,n])

    m.c_tok1 = Constraint(m.RN, rule=lambda m,r,n: m.T_in[r,n] >= m.Tok - m.M_temp*(1 - m.y_ok[r,n]))
    m.c_tok2 = Constraint(m.RN, rule=lambda m,r,n: m.T_in[r,n] <= m.Tok + m.M_temp*m.y_ok[r,n])

    # Overrule Memory (u variable)
    def u_memory_rule(m, r, n):
        tau_n = node_stage[n] 
        if tau_n == 1:
            # Logic for root depends on the passed state 'low_override'
            return m.u[r, n] <= low_override[r] + m.y_low[r, n]
        p_id = nodes_map[n]['parent_id']
        return m.u[r, n] <= m.u[r, p_id] + m.y_low[r, n]
    
    m.c_u1 = Constraint(m.RN, rule=lambda m,r,n: m.u[r,n] >= m.y_low[r,n])
    m.c_u2 = Constraint(m.RN, rule=u_memory_rule)
    m.c_u3 = Constraint(m.RN, rule=lambda m,r,n: m.Heat[r,n] >= m.Pr * m.u[r,n])
    def u_memory_rule2(m, r, n):
        tau_n = node_stage[n] 
        if tau_n == 1:
            return m.u[r, n] >= low_override[r] - m.y_ok[r, n]
        p_id = nodes_map[n]['parent_id']
        return m.u[r, n] >= m.u[r, p_id] - m.y_ok[r, n]
    m.c_u4 = Constraint(m.RN, rule=u_memory_rule2)
    m.c_u5 = Constraint(m.RN, rule=lambda m,r,n: m.u[r,n] <= 1 - m.y_ok[r,n])

    # ── 4. Ventilation Constraints ───────────────────────────────────────────
    def vent_start_rule(m, n):
        tau_n = node_stage[n] 
        if tau_n == 1:
            return m.Vstart[n] >= m.Vent[n] - v_prev
        p_id = nodes_map[n]['parent_id']
        return m.Vstart[n] >= m.Vent[n] - m.Vent[p_id]
    m.c_vstart = Constraint(m.N, rule=vent_start_rule)

    # CVstart2 : Vstart[nid] <= Vent[nid]
    def vent_start_rule_2(m, n):
        return m.Vstart[n] <= m.Vent[n]
    m.c_vstart2 = Constraint(m.N, rule=vent_start_rule_2)

    # CVstart3 : Vstart[nid] <= 1 - Vent[parent]
    def vent_start_rule_3(m, n):
        tau_n = node_stage[n]
        if tau_n == 1:
            return m.Vstart[n] <= 1 - v_prev
        p_id = nodes_map[n]['parent_id']
        return m.Vstart[n] <= 1 - m.Vent[p_id]
    m.c_vstart3 = Constraint(m.N, rule=vent_start_rule_3)

    m.Chains = {}
    for nid in decision_ids:
        m.Chains[nid] = get_descendant_chains(nid, value(m.U_vent), nodes_map)

    def min_uptime_rule(m, nid, chain_idx):
        chain = m.Chains[nid][chain_idx]
        return sum(m.Vent[k] for k in chain) >= len(chain) * m.Vstart[nid]

    m.NodeChainSet = Set(initialize=[(nid, i) for nid in decision_ids for i in range(len(m.Chains[nid]))])
    m.MinVentOn = Constraint(m.NodeChainSet, rule=min_uptime_rule)

    # Humidity threshold forces ventilation
    m.c_hum_limit = Constraint(m.N, rule=lambda m, n: m.Hum[n] <= m.Hhigh + m.M_hum * m.Vent[n])


    # ── 5. Objective: Minimize Expected Cost ──────────────────────────────────
    def obj_rule(m):
        # Total cost = Sum over all nodes (Prob_node * Cost_node)
        return sum(
            nodes_map[n]['probability'] * (
                m.prices[n] * sum(m.Heat[r, n] for r in m.R) +
                m.prices[n] * m.Pvent * m.Vent[n]
            ) for n in m.N 
        ) + state['price_t'] * (m.Heat0[1] + m.Heat0[2] + m.Pvent * m.Vent0)
    m.obj = Objective(rule=obj_rule, sense=minimize)

    return m    

def multiSP_policy(state):
    t         = state['current_time']
    remaining = DATA['num_timeslots'] - t
    horizon   = min(HORIZON_MULTI, remaining)
    tree = build_scenario_tree(state, horizon, S=BRANCHING_FACTOR, K=N_CLUSTERS)
    model = build_multisp_model(state, tree, horizon)
    solver = SolverFactory('gurobi')
    result = solver.solve(model, tee=False)
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
    # ── Extract first-stage decisions from the solved model ───────────────────
    p1 = value(model.Heat0[1])
    p2 = value(model.Heat0[2])
    v = value(model.Vent0)
    return {'HeatPowerRoom1': p1, 'HeatPowerRoom2': p2, 'VentilationON': v}

def select_action(state):
    return multiSP_policy(state)

