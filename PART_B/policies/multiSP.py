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
HORIZON_MULTI = 6    
N_CLUSTERS    = 3 
BRANCHING_FACTOR = 6

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
            probabilities = []
            for _ in range(S):
                price_next = price_model(node['price'], node['price_prev'])
                occ1_next, occ2_next = next_occupancy_levels(node['occupancy1'], node['occupancy2'])
                probability = node['probability'] / S 
                samples.append([price_next, occ1_next, occ2_next])
                probabilities.append(probability)

            # --- Cluster samples into K clusters ---
            scaler = StandardScaler()
            samples_scaled = scaler.fit_transform(samples)
            kmeans = KMeans(n_clusters=K, random_state=0).fit(samples_scaled)
            centers = scaler.inverse_transform(kmeans.cluster_centers_)

            # --- Create new nodes for each cluster center ---
            for k in range(K):
                # Conditional probability: p(cluster k | parent)
                cluster_count = sum(1 for i in range(S) if kmeans.labels_[i] == k)
                conditional_prob = cluster_count / S
                
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
    v_on_h       = vent_counter  
    low_override = {
        1: int(state['low_override_r1']),
        2: int(state['low_override_r2']),
    }

    # ── Node sets ─────────────────────────────────────────────────────────────
    # 1. Creiamo un dizionario piatto per accesso rapido: id_globale -> dati_nodo
    # Questo risolve il problema "tree[stage][nid]" che non funzionerebbe
    nodes_map = {node['id']: node for stage in tree for node in stage}

    # 2. Definiamo i set usando gli ID globali
    all_node_ids = list(nodes_map.keys())

    # Root è sempre il primo nodo del primo stage
    root_id = tree[0][0]['id']

    # Foglie: nodi che non hanno figli
    leaf_ids = [nid for nid, node in nodes_map.items() if not node['children']]

    # Nodi interni: hanno figli
    internal_ids = [nid for nid, node in nodes_map.items() if node['children']]

    # Decision nodes: solitamente sono tutti i nodi TRANNE le foglie 
    # (perché sulle foglie non prendi decisioni per il futuro, essendo la fine dell'orizzonte)
    decision_ids = internal_ids 

    # Set per Pyomo

    m = ConcreteModel()

    # ── Sets ──────────────────────────────────────────────────────────────────
    m.R      = Set(initialize=[1, 2])
    m.N = Set(initialize=all_node_ids)
    m.RN = Set(initialize = m.R* m.N)
    m.LeafNodes = Set(initialize=leaf_ids)
    m.InternalNodes = Set(initialize=internal_ids)

    
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
    m.Heat   = Var(m.RN, domain=NonNegativeReals, bounds=(0, d['heating_max_power']))
    m.Vent   = Var(m.N,  domain=Binary)
    m.Vstart = Var(m.N,  domain=Binary)
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
    
    # Root Node Initial Conditions
    def root_temp_init_rule(m, r):
        return m.T_in[r, root_id] == T_init[r]
    m.RootTempInit = Constraint(m.R, rule=root_temp_init_rule)

    def root_hum_init_rule(m):
        return m.Hum[root_id] == H_init
    m.RootHumInit = Constraint(rule=root_hum_init_rule)

    # Dynamics: Child node state = f(Parent state, Parent decision)
    def thermal_dynamics_rule(m, r, n):
        if n == root_id:
            return Constraint.Skip

        p_id = nodes_map[n]['parent_id']
        tau_n = node_stage[n]          # stage del nodo corrente
        r_other = 2 if r == 1 else 1
        t = state["current_time"] + node_stage[p_id]

        if tau_n == 1:  # nodo subito dopo root
            occ_term = occ1_root if r == 1 else occ2_root
        else:
            # per gli altri stage usa occupazione del parent (o del nodo, se preferisci)
            occ_term = m.O1[p_id] if r == 1 else m.O2[p_id]

        return m.T_in[r, n] == (
            m.T_in[r, p_id]
            + m.Zexch * (m.T_in[r_other, p_id] - m.T_in[r, p_id])
            + m.Zloss * (m.Tout[t] - m.T_in[r, p_id])
            + m.Zconv * m.Heat[r, p_id]
            - m.Zcool * m.Vent[p_id]
            + m.Zocc * occ_term
        )
    m.Temp_Dynamics = Constraint(m.R, m.N, rule=thermal_dynamics_rule)

    def humidity_dynamics_rule(m, n):
        if n == root_id: return Constraint.Skip
        
        p_id = nodes_map[n]['parent_id']
        tau_n = node_stage[n]          # stage del nodo corrente
        if tau_n == 1:  # nodo subito dopo root
            occ_term = occ1_root + occ2_root
        else:
            # per gli altri stage usa occupazione del parent (o del nodo, se preferisci)
            occ_term = m.O1[p_id] + m.O2[p_id]

        return m.Hum[n] == (
            m.Hum[p_id]
            - m.Hvent * m.Vent[p_id]
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
        if n == root_id:
            # Logic for root depends on the passed state 'low_override'
            return m.u[r, n] == low_override[r] 
        
        p_id = nodes_map[n]['parent_id']
        return m.u[r, n] <= m.u[r, p_id] + m.y_low[r, n]
    
    m.c_u1 = Constraint(m.RN, rule=lambda m,r,n: m.u[r,n] >= m.y_low[r,n])
    m.c_u2 = Constraint(m.RN, rule=u_memory_rule)
    m.c_u3 = Constraint(m.RN, rule=lambda m,r,n: m.Heat[r,n] >= m.Pr * m.u[r,n])
    def u_memory_rule(m, r, n):
        if n == root_id:
            return m.u[r, n] == low_override[r] 
        
        p_id = nodes_map[n]['parent_id']
        return m.u[r, n] >= m.u[r, p_id] - m.y_ok[r, n]

    m.c_u4 = Constraint(m.RN, rule=lambda m,r,n: m.u[r,n] <= 1 - m.y_ok[r,n])

    # ── 4. Ventilation Constraints ───────────────────────────────────────────
    def vent_start_rule(m, n):
        if n == root_id:
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
        if n == root_id:
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
        )
    m.obj = Objective(rule=obj_rule, sense=minimize)

    return m    

def multiSP_policy(state):
    t         = state['current_time']
    remaining = DATA['num_timeslots'] - t
    horizon   = min(HORIZON_MULTI, remaining)
    tree = build_scenario_tree(state, horizon, S=5, K=2)
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
    p1 = value(model.Heat[1, tree[0][0]['id']])
    p2 = value(model.Heat[2, tree[0][0]['id']])
    v = value(model.Vent[tree[0][0]['id']])
    return {'HeatPowerRoom1': p1, 'HeatPowerRoom2': p2, 'VentilationON': v}

def select_action(state):
    return multiSP_policy(state)

