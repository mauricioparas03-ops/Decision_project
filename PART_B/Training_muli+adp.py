import numpy as np
import csv
import warnings
from pathlib import Path
from sklearn.cluster import KMeans
from pyomo.environ import *
from sklearn.linear_model import Ridge  
import matplotlib
matplotlib.use('Agg')
from pyomo.opt import TerminationCondition
from EnvFunctions import apply_dynamics
from Data.PriceProcessRestaurant import price_model 
from Data.OccupancyProcessRestaurant import next_occupancy_levels
from Data.v2_SystemCharacteristics import get_fixed_data

# HYPERPARAMETERS & SETTINGS
N_SAMPLES = 120
K_SCENARIOS = 50
K_SCENARIOS_BACKWARD = 100
ITERATIONS_I = 80
T_HOURS = 10
SWEEPS_J = 6
BETA = 0.15
HORIZON_MULTI = 4    
N_CLUSTERS    = 3 
BRANCHING_FACTOR = 100

# ── System Parameters ─────────────────────────────────
data = get_fixed_data()
feature_cols = [
                "T1", 
                "T2", 
                "H", 
                "price_t", 
                "price_previous", 
                "Occ1", 
                "Occ2", 
                "vent_counter", 
                "low_override_r1", 
                "low_override_r2"
                ]

# Initialization (initial eta_1 guess)
vfa_weights = {}

for t in range(T_HOURS):
    vfa_weights[t] = {}
    
    # Inizializza tutte le feature a 0.0
    for feat in feature_cols:
        vfa_weights[t][feat] = 0.0
        
    # Aggiunge l'intercetta a 0.0
    vfa_weights[t]['intercept'] = 0.0


# ============================================================================
# 0. FUNZIONE DI NORMALIZZAZIONE (NUOVA)
# ============================================================================
def get_normalized_features(state):
    """Normalizza gli stati per stabilizzare la regressione lineare"""
    return {
        'T1': (state['T1'] - 22.0) / 8.0,
        'T2': (state['T2'] - 22.0) / 8.0,
        'H': (state['H'] - 40.0) / 40.0,
        'Occ1': (state['Occ1'] - 20.0) / 30.0,
        'Occ2': (state['Occ2'] - 10.0) / 20.0,
        'price_t': state['price_t'] / 10.0,
        'price_previous': state['price_previous'] / 10.0,
        'vent_counter': state['vent_counter'] / 3.0,
        'low_override_r1': state['low_override_r1'],
        'low_override_r2': state['low_override_r2']
    }
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
            samples_prices= []
            sample_occ1 = []
            sample_occ2 = []
            for _ in range(S):
                price_next = price_model(node['price'], node['price_prev'])
                occ1_next, occ2_next = next_occupancy_levels(node['occupancy1'], node['occupancy2'])
                samples_prices.append(price_next)
                sample_occ1.append(occ1_next)
                sample_occ2.append(occ2_next)

            # --- Cluster samples into K clusters (NO SCALING - align with MultiSP) ---
            X = np.column_stack([samples_prices, sample_occ1, sample_occ2])  # shape (n_samples, 3) — DO NOT use column_stack, it transposes!
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
def vent_counter_expr(n, nodes_map, m, v_prev, U_vent):
    """
    Restituisce un'espressione Pyomo lineare per il vent_counter al nodo n.
    Somma le variabili Vent lungo il percorso verso la root, fino a U_vent passi.
    
    Nota: è una somma (non conta solo i consecutivi) per restare lineare.
    """
    terms = []
    current_id = n
    steps = 0

    while steps < U_vent:
        if current_id not in nodes_map:
            break

        terms.append(m.Vent[current_id])
        steps += 1

        parent_id = nodes_map[current_id]['parent_id']

        if parent_id not in nodes_map:          # il padre è la root
            if steps < U_vent:
                terms.append(m.Vent0)           # variabile simbolica Pyomo
                steps += 1
            if steps < U_vent:
                terms.append(v_prev)            # costante (int 0/1)
                steps += 1
            break

        current_id = parent_id

    return sum(terms)

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
    
# ============================================================================
# 1. MILP FUNCTION (FIXED)
# ============================================================================
def solve_bellman_equation_milp(state, next_t_weights):
    d = data
    t         = state['current_time']
    remaining = d['num_timeslots'] - t
    horizon   = min(HORIZON_MULTI, remaining)
    tree = build_scenario_tree(state, horizon, S=BRANCHING_FACTOR, K=N_CLUSTERS)
    T_init = {1: state['T1'], 2: state['T2']}
    H_init = state['H']
    occ1_root = state['Occ1']
    occ2_root = state['Occ2']

    vent_counter = int(state['vent_counter'])
    v_prev       = 1 if vent_counter > 0 else 0
    low_override = {}
    for r, T_init_r, T_ok_threshold in [(1, state['T1'], data['temp_OK_threshold']),
                                        (2, state['T2'], data['temp_OK_threshold'])]:
        ov = state[f'low_override_r{r}']
        if T_init_r >= T_ok_threshold:
            ov = 0   # override già terminato
        low_override[r] = ov
    for r in [1, 2]:
        T_r = state[f'T{r}']
        if T_r > data['temp_max_comfort_threshold']:
            low_override[r] = 0 
    eps = 10e-6
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
                     initialize={t: d['outdoor_temperature'][min(t, len(d['outdoor_temperature']) - 1)] 
                                for t in range(state["current_time"], state["current_time"] + horizon)})
    price_by_node = {nid: node["price"] for nid, node in nodes_map.items()}
    occ1_by_node = {nid: node["occupancy1"] for nid, node in nodes_map.items()}
    occ2_by_node = {nid: node["occupancy2"] for nid, node in nodes_map.items()}
 
    m.prices = Param(m.N, initialize=price_by_node)
    m.O1     = Param(m.N, initialize=occ1_by_node)
    m.O2     = Param(m.N, initialize=occ2_by_node)

    # ── Decision variables (decision nodes: all non-root nodes) ──────────────────
    # root
    m.Heat0 = Var(m.R, domain=NonNegativeReals, bounds=(0, d['heating_max_power']))
    m.Vent0 = Var(domain=Binary)
    m.Vstart0 = Var(domain=Binary)
    m.y_low0  = Var(m.R, domain=Binary)  
    m.y_ok0   = Var(m.R, domain=Binary)  
    m.y_high0 = Var(m.R, domain=Binary)   
    m.u0      = Var(m.R, domain=Binary) 
    m.T_in0 = Var(m.R, domain=NonNegativeReals)
    m.Hum0  = Var(domain=NonNegativeReals)
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

    #constraints for root node
    m.c_real = Constraint(expr=m.T_in0[1]== T_init[1])
    m.c_real2 = Constraint(expr=m.T_in0[2]== T_init[2])
    m.c_hum = Constraint(expr=m.Hum0 == H_init)

    m.c_0thigh1 = Constraint(m.R, rule=lambda m,r: m.T_in0[r] >= eps + m.Thigh - m.M_temp*(1 - m.y_high0[r]))
    m.c_0thigh2 = Constraint(m.R, rule=lambda m,r: m.T_in0[r] <= m.Thigh + m.M_temp*m.y_high0[r])
    m.c_0heat_off = Constraint(m.R, rule=lambda m,r: m.Heat0[r] <= m.Pr*(1 - m.y_high0[r]))

    m.c_0tlow1 = Constraint(m.R, rule=lambda m,r: m.T_in0[r] <= m.Tmin + m.M_temp*(1 - m.y_low0[r]))
    m.c_0tlow2 = Constraint(m.R, rule=lambda m,r: m.T_in0[r] >= m.Tmin + eps - m.M_temp*m.y_low0[r])

    m.c_0tok1 = Constraint(m.R, rule=lambda m,r: m.T_in0[r] >= m.Tok - m.M_temp*(1 - m.y_ok0[r]))
    m.c_0tok2 = Constraint(m.R, rule=lambda m,r: m.T_in0[r] <= m.Tok + m.M_temp*m.y_ok0[r])


    m.c_0u1 = Constraint(m.R, rule=lambda m,r: m.u0[r] >= m.y_low0[r])
        # Overrule Memory (u variable)
    def u_memory_rule0(m, r):
        return m.u0[r] <= low_override[r] + m.y_low0[r]
    
    m.c_0u2 = Constraint(m.R, rule=u_memory_rule0)
    m.c_0u3 = Constraint(m.R, rule=lambda m,r: m.Heat0[r] >= m.Pr * m.u0[r])
    def u_memory_rule02(m, r):
        return m.u0[r] >= low_override[r] - m.y_ok0[r]
    m.c_0u4 = Constraint(m.R, rule=u_memory_rule02)
    m.c_0u5 = Constraint(m.R, rule=lambda m,r: m.u0[r] <= 1 - m.y_ok0[r])

    m.min_uptime_root = Constraint(expr=m.Vstart0 >= m.Vent0 - v_prev)
    m.min_uptime_root_2 = Constraint(expr=m.Vstart0 <= m.Vent0)
    m.min_uptime_root_3 = Constraint(expr=m.Vstart0 <= 1 - v_prev)
        # Root minimum-up chains: ensure Vstart0 enforces minimum up-time starting at root
    root_uptime_depth = min(value(m.U_vent), horizon)
    m.RootChains = []
    if root_uptime_depth == 1:
        m.RootChains = [[root_id]]
    elif len(tree) > 1:
        for child in tree[1]:
            child_chains = get_descendant_chains(child['id'], root_uptime_depth - 1, nodes_map)
            for chain in child_chains:
                m.RootChains.append([root_id] + chain)

    if m.RootChains:
        m.RootChainSet = Set(initialize=list(range(len(m.RootChains))))

        def min_uptime_root_rule(m, chain_idx):
            chain = m.RootChains[chain_idx]
            return m.Vent0 + sum(m.Vent[k] for k in chain[1:]) >= len(chain) * m.Vstart0

        m.MinVentOnRoot = Constraint(m.RootChainSet, rule=min_uptime_root_rule)

    # Humidity threshold forces ventilation
    m.c_hum_limit0 = Constraint(rule=lambda m: m.Hum0 <= m.Hhigh + m.M_hum * m.Vent0)

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
    m.c_thigh1 = Constraint(m.RN, rule=lambda m,r,n: m.T_in[r,n] >= eps + m.Thigh - m.M_temp*(1 - m.y_high[r,n]))
    m.c_thigh2 = Constraint(m.RN, rule=lambda m,r,n: m.T_in[r,n] <= m.Thigh + m.M_temp*m.y_high[r,n])
    m.c_heat_off = Constraint(m.RN, rule=lambda m,r,n: m.Heat[r,n] <= m.Pr*(1 - m.y_high[r,n]))

    m.c_tlow1 = Constraint(m.RN, rule=lambda m,r,n: m.T_in[r,n] <= m.Tmin  + m.M_temp*(1 - m.y_low[r,n]))
    m.c_tlow2 = Constraint(m.RN, rule=lambda m,r,n: m.T_in[r,n] >= m.Tmin + eps - m.M_temp*m.y_low[r,n])

    m.c_tok1 = Constraint(m.RN, rule=lambda m,r,n: m.T_in[r,n] >= m.Tok - m.M_temp*(1 - m.y_ok[r,n]))
    m.c_tok2 = Constraint(m.RN, rule=lambda m,r,n: m.T_in[r,n] <= m.Tok + m.M_temp*m.y_ok[r,n])

    # Overrule Memory (u variable)
    def u_memory_rule(m, r, n):
        tau_n = node_stage[n] 
        if tau_n == 1:
            # Logic for root depends on the passed state 'low_override'
            return m.u[r, n] <= m.u0[r] + m.y_low[r, n]
        p_id = nodes_map[n]['parent_id']
        return m.u[r, n] <= m.u[r, p_id] + m.y_low[r, n]
    
    m.c_u1 = Constraint(m.RN, rule=lambda m,r,n: m.u[r,n] >= m.y_low[r,n])
    m.c_u2 = Constraint(m.RN, rule=u_memory_rule)
    m.c_u3 = Constraint(m.RN, rule=lambda m,r,n: m.Heat[r,n] >= m.Pr * m.u[r,n])
    def u_memory_rule2(m, r, n):
        tau_n = node_stage[n] 
        if tau_n == 1:
            return m.u[r, n] >= m.u0[r] - m.y_ok[r, n]
        p_id = nodes_map[n]['parent_id']
        return m.u[r, n] >= m.u[r, p_id] - m.y_ok[r, n]
    m.c_u4 = Constraint(m.RN, rule=u_memory_rule2)
    m.c_u5 = Constraint(m.RN, rule=lambda m,r,n: m.u[r,n] <= 1 - m.y_ok[r,n])

    # ── 4. Ventilation Constraints ───────────────────────────────────────────
    def vent_start_rule(m, n):
        tau_n = node_stage[n] 
        if tau_n == 1:
            return m.Vstart[n] >= m.Vent[n] - m.Vent0
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
            return m.Vstart[n] <= 1 - m.Vent0
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
        running_cost_nodes = sum(
            nodes_map[n]['probability'] * (
                m.prices[n] * sum(m.Heat[r, n] for r in m.R) +
                m.prices[n] * m.Pvent * m.Vent[n]
            ) for n in m.N 
        )
        immediate_cost_root = state['price_t'] * (m.Heat0[1] + m.Heat0[2] + m.Pvent * m.Vent0)
        vfa_term = 0.0
        t_vfa = state["current_time"] + horizon  # Il tempo futuro in cui si trovano le foglie
        
        if next_t_weights is not None and t_vfa in next_t_weights:
            w = next_t_weights[t_vfa]
            
            for n in leaf_ids:
                prob = nodes_map[n]['probability']
                p_id = nodes_map[n]['parent_id']
                
                # Recupero dati esogeni del ramo per la normalizzazione
                price_t1 = nodes_map[n]['price']
                price_prev = nodes_map[p_id]['price'] if p_id != root_id else state['price_t']
                occ1_2 = nodes_map[n]['occupancy1']
                occ2_2 = nodes_map[n]['occupancy2']
                
                # Approssimazione lineare del vent_counter sulla foglia basata sul tempo minimo hardware
                vc_expr = vent_counter_expr(n, nodes_map, m, v_prev, int(value(m.U_vent)))

                # Equazione VFA lineare normalizzata
                node_vfa = (
                    w['intercept'] + 
                    w['T1'] * ((m.T_in[1, n] - 22.0) / 8.0) + 
                    w['T2'] * ((m.T_in[2, n] - 22.0) / 8.0) + 
                    w['H'] * ((m.Hum[n] - 40.0) / 40.0) +
                    w['vent_counter'] * (vc_expr / 3.0) + 
                    w['low_override_r1'] * m.u[1, n] + 
                    w['low_override_r2'] * m.u[2, n] + 
                    w['price_t'] * (price_t1 / 10.0) +
                    w['price_previous'] * (price_prev / 10.0) +  
                    w['Occ1'] * ((occ1_2 - 20.0) / 30.0) + 
                    w['Occ2'] * ((occ2_2 - 10.0) / 20.0)
                )
                
                # Somma pesata per la probabilità dello scenario
                vfa_term += prob * node_vfa
        return immediate_cost_root + running_cost_nodes + vfa_term
    m.obj = Objective(rule=obj_rule, sense=minimize)

    SolverFactory('gurobi').solve(m, tee=False)

    return {"HeatPowerRoom1": value(m.Heat0[1]),
            "HeatPowerRoom2": value(m.Heat0[2]),
            "VentilationON": int(value(m.Vent0))}
    
# ============================================================================
# 2. MATHEMATICAL FUNCTION (Used in Backward to calculate target on FIXED actions)
# ============================================================================
def evaluate_fixed_action(state, action, next_weights):
    """Calculates V*(x_{n,t}) = r(y_{n,t}, u_{n,t}) + E[ V^(y_{n,t+1} ; eta^j) ] """
    imm_cost = state['price_t'] * (action['HeatPowerRoom1'] + action['HeatPowerRoom2'] + action['VentilationON'] * data['ventilation_power'])
    if next_weights is None: return imm_cost

    expected_vfa = 0.0
    tout = data['outdoor_temperature'][int(state['current_time'])]

    for _ in range(K_SCENARIOS_BACKWARD):
        sc_p = price_model(state['price_t'], state['price_previous'])
        sc_o1, sc_o2 = next_occupancy_levels(state['Occ1'], state['Occ2'])

        # Deterministic Dynamics
        T1_n = state['T1'] + data['heat_exchange_coeff']*(state['T2']-state['T1']) + data['thermal_loss_coeff']*(tout-state['T1']) + data['heating_efficiency_coeff']*action['HeatPowerRoom1'] - data['heat_vent_coeff']*action['VentilationON'] + data['heat_occupancy_coeff']*state['Occ1']
        T2_n = state['T2'] + data['heat_exchange_coeff']*(state['T1']-state['T2']) + data['thermal_loss_coeff']*(tout-state['T2']) + data['heating_efficiency_coeff']*action['HeatPowerRoom2'] - data['heat_vent_coeff']*action['VentilationON'] + data['heat_occupancy_coeff']*state['Occ2']
        H_n = state['H'] - data['humidity_vent_coeff']*action['VentilationON'] + data['humidity_occupancy_coeff']*(state['Occ1']+state['Occ2'])
        vc_n = (state['vent_counter'] + 1) * action['VentilationON']

        # Override Memory Logic           
        if T1_n < data['temp_min_comfort_threshold']: ov1_n = 1
        elif T1_n >= data['temp_OK_threshold']: ov1_n = 0
        else: ov1_n = state['low_override_r1']

        if T2_n < data['temp_min_comfort_threshold']: ov2_n = 1
        elif T2_n >= data['temp_OK_threshold']: ov2_n = 0
        else: ov2_n = state['low_override_r2']

        # MODIFICA: Creazione dello stato futuro simulato per la normalizzazione
        next_state_sim = {
            'T1': T1_n, 
            'T2': T2_n, 
            'H': H_n,
            'Occ1': sc_o1, 
            'Occ2': sc_o2,
            'price_t': sc_p, 
            'price_previous': state['price_t'],
            'vent_counter': vc_n,
            'low_override_r1': ov1_n, 
            'low_override_r2': ov2_n
        }
        
        norm_feats = get_normalized_features(next_state_sim)

        # Linear approximation (usando le feature normalizzate)
        vfa_k = (next_weights['intercept'] + 
                 next_weights['T1']*norm_feats['T1'] + 
                 next_weights['T2']*norm_feats['T2'] + 
                 next_weights['H']*norm_feats['H'] + 
                 next_weights['price_t']*norm_feats['price_t'] + 
                 next_weights['price_previous']*norm_feats['price_previous'] + 
                 next_weights['Occ1']*norm_feats['Occ1'] + 
                 next_weights['Occ2']*norm_feats['Occ2'] + 
                 next_weights['vent_counter']*norm_feats['vent_counter'] + 
                 next_weights['low_override_r1']*norm_feats['low_override_r1'] + 
                 next_weights['low_override_r2']*norm_feats['low_override_r2'])
        
        expected_vfa += (1.0 / K_SCENARIOS_BACKWARD) * vfa_k

    return imm_cost + expected_vfa


# ============================================================================
# MAIN LOOP: APPROXIMATE POLICY ITERATION (Variant B)
# ============================================================================

for i in range(ITERATIONS_I):
    print(f"\nOUTER LOOP i={i+1}/{ITERATIONS_I} (Policy Improvement)")

    visited_states_actions = {t: [] for t in range(T_HOURS)}

    current_states = []
    for n in range(N_SAMPLES):
        state_n = get_fixed_data().copy()
        state_n['T1'] = np.random.uniform(18.0, 26.0)
        state_n['T2'] = np.random.uniform(18.0, 26.0)
        state_n['H'] = np.random.uniform(20.0, 70.0)
        state_n['Occ1'] = np.random.uniform(25.0, 35.0)
        state_n['Occ2'] = np.random.uniform(15.0, 25.0)
        state_n['price_t'] = np.random.uniform(0.0, 12.0)
        state_n['price_previous'] = np.random.uniform(0.0, 12.0)
        state_n['current_time']   = 0
        current_states.append(state_n)

    # Forward pass
    for t in range(T_HOURS):
        next_t_weights = vfa_weights[t + 1] if t < T_HOURS - 1 else None
        for n in range(N_SAMPLES):
            state_n = current_states[n]
            state_n['current_time'] = t

            action = solve_bellman_equation_milp(state_n, next_t_weights)
            visited_states_actions[t].append((state_n.copy(), action))

            next_state_n, _ = apply_dynamics(state_n, action, data)
            if t + 1 < T_HOURS:
                new_occ1, new_occ2 = next_occupancy_levels(state_n['Occ1'], state_n['Occ2'])
                new_price = price_model(state_n['price_t'], state_n['price_previous'])
                next_state_n['Occ1']           = new_occ1
                next_state_n['Occ2']           = new_occ2
                next_state_n['price_previous'] = state_n['price_t']
                next_state_n['price_t']        = new_price
            current_states[n] = next_state_n

            
    # BACKWARD PASS
    inner_weights = {t: {k: v for k,v in vfa_weights[t].items()} for t in range(T_HOURS)}
    
    for j in range(SWEEPS_J):
        print(f"  Inner Sweep j={j+1}/{SWEEPS_J} (Policy Evaluation)")
        for t in reversed(range(T_HOURS)):
            X_features, Y_targets = [], []
            next_t_weights_j = inner_weights[t + 1] if t < T_HOURS - 1 else None
            
            for state_n, action_n in visited_states_actions[t]:
                target_value = evaluate_fixed_action(state_n, action_n, next_t_weights_j)
                Y_targets.append(target_value)
                norm_feats = get_normalized_features(state_n)
                X_features.append([norm_feats[feat] for feat in feature_cols])
                        
            if len(X_features) > 0:
                regressor = Ridge(alpha=1.0, fit_intercept=True)
                regressor.fit(X_features, Y_targets)
                for idx, feat_name in enumerate(feature_cols):
                    inner_weights[t][feat_name] = regressor.coef_[idx]
                inner_weights[t]['intercept'] = regressor.intercept_

    # POLICY IMPROVEMENT
    for t in range(T_HOURS):
        for k in feature_cols + ['intercept']:
            vfa_weights[t][k] = (1 - BETA) * vfa_weights[t][k] + BETA * inner_weights[t][k]

# FINAL OUTPUT
print("\n=== FINAL RESULT: VFA_WEIGHTS ===")
print("VFA_WEIGHTS = {")
for t in range(T_HOURS):
    clean_weights = {k: round(float(v), 4) for k, v in vfa_weights[t].items()}
    print(f"    {t}: {clean_weights},")
print("}")

# Save final hybrid VFA weights to CSV for later policy usage
output_csv = Path(__file__).resolve().parent / "Data" / "hybrid_vfa_weights.csv"
output_csv.parent.mkdir(parents=True, exist_ok=True)

csv_columns = ["t"] + feature_cols + ["intercept"]
with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=csv_columns)
    writer.writeheader()
    for t in range(T_HOURS):
        row = {"t": t}
        for feat in feature_cols:
            row[feat] = float(vfa_weights[t][feat])
        row["intercept"] = float(vfa_weights[t]["intercept"])
        writer.writerow(row)

print(f"Saved hybrid VFA weights to CSV: {output_csv}")