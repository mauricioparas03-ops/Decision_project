import numpy as np
import csv
import warnings
from pathlib import Path
from sklearn.cluster import KMeans
from pyomo.environ import (
    ConcreteModel, Set, Param, Var, Objective, Constraint,
    SolverFactory, NonNegativeReals, minimize, Binary, value,
)
from pyomo.opt import TerminationCondition
from Data.v2_SystemCharacteristics import get_fixed_data
from Data.PriceProcessRestaurant import price_model
from Data.OccupancyProcessRestaurant import next_occupancy_levels
from sklearn.preprocessing import StandardScaler

# ── Hyper-parameters ───────────────────────────────────────────────────────────
HORIZON_MULTI = 4    
N_CLUSTERS    = 3 
BRANCHING_FACTOR = 100

# ── System Parameters ─────────────────────────────────
data = get_fixed_data()

def load_vfa_weights_from_csv():
    csv_path = Path(__file__).resolve().parents[1] / "Data" / "hybrid_vfa_weights.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Hybrid VFA weights CSV not found: {csv_path}")

    weights = {}
    with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = int(row["t"])
            weights[t] = {k: float(v) for k, v in row.items() if k != "t"}
    return weights


# ── weights for the hybrid policy ─────────────────────────────────────────────
VFA_WEIGHTS = load_vfa_weights_from_csv()

"""
    0: {'T1': -10.4107, 'T2': -9.858, 'H': 41.3139, 'price_t': 149.0293, 'price_previous': -55.8317, 'Occ1': 0.2401, 'Occ2': 3.1496, 'vent_counter': 0.0, 'low_override_r1': 0.0, 'low_override_r2': 0.0, 'intercept': 76.1364},
    1: {'T1': -7.079, 'T2': -2.3242, 'H': 40.3869, 'price_t': 122.1908, 'price_previous': 15.6402, 'Occ1': 0.7388, 'Occ2': 6.6406, 'vent_counter': 4.5014, 'low_override_r1': 0.0058, 'low_override_r2': 0.389, 'intercept': 32.672},
    2: {'T1': -11.8826, 'T2': -6.1858, 'H': 39.0304, 'price_t': 86.5111, 'price_previous': 49.1913, 'Occ1': 0.5189, 'Occ2': 4.0114, 'vent_counter': 20.272, 'low_override_r1': 0.7717, 'low_override_r2': 2.9775, 'intercept': 20.5864},
    3: {'T1': -6.0447, 'T2': -1.9932, 'H': 32.181, 'price_t': 77.5764, 'price_previous': 47.4754, 'Occ1': -1.0249, 'Occ2': 5.9025, 'vent_counter': 13.0823, 'low_override_r1': 11.2871, 'low_override_r2': 10.9403, 'intercept': 17.3355},
    4: {'T1': -7.4693, 'T2': -5.5469, 'H': 23.3097, 'price_t': 65.7235, 'price_previous': 43.4386, 'Occ1': 4.3104, 'Occ2': 2.4931, 'vent_counter': 3.9147, 'low_override_r1': 24.2356, 'low_override_r2': 22.2654, 'intercept': 12.4632},
    5: {'T1': -11.4844, 'T2': -10.7455, 'H': 21.067, 'price_t': 57.6215, 'price_previous': 35.6616, 'Occ1': 1.6597, 'Occ2': 2.0515, 'vent_counter': 1.5883, 'low_override_r1': 18.5726, 'low_override_r2': 17.4447, 'intercept': 1.768},
    6: {'T1': -18.5941, 'T2': -15.6486, 'H': 18.7309, 'price_t': 48.9972, 'price_previous': 27.9695, 'Occ1': -0.0251, 'Occ2': 3.0373, 'vent_counter': 2.8134, 'low_override_r1': 13.9636, 'low_override_r2': 13.9481, 'intercept': -12.6866},
    7: {'T1': -16.0351, 'T2': -16.4726, 'H': 19.575, 'price_t': 35.4553, 'price_previous': 24.7922, 'Occ1': 0.1854, 'Occ2': 4.0779, 'vent_counter': 1.5421, 'low_override_r1': 15.8584, 'low_override_r2': 15.5243, 'intercept': -22.489},
    8: {'T1': -13.8393, 'T2': -11.0145, 'H': 19.4111, 'price_t': 23.3659, 'price_previous': 19.801, 'Occ1': -1.9692, 'Occ2': 2.3989, 'vent_counter': 0.3947, 'low_override_r1': 15.813, 'low_override_r2': 15.6576, 'intercept': -25.1943},
    9: {'T1': 0.4197, 'T2': 0.6514, 'H': 11.0903, 'price_t': 12.2551, 'price_previous': 8.7014, 'Occ1': 0.7629, 'Occ2': 2.059, 'vent_counter': -0.3963, 'low_override_r1': 10.4557, 'low_override_r2': 10.6662, 'intercept': -13.5703},
}
"""

# ── Scenario tree construction ─────────────────────────────────────────────
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

        if parent_id not in nodes_map:         
            if steps < U_vent:
                terms.append(m.Vent0)           
                steps += 1
            if steps < U_vent:
                terms.append(v_prev)            
                steps += 1
            break

        current_id = parent_id

    return sum(terms)
    
def build_hybrid_model(state):
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
            ov = 0  
        low_override[r] = ov
    for r in [1, 2]:
        T_r = state[f'T{r}']
        if T_r > data['temp_max_comfort_threshold']:
            low_override[r] = 0 
    eps = 10e-6
    # ── Node sets ─────────────────────────────────────────────────────────────
    root_id = tree[0][0]['id']

    nodes_map = {
        node['id']: node
        for stage in tree
        for node in stage
        if node['id'] != root_id
    }

    all_node_ids = list(nodes_map.keys())

    leaf_ids = [nid for nid, node in nodes_map.items() if not node['children']]

    non_root_ids = [nid for nid in all_node_ids if nid != root_id]

    decision_ids = non_root_ids

    # Set for Pyomo

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

    def humidity_dynamics_rule(m, n):        
        p_id = nodes_map[n]['parent_id']
        tau_n = node_stage[n]         
        if tau_n == 1:  
            occ_term = occ1_root + occ2_root
            H_parent = H_init
            vent_parent = m.Vent0
        else:
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
        t_vfa = state["current_time"] + horizon  
        
        if t_vfa <= 9:
            w = VFA_WEIGHTS[t_vfa]
            
            for n in leaf_ids:
                prob = nodes_map[n]['probability']
                p_id = nodes_map[n]['parent_id']
                
                price_t1 = nodes_map[n]['price']
                price_prev = nodes_map[p_id]['price'] if p_id != root_id else state['price_t']
                occ1_2 = nodes_map[n]['occupancy1']
                occ2_2 = nodes_map[n]['occupancy2']
                
                vc_expr = vent_counter_expr(n, nodes_map, m, v_prev, int(value(m.U_vent)))

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
                vfa_term += prob * node_vfa
        else:
            vfa_term = 0.0
        return immediate_cost_root + running_cost_nodes + vfa_term
    m.obj = Objective(rule=obj_rule, sense=minimize)

    SolverFactory('gurobi').solve(m, tee=False)

    return {"HeatPowerRoom1": value(m.Heat0[1]),
            "HeatPowerRoom2": value(m.Heat0[2]),
            "VentilationON": int(value(m.Vent0))}
def select_action(state):
    decisions = build_hybrid_model(state)
    return {'HeatPowerRoom1': decisions["HeatPowerRoom1"], 'HeatPowerRoom2': decisions["HeatPowerRoom2"], 'VentilationON': decisions["VentilationON"]}