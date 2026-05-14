import numpy as np
from pyomo.environ import *
from sklearn.linear_model import Ridge  # MODIFICA: Importata Ridge al posto di LinearRegression
import matplotlib
matplotlib.use('Agg')

from EnvFunctions import apply_dynamics
from Data.PriceProcessRestaurant import price_model 
from Data.OccupancyProcessRestaurant import next_occupancy_levels
from Data.v2_SystemCharacteristics import get_fixed_data

# HYPERPARAMETERS & SETTINGS
N_SAMPLES = 80            # Number of initial scenarios {x_{n,0}}
K_SCENARIOS = 5           # Future samples for expectation E[]
ITERATIONS_I = 15         # MODIFICA: Aumentate le iterazioni a 15 per compensare il Soft Update
SWEEPS_J = 4              # INNER LOOP (Policy Evaluation)
T_HOURS = 10
EPSILON = 0.15            # Exploration
BETA = 0.2                # MODIFICA: Tasso di apprendimento per il Soft Update

data = get_fixed_data()
feature_cols = ["T1", "T2", "H", "price_t", "price_previous", "Occ1", "Occ2", "vent_counter", "low_override_r1", "low_override_r2"]

# Initialization (initial eta_1 guess)
vfa_weights = {t: {feat: 0.0 for feat in feature_cols} for t in range(T_HOURS)}
for t in range(T_HOURS): vfa_weights[t]['intercept'] = 0.0

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

# ============================================================================
# 1. MILP FUNCTION (Used ONLY in Forward Pass to find Optimal Policy)
# ============================================================================
def solve_bellman_equation_milp(state, next_t_weights):
    
    m = ConcreteModel()
    m.p1 = Var(bounds=(0, data['heating_max_power']))
    m.p2 = Var(bounds=(0, data['heating_max_power']))
    m.v = Var(domain=Binary)

    if state['T1'] > data['temp_max_comfort_threshold']: m.p1.fix(0)
    elif state['T1'] < data['temp_min_comfort_threshold'] or state['low_override_r1'] == 1: m.p1.fix(data['heating_max_power'])
    if state['T2'] > data['temp_max_comfort_threshold']: m.p2.fix(0)
    elif state['T2'] < data['temp_min_comfort_threshold'] or state['low_override_r2'] == 1: m.p2.fix(data['heating_max_power'])
    if state['H'] > data['humidity_threshold'] or state['vent_counter'] in [1, 2]: m.v.fix(1)

    scen_data = []
    for _ in range(K_SCENARIOS):
        sc_p = price_model(state['price_t'], state['price_previous'])
        sc_o1, sc_o2 = next_occupancy_levels(state['Occ1'], state['Occ2'])
        scen_data.append({'price_t': sc_p, 'Occ1': sc_o1, 'Occ2': sc_o2})

    m.Scen = RangeSet(0, K_SCENARIOS - 1)
    m.T1_next = Var(m.Scen); m.T2_next = Var(m.Scen); m.H_next = Var(m.Scen)
    m.ov1_next = Var(m.Scen, domain=Binary); m.ov2_next = Var(m.Scen, domain=Binary)
    m.vc_next = Var(domain=NonNegativeReals)
    
    # MODIFICA: rimosse variabili T1_vfa e T2_vfa

    m.c_vc = Constraint(expr=m.vc_next == (state['vent_counter'] + 1) * m.v)
    immediate_cost = state['price_t'] * (m.p1 + m.p2 + m.v * data['ventilation_power'])
    expected_future_cost = 0
    tout = data['outdoor_temperature'][int(state['current_time'])]
    M, eps = 500, 0.001 
    
    for k in m.Scen:
        s = scen_data[k]
        m.add_component(f"ct1_{k}", Constraint(expr=m.T1_next[k] == state['T1'] + data['heat_exchange_coeff']*(state['T2']-state['T1']) + data['thermal_loss_coeff']*(tout-state['T1']) + data['heating_efficiency_coeff']*m.p1 - data['heat_vent_coeff']*m.v + data['heat_occupancy_coeff']*state['Occ1']))
        m.add_component(f"ct2_{k}", Constraint(expr=m.T2_next[k] == state['T2'] + data['heat_exchange_coeff']*(state['T1']-state['T2']) + data['thermal_loss_coeff']*(tout-state['T2']) + data['heating_efficiency_coeff']*m.p2 - data['heat_vent_coeff']*m.v + data['heat_occupancy_coeff']*state['Occ2']))
        m.add_component(f"ch_{k}", Constraint(expr=m.H_next[k] == state['H'] - data['humidity_vent_coeff']*m.v + data['humidity_occupancy_coeff']*(state['Occ1']+state['Occ2'])))
        
        m.add_component(f"y_low_r1_{k}", Var(domain=Binary)); m.add_component(f"y_ok_r1_{k}", Var(domain=Binary))
        m.add_component(f"c_ylow_r1_a_{k}", Constraint(expr=m.T1_next[k] <= data['temp_min_comfort_threshold'] + eps + M*(1 - getattr(m, f"y_low_r1_{k}"))))
        m.add_component(f"c_ylow_r1_b_{k}", Constraint(expr=m.T1_next[k] >= data['temp_min_comfort_threshold'] + eps - M*getattr(m, f"y_low_r1_{k}")))
        m.add_component(f"c_yok_r1_a_{k}", Constraint(expr=m.T1_next[k] >= data['temp_OK_threshold'] - M*(1 - getattr(m, f"y_ok_r1_{k}"))))
        m.add_component(f"c_yok_r1_b_{k}", Constraint(expr=m.T1_next[k] <= data['temp_OK_threshold'] + M*getattr(m, f"y_ok_r1_{k}")))
        u_prev_r1 = state['low_override_r1']
        m.add_component(f"c_u_r1_a_{k}", Constraint(expr=m.ov1_next[k] >= getattr(m, f"y_low_r1_{k}")))
        m.add_component(f"c_u_r1_b_{k}", Constraint(expr=m.ov1_next[k] <= u_prev_r1 + getattr(m, f"y_low_r1_{k}")))
        m.add_component(f"c_u_r1_c_{k}", Constraint(expr=m.ov1_next[k] >= u_prev_r1 - getattr(m, f"y_ok_r1_{k}")))
        m.add_component(f"c_u_r1_d_{k}", Constraint(expr=m.ov1_next[k] <= 1 - getattr(m, f"y_ok_r1_{k}")))

        m.add_component(f"y_low_r2_{k}", Var(domain=Binary)); m.add_component(f"y_ok_r2_{k}", Var(domain=Binary))
        m.add_component(f"c_ylow_r2_a_{k}", Constraint(expr=m.T2_next[k] <= data['temp_min_comfort_threshold'] + eps + M*(1 - getattr(m, f"y_low_r2_{k}"))))
        m.add_component(f"c_ylow_r2_b_{k}", Constraint(expr=m.T2_next[k] >= data['temp_min_comfort_threshold'] + eps - M*getattr(m, f"y_low_r2_{k}")))
        m.add_component(f"c_yok_r2_a_{k}", Constraint(expr=m.T2_next[k] >= data['temp_OK_threshold'] - M*(1 - getattr(m, f"y_ok_r2_{k}"))))
        m.add_component(f"c_yok_r2_b_{k}", Constraint(expr=m.T2_next[k] <= data['temp_OK_threshold'] + M*getattr(m, f"y_ok_r2_{k}")))
        u_prev_r2 = state['low_override_r2']
        m.add_component(f"c_u_r2_a_{k}", Constraint(expr=m.ov2_next[k] >= getattr(m, f"y_low_r2_{k}")))
        m.add_component(f"c_u_r2_b_{k}", Constraint(expr=m.ov2_next[k] <= u_prev_r2 + getattr(m, f"y_low_r2_{k}")))
        m.add_component(f"c_u_r2_c_{k}", Constraint(expr=m.ov2_next[k] >= u_prev_r2 - getattr(m, f"y_ok_r2_{k}")))
        m.add_component(f"c_u_r2_d_{k}", Constraint(expr=m.ov2_next[k] <= 1 - getattr(m, f"y_ok_r2_{k}")))

        if next_t_weights:
            # MODIFICA: rimossa la Trust Region e applicata la normalizzazione direttamente alle variabili
            val_future_state = (next_t_weights['intercept'] + 
                                next_t_weights['T1'] * ((m.T1_next[k] - 22.0) / 8.0) + 
                                next_t_weights['T2'] * ((m.T2_next[k] - 22.0) / 8.0) + 
                                next_t_weights['H'] * ((m.H_next[k] - 40.0) / 40.0) + 
                                next_t_weights['price_t'] * (s['price_t'] / 10.0) + 
                                next_t_weights['price_previous'] * (state['price_t'] / 10.0) + 
                                next_t_weights['Occ1'] * (s['Occ1'] / 30.0) + 
                                next_t_weights['Occ2'] * (s['Occ2'] / 20.0) + 
                                next_t_weights['vent_counter'] * (m.vc_next / 3.0) + 
                                next_t_weights['low_override_r1'] * m.ov1_next[k] + 
                                next_t_weights['low_override_r2'] * m.ov2_next[k])
            expected_future_cost += (1.0 / K_SCENARIOS) * val_future_state

    m.obj = Objective(expr=immediate_cost + expected_future_cost, sense=minimize)
    SolverFactory('gurobi').solve(m, tee=False)
    
    return {"HeatPowerRoom1": value(m.p1), "HeatPowerRoom2": value(m.p2), "VentilationON": int(value(m.v))}

# ============================================================================
# 2. MATHEMATICAL FUNCTION (Used in Backward to calculate target on FIXED actions)
# ============================================================================
def evaluate_fixed_action(state, action, next_weights):
    """Calculates V*(x_{n,t}) = r(y_{n,t}, u_{n,t}) + E[ V^(y_{n,t+1} ; eta^j) ] in pure python"""
    imm_cost = state['price_t'] * (action['HeatPowerRoom1'] + action['HeatPowerRoom2'] + action['VentilationON'] * data['ventilation_power'])
    if next_weights is None: return imm_cost

    expected_vfa = 0.0
    tout = data['outdoor_temperature'][int(state['current_time'])]

    for _ in range(K_SCENARIOS):
        sc_p = price_model(state['price_t'], state['price_previous'])
        sc_o1, sc_o2 = next_occupancy_levels(state['Occ1'], state['Occ2'])

        # Deterministic Dynamics
        T1_n = state['T1'] + data['heat_exchange_coeff']*(state['T2']-state['T1']) + data['thermal_loss_coeff']*(tout-state['T1']) + data['heating_efficiency_coeff']*action['HeatPowerRoom1'] - data['heat_vent_coeff']*action['VentilationON'] + data['heat_occupancy_coeff']*state['Occ1']
        T2_n = state['T2'] + data['heat_exchange_coeff']*(state['T1']-state['T2']) + data['thermal_loss_coeff']*(tout-state['T2']) + data['heating_efficiency_coeff']*action['HeatPowerRoom2'] - data['heat_vent_coeff']*action['VentilationON'] + data['heat_occupancy_coeff']*state['Occ2']
        H_n = state['H'] - data['humidity_vent_coeff']*action['VentilationON'] + data['humidity_occupancy_coeff']*(state['Occ1']+state['Occ2'])
        vc_n = (state['vent_counter'] + 1) * action['VentilationON']

        # Override Memory Logic 
        if T1_n <= data['temp_min_comfort_threshold']: ov1_n = 1
        elif T1_n >= data['temp_OK_threshold']: ov1_n = 0
        else: ov1_n = state['low_override_r1']

        if T2_n <= data['temp_min_comfort_threshold']: ov2_n = 1
        elif T2_n >= data['temp_OK_threshold']: ov2_n = 0
        else: ov2_n = state['low_override_r2']

        # MODIFICA: Creazione dello stato futuro simulato per la normalizzazione
        next_state_sim = {
            'T1': T1_n, 'T2': T2_n, 'H': H_n,
            'Occ1': sc_o1, 'Occ2': sc_o2,
            'price_t': sc_p, 'price_previous': state['price_t'],
            'vent_counter': vc_n,
            'low_override_r1': ov1_n, 'low_override_r2': ov2_n
        }
        
        # MODIFICA: Applicazione della normalizzazione
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
        
        expected_vfa += (1.0 / K_SCENARIOS) * vfa_k

    return imm_cost + expected_vfa


# ============================================================================
# MAIN LOOP: APPROXIMATE POLICY ITERATION (Variant B)
# ============================================================================

for i in range(ITERATIONS_I):
    print(f"\nOUTER LOOP i={i+1}/{ITERATIONS_I} (Policy Improvement)")
      
    # Container to save (State, Action) decided by the current Policy
    visited_states_actions = {t: [] for t in range(T_HOURS)}
    current_states = [get_fixed_data().copy() for _ in range(N_SAMPLES)]
    
    # ----------------------------------------------------
    # FORWARD PASS (We determine the actions and FIX them)
    # ----------------------------------------------------
    for t in range(T_HOURS):
        next_t_weights = vfa_weights[t + 1] if t < T_HOURS - 1 else None
        
        for n in range(N_SAMPLES):
            state_n = current_states[n]
            state_n['current_time'] = t
            
            # Exploration or Exploitation MILP
            if np.random.rand() < EPSILON:
                p1_rand, p2_rand = np.random.uniform(0, data['heating_max_power'], 2)
                v_rand = np.random.choice([0, 1])
                if state_n['T1'] > data['temp_max_comfort_threshold']: p1_rand = 0.0
                elif state_n['T1'] < data['temp_min_comfort_threshold'] or state_n['low_override_r1'] == 1: p1_rand = data['heating_max_power']
                if state_n['T2'] > data['temp_max_comfort_threshold']: p2_rand = 0.0
                elif state_n['T2'] < data['temp_min_comfort_threshold'] or state_n['low_override_r2'] == 1: p2_rand = data['heating_max_power']
                if state_n['H'] > data['humidity_threshold'] or state_n['vent_counter'] in [1, 2]: v_rand = 1
                action = {"HeatPowerRoom1": p1_rand, "HeatPowerRoom2": p2_rand, "VentilationON": v_rand}
            else:
                action = solve_bellman_equation_milp(state_n, next_t_weights)
            
            # We save for Backward: "At this state, I took this action"
            visited_states_actions[t].append((state_n.copy(), action))
            
            # Dynamics for the next step
            next_state_n, _ = apply_dynamics(state_n, action, data)
            if t + 1 < T_HOURS:
                new_occ1, new_occ2 = next_occupancy_levels(state_n['Occ1'], state_n['Occ2'])
                new_price = price_model(state_n['price_t'], state_n['price_previous'])
                next_state_n['Occ1'] = new_occ1; next_state_n['Occ2'] = new_occ2
                next_state_n['price_previous'] = state_n['price_t']
                next_state_n['price_t'] = new_price
            current_states[n] = next_state_n

    # ----------------------------------------------------
    # BACKWARD PASS (Inner Loop j in J - Policy Evaluation)
    # ----------------------------------------------------
    # Initialize inner weights eta_hat for sweeps
    inner_weights = {t: {k: v for k,v in vfa_weights[t].items()} for t in range(T_HOURS)}
    
    for j in range(SWEEPS_J):
        print(f"  Inner Sweep j={j+1}/{SWEEPS_J} (Policy Evaluation)")
        
        for t in reversed(range(T_HOURS)):
            X_features, Y_targets = [], []
            next_t_weights_j = inner_weights[t + 1] if t < T_HOURS - 1 else None
            
            for state_n, action_n in visited_states_actions[t]:
                # Evaluate the FIXED action using pure math, updating with respect to eta_hat_j
                target_value = evaluate_fixed_action(state_n, action_n, next_t_weights_j)
                Y_targets.append(target_value)
                
                # MODIFICA: Normalizzazione prima della regressione
                norm_feats = get_normalized_features(state_n)
                X_features.append([norm_feats[feat] for feat in feature_cols])
            
            if len(X_features) > 0:
                # MODIFICA: Utilizzo della Ridge Regression per evitare l'overfitting sui campioni
                regressor = Ridge(alpha=1.0, fit_intercept=True)
                regressor.fit(X_features, Y_targets)
                for idx, feat_name in enumerate(feature_cols):
                    inner_weights[t][feat_name] = regressor.coef_[idx]
                inner_weights[t]['intercept'] = regressor.intercept_

    # ----------------------------------------------------
    # POLICY IMPROVEMENT (Soft Update)
    # ----------------------------------------------------
    # MODIFICA: Implementato il Soft Update per evitare instabilità
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