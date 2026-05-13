import numpy as np
from pyomo.environ import *
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('Agg')

from EnvFunctions import apply_dynamics
from Data.PriceProcessRestaurant import price_model 
from Data.OccupancyProcessRestaurant import next_occupancy_levels
from Data.v2_SystemCharacteristics import get_fixed_data

# HYPERPARAMETERS & SETTINGS
N_SAMPLES = 100           # Number of independent trajectories  
K_SCENARIOS = 5           # Scenarios for Bellman backup
ITERATIONS = 12           # Number of Forward-Backward loops
T_HOURS = 10
EPSILON = 0.15            # Probability of choosing a random action (Exploration)

data = get_fixed_data()

# 1. FIX: Added the state variables predicted by the formal MDP (Task 2)
feature_cols = [
    "T1", "T2", "H", 
    "price_t", "price_previous", "Occ1", "Occ2", 
    "vent_counter", "low_override_r1", "low_override_r2"
]

# Initialization of the Value Function Approximation (VFA)
vfa_weights = {}
for t in range(T_HOURS):
    vfa_weights[t] = {}
    for feat in feature_cols:
        vfa_weights[t][feat] = 0.0
    vfa_weights[t]['intercept'] = 0.0


# CORE OPTIMIZATION ENGINE (MILP for Bellman)
def solve_bellman_equation_milp(state, next_t_weights, mode="value"):
    m = ConcreteModel()
    
    # Decision Variables (Action u_t)
    m.p1 = Var(bounds=(0, data['heating_max_power']))
    m.p2 = Var(bounds=(0, data['heating_max_power']))
    m.v = Var(domain=Binary)

    # STEP 0: Overrule Rules (Current Hour t)
    if state['T1'] > data['temp_max_comfort_threshold']: 
        m.p1.fix(0)
    elif state['T1'] < data['temp_min_comfort_threshold'] or state['low_override_r1'] == 1: 
        m.p1.fix(data['heating_max_power'])
        
    if state['T2'] > data['temp_max_comfort_threshold']: 
        m.p2.fix(0)
    elif state['T2'] < data['temp_min_comfort_threshold'] or state['low_override_r2'] == 1: 
        m.p2.fix(data['heating_max_power'])
        
    if state['H'] > data['humidity_threshold'] or state['vent_counter'] in [1, 2]: 
        m.v.fix(1)

    # Scenario Generation
    scen_data = []
    for _ in range(K_SCENARIOS):
        sc_p = price_model(state['price_t'], state['price_previous'])
        sc_o1, sc_o2 = next_occupancy_levels(state['Occ1'], state['Occ2'])
        scenario_dict = {'price_t': sc_p, 'Occ1': sc_o1, 'Occ2': sc_o2}
        scen_data.append(scenario_dict)

    m.Scen = RangeSet(0, K_SCENARIOS - 1)
    
    # Future state variables (y_{n, k, t+1})
    m.T1_next = Var(m.Scen)
    m.T2_next = Var(m.Scen)
    m.H_next = Var(m.Scen)
    m.ov1_next = Var(m.Scen, domain=Binary) # Future memory u R1
    m.ov2_next = Var(m.Scen, domain=Binary) # Future memory u R2
    m.vc_next = Var(domain=NonNegativeReals)
    
    # Trust Region variables 
    m.T1_vfa = Var(m.Scen)
    m.T2_vfa = Var(m.Scen)

    m.c_vc = Constraint(expr=m.vc_next == (state['vent_counter'] + 1) * m.v)

    immediate_cost = state['price_t'] * (m.p1 + m.p2 + m.v * data['ventilation_power'])
    expected_future_cost = 0
    tout = data['outdoor_temperature'][int(state['current_time'])]
    M, eps = 500, 0.001 # 3. FIX: Increased Big-M and eps tolerance
    
    for k in m.Scen:
        s = scen_data[k]
        
        # Physical Dynamics
        m.add_component(f"ct1_{k}", Constraint(expr=m.T1_next[k] == state['T1'] + data['heat_exchange_coeff']*(state['T2']-state['T1']) + data['thermal_loss_coeff']*(tout-state['T1']) + data['heating_efficiency_coeff']*m.p1 - data['heat_vent_coeff']*m.v + data['heat_occupancy_coeff']*state['Occ1']))
        m.add_component(f"ct2_{k}", Constraint(expr=m.T2_next[k] == state['T2'] + data['heat_exchange_coeff']*(state['T1']-state['T2']) + data['thermal_loss_coeff']*(tout-state['T2']) + data['heating_efficiency_coeff']*m.p2 - data['heat_vent_coeff']*m.v + data['heat_occupancy_coeff']*state['Occ2']))
        m.add_component(f"ch_{k}", Constraint(expr=m.H_next[k] == state['H'] - data['humidity_vent_coeff']*m.v + data['humidity_occupancy_coeff']*(state['Occ1']+state['Occ2'])))
        
        # 3. FIX: Professor's rigorous logic for future override (Room 1)
        m.add_component(f"y_low_r1_{k}", Var(domain=Binary))
        m.add_component(f"y_ok_r1_{k}", Var(domain=Binary))
        
        m.add_component(f"c_ylow_r1_a_{k}", Constraint(expr=m.T1_next[k] <= data['temp_min_comfort_threshold'] + eps + M*(1 - getattr(m, f"y_low_r1_{k}"))))
        m.add_component(f"c_ylow_r1_b_{k}", Constraint(expr=m.T1_next[k] >= data['temp_min_comfort_threshold'] + eps - M*getattr(m, f"y_low_r1_{k}")))
        
        m.add_component(f"c_yok_r1_a_{k}", Constraint(expr=m.T1_next[k] >= data['temp_OK_threshold'] - M*(1 - getattr(m, f"y_ok_r1_{k}"))))
        m.add_component(f"c_yok_r1_b_{k}", Constraint(expr=m.T1_next[k] <= data['temp_OK_threshold'] + M*getattr(m, f"y_ok_r1_{k}")))

        u_prev_r1 = state['low_override_r1']
        m.add_component(f"c_u_r1_a_{k}", Constraint(expr=m.ov1_next[k] >= getattr(m, f"y_low_r1_{k}")))
        m.add_component(f"c_u_r1_b_{k}", Constraint(expr=m.ov1_next[k] <= u_prev_r1 + getattr(m, f"y_low_r1_{k}")))
        m.add_component(f"c_u_r1_c_{k}", Constraint(expr=m.ov1_next[k] >= u_prev_r1 - getattr(m, f"y_ok_r1_{k}")))
        m.add_component(f"c_u_r1_d_{k}", Constraint(expr=m.ov1_next[k] <= 1 - getattr(m, f"y_ok_r1_{k}")))

        # 3. FIX: Professor's rigorous logic for future override (Room 2)
        m.add_component(f"y_low_r2_{k}", Var(domain=Binary))
        m.add_component(f"y_ok_r2_{k}", Var(domain=Binary))
        
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
            # Trust Region application
            if next_t_weights['T1'] < 0:
                m.add_component(f"tr1a_{k}", Constraint(expr=m.T1_vfa[k] <= m.T1_next[k]))
                m.add_component(f"tr1b_{k}", Constraint(expr=m.T1_vfa[k] <= data['temp_max_comfort_threshold']))
            else:
                m.add_component(f"tr1_{k}", Constraint(expr=m.T1_vfa[k] == m.T1_next[k]))
                
            if next_t_weights['T2'] < 0:
                m.add_component(f"tr2a_{k}", Constraint(expr=m.T2_vfa[k] <= m.T2_next[k]))
                m.add_component(f"tr2b_{k}", Constraint(expr=m.T2_vfa[k] <= data['temp_max_comfort_threshold']))
            else:
                m.add_component(f"tr2_{k}", Constraint(expr=m.T2_vfa[k] == m.T2_next[k]))

            # Value Function Approximation
            val_future_state = (next_t_weights['intercept'] + 
                                next_t_weights['T1'] * m.T1_vfa[k] + 
                                next_t_weights['T2'] * m.T2_vfa[k] + 
                                next_t_weights['H'] * m.H_next[k] + 
                                next_t_weights['price_t'] * s['price_t'] + 
                                next_t_weights['price_previous'] * state['price_t'] + # The price_t of today becomes price_previous tomorrow!
                                next_t_weights['Occ1'] * s['Occ1'] + 
                                next_t_weights['Occ2'] * s['Occ2'] + 
                                next_t_weights['vent_counter'] * m.vc_next + 
                                next_t_weights['low_override_r1'] * m.ov1_next[k] + 
                                next_t_weights['low_override_r2'] * m.ov2_next[k])
            
            expected_future_cost += (1.0 / K_SCENARIOS) * val_future_state
        else:
            expected_future_cost += 0.0

    m.obj = Objective(expr=immediate_cost + expected_future_cost, sense=minimize)
    results = SolverFactory('gurobi').solve(m, tee=False)
    
    if mode == "value":
        return value(m.obj)
    else:
        return {
            "HeatPowerRoom1": value(m.p1), 
            "HeatPowerRoom2": value(m.p2), 
            "VentilationON": int(value(m.v))
        }


# ADP ALGORITHM: FORWARD-BACKWARD LOOP
for iteration in range(ITERATIONS):
    print(f" ITERATION {iteration + 1}/{ITERATIONS}")
      
    current_states = []
    for _ in range(N_SAMPLES):
        initial_state = get_fixed_data().copy()
        current_states.append(initial_state)
    
    visited_states_history = {t: [] for t in range(T_HOURS)}
    
    # FORWARD PASS
    for t in range(T_HOURS):
        next_t_weights = vfa_weights[t + 1] if t < T_HOURS - 1 else None
        
        for n in range(N_SAMPLES):
            state_n = current_states[n]
            state_n['current_time'] = t
            visited_states_history[t].append(state_n.copy())
            
            # Action selection
            if np.random.rand() < EPSILON or iteration == 0:
                p1_rand = np.random.uniform(0, data['heating_max_power'])
                p2_rand = np.random.uniform(0, data['heating_max_power'])
                v_rand = np.random.choice([0, 1])

                # 2. FIX: Safe exploration that respects hardware
                if state_n['T1'] > data['temp_max_comfort_threshold']: p1_rand = 0.0
                elif state_n['T1'] < data['temp_min_comfort_threshold'] or state_n['low_override_r1'] == 1: p1_rand = data['heating_max_power']
                
                if state_n['T2'] > data['temp_max_comfort_threshold']: p2_rand = 0.0
                elif state_n['T2'] < data['temp_min_comfort_threshold'] or state_n['low_override_r2'] == 1: p2_rand = data['heating_max_power']
                
                if state_n['H'] > data['humidity_threshold'] or state_n['vent_counter'] in [1, 2]: v_rand = 1
                
                action = {"HeatPowerRoom1": p1_rand, "HeatPowerRoom2": p2_rand, "VentilationON": v_rand}
            else:
                action = solve_bellman_equation_milp(state_n, next_t_weights, mode="action")
            
            next_state_n, _ = apply_dynamics(state_n, action, data)
            
            if t + 1 < T_HOURS:
                new_occ1, new_occ2 = next_occupancy_levels(state_n['Occ1'], state_n['Occ2'])
                new_price = price_model(state_n['price_t'], state_n['price_previous'])
                
                next_state_n['Occ1'] = new_occ1
                next_state_n['Occ2'] = new_occ2
                next_state_n['price_previous'] = state_n['price_t']
                next_state_n['price_t'] = new_price
            
            current_states[n] = next_state_n

    # BACKWARD PASS
    for t in reversed(range(T_HOURS)):
        X_features, Y_targets = [], []
        next_t_weights = vfa_weights[t + 1] if t < T_HOURS - 1 else None
        
        for saved_state in visited_states_history[t]:
            target_value = solve_bellman_equation_milp(saved_state, next_t_weights, mode="value")
            Y_targets.append(target_value)
            
            feature_values = [saved_state[feat] for feat in feature_cols]
            X_features.append(feature_values)
        
        if len(X_features) > 0:
            regressor = LinearRegression(fit_intercept=True)
            regressor.fit(X_features, Y_targets)
            
            for idx, feat_name in enumerate(feature_cols):
                vfa_weights[t][feat_name] = regressor.coef_[idx]
            vfa_weights[t]['intercept'] = regressor.intercept_

# FINAL OUTPUT
print("\n=== FINAL RESULT: VFA_WEIGHTS ===")
print("VFA_WEIGHTS = {")
for t in range(T_HOURS):
    clean_weights = {k: round(float(v), 4) for k, v in vfa_weights[t].items()}
    print(f"    {t}: {clean_weights},")
print("}")