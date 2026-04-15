import numpy as np
from pyomo.environ import *
from sklearn.linear_model import LinearRegression

from EnvFunctions import apply_dynamics
from Data.PriceProcessRestaurant import price_model 
from Data.OccupancyProcessRestaurant import next_occupancy_levels
from Data.v2_SystemCharacteristics import get_fixed_data

# --- HYPERPARAMETERS ---
N_STATES_TO_SAMPLE = 500  # Quante traiettorie (giorni) esplorare
K_SAMPLES = 5             # Scenari futuri per il calcolo dell'Expected Future Cost
T_HOURS = 10

data = get_fixed_data()
feature_cols = ["T1", "T2", "H", "price_t", "vent_counter", "low_override_r1", "low_override_r2"]

# Inizializzazione contenitori
states_by_time = {t: [] for t in range(T_HOURS)}
VFA_WEIGHTS = {t: {feat: 0.0 for feat in feature_cols} for t in range(T_HOURS)}
for t in range(T_HOURS): VFA_WEIGHTS[t]['intercept'] = 0.0

# ============================================================================
# 1. FORWARD PASS: SAMPLING STATES (Esplorazione Randomica)
# ============================================================================
print("Generazione degli stati tramite simulazione...")
for day in range(N_STATES_TO_SAMPLE):
    state = {
        "T1": np.random.uniform(data['temp_min_comfort_threshold'] - 2, data['temp_max_comfort_threshold'] + 2),
        "T2": np.random.uniform(data['temp_min_comfort_threshold'] - 2, data['temp_max_comfort_threshold'] + 2),
        "H": np.random.uniform(40, 85),
        "Occ1": np.random.uniform(25, 35), 
        "Occ2": np.random.uniform(15, 25), 
        "price_t": np.random.uniform(2, 8),  
        "price_previous": np.random.uniform(2, 8), 
        "vent_counter": np.random.choice([0, 1, 2]),
        "low_override_r1": np.random.choice([0, 1]),
        "low_override_r2": np.random.choice([0, 1]),
        "current_time": 0
    }

    for t in range(T_HOURS):
        state['current_time'] = t
        states_by_time[t].append(state.copy())
        
        # Azione puramente casuale per esplorare lo spazio degli stati
        decision = {
            "HeatPowerRoom1": np.random.uniform(0, data['heating_max_power']),
            "HeatPowerRoom2": np.random.uniform(0, data['heating_max_power']),
            "VentilationON": np.random.choice([0, 1])
        }
        
        state, _ = apply_dynamics(state, decision, data)
        
        if t + 1 < T_HOURS:
            next_o1, next_o2 = next_occupancy_levels(state['Occ1'], state['Occ2'])
            next_p = price_model(state['price_t'], state['price_previous'])
            state['Occ1'] = next_o1; state['Occ2'] = next_o2
            state['price_previous'] = state['price_t']; state['price_t'] = next_p


# ============================================================================
# 2. TARGET CALCULATION (1-step Bellman Equation)
# ============================================================================
def solve_one_step(state, future_scenarios, weights_next_step):
    m = ConcreteModel()
    
    m.p1 = Var(bounds=(0, data['heating_max_power']))
    m.p2 = Var(bounds=(0, data['heating_max_power']))
    m.v = Var(domain=Binary)

    # Overrules esatti del timestep corrente
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
        
    m.Scenarios = RangeSet(0, len(future_scenarios) - 1)
    m.T1_next = Var(m.Scenarios); m.T2_next = Var(m.Scenarios); m.H_next = Var(m.Scenarios)
    m.low_override_r1_next = Var(m.Scenarios, domain=Binary)
    m.low_override_r2_next = Var(m.Scenarios, domain=Binary)
    m.vent_counter_next = Var(domain=NonNegativeReals)
    
    m.c_vc = Constraint(expr=m.vent_counter_next == (state['vent_counter'] + 1) * m.v)
    immediate_cost = state['price_t'] * (m.p1 + m.p2 + m.v * data['ventilation_power'])
    
    expected_future_cost = 0
    prob = 1.0 / len(future_scenarios)
    tout = data['outdoor_temperature'][int(state['current_time'])]
    M = 100; eps = 0.001
    
    for k in m.Scenarios:
        scen = future_scenarios[k]
        
        m.add_component(f"c_t1_{k}", Constraint(expr=m.T1_next[k] == state['T1'] + data['heat_exchange_coeff'] * (state['T2'] - state['T1']) + data['thermal_loss_coeff'] * (tout - state['T1']) + data['heating_efficiency_coeff'] * m.p1 - data['heat_vent_coeff'] * m.v + data['heat_occupancy_coeff'] * state['Occ1']))
        m.add_component(f"c_t2_{k}", Constraint(expr=m.T2_next[k] == state['T2'] + data['heat_exchange_coeff'] * (state['T1'] - state['T2']) + data['thermal_loss_coeff'] * (tout - state['T2']) + data['heating_efficiency_coeff'] * m.p2 - data['heat_vent_coeff'] * m.v + data['heat_occupancy_coeff'] * state['Occ2']))
        m.add_component(f"c_h_{k}", Constraint(expr=m.H_next[k] == state['H'] - data['humidity_vent_coeff'] * m.v + data['humidity_occupancy_coeff'] * (state['Occ1'] + state['Occ2'])))
        
        thresh1 = (data['temp_min_comfort_threshold'] if state['low_override_r1'] == 0 else data['temp_OK_threshold']) + eps
        m.add_component(f"c_low1_a_{k}", Constraint(expr=m.T1_next[k] >= thresh1 - M * m.low_override_r1_next[k]))
        m.add_component(f"c_low1_b_{k}", Constraint(expr=m.T1_next[k] <= thresh1 + M * (1 - m.low_override_r1_next[k])))

        thresh2 = (data['temp_min_comfort_threshold'] if state['low_override_r2'] == 0 else data['temp_OK_threshold']) + eps
        m.add_component(f"c_low2_a_{k}", Constraint(expr=m.T2_next[k] >= thresh2 - M * m.low_override_r2_next[k]))
        m.add_component(f"c_low2_b_{k}", Constraint(expr=m.T2_next[k] <= thresh2 + M * (1 - m.low_override_r2_next[k])))

        # Calcolo del costo futuro basato sui pesi GIA' addestrati del timestep t+1
        if weights_next_step is not None:
            scen_value = (
                weights_next_step['intercept'] + 
                weights_next_step['T1'] * m.T1_next[k] + 
                weights_next_step['T2'] * m.T2_next[k] +
                weights_next_step['H'] * m.H_next[k] + 
                weights_next_step['price_t'] * scen['price_t'] +
                weights_next_step['vent_counter'] * m.vent_counter_next +
                weights_next_step['low_override_r1'] * m.low_override_r1_next[k] +
                weights_next_step['low_override_r2'] * m.low_override_r2_next[k]
            )
        else:
            scen_value = 0.0 
            
        expected_future_cost += prob * scen_value

    m.obj = Objective(expr=immediate_cost + expected_future_cost, sense=minimize)
    solver = SolverFactory('gurobi')
    solver.solve(m, tee=False)
    
    return value(m.obj)

# ============================================================================
# 3. BACKWARD INDUCTION (Singolo sweep a ritroso)
# ============================================================================
print("Inizio fase di Backward Induction...")
for t in reversed(range(T_HOURS)):
    print(f"Training pesi per t = {t}...")
    X = []
    y = []
    
    # Il segreto dell'SBI: la Value Function al tempo t+1 è già definitiva.
    weights_next_step = VFA_WEIGHTS[t+1] if t < T_HOURS - 1 else None
    
    for state in states_by_time[t]:
        future_scenarios = []
        for _ in range(K_SAMPLES):
            next_p = price_model(state['price_t'], state['price_previous'])
            next_o1, next_o2 = next_occupancy_levels(state['Occ1'], state['Occ2'])
            future_scenarios.append({'price_t': next_p, 'Occ1': next_o1, 'Occ2': next_o2})
        
        try:
            target_v = solve_one_step(state, future_scenarios, weights_next_step)
            y.append(target_v)
            X.append([state[feat] for feat in feature_cols])
        except Exception:
            continue 
            
    if len(X) > 0:
        model = LinearRegression(fit_intercept=True)
        model.fit(X, y)
        for idx, feat in enumerate(feature_cols):
            VFA_WEIGHTS[t][feat] = model.coef_[idx]
        VFA_WEIGHTS[t]['intercept'] = model.intercept_

print("\n--- RISULTATO TRAINING SBI ---")
print("VFA_WEIGHTS = {")
for t in range(T_HOURS):
    print(f"    {t}: {{", end="")
    for k, v in VFA_WEIGHTS[t].items():
        print(f"'{k}': {v:.4f}, ", end="")
    print("},")
print("}")