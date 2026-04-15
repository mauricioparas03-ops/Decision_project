import numpy as np
from pyomo.environ import *
from sklearn.linear_model import LinearRegression
from pathlib import Path

from EnvFunctions import apply_dynamics
from Data.PriceProcessRestaurant import price_model 
from Data.OccupancyProcessRestaurant import next_occupancy_levels
from Data.v2_SystemCharacteristics import get_fixed_data
from policies.dummy_policy import select_action as dummy_action

# HYPERPARAMETERS OF THE ALGORITHM
NUM_DAYS_SIMULATION = 100  
K_SAMPLES = 5             # future scenario for each state (Monte Carlo)
ITERATIONS = 10           # number of iterations of FVI
T_HOURS = 10

data = get_fixed_data()
feature_cols = [
    "T1", 
    "T2", 
    "H", 
    "price_t", 
    "vent_counter", 
    "low_override_r1", 
    "low_override_r2"
]


current_weights = {}
states_by_time = {}

# initialize 
for t in range(T_HOURS):
    current_weights[t] = {feat: 0.0 for feat in feature_cols}
    current_weights[t]['intercept'] = 0.0  
    states_by_time[t] = []

# STATE SAMPLING WITH TRAJECTORY SIMULATION (Epsilon-Greedy)
for day in range(NUM_DAYS_SIMULATION):
    state = get_fixed_data().copy()
    state['current_time'] = 0

    for t in range(T_HOURS):
        state['current_time'] = t
        states_by_time[t].append(state.copy())
        
        # Epsilon-greedy exploration: 20% random, 80% sensible policy
        if np.random.rand() < 0.20:
            decision = {
                "HeatPowerRoom1": np.random.uniform(0, data['heating_max_power']),
                "HeatPowerRoom2": np.random.uniform(0, data['heating_max_power']),
                "VentilationON": np.random.choice([0, 1])
            }
        else:
            decision = dummy_action(state)
        
        # Physical Dynamics
        state, _ = apply_dynamics(state, decision, data)
        
        # Exogenous Dynamics (Mathematical Models, NO CSV DATA)
        if t + 1 < T_HOURS:
            next_o1, next_o2 = next_occupancy_levels(state['Occ1'], state['Occ2'])
            next_p = price_model(state['price_t'], state['price_previous'])
            
            state['Occ1'] = next_o1
            state['Occ2'] = next_o2
            state['price_previous'] = state['price_t']
            state['price_t'] = next_p


# TARGET V* CALCULATION (one step lookahead)
def solve_one_step(state, future_scenarios, weights_next_step):
    m = ConcreteModel()
    
    # here and now variables
    m.p1 = Var(bounds=(0, data['heating_max_power']))
    m.p2 = Var(bounds=(0, data['heating_max_power']))
    m.v = Var(domain=Binary)

    # Immediate overrule
    if state['T1'] > data['temp_max_comfort_threshold']:
        m.p1.fix(0)
    elif state['low_override_r1'] == 1:
        m.p1.fix(data['heating_max_power'])

    if state['T2'] > data['temp_max_comfort_threshold']:
        m.p2.fix(0)
    elif state['low_override_r2'] == 1:
        m.p2.fix(data['heating_max_power'])

    if state['H'] > data['humidity_threshold'] or state['vent_counter'] in [1, 2]:
        m.v.fix(1)
        
    m.Scenarios = RangeSet(0, len(future_scenarios) - 1)
    
    # state variables
    m.T1_next = Var(m.Scenarios)
    m.T2_next = Var(m.Scenarios)
    m.H_next = Var(m.Scenarios)
    m.low_override_r1_next = Var(m.Scenarios, domain=Binary)
    m.low_override_r2_next = Var(m.Scenarios, domain=Binary)
    m.vent_counter_next = Var(domain=NonNegativeReals)
    
    # Trust Region Variables
    m.T1_vfa = Var(m.Scenarios)
    m.T2_vfa = Var(m.Scenarios)

    m.c_vc = Constraint(expr=m.vent_counter_next == (state['vent_counter'] + 1) * m.v)
    
    # Immediate cost
    immediate_cost = state['price_t'] * (m.p1 + m.p2 + m.v * data['ventilation_power'])
    
    expected_future_cost = 0
    prob = 1.0 / len(future_scenarios)
    tout = data['outdoor_temperature'][int(state['current_time'])]
    M = 100
    eps = 0.001
    
    # constraints for future k scenarios
    for k in m.Scenarios:
        scen = future_scenarios[k]
        
        m.add_component(f"c_t1_{k}", Constraint(expr=m.T1_next[k] == state['T1'] + data['heat_exchange_coeff'] * (state['T2'] - state['T1']) + data['thermal_loss_coeff'] * (tout - state['T1']) + data['heating_efficiency_coeff'] * m.p1 - data['heat_vent_coeff'] * m.v + data['heat_occupancy_coeff'] * state['Occ1']))
        m.add_component(f"c_t2_{k}", Constraint(expr=m.T2_next[k] == state['T2'] + data['heat_exchange_coeff'] * (state['T1'] - state['T2']) + data['thermal_loss_coeff'] * (tout - state['T2']) + data['heating_efficiency_coeff'] * m.p2 - data['heat_vent_coeff'] * m.v + data['heat_occupancy_coeff'] * state['Occ2']))
        m.add_component(f"c_h_{k}", Constraint(expr=m.H_next[k] == state['H'] - data['humidity_vent_coeff'] * m.v + data['humidity_occupancy_coeff'] * (state['Occ1'] + state['Occ2'])))
        
        # mapping for future overrule
        thresh1 = (data['temp_min_comfort_threshold'] if state['low_override_r1'] == 0 else data['temp_OK_threshold']) + eps
        m.add_component(f"c_low1_a_{k}", Constraint(expr=m.T1_next[k] >= thresh1 - M * m.low_override_r1_next[k]))
        m.add_component(f"c_low1_b_{k}", Constraint(expr=m.T1_next[k] <= thresh1 + M * (1 - m.low_override_r1_next[k])))

        thresh2 = (data['temp_min_comfort_threshold'] if state['low_override_r2'] == 0 else data['temp_OK_threshold']) + eps
        m.add_component(f"c_low2_a_{k}", Constraint(expr=m.T2_next[k] >= thresh2 - M * m.low_override_r2_next[k]))
        m.add_component(f"c_low2_b_{k}", Constraint(expr=m.T2_next[k] <= thresh2 + M * (1 - m.low_override_r2_next[k])))

        # bellman equation for Approximate Value Function
        if weights_next_step is not None:
            # Dynamic Trust Region for T1
            if weights_next_step['T1'] < 0:
                m.add_component(f"c_vfa_t1_a_{k}", Constraint(expr=m.T1_vfa[k] <= m.T1_next[k]))
                m.add_component(f"c_vfa_t1_b_{k}", Constraint(expr=m.T1_vfa[k] <= data['temp_max_comfort_threshold']))
            else:
                m.add_component(f"c_vfa_t1_{k}", Constraint(expr=m.T1_vfa[k] == m.T1_next[k]))

            # Dynamic Trust Region for T2
            if weights_next_step['T2'] < 0:
                m.add_component(f"c_vfa_t2_a_{k}", Constraint(expr=m.T2_vfa[k] <= m.T2_next[k]))
                m.add_component(f"c_vfa_t2_b_{k}", Constraint(expr=m.T2_vfa[k] <= data['temp_max_comfort_threshold']))
            else:
                m.add_component(f"c_vfa_t2_{k}", Constraint(expr=m.T2_vfa[k] == m.T2_next[k]))

            scen_value = (
                weights_next_step['intercept'] + 
                weights_next_step['T1'] * m.T1_vfa[k] + 
                weights_next_step['T2'] * m.T2_vfa[k] +
                weights_next_step['H'] * m.H_next[k] + 
                weights_next_step['price_t'] * scen['price_t'] +
                weights_next_step['vent_counter'] * m.vent_counter_next +
                weights_next_step['low_override_r1'] * m.low_override_r1_next[k] +
                weights_next_step['low_override_r2'] * m.low_override_r2_next[k]
            )
        else:
            scen_value = 0.0 # if t=9 future cost is 0 (Natural End-of-Horizon)
            
        expected_future_cost += prob * scen_value

    m.obj = Objective(expr=immediate_cost + expected_future_cost, sense=minimize)
    solver = SolverFactory('gurobi')
    solver.solve(m, tee=False)
    
    return value(m.obj)


# FITTED VALUE ITERATION (ETA UPDATE)
for i in range(ITERATIONS):
    for t in reversed(range(T_HOURS)):
        X = []
        y = []
        
        # next hour weights (0 if last hour)
        weights_next_step = current_weights[t+1] if t < T_HOURS - 1 else None
        
        for state in states_by_time[t]:
            future_scenarios = []
            
            # K future scenarios creation
            for _ in range(K_SAMPLES):
                next_p = price_model(state['price_t'], state['price_previous'])
                next_o1, next_o2 = next_occupancy_levels(state['Occ1'], state['Occ2'])
                future_scenarios.append({'price_t': next_p, 'Occ1': next_o1, 'Occ2': next_o2})
            
            try:
                # Target V* calculation
                target_v = solve_one_step(state, future_scenarios, weights_next_step)
                y.append(target_v)
                X.append([state[feat] for feat in feature_cols])
            except Exception:
                continue 
                
        # linear regression: learning weights (eta) for hour t and fitting with intercept
        if len(X) > 0:
            model = LinearRegression(fit_intercept=True)
            model.fit(X, y)
            for idx, feat in enumerate(feature_cols):
                current_weights[t][feat] = model.coef_[idx]
            current_weights[t]['intercept'] = model.intercept_

    # check
    print(f"Iteration {i} | weigth t=0 -> T1: {current_weights[0]['T1']:.4f}, vent: {current_weights[0]['vent_counter']:.4f}, low_ov1: {current_weights[0]['low_override_r1']:.4f}")


print("VFA_WEIGHTS = {")
for t in range(T_HOURS):
    print(f"    {t}: {{", end="")
    for k, v in current_weights[t].items():
        print(f"'{k}': {v:.4f}, ", end="")
    print("},")
print("}")