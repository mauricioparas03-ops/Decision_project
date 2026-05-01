import numpy as np
from pyomo.environ import *
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('Agg')

from EnvFunctions import apply_dynamics
from Data.PriceProcessRestaurant import price_model 
from Data.OccupancyProcessRestaurant import next_occupancy_levels
from Data.v2_SystemCharacteristics import get_fixed_data
from policies.dummy_policy import select_action as dummy_action


# HYPERPARAMETERS & SETTINGS

# We solve a MILP in the Forward Pass too, so we keep simulation days balanced.
N_SAMPLES = 100           # Number of independent trajectories  
K_SCENARIOS = 5           # Scenarios for Bellman backup
ITERATIONS = 12           # Number of Forward-Backward loops
T_HOURS = 10
EPSILON = 0.15            # Probability of choosing a random action (Exploration)

data = get_fixed_data()
feature_cols = ["T1", 
                "T2", 
                "H", 
                "price_t", 
                "vent_counter", 
                "low_override_r1", 
                "low_override_r2"
                ]

# Initialization of the Value Function Approximation (VFA)
# The weights (eta) start at 0.0 for each time step t
vfa_weights = {}
for t in range(T_HOURS):
    # For each hour, create a dictionary to hold the weights
    vfa_weights[t] = {}
    
    # Initialize each feature's weight to 0.0
    for feat in feature_cols:
        vfa_weights[t][feat] = 0.0
        
    # Also initialize the intercept to 0.0
    vfa_weights[t]['intercept'] = 0.0


# CORE OPTIMIZATION ENGINE (MILP for Bellman)


def solve_bellman_equation_milp(state, next_t_weights, mode="value"):
    """
    Solves the Bellman equation: min_u { Immediate_Cost + E[Future_VFA] }
    This function strictly implements the "Create a y_n,t+1 by solving Bellman" 
    and "Compute targets V*(x_n,t)".

    mode="value"  -> returns the expected total cost (V*) for training.
    mode="action" -> returns the optimal decision (u*) for forward simulation.
    """
    m = ConcreteModel()
    
    # Decision Variables (Action u_t)
    m.p1 = Var(bounds=(0, data['heating_max_power'])) # Heating power room 1
    m.p2 = Var(bounds=(0, data['heating_max_power'])) # Heating power room 2
    m.v = Var(domain=Binary)                          # Ventilation ON/OFF

    # Overrule Rules (Physical/Controller constraints)
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

    # Scenario Generation to compute Expectation E
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
    m.ov1_next = Var(m.Scen, domain=Binary)
    m.ov2_next = Var(m.Scen, domain=Binary)
    m.vc_next = Var(domain=NonNegativeReals)
    
    # Trust Region variables (Piece-wise linear logic mapping)
    m.T1_vfa = Var(m.Scen)
    m.T2_vfa = Var(m.Scen)

    m.c_vc = Constraint(expr=m.vc_next == (state['vent_counter'] + 1) * m.v)

    # Immediate Cost Calculation
    immediate_cost = state['price_t'] * (m.p1 + m.p2 + m.v * data['ventilation_power'])
    expected_future_cost = 0
    tout = data['outdoor_temperature'][int(state['current_time'])]
    

    # Weighted Future VFA Calculation over K scenarios
    for k in m.Scen:
        s = scen_data[k]
        
        # Physical Dynamics
        m.add_component(f"ct1_{k}", Constraint(expr=m.T1_next[k] == state['T1'] + data['heat_exchange_coeff']*(state['T2']-state['T1']) + data['thermal_loss_coeff']*(tout-state['T1']) + data['heating_efficiency_coeff']*m.p1 - data['heat_vent_coeff']*m.v + data['heat_occupancy_coeff']*state['Occ1']))
        m.add_component(f"ct2_{k}", Constraint(expr=m.T2_next[k] == state['T2'] + data['heat_exchange_coeff']*(state['T1']-state['T2']) + data['thermal_loss_coeff']*(tout-state['T2']) + data['heating_efficiency_coeff']*m.p2 - data['heat_vent_coeff']*m.v + data['heat_occupancy_coeff']*state['Occ2']))
        m.add_component(f"ch_{k}", Constraint(expr=m.H_next[k] == state['H'] - data['humidity_vent_coeff']*m.v + data['humidity_occupancy_coeff']*(state['Occ1']+state['Occ2'])))
        
        # Binary logic mapping for future overrides
        M, eps = 100, 0.001
        thresh1 = (data['temp_min_comfort_threshold'] if state['low_override_r1'] == 0 else data['temp_OK_threshold']) + eps
        m.add_component(f"cl1a_{k}", Constraint(expr=m.T1_next[k] >= thresh1 - M*m.ov1_next[k]))
        m.add_component(f"cl1b_{k}", Constraint(expr=m.T1_next[k] <= thresh1 + M*(1 - m.ov1_next[k])))
        
        thresh2 = (data['temp_min_comfort_threshold'] if state['low_override_r2'] == 0 else data['temp_OK_threshold']) + eps
        m.add_component(f"cl2a_{k}", Constraint(expr=m.T2_next[k] >= thresh2 - M*m.ov2_next[k]))
        m.add_component(f"cl2b_{k}", Constraint(expr=m.T2_next[k] <= thresh2 + M*(1 - m.ov2_next[k])))

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

            # Value Function Approximation: Linear evaluation of the future state
            val_future_state = (next_t_weights['intercept'] + 
                                next_t_weights['T1'] * m.T1_vfa[k] + 
                                next_t_weights['T2'] * m.T2_vfa[k] + 
                                next_t_weights['H'] * m.H_next[k] + 
                                next_t_weights['price_t'] * s['price_t'] + 
                                next_t_weights['vent_counter'] * m.vc_next + 
                                next_t_weights['low_override_r1'] * m.ov1_next[k] + 
                                next_t_weights['low_override_r2'] * m.ov2_next[k])
            
            expected_future_cost += (1.0 / K_SCENARIOS) * val_future_state
        else:
            # End of horizon: no future expected cost
            expected_future_cost += 0.0

    # Objective: Minimize Bellman equation
    m.obj = Objective(expr=immediate_cost + expected_future_cost, sense=minimize)
    results = SolverFactory('gurobi').solve(m, tee=False)
    if results.solver.termination_condition != TerminationCondition.optimal:
        raise RuntimeError(
            f"Bellman MILP did not solve to optimality: "
            f"status={results.solver.status}, "
            f"termination={results.solver.termination_condition}"
        )

    if mode == "value":
        return value(m.obj)
    else:
        action_dict = {
            "HeatPowerRoom1": value(m.p1), 
            "HeatPowerRoom2": value(m.p2), 
            "VentilationON": int(value(m.v))
        }
        return action_dict


# ADP ALGORITHM: FORWARD-BACKWARD LOOP


for iteration in range(ITERATIONS):
    print(f" ITERATION {iteration + 1}/{ITERATIONS}")
      
    # STATE INITIALIZATION
    # Sample N initial states {x_{n,0}}
    current_states = []
    for _ in range(N_SAMPLES):
        # Create a clean copy of the initial state data for each sample
        initial_state = get_fixed_data().copy()
        current_states.append(initial_state)
    
    # Dictionary to store visited states, essential for the Backward Pass target computation
    visited_states_history = {}
    for t in range(T_HOURS):
        visited_states_history[t] = []
    
    # FORWARD PASS
    
    for t in range(T_HOURS):
        if t < T_HOURS - 1:
            next_t_weights = vfa_weights[t + 1]
        else:
            next_t_weights = None
        
        for n in range(N_SAMPLES):
            # Retrieve the current state for the n-th trajectory
            state_n = current_states[n]
            state_n['current_time'] = t
            
            # Save the state so we can compute its V* Target during the Backward Pass
            visited_states_history[t].append(state_n.copy())
            
            # Action selection u_t (Epsilon greedy to avoid sticking to poor initial policies)
            if np.random.rand() < EPSILON or iteration == 0:
                action = {}
                action["HeatPowerRoom1"] = np.random.uniform(0, data['heating_max_power'])
                action["HeatPowerRoom2"] = np.random.uniform(0, data['heating_max_power'])
                action["VentilationON"] = np.random.choice([0, 1])
            else:
                # Exploit current weights (eta) by solving Bellman to find the optimal action
                action = solve_bellman_equation_milp(state_n, next_t_weights, mode="action")
            
            # Apply the action to get the deterministic next state (Physics)
            next_state_n, _ = apply_dynamics(state_n, action, data)
            
            # Exogenous variable evolution (Prices and Occupancy) for the next hour
            if t + 1 < T_HOURS:
                new_occ1, new_occ2 = next_occupancy_levels(state_n['Occ1'], state_n['Occ2'])
                new_price = price_model(state_n['price_t'], state_n['price_previous'])
                
                next_state_n['Occ1'] = new_occ1
                next_state_n['Occ2'] = new_occ2
                next_state_n['price_previous'] = state_n['price_t']
                next_state_n['price_t'] = new_price
            
            # Update the n-th state in the list, ready for hour t+1
            current_states[n] = next_state_n

    
    # BACKWARD PASS (Fitted Value Iteration)
    
    # Iterate backward in time: T-1, T-2, ..., 0
    for t in reversed(range(T_HOURS)):
        X_features = []
        Y_targets = []
        
        if t < T_HOURS - 1:
            next_t_weights = vfa_weights[t + 1]
        else:
            next_t_weights = None
        
        # For each state saved at this hour during the Forward Pass
        for saved_state in visited_states_history[t]:
            # Compute target V*(x_{n,t}) as: max_u { r + E[V(y)] }
            target_value = solve_bellman_equation_milp(saved_state, next_t_weights, mode="value")
            
            # Append the calculated target to Y
            Y_targets.append(target_value)
            
            # Extract the features for this state and append to X
            feature_values = []
            for feat in feature_cols:
                feature_values.append(saved_state[feat])
            X_features.append(feature_values)
        
        # "Update \eta_t using Fitted Value Iteration" (Multiple Linear Regression)
        if len(X_features) > 0:
            regressor = LinearRegression(fit_intercept=True)
            regressor.fit(X_features, Y_targets)
            
            # Save the new coefficients into the global weights dictionary
            for idx in range(len(feature_cols)):
                feat_name = feature_cols[idx]
                vfa_weights[t][feat_name] = regressor.coef_[idx]
                
            # Save the intercept separately
            vfa_weights[t]['intercept'] = regressor.intercept_

   


# FINAL TRAINED WEIGHTS OUTPUT

print("\n=== FINAL RESULT: VFA_WEIGHTS ===")
print("VFA_WEIGHTS = {")
for t in range(T_HOURS):
    
    clean_weights = {}
    for k, v in vfa_weights[t].items():
        clean_weights[k] = round(float(v), 4)
        
    print(f"    {t}: {clean_weights},")
print("}")