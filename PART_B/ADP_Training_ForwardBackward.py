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

# =============================================================================
# 1. HYPERPARAMETERS & SETTINGS
# =============================================================================
# We solve a MILP in the Forward Pass too, so we keep simulation days balanced.
NUM_DAYS_SIMULATION = 100  
K_SAMPLES = 5             # Scenarios for Bellman backup
ITERATIONS = 12           # Number of Forward-Backward loops
T_HOURS = 10
EPSILON = 0.15            # Probability of choosing a random action (Exploration)

data = get_fixed_data()
feature_cols = ["T1", "T2", "H", "price_t", "vent_counter", "low_override_r1", "low_override_r2"]

# Initialize weights to zero (they will evolve through iterations)
current_weights = {t: {feat: 0.0 for feat in feature_cols + ['intercept']} for t in range(T_HOURS)}

# =============================================================================
# 2. CORE OPTIMIZATION FUNCTION (MILP)
# =============================================================================

def solve_one_step_milp(state, weights_next, mode="value"):
    """
    Solves the 1-step lookahead problem.
    mode="value"  -> returns the expected total cost (V*) for training.
    mode="action" -> returns the optimal decision (u*) for forward simulation.
    """
    m = ConcreteModel()
    
    # Decision Variables
    m.p1 = Var(bounds=(0, data['heating_max_power']))
    m.p2 = Var(bounds=(0, data['heating_max_power']))
    m.v = Var(domain=Binary)

    # Immediate Overrules (Standard rules)
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

    # Scenario Generation for Expectation
    K = K_SAMPLES
    scenarios = []
    for _ in range(K):
        sc_p = price_model(state['price_t'], state['price_previous'])
        sc_o1, sc_o2 = next_occupancy_levels(state['Occ1'], state['Occ2'])
        scenarios.append({'price_t': sc_p, 'Occ1': sc_o1, 'Occ2': sc_o2})

    m.Scen = RangeSet(0, K-1)
    m.T1_next = Var(m.Scen)
    m.T2_next = Var(m.Scen)
    m.H_next = Var(m.Scen)
    m.ov1_next = Var(m.Scen, domain=Binary)
    m.ov2_next = Var(m.Scen, domain=Binary)
    m.vc_next = Var(domain=NonNegativeReals)
    
    # Trust Region Variables (Piece-wise Linear approximation)
    m.T1_vfa = Var(m.Scen)
    m.T2_vfa = Var(m.Scen)

    m.c_vc = Constraint(expr=m.vc_next == (state['vent_counter'] + 1) * m.v)
    
    imm_cost = state['price_t'] * (m.p1 + m.p2 + m.v * data['ventilation_power'])
    
    exp_future_cost = 0
    tout = data['outdoor_temperature'][int(state['current_time'])]

    for k in m.Scen:
        s = scenarios[k]
        # Physics Dynamics
        m.add_component(f"ct1_{k}", Constraint(expr=m.T1_next[k] == state['T1'] + data['heat_exchange_coeff']*(state['T2']-state['T1']) + data['thermal_loss_coeff']*(tout-state['T1']) + data['heating_efficiency_coeff']*m.p1 - data['heat_vent_coeff']*m.v + data['heat_occupancy_coeff']*state['Occ1']))
        m.add_component(f"ct2_{k}", Constraint(expr=m.T2_next[k] == state['T2'] + data['heat_exchange_coeff']*(state['T1']-state['T2']) + data['thermal_loss_coeff']*(tout-state['T2']) + data['heating_efficiency_coeff']*m.p2 - data['heat_vent_coeff']*m.v + data['heat_occupancy_coeff']*state['Occ2']))
        m.add_component(f"ch_{k}", Constraint(expr=m.H_next[k] == state['H'] - data['humidity_vent_coeff']*m.v + data['humidity_occupancy_coeff']*(state['Occ1']+state['Occ2'])))
        
        # Binary Mapping for future overrides
        M, eps = 100, 0.001
        thresh1 = (data['temp_min_comfort_threshold'] if state['low_override_r1'] == 0 else data['temp_OK_threshold']) + eps
        m.add_component(f"cl1a_{k}", Constraint(expr=m.T1_next[k] >= thresh1 - M*m.ov1_next[k]))
        m.add_component(f"cl1b_{k}", Constraint(expr=m.T1_next[k] <= thresh1 + M*(1 - m.ov1_next[k])))
        
        thresh2 = (data['temp_min_comfort_threshold'] if state['low_override_r2'] == 0 else data['temp_OK_threshold']) + eps
        m.add_component(f"cl2a_{k}", Constraint(expr=m.T2_next[k] >= thresh2 - M*m.ov2_next[k]))
        m.add_component(f"cl2b_{k}", Constraint(expr=m.T2_next[k] <= thresh2 + M*(1 - m.ov2_next[k])))

        if weights_next:
            # Trust Region (Piece-wise Linear Logic)
            if weights_next['T1'] < 0:
                m.add_component(f"tr1a_{k}", Constraint(expr=m.T1_vfa[k] <= m.T1_next[k]))
                m.add_component(f"tr1b_{k}", Constraint(expr=m.T1_vfa[k] <= data['temp_max_comfort_threshold']))
            else:
                m.add_component(f"tr1_{k}", Constraint(expr=m.T1_vfa[k] == m.T1_next[k]))
                
            if weights_next['T2'] < 0:
                m.add_component(f"tr2a_{k}", Constraint(expr=m.T2_vfa[k] <= m.T2_next[k]))
                m.add_component(f"tr2b_{k}", Constraint(expr=m.T2_vfa[k] <= data['temp_max_comfort_threshold']))
            else:
                m.add_component(f"tr2_{k}", Constraint(expr=m.T2_vfa[k] == m.T2_next[k]))

            # Value Function Approximation
            val = (weights_next['intercept'] + 
                   weights_next['T1'] * m.T1_vfa[k] + 
                   weights_next['T2'] * m.T2_vfa[k] + 
                   weights_next['H'] * m.H_next[k] + 
                   weights_next['price_t'] * s['price_t'] + 
                   weights_next['vent_counter'] * m.vc_next + 
                   weights_next['low_override_r1'] * m.ov1_next[k] + 
                   weights_next['low_override_r2'] * m.ov2_next[k])
            exp_future_cost += (1.0/K) * val
        else:
            # End of Horizon
            exp_future_cost += 0.0

    m.obj = Objective(expr=imm_cost + exp_future_cost, sense=minimize)
    SolverFactory('gurobi').solve(m, tee=False)

    if mode == "value":
        return value(m.obj)
    else:
        return {"HeatPowerRoom1": value(m.p1), "HeatPowerRoom2": value(m.p2), "VentilationON": int(value(m.v))}

# =============================================================================
# 3. MAIN FORWARD-BACKWARD LOOP
# =============================================================================

for i in range(ITERATIONS):
    print(f"\n--- FORWARD-BACKWARD ITERATION {i} ---")
    
    # -------------------------------------------------------------------------
    # FORWARD PASS: Sampling states based on the CURRENT policy (weights)
    # -------------------------------------------------------------------------
    print(f"Forward Pass: Simulating {NUM_DAYS_SIMULATION} days with current VFA...")
    states_by_time = {t: [] for t in range(T_HOURS)}
    
    for day in range(NUM_DAYS_SIMULATION):
        state = get_fixed_data().copy()
        
        for t in range(T_HOURS):
            state['current_time'] = t
            states_by_time[t].append(state.copy())
            
            # Choose Action: Epsilon-Greedy Exploration
            if np.random.rand() < EPSILON or i == 0:
                # Random exploration or initial cold start
                decision = {
                    "HeatPowerRoom1": np.random.uniform(0, data['heating_max_power']),
                    "HeatPowerRoom2": np.random.uniform(0, data['heating_max_power']),
                    "VentilationON": np.random.choice([0, 1])
                }
            else:
                # Exploitation: Use current VFA weights to pick the best action
                w_next = current_weights[t+1] if t < T_HOURS-1 else None
                decision = solve_one_step_milp(state, w_next, mode="action")
            
            # Apply Dynamics
            state, _ = apply_dynamics(state, decision, data)
            
            # Transition to next state (Exogenous)
            if t + 1 < T_HOURS:
                n_o1, n_o2 = next_occupancy_levels(state['Occ1'], state['Occ2'])
                n_p = price_model(state['price_t'], state['price_previous'])
                state['Occ1'], state['Occ2'] = n_o1, n_o2
                state['price_previous'], state['price_t'] = state['price_t'], n_p

    # -------------------------------------------------------------------------
    # BACKWARD PASS: Updating VFA weights (Fitted Value Iteration)
    # -------------------------------------------------------------------------
    print("Backward Pass: Updating VFA weights using newly visited states...")
    for t in reversed(range(T_HOURS)):
        X, y = [], []
        w_next = current_weights[t+1] if t < T_HOURS-1 else None
        
        for s in states_by_time[t]:
            try:
                # Calculate Target V* for each visited state
                target_v = solve_one_step_milp(s, w_next, mode="value")
                y.append(target_v)
                X.append([s[feat] for feat in feature_cols])
            except:
                continue
        
        # Regression update (Policy Evaluation step)
        if X:
            model = LinearRegression(fit_intercept=True).fit(X, y)
            for idx, feat in enumerate(feature_cols):
                current_weights[t][feat] = model.coef_[idx]
            current_weights[t]['intercept'] = model.intercept_

    print(f"Update: Iteration {i} finished. Weights for t=0 (T1): {current_weights[0]['T1']:.4f}")

# Final result output
print("\nVFA_WEIGHTS_FORWARD_BACKWARD = {")
for t in range(T_HOURS):
    clean_weights = {k: round(float(v), 4) for k, v in current_weights[t].items()}
    print(f"    {t}: {clean_weights},")
print("}")