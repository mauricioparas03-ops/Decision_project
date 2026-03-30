from EnvFunctions import apply_dynamics
from Data.PriceProcessRestaurant import price_model 
from Data.OccupancyProcessRestaurant import next_occupancy_levels
from Data.v2_SystemCharacteristics import get_fixed_data
import policies.Optimal_in_hindsight_policy as oih
import numpy as np

import pandas as pd
from sklearn.linear_model import LinearRegression

NUM_DAYS = 500
NUM_HOURS = 10
training_data = []


for day in range(NUM_DAYS):

    data = get_fixed_data()

    #price and occupancy trajectories for the current day
    daily_prices = [data['price_t']]
    daily_occ1 = [data['Occ1']]
    daily_occ2 = [data['Occ2']]
    
    curr_p = data['price_t']
    prev_p = data['price_previous']
    curr_o1 = data['Occ1']
    curr_o2 = data['Occ2']

    for _ in range(NUM_HOURS - 1):
        next_p = price_model(curr_p, prev_p)
        prev_p, curr_p = curr_p, next_p
        daily_prices.append(curr_p)
        
        curr_o1, curr_o2 = next_occupancy_levels(curr_o1, curr_o2)
        daily_occ1.append(curr_o1)
        daily_occ2.append(curr_o2)

    # solution with MILP with perfect foresight of the current day (optimal in hindsight)
    oih.initialize_policy(data, daily_prices, daily_occ1, daily_occ2)
    try:
        oih.solve_daily_milp()
    except Exception:
        continue

    state = {
        "T1": data['T1'], #Temperature of room 1
        "T2": data['T2'], #Temperature of room 2
        "H": data['H'], #Humidity
        "Occ1": daily_occ1[0], #Occupancy of room 1
        "Occ2": daily_occ2[0], #Occupancy of room 2
        "price_t": daily_prices[0], #Price
        "price_previous": data['price_previous'], #Previous Price
        "vent_counter": data['vent_counter'], #For how many consecutive hours has the ventilation been on
        "low_override_r1": data['low_override_r1'], #Is the low-temperature overrule controller of room 1 active
        "low_override_r2": data['low_override_r2'], #Is the low-temperature overrule controller of room 2 active
        "current_time": 0 #What is the hour of the day
    }

    #tracking of daily cost and state visited during this episode (day)
    hourly_costs = []
    states_visited = []

    for t in range(NUM_HOURS):
        states_visited.append(state.copy())

        #take optimal action from MILP solution
        decision = {
            "HeatPowerRoom1": oih._p1_opt[t],
            "HeatPowerRoom2": oih._p2_opt[t],
            "VentilationON": oih._v_opt[t]
        }

        state, real_cost = apply_dynamics(state, decision, data)
        hourly_costs.append(real_cost)

        if t + 1 < NUM_HOURS:
            state['Occ1'], state['Occ2'] = daily_occ1[t+1], daily_occ2[t+1]
            state['price_previous'], state['price_t'] = daily_prices[t], daily_prices[t+1]
            

    for t in range(NUM_HOURS):
        #target V is the sum of the costs-to-go from this hour until the end of the day 
        future_cost_total = sum(hourly_costs[t:])

        #save the last hour in the dataset
        s_t = states_visited[t]
        dataset_line = {
            "T1": s_t['T1'],
            "T2": s_t['T2'],
            "H": s_t['H'],
            "price_t": s_t['price_t'],
            "vent_counter": s_t['vent_counter'],
            "low_override_r1": s_t['low_override_r1'],
            "low_override_r2": s_t['low_override_r2'],
            'Target_v': future_cost_total   # target for regression
        }

        training_data.append(dataset_line)
    


# MACHINE LEARNING (FITTED VALUE ITERATION)
df = pd.DataFrame(training_data)
feature_cols = [
    "T1", 
    "T2", 
    "H", 
    "price_t", 
    "vent_counter", 
    "low_override_r1", 
    "low_override_r2"
]

X = df[feature_cols]
y = df['Target_v']

#Model training
model = LinearRegression()
model.fit(X, y)

#extract and print the learned coefficients
print(f"Intercept (eta_0): {model.intercept_:.4f}")
print("Coefficient (weight of the features):")
for col, coef in zip(feature_cols, model.coef_):
    print(f"  - {col}: {coef:.4f}")

#accuracy of approximation
score = model.score(X, y)
print(f"\nAccuracy of model (R^2): {score:.4f}")

