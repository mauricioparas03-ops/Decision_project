from EnvFunctions import apply_dynamics, check_feasibility
from policies.dummy_policy import select_action as dummy_action
#Import your policy here:
#from policies.dummy_policy import select_action
#from policies.Optimal_in_hindsight_policy import select_action, initialize_policy #UNCOMMENT THIS AND THE INITIALIZATION CALL IN THE SIMULATION LOOP IF YOU WANT TO TEST THE OPTIMAL IN HINDSIGHT POLICY
from policies.hybrid_policy import select_action
from Data.v2_SystemCharacteristics import get_fixed_data
import pandas as pd
from pathlib import Path
import numpy as np

data_dir = Path(__file__).resolve().parent / "Data"

occ1 = pd.read_csv(data_dir / "OccupancyRoom1.csv", header=None).values.flatten()
occ2 = pd.read_csv(data_dir / "OccupancyRoom2.csv", header=None).values.flatten()

df_prices = pd.read_csv(data_dir / "v2_PriceData.csv") # skip header, as we will access the columns by index

# extract the first column (previous price) as a separate array, if needed for the dynamics or the policy.
daily_previous_prices = df_prices.iloc[:, 0].values 
# extract the columns from 1 to 10: 100x10 matrix of hourly prices, flattened to 1000 elements
prices = df_prices.iloc[:, 1:11].values.flatten()

data = get_fixed_data()
power_max = {1: data['heating_max_power'], 2: data['heating_max_power']}


E_days = 100
T_hours = 10
daily_costs = np.zeros(E_days)

for day in range(E_days):
    # extracting the exact datas for the current day from the full trajectories
    day_occ1 = occ1[day * T_hours : (day + 1) * T_hours]
    day_occ2 = occ2[day * T_hours : (day + 1) * T_hours]
    day_prices = prices[day * T_hours : (day + 1) * T_hours]
    day_prev_price = daily_previous_prices[day] #first hour of the day, to be used as price_previous in the initial state

    #IF TESTING OPTIMAL IN HINDSIGHT POLICY, REMEMBER TO CALL initialize_policy(day_prices, day_occ1, day_occ2) BEFORE THE SIMULATION LOOP
    #initialize_policy(data, day_prices, day_occ1, day_occ2)

    state = {
        "T1": data['T1'], #Temperature of room 1
        "T2": data['T2'], #Temperature of room 2
        "H": data['H'], #Humidity
        "Occ1": day_occ1[0], #Occupancy of room 1
        "Occ2": day_occ2[0], #Occupancy of room 2
        "price_t": day_prices[0], #Price
        "price_previous": day_prev_price, #Previous Price
        "vent_counter": data['vent_counter'], #For how many consecutive hours has the ventilation been on
        "low_override_r1": data['low_override_r1'], #Is the low-temperature overrule controller of room 1 active
        "low_override_r2": data['low_override_r2'], #Is the low-temperature overrule controller of room 2 active
        "current_time": 0 #What is the hour of the day
    }



    cost_of_this_day = 0.0

    for t in range(T_hours):
        # DECISION (Here-and-now)
        decision = select_action(state)

        # VERIFY FEASIBILITY
        is_feasible = check_feasibility(decision, power_max)
        if not is_feasible:
            print(f"Day {day}, Time {t}: Infeasible! Using dummy.")
            decision = dummy_action(state)

        # COST AND DYNAMICS
        #cost afteroverrules
        state, real_cost = apply_dynamics(state, decision, data)
        cost_of_this_day += real_cost
    
    # 6. TRANSITION (exogenous - Uncertainty "Real" revealed by the historical CSV data)
        if t + 1 < T_hours:
            state['Occ1'] = day_occ1[t + 1]
            state['Occ2'] = day_occ2[t + 1]
            state['price_previous'] = state['price_t']
            state['price_t'] = day_prices[t + 1]
            # current_time already updated in apply_dynamics, so it will automatically move to the next hour in the next iteration
    
        if state["T1"] < 18 or state["T2"] < 18:
            print(f"ora: {t}, day: {day}, {state['T1']:.2f}, {state['T2']:.2f}, {state['H']:.2f}, {state['vent_counter']}, {state['low_override_r1']}, {state['low_override_r2']}")   
        #print(decision)

    # Save the total cost of this day
    daily_costs[day] = cost_of_this_day

    

print(f"Cost average over {E_days} days: {np.mean(daily_costs):.2f}")