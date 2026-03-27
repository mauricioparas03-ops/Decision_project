from EnvFunctions import apply_dynamics, check_feasibility, cost_function
from policies.dummy_policy import select_action as dummy_action
#Import your policy here:
from policies.dummy_policy import select_action
from Data.SystemCharacteristics import get_fixed_data
import pandas as pd
from pathlib import Path

data_dir = Path(__file__).resolve().parent / "Data"

occ1 = pd.read_csv(data_dir / "OccupancyRoom1.csv", header=None).values.flatten()
occ2 = pd.read_csv(data_dir / "OccupancyRoom2.csv", header=None).values.flatten()
prices = pd.read_csv(data_dir / "PriceData.csv", header=None).values.flatten()

power_max = {1: 3, 2:3}
data = get_fixed_data()
ventilation_power = data['ventilation_power']

state = {
    "T1": data['initial_temperature'], #Temperature of room 1
    "T2": data['initial_temperature'], #Temperature of room 2
    "H": data['initial_humidity'], #Humidity
    "Occ1": occ1[0], #Occupancy of room 1
    "Occ2": occ2[0], #Occupancy of room 2
    "price_t": prices[0], #Price
    "price_previous": prices[0], #Previous Price
    "vent_counter": 0, #For how many consecutive hours has the ventilation been on
    "low_override_r1": 0, #Is the low-temperature overrule controller of room 1 active
    "low_override_r2": 0, #Is the low-temperature overrule controller of room 2 active
    "current_time": 0 #What is the hour of the day
}

for t in range(1, 100*10):
    decision = select_action(state)

    #check feasibility and if it is not feasible, use dummy_decision
    is_feasible = check_feasibility(decision, power_max)


    if not is_feasible:
        print(f"Decision at time {t} is not feasible. Using dummy decision.")
        decision = dummy_action(state)

    cost = cost_function(decision, state, ventilation_power)
    state = apply_dynamics(state, decision, data)
    