from Checks import check_and_sanitize_action
from EnvFunctions import *
from Dummy_policy import dummy_action
#Import your policy here:
from "policy_name".py import select_action

power_max = {1: 3, 2:3}

for d in range(1, 100):
    init_state = {
        "T1": ..., #Temperature of room 1
        "T2": ..., #Temperature of room 2
        "H": ..., #Humidity
        "Occ1": ..., #Occupancy of room 1
        "Occ2": ..., #Occupancy of room 2
        "price_t": ..., #Price
        "price_previous": ..., #Previous Price
        "vent_counter": ..., #For how many consecutive hours has the ventilation been on 
        "low_override_r1": ..., #Is the low-temperature overrule controller of room 1 active 
        "low_override_r2": ..., #Is the low-temperature overrule controller of room 2 active 
        "current_time": ... #What is the hour of the day
    }
    if d == 1:
        state = init_state
    else:
        state = next_state

    decision = select_action(state)

    #check feasibility and if it is not feasible, use dummy_decision
    feasible_decision = check_feasibility(decision, power_max)


    if not feasible_decision:
        print(f"Decision at time {d} is not feasible. Using dummy decision.")
        decision = dummy_action(state)


    cost = cost_function(feasible_decision, state["price_t"])

    next_state = apply_dynamics(state, feasible_decision)
    