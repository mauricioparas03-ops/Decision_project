from Checks import check_and_sanitize_action
from Classes import RestaurantValidator
from Policies import Policies
for t in range(1, 100):
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
    dummy_decision = RestaurantValidator.get_dummy_action(init_state, data['demand_schedule'][t], data)
    state = RestaurantValidator.apply_dynamics(init_state, dummy_decision, data)

    decision = Policies.select_action(state)

    #check feasibility with checks.py and if it is not feasible, use dummy_decision
    feasible_decision = check_and_sanitize_action(policy, state, power_max)


    cost = RestaurantValidator.cost_function(decision, state['lambda_grid'])

    next_state = RestaurantValidator.apply_dynamics(state, decision, data) 
    