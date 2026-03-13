from SystemCharacteristics import get_fixed_data
    
def apply_dynamics(self, state, action, data):
    # Here you can implement the dynamics of the system, i.e., how the state evolves given the current state and action.
    # For example, you can use a simple linear model or a more complex one based on physical principles.
    # This is just a placeholder and should be replaced with the actual dynamics of your system.
    next_state = state.copy()  # Start with the current state
    # Update next_state based on action and data
    return next_state
def cost_function(decisions, state):
    params = get_fixed_data()

    ventilation_power = params["ventilation_power"]
    cost = state["price_t"] * (ventilation_power * decisions["v"] + decisions["p1"] + decisions["p2"])
    return cost


