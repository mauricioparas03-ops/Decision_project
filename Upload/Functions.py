from pyomo.environ import *
#objective function

def total_cost_rule(model):
    return sum(model.prices[d, t] * (model.Heat[r, t, d] + model.Vent[t, d] * model.Pvent) for r in model.R for t in model.T for d in model.D)

#constraints

#Constraint of room temperature and humidity dynamics for room and room 2

def room_thermal_balance_rule(model, r, t, d):
    # Innitialize the first time step with the initial temperature value
    if t == model.T.first():
        return model.T_in[r, t, d] == model.Tinitial
    
    # Define the previous time step
    # This works if model.T is a Set of sequential integers
    t_prev = t - 1 
    
    # Get occupancy value using iloc for row 0, column t_prev (time index)
    occupancy_value1 = model.Occ1[d, t_prev]
    occupancy_value2 = model.Occ2[d, t_prev]

    if r > 1: 
        room1 = 1
        room2 = 2
    else:
        room1 = 2
        room2 = 1
    
    if r == 1:
        occupancy = occupancy_value1
    else:
        occupancy = occupancy_value2


    # The Equation: T_now = T_prev + Internal_Exchange + External_Loss + Heating
    return model.T_in[r, t, d] == (
        model.T_in[r, t_prev, d] 
         + model.Zexch * (model.T_in[room1, t_prev, d] - model.T_in[room2, t_prev, d]) 
         + model.Zloss * (model.Tout[t_prev] - model.T_in[r, t_prev, d])
        + model.Zconv * (model.Heat[r, t_prev, d]) 
         - model.Zcool * model.Vent[t_prev, d]
         + model.Zocc * occupancy
    )


def humidity_balance_rule(model, t, d):
    # Initialize the first time step with the initial humidity value
    if t == model.T.first():
        return model.Hum[t, d] == model.Hinitial #initial humidity value
    
    # Define the previous time step
    t_prev = t - 1
    
    return model.Hum[t, d] == (
        model.Hum[t_prev, d]
        - model.Hvent * model.Vent[t_prev, d]
        + model.Hocc * (model.Occ1[d, t_prev] + model.Occ2[d, t_prev])
    )

# Max Heat when temperature is too low 
def max_temp_low_rule(block):

    # M is a large constant, larger than any possible temperature difference
    M = 100 
    m = block.model()

    # Constraint to force u = 1 if T_in <= Tmin
    def u_binary_logic_rule(block, r, t, d):
        return m.T_in[r, t, d] >= m.Tmin - M * m.u[r, t, d]
    block.u_logic = Constraint(m.RTD, rule=u_binary_logic_rule)

    # Force Heat to Max if w = 1
    def low_temp_heat_rule(block, r, t, d):
        return m.T_in[r, t, d] >= m.Tok - M * (1 - m.w[r, t, d])
    block.low_temp_heat = Constraint(m.RTD, rule=low_temp_heat_rule)

    # cannot have both u and w active at the same time
    def u_w_exclusivity_rule(block, r, t, d):    
        # Skip the first time step (initial condition)
        if t == m.T.first():
            m.u[r, t, d].fix(0)  # Force u to be 0 at the first time step
            m.w[r, t, d].fix(0)  # Force w to be 0 at the first time step
            return Constraint.Skip # Skip because initialized conditions are within Tlow and THigh at the first time step
        
        # Define the previous time step
        t_prev = t - 1

        return m.u[r, t, d] >= m.u[r,t_prev, d] - m.w[r, t, d]
    block.u_w_exclusivity = Constraint(m.RTD, rule=u_w_exclusivity_rule)

    #Set P to P max 
    def Set_max_power_rule(block, r, t, d):
        return m.Heat[r, t, d] >= m.Pr * m.u[r, t, d]
    block.Set_max_power = Constraint(m.RTD, rule=Set_max_power_rule)

# Power off when temperature is too high
def set_power_off_rule(block):
    # Force Heat to 0 if y = 1
    M = 100
    m = block.model()


    def z_binary_logic_rule(block, r, t, d):
        return m.T_in[r, t, d] <= m.Tok + M * (1 - m.z[r, t, d])
    block.z_logic = Constraint(m.RTD, rule=z_binary_logic_rule)

    def y_binary_logic_rule(block, r, t, d):
        return m.T_in[r, t, d] <= m.Thigh + M * m.y[r, t, d]
    block.y_logic = Constraint(m.RTD, rule=y_binary_logic_rule)

    def z_y_exclusivity_rule(block, r, t, d):    
        # Skip the first time step (initial condition)
        if t == m.T.first():
            m.z[r, t, d].fix(0)  # Force z to be 0 at the first time step
            m.y[r, t, d].fix(0)  # Force y to be 0 at the first time step
            return Constraint.Skip # Skip because initialized conditions are within Tok and Thigh at the first time step
        
        # Define the previous time step
        t_prev = t - 1

        return m.y[r, t, d] >= m.y[r,t_prev, d] - m.z[r, t, d]
    block.z_y_exclusivity = Constraint(m.RTD, rule=z_y_exclusivity_rule)

    def Set_power_off_rule(block, r, t, d):
        return m.Heat[r, t, d] <= m.Pr * (1 - m.y[r, t, d])
    block.Set_power_off = Constraint(m.RTD, rule=Set_power_off_rule)

# Force Vent ON if Hum >= Hhigh
def set_vent_on_rule(model,t,d):
    M = 100
    return model.Hum[t, d] <= model.Hhigh + M * model.Uon[t, d]

#Operational constraints for ventilation system
def on_off_limit_rule(model, t, d):
    return model.Uon[t, d] + model.Uoff[t, d] <= 1

# Constraint 2: U_off <= 1 - e
def off_le_e_rule(model, t, d):
    return model.Uoff[t, d] <=1 - model.Vent[t, d]

# Constraint 3: U_on <= e
def on_le_one_minus_e_rule(model, t, d):
    return model.Uon[t, d] <= model.Vent[t, d]

def set_vent_inertia_rule(model,t,d):
   if t == model.T.first():

      return Constraint.Skip # Skip because there is no previous time step at the first time step
   # Define the previous time step
   t_prev = t - 1
   return model.Uon[t, d] >= model.Vent[t, d] - model.Vent[t_prev, d]


### check that this funciton is working
def min_up_time_ventilation_rule(model, t, d):
    # We can't check 't-1' for the very first hour
    if t == model.T.first():
        return Constraint.Skip
    
    # Define the 'Up-Time' duration
    L = 3 
    
    # If we are too close to the end of the day to look L steps ahead,
    # we adjust the summation range to not exceed the time set.
    remaining_steps = [k for k in range(t, t + L) if k <= model.T.last()]
    
    # Logic: Sum of future states >= duration * (Current - Previous)
    return sum(model.Vent[k, d] for k in remaining_steps) >= \
           len(remaining_steps) * (model.Vent[t, d] - model.Vent[t-1, d])

# Max Temperature when temperature is too low
def low_to_max_temp_rule(model, t, r, d):
    if model.T_in[r, t, d] <= model.Tmin:
        return model.Heat[r, t, d] == model.Pr

# Power off when temperature is too high
def high_to_zero_temp_rule(model, t, r, d):
    if model.T_in[r, t, d] >= model.Thigh:
        return model.Heat[r, t, d] == 0
    
# High humidity forces ventilation ON
def high_humidity_ventilation_rule(model, t, d):
    if model.Hum[t, d] >= model.Hhigh:
        return model.Vent[t, d] == 1


