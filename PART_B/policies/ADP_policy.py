from pyomo.environ import *
from Data.v2_SystemCharacteristics import get_fixed_data


# definitive wheights for the VFA, obtained after training on 500 days with a linear regression on the collected data
VFA_WEIGHTS = {
    'T1': -4.2525,               # AVERAGE OF THE 2 ROOMS
    'T2': -4.2525,               # AVERAGE OF THE 2 ROOMS
    'H': -1.3122,
    'price_t': 22.0537,
    'vent_counter': -7.9100,
    'low_override_r1': 10.2428,  # AVERAGE OF THE 2 ROOMS
    'low_override_r2': 10.2428   # AVERAGE OF THE 2 ROOMS
}

def select_action(state):
    data = get_fixed_data()
    m = ConcreteModel()

    # DECISION VARIABLES
    m.p1 = Var(bounds=(0, data['heating_max_power']))
    m.p2 = Var(bounds=(0, data['heating_max_power']))
    m.v = Var(domain=Binary)

    m.T1_next = Var()
    m.T2_next = Var()
    m.H_next = Var()

    #OVERRULE FOR CURRENT STEP
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
    
    #DYNAMICS OF NEXT STATE
    tout = data['outdoor_temperature'][int(state['current_time'])]

    m.c_t1 = Constraint(expr=m.T1_next == state['T1'] + data['heat_exchange_coeff'] * (state['T2'] - state['T1']) 
                        + data['thermal_loss_coeff'] * (tout - state['T1']) + data['heating_efficiency_coeff'] * m.p1 
                        - data['heat_vent_coeff'] * m.v + data['heat_occupancy_coeff'] * state['Occ1'])

    m.c_t2 = Constraint(expr=m.T2_next == state['T2'] + data['heat_exchange_coeff'] * (state['T1'] - state['T2']) 
                        + data['thermal_loss_coeff'] * (tout - state['T2']) + data['heating_efficiency_coeff'] * m.p2 
                        - data['heat_vent_coeff'] * m.v + data['heat_occupancy_coeff'] * state['Occ2'])

    m.c_h = Constraint(expr=m.H_next == state['H'] - data['humidity_vent_coeff'] * m.v 
                       + data['humidity_occupancy_coeff'] * (state['Occ1'] + state['Occ2']))

    m.vent_counter_next = Var(domain=NonNegativeReals)
    m.c_vc = Constraint(expr=m.vent_counter_next == (state['vent_counter'] + 1) * m.v)

    #MAPPING FOR FUTURE OVERRULES
    M = 100
    eps = 0.001 # small constant to avoid numerical issues in the thresholding of the overrules (e.g., if T1 is exactly equal to temp_min_comfort_threshold, we want to be sure that we are in the low-temperature override state)
    
    thresh1 = (data['temp_min_comfort_threshold'] if state['low_override_r1'] == 0 else data['temp_OK_threshold']) + eps
    m.low_override_r1_next = Var(domain=Binary)
    m.c_low1_a = Constraint(expr=m.T1_next >= thresh1 - M * m.low_override_r1_next)
    m.c_low1_b = Constraint(expr=m.T1_next <= thresh1 + M * (1 - m.low_override_r1_next))

    thresh2 = (data['temp_min_comfort_threshold'] if state['low_override_r2'] == 0 else data['temp_OK_threshold']) + eps
    m.low_override_r2_next = Var(domain=Binary)
    m.c_low2_a = Constraint(expr=m.T2_next >= thresh2 - M * m.low_override_r2_next)
    m.c_low2_b = Constraint(expr=m.T2_next <= thresh2 + M * (1 - m.low_override_r2_next))

    #APPROXIMATE VALUE FUNCTION 
    immediate_cost = state['price_t'] * (m.p1 + m.p2 + m.v * data['ventilation_power'])
    
    future_cost = (
        VFA_WEIGHTS['T1'] * m.T1_next + VFA_WEIGHTS['T2'] * m.T2_next +
        VFA_WEIGHTS['H'] * m.H_next + VFA_WEIGHTS['vent_counter'] * m.vent_counter_next +
        VFA_WEIGHTS['low_override_r1'] * m.low_override_r1_next +
        VFA_WEIGHTS['low_override_r2'] * m.low_override_r2_next
    )

    m.obj = Objective(expr=immediate_cost + future_cost, sense=minimize)

    #SOLVER
    solver = SolverFactory('gurobi')
    solver.solve(m, tee=False)

    return {
        "HeatPowerRoom1": value(m.p1),
        "HeatPowerRoom2": value(m.p2),
        "VentilationON": int(value(m.v) + 0.5)
    }