from pyomo.environ import *
from Data.v2_SystemCharacteristics import get_fixed_data


# definitive wheights for the VFA, obtained after training on 500 days with a linear regression on the collected data
VFA_WEIGHTS = {
    0: {'T1': -4.7728, 'T2': -4.2723, 'H': 2.0153, 'price_t': 63.9635, 'vent_counter': 2.8734, 'low_override_r1': 19.5118, 'low_override_r2': 4.5856, },
    1: {'T1': -3.4651, 'T2': -5.2359, 'H': 1.4317, 'price_t': 40.1735, 'vent_counter': -2.4675, 'low_override_r1': 21.6570, 'low_override_r2': 3.9929, },
    2: {'T1': -5.2616, 'T2': -5.4729, 'H': 1.5572, 'price_t': 31.5471, 'vent_counter': -1.2298, 'low_override_r1': 16.2080, 'low_override_r2': 1.6170, },
    3: {'T1': -5.7597, 'T2': -8.1122, 'H': 1.7312, 'price_t': 26.3297, 'vent_counter': -0.6071, 'low_override_r1': 10.4642, 'low_override_r2': -2.7964, },
    4: {'T1': -7.2416, 'T2': -7.4703, 'H': 1.6601, 'price_t': 22.7137, 'vent_counter': 3.1988, 'low_override_r1': 1.8259, 'low_override_r2': 5.1817, },
    5: {'T1': -5.7665, 'T2': -8.8734, 'H': 1.5656, 'price_t': 19.6383, 'vent_counter': -0.1712, 'low_override_r1': 5.1719, 'low_override_r2': 5.1719, },
    6: {'T1': -5.8720, 'T2': -4.7384, 'H': 1.3976, 'price_t': 14.4499, 'vent_counter': 1.4589, 'low_override_r1': 12.6062, 'low_override_r2': 5.0689, },
    7: {'T1': -4.8330, 'T2': -2.7647, 'H': 0.9841, 'price_t': 12.0165, 'vent_counter': 1.6978, 'low_override_r1': 18.7546, 'low_override_r2': 7.6453, },
    8: {'T1': -0.3268, 'T2': -3.5952, 'H': 0.4287, 'price_t': 7.9712, 'vent_counter': -0.4391, 'low_override_r1': 10.0615, 'low_override_r2': 16.4810, },
    9: {'T1': -0.6808, 'T2': -0.1249, 'H': 0.3436, 'price_t': 4.1506, 'vent_counter': 0.6065, 'low_override_r1': 10.3152, 'low_override_r2': 10.3152, },
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

    m.vent_counter_next = Var(domain=NonNegativeReals)
    m.low_override_r1_next = Var(domain=Binary)
    m.low_override_r2_next = Var(domain=Binary)

    #VFA variables to calculate reward
    m.T1_vfa = Var()
    m.T2_vfa = Var()
    m.H_vfa = Var()

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

    m.c_vc = Constraint(expr=m.vent_counter_next == (state['vent_counter'] + 1) * m.v)

    #MAPPING FOR FUTURE OVERRULES
    M = 100
    eps = 0.001 # small constant to avoid numerical issues in the thresholding of the overrules (e.g., if T1 is exactly equal to temp_min_comfort_threshold, we want to be sure that we are in the low-temperature override state)
    
    thresh1 = (data['temp_min_comfort_threshold'] if state['low_override_r1'] == 0 else data['temp_OK_threshold']) + eps

    m.c_low1_a = Constraint(expr=m.T1_next >= thresh1 - M * m.low_override_r1_next)
    m.c_low1_b = Constraint(expr=m.T1_next <= thresh1 + M * (1 - m.low_override_r1_next))

    thresh2 = (data['temp_min_comfort_threshold'] if state['low_override_r2'] == 0 else data['temp_OK_threshold']) + eps

    m.c_low2_a = Constraint(expr=m.T2_next >= thresh2 - M * m.low_override_r2_next)
    m.c_low2_b = Constraint(expr=m.T2_next <= thresh2 + M * (1 - m.low_override_r2_next))

    T_TARGET = data['temp_max_comfort_threshold']
    # Relaxed humidity target to avoid forcing unnecessary ventilation when we are already below the risk threshold
    H_TARGET = data['humidity_threshold'] - 2.0

    # If w < 0, the solver pushes to maximize T_vfa. We lock it at the target, turning off the incentive beyond that threshold.
    m.c_vfa_t1_a = Constraint(expr=m.T1_vfa <= m.T1_next)
    m.c_vfa_t1_b = Constraint(expr=m.T1_vfa <= T_TARGET)

    m.c_vfa_t2_a = Constraint(expr=m.T2_vfa <= m.T2_next)
    m.c_vfa_t2_b = Constraint(expr=m.T2_vfa <= T_TARGET)

# If w > 0, the solver pushes to minimize H_vfa. We lock it to the target (it doesn't go below it), turning off the incentive.
    m.c_vfa_h_a = Constraint(expr=m.H_vfa >= m.H_next)
    m.c_vfa_h_b = Constraint(expr=m.H_vfa >= H_TARGET)

    #APPROXIMATE VALUE FUNCTION 
    immediate_cost = state['price_t'] * (m.p1 + m.p2 + m.v * data['ventilation_power'])
    

    t = int(state['current_time'])
    #If not at the last hour use weiths to forecast future
    if t < 9:
        w = VFA_WEIGHTS[t+1]
        
        # Cleanup of negative weights on penalty variables
        # If the regression has assigned a negative weight to trigger an alarm, we reset it.
        w_v_count = max(0, w['vent_counter'])
        w_ov1 = max(0, w['low_override_r1'])
        w_ov2 = max(0, w['low_override_r2'])
        
        expected_future_cost = (
            w['T1'] * m.T1_vfa + 
            w['T2'] * m.T2_vfa + 
            w['H'] * m.H_vfa + 
            w_v_count * m.vent_counter_next + 
            w_ov1 * m.low_override_r1_next + 
            w_ov2 * m.low_override_r2_next
        )
    else:
        # end of the day no cost
        expected_future_cost = 0.0

    m.obj = Objective(expr=immediate_cost + expected_future_cost, sense=minimize)

    #SOLVER
    solver = SolverFactory('gurobi')
    solver.solve(m, tee=False)


    return {
        "HeatPowerRoom1": value(m.p1),
        "HeatPowerRoom2": value(m.p2),
        "VentilationON": int(value(m.v))
    }