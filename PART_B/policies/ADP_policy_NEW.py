from pyomo.environ import *
from Data.v2_SystemCharacteristics import get_fixed_data
from Data.PriceProcessRestaurant import price_model
from Data.OccupancyProcessRestaurant import next_occupancy_levels 


VFA_WEIGHTS = {
    0: {'T1': 0.0, 'T2': -0.0, 'H': 0.0, 'price_t': 35.9197, 'price_previous': -19.1043, 'Occ1': -0.3504, 'Occ2': -0.4823, 'vent_counter': 0.0, 'low_override_r1': 0.0, 'low_override_r2': 0.0, 'intercept': 41.5499},
    1: {'T1': -8.4267, 'T2': -5.1972, 'H': -0.705, 'price_t': 35.5329, 'price_previous': -20.4844, 'Occ1': 0.5823, 'Occ2': 0.8666, 'vent_counter': -25.9457, 'low_override_r1': 0.0, 'low_override_r2': 0.0, 'intercept': 302.7767},
    2: {'T1': -5.8296, 'T2': -8.2715, 'H': -0.1894, 'price_t': 38.0654, 'price_previous': -23.0358, 'Occ1': 0.3091, 'Occ2': 0.8031, 'vent_counter': -22.1988, 'low_override_r1': 21.3882, 'low_override_r2': 50.3856, 'intercept': 293.3101},
    3: {'T1': -6.8466, 'T2': -4.4059, 'H': 0.5919, 'price_t': 38.8714, 'price_previous': -22.6028, 'Occ1': 0.2916, 'Occ2': 0.9008, 'vent_counter': -8.1946, 'low_override_r1': 15.3258, 'low_override_r2': 29.3167, 'intercept': 175.2718},
    4: {'T1': -5.237, 'T2': -4.1989, 'H': 1.0355, 'price_t': 32.7017, 'price_previous': -15.9855, 'Occ1': 0.0574, 'Occ2': 0.187, 'vent_counter': 0.9435, 'low_override_r1': 11.2276, 'low_override_r2': 25.7309, 'intercept': 115.3034},
    5: {'T1': -4.6849, 'T2': -6.3935, 'H': 0.7406, 'price_t': 25.8907, 'price_previous': -11.3308, 'Occ1': -0.1703, 'Occ2': 0.0171, 'vent_counter': 1.3153, 'low_override_r1': 11.0832, 'low_override_r2': 19.718, 'intercept': 169.5322},
    6: {'T1': -6.81, 'T2': -3.9558, 'H': 0.4173, 'price_t': 20.385, 'price_previous': -8.9296, 'Occ1': 0.0207, 'Occ2': 0.0188, 'vent_counter': 1.3412, 'low_override_r1': 13.1749, 'low_override_r2': 20.5421, 'intercept': 176.8147},
    7: {'T1': -2.8955, 'T2': -5.3523, 'H': 0.2851, 'price_t': 15.9023, 'price_previous': -7.0113, 'Occ1': -0.1154, 'Occ2': 0.6538, 'vent_counter': 0.2889, 'low_override_r1': 11.3646, 'low_override_r2': 23.0011, 'intercept': 123.7202},
    8: {'T1': -3.157, 'T2': -1.1498, 'H': 0.2721, 'price_t': 7.0695, 'price_previous': -0.2905, 'Occ1': -0.0921, 'Occ2': 0.0406, 'vent_counter': 1.6258, 'low_override_r1': 16.4888, 'low_override_r2': 16.9401, 'intercept': 60.5083},
    9: {'T1': -0.2234, 'T2': -0.3708, 'H': 0.1329, 'price_t': 2.5882, 'price_previous': 0.4347, 'Occ1': 0.0162, 'Occ2': 0.0831, 'vent_counter': 1.5048, 'low_override_r1': 11.965, 'low_override_r2': 12.1878, 'intercept': -6.7958},
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

    # VFA Trust Region variables
    m.T1_vfa = Var()
    m.T2_vfa = Var()

    # OVERRULE FOR CURRENT STEP (Ora t)
    if state['T1'] > data['temp_max_comfort_threshold']:
        m.p1.fix(0)
    elif state['T1'] < data['temp_min_comfort_threshold'] or state['low_override_r1'] == 1:
        m.p1.fix(data['heating_max_power'])

    if state['T2'] > data['temp_max_comfort_threshold']:
        m.p2.fix(0)
    elif state['T2'] < data['temp_min_comfort_threshold'] or state['low_override_r2'] == 1:
        m.p2.fix(data['heating_max_power'])

    if state['H'] > data['humidity_threshold'] or state['vent_counter'] in [1, 2]:
        m.v.fix(1)
    
    # DYNAMICS OF NEXT STATE
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

    # ==========================================
    # RIGOROUS OVERRIDE LOGIC FOR FUTURE (As per Training and Task 1)
    # ==========================================
    M, eps = 500, 0.001 
    
    # --- Room 1 ---
    m.y_low_r1_next = Var(domain=Binary)
    m.y_ok_r1_next = Var(domain=Binary)
    m.ov1_next = Var(domain=Binary) # Equivalent to your old low_override_r1_next

    m.c_ylow_r1_a = Constraint(expr=m.T1_next <= data['temp_min_comfort_threshold'] + eps + M*(1 - m.y_low_r1_next))
    m.c_ylow_r1_b = Constraint(expr=m.T1_next >= data['temp_min_comfort_threshold'] + eps - M*m.y_low_r1_next)
    
    m.c_yok_r1_a = Constraint(expr=m.T1_next >= data['temp_OK_threshold'] - M*(1 - m.y_ok_r1_next))
    m.c_yok_r1_b = Constraint(expr=m.T1_next <= data['temp_OK_threshold'] + M*m.y_ok_r1_next)

    u_prev_r1 = state['low_override_r1']
    m.c_u_r1_a = Constraint(expr=m.ov1_next >= m.y_low_r1_next)
    m.c_u_r1_b = Constraint(expr=m.ov1_next <= u_prev_r1 + m.y_low_r1_next)
    m.c_u_r1_c = Constraint(expr=m.ov1_next >= u_prev_r1 - m.y_ok_r1_next)
    m.c_u_r1_d = Constraint(expr=m.ov1_next <= 1 - m.y_ok_r1_next)

    # --- Room 2 ---
    m.y_low_r2_next = Var(domain=Binary)
    m.y_ok_r2_next = Var(domain=Binary)
    m.ov2_next = Var(domain=Binary)

    m.c_ylow_r2_a = Constraint(expr=m.T2_next <= data['temp_min_comfort_threshold'] + eps + M*(1 - m.y_low_r2_next))
    m.c_ylow_r2_b = Constraint(expr=m.T2_next >= data['temp_min_comfort_threshold'] + eps - M*m.y_low_r2_next)
    
    m.c_yok_r2_a = Constraint(expr=m.T2_next >= data['temp_OK_threshold'] - M*(1 - m.y_ok_r2_next))
    m.c_yok_r2_b = Constraint(expr=m.T2_next <= data['temp_OK_threshold'] + M*m.y_ok_r2_next)

    u_prev_r2 = state['low_override_r2']
    m.c_u_r2_a = Constraint(expr=m.ov2_next >= m.y_low_r2_next)
    m.c_u_r2_b = Constraint(expr=m.ov2_next <= u_prev_r2 + m.y_low_r2_next)
    m.c_u_r2_c = Constraint(expr=m.ov2_next >= u_prev_r2 - m.y_ok_r2_next)
    m.c_u_r2_d = Constraint(expr=m.ov2_next <= 1 - m.y_ok_r2_next)

    # ==========================================
    # APPROXIMATE VALUE FUNCTION 
    # ==========================================
    immediate_cost = state['price_t'] * (m.p1 + m.p2 + m.v * data['ventilation_power'])
    
    t = int(state['current_time'])

    if t < 9:
        w = VFA_WEIGHTS[t+1]
        
        # Dynamic Trust Region for T1
        if w['T1'] < 0:
            m.c_vfa_t1_a = Constraint(expr=m.T1_vfa <= m.T1_next)
            m.c_vfa_t1_b = Constraint(expr=m.T1_vfa <= data['temp_max_comfort_threshold'])
        else:
            m.c_vfa_t1 = Constraint(expr=m.T1_vfa == m.T1_next)

        # Dynamic Trust Region for T2
        if w['T2'] < 0:
            m.c_vfa_t2_a = Constraint(expr=m.T2_vfa <= m.T2_next)
            m.c_vfa_t2_b = Constraint(expr=m.T2_vfa <= data['temp_max_comfort_threshold'])
        else:
            m.c_vfa_t2 = Constraint(expr=m.T2_vfa == m.T2_next)

        # Forecasts for exogenous variables at step t+1
        expected_price_next = price_model(state['price_t'], state['price_previous'])
        expected_occ1_next, expected_occ2_next = next_occupancy_levels(state['Occ1'], state['Occ2'])
        
        expected_future_cost = (
            w['intercept'] +   
            w['T1'] * m.T1_vfa +  
            w['T2'] * m.T2_vfa + 
            w['H'] * m.H_next +
            w['vent_counter'] * m.vent_counter_next + 
            w['low_override_r1'] * m.ov1_next + 
            w['low_override_r2'] * m.ov2_next +
            w['price_t'] * expected_price_next +
            w['price_previous'] * state['price_t'] + # The price_t of today is the price_previous of tomorrow
            w['Occ1'] * expected_occ1_next +
            w['Occ2'] * expected_occ2_next
        )
    else:
        # end of the day: future cost is zero 
        expected_future_cost = 0.0

    m.obj = Objective(expr=immediate_cost + expected_future_cost, sense=minimize)

    # SOLVER
    solver = SolverFactory('gurobi')
    solver.solve(m, tee=False)

    return {
        "HeatPowerRoom1": value(m.p1),
        "HeatPowerRoom2": value(m.p2),
        "VentilationON": int(value(m.v))
    }