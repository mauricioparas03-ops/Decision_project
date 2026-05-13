from pyomo.environ import *
from Data.v2_SystemCharacteristics import get_fixed_data
from Data.PriceProcessRestaurant import price_model
from Data.OccupancyProcessRestaurant import next_occupancy_levels 


VFA_WEIGHTS = {
    0: {'T1': 0.0, 'T2': -0.0, 'H': 0.0, 'price_t': 26.825, 'price_previous': -17.4572, 'Occ1': 1.1663, 'Occ2': 1.2161, 'vent_counter': 0.0, 'low_override_r1': 0.0, 'low_override_r2': 0.0, 'intercept': 61.0359},
    1: {'T1': -0.8019, 'T2': -1.0771, 'H': 9.258, 'price_t': 29.4533, 'price_previous': -18.4912, 'Occ1': -1.1551, 'Occ2': -0.7443, 'vent_counter': 141.6396, 'low_override_r1': 0.0, 'low_override_r2': 0.0, 'intercept': -271.8814},
    2: {'T1': -1.2397, 'T2': -1.682, 'H': 1.1047, 'price_t': 33.4371, 'price_previous': -21.8444, 'Occ1': -0.6305, 'Occ2': 0.3539, 'vent_counter': -6.4357, 'low_override_r1': 75.7665, 'low_override_r2': 61.5993, 'intercept': 93.57},
    3: {'T1': -4.7691, 'T2': 0.8629, 'H': -0.4243, 'price_t': 34.2862, 'price_previous': -20.8241, 'Occ1': 0.7083, 'Occ2': 1.4613, 'vent_counter': -25.8287, 'low_override_r1': 39.3282, 'low_override_r2': 29.3616, 'intercept': 126.8804},
    4: {'T1': 2.2995, 'T2': -3.0611, 'H': 1.2506, 'price_t': 31.1741, 'price_previous': -15.9515, 'Occ1': 0.1228, 'Occ2': 1.0082, 'vent_counter': 3.1702, 'low_override_r1': 36.344, 'low_override_r2': 30.1224, 'intercept': -56.1332},
    5: {'T1': 2.8862, 'T2': -3.5986, 'H': 1.0798, 'price_t': 22.0975, 'price_previous': -8.9023, 'Occ1': 0.0378, 'Occ2': 0.2463, 'vent_counter': 2.5999, 'low_override_r1': 27.4047, 'low_override_r2': 24.5568, 'intercept': -36.5503},
    6: {'T1': -1.3993, 'T2': 1.1512, 'H': 0.5505, 'price_t': 16.2527, 'price_previous': -5.6775, 'Occ1': 0.5306, 'Occ2': 0.3416, 'vent_counter': 3.7591, 'low_override_r1': 23.633, 'low_override_r2': 23.0267, 'intercept': -48.5741},
    7: {'T1': -0.5691, 'T2': -0.8967, 'H': 0.3382, 'price_t': 12.511, 'price_previous': -4.297, 'Occ1': 0.8723, 'Occ2': -0.0633, 'vent_counter': 1.0209, 'low_override_r1': 15.3075, 'low_override_r2': 22.3047, 'intercept': -15.122},
    8: {'T1': -1.0249, 'T2': -0.4937, 'H': 0.3589, 'price_t': 8.1363, 'price_previous': -1.7101, 'Occ1': 0.5735, 'Occ2': 0.2538, 'vent_counter': 1.2943, 'low_override_r1': 9.6584, 'low_override_r2': 18.9561, 'intercept': -19.9921},
    9: {'T1': -0.1823, 'T2': 0.4903, 'H': 0.1723, 'price_t': 3.9434, 'price_previous': 0.1421, 'Occ1': -0.1386, 'Occ2': -0.026, 'vent_counter': 1.6066, 'low_override_r1': 9.8868, 'low_override_r2': 10.3678, 'intercept': -20.732},
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