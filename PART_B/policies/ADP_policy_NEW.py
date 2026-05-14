from pyomo.environ import *
from Data.v2_SystemCharacteristics import get_fixed_data
from Data.PriceProcessRestaurant import price_model
from Data.OccupancyProcessRestaurant import next_occupancy_levels 

VFA_WEIGHTS = {
    0: {'T1': 0.0, 'T2': 0.0, 'H': 0.0, 'price_t': 187.6667, 'price_previous': -62.3012, 'Occ1': 1.0089, 'Occ2': 3.1033, 'vent_counter': 0.0, 'low_override_r1': 0.0, 'low_override_r2': 0.0, 'intercept': 89.887},
    1: {'T1': -9.6641, 'T2': -7.4988, 'H': 6.7825, 'price_t': 151.4188, 'price_previous': 47.0275, 'Occ1': 1.7654, 'Occ2': 4.4254, 'vent_counter': -5.4744, 'low_override_r1': 0.0, 'low_override_r2': 0.0, 'intercept': 44.2343},
    2: {'T1': -20.3638, 'T2': -18.6061, 'H': 14.9716, 'price_t': 106.136, 'price_previous': 64.4833, 'Occ1': 2.9498, 'Occ2': 3.9021, 'vent_counter': -12.2186, 'low_override_r1': 15.8448, 'low_override_r2': 14.7889, 'intercept': 34.7689},
    3: {'T1': -15.6701, 'T2': -15.2939, 'H': 18.9105, 'price_t': 85.1026, 'price_previous': 59.0587, 'Occ1': 6.0099, 'Occ2': 4.0709, 'vent_counter': -12.2869, 'low_override_r1': 19.4058, 'low_override_r2': 15.9349, 'intercept': 31.4082},
    4: {'T1': -14.1999, 'T2': -12.498, 'H': 28.6464, 'price_t': 75.633, 'price_previous': 51.2581, 'Occ1': 2.5462, 'Occ2': -1.8861, 'vent_counter': 0.01, 'low_override_r1': 16.1628, 'low_override_r2': 12.7883, 'intercept': 21.0121},
    5: {'T1': -15.1683, 'T2': -15.928, 'H': 26.523, 'price_t': 61.2616, 'price_previous': 42.5076, 'Occ1': -1.8042, 'Occ2': -0.4127, 'vent_counter': 2.1161, 'low_override_r1': 12.3775, 'low_override_r2': 11.2232, 'intercept': 9.2689},
    6: {'T1': -20.201, 'T2': -14.626, 'H': 18.6769, 'price_t': 50.237, 'price_previous': 30.8139, 'Occ1': -4.5278, 'Occ2': -4.4701, 'vent_counter': 2.5054, 'low_override_r1': 11.4128, 'low_override_r2': 10.2971, 'intercept': 6.3677},
    7: {'T1': -16.8653, 'T2': -14.8129, 'H': 6.5894, 'price_t': 38.6177, 'price_previous': 23.5582, 'Occ1': 1.9337, 'Occ2': 1.865, 'vent_counter': 0.74, 'low_override_r1': 13.7811, 'low_override_r2': 11.6375, 'intercept': -3.3874},
    8: {'T1': -10.1996, 'T2': -11.7665, 'H': 9.009, 'price_t': 29.3922, 'price_previous': 20.8786, 'Occ1': 1.8752, 'Occ2': 0.7728, 'vent_counter': -1.0947, 'low_override_r1': 13.73, 'low_override_r2': 14.4303, 'intercept': -9.3937},
    9: {'T1': -1.7892, 'T2': -1.9195, 'H': 5.0565, 'price_t': 16.2, 'price_previous': 10.4758, 'Occ1': 1.6312, 'Occ2': 0.3141, 'vent_counter': 2.1061, 'low_override_r1': 11.2347, 'low_override_r2': 11.3342, 'intercept': -8.4416},
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
    # RIGOROUS OVERRIDE LOGIC FOR FUTURE 
    # ==========================================
    M, eps = 500, 0.001 
    
    # --- Room 1 ---
    m.y_low_r1_next = Var(domain=Binary)
    m.y_ok_r1_next = Var(domain=Binary)
    m.ov1_next = Var(domain=Binary)

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
        
        # Forecasts for exogenous variables at step t+1
        expected_price_next = price_model(state['price_t'], state['price_previous'])
        expected_occ1_next, expected_occ2_next = next_occupancy_levels(state['Occ1'], state['Occ2'])
        
        # APPLICHIAMO LA NORMALIZZAZIONE DIRETTAMENTE ALLE VARIABILI DEL MILP
        expected_future_cost = (
            w['intercept'] +   
            w['T1'] * ((m.T1_next - 22.0) / 8.0) +  
            w['T2'] * ((m.T2_next - 22.0) / 8.0) + 
            w['H'] * ((m.H_next - 40.0) / 40.0) +
            w['vent_counter'] * (m.vent_counter_next / 3.0) + 
            w['low_override_r1'] * m.ov1_next + 
            w['low_override_r2'] * m.ov2_next +
            w['price_t'] * (expected_price_next / 10.0) +
            w['price_previous'] * (state['price_t'] / 10.0) + 
            w['Occ1'] * ((expected_occ1_next - 20.0) / 30.0) +
            w['Occ2'] * ((expected_occ2_next - 10.0) / 20.0)
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