from pyomo.environ import *
from Data.v2_SystemCharacteristics import get_fixed_data
from Data.PriceProcessRestaurant import price_model
from Data.OccupancyProcessRestaurant import next_occupancy_levels 

VFA_WEIGHTS = {
    0: {'T1': -10.5156, 'T2': -9.4746, 'H': 32.4973, 'price_t': 215.5152, 'price_previous': -71.0935, 'Occ1': 2.7165, 'Occ2': -2.0341, 'vent_counter': 0.0, 'low_override_r1': 0.0, 'low_override_r2': 0.0, 'intercept': 67.1916},
    1: {'T1': -10.4341, 'T2': -12.742, 'H': 27.6924, 'price_t': 190.4269, 'price_previous': 14.803, 'Occ1': 3.1078, 'Occ2': -3.9389, 'vent_counter': 24.314, 'low_override_r1': 26.5585, 'low_override_r2': 29.4022, 'intercept': 19.0384},
    2: {'T1': -7.356, 'T2': -9.252, 'H': 22.438, 'price_t': 130.5478, 'price_previous': 65.1923, 'Occ1': -0.0628, 'Occ2': -4.0374, 'vent_counter': 16.7436, 'low_override_r1': 19.1924, 'low_override_r2': 17.7823, 'intercept': 16.3154},
    3: {'T1': -20.0074, 'T2': -14.0075, 'H': 17.7255, 'price_t': 99.7337, 'price_previous': 64.4337, 'Occ1': -1.0423, 'Occ2': -0.9566, 'vent_counter': 2.35, 'low_override_r1': 10.5561, 'low_override_r2': 6.9936, 'intercept': 17.71},
    4: {'T1': -27.4527, 'T2': -23.3013, 'H': 10.2402, 'price_t': 74.9055, 'price_previous': 52.2611, 'Occ1': 2.6386, 'Occ2': 4.3732, 'vent_counter': 1.1484, 'low_override_r1': 6.3508, 'low_override_r2': 3.4272, 'intercept': 11.2701},
    5: {'T1': -19.8706, 'T2': -19.7596, 'H': 7.8631, 'price_t': 49.9835, 'price_previous': 32.9094, 'Occ1': 3.7036, 'Occ2': 3.2115, 'vent_counter': -3.1071, 'low_override_r1': 14.8696, 'low_override_r2': 6.9006, 'intercept': 8.9103},
    6: {'T1': -14.9846, 'T2': -15.6679, 'H': 9.8546, 'price_t': 38.9655, 'price_previous': 18.6839, 'Occ1': 2.9592, 'Occ2': 0.8702, 'vent_counter': -2.5272, 'low_override_r1': 17.8959, 'low_override_r2': 11.7734, 'intercept': 5.6151},
    7: {'T1': -14.6893, 'T2': -13.1614, 'H': 9.1806, 'price_t': 36.9371, 'price_previous': 14.4694, 'Occ1': 1.4109, 'Occ2': -0.7224, 'vent_counter': -1.2533, 'low_override_r1': 16.8426, 'low_override_r2': 18.3554, 'intercept': -2.5491},
    8: {'T1': -4.4207, 'T2': -5.4185, 'H': 6.6666, 'price_t': 36.4296, 'price_previous': 18.2789, 'Occ1': -1.8459, 'Occ2': 0.0836, 'vent_counter': 0.3589, 'low_override_r1': 18.5898, 'low_override_r2': 20.8329, 'intercept': -10.1587},
    9: {'T1': 2.6882, 'T2': 2.081, 'H': 4.8251, 'price_t': 23.8798, 'price_previous': 15.9821, 'Occ1': -0.5158, 'Occ2': 0.6465, 'vent_counter': -1.2958, 'low_override_r1': 11.1143, 'low_override_r2': 10.9582, 'intercept': -9.1359},
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
    # NOMI AGGIORNATI: Logica di Isteresi (Memoria del termostato)
    m.c_force_override_ON_if_cold_r1     = Constraint(expr=m.ov1_next >= m.y_low_r1_next)
    m.c_prevent_override_without_cold_r1 = Constraint(expr=m.ov1_next <= u_prev_r1 + m.y_low_r1_next)
    m.c_keep_override_ON_until_ok_r1     = Constraint(expr=m.ov1_next >= u_prev_r1 - m.y_ok_r1_next)
    m.c_force_override_OFF_if_ok_r1      = Constraint(expr=m.ov1_next <= 1 - m.y_ok_r1_next)

    # --- Room 2 ---
    m.y_low_r2_next = Var(domain=Binary)
    m.y_ok_r2_next = Var(domain=Binary)
    m.ov2_next = Var(domain=Binary)

    m.c_ylow_r2_a = Constraint(expr=m.T2_next <= data['temp_min_comfort_threshold'] + eps + M*(1 - m.y_low_r2_next))
    m.c_ylow_r2_b = Constraint(expr=m.T2_next >= data['temp_min_comfort_threshold'] + eps - M*m.y_low_r2_next)
    
    m.c_yok_r2_a = Constraint(expr=m.T2_next >= data['temp_OK_threshold'] - M*(1 - m.y_ok_r2_next))
    m.c_yok_r2_b = Constraint(expr=m.T2_next <= data['temp_OK_threshold'] + M*m.y_ok_r2_next)

    u_prev_r2 = state['low_override_r2']
    # NOMI AGGIORNATI: Logica di Isteresi (Memoria del termostato)
    m.c_force_override_ON_if_cold_r2     = Constraint(expr=m.ov2_next >= m.y_low_r2_next)
    m.c_prevent_override_without_cold_r2 = Constraint(expr=m.ov2_next <= u_prev_r2 + m.y_low_r2_next)
    m.c_keep_override_ON_until_ok_r2     = Constraint(expr=m.ov2_next >= u_prev_r2 - m.y_ok_r2_next)
    m.c_force_override_OFF_if_ok_r2      = Constraint(expr=m.ov2_next <= 1 - m.y_ok_r2_next)
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