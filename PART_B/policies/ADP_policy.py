from pyomo.environ import *
from Data.v2_SystemCharacteristics import get_fixed_data
from Data.PriceProcessRestaurant import price_model
from Data.OccupancyProcessRestaurant import next_occupancy_levels 

VFA_WEIGHTS = {
    0: {'T1': -19.3152, 'T2': -21.1865, 'H': 31.6621, 'price_t': 168.9147, 'price_previous': -50.1968, 'Occ1': -2.5972, 'Occ2': 0.3543, 'vent_counter': 0.0, 'low_override_r1': 0.0, 'low_override_r2': 0.0, 'intercept': 69.9217},
    1: {'T1': -22.4024, 'T2': -18.5014, 'H': 37.4839, 'price_t': 140.5326, 'price_previous': 13.0276, 'Occ1': -2.5744, 'Occ2': -2.5641, 'vent_counter': 14.7577, 'low_override_r1': 11.1136, 'low_override_r2': 12.162, 'intercept': 34.108},
    2: {'T1': -19.091, 'T2': -12.7801, 'H': 34.4687, 'price_t': 98.2028, 'price_previous': 50.8459, 'Occ1': 0.4662, 'Occ2': -2.3167, 'vent_counter': 28.5547, 'low_override_r1': 15.1602, 'low_override_r2': 17.5706, 'intercept': 20.3403},
    3: {'T1': -8.905, 'T2': -6.5532, 'H': 27.4922, 'price_t': 80.716, 'price_previous': 53.2317, 'Occ1': 1.0383, 'Occ2': 1.494, 'vent_counter': 12.1014, 'low_override_r1': 20.569, 'low_override_r2': 17.7721, 'intercept': 18.8862},
    4: {'T1': -6.6766, 'T2': -6.2641, 'H': 18.553, 'price_t': 69.3743, 'price_previous': 42.7147, 'Occ1': 3.6594, 'Occ2': 1.899, 'vent_counter': -1.9364, 'low_override_r1': 20.5159, 'low_override_r2': 15.9175, 'intercept': 21.4158},
    5: {'T1': -5.7534, 'T2': -7.7121, 'H': 14.7422, 'price_t': 62.1846, 'price_previous': 39.0899, 'Occ1': 2.9084, 'Occ2': 2.1016, 'vent_counter': -3.6857, 'low_override_r1': 12.9591, 'low_override_r2': 18.8723, 'intercept': 13.6043},
    6: {'T1': -7.1136, 'T2': -6.4891, 'H': 15.5338, 'price_t': 51.2006, 'price_previous': 35.3677, 'Occ1': 1.6336, 'Occ2': 0.15, 'vent_counter': -1.8828, 'low_override_r1': 17.325, 'low_override_r2': 21.0554, 'intercept': 5.2841},
    7: {'T1': -7.9547, 'T2': -7.2715, 'H': 14.8001, 'price_t': 39.9622, 'price_previous': 28.5365, 'Occ1': -2.0356, 'Occ2': 0.221, 'vent_counter': -0.0837, 'low_override_r1': 16.4371, 'low_override_r2': 17.1074, 'intercept': -3.4841},
    8: {'T1': -5.4891, 'T2': -6.374, 'H': 11.936, 'price_t': 23.8296, 'price_previous': 17.6962, 'Occ1': -1.5938, 'Occ2': -1.7513, 'vent_counter': -0.0117, 'low_override_r1': 14.2927, 'low_override_r2': 16.77, 'intercept': -5.1312},
    9: {'T1': -1.2377, 'T2': -1.7029, 'H': 5.4459, 'price_t': 5.7299, 'price_previous': 4.9878, 'Occ1': 0.9003, 'Occ2': 0.2118, 'vent_counter': -0.1173, 'low_override_r1': 14.661, 'low_override_r2': 15.7373, 'intercept': -5.1025},
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
    M, eps = 500, 10e-6
    
    # --- Room 1 ---
    m.y_low_r1_next = Var(domain=Binary)
    m.y_ok_r1_next = Var(domain=Binary)
    m.ov1_next = Var(domain=Binary)

    m.c_ylow_r1_a = Constraint(expr=m.T1_next <= data['temp_min_comfort_threshold'] + M*(1 - m.y_low_r1_next))
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

    m.c_ylow_r2_a = Constraint(expr=m.T2_next <= data['temp_min_comfort_threshold'] + M*(1 - m.y_low_r2_next))
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
        
        K_POLICY = 15
        scen_data = []
        for _ in range(K_POLICY):
            sc_p = price_model(state['price_t'], state['price_previous'])
            sc_o1, sc_o2 = next_occupancy_levels(state['Occ1'], state['Occ2'])
            scen_data.append({'price_t': sc_p, 'Occ1': sc_o1, 'Occ2': sc_o2})

        expected_future_cost = sum(
            (1.0 / K_POLICY) * (
                w['intercept'] +
                w['T1'] * ((m.T1_next - 22.0) / 8.0) +
                w['T2'] * ((m.T2_next - 22.0) / 8.0) +
                w['H'] * ((m.H_next - 40.0) / 40.0) +
                w['vent_counter'] * (m.vent_counter_next / 3.0) +
                w['low_override_r1'] * m.ov1_next +
                w['low_override_r2'] * m.ov2_next +
                w['price_t'] * (scen_data[k]['price_t'] / 10.0) +
                w['price_previous'] * (state['price_t'] / 10.0) +
                w['Occ1'] * ((scen_data[k]['Occ1'] - 20.0) / 30.0) +
                w['Occ2'] * ((scen_data[k]['Occ2'] - 10.0) / 20.0)
            ) for k in range(K_POLICY)
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