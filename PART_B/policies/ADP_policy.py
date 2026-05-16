from pyomo.environ import *
from Data.v2_SystemCharacteristics import get_fixed_data
from Data.PriceProcessRestaurant import price_model
from Data.OccupancyProcessRestaurant import next_occupancy_levels 

VFA_WEIGHTS = {
    0: {'T1': -15.1817, 'T2': -11.0521, 'H': 39.4454, 'price_t': 212.7638, 'price_previous': -63.8921, 'Occ1': 1.1614, 'Occ2': 2.6545, 'vent_counter': 0.0, 'low_override_r1': 0.0, 'low_override_r2': 0.0, 'intercept': 61.2015},
    1: {'T1': -14.1244, 'T2': -13.8221, 'H': 32.051, 'price_t': 177.6584, 'price_previous': 21.2621, 'Occ1': 3.8173, 'Occ2': 1.1884, 'vent_counter': 29.6988, 'low_override_r1': 29.0793, 'low_override_r2': 39.2134, 'intercept': 15.148},
    2: {'T1': -9.1318, 'T2': -9.182, 'H': 24.4011, 'price_t': 126.1836, 'price_previous': 61.5023, 'Occ1': 5.7478, 'Occ2': 3.3838, 'vent_counter': 16.4466, 'low_override_r1': 22.6582, 'low_override_r2': 24.7516, 'intercept': 10.4387},
    3: {'T1': -17.557, 'T2': -16.5797, 'H': 15.3877, 'price_t': 99.3028, 'price_previous': 59.9706, 'Occ1': 6.1186, 'Occ2': 4.0846, 'vent_counter': 2.3173, 'low_override_r1': 12.7133, 'low_override_r2': 14.0686, 'intercept': 11.5319},
    4: {'T1': -26.0576, 'T2': -27.876, 'H': 11.4536, 'price_t': 77.0006, 'price_previous': 50.8787, 'Occ1': 5.2653, 'Occ2': 2.6675, 'vent_counter': 0.5803, 'low_override_r1': 7.1369, 'low_override_r2': 6.5221, 'intercept': 10.2653},
    5: {'T1': -22.763, 'T2': -22.5669, 'H': 10.6605, 'price_t': 51.6956, 'price_previous': 30.0293, 'Occ1': 2.148, 'Occ2': 1.9294, 'vent_counter': -2.972, 'low_override_r1': 13.2232, 'low_override_r2': 10.1819, 'intercept': 10.8758},
    6: {'T1': -14.6925, 'T2': -15.7399, 'H': 10.6009, 'price_t': 38.4131, 'price_previous': 17.1451, 'Occ1': 1.4967, 'Occ2': 3.4037, 'vent_counter': -2.0077, 'low_override_r1': 19.8444, 'low_override_r2': 15.3903, 'intercept': 4.8376},
    7: {'T1': -13.2809, 'T2': -13.4543, 'H': 8.1404, 'price_t': 37.288, 'price_previous': 15.1328, 'Occ1': -3.1742, 'Occ2': -0.3237, 'vent_counter': -2.7326, 'low_override_r1': 16.1651, 'low_override_r2': 19.3024, 'intercept': 0.6569},
    8: {'T1': -8.2088, 'T2': -7.0544, 'H': 8.1414, 'price_t': 37.3914, 'price_previous': 19.8036, 'Occ1': -1.6459, 'Occ2': 0.1354, 'vent_counter': -1.6509, 'low_override_r1': 19.0625, 'low_override_r2': 18.5755, 'intercept': -11.3484},
    9: {'T1': 0.0892, 'T2': 0.0221, 'H': 5.9628, 'price_t': 25.3733, 'price_previous': 15.9955, 'Occ1': 0.5999, 'Occ2': 0.5879, 'vent_counter': -1.6118, 'low_override_r1': 11.2778, 'low_override_r2': 11.1858, 'intercept': -12.1428},
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