from pyomo.environ import *
from Data.v2_SystemCharacteristics import get_fixed_data
from Data.PriceProcessRestaurant import price_model
from Data.OccupancyProcessRestaurant import next_occupancy_levels 

VFA_WEIGHTS = {
    0: {'T1': -18.7167, 'T2': -15.3049, 'H': 36.3739, 'price_t': 174.3789, 'price_previous': -53.1764, 'Occ1': -4.6999, 'Occ2': 0.1533, 'vent_counter': 0.0, 'low_override_r1': 0.0, 'low_override_r2': 0.0, 'intercept': 74.3299},
    1: {'T1': -18.1261, 'T2': -21.7064, 'H': 39.7965, 'price_t': 145.3838, 'price_previous': 13.8824, 'Occ1': -0.0434, 'Occ2': 1.5164, 'vent_counter': 18.5453, 'low_override_r1': 3.5211, 'low_override_r2': 17.1537, 'intercept': 32.4133},
    2: {'T1': -19.612, 'T2': -12.6947, 'H': 36.8506, 'price_t': 105.5618, 'price_previous': 48.8964, 'Occ1': 2.2471, 'Occ2': -1.1857, 'vent_counter': 31.9703, 'low_override_r1': 21.7403, 'low_override_r2': 21.3257, 'intercept': 20.4053},
    3: {'T1': -9.9476, 'T2': -10.9026, 'H': 28.325, 'price_t': 84.8963, 'price_previous': 55.5871, 'Occ1': 2.9794, 'Occ2': -1.2659, 'vent_counter': 15.7752, 'low_override_r1': 20.5333, 'low_override_r2': 23.9453, 'intercept': 17.4612},
    4: {'T1': -10.8574, 'T2': -11.6557, 'H': 18.8052, 'price_t': 70.5663, 'price_previous': 45.5688, 'Occ1': -0.0426, 'Occ2': -2.627, 'vent_counter': -0.8769, 'low_override_r1': 24.0487, 'low_override_r2': 20.2653, 'intercept': 22.9409},
    5: {'T1': -8.6935, 'T2': -9.9432, 'H': 15.7539, 'price_t': 62.6761, 'price_previous': 41.5705, 'Occ1': -0.8214, 'Occ2': -1.4374, 'vent_counter': -3.7855, 'low_override_r1': 16.8457, 'low_override_r2': 23.2125, 'intercept': 15.8114},
    6: {'T1': -8.8508, 'T2': -6.6035, 'H': 16.3716, 'price_t': 53.5042, 'price_previous': 35.3989, 'Occ1': 0.3332, 'Occ2': -1.5543, 'vent_counter': -2.8252, 'low_override_r1': 17.4996, 'low_override_r2': 16.5994, 'intercept': 6.1139},
    7: {'T1': -8.1283, 'T2': -8.1708, 'H': 16.0716, 'price_t': 39.5477, 'price_previous': 28.9486, 'Occ1': 0.0778, 'Occ2': -2.2621, 'vent_counter': -0.5009, 'low_override_r1': 15.9565, 'low_override_r2': 15.3527, 'intercept': -3.5235},
    8: {'T1': -6.3418, 'T2': -5.9387, 'H': 12.9215, 'price_t': 25.4014, 'price_previous': 18.8155, 'Occ1': -0.4181, 'Occ2': -2.5778, 'vent_counter': -0.2622, 'low_override_r1': 15.4762, 'low_override_r2': 15.6067, 'intercept': -6.4093},
    9: {'T1': 0.9961, 'T2': 0.2475, 'H': 6.7062, 'price_t': 6.9417, 'price_previous': 5.0313, 'Occ1': 0.4365, 'Occ2': -0.0945, 'vent_counter': -0.3673, 'low_override_r1': 13.0701, 'low_override_r2': 13.6237, 'intercept': -3.3634},
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