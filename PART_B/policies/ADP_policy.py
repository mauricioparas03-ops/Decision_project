from pyomo.environ import *
from Data.v2_SystemCharacteristics import get_fixed_data
from Data.PriceProcessRestaurant import price_model
from Data.OccupancyProcessRestaurant import next_occupancy_levels 

VFA_WEIGHTS = {
    0: {'T1': -35.4687, 'T2': -36.5209, 'H': 38.8159, 'price_t': 236.06, 'price_previous': -72.7715, 'Occ1': 2.8919, 'Occ2': 0.999, 'vent_counter': 0.0, 'low_override_r1': 0.0, 'low_override_r2': 0.0, 'intercept': 97.1825},
    1: {'T1': -32.6067, 'T2': -32.3931, 'H': 32.5672, 'price_t': 190.2126, 'price_previous': 14.2626, 'Occ1': 3.4663, 'Occ2': 6.7688, 'vent_counter': 27.2, 'low_override_r1': 16.0875, 'low_override_r2': 21.9745, 'intercept': 46.1404},
    2: {'T1': -16.2663, 'T2': -19.0434, 'H': 23.6845, 'price_t': 128.2837, 'price_previous': 64.0347, 'Occ1': -1.0931, 'Occ2': -1.1827, 'vent_counter': 16.2031, 'low_override_r1': 15.573, 'low_override_r2': 16.2804, 'intercept': 42.3815},
    3: {'T1': -18.8182, 'T2': -20.1951, 'H': 16.2384, 'price_t': 102.0203, 'price_previous': 63.263, 'Occ1': -0.9118, 'Occ2': -1.3604, 'vent_counter': 3.3014, 'low_override_r1': 8.6422, 'low_override_r2': 7.446, 'intercept': 40.1767},
    4: {'T1': -22.9297, 'T2': -21.9709, 'H': 10.5439, 'price_t': 77.6107, 'price_previous': 50.9333, 'Occ1': -1.1049, 'Occ2': -0.6918, 'vent_counter': -1.8514, 'low_override_r1': 5.1366, 'low_override_r2': 6.4963, 'intercept': 38.4176},
    5: {'T1': -24.4634, 'T2': -22.7247, 'H': 9.3645, 'price_t': 56.7578, 'price_previous': 33.4102, 'Occ1': -0.8202, 'Occ2': -1.764, 'vent_counter': -2.5723, 'low_override_r1': 6.855, 'low_override_r2': 6.8578, 'intercept': 33.2982},
    6: {'T1': -27.7624, 'T2': -26.946, 'H': 8.299, 'price_t': 39.7313, 'price_previous': 19.0929, 'Occ1': -0.1903, 'Occ2': 3.4308, 'vent_counter': -3.8012, 'low_override_r1': 5.9577, 'low_override_r2': 6.0968, 'intercept': 26.2539},
    7: {'T1': -28.2931, 'T2': -27.5167, 'H': 8.5443, 'price_t': 27.9958, 'price_previous': 11.6969, 'Occ1': 0.9816, 'Occ2': 4.6305, 'vent_counter': -3.2494, 'low_override_r1': 9.4378, 'low_override_r2': 6.6737, 'intercept': 15.4005},
    8: {'T1': -19.2011, 'T2': -20.6518, 'H': 6.9868, 'price_t': 21.4821, 'price_previous': 14.278, 'Occ1': 1.1969, 'Occ2': 5.2746, 'vent_counter': -0.9063, 'low_override_r1': 11.5515, 'low_override_r2': 12.0323, 'intercept': -2.4843},
    9: {'T1': 0.828, 'T2': -0.5646, 'H': 5.7386, 'price_t': 18.6641, 'price_previous': 10.5629, 'Occ1': 2.4606, 'Occ2': 1.1231, 'vent_counter': -0.5659, 'low_override_r1': 12.9957, 'low_override_r2': 10.8021, 'intercept': -9.4625},
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

    m.c_ylow_r1_a = Constraint(expr=m.T1_next <= data['temp_min_comfort_threshold'] + M*(1 - m.y_low_r1_next))
    m.c_ylow_r1_b = Constraint(expr=m.T1_next >= data['temp_min_comfort_threshold'] - M*m.y_low_r1_next)

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
    m.c_ylow_r2_b = Constraint(expr=m.T2_next >= data['temp_min_comfort_threshold'] - M*m.y_low_r2_next)
    
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