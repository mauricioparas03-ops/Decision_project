from pyomo.environ import *
from Data.v2_SystemCharacteristics import get_fixed_data
from Data.PriceProcessRestaurant import price_model
from Data.OccupancyProcessRestaurant import next_occupancy_levels 

VFA_WEIGHTS = {
    0: {'T1': -8.3368, 'T2': -8.4183, 'H': 29.7709, 'price_t': 184.571, 'price_previous': -57.4942, 'Occ1': -3.552, 'Occ2': 2.8204, 'vent_counter': 0.0, 'low_override_r1': 0.0, 'low_override_r2': 0.0, 'intercept': 61.8075},
    1: {'T1': -9.0447, 'T2': -9.7995, 'H': 32.2628, 'price_t': 165.7444, 'price_previous': 14.3447, 'Occ1': -0.3919, 'Occ2': -0.5644, 'vent_counter': 19.7049, 'low_override_r1': 35.0862, 'low_override_r2': 36.1438, 'intercept': 18.7716},
    2: {'T1': -4.9311, 'T2': -5.4965, 'H': 28.4026, 'price_t': 121.5497, 'price_previous': 57.3255, 'Occ1': -4.0862, 'Occ2': -2.1746, 'vent_counter': 30.4878, 'low_override_r1': 28.0968, 'low_override_r2': 24.6015, 'intercept': 11.5262},
    3: {'T1': -13.4041, 'T2': -13.0665, 'H': 24.7283, 'price_t': 93.6153, 'price_previous': 61.0787, 'Occ1': 1.1298, 'Occ2': 1.5432, 'vent_counter': 16.2653, 'low_override_r1': 15.5998, 'low_override_r2': 12.6422, 'intercept': 4.0037},
    4: {'T1': -20.4344, 'T2': -21.65, 'H': 15.8882, 'price_t': 70.4439, 'price_previous': 46.0781, 'Occ1': 1.1665, 'Occ2': 1.5098, 'vent_counter': 2.8218, 'low_override_r1': 12.3018, 'low_override_r2': 6.2533, 'intercept': 9.0091},
    5: {'T1': -16.5171, 'T2': -17.8856, 'H': 14.1182, 'price_t': 50.727, 'price_previous': 30.1955, 'Occ1': 0.5515, 'Occ2': 3.104, 'vent_counter': -0.8036, 'low_override_r1': 12.8945, 'low_override_r2': 8.8533, 'intercept': 4.2197},
    6: {'T1': -14.2408, 'T2': -15.6891, 'H': 11.5555, 'price_t': 37.6434, 'price_previous': 18.5224, 'Occ1': 2.8728, 'Occ2': -4.854, 'vent_counter': -2.4179, 'low_override_r1': 16.3108, 'low_override_r2': 10.3472, 'intercept': 8.6357},
    7: {'T1': -13.694, 'T2': -12.6146, 'H': 6.6726, 'price_t': 40.0104, 'price_previous': 16.3184, 'Occ1': 3.6869, 'Occ2': -0.5152, 'vent_counter': -1.6426, 'low_override_r1': 21.0921, 'low_override_r2': 17.5114, 'intercept': -3.8933},
    8: {'T1': -6.7134, 'T2': -7.4186, 'H': 6.4815, 'price_t': 42.0865, 'price_previous': 22.7026, 'Occ1': -0.0187, 'Occ2': -1.4704, 'vent_counter': -1.3588, 'low_override_r1': 17.507, 'low_override_r2': 18.3879, 'intercept': -13.812},
    9: {'T1': 0.603, 'T2': -0.3511, 'H': 5.1107, 'price_t': 27.7555, 'price_previous': 19.7616, 'Occ1': 0.5282, 'Occ2': -0.2876, 'vent_counter': -0.6616, 'low_override_r1': 10.8383, 'low_override_r2': 10.3956, 'intercept': -14.3331},
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
    elif state['T1'] < data['temp_min_comfort_threshold'] and state['low_override_r1'] == 1:
        m.p1.fix(data['heating_max_power'])

    if state['T2'] > data['temp_max_comfort_threshold']:
        m.p2.fix(0)
    elif state['T2'] < data['temp_min_comfort_threshold'] and state['low_override_r2'] == 1:
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