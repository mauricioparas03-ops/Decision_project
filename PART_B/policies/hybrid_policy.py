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
    
    t = int(state['current_time'])
    M = 500  # Big-M for rigorous linearization
    eps = 0.001 

    # ==========================================================
    # --- STEP 0: HERE AND NOW (Immediate action at time t) ---
    # ==========================================================
    m.p1_0 = Var(bounds=(0, data['heating_max_power']))
    m.p2_0 = Var(bounds=(0, data['heating_max_power']))
    m.v_0 = Var(domain=Binary)

    # Hardware forcing (At time t)
    if state['T1'] > data['temp_max_comfort_threshold']:
        m.p1_0.fix(0)
    elif state['T1'] < data['temp_min_comfort_threshold'] or state['low_override_r1'] == 1:
        m.p1_0.fix(data['heating_max_power'])

    if state['T2'] > data['temp_max_comfort_threshold']:
        m.p2_0.fix(0)
    elif state['T2'] < data['temp_min_comfort_threshold'] or state['low_override_r2'] == 1:
        m.p2_0.fix(data['heating_max_power'])

    if state['H'] > data['humidity_threshold'] or state['vent_counter'] in [1, 2]:
        m.v_0.fix(1)

    # Physical dynamics towards t+1
    tout_0 = data['outdoor_temperature'][t]
    m.T1_1 = Var()
    m.T2_1 = Var()
    m.H_1 = Var()
    m.vent_counter_1 = Var(domain=NonNegativeReals)

    m.c_t1_1 = Constraint(expr=m.T1_1 == state['T1'] + data['heat_exchange_coeff'] * (state['T2'] - state['T1']) 
                        + data['thermal_loss_coeff'] * (tout_0 - state['T1']) + data['heating_efficiency_coeff'] * m.p1_0 
                        - data['heat_vent_coeff'] * m.v_0 + data['heat_occupancy_coeff'] * state['Occ1'])

    m.c_t2_1 = Constraint(expr=m.T2_1 == state['T2'] + data['heat_exchange_coeff'] * (state['T1'] - state['T2']) 
                        + data['thermal_loss_coeff'] * (tout_0 - state['T2']) + data['heating_efficiency_coeff'] * m.p2_0 
                        - data['heat_vent_coeff'] * m.v_0 + data['heat_occupancy_coeff'] * state['Occ2'])

    m.c_h_1 = Constraint(expr=m.H_1 == state['H'] - data['humidity_vent_coeff'] * m.v_0 
                       + data['humidity_occupancy_coeff'] * (state['Occ1'] + state['Occ2']))

    m.c_vc_1 = Constraint(expr=m.vent_counter_1 == (state['vent_counter'] + 1) * m.v_0)

    immediate_cost = state['price_t'] * (m.p1_0 + m.p2_0 + m.v_0 * data['ventilation_power'])
    expected_future_cost = 0.0

    # ==========================================================
    # --- STEP 1: DETERMINISTIC LOOKAHEAD (MPC for time t+1) ---
    # ==========================================================
    if t < 9:
        price_t1 = price_model(state['price_t'], state['price_previous'])
        occ1_1, occ2_1 = next_occupancy_levels(state['Occ1'], state['Occ2'])
        tout_1 = data['outdoor_temperature'][t+1]
        
        m.p1_1 = Var(bounds=(0, data['heating_max_power']))
        m.p2_1 = Var(bounds=(0, data['heating_max_power']))
        m.v_1 = Var(domain=Binary)

        # -----------------------------------------------------
        # Formal Override Logic for t+1 (Room 1)
        # -----------------------------------------------------
        m.y_low_r1_1  = Var(domain=Binary)
        m.y_ok_r1_1   = Var(domain=Binary)
        m.y_high_r1_1 = Var(domain=Binary)
        m.u_r1_1      = Var(domain=Binary) 

        # Tmin detection (with eps tolerance)
        m.c_ylow_r1_1a = Constraint(expr=m.T1_1 <= data['temp_min_comfort_threshold'] + eps + M*(1 - m.y_low_r1_1))
        m.c_ylow_r1_1b = Constraint(expr=m.T1_1 >= data['temp_min_comfort_threshold'] + eps - M*m.y_low_r1_1)
        # Tcomfort detection
        m.c_yok_r1_1a  = Constraint(expr=m.T1_1 >= data['temp_OK_threshold'] - M*(1 - m.y_ok_r1_1))
        m.c_yok_r1_1b  = Constraint(expr=m.T1_1 <= data['temp_OK_threshold'] + M*m.y_ok_r1_1)
        # Thigh detection
        m.c_yhigh_r1_1a = Constraint(expr=m.T1_1 >= data['temp_max_comfort_threshold'] + eps - M*(1 - m.y_high_r1_1))
        
        # Hardware memory propagation (from t to t+1)
        u_prev_r1 = state['low_override_r1']
        m.c_u_r1_1a = Constraint(expr=m.u_r1_1 >= m.y_low_r1_1)
        m.c_u_r1_1b = Constraint(expr=m.u_r1_1 <= u_prev_r1 + m.y_low_r1_1)
        m.c_u_r1_1c = Constraint(expr=m.u_r1_1 >= u_prev_r1 - m.y_ok_r1_1)
        m.c_u_r1_1d = Constraint(expr=m.u_r1_1 <= 1 - m.y_ok_r1_1)

        # Azione forzata a t+1
        m.c_p1_1_max = Constraint(expr=m.p1_1 <= data['heating_max_power'] * (1 - m.y_high_r1_1))
        m.c_p1_1_min = Constraint(expr=m.p1_1 >= data['heating_max_power'] * m.u_r1_1)

        # -----------------------------------------------------
        # Formal Override Logic for t+1 (Room 2)
        # -----------------------------------------------------
        m.y_low_r2_1  = Var(domain=Binary)
        m.y_ok_r2_1   = Var(domain=Binary)
        m.y_high_r2_1 = Var(domain=Binary)
        m.u_r2_1      = Var(domain=Binary) 

        m.c_ylow_r2_1a = Constraint(expr=m.T2_1 <= data['temp_min_comfort_threshold'] + eps + M*(1 - m.y_low_r2_1))
        m.c_ylow_r2_1b = Constraint(expr=m.T2_1 >= data['temp_min_comfort_threshold'] + eps - M*m.y_low_r2_1)
        m.c_yok_r2_1a  = Constraint(expr=m.T2_1 >= data['temp_OK_threshold'] - M*(1 - m.y_ok_r2_1))
        m.c_yok_r2_1b  = Constraint(expr=m.T2_1 <= data['temp_OK_threshold'] + M*m.y_ok_r2_1)
        m.c_yhigh_r2_1a = Constraint(expr=m.T2_1 >= data['temp_max_comfort_threshold'] + eps - M*(1 - m.y_high_r2_1))
        
        u_prev_r2 = state['low_override_r2']
        m.c_u_r2_1a = Constraint(expr=m.u_r2_1 >= m.y_low_r2_1)
        m.c_u_r2_1b = Constraint(expr=m.u_r2_1 <= u_prev_r2 + m.y_low_r2_1)
        m.c_u_r2_1c = Constraint(expr=m.u_r2_1 >= u_prev_r2 - m.y_ok_r2_1)
        m.c_u_r2_1d = Constraint(expr=m.u_r2_1 <= 1 - m.y_ok_r2_1)

        m.c_p2_1_max = Constraint(expr=m.p2_1 <= data['heating_max_power'] * (1 - m.y_high_r2_1))
        m.c_p2_1_min = Constraint(expr=m.p2_1 >= data['heating_max_power'] * m.u_r2_1)

        # -----------------------------------------------------
        # Ventilation Overrides at t+1
        # -----------------------------------------------------
        m.high_H_1 = Var(domain=Binary)
        m.c_hh_1a = Constraint(expr=m.H_1 >= data['humidity_threshold'] + eps - M*(1 - m.high_H_1))
        m.c_v1_hum = Constraint(expr=m.v_1 >= m.high_H_1)

        if state['vent_counter'] == 1:
            m.c_v1_ine = Constraint(expr=m.v_1 >= 1)
        elif state['vent_counter'] == 0:
            m.c_v1_ine = Constraint(expr=m.v_1 >= m.v_0)

        # Physical dynamics towards t+2
        m.T1_2 = Var()
        m.T2_2 = Var()
        m.H_2 = Var()
        m.vent_counter_2 = Var(domain=NonNegativeReals)

        m.c_t1_2 = Constraint(expr=m.T1_2 == m.T1_1 + data['heat_exchange_coeff'] * (m.T2_1 - m.T1_1) 
                            + data['thermal_loss_coeff'] * (tout_1 - m.T1_1) + data['heating_efficiency_coeff'] * m.p1_1 
                            - data['heat_vent_coeff'] * m.v_1 + data['heat_occupancy_coeff'] * occ1_1)

        m.c_t2_2 = Constraint(expr=m.T2_2 == m.T2_1 + data['heat_exchange_coeff'] * (m.T1_1 - m.T2_1) 
                            + data['thermal_loss_coeff'] * (tout_1 - m.T2_1) + data['heating_efficiency_coeff'] * m.p2_1 
                            - data['heat_vent_coeff'] * m.v_1 + data['heat_occupancy_coeff'] * occ2_1)

        m.c_h_2 = Constraint(expr=m.H_2 == m.H_1 - data['humidity_vent_coeff'] * m.v_1 
                           + data['humidity_occupancy_coeff'] * (occ1_1 + occ2_1))

        m.c_vc2_a = Constraint(expr=m.vent_counter_2 <= M * m.v_1)
        m.c_vc2_b = Constraint(expr=m.vent_counter_2 <= m.vent_counter_1 + 1 + M * (1 - m.v_1))
        m.c_vc2_c = Constraint(expr=m.vent_counter_2 >= m.vent_counter_1 + 1 - M * (1 - m.v_1))
        m.c_vc2_d = Constraint(expr=m.vent_counter_2 >= 0)

        lookahead_cost = price_t1 * (m.p1_1 + m.p2_1 + m.v_1 * data['ventilation_power'])

        # ==========================================================
        # --- STEP 2: ADP TAIL (VFA evaluated at start of time t+2) ---
        # ==========================================================
        vfa_tail = 0.0
        if t < 8:
            w = VFA_WEIGHTS[t+2]
            
            # -----------------------------------------------------
            # Override Memory for t+2 (Necessary for VFA!)
            # -----------------------------------------------------
            m.y_low_r1_2 = Var(domain=Binary)
            m.y_ok_r1_2  = Var(domain=Binary)
            m.u_r1_2     = Var(domain=Binary) 
            
            m.c_ylow_r1_2a = Constraint(expr=m.T1_2 <= data['temp_min_comfort_threshold'] + eps + M*(1 - m.y_low_r1_2))
            m.c_ylow_r1_2b = Constraint(expr=m.T1_2 >= data['temp_min_comfort_threshold'] + eps - M*m.y_low_r1_2)
            m.c_yok_r1_2a  = Constraint(expr=m.T1_2 >= data['temp_OK_threshold'] - M*(1 - m.y_ok_r1_2))
            m.c_yok_r1_2b  = Constraint(expr=m.T1_2 <= data['temp_OK_threshold'] + M*m.y_ok_r1_2)

            m.c_u_r1_2a = Constraint(expr=m.u_r1_2 >= m.y_low_r1_2)
            m.c_u_r1_2b = Constraint(expr=m.u_r1_2 <= m.u_r1_1 + m.y_low_r1_2) 
            m.c_u_r1_2c = Constraint(expr=m.u_r1_2 >= m.u_r1_1 - m.y_ok_r1_2)
            m.c_u_r1_2d = Constraint(expr=m.u_r1_2 <= 1 - m.y_ok_r1_2)

            m.y_low_r2_2 = Var(domain=Binary)
            m.y_ok_r2_2  = Var(domain=Binary)
            m.u_r2_2     = Var(domain=Binary) 
            
            m.c_ylow_r2_2a = Constraint(expr=m.T2_2 <= data['temp_min_comfort_threshold'] + eps + M*(1 - m.y_low_r2_2))
            m.c_ylow_r2_2b = Constraint(expr=m.T2_2 >= data['temp_min_comfort_threshold'] + eps - M*m.y_low_r2_2)
            m.c_yok_r2_2a  = Constraint(expr=m.T2_2 >= data['temp_OK_threshold'] - M*(1 - m.y_ok_r2_2))
            m.c_yok_r2_2b  = Constraint(expr=m.T2_2 <= data['temp_OK_threshold'] + M*m.y_ok_r2_2)

            m.c_u_r2_2a = Constraint(expr=m.u_r2_2 >= m.y_low_r2_2)
            m.c_u_r2_2b = Constraint(expr=m.u_r2_2 <= m.u_r2_1 + m.y_low_r2_2)
            m.c_u_r2_2c = Constraint(expr=m.u_r2_2 >= m.u_r2_1 - m.y_ok_r2_2)
            m.c_u_r2_2d = Constraint(expr=m.u_r2_2 <= 1 - m.y_ok_r2_2)

            # -----------------------------------------------------
            # Exogenous Forecast for t+2 and VFA Calculation
            # -----------------------------------------------------
            price_t2 = price_model(price_t1, state['price_t'])
            occ1_2, occ2_2 = next_occupancy_levels(occ1_1, occ2_1)

            # MODIFICA: Rimossa la Trust Region
            # MODIFICA: Normalizzazione applicata direttamente all'equazione della VFA
            vfa_tail = (
                w['intercept'] + 
                w['T1'] * ((m.T1_2 - 22.0) / 8.0) + 
                w['T2'] * ((m.T2_2 - 22.0) / 8.0) + 
                w['H'] * ((m.H_2 - 40.0) / 40.0) +
                w['vent_counter'] * (m.vent_counter_2 / 3.0) + 
                w['low_override_r1'] * m.u_r1_2 + 
                w['low_override_r2'] * m.u_r2_2 + 
                w['price_t'] * (price_t2 / 10.0) +
                w['price_previous'] * (price_t1 / 10.0) +  
                w['Occ1'] * ((occ1_2 - 20.0) / 30.0) + 
                w['Occ2'] * ((occ2_2 - 10.0) / 20.0)
            )

        expected_future_cost = lookahead_cost + vfa_tail

    m.obj = Objective(expr=immediate_cost + expected_future_cost, sense=minimize)

    SolverFactory('gurobi').solve(m, tee=False)

    return {
        "HeatPowerRoom1": value(m.p1_0),
        "HeatPowerRoom2": value(m.p2_0),
        "VentilationON": int(value(m.v_0))
    }