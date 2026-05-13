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

        # FIX: Inertia chained to action t=0
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

        # Linearizzazione esatta per vent_counter_2
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
            m.u_r1_2     = Var(domain=Binary) # low_override_r1_next per la VFA
            
            m.c_ylow_r1_2a = Constraint(expr=m.T1_2 <= data['temp_min_comfort_threshold'] + eps + M*(1 - m.y_low_r1_2))
            m.c_ylow_r1_2b = Constraint(expr=m.T1_2 >= data['temp_min_comfort_threshold'] + eps - M*m.y_low_r1_2)
            m.c_yok_r1_2a  = Constraint(expr=m.T1_2 >= data['temp_OK_threshold'] - M*(1 - m.y_ok_r1_2))
            m.c_yok_r1_2b  = Constraint(expr=m.T1_2 <= data['temp_OK_threshold'] + M*m.y_ok_r1_2)

            m.c_u_r1_2a = Constraint(expr=m.u_r1_2 >= m.y_low_r1_2)
            m.c_u_r1_2b = Constraint(expr=m.u_r1_2 <= m.u_r1_1 + m.y_low_r1_2) # Depends on u_r1_1 (Step 1)
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

            m.T1_vfa = Var()
            m.T2_vfa = Var()
            
            # Trust Region
            if w['T1'] < 0:
                m.c_vfa_t1_a = Constraint(expr=m.T1_vfa <= m.T1_2)
                m.c_vfa_t1_b = Constraint(expr=m.T1_vfa <= data['temp_max_comfort_threshold'])
            else:
                m.c_vfa_t1 = Constraint(expr=m.T1_vfa == m.T1_2)

            if w['T2'] < 0:
                m.c_vfa_t2_a = Constraint(expr=m.T2_vfa <= m.T2_2)
                m.c_vfa_t2_b = Constraint(expr=m.T2_vfa <= data['temp_max_comfort_threshold'])
            else:
                m.c_vfa_t2 = Constraint(expr=m.T2_vfa == m.T2_2)

            # Injection of 10 VFA Variables
            vfa_tail = (w['intercept'] + 
                        w['T1'] * m.T1_vfa + 
                        w['T2'] * m.T2_vfa + 
                        w['H'] * m.H_2 +
                        w['vent_counter'] * m.vent_counter_2 + 
                        w['low_override_r1'] * m.u_r1_2 + 
                        w['low_override_r2'] * m.u_r2_2 + 
                        w['price_t'] * price_t2 +
                        w['price_previous'] * price_t1 +  # The past price at t+2 is the price of t+1
                        w['Occ1'] * occ1_2 + 
                        w['Occ2'] * occ2_2)

        expected_future_cost = lookahead_cost + vfa_tail

    m.obj = Objective(expr=immediate_cost + expected_future_cost, sense=minimize)

    SolverFactory('gurobi').solve(m, tee=False)

    return {
        "HeatPowerRoom1": value(m.p1_0),
        "HeatPowerRoom2": value(m.p2_0),
        "VentilationON": int(value(m.v_0))
    }