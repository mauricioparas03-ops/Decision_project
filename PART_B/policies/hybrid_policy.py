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