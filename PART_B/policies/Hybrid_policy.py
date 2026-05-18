from pyomo.environ import *
from Data.v2_SystemCharacteristics import get_fixed_data
from Data.PriceProcessRestaurant import price_model
from Data.OccupancyProcessRestaurant import next_occupancy_levels

VFA_WEIGHTS = {
    0: {'T1': 0.0, 'T2': 0.0, 'H': 0.0, 'price_t': 166.811, 'price_previous': -52.4137, 'Occ1': 2.3454, 'Occ2': -6.5845, 'vent_counter': 0.0, 'low_override_r1': 0.0, 'low_override_r2': 0.0, 'intercept': 81.7666},
    1: {'T1': -3.6597, 'T2': -6.6471, 'H': 3.2715, 'price_t': 136.232, 'price_previous': 43.6886, 'Occ1': 0.4884, 'Occ2': -3.691, 'vent_counter': -3.1349, 'low_override_r1': 0.0, 'low_override_r2': 0.0, 'intercept': 45.5579},
    2: {'T1': -17.8462, 'T2': -16.712, 'H': 17.0878, 'price_t': 94.55, 'price_previous': 60.9206, 'Occ1': 1.7097, 'Occ2': -1.4284, 'vent_counter': -14.5423, 'low_override_r1': 10.304, 'low_override_r2': 16.7219, 'intercept': 34.5378},
    3: {'T1': -12.4565, 'T2': -10.464, 'H': 20.5213, 'price_t': 76.5996, 'price_previous': 53.3461, 'Occ1': 1.7842, 'Occ2': 0.6308, 'vent_counter': -14.0177, 'low_override_r1': 18.7843, 'low_override_r2': 24.1062, 'intercept': 34.9512},
    4: {'T1': -7.7689, 'T2': -9.1995, 'H': 25.1733, 'price_t': 71.1307, 'price_previous': 47.4648, 'Occ1': -1.3999, 'Occ2': -2.4511, 'vent_counter': -5.531, 'low_override_r1': 17.9065, 'low_override_r2': 22.11, 'intercept': 27.4075},
    5: {'T1': -10.7988, 'T2': -9.031, 'H': 25.8565, 'price_t': 58.8176, 'price_previous': 43.0421, 'Occ1': -4.0887, 'Occ2': -2.6151, 'vent_counter': 2.2132, 'low_override_r1': 17.9992, 'low_override_r2': 19.7543, 'intercept': 11.6112},
    6: {'T1': -12.1684, 'T2': -10.1931, 'H': 17.439, 'price_t': 45.3847, 'price_previous': 33.2699, 'Occ1': -2.407, 'Occ2': 0.0989, 'vent_counter': 4.8596, 'low_override_r1': 14.7093, 'low_override_r2': 17.8049, 'intercept': 1.5964},
    7: {'T1': -12.2613, 'T2': -11.4534, 'H': 12.7152, 'price_t': 33.5577, 'price_previous': 22.9294, 'Occ1': 3.7094, 'Occ2': 4.1436, 'vent_counter': 0.3337, 'low_override_r1': 17.4723, 'low_override_r2': 18.7128, 'intercept': -8.3538},
    8: {'T1': -8.5566, 'T2': -9.4265, 'H': 14.9203, 'price_t': 30.1314, 'price_previous': 20.0247, 'Occ1': 0.7527, 'Occ2': 3.4089, 'vent_counter': 4.6502, 'low_override_r1': 14.0543, 'low_override_r2': 15.5494, 'intercept': -16.3612},
    9: {'T1': -1.7524, 'T2': -2.5389, 'H': 5.123, 'price_t': 12.532, 'price_previous': 9.7659, 'Occ1': 1.0408, 'Occ2': -0.2031, 'vent_counter': 3.8551, 'low_override_r1': 13.4865, 'low_override_r2': 14.3706, 'intercept': -8.692},
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