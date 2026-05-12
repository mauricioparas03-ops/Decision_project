from pyomo.environ import *
from Data.v2_SystemCharacteristics import get_fixed_data
from Data.PriceProcessRestaurant import price_model

# Definitive weights for the VFA, obtained after training on 100 days (Task 4)
VFA_WEIGHTS = {
    0: {'T1': 0.0, 'T2': 0.0, 'H': 0.0, 'price_t': 42.2762, 'vent_counter': 0.0, 'low_override_r1': 0.0, 'low_override_r2': 0.0, 'intercept': -78.7346},
    1: {'T1': 2.2541, 'T2': -3.3547, 'H': 7.0211, 'price_t': 29.0863, 'vent_counter': 114.0406, 'low_override_r1': 0.0, 'low_override_r2': 0.0, 'intercept': -332.6881},
    2: {'T1': -2.2899, 'T2': -1.1332, 'H': 3.408, 'price_t': 24.4825, 'vent_counter': 46.9638, 'low_override_r1': 18.5807, 'low_override_r2': 27.6635, 'intercept': -125.4662},
    3: {'T1': -2.6706, 'T2': -0.3753, 'H': 2.0655, 'price_t': 21.4215, 'vent_counter': 22.3914, 'low_override_r1': 18.9324, 'low_override_r2': 17.3069, 'intercept': -67.0107},
    4: {'T1': -2.151, 'T2': -4.7714, 'H': 0.7994, 'price_t': 17.5064, 'vent_counter': 1.878, 'low_override_r1': 9.9414, 'low_override_r2': 13.6788, 'intercept': 94.3381},
    5: {'T1': -5.9482, 'T2': -0.3145, 'H': 0.7534, 'price_t': 13.0689, 'vent_counter': 2.4829, 'low_override_r1': 10.459, 'low_override_r2': 10.0252, 'intercept': 85.3074},
    6: {'T1': -4.8076, 'T2': -2.0748, 'H': 0.5798, 'price_t': 7.7821, 'vent_counter': -0.6056, 'low_override_r1': 3.8225, 'low_override_r2': 17.4337, 'intercept': 117.1084},
    7: {'T1': -2.4897, 'T2': -1.1346, 'H': 0.4077, 'price_t': 4.7985, 'vent_counter': -1.7045, 'low_override_r1': 17.0495, 'low_override_r2': 28.5045, 'intercept': 52.5686},
    8: {'T1': -0.6193, 'T2': -0.8051, 'H': 0.4395, 'price_t': 4.5277, 'vent_counter': 1.7981, 'low_override_r1': 28.2602, 'low_override_r2': 22.8713, 'intercept': -5.571},
    9: {'T1': -0.078, 'T2': 0.3872, 'H': 0.2915, 'price_t': 2.2303, 'vent_counter': 2.197, 'low_override_r1': 13.4133, 'low_override_r2': 14.1389, 'intercept': -30.6712},
}

def select_action(state):
    data = get_fixed_data()
    m = ConcreteModel()
    
    t = int(state['current_time'])
    M = 100
    eps = 0.001 

    # --- STEP 0: HERE AND NOW (Current hour t) ---
    m.p1_0 = Var(bounds=(0, data['heating_max_power']))
    m.p2_0 = Var(bounds=(0, data['heating_max_power']))
    m.v_0 = Var(domain=Binary)

    # Current Overrule rules
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

    # Dynamics for next hour t+1
    tout_0 = data['outdoor_temperature'][t]
    m.T1_1 = Var()
    m.T2_1 = Var()
    m.H_1 = Var()
    m.vent_counter_1 = Var(domain=NonNegativeReals)
    m.low_override_r1_1 = Var(domain=Binary)
    m.low_override_r2_1 = Var(domain=Binary)

    m.c_t1_1 = Constraint(expr=m.T1_1 == state['T1'] + data['heat_exchange_coeff'] * (state['T2'] - state['T1']) 
                        + data['thermal_loss_coeff'] * (tout_0 - state['T1']) + data['heating_efficiency_coeff'] * m.p1_0 
                        - data['heat_vent_coeff'] * m.v_0 + data['heat_occupancy_coeff'] * state['Occ1'])

    m.c_t2_1 = Constraint(expr=m.T2_1 == state['T2'] + data['heat_exchange_coeff'] * (state['T1'] - state['T2']) 
                        + data['thermal_loss_coeff'] * (tout_0 - state['T2']) + data['heating_efficiency_coeff'] * m.p2_0 
                        - data['heat_vent_coeff'] * m.v_0 + data['heat_occupancy_coeff'] * state['Occ2'])

    m.c_h_1 = Constraint(expr=m.H_1 == state['H'] - data['humidity_vent_coeff'] * m.v_0 
                       + data['humidity_occupancy_coeff'] * (state['Occ1'] + state['Occ2']))

    m.c_vc_1 = Constraint(expr=m.vent_counter_1 == (state['vent_counter'] + 1) * m.v_0)

    # Mapping future overrules for t+1
    thresh1_0 = (data['temp_min_comfort_threshold'] if state['low_override_r1'] == 0 else data['temp_OK_threshold']) + eps
    m.c_low1_1a = Constraint(expr=m.T1_1 >= thresh1_0 - M * m.low_override_r1_1)
    m.c_low1_1b = Constraint(expr=m.T1_1 <= thresh1_0 + M * (1 - m.low_override_r1_1))

    thresh2_0 = (data['temp_min_comfort_threshold'] if state['low_override_r2'] == 0 else data['temp_OK_threshold']) + eps
    m.c_low2_1a = Constraint(expr=m.T2_1 >= thresh2_0 - M * m.low_override_r2_1)
    m.c_low2_1b = Constraint(expr=m.T2_1 <= thresh2_0 + M * (1 - m.low_override_r2_1))

    # --- STEP 1: DETERMINISTIC LOOKAHEAD (MPC for hour t+1) ---
    immediate_cost = state['price_t'] * (m.p1_0 + m.p2_0 + m.v_0 * data['ventilation_power'])
    expected_future_cost = 0.0

    if t < 9:
        price_t1 = price_model(state['price_t'], state['price_previous'])
        tout_1 = data['outdoor_temperature'][t+1]
        
        m.p1_1 = Var(bounds=(0, data['heating_max_power']))
        m.p2_1 = Var(bounds=(0, data['heating_max_power']))
        m.v_1 = Var(domain=Binary)

        # Future Overrule logic for t+1 (Big-M)
        m.high_T1_1 = Var(domain=Binary)
        m.c_ht1_1 = Constraint(expr=m.T1_1 >= data['temp_max_comfort_threshold'] + eps - M*(1-m.high_T1_1))
        m.c_p1_1_up = Constraint(expr=m.p1_1 <= data['heating_max_power'] * (1 - m.high_T1_1))
        m.c_p1_1_lo = Constraint(expr=m.p1_1 >= data['heating_max_power'] * m.low_override_r1_1)

        m.high_T2_1 = Var(domain=Binary)
        m.c_ht2_1 = Constraint(expr=m.T2_1 >= data['temp_max_comfort_threshold'] + eps - M*(1-m.high_T2_1))
        m.c_p2_1_up = Constraint(expr=m.p2_1 <= data['heating_max_power'] * (1 - m.high_T2_1))
        m.c_p2_1_lo = Constraint(expr=m.p2_1 >= data['heating_max_power'] * m.low_override_r2_1)

        # Humidity and Inertia for t+1
        m.high_H_1 = Var(domain=Binary)
        m.c_hh_1 = Constraint(expr=m.H_1 >= data['humidity_threshold'] + eps - M*(1-m.high_H_1))
        m.c_v1_hum = Constraint(expr=m.v_1 >= m.high_H_1)
        m.c_v1_ine = Constraint(expr=m.v_1 >= (1 if state['vent_counter'] in [1,2] else m.v_0))

        # Dynamics for next hour t+2
        m.T1_2 = Var()
        m.T2_2 = Var()
        m.H_2 = Var()
        m.vent_counter_2 = Var(domain=NonNegativeReals)
        m.low_override_r1_2 = Var(domain=Binary)
        m.low_override_r2_2 = Var(domain=Binary)

        m.c_t1_2 = Constraint(expr=m.T1_2 == m.T1_1 + data['heat_exchange_coeff'] * (m.T2_1 - m.T1_1) 
                            + data['thermal_loss_coeff'] * (tout_1 - m.T1_1) + data['heating_efficiency_coeff'] * m.p1_1 
                            - data['heat_vent_coeff'] * m.v_1 + data['heat_occupancy_coeff'] * state['Occ1'])

        m.c_t2_2 = Constraint(expr=m.T2_2 == m.T2_1 + data['heat_exchange_coeff'] * (m.T1_1 - m.T2_1) 
                            + data['thermal_loss_coeff'] * (tout_1 - m.T2_1) + data['heating_efficiency_coeff'] * m.p2_1 
                            - data['heat_vent_coeff'] * m.v_1 + data['heat_occupancy_coeff'] * state['Occ2'])

        m.c_h_2 = Constraint(expr=m.H_2 == m.H_1 - data['humidity_vent_coeff'] * m.v_1 
                           + data['humidity_occupancy_coeff'] * (state['Occ1'] + state['Occ2']))

        m.c_vc_2 = Constraint(expr=m.vent_counter_2 == (m.vent_counter_1 + 1) * m.v_1)

        # Future Overrule mapping for t+2
        m.thresh1_1 = Var()
        m.c_th1_1 = Constraint(expr=m.thresh1_1 == data['temp_min_comfort_threshold'] + eps + (data['temp_OK_threshold'] - data['temp_min_comfort_threshold']) * m.low_override_r1_1)
        m.c_low1_2a = Constraint(expr=m.T1_2 >= m.thresh1_1 - M * m.low_override_r1_2)
        m.c_low1_2b = Constraint(expr=m.T1_2 <= m.thresh1_1 + M * (1 - m.low_override_r1_2))

        m.thresh2_1 = Var()
        m.c_th2_1 = Constraint(expr=m.thresh2_1 == data['temp_min_comfort_threshold'] + eps + (data['temp_OK_threshold'] - data['temp_min_comfort_threshold']) * m.low_override_r2_1)
        m.c_low2_2a = Constraint(expr=m.T2_2 >= m.thresh2_1 - M * m.low_override_r2_2)
        m.c_low2_2b = Constraint(expr=m.T2_2 <= m.thresh2_1 + M * (1 - m.low_override_r2_2))

        lookahead_cost = price_t1 * (m.p1_1 + m.p2_1 + m.v_1 * data['ventilation_power'])

        # --- STEP 2: ADP TAIL (VFA at hour t+2) ---
        vfa_tail = 0.0
        if t < 8:
            w = VFA_WEIGHTS[t+2]
            m.T1_vfa = Var()
            m.T2_vfa = Var()

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

            price_t2 = price_model(price_t1, state['price_t'])
            vfa_tail = (w['intercept'] + w['T1'] * m.T1_vfa + w['T2'] * m.T2_vfa + w['H'] * m.H_2 +
                        w['vent_counter'] * m.vent_counter_2 + w['low_override_r1'] * m.low_override_r1_2 + 
                        w['low_override_r2'] * m.low_override_r2_2 + w['price_t'] * price_t2)

        expected_future_cost = lookahead_cost + vfa_tail

    m.obj = Objective(expr=immediate_cost + expected_future_cost, sense=minimize)

    SolverFactory('gurobi').solve(m, tee=False)

    return {
        "HeatPowerRoom1": value(m.p1_0),
        "HeatPowerRoom2": value(m.p2_0),
        "VentilationON": int(value(m.v_0))
    }