from pyomo.environ import *
from Data.v2_SystemCharacteristics import get_fixed_data

# definitive wheights for the VFA, obtained after training on 50 days with a linear regression on the collected data
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

    # DECISION VARIABLES
    m.p1 = Var(bounds=(0, data['heating_max_power']))
    m.p2 = Var(bounds=(0, data['heating_max_power']))
    m.v = Var(domain=Binary)

    m.T1_next = Var()
    m.T2_next = Var()
    m.H_next = Var()

    m.vent_counter_next = Var(domain=NonNegativeReals)
    m.low_override_r1_next = Var(domain=Binary)
    m.low_override_r2_next = Var(domain=Binary)

    # VFA Trust Region variables
    m.T1_vfa = Var()
    m.T2_vfa = Var()

    # OVERRULE FOR CURRENT STEP
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

    # MAPPING FOR FUTURE OVERRULES
    M = 100
    eps = 0.001 
    
    thresh1 = (data['temp_min_comfort_threshold'] if state['low_override_r1'] == 0 else data['temp_OK_threshold']) + eps
    m.c_low1_a = Constraint(expr=m.T1_next >= thresh1 - M * m.low_override_r1_next)
    m.c_low1_b = Constraint(expr=m.T1_next <= thresh1 + M * (1 - m.low_override_r1_next))

    thresh2 = (data['temp_min_comfort_threshold'] if state['low_override_r2'] == 0 else data['temp_OK_threshold']) + eps
    m.c_low2_a = Constraint(expr=m.T2_next >= thresh2 - M * m.low_override_r2_next)
    m.c_low2_b = Constraint(expr=m.T2_next <= thresh2 + M * (1 - m.low_override_r2_next))

    # APPROXIMATE VALUE FUNCTION 
    immediate_cost = state['price_t'] * (m.p1 + m.p2 + m.v * data['ventilation_power'])
    
    t = int(state['current_time'])

    # If not at the last hour use weights to forecast future
    if t < 9:
        w = VFA_WEIGHTS[t+1]
        
        # Dynamic Trust Region for T1
        if w['T1'] < 0:
            m.c_vfa_t1_a = Constraint(expr=m.T1_vfa <= m.T1_next)
            m.c_vfa_t1_b = Constraint(expr=m.T1_vfa <= data['temp_max_comfort_threshold'])
        else:
            m.c_vfa_t1 = Constraint(expr=m.T1_vfa == m.T1_next)

        # Dynamic Trust Region for T2
        if w['T2'] < 0:
            m.c_vfa_t2_a = Constraint(expr=m.T2_vfa <= m.T2_next)
            m.c_vfa_t2_b = Constraint(expr=m.T2_vfa <= data['temp_max_comfort_threshold'])
        else:
            m.c_vfa_t2 = Constraint(expr=m.T2_vfa == m.T2_next)
        
        expected_future_cost = (
            w['intercept'] +   
            w['T1'] * m.T1_vfa +  
            w['T2'] * m.T2_vfa + 
            w['H'] * m.H_next +
            w['vent_counter'] * m.vent_counter_next + 
            w['low_override_r1'] * m.low_override_r1_next + 
            w['low_override_r2'] * m.low_override_r2_next
        )
    else:
        # end of the day: future cost is zero (Natural End-of-Horizon)
        expected_future_cost = 0.0

    m.obj = Objective(expr=immediate_cost + expected_future_cost, sense=minimize)

    # SOLVER
    solver = SolverFactory('gurobi')
    solver.solve(m, tee=False)

    # =====================================================================
    # INIZIO BLOCCO DI DEBUG VFA (Aggiornato con Trust Region)
    # =====================================================================
    try:
        # 1. Estrazione stati fisici (realtà termodinamica)
        val_T1_next = value(m.T1_next)
        val_T2_next = value(m.T2_next)
        val_H_next  = value(m.H_next)
        val_vent    = value(m.vent_counter_next)
        val_ov1     = value(m.low_override_r1_next)
        val_ov2     = value(m.low_override_r2_next)
        
        val_costo_immediato = value(immediate_cost)
        
        if t < 9:
            val_costo_futuro = value(expected_future_cost)
            # 2. Estrazione stati VFA (visione matematica limitata dalla Trust Region)
            val_T1_vfa = value(m.T1_vfa)
            val_T2_vfa = value(m.T2_vfa)
        else:
            val_costo_futuro = 0.0
            
        val_costo_totale = val_costo_immediato + val_costo_futuro

        print(f"\n{'='*65}")
        print(f"🔎 DEBUG TIMESTEP t = {t}")
        print(f"{'='*65}")
        print(f"Costo Immediato (Oggi) : {val_costo_immediato:>8.2f}")
        print(f"Costo Futuro (VFA)     : {val_costo_futuro:>8.2f}")
        print(f"Obiettivo Totale Solver: {val_costo_totale:>8.2f}")
        print("-" * 65)
        
        if t < 9:
            w = VFA_WEIGHTS[t+1]
            
            # Pesi corretti (Manteniamo la tua logica di visualizzazione)
            w_v_count = max(0, w['vent_counter'])
            w_ov1 = max(0, w['low_override_r1'])
            w_ov2 = max(0, w['low_override_r2'])

            print("Dettaglio fisica vs Trust Region VFA scelti dal solver:")
            print(f"  T1: fisico {val_T1_next:>5.2f} -> VFA vista: {val_T1_vfa:>5.2f} | Peso: {w['T1']:>7.4f} ")
            print(f"  T2: fisico {val_T2_next:>5.2f} -> VFA vista: {val_T2_vfa:>5.2f} | Peso: {w['T2']:>7.4f} ")
            print(f"  H:  fisico {val_H_next:>5.2f}  -> Peso: {w['H']:>7.4f} ")
            print(f"  vent_c:  {val_vent:>5.2f} | Peso adj: {w_v_count:>7.4f} | Impatto: {w_v_count * val_vent:>8.2f}")
            print(f"  over_r1: {val_ov1:>5.2f} | Peso adj: {w_ov1:>7.4f} | Impatto: {w_ov1 * val_ov1:>8.2f}")
            print(f"  over_r2: {val_ov2:>5.2f} | Peso adj: {w_ov2:>7.4f} | Impatto: {w_ov2 * val_ov2:>8.2f}")
        print(f"{'='*65}\n")
        
    except Exception as e:
        print(f"Errore nel blocco di debug VFA: {e}")
    # =====================================================================
    # FINE BLOCCO DI DEBUG VFA

    return {
        "HeatPowerRoom1": value(m.p1),
        "HeatPowerRoom2": value(m.p2),
        "VentilationON": int(value(m.v))
    }