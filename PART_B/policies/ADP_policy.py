from pyomo.environ import *
from Data.v2_SystemCharacteristics import get_fixed_data


# definitive wheights for the VFA, obtained after training on 500 days with a linear regression on the collected data
VFA_WEIGHTS = {
    0: {'T1': -7.1132, 'T2': -4.9259, 'H': 0.3993, 'price_t': 47.3246, 'vent_counter': 14.0540, 'low_override_r1': 31.9754, 'low_override_r2': -1.3139, 'intercept': 108.8559, },
    1: {'T1': -4.5161, 'T2': -7.5977, 'H': 0.4620, 'price_t': 26.4468, 'vent_counter': 1.4653, 'low_override_r1': 21.9589, 'low_override_r2': 2.1926, 'intercept': 201.5129, },
    2: {'T1': -5.8139, 'T2': -7.4732, 'H': 0.5764, 'price_t': 19.4310, 'vent_counter': -2.9355, 'low_override_r1': 21.0780, 'low_override_r2': -3.0473, 'intercept': 244.7286, },
    3: {'T1': -7.7707, 'T2': -4.1875, 'H': 0.5017, 'price_t': 15.5271, 'vent_counter': -1.4120, 'low_override_r1': 21.7614, 'low_override_r2': 3.4125, 'intercept': 218.3277, },
    4: {'T1': -3.3780, 'T2': -5.8786, 'H': 0.4014, 'price_t': 12.8491, 'vent_counter': -0.6709, 'low_override_r1': 22.7219, 'low_override_r2': 6.6950, 'intercept': 165.2901, },
    5: {'T1': -3.6561, 'T2': -3.2642, 'H': 0.5472, 'price_t': 10.5746, 'vent_counter': 0.5681, 'low_override_r1': 24.1337, 'low_override_r2': 14.9623, 'intercept': 104.4287, },
    6: {'T1': -3.9434, 'T2': -2.8828, 'H': 0.5332, 'price_t': 9.7436, 'vent_counter': -0.6949, 'low_override_r1': 23.9571, 'low_override_r2': 24.2764, 'intercept': 97.6755, },
    7: {'T1': 0.3850, 'T2': -4.2068, 'H': 0.3613, 'price_t': 8.6409, 'vent_counter': 0.1511, 'low_override_r1': 24.9123, 'low_override_r2': 30.9137, 'intercept': 38.7269, },
    8: {'T1': -2.0624, 'T2': -0.5792, 'H': 0.3323, 'price_t': 6.5563, 'vent_counter': 0.1911, 'low_override_r1': 23.2699, 'low_override_r2': 23.5573, 'intercept': 16.3067, },
    9: {'T1': -0.7604, 'T2': 1.0433, 'H': 0.0535, 'price_t': 3.2502, 'vent_counter': -0.0238, 'low_override_r1': 11.5441, 'low_override_r2': 10.6425, 'intercept': -17.6645, },
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

    # #VFA variables to calculate reward
    # m.T1_vfa = Var()
    # m.T2_vfa = Var()
    # m.H_vfa = Var()

    #OVERRULE FOR CURRENT STEP
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
    
    #DYNAMICS OF NEXT STATE
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

    #MAPPING FOR FUTURE OVERRULES
    M = 100
    eps = 0.001 # small constant to avoid numerical issues in the thresholding of the overrules (e.g., if T1 is exactly equal to temp_min_comfort_threshold, we want to be sure that we are in the low-temperature override state)
    
    thresh1 = (data['temp_min_comfort_threshold'] if state['low_override_r1'] == 0 else data['temp_OK_threshold']) + eps

    m.c_low1_a = Constraint(expr=m.T1_next >= thresh1 - M * m.low_override_r1_next)
    m.c_low1_b = Constraint(expr=m.T1_next <= thresh1 + M * (1 - m.low_override_r1_next))

    thresh2 = (data['temp_min_comfort_threshold'] if state['low_override_r2'] == 0 else data['temp_OK_threshold']) + eps

    m.c_low2_a = Constraint(expr=m.T2_next >= thresh2 - M * m.low_override_r2_next)
    m.c_low2_b = Constraint(expr=m.T2_next <= thresh2 + M * (1 - m.low_override_r2_next))

#     T_TARGET = data['temp_max_comfort_threshold']
#     # Relaxed humidity target to avoid forcing unnecessary ventilation when we are already below the risk threshold
#     H_TARGET = data['humidity_threshold'] - 2.0

#     # If w < 0, the solver pushes to maximize T_vfa. We lock it at the target, turning off the incentive beyond that threshold.
#     m.c_vfa_t1_a = Constraint(expr=m.T1_vfa <= m.T1_next)
#     m.c_vfa_t1_b = Constraint(expr=m.T1_vfa <= T_TARGET)

#     m.c_vfa_t2_a = Constraint(expr=m.T2_vfa <= m.T2_next)
#     m.c_vfa_t2_b = Constraint(expr=m.T2_vfa <= T_TARGET)

# # If w > 0, the solver pushes to minimize H_vfa. We lock it to the target (it doesn't go below it), turning off the incentive.
#     m.c_vfa_h_a = Constraint(expr=m.H_vfa >= m.H_next)
#     m.c_vfa_h_b = Constraint(expr=m.H_vfa >= H_TARGET)

    #APPROXIMATE VALUE FUNCTION 
    immediate_cost = state['price_t'] * (m.p1 + m.p2 + m.v * data['ventilation_power'])
    

    t = int(state['current_time'])
    #If not at the last hour use weiths to forecast future
    if t < 9:
        w = VFA_WEIGHTS[t+1]
        
        
        expected_future_cost = (
            w['intercept'] +   
            w['T1'] * m.T1_next +  
            w['T2'] * m.T2_next + 
            w['H'] * m.H_next +
            w['vent_counter'] * m.vent_counter_next + 
            w['low_override_r1'] * m.low_override_r1_next + 
            w['low_override_r2'] * m.low_override_r2_next
        )
    else:
        # end of the day no cost
        expected_future_cost = 0.0

    m.obj = Objective(expr=immediate_cost + expected_future_cost, sense=minimize)

    #SOLVER
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
        
        # # 2. Estrazione stati VFA (visione matematica limitata)
        # val_T1_vfa = value(m.T1_vfa)
        # val_T2_vfa = value(m.T2_vfa)
        # val_H_vfa  = value(m.H_vfa)
        
        val_costo_immediato = value(immediate_cost)
        
        if t < 9:
            val_costo_futuro = value(expected_future_cost)
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
            
            # Pesi corretti come nel modello
            w_v_count = max(0, w['vent_counter'])
            w_ov1 = max(0, w['low_override_r1'])
            w_ov2 = max(0, w['low_override_r2'])

            print("Dettaglio fisica vs Trust Region VFA scelti dal solver:")
            print(f"  T1: fisico {val_T1_next:>5.2f} ->  Peso: {w['T1']:>7.4f} ")
            print(f"  T2: fisico {val_T2_next:>5.2f} ->  Peso: {w['T2']:>7.4f} ")
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