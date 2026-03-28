from Data.PriceProcessRestaurant import price_model 
from Data.OccupancyProcessRestaurant import next_occupancy_levels

def apply_dynamics(state, decisions, data):
    """
    Advance the real system state by one timestep.

    Parameters
    ----------
    state : dict with keys
        T1, T2, H, Occ1, Occ2, price_t, price_previous,
        vent_counter, low_override_r1, low_override_r2, current_time

    decisions : dict with keys
        HeatPowerRoom1, HeatPowerRoom2, VentilationON

    Returns
    -------
    next_state : dict with same structure as init_state in Environment.py.
    """
    t = int(state['current_time'])
    t_idx = max(0, min(t, len(data['outdoor_temperature']) - 1))

    p1 = float(decisions['HeatPowerRoom1'])
    p2 = float(decisions['HeatPowerRoom2'])
    v = int(decisions['VentilationON'])

    o1 = float(state['Occ1'])
    o2 = float(state['Occ2'])
    T1 = float(state['T1'])
    T2 = float(state['T2'])
    H = float(state['H'])
    Tout = data['outdoor_temperature'][t_idx]

    heat_exchange = data['heat_exchange_coeff']
    thermal_loss = data['thermal_loss_coeff']
    heating_eff = data['heating_efficiency_coeff']
    vent_cooling = data['heat_vent_coeff']
    occ_heat = data['heat_occupancy_coeff']
    humidity_vent = data['humidity_vent_coeff']
    humidity_occ = data['humidity_occupancy_coeff']

    # --- OVERRULES (ambient) ---
    temp_min = data['temp_min_comfort_threshold']
    temp_ok = data['temp_OK_threshold']
    temp_max = data['temp_max_comfort_threshold']
    hum_high = data['humidity_threshold']
    p_max = data['heating_max_power']

    # Overrule humidity and ventilation
    if H > hum_high or state['vent_counter'] in [1, 2]:
        v_eff = 1
    else:
        v_eff = v

    # Overrule temperature room 1
    if T1 > temp_max:
        p1_eff = 0
    elif state['low_override_r1'] == 1: # if we were in a low-temperature override state in the previous step, we stay in it until we reach temp_ok
        p1_eff = p_max
    else:
        p1_eff = p1

    # Overrule temperature room 2
    if T2 > temp_max:
        p2_eff = 0
    elif state['low_override_r2'] == 1: # if we were in a low-temperature override state in the previous step, we stay in it until we reach temp_ok
        p2_eff = p_max
    else:
        p2_eff = p2

    # ── Temperature dynamics ──────────────────────────────────────────────────
    T1_next = (T1
            + heat_exchange * (T2   - T1)
            + thermal_loss  * (Tout - T1)
            + heating_eff   * p1_eff
            - vent_cooling  * v_eff
            + occ_heat      * o1)

    T2_next = (T2
            + heat_exchange * (T1   - T2)
            + thermal_loss  * (Tout - T2)
            + heating_eff   * p2_eff
            - vent_cooling  * v_eff
            + occ_heat      * o2)

    # ── Humidity dynamics ─────────────────────────────────────────────────────
    H_next = max(0.0,
                H
                - humidity_vent * v_eff
                + humidity_occ  * (o1 + o2))

    # ── Ventilation inertia counter ───────────────────────────────────────────
    if v_eff == 1:
        vent_counter_next = state['vent_counter'] + 1
    else:
        vent_counter_next = 0

    # ── Low-temperature hysteresis update ───────────────────────────────────────────────
    if T1_next <= temp_min:
        low_override_r1_next = 1
    elif state['low_override_r1'] == 1 and T1_next < temp_ok:
        low_override_r1_next = 1
    else:
        low_override_r1_next = 0
        
    if T2_next <= temp_min:
        low_override_r2_next = 1
    elif state['low_override_r2'] == 1 and T2_next < temp_ok:
        low_override_r2_next = 1
    else:
        low_override_r2_next = 0


    #we don't need to model the price and occupancy dynamics here, as they are exogenous and will be read from the data in Environment.py. 
    #occ1_next, occ2_next = next_occupancy_levels(o1, o2)
    #price_next = price_model(state['price_t'], state['price_previous'])
    
    
    # REAL COST
    # cost after overrules
    real_cost = state['price_t'] * (
        vent_cooling * v_eff 
        + p1_eff
        + p2_eff
    )

    real_cost = state['price_t'] * (
        data['ventilation_power'] * v_eff 
        + p1_eff 
        + p2_eff
    )
    
    next_state = {
        'T1': T1_next,
        'T2': T2_next,
        'H': H_next,
        'Occ1': o1,
        'Occ2': o2,
        'price_t': state['price_t'], 
        'price_previous': state['price_previous'], 
        'vent_counter': vent_counter_next,
        'low_override_r1': low_override_r1_next,
        'low_override_r2': low_override_r2_next,
        'current_time': t + 1,
    }
    
    return next_state, real_cost

def check_feasibility(decisions, power_max):
    p1 = decisions["HeatPowerRoom1"]
    p2 = decisions["HeatPowerRoom2"]
    v = decisions["VentilationON"]

    if p1 < 0 or p1 > power_max[1]:
        return False
    if p2 < 0 or p2 > power_max[2]:
        return False
    if v not in [0, 1]:
        return False
    

    return True


