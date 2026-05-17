from Data.PriceProcessRestaurant import price_model 
from Data.OccupancyProcessRestaurant import next_occupancy_levels

def apply_dynamics(state, decisions, data):
    """
    Advance the real system state by one timestep.
    Follows exactly the formal MDP (Assignment Part A, Task 2).
 
    Key alignment with MDP:
      - Occupancy/Tout at time t used in dynamics (NOT t-1)       [T_{r,t+1} formula]
      - y_Low update uses <= T_low  (not strict <)                [hysteresis formula]
      - Vent overrule uses data['vent_min_up_time'] (not hardcoded)
      - High-temp overrule uses strict > T_High                   [pf formula]
    """
    t  = int(state['current_time'])
 
    p1 = float(decisions['HeatPowerRoom1'])
    p2 = float(decisions['HeatPowerRoom2'])
    v  = int(decisions['VentilationON'])
 
    o1   = float(state['Occ1'])            # kappa_{r,t}: occupancy at t  ✓ MDP
    o2   = float(state['Occ2'])
    T1   = float(state['T1'])
    T2   = float(state['T2'])
    H    = float(state['H'])
    Tout = data['outdoor_temperature'][t]  # Tout_t: outdoor temp at t    ✓ MDP
 
    heat_exchange = data['heat_exchange_coeff']
    thermal_loss  = data['thermal_loss_coeff']
    heating_eff   = data['heating_efficiency_coeff']
    vent_cooling  = data['heat_vent_coeff']
    occ_heat      = data['heat_occupancy_coeff']
    humidity_vent = data['humidity_vent_coeff']
    humidity_occ  = data['humidity_occupancy_coeff']
 
    temp_min    = data['temp_min_comfort_threshold']   # T_low
    temp_ok     = data['temp_OK_threshold']            # T_ok
    temp_max    = data['temp_max_comfort_threshold']   # T_High
    hum_high    = data['humidity_threshold']
    p_max       = data['heating_max_power']
    vent_min_up = data['vent_min_up_time']             # U_vent — from data, not hardcoded
 
    # ── Ventilation overrule ──────────────────────────────────────────────────
    # MDP: ve_t = 1 if H_t > H_high  OR  c_t in {1, ..., U_vent - 1}
    if H > hum_high or (0 < state['vent_counter'] < vent_min_up):
        v_eff = 1
    else:
        v_eff = v
 
    # ── Heating overrule room 1 ───────────────────────────────────────────────
    # MDP: pf = 0 if T > T_High (high-temp priority),  pf = P_r if y_Low = 1
    if T1 > temp_max:
        p1_eff = 0
    elif state['low_override_r1']:
        p1_eff = p_max
    else:
        p1_eff = p1
 
    # ── Heating overrule room 2 ───────────────────────────────────────────────
    if T2 > temp_max:
        p2_eff = 0
    elif state['low_override_r2']:
        p2_eff = p_max
    else:
        p2_eff = p2
 
    # ── Temperature dynamics ──────────────────────────────────────────────────
    # MDP: T_{r,t+1} = T_{r,t} + zeta_exch*(T_{r',t}-T_{r,t})
    #                           + zeta_loss*(Tout_t - T_{r,t})
    #                           + zeta_conv*pf_{r,t}
    #                           - zeta_cool*ve_t
    #                           + zeta_occ*kappa_{r,t}
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
 
    # ── Ventilation counter ───────────────────────────────────────────────────
    vent_counter_next = (state['vent_counter'] + 1) if v_eff == 1 else 0
 
    # ── Low-temperature hysteresis ────────────────────────────────────────────
    # MDP: y_Low_{t+1} = 1  if T_next <= T_low        <- <= (not strict <)
    #                  = 1  if y_Low==1 AND T_next < T_ok
    #                  = 0  otherwise
    if T1_next <= temp_min:
        low_override_r1_next = 1
    elif state['low_override_r1'] and T1_next < temp_ok:
        low_override_r1_next = 1
    else:
        low_override_r1_next = 0
 
    if T2_next <= temp_min:
        low_override_r2_next = 1
    elif state['low_override_r2'] and T2_next < temp_ok:
        low_override_r2_next = 1
    else:
        low_override_r2_next = 0
 
    # ── Real cost ─────────────────────────────────────────────────────────────
    # MDP: r(s,a) = -lambda_t * (sum_r pf_{r,t} + P_vent * ve_t)
    real_cost = state['price_t'] * (
        data['ventilation_power'] * v_eff
        + p1_eff
        + p2_eff
    )
 
    next_state = {
        'T1':              T1_next,
        'T2':              T2_next,
        'H':               H_next,
        'Occ1':            o1,
        'Occ2':            o2,
        'price_t':         state['price_t'],
        'price_previous':  state['price_previous'],
        'vent_counter':    vent_counter_next,
        'low_override_r1': low_override_r1_next,
        'low_override_r2': low_override_r2_next,
        'current_time':    t + 1,
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


