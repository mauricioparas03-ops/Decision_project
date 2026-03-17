from SystemCharacteristics import get_fixed_data
    
def apply_dynamics(self, state, action, data):
    """
    Advance the real system state by one timestep.

    Parameters
    ----------
    state : dict with keys
        T_in_r1       – room 1 temperature (°C)
        T_in_r2       – room 2 temperature (°C)
        humidity      – humidity level (%)
        vent_prev     – ventilation status at current t (0 or 1)
        vent_on_count – consecutive hours ventilation has been ON
        t             – current hour index (0-based)
        occ_r1        – room 1 occupancy at current t
        occ_r2        – room 2 occupancy at current t

    decisions : dict with keys
        p1  – heating power room 1 (kW)
        p2  – heating power room 2 (kW)
        v   – ventilation (0 or 1)

    Returns
    -------
    next_state : dict with same keys as state, advanced by one step.
                occ_r1 and occ_r2 are set to None — filled by environment.
    """
    d    = get_fixed_data()
    t    = state['t']
    p1   = decisions[0]
    p2   = decisions[1]
    v    = decisions[2]
    o1   = gen_data[1][t]
    o2   = gen_data[2][t]
    T1   = state['T_in_r1']
    T2   = state['T_in_r2']
    H    = state['humidity']
    Tout = d['outdoor_temperature'][t]

    # ── Temperature dynamics ──────────────────────────────────────────────────
    T1_next = (T1
            + d['heat_exchange_coeff']      * (T2   - T1)
            + d['thermal_loss_coeff']       * (Tout - T1)
            + d['heating_efficiency_coeff'] * p1
            - d['heat_vent_coeff']          * v
            + d['heat_occupancy_coeff']     * o1)

    T2_next = (T2
            + d['heat_exchange_coeff']      * (T1   - T2)
            + d['thermal_loss_coeff']       * (Tout - T2)
            + d['heating_efficiency_coeff'] * p2
            - d['heat_vent_coeff']          * v
            + d['heat_occupancy_coeff']     * o2)

    # ── Humidity dynamics ─────────────────────────────────────────────────────
    H_next = max(0.0,
                H
                - d['humidity_vent_coeff']      * v
                + d['humidity_occupancy_coeff'] * (o1 + o2))

    # ── Ventilation inertia counter ───────────────────────────────────────────
    if v == 1 and state['vent_status'] == 0:
        vent_on_count_next = 1
    elif v == 1:
        vent_on_count_next = min(state['vent_on_count'] + 1, 3)
    else:
        vent_on_count_next = 0 

    return {
        'T_in_r1'      : T1_next,
        'T_in_r2'      : T2_next,
        'humidity'     : H_next,
        'vent_status'    : v,
        'vent_on_count': vent_on_count_next,
        't'            : t + 1,
        'occ_r1'       : None,   # filled by environment at next step
        'occ_r2'       : None,   # filled by environment at next step
        'price_now'    : None,
        'price_prev'   : state['price_now'],  # for next step's scenario generation
    } #returns next_state dictionary
    return next_state
def cost_function(decisions, state):
    params = get_fixed_data()

    ventilation_power = params["ventilation_power"]
    cost = state["price_t"] * (ventilation_power * decisions["v"] + decisions["p1"] + decisions["p2"])
    return cost

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


