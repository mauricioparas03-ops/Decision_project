from pyomo.environ import *

_data_fixed = None
_prices = None
_occ_r1 = None
_occ_r2 = None
_p1_opt = []
_p2_opt = []
_v_opt = []
_is_solved = False


def initialize_policy(data_fixed, daily_prices, daily_occ_r1, daily_occ_r2):
    """
    Initializes the policy with perfect foresight data for the current day.
    daily_prices, daily_occ_r1, daily_occ_r2 are expected to be lists/arrays of 10 elements.
    """
    global _data_fixed, _prices, _occ_r1, _occ_r2, _p1_opt, _p2_opt, _v_opt, _is_solved

    _data_fixed = data_fixed
    _prices = daily_prices
    _occ_r1 = daily_occ_r1
    _occ_r2 = daily_occ_r2

    num_timeslots = _data_fixed['num_timeslots']
    _p1_opt = [0.0] * num_timeslots
    _p2_opt = [0.0] * num_timeslots
    _v_opt = [0] * num_timeslots
    _is_solved = False


def solve_daily_milp():
    global _is_solved

    if _data_fixed is None:
        raise ValueError("Policy data not initialized. Call initialize_policy(...) first.")

    model = ConcreteModel()

    # SETS
    model.R = Set(initialize=[1, 2])
    model.T = Set(initialize=range(_data_fixed['num_timeslots']))
    model.RT = model.R * model.T

    # PARAMETERS
    model.Prices = Param(model.T, initialize=lambda m, t: _prices[t])
    model.Occ1 = Param(model.T, initialize=lambda m, t: _occ_r1[t])
    model.Occ2 = Param(model.T, initialize=lambda m, t: _occ_r2[t])
    model.Tout = Param(model.T, initialize=lambda m, t: _data_fixed['outdoor_temperature'][t])

    model.Pr = Param(initialize=_data_fixed['heating_max_power'])
    model.Pvent = Param(initialize=_data_fixed['ventilation_power'])

    model.Zexch = Param(initialize=_data_fixed['heat_exchange_coeff'])
    model.Zconv = Param(initialize=_data_fixed['heating_efficiency_coeff'])
    model.Zloss = Param(initialize=_data_fixed['thermal_loss_coeff'])
    model.Zcool = Param(initialize=_data_fixed['heat_vent_coeff'])
    model.Zocc = Param(initialize=_data_fixed['heat_occupancy_coeff'])

    model.Hocc = Param(initialize=_data_fixed['humidity_occupancy_coeff'])
    model.Hvent = Param(initialize=_data_fixed['humidity_vent_coeff'])

    model.Tmin = Param(initialize=_data_fixed['temp_min_comfort_threshold'])
    model.Tok = Param(initialize=_data_fixed['temp_OK_threshold'])
    model.Thigh = Param(initialize=_data_fixed['temp_max_comfort_threshold'])
    model.Hhigh = Param(initialize=_data_fixed['humidity_threshold'])

    model.Tinitial = Param(initialize=_data_fixed['initial_temperature'])
    model.Hinitial = Param(initialize=_data_fixed['initial_humidity'])

    # DECISION VARIABLES
    model.Vent = Var(model.T, domain=Binary)
    model.Heat = Var(model.RT, domain=NonNegativeReals, bounds=(0, model.Pr))

    model.w = Var(model.RT, domain=Binary)
    model.u = Var(model.RT, domain=Binary)
    model.y = Var(model.RT, domain=Binary)

    model.T_in = Var(model.RT, domain=NonNegativeReals)
    model.Hum = Var(model.T, domain=NonNegativeReals)

    def total_cost_rule(model):
        heat_cost = sum(model.Prices[t] * model.Heat[r, t] for r in model.R for t in model.T)
        vent_cost = sum(model.Prices[t] * model.Vent[t] * model.Pvent for t in model.T)
        return heat_cost + vent_cost

    model.obj = Objective(rule=total_cost_rule, sense=minimize)

    def room_thermal_balance_rule(model, r, t):
        if t == model.T.first():
            return model.T_in[r, t] == model.Tinitial

        t_prev = t - 1
        occ = model.Occ1[t_prev] if r == 1 else model.Occ2[t_prev]
        r_other = 2 if r == 1 else 1

        return model.T_in[r, t] == (
            model.T_in[r, t_prev]
            + model.Zexch * (model.T_in[r_other, t_prev] - model.T_in[r, t_prev])
            + model.Zloss * (model.Tout[t_prev] - model.T_in[r, t_prev])
            + model.Zconv * model.Heat[r, t_prev]
            - model.Zcool * model.Vent[t_prev]
            + model.Zocc * occ
        )

    model.Temp_Room_Dynamics = Constraint(model.RT, rule=room_thermal_balance_rule)

    def humidity_balance_rule(model, t):
        if t == model.T.first():
            return model.Hum[t] == model.Hinitial

        t_prev = t - 1
        return model.Hum[t] == (
            model.Hum[t_prev]
            - model.Hvent * model.Vent[t_prev]
            + model.Hocc * (model.Occ1[t_prev] + model.Occ2[t_prev])
        )

    model.Hum_Room_Dynamics = Constraint(model.T, rule=humidity_balance_rule)

    def max_temp_low_rule(block):
        M = 100
        m_ref = block.model()
        eps = 0.0001

        def u_binary_logic_rule(block, r, t):
            return m_ref.T_in[r, t] >= m_ref.Tmin - M * m_ref.u[r, t]

        block.u_logic = Constraint(m_ref.RT, rule=u_binary_logic_rule)

        def low_temp_heat_rule(block, r, t):
            return m_ref.T_in[r, t] >= m_ref.Tok + eps - M * (1 - m_ref.w[r, t])

        block.low_temp_heat = Constraint(m_ref.RT, rule=low_temp_heat_rule)

        def u_w_exclusivity_rule(block, r, t):
            if t == m_ref.T.first():
                m_ref.u[r, t].fix(0)
                m_ref.w[r, t].fix(0)
                return Constraint.Skip
            return m_ref.u[r, t] >= m_ref.u[r, t - 1] - m_ref.w[r, t]

        block.u_w_exclusivity = Constraint(m_ref.RT, rule=u_w_exclusivity_rule)

        def set_max_power_rule(block, r, t):
            return m_ref.Heat[r, t] >= m_ref.Pr * m_ref.u[r, t]

        block.set_max_power = Constraint(m_ref.RT, rule=set_max_power_rule)

    model.set_max_temp_rule = Block(rule=max_temp_low_rule)

    def set_power_off_rule(block):
        M = 100
        m_ref = block.model()

        def y_binary_logic_rule(block, r, t):
            return m_ref.T_in[r, t] <= m_ref.Thigh + M * m_ref.y[r, t]

        block.y_logic = Constraint(m_ref.RT, rule=y_binary_logic_rule)

        def power_off_rule(block, r, t):
            return m_ref.Heat[r, t] <= m_ref.Pr * (1 - m_ref.y[r, t])

        block.set_power_off = Constraint(m_ref.RT, rule=power_off_rule)

    model.set_power_off_rule = Block(rule=set_power_off_rule)

    def set_vent_on_rule(model, t):
        M = 100
        return model.Hum[t] <= model.Hhigh + M * model.Vent[t]

    model.vent_on_rule = Constraint(model.T, rule=set_vent_on_rule)

    def min_up_time_ventilation_rule(model, t):
        L = 3
        remaining_steps = [k for k in range(t, t + L) if k <= model.T.last()]
        v_prev = model.Vent[t - 1] if t > model.T.first() else 0
        return sum(model.Vent[k] for k in remaining_steps) >= len(remaining_steps) * (model.Vent[t] - v_prev)

    model.min_vent_on = Constraint(model.T, rule=min_up_time_ventilation_rule)

    solver = SolverFactory('gurobi')
    solver.solve(model, tee=False)

    for t in model.T:
        _p1_opt[t] = value(model.Heat[1, t])
        _p2_opt[t] = value(model.Heat[2, t])
        _v_opt[t] = int(round(value(model.Vent[t])))

    _is_solved = True


def select_action(state):
    """
    Decision rule called by the environment at every hour t.
    """
    current_t = state["current_time"]

    if current_t == 0 and not _is_solved:
        solve_daily_milp()

    HereAndNowActions = {
        "HeatPowerRoom1": float(_p1_opt[current_t]),
        "HeatPowerRoom2": float(_p2_opt[current_t]),
        "Ventilation": int(_v_opt[current_t]),
    }

    return HereAndNowActions