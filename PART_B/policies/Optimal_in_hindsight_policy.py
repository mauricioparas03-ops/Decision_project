from xml.parsers.expat import model

from numpy import block
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

    model.Tinitial = Param(initialize=_data_fixed['T1'])  # Initial temperature (same for both rooms)
    model.Hinitial = Param(initialize=_data_fixed['H'])  # Initial humidity

    # DECISION VARIABLES
    model.Vent = Var(model.T, domain=Binary)
    model.s = Var(model.T, domain=Binary)  # startuo ventilation
    model.Heat = Var(model.RT, domain=NonNegativeReals, bounds=(0, model.Pr))

    # sensor variables for overrules
    model.y_low = Var(model.RT, domain=Binary)
    model.y_ok = Var(model.RT, domain=Binary)
    model.y_high = Var(model.RT, domain=Binary)

    # Memory for overrules
    model.u = Var(model.RT, domain=Binary)

    model.T_in = Var(model.RT, domain=NonNegativeReals)
    model.Hum = Var(model.T, domain=NonNegativeReals)


    # OBJECTIVE 
    def total_cost_rule(model):
        heat_cost = sum(model.Prices[t] * model.Heat[r, t] for r in model.R for t in model.T)
        vent_cost = sum(model.Prices[t] * model.Vent[t] * model.Pvent for t in model.T)
        return heat_cost + vent_cost

    model.obj = Objective(rule=total_cost_rule, sense=minimize)


    # DYNAMICS 
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


    # CONSTRAINTS
    M_temp = 100
    M_hum = 100
    U_vent = 3

    # Temperature over high threshold
    def t_high_rule1(m, r, t): 
        return m.T_in[r,t] >= m.Thigh - M_temp*(1 - m.y_high[r,t])
    model.c_thigh1 = Constraint(model.RT, rule=t_high_rule1)

    def t_high_rule2(m, r, t): 
        return m.T_in[r,t] <= m.Thigh + M_temp*m.y_high[r,t]
    model.c_thigh2 = Constraint(model.RT, rule=t_high_rule2)

    #Overrule to turn off heating
    def heat_off_rule(m, r, t): 
        return m.Heat[r,t] <= m.Pr*(1 - m.y_high[r,t])
    model.c_heat_off = Constraint(model.RT, rule=heat_off_rule)

    #temperature under low threshold
    def t_low_rule1(m, r, t): 
        return m.T_in[r,t] <= m.Tmin + M_temp*(1 - m.y_low[r,t])
    model.c_tlow1 = Constraint(model.RT, rule=t_low_rule1)

    def t_low_rule2(m, r, t): 
        return m.T_in[r,t] >= m.Tmin - M_temp*m.y_low[r,t]
    model.c_tlow2 = Constraint(model.RT, rule=t_low_rule2)

    #Temperature over OK threshold
    def t_ok_rule1(m, r, t): 
        return m.T_in[r,t] >= m.Tok - M_temp*(1 - m.y_ok[r,t])
    model.c_tok1 = Constraint(model.RT, rule=t_ok_rule1)

    def t_ok_rule2(m, r, t): 
        return m.T_in[r,t] <= m.Tok + M_temp*m.y_ok[r,t]
    model.c_tok2 = Constraint(model.RT, rule=t_ok_rule2)

    #Overrule control logic
    def u_rule1(m, r, t): 
        return m.u[r,t] >= m.y_low[r,t]
    model.c_u1 = Constraint(model.RT, rule=u_rule1)

    def u_rule2(m, r, t):
        u_prev = m.u[r,t-1] if t > m.T.first() else 0
        return m.u[r,t] <= u_prev + m.y_low[r,t]
    model.c_u2 = Constraint(model.RT, rule=u_rule2)

    def heat_max_rule(m, r, t): 
        return m.Heat[r,t] >= m.Pr * m.u[r,t]
    model.c_heat_max = Constraint(model.RT, rule=heat_max_rule)

    def u_rule3(m, r, t):
        u_prev = m.u[r,t-1] if t > m.T.first() else 0
        return m.u[r,t] >= u_prev - m.y_ok[r,t]
    model.c_u3 = Constraint(model.RT, rule=u_rule3)

    def u_rule4(m, r, t):
        return m.u[r,t] <= 1 - m.y_ok[r,t]
    model.c_u4 = Constraint(model.RT, rule=u_rule4)

    # Ventilation Startup and Minimum Up-Time
    def s_rule1(m, t):
        v_prev = m.Vent[t-1] if t > m.T.first() else 0
        return m.s[t] >= m.Vent[t] - v_prev
    model.c_s1 = Constraint(model.T, rule=s_rule1)

    def s_rule2(m, t): 
        return m.s[t] <= m.Vent[t]
    model.c_s2 = Constraint(model.T, rule=s_rule2)

    def s_rule3(m, t):
        v_prev = m.Vent[t-1] if t > m.T.first() else 0
        return m.s[t] <= 1 - v_prev
    model.c_s3 = Constraint(model.T, rule=s_rule3)

    def min_up_time_ventilation_rule(m, t):
        L_total = len(m.T)
        end_idx = min(t + U_vent - 1, L_total - 1)
        sum_vent = sum(m.Vent[tau] for tau in range(t, end_idx + 1))
        min_val = min(U_vent, L_total - t)
        return sum_vent >= min_val * m.s[t]
    model.min_vent_on = Constraint(model.T, rule=min_up_time_ventilation_rule)

    #ventilation fored by humidity
    def hum_rule(m, t): 
        return m.Hum[t] <= m.Hhigh + M_hum * m.Vent[t]
    model.c_hum = Constraint(model.T, rule=hum_rule)


    # SOLVER
    solver = SolverFactory('gurobi')
    solver.solve(model, tee=False)

    for t in model.T:
        _p1_opt[t] = value(model.Heat[1, t])
        _p2_opt[t] = value(model.Heat[2, t])
        _v_opt[t] = int(round(value(model.Vent[t])))

    _is_solved = True


def select_action(state):

    current_t = int(state["current_time"])

    if current_t == 0 and not _is_solved:
        solve_daily_milp()

    HereAndNowActions = {
        "HeatPowerRoom1": float(_p1_opt[current_t]),
        "HeatPowerRoom2": float(_p2_opt[current_t]),
        "VentilationON": int(_v_opt[current_t]),
    }

    return HereAndNowActions