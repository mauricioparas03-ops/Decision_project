#%%
import pandas as pd
import numpy as np
from pathlib import Path
from pyomo.environ import *

# -----------------------------------------------------------------------------
# 1) Data Loading and System Parameters
# -----------------------------------------------------------------------------
from Data.v2_SystemCharacteristics import get_fixed_data

_data_fixed = get_fixed_data()
_num_timeslots = int(_data_fixed['num_timeslots'])
days = 100

# Define data directory path
data_dir = Path(__file__).resolve().parent / "Data"
# %%
# Load and flatten temporal data from CSVs
_occ1 = pd.read_csv(data_dir / "OccupancyRoom1.csv").values.flatten()
_occ2 = pd.read_csv(data_dir / "OccupancyRoom2.csv").values.flatten()
_df_prices = pd.read_csv(data_dir / "v2_PriceData.csv").values.flatten()
# %%
# -----------------------------------------------------------------------------
# 2) Pyomo Model Initialization
# -----------------------------------------------------------------------------
model = ConcreteModel()

# SETS
model.R = Set(initialize=[1, 2]) # Rooms
model.T = Set(initialize=range(days * _num_timeslots)) # Total time horizon
model.RT = model.R * model.T

# PARAMETERS (Exogenous Data)
model.Prices = Param(model.T, initialize=lambda m, t: float(_df_prices[t]))
model.Occ = Param(
    model.RT,
    initialize=lambda m, r, t: float(_occ1[t]) if r == 1 else float(_occ2[t])
)
model.Tout   = Param(model.T, initialize={t: _data_fixed['outdoor_temperature'][t % _num_timeslots] for t in model.T}) 

# Physical constants and thresholds
model.Pr     = Param(initialize=_data_fixed['heating_max_power'])
model.Pvent  = Param(initialize=_data_fixed['ventilation_power'])
model.Zexch  = Param(initialize=_data_fixed['heat_exchange_coeff'])
model.Zconv  = Param(initialize=_data_fixed['heating_efficiency_coeff'])
model.Zloss  = Param(initialize=_data_fixed['thermal_loss_coeff'])
model.Zcool  = Param(initialize=_data_fixed['heat_vent_coeff'])
model.Zocc   = Param(initialize=_data_fixed['heat_occupancy_coeff'])
model.Hocc   = Param(initialize=_data_fixed['humidity_occupancy_coeff'])
model.Hvent  = Param(initialize=_data_fixed['humidity_vent_coeff'])
model.Tmin   = Param(initialize=_data_fixed['temp_min_comfort_threshold'])
model.Tok    = Param(initialize=_data_fixed['temp_OK_threshold'])
model.Thigh  = Param(initialize=_data_fixed['temp_max_comfort_threshold'])
model.Hhigh  = Param(initialize=_data_fixed['humidity_threshold'])
model.Tinitial = Param(initialize=_data_fixed['T1'])
model.Hinitial = Param(initialize=_data_fixed['H'])
model.L = Param(initialize=4, within=NonNegativeIntegers)
model.M_temp = Param(initialize=100)
model.M_hum = Param(initialize=100)
model.U_vent = Param(initialize=3) # Minimum ventilation up-time

# -----------------------------------------------------------------------------
# 3) Decision Variables
# -----------------------------------------------------------------------------
model.Vent = Var(model.T, domain=Binary)    # Ventilation ON/OFF
model.s    = Var(model.T, domain=Binary)    # Ventilation startup signal
model.Heat = Var(model.RT, domain=NonNegativeReals, bounds=(0, model.Pr))

# Sensor variables for overrule logic (Big-M)
model.y_low  = Var(model.RT, domain=Binary) # 1 if T < Tmin
model.y_ok   = Var(model.RT, domain=Binary) # 1 if T > Tok
model.y_high = Var(model.RT, domain=Binary) # 1 if T > Thigh
# State variable (Memory) for heating maintenance
model.u = Var(model.RT, domain=Binary)

# Physical state variables
model.T_in = Var(model.RT, domain=NonNegativeReals)
model.Hum  = Var(model.T, domain=NonNegativeReals)

# -----------------------------------------------------------------------------
# 4) Objective Function
# -----------------------------------------------------------------------------
def total_cost_rule(model):
    heat_cost = sum(model.Prices[t] * model.Heat[r, t] for r in model.R for t in model.T)
    vent_cost = sum(model.Prices[t] * model.Vent[t] * model.Pvent for t in model.T)
    return heat_cost + vent_cost

model.obj = Objective(rule=total_cost_rule, sense=minimize)

# -----------------------------------------------------------------------------
# 5) Dynamic Constraints (Thermal and Humidity Balance)
# -----------------------------------------------------------------------------
def room_thermal_balance_rule(model, r, t):
    if t == model.T.first():
        return model.T_in[r, t] == model.Tinitial
    else:
        t_prev = t - 1
        occ = model.Occ[1, t_prev] if r == 1 else model.Occ[2, t_prev]
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
    else:
        t_prev = t - 1
        return model.Hum[t] == (
            model.Hum[t_prev]
            - model.Hvent * model.Vent[t_prev]
            + model.Hocc * (model.Occ[1, t_prev] + model.Occ[2, t_prev])
    )
model.Hum_Room_Dynamics = Constraint(model.T, rule=humidity_balance_rule)

# -----------------------------------------------------------------------------
# 6) Logic and Overrule Constraints (Big-M)
# -----------------------------------------------------------------------------

# High Temperature Logic (Forced heating shutdown)
model.c_thigh1 = Constraint(model.RT, rule=lambda m,r,t: m.T_in[r,t] >= m.Thigh - model.M_temp*(1 - m.y_high[r,t]))
model.c_thigh2 = Constraint(model.RT, rule=lambda m,r,t: m.T_in[r,t] <= m.Thigh + model.M_temp*m.y_high[r,t])
model.c_heat_off = Constraint(model.RT, rule=lambda m,r,t: m.Heat[r,t] <= m.Pr*(1 - m.y_high[r,t]))

# Low Temperature Logic (Overrule Activation)
model.c_tlow1 = Constraint(model.RT, rule=lambda m,r,t: m.T_in[r,t] <= m.Tmin + model.M_temp*(1 - m.y_low[r,t]))
model.c_tlow2 = Constraint(model.RT, rule=lambda m,r,t: m.T_in[r,t] >= m.Tmin - model.M_temp*m.y_low[r,t])

# Temperature OK Logic (Overrule Deactivation)
model.c_tok1 = Constraint(model.RT, rule=lambda m,r,t: m.T_in[r,t] >= m.Tok - model.M_temp*(1 - m.y_ok[r,t]))
model.c_tok2 = Constraint(model.RT, rule=lambda m,r,t: m.T_in[r,t] <= m.Tok + model.M_temp*m.y_ok[r,t])

# Overrule Memory Management (u)
model.c_u1 = Constraint(model.RT, rule=lambda m,r,t: m.u[r,t] >= m.y_low[r,t])
def u_rule2(m, r, t):
    u_prev = m.u[r,t-1] if t > m.T.first() else 0
    return m.u[r,t] <= u_prev + m.y_low[r,t]
model.c_u2 = Constraint(model.RT, rule=u_rule2)

model.c_heat_max = Constraint(model.RT, rule=lambda m,r,t: m.Heat[r,t] >= m.Pr * m.u[r,t])

def u_rule3(m, r, t):
    u_prev = m.u[r,t-1] if t > m.T.first() else 0
    return m.u[r,t] >= u_prev - m.y_ok[r,t]
model.c_u3 = Constraint(model.RT, rule=u_rule3)
model.c_u4 = Constraint(model.RT, rule=lambda m,r,t: m.u[r,t] <= 1 - m.y_ok[r,t])

# -----------------------------------------------------------------------------
# 7) Ventilation Constraints (Startup, Min-Up Time, Humidity)
# -----------------------------------------------------------------------------
def s_rule1(m, t):
    v_prev = m.Vent[t-1] if t > m.T.first() else 0
    return m.s[t] >= m.Vent[t] - v_prev
model.c_s1 = Constraint(model.T, rule=s_rule1)
model.c_s2 = Constraint(model.T, rule=lambda m, t: m.s[t] <= m.Vent[t])
model.c_s3 = Constraint(model.T, rule=lambda m, t: m.s[t] <= 1 - (m.Vent[t-1] if t > m.T.first() else 0))

def min_up_time_ventilation_rule(m, t):
    end_idx = min(t + m.U_vent - 1, m.L - 1)
    sum_vent = sum(m.Vent[tau] for tau in range(t, end_idx + 1))
    min_val = min(m.U_vent, m.L - t)
    return sum_vent >= min_val * m.s[t]
model.min_vent_on = Constraint(model.T, rule=min_up_time_ventilation_rule)

model.c_hum = Constraint(model.T, rule=lambda m, t: m.Hum[t] <= m.Hhigh + model.M_hum * m.Vent[t])

# -----------------------------------------------------------------------------
# 8) Model Solving
# -----------------------------------------------------------------------------
# Solver Configuration
solver = SolverFactory('gurobi')
results = solver.solve(model, tee=True)

# -----------------------------------------------------------------------------
# 9) Result Reporting
# -----------------------------------------------------------------------------
if results.solver.termination_condition == TerminationCondition.optimal:
    total_cost = value(model.obj)
    print(f"\n{'='*40}")
    print(f"{'STATUS:':<20} OPTIMAL SOLUTION FOUND")
    print(f"{'TOTAL COST:':<20} {total_cost:,.2f} DKK")
    print(f"{'SIMULATED DAYS:':<20} {days}")
    print(f"{'='*40}")
else:
    print(f"Warning: Solver terminated with status {results.solver.termination_condition}")