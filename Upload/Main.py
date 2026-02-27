import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pyomo.environ import *
import gurobipy as grb
from PlotsRestaurant import *
from SystemCharacteristics import *
from Functions import *

# -----------------------------------------------------------------------------
# 1) Locate input data files
#    Try current script folder first, then parent folder.
# -----------------------------------------------------------------------------
script_dir = Path(__file__).resolve().parent
required_files = ("OccupancyRoom1.csv", "OccupancyRoom2.csv", "PriceData.csv")
candidate_dirs = (script_dir, script_dir.parent)

data_dir = None
for folder in candidate_dirs:
    if all((folder / filename).exists() for filename in required_files):
        data_dir = folder
        break

if data_dir is None:
    raise FileNotFoundError(
        f"Could not find required data files {required_files} in {script_dir} or {script_dir.parent}."
    )

# -----------------------------------------------------------------------------
# 2) Load CSV inputs (occupancy and electricity prices)
# -----------------------------------------------------------------------------
Occ_r1 = pd.read_csv(data_dir / "OccupancyRoom1.csv")
Occ_r2 = pd.read_csv(data_dir / "OccupancyRoom2.csv")
prices = pd.read_csv(data_dir / "PriceData.csv")

# Ensure hour labels are integer-indexed for consistent dictionary keys.
Occ_r1.columns = Occ_r1.columns.astype(int)
Occ_r2.columns = Occ_r2.columns.astype(int)
prices.columns = prices.columns.astype(int)

# -----------------------------------------------------------------------------
# 3) Prepare fixed parameters and build time-price lookup dictionary
# -----------------------------------------------------------------------------
data_fixed = get_fixed_data()
prices_dict = {}
for day_idx, row in prices.iterrows():
    for hour in prices.columns:
        prices_dict[(day_idx, int(hour))] = row[hour]


# -----------------------------------------------------------------------------
# 4) Create Pyomo model and index sets
# -----------------------------------------------------------------------------
model = ConcreteModel()
day = 0
# Sets
model.R = Set(initialize=[1, 2])  # Rooms
model.T = Set(initialize=range(data_fixed['num_timeslots'])) #Time steps
model.D = Set(initialize=range(Occ_r1.shape[0])) # Days (for occupancy data)
model.TD = model.T * model.D # Time-Day combinations
model.RTD = model.R*model.T*model.D # Room-Time-Day combinations


# -----------------------------------------------------------------------------
# 5) Define model parameters (physical coefficients, limits, and exogenous data)
# -----------------------------------------------------------------------------
# Parameters
model.Pr = Param(initialize=data_fixed['heating_max_power']) # Maximum heating power (kW)
model.Zexch = Param(initialize=data_fixed['heat_exchange_coeff']) # Heat exchange coefficient between rooms
model.Zconv = Param(initialize=data_fixed['heating_efficiency_coeff']) # Heating efficiency
model.Zloss = Param(initialize=data_fixed['thermal_loss_coeff']) # Thermal loss coefficient
model.Zcool = Param(initialize=data_fixed['heat_vent_coeff']) # Ventilation cooling effect
model.Zocc = Param(initialize=data_fixed['heat_occupancy_coeff']) # Occupancy
model.Tmin = Param(initialize=data_fixed['temp_min_comfort_threshold']) # Lower threshold for Overrule heater activation
model.Tok = Param(initialize=data_fixed['temp_OK_threshold']) # Temperature above which the Overrule controller is deactived
model.Thigh = Param(initialize=data_fixed['temp_max_comfort_threshold']) # Hard upper limit: when exceeded, heater must be OFF
model.Hhigh = Param(initialize=data_fixed['humidity_threshold']) # Humidity threshold above which overrule controller forces ventilation ON (%)
model.Pvent = Param(initialize=data_fixed['ventilation_power']) # Electrical power consumption of ventilation when ON (kW)
model.Hocc = Param(initialize=data_fixed['humidity_occupancy_coeff']) # Degrees of humidity increase per hour per person
model.Hvent = Param(initialize=data_fixed['humidity_vent_coeff']) # Degrees of humidity decrease per hour that ventilation is ON
model.Tinitial = Param(initialize=data_fixed['initial_temperature']) # Initial temperature value
model.Hinitial = Param(initialize=data_fixed['initial_humidity']) # Initial humidity value
model.Tout = Param(model.T, initialize={t: data_fixed['outdoor_temperature'][t] for t in model.T}) # Outdoor temperature (°C)
model.Occ1 = Param(model.D, model.T, initialize=Occ_r1.stack().to_dict())
model.Occ2 = Param(model.D, model.T, initialize=Occ_r2.stack().to_dict())
model.prices = Param(model.D, model.T, initialize=prices_dict)

# -----------------------------------------------------------------------------
# 6) Define decision variables
# -----------------------------------------------------------------------------
# Variables
model.Vent = Var(model.TD, domain=Binary) # Ventilation ON/OFF
model.Uon = Var(model.TD, domain=Binary) # Ventilation start-up
model.Uoff = Var(model.TD, domain=Binary) # Ventilation shut-down
model.Heat = Var(model.RTD, domain=NonNegativeReals, bounds=(0, model.Pr)) # Heating power (kW)
model.w = Var(model.RTD, domain=Binary) # T is higher thatn Tok
model.u = Var(model.RTD, domain=Binary) # T is lower than Tlow
model.z = Var(model.RTD, domain=Binary) # T is lower than Tok
model.y = Var(model.RTD, domain=Binary) # T is higher than Thigh
model.T_in = Var(model.RTD, domain=NonNegativeReals) # Indoor temperature (°C)
model.Hum = Var(model.TD, domain=NonNegativeReals) # Indoor humidity (%)

# -----------------------------------------------------------------------------
# 7) Add objective function (minimize total operating cost)
# -----------------------------------------------------------------------------
model.obj = Objective(rule=total_cost_rule, sense=minimize)

# -----------------------------------------------------------------------------
# 8) Add dynamic and logic constraints
# -----------------------------------------------------------------------------
# Enforces indoor temperature evolution each hour from heat exchange, losses, heating, ventilation, and occupancy.
model.Temp_Room_Dynamics = Constraint(model.RTD, rule=room_thermal_balance_rule)
# Enforces indoor humidity evolution each hour from ventilation removal and occupancy generation.
model.Hum_Room_Dynamics = Constraint(model.TD, rule=humidity_balance_rule)

# Adds low-temperature control logic (binary activation) that can force heater power up when too cold.
model.set_max_temp_rule = Block(rule=max_temp_low_rule)

# Adds high-temperature safety logic (binary activation) that can force heater power down/off when too hot.
model.set_power_off_rule = Block(rule=set_power_off_rule)

# Links high humidity to ventilation start signal through a big-M trigger condition.
model.vent_on_rule = Constraint(model.TD, rule=set_vent_on_rule)

# Prevents simultaneous ventilation start and stop commands in the same time step.
model.on_off_limit = Constraint(model.TD, rule=on_off_limit_rule)

# Allows a shut-down command only when ventilation is currently ON.
model.off_le_e = Constraint(model.TD, rule=off_le_e_rule)

# Allows a start-up command only when ventilation is currently OFF.
model.on_le_one_minus_e = Constraint(model.TD, rule=on_le_one_minus_e_rule)

# Enforces ventilation inertia by linking current ON state to previous ON state and start-up decision.
model.vent_inertia_rule = Constraint(model.TD, rule=set_vent_inertia_rule)

# Enforces minimum ON-time (up-time) so ventilation stays active for a minimum duration after start.
model.min_vent_on = Constraint(model.TD, rule=min_up_time_ventilation_rule)

# -----------------------------------------------------------------------------
# 9) Select solver and solve optimization problem
# -----------------------------------------------------------------------------
# 1. Select the solver
solver = SolverFactory('gurobi') 

# 2. Solve the model
results = solver.solve(model, tee=False)

# Export LP model for inspection/debugging.
model.write("model.lp", io_options={'symbolic_solver_labels': True})

# -----------------------------------------------------------------------------
# 10) Check solver termination condition and report objective
# -----------------------------------------------------------------------------
from pyomo.opt import TerminationCondition

if results.solver.termination_condition == TerminationCondition.optimal:
    print(value(model.obj))

elif results.solver.termination_condition == TerminationCondition.infeasible:
    print("Failure: The model is infeasible (the constraints contradict each other).")
else:
    print(f"Solver Status: {results.solver.termination_condition}")

# -----------------------------------------------------------------------------
# 11) Visualize optimization results
# -----------------------------------------------------------------------------
plot_HVAC_results(model)
plot_all_days_sequential(model)