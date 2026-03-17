from pyomo.environ import *


class DummyPolicy:

    def select_action(self, state):
        """
        Dummy policy: never turns on the ventilation nor any heater actively.
        Everything is left up to the overrule controllers.
        """
        HereAndNowActions = {
            "p1": 0.0,  # no heating room 1
            "p2": 0.0,  # no heating room 2
            "v": 0      # no ventilation
        }
        
        return HereAndNowActions
    


class OptimalInHindsightPolicy:
    def __init__(self, data_fixed, daily_prices, daily_occ_r1, daily_occ_r2):
        """
        Initializes the policy with perfect foresight data for the current day.
        daily_prices, daily_occ_r1, daily_occ_r2 are expected to be lists/arrays of 10 elements.
        """
        self.data_fixed = data_fixed
        self.prices = daily_prices
        self.occ_r1 = daily_occ_r1
        self.occ_r2 = daily_occ_r2
        
        # Prepare memory to store the optimal actions for the 10 hours
        self.p1_opt = [0.0] * 10
        self.p2_opt = [0.0] * 10
        self.v_opt = [0] * 10
        
        # ensure Gurobi is called only once per day
        self.is_solved = False

    def solve_daily_milp(self):
        model = ConcreteModel()
        
        # SETS 
        model.R = Set(initialize=[1, 2]) # Rooms
        model.T = Set(initialize=range(self.data_fixed['num_timeslots'])) #Time steps
        model.RT = model.R * model.T # Time-Day combinations


        # PARAMETERS
        # Exogenous data for the current day (loaded from the instance attributes)
        model.Prices = Param(model.T, initialize=lambda m, t: self.prices[t])
        model.Occ1 = Param(model.T, initialize=lambda m, t: self.occ_r1[t])
        model.Occ2 = Param(model.T, initialize=lambda m, t: self.occ_r2[t])
        model.Tout = Param(model.T, initialize=lambda m, t: self.data_fixed['outdoor_temperature'][t])
        
        # Physical system characteristics and limits
        model.Pr = Param(initialize=self.data_fixed['heating_max_power'])
        model.Pvent = Param(initialize=self.data_fixed['ventilation_power'])
        
        # Thermal and humidity dynamics coefficients
        model.Zexch = Param(initialize=self.data_fixed['heat_exchange_coeff'])
        model.Zconv = Param(initialize=self.data_fixed['heating_efficiency_coeff'])
        model.Zloss = Param(initialize=self.data_fixed['thermal_loss_coeff'])
        model.Zcool = Param(initialize=self.data_fixed['heat_vent_coeff'])
        model.Zocc = Param(initialize=self.data_fixed['heat_occupancy_coeff'])
        
        model.Hocc = Param(initialize=self.data_fixed['humidity_occupancy_coeff'])
        model.Hvent = Param(initialize=self.data_fixed['humidity_vent_coeff'])

        # Overrule controller thresholds
        model.Tmin = Param(initialize=self.data_fixed['temp_min_comfort_threshold'])
        model.Tok = Param(initialize=self.data_fixed['temp_OK_threshold'])
        model.Thigh = Param(initialize=self.data_fixed['temp_max_comfort_threshold'])
        model.Hhigh = Param(initialize=self.data_fixed['humidity_threshold'])
        
        # Initial states at t=0
        model.Tinitial = Param(initialize=self.data_fixed['initial_temperature'])
        model.Hinitial = Param(initialize=self.data_fixed['initial_humidity'])
        
        
        # DECISION VARIABLES
        # Action variables (Here-and-now decisions mapped for the whole day)
        model.Vent = Var(model.T, domain=Binary) # Ventilation ON/OFF state
        model.Heat = Var(model.RT, domain=NonNegativeReals, bounds=(0, model.Pr)) # Heating power in kW
        
        # 2. Auxiliary binary variables for overrule logic (from Task 1)
        model.w = Var(model.RT, domain=Binary) # 1 if Temperature > Tok (deactivates low-temp overrule)
        model.u = Var(model.RT, domain=Binary) # 1 if Temperature < Tlow (activates low-temp overrule)
        model.y = Var(model.RT, domain=Binary) # 1 if Temperature > Thigh (forces heater OFF)
        
        # 3. State variables (Physical dynamics)
        model.T_in = Var(model.RT, domain=NonNegativeReals) # Indoor temperature in °C
        model.Hum = Var(model.T, domain=NonNegativeReals)   # Indoor humidity in %

        # OBJECTIVE FUNCTION
        def total_cost_rule(model):
            """
            Calculates the total electricity cost for the current day.
            Cost = Price * (Heating Power + Ventilation Power) for all time slots.
            """
            heat_cost = sum(model.Prices[t] * model.Heat[r, t] for r in model.R for t in model.T)
            vent_cost = sum(model.Prices[t] * model.Vent[t] * model.Pvent for t in model.T)
            return heat_cost + vent_cost
            
        # Minimize the expected cost (which is deterministic in hindsight)
        model.obj = Objective(rule=total_cost_rule, sense=minimize)
        
        
        # CONSTRAINTS
        # Temperature Dynamics
        def room_thermal_balance_rule(model, r, t):
            """
            Enforces indoor temperature evolution each hour from heat exchange, 
            losses, heating, ventilation, and occupancy.
            """
            # Initialize the first time step with the initial temperature value
            if t == model.T.first():
                return model.T_in[r, t] == model.Tinitial
            
            t_prev = t - 1 
            
            # Determine occupancy for the current room
            occ = model.Occ1[t_prev] if r == 1 else model.Occ2[t_prev]
            # Determine the other room for heat exchange
            r_other = 2 if r == 1 else 1
            
            # The Equation: T_now = T_prev + Internal_Exchange + External_Loss + Heating - Cooling + Occupancy
            return model.T_in[r, t] == (
                model.T_in[r, t_prev] 
                + model.Zexch * (model.T_in[r_other, t_prev] - model.T_in[r, t_prev]) 
                + model.Zloss * (model.Tout[t_prev] - model.T_in[r, t_prev])
                + model.Zconv * model.Heat[r, t_prev] 
                - model.Zcool * model.Vent[t_prev]
                + model.Zocc * occ
            )
        model.Temp_Room_Dynamics = Constraint(model.RT, rule=room_thermal_balance_rule)

        # Humidity Dynamics
        def humidity_balance_rule(model, t):
            """
            Enforces indoor humidity evolution each hour from ventilation removal 
            and occupancy generation.
            """
            if t == model.T.first():
                return model.Hum[t] == model.Hinitial
            
            t_prev = t - 1
            return model.Hum[t] == (
                model.Hum[t_prev]
                - model.Hvent * model.Vent[t_prev]
                + model.Hocc * (model.Occ1[t_prev] + model.Occ2[t_prev])
            )
        model.Hum_Room_Dynamics = Constraint(model.T, rule=humidity_balance_rule)
               
        # Low-Temperature Overrule Controller
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
                return m_ref.u[r, t] >= m_ref.u[r, t-1] - m_ref.w[r, t]
            block.u_w_exclusivity = Constraint(m_ref.RT, rule=u_w_exclusivity_rule)

            def set_max_power_rule(block, r, t):
                return m_ref.Heat[r, t] >= m_ref.Pr * m_ref.u[r, t]
            block.set_max_power = Constraint(m_ref.RT, rule=set_max_power_rule)
            
        model.set_max_temp_rule = Block(rule=max_temp_low_rule)

        # High-Temperature Safety Logic
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

        # Humidity-Triggered Ventilation
        def set_vent_on_rule(model, t):
            M = 100
            return model.Hum[t] <= model.Hhigh + M * model.Vent[t]
        model.vent_on_rule = Constraint(model.T, rule=set_vent_on_rule)

        # Minimum Up-Time for Ventilation
        def min_up_time_ventilation_rule(model, t):
            L = 3 
            remaining_steps = [k for k in range(t, t + L) if k <= model.T.last()]
            v_prev = model.Vent[t-1] if t > model.T.first() else 0
            return sum(model.Vent[k] for k in remaining_steps) >= len(remaining_steps) * (model.Vent[t] - v_prev)
        model.min_vent_on = Constraint(model.T, rule=min_up_time_ventilation_rule)

        # SOLVER EXECUTION & RESULT EXTRACTION
        solver = SolverFactory('gurobi') 
        results = solver.solve(model, tee=False)

        # Extract optimal actions into our pre-allocated memory lists
        for t in model.T:
            self.p1_opt[t] = value(model.Heat[1, t])
            self.p2_opt[t] = value(model.Heat[2, t])
            self.v_opt[t]  = int(round(value(model.Vent[t])))

        self.is_solved = True 


    # INTERFACE WITH THE ENVIRONMENT
    def select_action(self, state):
        """
        This is the decision rule called by the environment at every hour t.
        """
        current_t = state["current_time"]
        
        # If it's the start of the day and we haven't solved yet, do it now!
        if current_t == 0 and not self.is_solved:
            self.solve_daily_milp()
            
        # Return the pre calculated action for the current hour
        HereAndNowActions = {
            "p1": float(self.p1_opt[current_t]),
            "p2": float(self.p2_opt[current_t]),
            "v":  int(self.v_opt[current_t])
        }
        
        return HereAndNowActions