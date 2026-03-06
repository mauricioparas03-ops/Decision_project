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
        Inizializza la policy passandole "dal futuro" i dati dell'intero giorno.
        daily_prices, daily_occ_r1, daily_occ_r2 devono essere dizionari o array lunghi 10 (per le ore t=0..9).
        """
        self.data_fixed = data_fixed
        self.prices = daily_prices
        self.occ_r1 = daily_occ_r1
        self.occ_r2 = daily_occ_r2
        
        # Liste per salvare le decisioni ottime
        self.p1_opt = [0.0] * 10
        self.p2_opt = [0.0] * 10
        self.v_opt = [0] * 10
        
        self.is_solved = False

    def solve_daily_milp(self):
        model = ConcreteModel()
        
        # 1. Sets (Senza i giorni!)
        model.R = Set(initialize=[1, 2])
        model.T = Set(initialize=range(self.data_fixed['num_timeslots']))
        model.RT = model.R * model.T

        # 2. Parametri (Adattati per leggere i dati giornalieri passati alla classe)
        model.Prices = Param(model.T, initialize=lambda m, t: self.prices[t])
        model.Occ1 = Param(model.T, initialize=lambda m, t: self.occ_r1[t])
        model.Occ2 = Param(model.T, initialize=lambda m, t: self.occ_r2[t])
        
        # ... [QUI INSERIRAI TUTTI GLI ALTRI PARAMETRI DAL TUO CODICE data_fixed] ...
        model.Pr = Param(initialize=self.data_fixed['heating_max_power'])
        model.Pvent = Param(initialize=self.data_fixed['ventilation_power'])
        # (Aggiungi Zexch, Zconv, ecc. esattamente come nel tuo codice)
        
        # 3. Variabili (Senza l'indice d)
        model.Vent = Var(model.T, domain=Binary)
        model.Heat = Var(model.RT, domain=NonNegativeReals, bounds=(0, model.Pr))
        # (Aggiungi T_in, Hum, w, u, y come nel tuo codice, togliendo model.D)

        # 4. Funzione Obiettivo (Senza ciclo sui giorni)
        def total_cost_rule(model):
            heat_cost = sum(model.Prices[t] * model.Heat[r, t] for r in model.R for t in model.T)
            vent_cost = sum(model.Prices[t] * model.Vent[t] * model.Pvent for t in model.T)
            return heat_cost + vent_cost
        model.obj = Objective(rule=total_cost_rule, sense=minimize)

        # 5. Vincoli
        # Esempio di come si semplifica il bilancio termico (togliendo la 'd'):
        # def room_thermal_balance_rule(model, r, t):
        #     if t == model.T.first(): ...
        # model.Temp_Room_Dynamics = Constraint(model.RT, rule=room_thermal_balance_rule)
        
        # ... [QUI AGGIUNGI TUTTE LE TUE REGOLE PYOMO SNELLITE] ...

        # 6. Risoluzione
        solver = SolverFactory('gurobi')
        solver.solve(model, tee=False)

        # 7. Salvataggio dei risultati nelle liste interne
        for t in model.T:
            self.p1_opt[t] = value(model.Heat[1, t])
            self.p2_opt[t] = value(model.Heat[2, t])
            self.v_opt[t]  = round(value(model.Vent[t])) # Arrotondiamo per sicurezza il binario

        self.is_solved = True

    def select_action(self, state):
        current_t = state["current_time"]
        
        # Il trucco dell'Hindsight: se siamo a t=0, guardiamo nel futuro e risolviamo tutto
        if current_t == 0 and not self.is_solved:
            self.solve_daily_milp()
            
        # Restituiamo l'azione pre-calcolata, usando le chiavi esatte richieste
        return {
            "p1": float(self.p1_opt[current_t]),
            "p2": float(self.p2_opt[current_t]),
            "v":  int(self.v_opt[current_t])
        }