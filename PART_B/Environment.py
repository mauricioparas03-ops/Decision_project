from EnvFunctions import apply_dynamics, check_feasibility
from policies.dummy_policy import select_action as dummy_action
#from policies.dummy_policy import select_action
#from policies.lookahead_policy import select_action
#from policies.SP_policy import select_action
#from policies.ADP_policy import select_action
#from policies.multiSP import select_action
from policies.Hybrid_multi_and_adp import select_action

from Data.v2_SystemCharacteristics import get_fixed_data
import pandas as pd
from pathlib import Path
import numpy as np
from v2_Checks import check_and_sanitize_action

policy_name = "Hybrid_multi_and_adp" #choose from : "dummy", "lookahead", "SP", "multiSP1", "Hybrid_multi_and_adp", "ADP"


data_dir    = Path(__file__).resolve().parent / "Data"
outputs_dir = Path(__file__).resolve().parent / "outputs"
outputs_dir.mkdir(exist_ok=True)

occ1 = pd.read_csv(data_dir / "OccupancyRoom1.csv").values.flatten()
occ2 = pd.read_csv(data_dir / "OccupancyRoom2.csv").values.flatten()

df_prices = pd.read_csv(data_dir / "v2_PriceData.csv")
daily_previous_prices = df_prices.iloc[:, 0].values
prices = df_prices.iloc[:, 1:11].values.flatten()

data      = get_fixed_data()
power_max = {1: data['heating_max_power'], 2: data['heating_max_power']}

E_days  = 100
T_hours = 10

daily_costs         = np.zeros(E_days)
daily_vent_hours    = np.zeros(E_days)
daily_overrule_acts = np.zeros(E_days)
daily_energy_kwh    = np.zeros(E_days)
daily_pw_cost       = np.zeros(E_days)

print(f"Running simulation for {E_days} days...", flush=True)

for day in range(E_days):
    day_occ1       = occ1  [day * T_hours : (day + 1) * T_hours]
    day_occ2       = occ2  [day * T_hours : (day + 1) * T_hours]
    day_prices     = prices[day * T_hours : (day + 1) * T_hours]
    day_prev_price = daily_previous_prices[day]

    # Initial override state consistent with MDP hysteresis rule (uses <=)
    init_t1 = float(data['T1'])
    init_t2 = float(data['T2'])
    temp_min = data['temp_min_comfort_threshold']

    init_override_r1 = int(init_t1 <= temp_min)
    init_override_r2 = int(init_t2 <= temp_min)

    state = {
        "T1":              init_t1,
        "T2":              init_t2,
        "H":               float(data['H']),
        "Occ1":            day_occ1[0],
        "Occ2":            day_occ2[0],
        "price_t":         day_prices[0],
        "price_previous":  day_prev_price,
        "vent_counter":    int(data['vent_counter']),
        "low_override_r1": init_override_r1,
        "low_override_r2": init_override_r2,
        "current_time":    0,
    }

    cost_of_this_day = 0.0

    for t in range(T_hours):
        # ── Decision ─────────────────────────────────────────────────────────
        print(f"Day {day}, Time {t}: Temperatures = ({state['T1']:.1f}°C, {state['T2']:.1f}°C) | Humidity = {state['H']:.1f}%")
        decision = check_and_sanitize_action(select_action, state, power_max)
        print(f"Day {day}, Time {t}: heater1 = {decision.get('HeatPowerRoom1', 0):.1f}kW, heater2 = {decision.get('HeatPowerRoom2', 0):.1f}kW, ventilation = {'ON' if decision.get('VentilationON', 0) else 'OFF'}", flush=True)
        is_feasible = check_feasibility(decision, power_max)
        if not is_feasible:
            print(f"Day {day}, Time {t}: Infeasible! Using dummy.")
            decision = dummy_action(state)
            fallback_count += 1
        # ── Dynamics + cost ───────────────────────────────────────────────────
        state, real_cost = apply_dynamics(state, decision, data)
        cost_of_this_day += real_cost

        # ── Per-timestep logging ──────────────────────────────────────────────
        vent_was_on = int(state['vent_counter'] > 0)
        p1          = decision.get('HeatPowerRoom1', 0)
        p2          = decision.get('HeatPowerRoom2', 0)
        step_energy = p1 + p2 + vent_was_on * data['ventilation_power']

        overrule_fired = int(
            state['low_override_r1']
            or state['low_override_r2']
            or (state['H'] > data['humidity_threshold'])
            or (0 < state['vent_counter'] < data['vent_min_up_time'])
        )

        daily_vent_hours   [day] += vent_was_on
        daily_overrule_acts[day] += overrule_fired
        daily_energy_kwh   [day] += step_energy

        # ── Exogenous transition ──────────────────────────────────────────────
        if t + 1 < T_hours:
            state['Occ1']          = day_occ1  [t + 1]
            state['Occ2']          = day_occ2  [t + 1]
            state['price_previous'] = state['price_t']
            state['price_t']       = day_prices[t + 1]
       

    daily_costs  [day] = cost_of_this_day
    daily_pw_cost[day] = (cost_of_this_day / daily_energy_kwh[day]
                          if daily_energy_kwh[day] > 0 else 0.0)

    running_mean = np.mean(daily_costs[:day + 1])
    print(
        f"Day {day + 1:>3}: "
        f"daily cost = {cost_of_this_day:>7.2f}"
        f" | running average = {running_mean:>7.2f}",
        flush=True,
    )

    if (day + 1) % 10 == 0:
        print(f"Completed day {day + 1}/{E_days}", flush=True)

print(f"\nCost average over {E_days} days: {np.mean(daily_costs):.2f}", flush=True)


np.savez(
    outputs_dir / f"results_{policy_name}.npz",
    daily_costs         = daily_costs,
    daily_vent_hours    = daily_vent_hours,
    daily_overrule_acts = daily_overrule_acts,
    daily_energy_kwh    = daily_energy_kwh,
    daily_pw_cost       = daily_pw_cost,
)
print(f"Results saved to results_{policy_name}.npz", flush=True)