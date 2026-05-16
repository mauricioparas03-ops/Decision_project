from EnvFunctions import apply_dynamics, check_feasibility
from policies.dummy_policy import select_action as dummy_action
#Import your policy here:
#from policies.dummy_policy import select_action
# from policies.lookahead_policy import select_action
#from policies.SP_policy import select_action
from policies.multSP_policy import select_action

from Data.v2_SystemCharacteristics import get_fixed_data
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from v2_Checks import check_and_sanitize_action

data_dir = Path(__file__).resolve().parent / "Data"

occ1 = pd.read_csv(data_dir / "OccupancyRoom1.csv").values.flatten()
occ2 = pd.read_csv(data_dir / "OccupancyRoom2.csv").values.flatten()

df_prices = pd.read_csv(data_dir / "v2_PriceData.csv") # skip header, as we will access the columns by index

# extract the first column (previous price) as a separate array, if needed for the dynamics or the policy.
daily_previous_prices = df_prices.iloc[:, 0].values 
# extract the columns from 1 to 10: 100x10 matrix of hourly prices, flattened to 1000 elements
prices = df_prices.iloc[:, 1:11].values.flatten()

data = get_fixed_data()
power_max = {1: data['heating_max_power'], 2: data['heating_max_power']}

def init_history():
    return {
        "T1": [],
        "T2": [],
        "H": [],
        "Heat1": [],
        "Heat2": [],
        "Vent": [],
        "Price": [],
        "Occ1": [],
        "Occ2": [],
        "Cost": [],
        "Cumulative_Cost": [],
    }


def plot_average_profile(history, T_hours, title_prefix):
    total_hours = len(history["Cost"])
    if total_hours == 0:
        return

    n_days = total_hours // T_hours
    if n_days == 0:
        return

    day_axis = np.arange(1, n_days + 1)

    def reshape_metric(name):
        values = np.asarray(history[name], dtype=float)
        return values[: n_days * T_hours].reshape(n_days, T_hours)

    daily_t1 = reshape_metric("T1").mean(axis=1)
    daily_t2 = reshape_metric("T2").mean(axis=1)
    daily_h = reshape_metric("H").mean(axis=1)
    daily_heat1 = reshape_metric("Heat1").mean(axis=1)
    daily_heat2 = reshape_metric("Heat2").mean(axis=1)
    daily_vent = reshape_metric("Vent").mean(axis=1)
    daily_price = reshape_metric("Price").mean(axis=1)
    daily_occ1 = reshape_metric("Occ1").mean(axis=1)
    daily_occ2 = reshape_metric("Occ2").mean(axis=1)
    daily_cost = reshape_metric("Cost").sum(axis=1)
    daily_cumulative_cost = reshape_metric("Cumulative_Cost").max(axis=1)

    fig, axs = plt.subplots(4, 2, figsize=(16, 14), sharex=True)
    fig.suptitle(f"{title_prefix}: Daily Averages over {n_days} Days", fontsize=16, fontweight='bold')

    plots = [
        (axs[0, 0], daily_heat1, daily_heat2, "Heating Power", "kW", "Heat1", "Heat2"),
        (axs[0, 1], daily_t1, daily_t2, "Temperature", "°C", "T1", "T2"),
        (axs[1, 0], daily_vent, None, "Ventilation", "Average On Fraction", "Vent", None),
        (axs[1, 1], daily_price, None, "Electricity Price", "Price", "Price", None),
        (axs[2, 0], daily_h, None, "Humidity", "Humidity", "Humidity", None),
        (axs[2, 1], daily_occ1, daily_occ2, "Occupancy", "Occupancy", "Occ1", "Occ2"),
        (axs[3, 0], daily_cost, None, "Overall Cost", "Cost", "Daily Cost", None),
        (axs[3, 1], daily_cumulative_cost, None, "End-of-Day Cumulative Cost", "Cost", "Cumulative Cost", None),
    ]

    for ax, series_a, series_b, title, ylabel, label_a, label_b in plots:
        ax.plot(day_axis, series_a, label=label_a, color='tab:blue', marker='o')
        if series_b is not None:
            ax.plot(day_axis, series_b, label=label_b, color='tab:orange', marker='s')
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    for ax in axs[-1, :]:
        ax.set_xlabel("Day")

    plt.tight_layout()
    plt.show()


def plot_day_profile(history, day_to_plot, T_hours, title_prefix):
    total_hours = len(history["Cost"])
    if total_hours == 0:
        return

    n_days = total_hours // T_hours
    if n_days == 0:
        return

    day_index = max(0, min(day_to_plot, n_days - 1))
    day_label = f"Day {day_index + 1}"
    hour_axis = np.arange(T_hours)

    day_slice = slice(day_index * T_hours, (day_index + 1) * T_hours)
    day_t1 = np.asarray(history["T1"][day_slice], dtype=float)
    day_t2 = np.asarray(history["T2"][day_slice], dtype=float)
    day_h = np.asarray(history["H"][day_slice], dtype=float)
    day_heat1 = np.asarray(history["Heat1"][day_slice], dtype=float)
    day_heat2 = np.asarray(history["Heat2"][day_slice], dtype=float)
    day_vent = np.asarray(history["Vent"][day_slice], dtype=float)
    day_price = np.asarray(history["Price"][day_slice], dtype=float)
    day_occ1 = np.asarray(history["Occ1"][day_slice], dtype=float)
    day_occ2 = np.asarray(history["Occ2"][day_slice], dtype=float)
    day_cost = np.asarray(history["Cost"][day_slice], dtype=float)
    day_cumulative_cost = np.asarray(history["Cumulative_Cost"][day_slice], dtype=float)

    fig, axs = plt.subplots(4, 2, figsize=(16, 14), sharex=True)
    fig.suptitle(f"{title_prefix}: {day_label} Profile", fontsize=16, fontweight='bold')

    plots = [
        (axs[0, 0], day_heat1, day_heat2, "Heating Power", "kW", "Heat1", "Heat2"),
        (axs[0, 1], day_t1, day_t2, "Temperature", "°C", "T1", "T2"),
        (axs[1, 0], day_vent, None, "Ventilation", "On / Off", "Vent", None),
        (axs[1, 1], day_price, None, "Electricity Price", "Price", "Price", None),
        (axs[2, 0], day_h, None, "Humidity", "Humidity", "Humidity", None),
        (axs[2, 1], day_occ1, day_occ2, "Occupancy", "Occupancy", "Occ1", "Occ2"),
        (axs[3, 0], day_cost, None, "Hourly Cost", "Cost", "Cost", None),
        (axs[3, 1], day_cumulative_cost, None, "Cumulative Cost", "Cost", "Cumulative Cost", None),
    ]

    for ax, series_a, series_b, title, ylabel, label_a, label_b in plots:
        ax.plot(hour_axis, series_a, label=f"{day_label} {label_a}", color='black', linestyle='--', marker='x')
        if series_b is not None:
            ax.plot(hour_axis, series_b, label=f"{day_label} {label_b}", color='gray', linestyle=':', marker='^')
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    for ax in axs[-1, :]:
        ax.set_xlabel("Hour of the Day")

    plt.tight_layout()
    plt.show()

E_days = 100
T_hours = 10
daily_costs = np.zeros(E_days)
history = init_history()

print(f"Running simulation for {E_days} days...", flush=True)

for day in range(E_days):
    # extracting the exact datas for the current day from the full trajectories
    day_occ1 = occ1[day * T_hours : (day + 1) * T_hours]
    day_occ2 = occ2[day * T_hours : (day + 1) * T_hours]
    day_prices = prices[day * T_hours : (day + 1) * T_hours]
    day_prev_price = daily_previous_prices[day] #first hour of the day, to be used as price_previous in the initial state

    state = {
        "T1": data['T1'], #Temperature of room 1
        "T2": data['T2'], #Temperature of room 2
        "H": data['H'], #Humidity
        "Occ1": day_occ1[0], #Occupancy of room 1
        "Occ2": day_occ2[0], #Occupancy of room 2
        "price_t": day_prices[0], #Price
        "price_previous": day_prev_price, #Previous Price
        "vent_counter": data['vent_counter'], #For how many consecutive hours has the ventilation been on
        "low_override_r1": data['low_override_r1'], #Is the low-temperature overrule controller of room 1 active
        "low_override_r2": data['low_override_r2'], #Is the low-temperature overrule controller of room 2 active
        "current_time": 0 #What is the hour of the day
    }



    cost_of_this_day = 0.0

    for t in range(T_hours):
        print(f"Day {day}, Time {t}: Current state: {state}", flush=True)
        # DECISION (Here-and-now)
        decision = select_action(state)

        # VERIFY FEASIBILITY
        #decision = action = check_and_sanitize_action(select_action, state, power_max)
        is_feasible = check_feasibility(decision, power_max)
        if not is_feasible:
            print(f"Day {day}, Time {t}: Infeasible! Using dummy.")
            decision = dummy_action(state)

        history["Heat1"].append(float(decision["HeatPowerRoom1"]))
        history["Heat2"].append(float(decision["HeatPowerRoom2"]))
        history["Vent"].append(float(decision["VentilationON"]))
        history["T1"].append(float(state["T1"]))
        history["T2"].append(float(state["T2"]))
        history["H"].append(float(state["H"]))
        history["Price"].append(float(state["price_t"]))
        history["Occ1"].append(float(state["Occ1"]))
        history["Occ2"].append(float(state["Occ2"]))

        # COST AND DYNAMICS
        # cost after overrules; pass the day's exogenous arrays so the
        # environment values from the CSV are used for the "real" next state
        state, real_cost = apply_dynamics(state, decision, data,
                          day_occ1=day_occ1,
                          day_occ2=day_occ2,
                          day_prices=day_prices)
        cost_of_this_day += real_cost
        history["Cost"].append(float(real_cost))
        history["Cumulative_Cost"].append(float(cost_of_this_day))
    
    # 6. TRANSITION (exogenous - Uncertainty "Real" revealed by the historical CSV data)
        if t + 1 < T_hours:
            state['Occ1'] = day_occ1[t + 1]
            state['Occ2'] = day_occ2[t + 1]
            state['price_previous'] = state['price_t']
            state['price_t'] = day_prices[t + 1]
            # current_time already updated in apply_dynamics, so it will automatically move to the next hour in the next iteration
        print(f"Day {day}, Time {t}: Decision taken: {decision}", flush=True)
    # Save the total cost of this day
    daily_costs[day] = cost_of_this_day

    if (day + 1) % 10 == 0:
        print(f"Completed day {day + 1}/{E_days}", flush=True)

print(f"Cost average over {E_days} days: {np.mean(daily_costs):.2f}", flush=True)
plot_average_profile(history, T_hours=T_hours, title_prefix="Policy-selected Environment")
plot_day_profile(history, day_to_plot=42, T_hours=T_hours, title_prefix="Policy-selected Environment")