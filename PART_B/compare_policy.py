
"""compare policies"""

import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# 1. IMPORTS: Environment & Policies
# =============================================================================
from EnvFunctions import apply_dynamics, check_feasibility
from Data.v2_SystemCharacteristics import get_fixed_data

# --- Import Policy A (Baseline) ---
from policies.lookahead_policy import select_action as policy_A_action

# --- Import Policy B (Challenger) ---
from policies.multiSP_policy import select_action as policy_B_action 

# Fallback dummy action for infeasible decisions
from policies.dummy_policy import select_action as dummy_action

# =============================================================================
# 2. HELPER FUNCTIONS: Tracking & Plotting
# =============================================================================
def init_tracker():
    """Initializes a dictionary to store the time-series history of a policy."""
    return {
        'T1': [], 'T2': [], 'H': [], 
        'Heat1': [], 'Heat2': [], 'Vent': [], 
        'Cost': [], 'Cumulative_Cost': [],
        'Price': [], 'Occ1': [], 'Occ2': []
    }

def plot_policy_comparison(day_to_plot, T_hours, hist_A, hist_B, name_A="Policy A", name_B="Policy B"):
    """Generates a 4-panel diagnostic plot comparing two policies for a specific day."""
    start_idx = day_to_plot * T_hours
    end_idx = start_idx + T_hours
    time_steps = np.arange(T_hours)

    fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    fig.suptitle(f"Policy Diagnostic: Day {day_to_plot}", fontsize=16, fontweight='bold')

    # --- Plot 1: Temperatures ---
    axs[0].plot(time_steps, hist_A['T1'][start_idx:end_idx], label=f'{name_A} (T1)', color='blue', marker='o')
    axs[0].plot(time_steps, hist_B['T1'][start_idx:end_idx], label=f'{name_B} (T1)', color='red', marker='x', linestyle='--')
    axs[0].axhline(y=20, color='gray', linestyle=':', label='Min Temp (20C)')
    axs[0].set_ylabel('Temperature (°C)')
    axs[0].set_title('Room 1 Temperature Dynamics')
    axs[0].legend(loc='upper right')
    axs[0].grid(True, alpha=0.3)

    # --- Plot 2: Actions (Heating) ---
    axs[1].step(time_steps, hist_A['Heat1'][start_idx:end_idx], label=f'{name_A} (Heat1)', color='blue', where='mid')
    axs[1].step(time_steps, hist_B['Heat1'][start_idx:end_idx], label=f'{name_B} (Heat1)', color='red', linestyle='--', where='mid')
    axs[1].set_ylabel('Power (kW)')
    axs[1].set_title('Heating Decisions (Room 1)')
    axs[1].legend(loc='upper right')
    axs[1].grid(True, alpha=0.3)

    # --- Plot 3: Environment (Price & Occupancy) ---
    axs[2].plot(time_steps, hist_A['Price'][start_idx:end_idx], label='Price', color='green', drawstyle='steps-mid')
    axs[2].fill_between(time_steps, 0, hist_A['Occ1'][start_idx:end_idx], color='purple', alpha=0.2, label='Occupancy', step='mid')
    axs[2].set_ylabel('Price / Occupancy')
    axs[2].set_title('Exogenous Factors (Price & Occ)')
    axs[2].legend(loc='upper right')
    axs[2].grid(True, alpha=0.3)

    # --- Plot 4: Cumulative Cost ---
    axs[3].plot(time_steps, hist_A['Cumulative_Cost'][start_idx:end_idx], label=f'{name_A} Cost', color='blue')
    axs[3].plot(time_steps, hist_B['Cumulative_Cost'][start_idx:end_idx], label=f'{name_B} Cost', color='red', linestyle='--')
    axs[3].set_ylabel('Cumulative Cost')
    axs[3].set_xlabel('Hour of the Day')
    axs[3].set_title('Financial Impact')
    axs[3].legend(loc='upper left')
    axs[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# =============================================================================
# 3. MAIN SIMULATION EXECUTION
# =============================================================================
if __name__ == "__main__":
    # --- Load Data ---
    data_dir = Path(__file__).resolve().parent / "Data"
    occ1 = pd.read_csv(data_dir / "OccupancyRoom1.csv", header=None).values.flatten()
    occ2 = pd.read_csv(data_dir / "OccupancyRoom2.csv", header=None).values.flatten()
    df_prices = pd.read_csv(data_dir / "v2_PriceData.csv") 

    daily_previous_prices = df_prices.iloc[:, 0].values 
    prices = df_prices.iloc[:, 1:11].values.flatten()
    data = get_fixed_data()
    power_max = {1: data['heating_max_power'], 2: data['heating_max_power']}

    # --- Simulation Parameters ---
    E_days = 100
    T_hours = 10
    
    total_cost_A = np.zeros(E_days)
    total_cost_B = np.zeros(E_days)

    history_A = init_tracker()
    history_B = init_tracker()

    print("Starting parallel policy simulation...")

    for day in range(E_days):
        # Extract day-specific data
        day_occ1 = occ1[day * T_hours : (day + 1) * T_hours]
        day_occ2 = occ2[day * T_hours : (day + 1) * T_hours]
        day_prices = prices[day * T_hours : (day + 1) * T_hours]
        day_prev_price = daily_previous_prices[day] 

        # Initial baseline state for Hour 0
        base_state = {
            "T1": data['T1'], "T2": data['T2'], "H": data['H'], 
            "Occ1": day_occ1[0], "Occ2": day_occ2[0], 
            "price_t": day_prices[0], "price_previous": day_prev_price, 
            "vent_counter": data['vent_counter'], 
            "low_override_r1": data['low_override_r1'], 
            "low_override_r2": data['low_override_r2'], 
            "current_time": 0 
        }

        # Create distinct physical environments for both policies
        state_A = copy.deepcopy(base_state)
        state_B = copy.deepcopy(base_state)

        cum_cost_A = 0.0
        cum_cost_B = 0.0

        for t in range(T_hours):
            # ---------------------------------------------------------
            # POLICY A EXECUTION
            # ---------------------------------------------------------
            dec_A = policy_A_action(state_A)
            if not check_feasibility(dec_A, power_max): 
                print(f"Policy A: Day {day}, Time {t}: Infeasible! Using dummy.")
                dec_A = dummy_action(state_A)
            
            # Log Policy A state and decisions
            history_A['Heat1'].append(dec_A['HeatPowerRoom1'])
            history_A['Heat2'].append(dec_A['HeatPowerRoom2'])
            history_A['Vent'].append(dec_A['VentilationON'])
            history_A['T1'].append(state_A['T1'])
            history_A['T2'].append(state_A['T2'])
            history_A['Price'].append(state_A['price_t'])
            history_A['Occ1'].append(state_A['Occ1'])
            
            # Apply Dynamics A
            state_A, cost_A = apply_dynamics(state_A, dec_A, data)
            cum_cost_A += cost_A
            history_A['Cost'].append(cost_A)
            history_A['Cumulative_Cost'].append(cum_cost_A)

            # ---------------------------------------------------------
            # POLICY B EXECUTION
            # ---------------------------------------------------------
            dec_B = policy_B_action(state_B)
            if not check_feasibility(dec_B, power_max): 
                print(f"Policy B: Day {day}, Time {t}: Infeasible! Using dummy.")
                dec_B = dummy_action(state_B)
            
            # Log Policy B state and decisions
            history_B['Heat1'].append(dec_B['HeatPowerRoom1'])
            history_B['Heat2'].append(dec_B['HeatPowerRoom2'])
            history_B['Vent'].append(dec_B['VentilationON'])
            history_B['T1'].append(state_B['T1'])
            history_B['T2'].append(state_B['T2'])
            history_B['Price'].append(state_B['price_t']) 
            history_B['Occ1'].append(state_B['Occ1'])

            # Apply Dynamics B
            state_B, cost_B = apply_dynamics(state_B, dec_B, data)
            cum_cost_B += cost_B
            history_B['Cost'].append(cost_B)
            history_B['Cumulative_Cost'].append(cum_cost_B)

            # ---------------------------------------------------------
            # EXOGENOUS UPDATES (Same for both environments)
            # ---------------------------------------------------------
            if t + 1 < T_hours:
                for s in [state_A, state_B]:
                    s['Occ1'] = day_occ1[t + 1]
                    s['Occ2'] = day_occ2[t + 1]
                    s['price_previous'] = s['price_t']
                    s['price_t'] = day_prices[t + 1]
                    # current_time is updated automatically in apply_dynamics

        # Store end-of-day costs
        total_cost_A[day] = cum_cost_A
        total_cost_B[day] = cum_cost_B

    # --- Final Results & Plotting ---
    print("\n--- Simulation Complete ---")
    print(f"Policy A Average Cost: {np.mean(total_cost_A):.2f}")
    print(f"Policy B Average Cost: {np.mean(total_cost_B):.2f}")

    # To visualize a specific day (e.g., Day 42), uncomment the line below:
    plot_policy_comparison(42, T_hours, history_A, history_B, name_A="Lookahead Policy", name_B="Multi-Stage Policy")