from matplotlib import axes

T = list(model.T)
d = 99

HVAC_results = {
    "T": T,
    "Temp_r1": [value(model.T_in[1, t, d]) for t in T],
    "Temp_r2": [value(model.T_in[2, t, d]) for t in T],
    "h_r1": [value(model.Heat[1, t, d]) for t in T],
    "h_r2": [value(model.Heat[2, t, d]) for t in T],
    "v": [value(model.Vent[t, d]) for t in T],
    "Hum": [value(model.Hum[t, d]) for t in T],
    "price": [value(model.prices[d, t]) for t in T],
    "Occ_r1": [Occ_r1.iloc[d, t] for t in T],
    "Occ_r2": [Occ_r2.iloc[d, t] for t in T],
}

def plot_HVAC_results(HVAC_results):
    import matplotlib.pyplot as plt
    T = HVAC_results["T"]   
    
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)

    ## Temp_r1, Temp_r2, etc are imported from HVAC_results

    # Room Temperatures
    axes[0].plot(T, HVAC_results["Temp_r1"], label='Room 1 Temp', marker='o')
    axes[0].plot(T, HVAC_results["Temp_r2"], label='Room 2 Temp', marker='s')
    axes[0].axhline(18, color='gray', linestyle='--', alpha=0.5)
    axes[0].axhline(20, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].set_title(f"Room Temperatures day {d+1}")
    axes[0].legend()
    axes[0].grid(True)
    
    # Heater consumption
    axes[1].bar(T, HVAC_results["h_r1"], width=0.4, label='Room 1 Heater', alpha=0.7)
    axes[1].bar(T, HVAC_results["h_r2"], width=0.4, bottom=HVAC_results["h_r1"], label='Room 2 Heater', alpha=0.7)
    axes[1].set_ylabel("Heater Power (kW)")
    axes[1].set_title(f"Heater Consumption day {d+1}")
    axes[1].legend()
    axes[1].grid(True)
    
    # Ventilation and Humidity
    ax2 = axes[2]
    ax2_right = ax2.twinx()  # Crea un secondo asse Y sulla destra

    # Humidity left axis
    ax2.plot(T, HVAC_results["Hum"], label='Humidity (%)', color='tab:orange', marker='o', linewidth=2)
    ax2.axhline(45, color='gray', linestyle='--', alpha=0.5)
    ax2.axhline(70, color='gray', linestyle='--', alpha=0.5)
    ax2.set_ylabel("Humidity (%)", color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    # Ventilation right axis
    ax2_right.step(T, HVAC_results["v"], where='mid', label='Ventilation ON', color='tab:blue', linewidth=2)
    ax2_right.set_ylabel("Ventilation Status", color='tab:blue')
    ax2_right.set_ylim(-0.1, 1.1)  
    ax2_right.set_yticks([0, 1])
    ax2_right.set_yticklabels(['OFF', 'ON'])
    ax2_right.tick_params(axis='y', labelcolor='tab:blue')

    ax2.set_title(f"Ventilation Status and Humidity day {d+1}")
    ax2.grid(True)

    # legend combination
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_right.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Electricity price and occupancy
    axes[3].plot(T, HVAC_results["price"], label='TOU Price (€/kWh)', color='tab:red', marker='x')
    axes[3].bar(T, HVAC_results["Occ_r1"], label='Occupancy Room 1', alpha=0.5)
    axes[3].bar(T, HVAC_results["Occ_r2"], bottom=HVAC_results["Occ_r1"], label='Occupancy Room 2', alpha=0.5)
    axes[3].set_ylabel("Price / Occupancy")
    axes[3].set_xlabel("Time (hours)")
    axes[3].set_title(f"Electricity Price and Occupancy day {d+1}")
    axes[3].legend()
    axes[3].grid(True)
    
    plt.tight_layout()
    plt.show()

    plot_HVAC_results(HVAC_results)