def plot_HVAC_results(model):
    import matplotlib.pyplot as plt
    from pyomo.environ import value

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
        "Occ_r1": [value(model.Occ1[d, t]) for t in T],
        "Occ_r2": [value(model.Occ2[d, t]) for t in T],
    }
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

def plot_all_days_sequential(model):
    import matplotlib.pyplot as plt
    from pyomo.environ import value
        # data estraction for every day
    T_list = list(model.T)
    D_list = list(model.D)
    num_timeslots = len(T_list)
    num_days = len(D_list)

    # concatenate list for every day
    all_days_data = {
        "time": [],
        "Temp_r1": [],
        "Temp_r2": [],
        "h_r1": [],
        "h_r2": [],
        "v": [],
        "Hum": [],
        "price": [],
        "Occ_r1": [],
        "Occ_r2": [],
    }

    # concatenate data for all days
    for d in D_list:
        for t in T_list:
            all_days_data["time"].append(d * num_timeslots + t)
            all_days_data["Temp_r1"].append(value(model.T_in[1, t, d]))
            all_days_data["Temp_r2"].append(value(model.T_in[2, t, d]))
            all_days_data["h_r1"].append(value(model.Heat[1, t, d]))
            all_days_data["h_r2"].append(value(model.Heat[2, t, d]))
            all_days_data["v"].append(value(model.Vent[t, d]))
            all_days_data["Hum"].append(value(model.Hum[t, d]))
            all_days_data["price"].append(value(model.prices[d, t]))
            all_days_data["Occ_r1"].append(value(model.Occ1[d, t]))
            all_days_data["Occ_r2"].append(value(model.Occ2[d, t]))

    """
    Plot all the data for all days sequentially in a single figure with 4 subplots:
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(4, 1, figsize=(20, 14), sharex=True)
    
    time = all_days_data["time"]
    
    # Room Temperatures
    axes[0].plot(time, all_days_data["Temp_r1"], label='Room 1 Temp', alpha=0.7, linewidth=0.8)
    axes[0].plot(time, all_days_data["Temp_r2"], label='Room 2 Temp', alpha=0.7, linewidth=0.8)
    axes[0].axhline(18, color='gray', linestyle='--', alpha=0.5)
    axes[0].axhline(22, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].set_title("Room Temperatures - All Days Sequential")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Heater consumption
    axes[1].fill_between(time, 0, all_days_data["h_r1"], label='Room 1 Heater', alpha=0.6)
    axes[1].fill_between(time, all_days_data["h_r1"], 
                         [h1 + h2 for h1, h2 in zip(all_days_data["h_r1"], all_days_data["h_r2"])], 
                         label='Room 2 Heater', alpha=0.6)
    axes[1].set_ylabel("Heater Power (kW)")
    axes[1].set_title("Heater Consumption")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Ventilation and Humidity
    axes[2].fill_between(time, 0, all_days_data["v"], label='Ventilation ON', 
                         color='tab:blue', alpha=0.3, step='mid')
    axes[2].plot(time, all_days_data["Hum"], label='Humidity (%)', 
                 color='tab:orange', linewidth=0.8)
    axes[2].axhline(45, color='gray', linestyle='--', alpha=0.5)
    axes[2].axhline(70, color='gray', linestyle='--', alpha=0.5)
    axes[2].set_ylabel("Ventilation / Humidity")
    axes[2].set_title("Ventilation Status and Humidity")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Electricity price and total occupancy
    ax3a = axes[3]
    ax3b = ax3a.twinx()
    
    ax3a.plot(time, all_days_data["price"], label='TOU Price (€/kWh)', 
             color='tab:red', linewidth=0.8, alpha=0.7)
    ax3a.set_ylabel("Price (€/kWh)", color='tab:red')
    ax3a.tick_params(axis='y', labelcolor='tab:red')
    
    total_occ = [o1 + o2 for o1, o2 in zip(all_days_data["Occ_r1"], all_days_data["Occ_r2"])]
    ax3b.fill_between(time, 0, total_occ, label='Total Occupancy', 
                     alpha=0.3, color='tab:green')
    ax3b.set_ylabel("Total Occupancy", color='tab:green')
    ax3b.tick_params(axis='y', labelcolor='tab:green')
    
    axes[3].set_xlabel("Time (hours)")
    axes[3].set_title("Electricity Price and Total Occupancy")
    ax3a.grid(True, alpha=0.3)
    
    # Add vertical lines to separate days
    for d in range(1, num_days):
        for ax in axes:
            ax.axvline(d * num_timeslots, color='black', linestyle=':', alpha=0.2, linewidth=0.5)
    
    plt.tight_layout()
    plt.show()