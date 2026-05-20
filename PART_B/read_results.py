import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

policies = ["dummy", "lookahead", "SP", "multiSP", "hindsight", "Hybrid_multi_and_adp"]  # List of policy identifiers to load results for

policy_labels = {
    "dummy":     "Dummy",
    "lookahead": "Det. Lookahead",
    "SP":        "2-Stage SP",
    "multiSP":   "Multi-Stage SP",
    "hindsight": "Opt. Hindsight",
    "ADP":       "ADP",
    "Hybrid_multi_and_adp":    "Hybrid",
}

colors = {
    "dummy":     "#B4B2A9",
    "lookahead": "#85B7EB",
    "SP":        "#378ADD",
    "multiSP":   "#5DCAA5",
    "hindsight": "#639922",
    "ADP":       "#EF9F27",
    "Hybrid_multi_and_adp":    "#D85A30",
}

results = {}
for name in policies:
    data = np.load(f"results_{name}.npz")
    results[name] = {key: data[key] for key in data.files}
    print(f"\n{'='*55}")
    print(f"Summary for: {name}")
    print(f"{'='*55}")
    print(f"  Avg daily cost         : {np.mean(results[name]['daily_costs']):.4f}  ± {np.std(results[name]['daily_costs']):.4f}")
    print(f"  Avg vent hours/day     : {np.mean(results[name]['daily_vent_hours']):.2f}   ± {np.std(results[name]['daily_vent_hours']):.2f}")
    print(f"  Avg overrule acts/day  : {np.mean(results[name]['daily_overrule_acts']):.2f}   ± {np.std(results[name]['daily_overrule_acts']):.2f}")
    print(f"  Avg energy/day (kWh)   : {np.mean(results[name]['daily_energy_kwh']):.4f}  ± {np.std(results[name]['daily_energy_kwh']):.4f}")
    print(f"  Avg price-wtd cost     : {np.mean(results[name]['daily_pw_cost']):.4f}  ± {np.std(results[name]['daily_pw_cost']):.4f}")

# ── Sort policies by average daily cost ───────────────────────────────────────
sorted_policies = sorted(policies, key=lambda p: np.mean(results[p]['daily_costs']))
labels    = [policy_labels[p] for p in sorted_policies]
means     = [np.mean(results[p]['daily_costs']) for p in sorted_policies]
stds      = [np.std(results[p]['daily_costs'])  for p in sorted_policies]
bar_colors = [colors[p] for p in sorted_policies]

# =============================================================================
# Figure 1: Bar chart — average daily cost with std error bars
# =============================================================================
fig1, ax1 = plt.subplots(figsize=(9, 5))

bars = ax1.bar(labels, means, yerr=stds, capsize=5,
               color=bar_colors, edgecolor='white', linewidth=0.8,
               error_kw=dict(elinewidth=1.2, ecolor='#444441'))

ax1.set_ylabel("Average Daily Cost", fontsize=11)
ax1.set_title("Average Daily Electricity Cost per Policy (100 experiments)", fontsize=12)
ax1.set_ylim(0, max(means) * 1.45)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.tick_params(axis='x', labelsize=10)

# Value labels on bars
for bar, mean, std in zip(bars, means, stds):
    ax1.text(bar.get_x() + bar.get_width() / 2,
             mean / 2,  # center of bar
             f"{mean:.1f}", ha='center', va='center', fontsize=9, 
             color='white', fontweight='bold')


plt.tight_layout()
plt.savefig("fig_avg_cost_comparison.pdf", bbox_inches='tight')
plt.savefig("fig_avg_cost_comparison.png", dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# Figure 2: Box plots — daily cost distributions
# =============================================================================
fig2, ax2 = plt.subplots(figsize=(9, 5))

box_data = [results[p]['daily_costs'] for p in sorted_policies]

bp = ax2.boxplot(box_data, patch_artist=True, notch=False,
                 medianprops=dict(color='white', linewidth=2),
                 whiskerprops=dict(linewidth=1.2),
                 capprops=dict(linewidth=1.2),
                 flierprops=dict(marker='o', markersize=4, linestyle='none', alpha=0.5))

for patch, p in zip(bp['boxes'], sorted_policies):
    patch.set_facecolor(colors[p])
    patch.set_alpha(0.85)

for flier, p in zip(bp['fliers'], sorted_policies):
    flier.set_markerfacecolor(colors[p])
    flier.set_markeredgecolor(colors[p])

ax2.set_xticklabels(labels, fontsize=10)
ax2.set_ylabel("Daily Cost", fontsize=11)
ax2.set_title("Daily Cost Distribution per Policy (100 experiments)", fontsize=12)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("fig_cost_distribution.pdf", bbox_inches='tight')
plt.savefig("fig_cost_distribution.png", dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# Figure 3: Summary table — all metrics
# =============================================================================
metric_keys   = ['daily_costs', 'daily_vent_hours', 'daily_overrule_acts',
                 'daily_energy_kwh', 'daily_pw_cost']
metric_labels = ['Avg Cost', 'Vent Hours/day', 'Overrule Acts/day',
                 'Energy/day (kWh)', 'Price-wtd Cost']

fig3, ax3 = plt.subplots(figsize=(11, 3))
ax3.axis('off')

table_data = []
for p in sorted_policies:
    row = [policy_labels[p]]
    for key in metric_keys:
        m = np.mean(results[p][key])
        s = np.std(results[p][key])
        row.append(f"{m:.2f} ± {s:.2f}")
    table_data.append(row)

col_labels = ['Policy'] + metric_labels
table = ax3.table(cellText=table_data, colLabels=col_labels,
                  cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.6)

# Header styling
for j in range(len(col_labels)):
    table[0, j].set_facecolor('#3C3489')
    table[0, j].set_text_props(color='white', fontweight='bold')

# Row styling
for i, p in enumerate(sorted_policies, start=1):
    for j in range(len(col_labels)):
        table[i, j].set_facecolor(colors[p] + '33')  # light tint

plt.title("Policy Comparison — All Metrics (100 experiments)",
          fontsize=11, pad=12)
plt.tight_layout()
plt.savefig("fig_metrics_table.pdf", bbox_inches='tight')
plt.savefig("fig_metrics_table.png", dpi=150, bbox_inches='tight')
plt.show()