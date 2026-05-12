import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from pyomo.environ import *
from Data.DataTask7 import fetch_data


# ==============================================================================
# SECTION 2: LOCAL STORE OPTIMIZATION
# ==============================================================================

def optimize_store_locally(store_weight, occupancies, data, lambda_multipliers):
    model = ConcreteModel()

    model.T   = Set(initialize=range(data['num_timeslots']))
    model.R   = Set(initialize=[1, 2])
    model.RT  = model.R * model.T
    model.Tp  = Set(initialize=range(data['num_timeslots'] - 1))
    model.RTp = model.R * model.Tp

    model.Occ  = Param(model.RT,
                       initialize=lambda m, r, t: float(occupancies.iloc[r - 1, t]))
    model.Tout = Param(model.T,
                       initialize={t: data['outdoor_temperature'][t] for t in range(data['num_timeslots'])})

    model.Pr      = Param(initialize=data['heating_max_power'])
    model.Zexch   = Param(initialize=data['heat_exchange_coeff'])
    model.Zconv   = Param(initialize=data['heating_efficiency_coeff'])
    model.Zloss   = Param(initialize=data['thermal_loss_coeff'])
    model.Zcool   = Param(initialize=data['heat_vent_coeff'])
    model.Zocc    = Param(initialize=data['heat_occupancy_coeff'])
    model.Tinitial = Param(initialize=data['initial_temperature'])
    model.Tref    = Param(initialize=data['Temperature_reference'])

    model.p   = Var(model.RTp, bounds=(0, model.Pr))
    model.T_in = Var(model.RT)

    def objective_rule(m):
        discomfort = sum(store_weight * (m.T_in[r, t] - m.Tref) ** 2
                         for r in m.R for t in m.T)
        dual_term  = sum(lambda_multipliers[t] * sum(m.p[r, t] for r in m.R)
                         for t in m.Tp)
        return discomfort + dual_term

    model.obj = Objective(rule=objective_rule, sense=minimize)

    def thermal_dynamics_rule(model, r, t):
        if t == 0:
            return model.T_in[r, t] == model.Tinitial
        t_prev  = t - 1
        r_other = 2 if r == 1 else 1
        occ     = model.Occ[r, t_prev]          # FIX: use r, not r_other
        return model.T_in[r, t] == (
            model.T_in[r, t_prev]
            + model.Zexch * (model.T_in[r_other, t_prev] - model.T_in[r, t_prev])
            + model.Zloss * (model.Tout[t_prev]  - model.T_in[r, t_prev])
            + model.Zconv * model.p[r, t_prev]
            - model.Zcool
            + model.Zocc  * occ
        )

    model.Temp_Room_Dynamics = Constraint(model.RT, rule=thermal_dynamics_rule)

    solver = SolverFactory('gurobi')
    solver.solve(model, tee=False)

    p_profile = np.zeros(data['num_timeslots'], dtype=float)
    for t in model.Tp:
        p_profile[t] = sum(value(model.p[r, t], exception=False) or 0.0
                           for r in model.R)
    T_in = np.array([[value(model.T_in[r, t]) for t in model.T]
                     for r in model.R], dtype=float)
    return p_profile, T_in


# ==============================================================================
# SECTION 3: COORDINATOR (DUAL UPDATE)
# ==============================================================================

def coordinator(lambda_prev, alpha_step, data, p_profiles):
    sum_p     = np.sum(p_profiles, axis=0)          # shape (T,)
    violation = sum_p - data['P_mall']              # g_t = Σ p_n,t − P_mall
    lambda_next = np.maximum(0.0, lambda_prev + alpha_step * violation)
    return lambda_next, sum_p, violation


# ==============================================================================
# SECTION 4: CENTRALIZED BENCHMARK
# ==============================================================================

def centralized_optimal_solution(n_stores, store_weights, occupancies, data):
    """Solve the joint problem over ALL stores for ONE day (num_timeslots)."""
    model = ConcreteModel()

    model.T   = Set(initialize=range(data['num_timeslots']))
    model.R   = Set(initialize=[1, 2])
    model.RT  = model.R * model.T
    model.S   = Set(initialize=range(n_stores))
    model.SRT = model.S * model.R * model.T
    model.Tp  = Set(initialize=range(data['num_timeslots'] - 1))
    model.SRTp = model.S * model.R * model.Tp

    model.Occ  = Param(model.SRT,
                       initialize=lambda m, s, r, t: float(occupancies.iloc[r - 1, t]))
    model.Tout = Param(model.T,
                       initialize={t: data['outdoor_temperature'][t]
                                   for t in range(data['num_timeslots'])})

    model.Pr      = Param(initialize=data['heating_max_power'])
    model.Zexch   = Param(initialize=data['heat_exchange_coeff'])
    model.Zconv   = Param(initialize=data['heating_efficiency_coeff'])
    model.Zloss   = Param(initialize=data['thermal_loss_coeff'])
    model.Zcool   = Param(initialize=data['heat_vent_coeff'])
    model.Zocc    = Param(initialize=data['heat_occupancy_coeff'])
    model.Tinitial = Param(initialize=data['initial_temperature'])
    model.Tref    = Param(initialize=data['Temperature_reference'])

    model.p    = Var(model.SRTp, bounds=(0, model.Pr))
    model.T_in = Var(model.SRT)

    def objective_rule(m):
        return sum(store_weights[s] * (m.T_in[s, r, t] - m.Tref) ** 2
                   for s in m.S for r in m.R for t in m.T)

    model.obj = Objective(rule=objective_rule, sense=minimize)

    def thermal_dynamics_rule(model, s, r, t):
        if t == 0:
            return model.T_in[s, r, t] == model.Tinitial
        t_prev  = t - 1
        r_other = 2 if r == 1 else 1
        occ     = model.Occ[s, r, t_prev]          # FIX: use r, not r_other
        return model.T_in[s, r, t] == (
            model.T_in[s, r, t_prev]
            + model.Zexch * (model.T_in[s, r_other, t_prev] - model.T_in[s, r, t_prev])
            + model.Zloss * (model.Tout[t_prev]  - model.T_in[s, r, t_prev])
            + model.Zconv * model.p[s, r, t_prev]
            - model.Zcool
            + model.Zocc  * occ
        )

    model.Temp_Room_Dynamics = Constraint(model.SRT, rule=thermal_dynamics_rule)

    def mall_power_limit_rule(model, t):
        return sum(model.p[s, r, t] for s in model.S for r in model.R) <= data['P_mall']

    model.Mall_Power_Limit = Constraint(model.T, rule=mall_power_limit_rule)

    solver = SolverFactory('gurobi')
    solver.solve(model, tee=False)
    return value(model.obj)


# ==============================================================================
# SECTION 5: DISTRIBUTED ALGORITHM  (fixed OR adaptive step)
# ==============================================================================

def run_distributed_algorithm(data, occupancies, store_weights,
                               num_iterations=100,
                               alpha_fixed=None,
                               alpha_0=None):
    """
    Run the subgradient-based distributed algorithm.

    Parameters
    ----------
    alpha_fixed : float or None
        Fixed step size.  If None, adaptive mode is used.
    alpha_0     : float or None
        Initial step size for adaptive schedule α_k = α_0 / (1 + k).
        Required when alpha_fixed is None.
    """
    if alpha_fixed is None and alpha_0 is None:
        raise ValueError("Provide either alpha_fixed or alpha_0.")

    num_timeslots = data['num_timeslots']
    lambda_prev   = np.zeros(num_timeslots)

    history = {'objective_value': [], 'lambda': [], 'violation': []}

    for k in range(num_iterations):

        # --- step size for this iteration ---
        if alpha_fixed is not None:
            alpha_k = alpha_fixed
        else:
            alpha_k = alpha_0 / (1.0 + k)          # adaptive: α_k = α_0 / (1+k)

        # --- local sub-problems (one per store) ---
        p_profiles, T_ins = [], []
        for store in range(len(store_weights)):
            p_profile, T_in = optimize_store_locally(
                store_weights[store], occupancies, data, lambda_prev)
            p_profiles.append(p_profile)
            T_ins.append(T_in)

        p_profiles = np.array(p_profiles)   # (N, T)
        T_ins      = np.array(T_ins)         # (N, 2, T)

        # --- primal objective (discomfort, no λ term) ---
        objective_value = sum(
            store_weights[s] * sum(
                (T_ins[s, r, t] - data['Temperature_reference']) ** 2
                for r in range(2) for t in range(num_timeslots)
            )
            for s in range(len(store_weights))
        )

        # --- dual update ---
        lambda_next, _, violation = coordinator(lambda_prev, alpha_k, data, p_profiles)

        history['objective_value'].append(objective_value)
        history['lambda'].append(lambda_next.copy())
        history['violation'].append(violation.copy())

        lambda_prev = lambda_next

    history['objective_value'] = np.asarray(history['objective_value'], dtype=float)
    history['lambda']          = np.asarray(history['lambda'],          dtype=float)
    history['violation']       = np.asarray(history['violation'],       dtype=float)
    return history


# ==============================================================================
# SECTION 6: PLOTS
# ==============================================================================

def plot_objective_histories(results_by_label, output_path=None,
                              result_centralized=None):
    fig, ax = plt.subplots(figsize=(11, 6))

    # colour for the adaptive curve so it stands out
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, (label, history) in enumerate(results_by_label.items()):
        iters = np.arange(1, len(history['objective_value']) + 1)
        ls    = '--' if 'adaptive' in label.lower() else '-'
        lw    = 2.5  if 'adaptive' in label.lower() else 1.8
        ax.plot(iters, history['objective_value'], lw=lw, ls=ls,
                color=colors[i % len(colors)], label=label)

    if result_centralized is not None:
        ax.axhline(y=result_centralized, color='red', linestyle=':', lw=2,
                   label='Centralized optimum')

    ax.set_xlabel('Iteration')
    ax.set_ylabel('System objective value')
    ax.set_title('Task 7 – objective value across 100 iterations')
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.legend()
    plt.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.show()


def plot_lambda_histories(results_by_label, output_path=None):
    n_cases = len(results_by_label)
    fig, axes = plt.subplots(1, n_cases, figsize=(4 * n_cases, 5), sharey=True)
    if n_cases == 1:
        axes = [axes]

    for ax, (label, history) in zip(axes, results_by_label.items()):
        lambdas = history['lambda']          # (iterations, T)
        for t in range(lambdas.shape[1]):
            ax.plot(np.arange(1, len(lambdas) + 1), lambdas[:, t],
                    lw=1.2, label=f't={t}')
        ax.set_title(label)
        ax.set_xlabel('Iteration')
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel('λ_t')
    fig.suptitle('Evolution of dual multipliers λ_t across iterations', y=1.01)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', ncol=2, fontsize=8)
    plt.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.show()


def plot_violation_histories(results_by_label, output_path=None):
    n_cases = len(results_by_label)
    fig, axes = plt.subplots(1, n_cases, figsize=(4 * n_cases, 5), sharey=True)
    if n_cases == 1:
        axes = [axes]

    for ax, (label, history) in zip(axes, results_by_label.items()):
        violations = history['violation']    # (iterations, T)
        for t in range(violations.shape[1]):
            ax.plot(np.arange(1, len(violations) + 1), violations[:, t],
                    lw=1.2, label=f't={t}')
        ax.axhline(0, color='k', lw=0.8, ls='--')
        ax.set_title(label)
        ax.set_xlabel('Iteration')
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel('Σ p_n,t − P_mall  (kW)')
    fig.suptitle('Constraint violation Σp_n,t − P_mall across iterations', y=1.01)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', ncol=2, fontsize=8)
    plt.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.show()


# ==============================================================================
# SECTION 7: MAIN
# ==============================================================================

data        = fetch_data()
occupancies = pd.read_csv(
    Path(__file__).parent / 'Data' / 'Task7Occupancies.csv',
    header=None, skiprows=1).iloc[:, :-1]

days          = 100
N_stores      = 15
# wn = n+1 for n in [1..15]  →  [2, 3, ..., 16]
store_weights = [float(n + 1) for n in range(1, N_stores + 1)]

# ------------------------------------------------------------------
# Step 1 – Centralized benchmark  (one day = num_timeslots)
# ------------------------------------------------------------------
print("[1/3] Computing centralized optimal solution...")
centralized_solution = centralized_optimal_solution(
    n_stores=N_stores,
    store_weights=store_weights,
    occupancies=occupancies,
    data=data,
)
print(f"      Centralized optimum = {centralized_solution:.4f}")

# ------------------------------------------------------------------
# Step 2 – Fixed step sizes
# ------------------------------------------------------------------
step_sizes     = [0.001, 0.01, 0.1, 1, 10]
results_by_label = {}

for alpha in step_sizes:
    print(f"[2/3] Running distributed algorithm  α = {alpha}")
    results_by_label[f'α = {alpha}'] = run_distributed_algorithm(
        data=data,
        occupancies=occupancies,
        store_weights=store_weights,
        num_iterations=days,
        alpha_fixed=alpha,
    )

# ------------------------------------------------------------------
# Step 3 – Adaptive step size  α_k = α_0 / (1 + k),  α_0 = 5
# ------------------------------------------------------------------
print("[3/3] Running distributed algorithm  adaptive α_0 = 5")
results_by_label['Adaptive α₀=5'] = run_distributed_algorithm(
    data=data,
    occupancies=occupancies,
    store_weights=store_weights,
    num_iterations=days,
    alpha_0=5.0,
)

# ------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------
output_dir = Path(__file__).parent / 'outputs'

plot_objective_histories(
    results_by_label,
    output_dir / 'task7_objective_history.pdf',
    result_centralized=centralized_solution,
)

plot_lambda_histories(
    results_by_label,
    output_dir / 'task7_lambda_history.pdf',
)

plot_violation_histories(
    results_by_label,
    output_dir / 'task7_violation_history.pdf',
)

print("\n" + "=" * 60)
print("Task 7 complete!")
print("=" * 60)