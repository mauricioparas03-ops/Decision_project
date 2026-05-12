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

    #SETS
    model.T = Set(initialize=range(data['num_timeslots']))
    model.R = Set(initialize=[1,2]) 
    model.RT = model.R * model.T
    model.Tp = Set(initialize=range(data['num_timeslots'] - 1))
    model.RTp = model.R * model.Tp

    #PARAMETERS
    model.Occ = Param(
        model.RT,
        initialize=lambda m, r, t: float(occupancies.iloc[r - 1, t])
    )
    model.Tout = Param(model.T, initialize={t: data['outdoor_temperature'][t] for t in model.T})
        
    # Physical constants and thresholds
    model.Pr     = Param(initialize=data['heating_max_power'])
    model.Zexch  = Param(initialize=data['heat_exchange_coeff'])
    model.Zconv  = Param(initialize=data['heating_efficiency_coeff'])
    model.Zloss  = Param(initialize=data['thermal_loss_coeff'])
    model.Zcool  = Param(initialize=data['heat_vent_coeff'])
    model.Zocc   = Param(initialize=data['heat_occupancy_coeff'])
    model.Tinitial = Param(initialize=data['initial_temperature'])
    model.Tref = Param(initialize=data['Temperature_reference'])

    # DECISION VARIABLES
    model.p = Var(model.RTp, bounds=(0, model.Pr))  # Heating power per room
    model.T_in = Var(model.RT)  # Indoor temperature

    # OBJECTIVE: Minimize local weighted discomfort + dual penalty term
    def objective_rule(m):
        discomfort = sum(store_weight * (m.T_in[r, t] - m.Tref) ** 2 for r in m.R for t in m.T)
        dual_term = sum(lambda_multipliers[t] * sum(m.p[r, t] for r in m.R) for t in m.Tp)
        return discomfort + dual_term

    model.obj = Objective(rule=objective_rule, sense = minimize)
    # CONSTRAINTS
    def thermal_dynamics_rule(model, r, t):
        if t == 0:
            return model.T_in[r, t] == model.Tinitial
        else:
            t_prev = t - 1
            r_other = 2 if r == 1 else 1
            occ = model.Occ[1, t_prev] if r == 1 else model.Occ[2, t_prev]
            return model.T_in[r, t] == (
                        model.T_in[r, t_prev]
                        + model.Zexch * (model.T_in[r_other, t_prev] - model.T_in[r, t_prev])
                        + model.Zloss * (model.Tout[t_prev] - model.T_in[r, t_prev])
                        + model.Zconv * model.p[r, t_prev]
                        - model.Zcool
                        + model.Zocc * occ)
    
    model.Temp_Room_Dynamics = Constraint(model.RT, rule=thermal_dynamics_rule)

    # Solver Configuration
    solver = SolverFactory('gurobi')
    solver.solve(model, tee=True)
    p_profile = np.zeros(data['num_timeslots'], dtype=float)
    for t in model.Tp:
        p_profile[t] = sum(value(model.p[r, t], exception=False) or 0.0 for r in model.R)
    T_in = np.array([[value(model.T_in[r, t]) for t in model.T] for r in model.R], dtype=float)
    return p_profile, T_in



# ==============================================================================
# SECTION 3: LAGRANGIAN RELAXATION - MAIN DISTRIBUTED ALGORITHM
# ==============================================================================

def coordinator(lambda_prev, alpha_step, data, p_profiles):
    # p_profiles shape: (N_stores, T_hours)
    sum_p = np.sum(p_profiles, axis=0)  # shape (T_hours,)
    violation = sum_p - data['P_mall']  # g_t = Σ_n p_n,t - P_mall
    lambda_next = np.maximum(0.0, lambda_prev + alpha_step * violation)
    return lambda_next, sum_p, violation

# ==============================================================================
# SECTION 4: CENTRALIZED OPTIMAL SOLUTION (BENCHMARK)
# ==============================================================================

def centralized_optimal_solution(iterations, n_stores, store_weights, occupancies, data):
    model = ConcreteModel()

    #SETS
    model.T = Set(initialize=range(data['num_timeslots'] * iterations))
    model.R = Set(initialize=[1,2]) 
    model.RT = model.R * model.T
    model.S = Set(initialize=range(n_stores))
    model.SRT = model.S * model.R * model.T

    #PARAMETERS
    model.Occ = Param(
        model.SRT,
        initialize=lambda m, s, r, t: float(occupancies.iloc[r - 1, t % data['num_timeslots']])
    )
    model.Tout = Param(model.T, initialize={t: data['outdoor_temperature'][t % data['num_timeslots']] for t in model.T})
        
    # Physical constants and thresholds
    model.Pr     = Param(initialize=data['heating_max_power'])
    model.Zexch  = Param(initialize=data['heat_exchange_coeff'])
    model.Zconv  = Param(initialize=data['heating_efficiency_coeff'])
    model.Zloss  = Param(initialize=data['thermal_loss_coeff'])
    model.Zcool  = Param(initialize=data['heat_vent_coeff'])
    model.Zocc   = Param(initialize=data['heat_occupancy_coeff'])
    model.Tinitial = Param(initialize=data['initial_temperature'])
    model.Tref = Param(initialize=data['Temperature_reference'])

    # DECISION VARIABLES
    model.p = Var(model.SRT, bounds=(0, model.Pr))  # Heating power per room
    model.T_in = Var(model.SRT)  # Indoor temperature

    # OBJECTIVE: Minimize local weighted discomfort + dual penalty term
    def objective_rule(m):
        return sum(store_weights[s] * (m.T_in[s, r, t] - m.Tref) ** 2 for s in m.S for r in m.R for t in m.T)

    model.obj = Objective(rule=objective_rule, sense = minimize)
    # CONSTRAINTS
    def thermal_dynamics_rule(model, s, r, t):
        if t == 0:
            return model.T_in[s, r, t] == model.Tinitial
        else:
            t_prev = t - 1
            r_other = 2 if r == 1 else 1
            occ = model.Occ[s, r, t_prev] if r == 1 else model.Occ[s, r_other, t_prev]
            return model.T_in[s, r, t] == (
                        model.T_in[s, r, t_prev]
                        + model.Zexch * (model.T_in[s, r_other, t_prev] - model.T_in[s, r, t_prev])
                        + model.Zloss * (model.Tout[t_prev] - model.T_in[s, r, t_prev])
                        + model.Zconv * model.p[s, r, t_prev]
                        - model.Zcool
                        + model.Zocc * occ)
    
    model.Temp_Room_Dynamics = Constraint(model.SRT, rule=thermal_dynamics_rule)
    def mall_power_limit_rule(model, t):
        return sum(model.p[s, r, t] for s in model.S for r in model.R) <= data['P_mall']
    model.Mall_Power_Limit = Constraint(model.T, rule=mall_power_limit_rule)
    # Solver Configuration
    solver = SolverFactory('gurobi')
    solver.solve(model, tee=True)
    return value(model.obj) 


# ==============================================================================
# SECTION 5: VISUALIZATION AND ANALYSIS
# ==============================================================================

def run_distributed_algorithm(alpha_step, data, occupancies, store_weights, num_iterations=100):
    num_timeslots = data['num_timeslots']
    lambda_prev = np.zeros(num_timeslots)

    history = {
        'objective_value': [],
        'lambda': [],
        'violation': []
    }

    for _ in range(num_iterations):
        p_profiles = []
        T_ins = []

        for store in range(len(store_weights)):
            p_profile, T_in = optimize_store_locally(
                store_weights[store],
                occupancies,
                data,
                lambda_prev,
            )
            p_profiles.append(p_profile)
            T_ins.append(T_in)

        p_profiles = np.array(p_profiles)
        T_ins = np.array(T_ins)

        objective_value = sum(
            store_weights[store] * sum(
                (T_ins[store, r, t] - data["Temperature_reference"]) ** 2
                for r in range(2)
                for t in range(data["num_timeslots"])
            )
            for store in range(len(store_weights))
        )

        lambda_next, _, violation = coordinator(lambda_prev, alpha_step, data, p_profiles)

        history['objective_value'].append(objective_value)
        history['lambda'].append(lambda_next)
        history['violation'].append(violation)

        lambda_prev = lambda_next

    history['objective_value'] = np.asarray(history['objective_value'], dtype=float)
    history['lambda'] = np.asarray(history['lambda'], dtype=float)
    history['violation'] = np.asarray(history['violation'], dtype=float)
    return history


def plot_objective_histories(results_by_alpha, output_path=None, result_centralized=None):
    fig, ax = plt.subplots(figsize=(11, 6))

    for alpha, history in results_by_alpha.items():
        iterations = np.arange(1, len(history['objective_value']) + 1)
        ax.plot(iterations, history['objective_value'], lw=2, label=f'α = {alpha}')

    if result_centralized is not None:
        ax.axhline(y=result_centralized, color='red', linestyle='--', lw=2, label='Centralized optimum')

    ax.set_xlabel('Iteration')
    ax.set_ylabel('System objective value')
    ax.set_title('Task 7 objective value across 100 iterations')
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.legend()

    plt.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.show()

# def analyze_and_comment_results(results_dict, step_sizes, store_data):

# ==============================================================================
# SECTION 6: MAIN EXECUTION
# ==============================================================================

#load data
data = fetch_data()
occupancies = pd.read_csv(Path(__file__).parent / 'Data' / 'Task7Occupancies.csv', header=None, skiprows = 1).iloc[:, :-1]
days = 100
N_stores = 15
store_weights = [float(n + 1) for n in range(N_stores)]

# TODO: Step 2 - Compute centralized optimal solution
print("[2/5] Computing centralized optimal solution...")
centralized_solution = centralized_optimal_solution(
    iterations=days,
    n_stores=N_stores,
    store_weights=store_weights,
    occupancies=occupancies,
    data=data
)

# Run distributed algorithm for the five step sizes requested in the task
step_sizes = [0.001, 0.01, 0.1, 1, 10]
results_by_alpha = {}
for alpha in step_sizes:
    print(f"  Running distributed algorithm for α = {alpha}")
    results_by_alpha[alpha] = run_distributed_algorithm(
        alpha_step=alpha,
        data=data,
        occupancies=occupancies,
        store_weights=store_weights,
        num_iterations=days,
    )



# Run adaptive step size case
print(f"\n  Adaptive step size α_0 = 5")
# adaptive_results = distributed_optimization_algorithm(
#     N_stores=N_stores,
#     num_iterations=100,
#     alpha_step=None,
#     adaptive=True,
#     alpha_0=5.0
# )

# TODO: Step 4 - Generate visualizations
print("[4/5] Generating visualizations...")
output_dir = Path(__file__).parent / 'outputs'
plot_objective_histories(results_by_alpha, output_dir / 'task7_objective_history.pdf', result_centralized = centralized_solution)

# TODO: Step 5 - Analyze and save results
print("[5/5] Analyzing results and generating report...")

print("\n" + "="*80)
print("Task 7 complete!")
print("="*80)

