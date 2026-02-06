# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 16:11:01 2025

@author: geots
"""

# student_plan.py
from pyomo.environ import *
import random

# --------------------------
# Problem setup (numeric)
# --------------------------

# Target grade
Target = 95.0
LectureQuality = 0.35


weeks = list(range(1, 15))   
tasks = list(range(1, 7))    # 6 tasks

# Hours available per week
Hours = {t: 12 for t in weeks}

# threshold: if fatigue > Threshold then stressed
Threshold = 1

# Big-M: choose conservatively as 1
M = 10  # safe-large value

# Task difficulty, measured in hours needed to tackle it to perfection
Difficulty = {1: 10, 2: 10, 3: 20, 4: 20, 5: 25, 6: 20}

#weight of how much each Task counts towards the grade
Weight =     {1: 0.1, 2: 0.1, 3: 0.2, 4: 0.2, 5: 0.3, 6: 0.1}

# For each (week,task) an Effect of attending that week's lecture on ability and, subsequently, grade
random.seed(0)
Effect = {}
for t in weeks:
    for c in tasks:
        # generate reasonable structure: earlier lectures help tasks that start earlier more
        Effect[(t,c)] = round(max(0.0, min(1.0, 0.6 + 0.4*(0.5 - abs(t - (2*c))/max(1, len(weeks))))), 3)*LectureQuality

# Task availability windows (start_week, end_week) inclusive
start = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 2}
end   = {1: 3, 2: 3, 3: 12, 4: 12, 5: 12, 6: 12}

# Construct availability parameter avail[w,c] = 1 if w in [start_c, end_c], else 0
avail = {}
for t in weeks:
    for c in tasks:
        avail[(t,c)] = 1 if (t >= start[c] and t <= end[c]) else 0

#Fatigue Accumulation Factor
xi = 0.5



# --------------------------
# Pyomo model
# --------------------------
model = ConcreteModel()

# Sets
model.W = Set(initialize=weeks, ordered=True)
model.C = Set(initialize=tasks, ordered=True)
model.WC = Set(initialize=[(t,c) for t in weeks for c in tasks])  # will use avail to restrict

# Parameters
model.Hours = Param(model.W, initialize=Hours)
model.avail = Param(model.WC, initialize=avail, within=Binary)
model.Effect = Param(model.WC, initialize=Effect)
model.Difficulty = Param(model.C, initialize=Difficulty)
model.Weight = Param(model.C, initialize=Weight)
model.M = Param(initialize=M)
model.Target = Param(initialize=Target)
model.Threshold = Param(initialize=Threshold)
model.xi = Param(initialize=xi)

# Variables
# Effort hours for week w, task c (continuous, >=0). Bound to Hours[w] and avail.
#Cannot work on task before given/after deadline -> enforced by h upper bound via avail
def h_bounds(model,t,c):
    return (0.0, model.Hours[t]*model.avail[t,c])
model.h = Var(model.WC, domain=NonNegativeReals, bounds=h_bounds)

# attendance binary (4 hours per attended lecture)
model.a = Var(model.W, domain=Binary)

# identify extra weeks 
last_teaching_week = 12
extra_weeks = [t for t in model.W if t > last_teaching_week]

# force no attendance
for t in extra_weeks:
    model.a[t].fix(0)


# stressed binary for week w
model.s = Var(model.W, domain=Binary)

# grade (continuous)
model.G = Var(domain=NonNegativeReals)

# fatigue (continuous)
model.F = Var(model.W, domain=NonNegativeReals)

# --------------------------
# Constraints
# --------------------------

# Grade definition
def grade_rule(m):
    return m.G == sum(
        (m.h[t,c] + 4*m.a[t]*m.Effect[t,c]) *
        (m.Weight[c] / m.Difficulty[c])*100
        for t in m.W for c in m.C
    )

model.grade_def = Constraint(rule=grade_rule)


# Achieve target
model.target_constr = Constraint(expr = model.G >= model.Target)


# Weekly working limit (if stressed, zero available)
def weekly_limit_rule(m, t):
    return sum(m.h[t,c] for c in m.C) + 4*m.a[t] <= (1 - m.s[t]) * m.Hours[t]
model.weekly_limit = Constraint(model.W, rule=weekly_limit_rule)


# Fatigue Accumulation Dynamics
def fatigue_rule(m, t):
    if t == max(m.W):
        return Constraint.Skip
    return m.F[t+1] == m.xi*m.F[t] + (1/m.Hours[t])*(sum(m.h[t,c] for c in m.C) + 4*m.a[t])

model.fatigue_acc = Constraint(model.W, rule=fatigue_rule)

# If Fatigue >= Threshold --> Stress

# big-M inequalities:
def stress_upper_rule(m,t):
    return m.F[t] - m.Threshold <= m.s[t] * m.M
model.stress_upper = Constraint(model.W, rule=stress_upper_rule)

def stress_lower_rule(m,t):
    return m.F[t] - m.Threshold >= (m.s[t] - 1.0) * m.M
model.stress_lower = Constraint(model.W, rule=stress_lower_rule)


# Initial Fatigue
model.F[1].fix(0.5)


# Fatigue Accumulation Dynamics
def fatigue_rule(m, t):
    if t == max(m.W):
        return Constraint.Skip
    return m.F[t+1] == m.xi*m.F[t] + (1/m.Hours[t])*(sum(m.h[t,c] for c in m.C) + 4*m.a[t])

model.fatigue_acc = Constraint(model.W, rule=fatigue_rule)


# Objective: minimize total fatigue + stress
def obj_rule(m):
#    return sum(m.h[t,c] for t in m.W for c in m.C) + 4*sum(m.a[t] + 10*m.s[t] for t in m.W)
    return sum(m.F[t] + 5*m.s[t] for t in m.W)
model.obj = Objective(rule=obj_rule, sense=minimize)

# --------------------------
# Solve with Gurobi
# --------------------------
if __name__ == "__main__":
    solver = SolverFactory('gurobi')
    if not solver.available():
        raise RuntimeError("Gurobi solver is not available in your environment.")
    results = solver.solve(model, tee=True)
    results.write()

    # Print summary
    print("\nObjective (total hours + attendance-hours):", value(model.obj))
    print("Grade achieved:", value(model.G))
    print("\nAttendance by week (a[w]=1 means attend):")
    for w in model.W:
        if value(model.a[w]) > 0.5:
            print(f"  Week {w}: attend")
    print("\nStressed weeks (s[w]=1):")
    for w in model.W:
        if value(model.s[w]) > 0.5:
            print(f"  Week {w}: stressed")
    print("\nNonzero effort (h[w,c]):")
    for w in model.W:
        for c in model.C:
            hv = value(model.h[w,c])
            if hv > 1e-6:
                print(f"  Week {w}, Task {c}: {hv:.2f} hours")
                
                
    import matplotlib.pyplot as plt
    import numpy as np
    
    # ============================================================
    # Extract optimal solution
    # ============================================================
    
    # Weeks and tasks in sorted order
    W = list(model.W.data())
    C = list(model.C.data())
    
    # Effort matrix
    Hmat = np.zeros((len(W), len(C)))
    for i, w in enumerate(W):
        for j, c in enumerate(C):
            Hmat[i, j] = float(model.h[w, c].value)
    
    # Attendance vector
    A = np.array([float(model.a[w].value) for w in W])
    
    # Stress vector
    S = np.array([float(model.s[w].value) for w in W])
    
    # Weekly total effort (h + 4*a)
    weekly_effort = np.array([
        sum(float(model.h[w, c].value) for c in C) + 4 * float(model.a[w].value)
        for w in W
    ])
    
    
    # ============================================================
    # Heatmap of Effort by Week and Task
    # ============================================================
    
    plt.figure(figsize=(10, 5))
    plt.imshow(Hmat, cmap="viridis", aspect="auto")
    plt.colorbar(label="Effort (hours)")
    plt.xlabel("Task")
    plt.ylabel("Week")
    plt.xticks(range(len(C)), C)
    plt.yticks(range(len(W)), W)
    plt.title("Effort Hours by Week and Task (Optimal Solution)")
    plt.tight_layout()
    plt.show()
    
    
    # ============================================================
    # Weekly Total Effort – STACKED BARS (attendance + study hours)
    # ============================================================
    
    # Attendance hours (bottom layer)
    attendance_hours = np.array([4 * float(model.a[w].value) for w in W])
    
    # Study hours (top layer)
    study_hours = np.array([sum(float(model.h[w, c].value) for c in C) for w in W])
    
    plt.figure(figsize=(10, 4))
    plt.bar(W, attendance_hours, width=0.6, label="Attendance Hours (4*a[w])")
    plt.bar(W, study_hours, width=0.6, bottom=attendance_hours, 
            label="Study Hours (sum h[w,c])")
    
    plt.xlabel("Week")
    plt.ylabel("Hours")
    plt.title("Weekly Effort (Stacked): Attendance + Study")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # ============================================================
    # Weekly Total Effort – STACKED BARS + Fatigue (2nd y-axis)
    # ============================================================
    
    attendance_hours = np.array([4 * float(model.a[w].value) for w in W])
    study_hours = np.array([sum(float(model.h[w, c].value) for c in C) for w in W])
    fatigue = np.array([float(model.F[w].value) for w in W])
    
    fig, ax1 = plt.subplots(figsize=(10, 4))
    
    # --- Primary axis: effort (stacked bars)
    ax1.bar(W, attendance_hours, width=0.6, label="Attendance Hours (4·a[w])")
    ax1.bar(W, study_hours, width=0.6, bottom=attendance_hours,
            label="Study Hours (∑ h[w,c])")
    
    ax1.set_xlabel("Week")
    ax1.set_ylabel("Hours")
    ax1.set_title("Weekly Effort and Fatigue")
    ax1.legend(loc="upper left")
    
    # --- Secondary axis: fatigue (RED)
    ax2 = ax1.twinx()
    ax2.plot(W, fatigue, color="red", marker="o", linewidth=2,
             label="Fatigue F[w]")
    ax2.axhline(y=Threshold, color="red", linestyle="--", linewidth=1.5,
                label="Fatigue Threshold")
    
    ax2.set_ylabel("Fatigue", color="red")
    ax2.tick_params(axis="y", colors="red")
    
    # Fix fatigue axis scale to create vertical room
    ax2.set_ylim(0, 1.5)
    ax2.set_yticks(np.linspace(0, 1.5, 6))

    
    # --- Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="upper center", ncol=3)
    
    plt.tight_layout()
    plt.show()
    
    
            
        
            
