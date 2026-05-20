# Part B — Policy Evaluation

## How to run

The only file you need to interact with is [Environment.py](Environment.py). For optimal in hindsight policy run the Optimal_in_Hindsight.py directly

### 1. Select a policy

At the top of `Environment.py`, uncomment the import for the policy you want to evaluate and comment out the others:

```python
# from policies.dummy_policy          import select_action   # Dummy (baseline)
# from policies.lookahead_policy      import select_action   # Deterministic Lookahead
# from policies.SP_policy             import select_action   # 2-Stage Stochastic 
# from policies.multiSP               import select_action   # Multi-Stage SP
# from policies.ADP_policy            import select_action   # ADP
# from policies.Hybrid_multi_and_adp  import select_action   # Hybrid ADP
```

### 2. Set the policy name

Near the bottom of `Environment.py`, update `policy_name` to match the policy you selected:

```python
policy_name = "multiSP"   # change to: dummy | lookahead | SP | multiSP | ADP | Hybrid_multi_and_adp | hindsight
```

This name is used for the output file — make sure it matches one of the keys recognised by `read_results.py` (see table below).

### 3. Run the simulation

```bash
cd PART_B
python Environment.py
```

The script runs 100 simulated days and saves the results to `outputs/results_<policy_name>.npz`.

---

## Comparing policies

Once you have generated `.npz` files for the policies you want to compare, open [read_results.py](read_results.py) and list them in the `policies` variable at the top:

```python
policies = ["dummy", "lookahead", "SP", "multiSP1", "hindsight", "Hybrid_multi_and_adp", "ADP"]
```

Then run:

```bash
python read_results.py
```

This prints a summary table and produces three figures, all saved to the `outputs/` folder:

| File | Content |
|---|---|
| `outputs/fig_avg_cost_comparison.png/pdf` | Bar chart — average daily cost with std error bars |
| `outputs/fig_cost_distribution.png/pdf` | Box plots — daily cost distributions |
| `outputs/fig_metrics_table.png/pdf` | Full metrics table (cost, ventilation hours, overrule acts, energy, price-weighted cost) |

---

## Available policies

| `policy_name` key | Module | Description |
|---|---|---|
| `dummy` | `policies/dummy_policy.py` | Baseline rule-based policy |
| `lookahead` | `policies/lookahead_policy.py` | Deterministic lookahead |
| `SP` | `policies/SP_policy.py` | 2-stage stochastic program |
| `multiSP` | `policies/multiSP.py` | Multi-stage stochastic program |
| `ADP` | `policies/ADP_policy.py` | Approximate Dynamic Programming |
| `hybrid` | `policies/Hybrid_multi_and_adp.py` | Hybrid ADP |
| `hindsight` | `Optimal_in_Hindsight.py` | Optimal in hindsight (lower bound) |


