# Feature-set selection

Optuna can search over **named feature subsets** — not just hyperparameters.
This formalizes the pattern from `notebooks/optuna/more_advanced_workflow.ipynb`.

## Concept

Define named feature subsets in the experiment YAML. Optuna samples a subset
name categorically per trial and slices `X` to that subset before
cross-validation. The chosen subset is stored as a `user_attr` for inspection.

## Example: diabetes

```yaml
name: diabetes_feature_set_selection
task: regression
dataset: diabetes
model: ridge
metric: r2
scaler: standard_scaler

feature_sets:
  clinical:    ["age", "sex", "bmi", "bp"]
  serum:       ["s1", "s2", "s3", "s4", "s5", "s6"]
  full:        ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]
  bmi_focus:   ["bmi", "s5", "bp", "s3"]

optimization:
  n_trials: 60
  direction: maximize
```

## Run

```bash
containeer-optuna run config/experiments/diabetes_feature_set_selection.yaml --n-trials 30
```

## What you learn

The Optuna dashboard's parallel-coordinate and slice plots reveal which feature
subsets perform best. For diabetes, the `full` set usually wins (Ridge benefits
from all 10 features), but you may find that a small subset (e.g. `bmi_focus`)
gives 90% of the performance with far fewer features — valuable for
interpretability and deployment.

## From Python

```python
from containeer_optuna import load_config, OptunaRunner

cfg = load_config("config/experiments/diabetes_feature_set_selection.yaml")
study = OptunaRunner(cfg).run(n_trials=30)
print("best feature_set:", study.best_params["feature_set"])
```
