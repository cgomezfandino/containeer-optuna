# Regression tutorial

This tutorial reproduces the Auto MPG regression study from
`notebooks/optuna/more_advanced_workflow.ipynb` using the framework.

## Setup

Download the UCI Auto MPG dataset to `./data/auto-mpg.data` (whitespace-
separated, `?` for missing values).

## Run

```bash
containeer-optuna run config/experiments/auto_mpg_model_selection.yaml --n-trials 50
```

## What it does

- Loads `auto_mpg` (target: `horsepower`) using the YAML preprocessing recipe
  (`?` → NaN, numeric coercion, dropna).
- Builds a `StandardScaler → Ridge` pipeline.
- Tunes `alpha` and `tol` over ShuffleSplit(5) CV, maximizing R².
- Stores the study at `sqlite:///optuna_studies.db` (resumable).

## From Python

```python
from containeer_optuna import load_config, OptunaRunner

cfg = load_config("config/experiments/auto_mpg_model_selection.yaml")
study = OptunaRunner(cfg).run(n_trials=50)
print("best R²:", study.best_value)
```

See also the [mirror script](https://github.com/cgomezfandino/containeer-optuna/tree/main/scripts)
and the [Levels of Maturity](levels_of_maturity.md) tutorial.


## M1: New models + pluggable metrics + diabetes

M1 adds 5 regression models (ElasticNet, DecisionTree, RandomForest,
GradientBoosting, SVR), pluggable metrics, and a bundled diabetes dataset
(no download). Compare them:

```bash
containeer-optuna describe random_forest
containeer-optuna describe svr
```

Run RandomForest on diabetes (R squared, no external data needed):

```bash
containeer-optuna run config/experiments/diabetes_regression.yaml --n-trials 30
```

Optimize for MAE instead of R squared (direction auto-derives to minimize):

```yaml
metric: mae
```

See [Feature-set selection](feature_set_selection.md) to search over named
feature subsets.
