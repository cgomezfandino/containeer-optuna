# Quickstart

## Run a clustering experiment

```bash
containeer-optuna run config/experiments/clustering_optimization.yaml --n-trials 20
```

This optimizes KMeans over Iris data (PCA-reduced, StandardScaler) for
maximum Silhouette across 3-fold CV, storing Calinski-Harabasz and
Davies-Bouldin as user attributes.

## Run from Python

```python
from containeer_optuna import load_config, OptunaRunner

cfg = load_config("config/experiments/clustering_optimization.yaml")
runner = OptunaRunner(cfg)
study = runner.run(n_trials=20)
print("best silhouette:", study.best_value)
print("best params:", study.best_params)
print("user attrs:", study.best_trial.user_attrs)
```

## Explore the model catalog

```bash
containeer-optuna list-models
containeer-optuna list-models --type clustering
containeer-optuna describe kmeans
```

## Scaffold a new experiment

```bash
containeer-optuna init my_experiment --task regression --dataset auto_mpg --model ridge
containeer-optuna run config/experiments/my_experiment.yaml
```

## Inspect a study with the dashboard

```bash
containeer-optuna dashboard --storage sqlite:///clustering_optuna.db
# Then run the printed command
optuna-dashboard sqlite:///clustering_optuna.db
```
