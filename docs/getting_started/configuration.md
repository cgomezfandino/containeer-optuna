# Configuration

Experiments are defined in YAML files under `config/experiments/`. Each is
validated by a pydantic `ExperimentConfig`.

## Minimal example

```yaml
name: my_clustering
task: clustering
dataset: iris
model: kmeans
scaler: standard_scaler
reducer: pca
random_state: 42

cv:
  strategy: kfold
  n_splits: 3

optimization:
  enabled: true
  n_trials: 50
  direction: maximize
  storage: sqlite:///clustering_optuna.db
  load_if_exists: true
  sampler: tpe
  pruner: null
```

## Top-level fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | str | Experiment name (used as default study name suffix). |
| `task` | `regression` \| `clustering` \| `classification` | Optimization task. |
| `dataset` | str | Dataset name in `config/datasets.yaml`. |
| `model` | str | Model name in `config/models.yaml`. |
| `scaler` | str? | Optional scaler (`standard_scaler`, `minmax_scaler`). |
| `reducer` | str? | Optional reducer (`pca`, `umap`). |
| `cv` | CVConfig | Cross-validation config. |
| `optimization` | OptimizationConfig | Optuna study config. |
| `random_state` | int | Global seed (default 42). |

## Environment variables

Settings are read from `.env` with the `CO_` prefix:

```bash
CO_DATA_DIR=./data
CO_RESULTS_DIR=./results
CO_LOG_LEVEL=INFO
```

See `.env.example`.
