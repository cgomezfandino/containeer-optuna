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

## M1 fields

### `metric` (regression only)

Choose the optimization objective. Drives both the scorer AND the direction:

```yaml
metric: mse   # r2 (default) | mse | rmse | mae
```

### `feature_sets` (regression only)

Named feature subsets for feature-set selection. Optuna samples one per trial:

```yaml
feature_sets:
  small: ["a", "b"]
  full:  ["a", "b", "c", "d"]
```

### `source` (datasets.yaml)

Where the dataset comes from:

| Source | Behavior |
|--------|----------|
| `local` (default) | Read from `path`. |
| `kaggle` | Download via `kagglehub` (`kaggle_dataset` required). |
| `sklearn` | Bundled sklearn dataset (`sklearn_name`: iris, diabetes, wine, breast_cancer). |

```yaml
- name: diabetes
  source: sklearn
  sklearn_name: diabetes
  target_column: target
```
