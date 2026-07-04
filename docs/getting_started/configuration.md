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
| `model` | str | Model name in `config/models.yaml` (default/fallback model). |
| `scaler` | str? | Optional scaler (`standard_scaler`, `minmax_scaler`). |
| `reducer` | str? | Optional reducer (`pca`, `umap`, `truncated_svd`, `factor_analysis`). |
| `cv` | CVConfig | Cross-validation config. |
| `optimization` | OptimizationConfig | Optuna study config. |
| `random_state` | int | Global seed (default 42). |
| `metric` | str? | Optimization metric — drives scorer AND direction. See below. |
| `feature_sets` | dict? | Named feature subsets for feature-set selection (regression). |
| `models` | list[str]? | Model families for model-selection-as-categorical. See below. |

## `metric` — pluggable optimization metric

When set, drives both the scorer and the optimization direction automatically.

**Regression** (`r2`→maximize, others→minimize):

```yaml
metric: mse   # r2 (default) | mse | rmse | mae
```

**Classification** (all maximize):

```yaml
metric: accuracy   # accuracy (default) | f1 | f1_weighted | roc_auc | roc_auc_ovr
```

## `feature_sets` — feature-set selection (regression)

Named feature subsets. Optuna samples one per trial:

```yaml
feature_sets:
  small: ["a", "b"]
  full:  ["a", "b", "c", "d"]
```

## `models` — model-selection-as-categorical (M2)

When set (non-empty), the runner searches across the listed model families
in a single Optuna study. Each trial samples `model_type` categorically and
builds the chosen model with its own namespaced `param_space`. Works for
regression, clustering, and classification:

```yaml
models:
  - ridge
  - random_forest
  - gradient_boosting
```

## `source` (datasets.yaml)

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

## Auto-derived CV defaults

| Task | Default CV strategy |
|------|-------------------|
| regression | `shuffle_split` |
| clustering | `kfold` |
| classification | `stratified_kfold` |

These are only applied when you don't explicitly set `cv.strategy` in YAML.

## Environment variables

Settings are read from `.env` with the `CO_` prefix:

```bash
CO_DIR=.                           # Base directory for config/ lookups
CO_DATA_DIR=data                   # Data root for local datasets
CO_RESULTS_DIR=results             # Artifacts directory
CO_LOG_LEVEL=INFO                  # Logging level
CO_OPTUNA_STORAGE=sqlite:///optuna_studies.db
CO_CLUSTERING_STORAGE=sqlite:///clustering_optuna.db
```

See `.env.example`.
