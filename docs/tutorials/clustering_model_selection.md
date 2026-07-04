# Clustering model selection

Optuna can search across clustering **model families** in a single study.
This is the model-selection-as-categorical pattern.

## Concept

Set `models: [...]` on the experiment. Each trial samples `model_type`
categorically, then builds the chosen model with its own namespaced
`param_space` (a conditional search space). The chosen family is stored as a
`user_attr` so the dashboard reveals which algorithm won.

## Example

```yaml
name: clustering_model_selection
task: clustering
dataset: iris
model: kmeans          # fallback (required field)
models:                # M2 — search across families
  - kmeans
  - agglomerative
  - spectral
  - birch
scaler: standard_scaler
reducer: pca
```

## Run

```bash
containeer-optuna run config/experiments/clustering_model_selection.yaml --n-trials 60
```

## What you learn

The param-importances plot shows `model_type` as the dominant factor (or not).
The parallel-coordinate plot reveals which families + hyperparameters cluster
together. This works for **regression too** — set `models: [ridge,
random_forest, gradient_boosting]`.

## Works for regression too

```yaml
task: regression
models: [ridge, random_forest, gradient_boosting, svr]
metric: r2
```
