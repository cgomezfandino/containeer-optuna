# Optimization

`OptunaRunner` is the central execution entry point. Given an
`ExperimentConfig`, it:

1. Loads the dataset.
2. Builds an objective (regression: mean CV R²; clustering: mean CV Silhouette
   with CH/DB as user attrs).
3. Constructs the study (sampler + pruner + storage + study_name).
4. Runs `study.optimize(...)` and returns the study.

## Samplers

- `tpe` (default) — Tree-structured Parzen Estimator, multivariate.
- `random` — Random search.
- `cmaes` — CMA-ES (falls back to TPE if unavailable).
- `nsgaii` — NSGA-II for multi-objective (falls back to TPE).

## Pruners

- `null` / `none` (default in M0 notebooks) — no pruning.
- `median` — MedianPruner.
- `percentile` — PercentilePruner.
- `hyperband` — HyperbandPruner.

## Clustering objective notes

The clustering objective fixes two issues from the original notebooks:

1. The full pipeline (scaler → reducer → clusterer) is applied end-to-end.
   The notebook used to refit the clusterer directly on the test fold,
   bypassing the scaler/reducer.
2. DBSCAN noise points (label `-1`) are stripped before computing metrics.

## Visualization

```python
runner.plot_standard()
# Returns: optimization_history, param_importances, slice, contour
```
