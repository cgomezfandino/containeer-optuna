# Changelog

## [Unreleased] — M3 — Dimensionality Reduction

### Added
- **3 new reducers**: t-SNE, TruncatedSVD, FactorAnalysis — each with full
  model cards (pros/cons/when-to-use/assumptions/complexity).
- **Visualization module** (`evaluation/plotting.py`): `plot_embedding_2d`
  (2D scatter colored by cluster labels, with noise handling) and `plot_scree`
  (explained-variance bar+cumulative chart for PCA/SVD).
- **`runner.plot_best_embedding()`**: rebuilds the best-trial pipeline,
  projects `self.X` to 2D via `fit_transform`, and returns a matplotlib Figure
  of the cluster structure. Works with ALL reducers including t-SNE (applies
  steps manually, bypassing sklearn Pipeline's transform requirement).
- **New experiment**: `clustering_truncated_svd.yaml`.
- **18 new tests** (124 total): M3 reducer instantiation + sampled params,
  reducer step types, t-SNE-in-pipeline TypeError guard, plotting smoke
  tests, `plot_best_embedding` e2e, regression-task guard.

### Design note (t-SNE)
t-SNE does NOT implement `transform()`, so it cannot be an intermediate step
in a sklearn Pipeline (which validates transform on all intermediate steps).
Therefore t-SNE is **not usable as a YAML reducer** in an experiment. It IS
usable via `plot_best_embedding()` (which applies steps manually via
`fit_transform`). For a non-linear YAML reducer, use UMAP (which has
transform). The t-SNE model card documents this clearly.

## [0.3.0] — M2 — Clustering Maturity

5 new clustering models, model-selection-as-categorical, DBSCAN fix. See PR #3.

## [0.2.0] — M1 — Regression Maturity

5 new regression models, pluggable metrics, diabetes dataset, feature-set
selection. See PR #2.

## [0.1.0] — M0 — Foundation

See PR #1.
