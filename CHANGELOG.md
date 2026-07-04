# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased] — M2 — Clustering Maturity

### Added
- **5 new clustering models**: HDBSCAN, Agglomerative, Spectral, Birch, OPTICS
  — each with industry-aligned hyperparameter ranges and full model cards.
  HDBSCAN imports defensively (requires sklearn >= 1.3; clear ImportError on
  older versions).
- **Model-selection-as-categorical**: the `models: list[str]` field on
  `ExperimentConfig` lets a single Optuna study search across model families.
  Each trial samples `model_type` categorically and builds the chosen model
  with its own namespaced param_space (conditional search space). Works for
  both regression and clustering.
- **New experiments**: `clustering_hdbscan.yaml`, `clustering_model_selection.yaml`.
- **21 new tests** (106 total): DBSCAN fix regression, per-model e2e for the
  5 new clusterers, model-selection e2e (clustering + regression), M2 cards.

### Fixed
- **DBSCAN clustering bug (latent since M0)**: the clustering objective used
  `pipeline.fit(X_train)` + `pipeline.predict(X_test)`, but DBSCAN (and
  HDBSCAN, Agglomerative, Spectral, OPTICS) don't implement `predict()`. The
  `except Exception` swallowed the error, so every DBSCAN trial silently scored
  -1.0. Rewrote the per-fold loop to use `pipeline.fit_predict(X_fold)` (the
  standard clustering stability-CV pattern). DBSCAN now produces valid
  Silhouette scores (verified: ~0.977 on well-separated blobs).
- **random_state injection for non-accepting clusterers**: `get_pipeline` now
  filters overrides (like `random_state`) to only kwargs the estimator's
  constructor accepts, so DBSCAN/Agglomerative/OPTICS/Birch no longer raise
  `TypeError`.

### Changed
- `ExperimentConfig` gained `models: list[str] | None`.
- The clustering objective now uses a shared `_evaluate_clustering_fold` helper.
- Docs regenerated: 20 model cards (was 15), new clustering-model-selection
  tutorial.

## [0.2.0] — M1 — Regression Maturity

5 new regression models, pluggable metrics, diabetes dataset, feature-set
selection. See PR #2.

## [0.1.0] — M0 — Foundation

See PR #1.
