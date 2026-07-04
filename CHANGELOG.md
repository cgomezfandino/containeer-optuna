# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — M1 — Regression Maturity

### Added
- **5 new regression models**: ElasticNet, DecisionTree, RandomForest,
  GradientBoosting, SVR — each with industry-aligned hyperparameter ranges
  (search spaces cross-referenced with Optuna/sklearn community benchmarks)
  and full model cards (pros/cons/when-to-use/assumptions/complexity).
- **Pluggable regression metrics**: the `metric` field on `ExperimentConfig`
  (`r2`/`mse`/`rmse`/`mae`) drives both the CV scorer and the optimization
  direction automatically. New `get_regression_scorer()` helper.
- **Feature-set selection**: the `feature_sets` field lets Optuna search over
  named feature subsets per trial (formalizes the notebook pattern). The
  chosen subset is stored as a `user_attr`.
- **Bundled diabetes dataset** (`source: sklearn`): 442×10 regression dataset
  with no download required — enables true end-to-end regression CI tests.
  Generic `source` field on `DatasetConfig` replaces the hard-coded Iris branch
  (`local`/`kaggle`/`sklearn`).
- **New experiments**: `diabetes_regression.yaml` (RF on diabetes),
  `diabetes_feature_set_selection.yaml` (feature-set selection demo).
- **28 new tests** (85 total): per-model instantiation + sampling tests, the
  `"None"` → `None` max_depth coercion, metric→direction derivation, diabetes
  loader, MSE objective e2e, feature-set selection e2e, RF e2e.

### Changed
- `max_depth` for RF/DT uses a categorical with a `"None"` sentinel; the
  registry coerces it to Python `None` before construction (no dispatcher change).
- The Iris dataset loader now uses the generic `source: sklearn` dispatch.
- `DatasetConfig` gained `source` and `sklearn_name` fields.
- `ExperimentConfig` gained `metric` and `feature_sets` fields.
- Docs regenerated: 15 model cards (was 10), new feature-set-selection
  tutorial, updated evaluation/models/configuration guides.

## [0.1.0] — M0 — Foundation

See the M0 release notes in the merged PR #1.
