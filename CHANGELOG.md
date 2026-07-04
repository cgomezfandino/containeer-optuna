# Changelog

## [1.0.0] — M9 — Productionization

### Added
- **Model serialization**: `save_model: true` in experiment YAML now persists
  the best model as `.joblib` (via joblib). New `fit_best_pipeline()` method
  on OptunaRunner reconstructs and fits the best-trial pipeline on the full
  dataset.
- **Prediction saving**: `save_predictions: true` now persists predictions
  as `.npy`. Previously these config fields were declared but dead.
- **MLflow tracking** (optional `[mlflow]` extra): set `tracking: mlflow` in
  optimization config to log params/metrics to MLflow. Lazy import.
- **`__main__.py`**: `python -m containeer_optuna --help` now works.
- **PyPI release metadata**: `[project.urls]`, classifier bumped Alpha→Beta,
  requires-python bumped ≥3.10 (matches CI matrix).
- **Publish workflow** (`.github/workflows/publish.yml`): auto-publishes to
  PyPI on tag/release push via `pypa/gh-action-pypi-publish`.
- **7 new tests** (192 total): fit_best_pipeline, save_model, save_predictions,
  save_model via config, python -m, tracking field, DL rejection.

### Changed
- Version bumped: `0.6.0` → `1.0.0`.
- `requires-python` bumped from `>=3.9` to `>=3.10` (matches CI matrix).
- Development Status classifier: `3 - Alpha` → `4 - Beta`.

## [0.6.0] — M5–M6

Statistics + DL MLP. See PRs #6–#8.

## [0.7.0] — M7–M8

DL CNN/RNN + NLP transformers. See PRs #9–#10.
