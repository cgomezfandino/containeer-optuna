# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — Unreleased (M0 — Foundation)

### Added
- `containeer_optuna` package: importable from `src/`.
- Subpackages: `config`, `data`, `models`, `pipelines`, `optimization`,
  `evaluation`, `utils`, `cli`.
- `OptunaRunner` — YAML-driven Optuna study execution with samplers (TPE,
  Random, CMA-ES, NSGA-II) and pruners (Median, Percentile, Hyperband, None).
- Generic `param_space` dispatcher (float/int/categorical → `trial.suggest_*`).
- Model registry: Ridge, Lasso, OLS, KMeans, DBSCAN, GMM, PCA, UMAP,
  StandardScaler, MinMaxScaler.
- Dataset registry: Auto MPG (local), Iris (bundled), customer_personality,
  credit_card (Kaggle).
- Clustering objective with DBSCAN noise stripping and full-pipeline scoring
  (fixes two latent bugs in the source notebook).
- `containeer-optuna` CLI: run, list-models, list-datasets, list-experiments,
  describe (model cards), dashboard, init.
- Model cards (pros/cons/when-to-use/assumptions/complexity) for all 10 models.
- Experiment catalog: `config/experiments/{clustering_optimization,
  auto_mpg_model_selection, auto_mpg_feature_set_selection}.yaml`.
- pytest suite: 57 tests across config, data, models, pipelines, optimization,
  evaluation, CLI.
- mkdocs-material documentation site (English): getting started, user guide,
  model cards, tutorials, API reference, roadmap, contributing.
- Pre-commit hooks, GitHub Actions CI, LICENSE (MIT), CHANGELOG, .env.example.

### Changed
- `config.py`: fixed path resolution bug (was looking at `src/config/` instead
  of `<repo>/config/`); widened `ModelConfig.type` to accept reducer/scaler;
  fixed `study_name` validator to use the experiment name; added `CO_` env
  prefix to `Settings`.
- `pyproject.toml`: removed the broken inline `[tool.mkdocs]` block; moved ruff
  config to `[tool.ruff.lint]`; bumped mypy `python_version` to 3.10.
- `requirements.txt`: realigned with `pyproject.toml` dependencies (was a
  notebook-only freeze missing pydantic/typer/rich/etc.). Old freeze moved to
  `requirements-notebooks.txt`.

### Removed
- `main.py` (dead stub).
- `src/containeer_optuna/optimization/{objectives}/` (stray literal-brace dir).
