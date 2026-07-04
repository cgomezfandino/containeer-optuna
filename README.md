# containeer-optuna

[![CI](https://github.com/cgomezfandino/containeer-optuna/actions/workflows/ci.yml/badge.svg)](https://github.com/cgomezfandino/containeer-optuna/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **An Optuna-first data science framework** for ML, DL, AI, and statistics.

`containeer_optuna` is a modular framework for running Optuna hyperparameter
optimization experiments across regression, clustering, dimensionality
reduction, and (in later milestones) classification, statistics, and deep
learning. Configured via YAML, validated with pydantic, runnable from the CLI.

## Why use it

- **Declarative experiments** — one YAML per experiment, validated by typed models.
- **Optuna-first** — every experiment is a study; samplers, pruners, storage, and visualization are first-class.
- **Reusable abstractions** — datasets, models, pipelines, and objectives are registry-driven. Adding a model takes two lines.
- **Model cards** — every model ships with pros/cons, assumptions, and when-to-use, surfaced in docs and `containeer-optuna describe`.
- **Reproducible** — global seeding, SQLite storage, resumable studies.

## Quick install

```bash
pip install -e ".[dev]"
```

## Quick start

### CLI

```bash
# Run a clustering study
containeer-optuna run config/experiments/clustering_optimization.yaml --n-trials 20

# Explore the catalog
containeer-optuna list-models
containeer-optuna describe kmeans     # prints the model card (pros/cons)
containeer-optuna dashboard --storage sqlite:///clustering_optuna.db
```

### Python

```python
from containeer_optuna import load_config, OptunaRunner

cfg = load_config("config/experiments/clustering_optimization.yaml")
study = OptunaRunner(cfg).run(n_trials=20)

print("best silhouette:", study.best_value)
print("best params:", study.best_params)
print("user attrs:", study.best_trial.user_attrs)
```

## What's included (M0 — Foundation)

| Component | Status |
|-----------|--------|
| Regression (Ridge, Lasso, OLS) | ✅ |
| Clustering (KMeans, DBSCAN, GMM) | ✅ |
| Reducers (PCA, UMAP) | ✅ |
| Scalers (Standard, MinMax) | ✅ |
| YAML + pydantic config | ✅ |
| CLI (run, describe, list-*, init, dashboard) | ✅ |
| Model cards (pros/cons for all 10 models) | ✅ |
| Optuna samplers (TPE, Random, CMA-ES, NSGA-II) + pruners | ✅ |
| Tests (57) | ✅ |
| Docs (mkdocs-material) | ✅ |
| CI (GitHub Actions) + pre-commit | ✅ |

## Documentation

```bash
mkdocs serve
```

Or read the markdown sources under [`docs/`](docs/). Highlights:

- [Getting Started](docs/getting_started/installation.md)
- [Model Cards](docs/model_cards/index.md) — pros/cons per model
- [Tutorials](docs/tutorials/regression_tutorial.md) — adapted from the notebooks
- [Roadmap](docs/roadmap.md) — milestones M0–M9

## Notebooks & mirror scripts

The original Optuna experiments live under `notebooks/` and remain runnable
standalone. Mirror `.py` scripts (extracted from the notebooks) live under
`scripts/`:

| Notebook | Mirror script | Topic |
|----------|---------------|-------|
| `notebooks/optuna/more_advanced_workflow.ipynb` | `scripts/more_advanced_workflow.py` | Model & feature-set selection |
| `notebooks/optuna/hyperparam_tuning_levels_of_maturity.ipynb` | `scripts/hyperparam_tuning_levels_of_maturity.py` | Manual → Grid → Optuna progression |
| `notebooks/optuna/perplexity.ipynb` | `scripts/perplexity.py` | Joint clustering pipeline optimization |
| `notebooks/optuna/kaggle_dataset.ipynb` | `scripts/kaggle_dataset.py` | Kaggle data download |
| `notebooks/customer_personality_analysis/model.ipynb` | `scripts/customer_personality_model.py` | Customer segmentation demo |

## Project layout

```
containeer-optuna/
├── src/containeer_optuna/      # the package (config, data, models, pipelines, optimization, evaluation, utils, cli)
├── config/                     # datasets.yaml, models.yaml, experiments/*.yaml
├── docs/                       # mkdocs-material site (English)
├── notebooks/                  # original Optuna experiments (runnable)
├── scripts/                    # mirror .py scripts extracted from notebooks
├── tests/                      # pytest suite (57 tests)
├── mkdocs.yml
├── pyproject.toml
└── .github/workflows/ci.yml
```

## Roadmap

The framework grows incrementally (Agile milestones). M0 is this release.
Later milestones add: more regression models (M1), more clustering (M2),
dimensionality reduction incl. t-SNE (M3), classification (M4), statistics
(M5), deep learning (M6–M7), NLP/AI (M8), and productionization (M9). See
[Roadmap](docs/roadmap.md).

## License

[MIT](LICENSE) © Carlos Gomez
