# containeer-optuna

[![CI](https://github.com/cgomezfandino/containeer-optuna/actions/workflows/ci.yml/badge.svg)](https://github.com/cgomezfandino/containeer-optuna/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-167-brightgreen.svg)](tests/)

> **An Optuna-first data science framework** for ML, statistics, and (soon) deep learning.

`containeer_optuna` is a modular framework for running Optuna hyperparameter optimization experiments across regression, clustering, classification, and dimensionality reduction — plus a full statistics toolkit for EDA. Configured via YAML, validated with pydantic, runnable from the CLI.

## Why use it

- **27 models** across regression (8), clustering (8), classification (4), reducers (5), scalers (2) — each with a model card (pros/cons/when-to-use).
- **3 tasks**: regression, clustering, classification — all with pluggable metrics and auto-derived CV strategies.
- **Model-selection-as-categorical**: one Optuna study searches across model families.
- **Statistics toolkit**: hypothesis tests, normality, correlation, chi-square, descriptive — powered by `scipy.stats`.
- **Declarative experiments** — one YAML per experiment, validated by typed models.
- **Reproducible** — global seeding, SQLite storage, resumable studies.
- **167 tests**, mkdocs-material docs site, CI on Python 3.10/3.11/3.12.

## Quick install

```bash
pip install -e ".[dev]"
```

## Quick start

### CLI

```bash
# Run a clustering study
containeer-optuna run config/experiments/clustering_optimization.yaml --n-trials 20

# Run a classification study
containeer-optuna run config/experiments/breast_cancer_classification.yaml --n-trials 30

# Explore the catalog
containeer-optuna list-models
containeer-optuna describe random_forest     # prints the model card (pros/cons)

# Statistical analysis
containeer-optuna stats describe iris
containeer-optuna stats anova iris_classification --group-by _target --feature sepal_length
containeer-optuna stats correlation diabetes --method pearson --threshold 0.3

# Optuna dashboard
containeer-optuna dashboard --storage sqlite:///clustering_optuna.db
```

### Python

```python
from containeer_optuna import load_config, OptunaRunner

cfg = load_config("config/experiments/clustering_optimization.yaml")
study = OptunaRunner(cfg).run(n_trials=20)

print("best silhouette:", study.best_value)
print("best params:", study.best_params)
```

Statistics:

```python
from containeer_optuna import two_sample_ttest, shapiro_test, describe

r = two_sample_ttest(group_a, group_b)
print(r.test_name, r.statistic, r.pvalue)
```

## What's included

| Component | Details |
|-----------|---------|
| Regression | Ridge, Lasso, OLS, ElasticNet, DecisionTree, RandomForest, GradientBoosting, SVR |
| Clustering | KMeans, DBSCAN, GMM, HDBSCAN, Agglomerative, Spectral, Birch, OPTICS |
| Classification | LogisticRegression, KNN, SVC, DecisionTreeClassifier |
| Reducers | PCA, UMAP, t-SNE, TruncatedSVD, FactorAnalysis |
| Scalers | StandardScaler, MinMaxScaler |
| Statistics | t-test, ANOVA, Mann-Whitney, Kruskal-Wallis, Shapiro, Anderson, Pearson/Spearman/Kendall, chi-square, descriptive |
| Metrics | R²/MSE/RMSE/MAE (regression); accuracy/F1/F1_weighted/roc_auc (classification); Silhouette/CH/DB (clustering) |
| Model-selection | Search across model families in one study (regression + clustering + classification) |
| Visualization | 2D embedding scatter, scree plots, `runner.plot_best_embedding()` |
| Model cards | 27 cards with pros/cons/when-to-use/assumptions/complexity |
| Tests | 167 passing |
| Docs | mkdocs-material (English) |

## Documentation

```bash
mkdocs serve
```

Highlights:

- [Getting Started](docs/getting_started/installation.md)
- [Model Cards](docs/model_cards/index.md) — pros/cons per model
- [Tutorials](docs/tutorials/regression_tutorial.md) — regression, clustering, classification, feature-set selection, statistics
- [Statistics](docs/user_guide/statistics.md) — hypothesis tests, normality, correlation
- [Roadmap](docs/roadmap.md) — milestones M0–M5 done, M6–M9 planned

## Notebooks & mirror scripts

The original Optuna experiments live under `notebooks/` and remain runnable standalone. Mirror `.py` scripts (extracted from the notebooks) live under `scripts/`.

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
├── src/containeer_optuna/      # the package (config, data, models, pipelines, optimization, evaluation, statistics, utils, cli)
├── config/                     # datasets.yaml, models.yaml, experiments/*.yaml
├── docs/                       # mkdocs-material site (English)
├── notebooks/                  # original Optuna experiments (runnable)
├── scripts/                    # mirror .py scripts extracted from notebooks
├── tests/                      # pytest suite (167 tests)
├── mkdocs.yml
├── pyproject.toml
└── .github/workflows/ci.yml
```

## Roadmap

Milestones M0–M5 are shipped. Later milestones add deep learning (M6–M7), NLP/AI (M8), and productionization (M9). See [Roadmap](docs/roadmap.md).

## License

[MIT](LICENSE) © Carlos Gomez
