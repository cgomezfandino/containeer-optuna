# containeer-optuna

**An Optuna-first data science framework.**

`containeer_optuna` is a modular framework for running Optuna hyperparameter
optimization experiments across regression, clustering, classification, and
dimensionality reduction — plus a full statistics toolkit for EDA. Configured
via YAML, validated with pydantic, runnable from the CLI.

## Why use it

- **27 models** across regression (8), clustering (8), classification (4),
  reducers (5), scalers (2) — each with a model card (pros/cons/when-to-use).
- **3 tasks** with pluggable metrics and auto-derived CV strategies.
- **Model-selection-as-categorical** — one study searches across model families.
- **Statistics toolkit** — hypothesis tests, normality, correlation, chi-square,
  descriptive — powered by `scipy.stats` (zero new deps).
- **Declarative experiments** — one YAML per experiment, validated by typed models.
- **Reproducible** — global seeding, SQLite storage, resumable studies.

## Quick install

```bash
pip install -e ".[dev]"
```

## Quick start

```python
from containeer_optuna import load_config, OptunaRunner

cfg = load_config("config/experiments/clustering_optimization.yaml")
study = OptunaRunner(cfg).run(n_trials=20)
print(study.best_value, study.best_params)
```

Or from the CLI:

```bash
containeer-optuna run config/experiments/clustering_optimization.yaml --n-trials 20
containeer-optuna describe random_forest
containeer-optuna stats describe iris
```

## Roadmap (Agile milestones)

| Milestone | Scope | Status |
|-----------|-------|--------|
| **M0** Foundation | Package + CLI + tests + docs (regression, clustering) | ✅ |
| **M1** Regression | Trees, ensembles, SVR, pluggable metrics, feature-set selection | ✅ |
| **M2** Clustering | HDBSCAN, Agglomerative, Spectral, Birch, OPTICS, model-selection | ✅ |
| **M3** Dim. Reduction | t-SNE, TruncatedSVD, FactorAnalysis, visualization | ✅ |
| **M4** Classification | LogReg, KNN, SVC, DT, accuracy/F1/AUC, StratifiedKFold | ✅ |
| **M5** Statistics | Hypothesis tests, normality, correlation, chi-square (scipy.stats) | ✅ |
| M6 | Deep Learning (PyTorch + Optuna) | Planned |
| M7 | DL advanced (CNN, RNN/Transformers) | Planned |
| M8 | AI/NLP (HuggingFace fine-tuning) | Planned |
| M9 | Production (serialization, MLflow, PyPI release) | Planned |

See [Roadmap](roadmap.md) for details.
