# containeer-optuna

**An Optuna-first data science framework.**

`containeer_optuna` is a modular framework for running Optuna hyperparameter
optimization experiments across regression, clustering, dimensionality
reduction, and (in later milestones) classification, statistics, and deep
learning. Configured via YAML, validated with pydantic, runnable from the CLI.

## Why use it

- **Declarative experiments** — one YAML file per experiment, validated by
  typed pydantic models.
- **Optuna-first** — every experiment is a study; samplers, pruners, storage,
  and visualization are first-class.
- **Reusable abstractions** — datasets, models, pipelines, and objectives are
  all registry-driven. Add a model by editing two lines.
- **Model cards** — every model ships with pros/cons, assumptions, and when-to-use,
  surfaced in docs and `containeer-optuna describe <model>`.
- **Reproducible** — global seeding, SQLite storage, `load_if_exists` resume.

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
containeer-optuna describe kmeans
```

## Roadmap (Agile milestones)

| Milestone | Scope | Status |
|-----------|-------|--------|
| **M0** Foundation | Package + CLI + tests + docs (ML: regression, clustering) | ✅ This release |
| M1 | Regression maturity (trees, ensembles, more metrics, feature selection) | Planned |
| M2 | Clustering maturity (HDBSCAN, Spectral, OPTICS) | Planned |
| M3 | Dimensionality reduction (t-SNE, TruncatedSVD) | Planned |
| M4 | Classification | Planned |
| M5 | Statistics (statsmodels, hypothesis tests) | Planned |
| M6 | Deep Learning (PyTorch + Optuna) | Planned |
| M7 | DL advanced (CNN, RNN/Transformers) | Planned |
| M8 | AI/NLP (HuggingFace fine-tuning) | Planned |
| M9 | Production (serialization, MLflow, PyPI release) | Planned |

See [Roadmap](roadmap.md) for details.
