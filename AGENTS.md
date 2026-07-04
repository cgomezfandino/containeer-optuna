# AGENTS.md - containeer-optuna

## State of the repo

The Python package `containeer_optuna` is **fully implemented** (M0–M5, version 0.6.0). It covers regression (8 models), clustering (8 models), classification (4 models), dimensionality reduction (5 reducers), 2 scalers, a statistics toolkit (scipy.stats-based), model-selection-as-categorical, and visualization utilities.

Importing works: `from containeer_optuna import load_config, OptunaRunner, two_sample_ttest`.
56 public symbols, 27 model cards, 167 tests, all passing.

## Commands

```bash
# Install (editable + dev tooling)
pip install -e ".[dev]"

# Quality gates (run before commits)
ruff check src tests
ruff format --check src tests
mypy src --ignore-missing-imports
pytest -q
mkdocs build --strict

# CLI
containeer-optuna run config/experiments/clustering_optimization.yaml --n-trials 20
containeer-optuna describe random_forest
containeer-optuna list-models --type clustering
containeer-optuna stats describe iris
containeer-optuna stats anova iris_classification --group-by _target --feature sepal_length

# Optuna dashboard
optuna-dashboard sqlite:///clustering_optuna.db   # clustering studies
optuna-dashboard sqlite:///optuna_studies.db      # regression + classification studies
```

## Package layout

```
src/containeer_optuna/
  __init__.py            # public API (56 symbols)
  config.py              # pydantic config + YAML loaders (Settings uses CO_ env prefix)
  data/                  # BaseDataset, YamlDatasetLoader, get_dataset (source: local/kaggle/sklearn)
  models/
    classes.py           # name → sklearn class mappings (REGRESSION/CLASSIFICATION/CLUSTERING/REDUCER/SCALER)
    registry.py          # BaseModel, get_model, suggest_params (param_space dispatcher)
  pipelines/pipelines.py # BasePipeline, get_pipeline (scaler? → reducer? → model)
  optimization/
    runner.py            # OptunaRunner (sampler + pruner + storage + study + plot_best_embedding)
    objectives/factories.py  # regression/classification/clustering/model-selection objectives
  evaluation/
    metrics.py           # regression/clustering/classification metrics + scorers
    model_cards.py       # 27 ModelCards (pros/cons)
    plotting.py          # plot_embedding_2d, plot_scree
  statistics/            # M5: hypothesis tests, normality, correlation, chi-square, descriptive
    types.py             # StatResult dataclass
    hypothesis.py, normality.py, correlation.py, categorical.py, descriptive.py
  utils/utils.py         # seed_all, get_logger, io helpers
  cli/cli.py             # typer app + stats subcommand group
config/
  datasets.yaml          # 8 datasets (auto_mpg, iris, diabetes, breast_cancer, wine, iris_classification, ...)
  models.yaml            # 27 models (regression, classification, clustering, reducer, scaler)
  experiments/           # 10 runnable experiment YAMLs
```

## Three task types

- **regression** — `ShuffleSplit` default; metrics `r2`/`mse`/`rmse`/`mae`; supports `feature_sets` selection.
- **clustering** — `KFold` default; objective uses `fit_predict` per fold (fixes DBSCAN bug); metrics Silhouette/CH/DB.
- **classification** — `StratifiedKFold` default; metrics `accuracy`/`f1`/`f1_weighted`/`roc_auc`/`roc_auc_ovr`.

All three support **model-selection-as-categorical** via the `models: list[str]` field.

## Statistics (M5)

Standalone utilities (NOT Optuna objectives). Return a uniform `StatResult(test_name, statistic, pvalue, extra)`. Powered by `scipy.stats` — zero new dependencies. Available as `from containeer_optuna import two_sample_ttest, ...` and via the `containeer-optuna stats` CLI group.

## Key conventions

- Storage is SQLite; two files: `optuna_studies.db` (regression + classification) and `clustering_optuna.db` (clustering).
- All studies default to `direction="maximize"` unless a metric overrides (e.g. `mse`→minimize).
- `random_state=42` by default; overridable per experiment.
- Package uses `src/` layout; editable install is `pip install -e .`.
- Python ≥3.9 required; CI runs on 3.10/3.11/3.12.
- The clustering objective uses `pipeline.fit_predict(X_fold)` per fold (not `fit`+`predict`), which correctly handles DBSCAN/HDBSCAN/Agglomerative/Spectral/OPTICS (no `transform`/`predict`).
- t-SNE is a visualization-only reducer: it lacks `transform()` so it cannot be a YAML reducer in a sklearn Pipeline. Use it via `runner.plot_best_embedding()` instead.

## Adding a new model

1. `config/models.yaml` — add entry with `type`, `default_params`, `param_space`.
2. `src/containeer_optuna/models/classes.py` — map the name to the sklearn class.
3. `src/containeer_optuna/evaluation/model_cards.py` — add a `ModelCard` (pros/cons).
4. Add a test that it instantiates.

The `get_model` + `suggest_params` dispatcher handles the Optuna sampling generically from `param_space` (types: `float`/`int`/`categorical` with `low`/`high`/`log`/`choices`). The `"None"` sentinel string in categoricals is coerced to Python `None` (for optional params like `max_depth`).

## Roadmap

M0–M5 are complete. Later milestones: M6–M7 (deep learning with PyTorch), M8 (NLP/AI), M9 (productionization). See `docs/roadmap.md`.
