# AGENTS.md - containeer-optuna

## State of the repo

The repo is now a **working package** (M0 — Foundation complete). The Python
package `containeer_optuna` is fully implemented, importable, tested, and
documented, alongside the original notebooks (which remain runnable
standalone).

1. **Python package `containeer_optuna`** (implemented) — `src/containeer_optuna/`
   contains working subpackages: `config`, `data`, `models`, `pipelines`,
   `optimization` (with `objectives/`), `evaluation`, `utils`, `cli`. Importing
   works: `from containeer_optuna import load_config, OptunaRunner`.
2. **Notebooks** (real, runnable) — `notebooks/optuna/*.ipynb` and
   `notebooks/customer_personality_analysis/model.ipynb`. Mirror `.py` scripts
   extracted from them live under `scripts/`.
3. **Docs** — mkdocs-material site under `docs/` (English); build with `mkdocs serve`.
4. **Tests** — pytest suite under `tests/` (57 tests, all passing).
5. **CI** — `.github/workflows/ci.yml` runs ruff + mypy + pytest + mkdocs build.

`from containeer_optuna import ...` works. The CLI `containeer-optuna` is
functional (`run`, `list-models`, `list-datasets`, `list-experiments`,
`describe`, `init`, `dashboard`).

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
containeer-optuna describe kmeans
containeer-optuna list-models --type clustering

# Optuna dashboard — storage filenames the experiments use:
optuna-dashboard sqlite:///clustering_optuna.db   # clustering_optimization
optuna-dashboard sqlite:///optuna_studies.db      # auto_mpg_* (regression)
```

## Package layout

```
src/containeer_optuna/
  __init__.py            # public API (36 symbols)
  config.py              # pydantic config + YAML loaders (Settings uses CO_ env prefix)
  data/                  # BaseDataset, YamlDatasetLoader, get_dataset
  models/
    classes.py           # name → sklearn class mappings
    registry.py          # BaseModel, get_model, suggest_params (param_space dispatcher)
  pipelines/pipelines.py # BasePipeline, get_pipeline (scaler? → reducer? → model)
  optimization/
    runner.py            # OptunaRunner (sampler + pruner + storage + study)
    objectives/factories.py  # regression/clustering objective factories
  evaluation/
    metrics.py           # regression_metrics, clustering_metrics
    model_cards.py       # ModelCard registry (pros/cons for every model)
  utils/utils.py         # seed_all, get_logger, io helpers
  cli/cli.py             # typer app
config/
  datasets.yaml          # auto_mpg, iris, customer_personality, credit_card
  models.yaml            # ridge, lasso, ols, kmeans, dbscan, gmm, pca, umap, scalers
  experiments/           # clustering_optimization, auto_mpg_model_selection, auto_mpg_feature_set_selection
```

## Public API (re-exported from `containeer_optuna`)

`load_config`, `ExperimentConfig`, `CVConfig`, `OptimizationConfig`,
`DatasetConfig`, `ModelConfig`, `load_dataset_config`, `load_model_config`,
`get_experiment_configs`, `settings`, `BaseDataset`, `get_dataset`,
`BaseModel`, `get_model`, `suggest_params`, `BasePipeline`, `get_pipeline`,
`OptunaRunner`, `make_regression_objective`, `make_clustering_objective`,
`regression_metrics`, `clustering_metrics`, `ModelCard`, `get_model_card`,
`all_model_cards`, `seed_all`, `get_logger`, `save_json`, `study_summary`.

## Conventions worth knowing

- Storage is SQLite; two files: `optuna_studies.db` (regression) and
  `clustering_optuna.db` (clustering). Both gitignored under `notebooks/optuna/`.
- All studies default to `direction="maximize"` (R² for regression, Silhouette
  for clustering). CH + DB stored as `user_attrs` for clustering.
- `random_state=42` by default; overridable per experiment.
- Cross-validation: `ShuffleSplit` for regression, `KFold` for clustering
  (auto-selected by the task type via a model_validator on `ExperimentConfig`).
- Package uses `src/` layout (`[tool.setuptools.packages.find] where = ["src"]`),
  so editable install is `pip install -e .`.
- Python ≥3.9 required; mypy `python_version=3.10`.
- The clustering objective fixes two latent bugs from `perplexity.ipynb`:
  (1) full pipeline applied end-to-end per fold (notebook re-fit the clusterer
  on `X_test` bypassing scaler/reducer); (2) DBSCAN noise (`-1`) stripped
  before scoring.

## Adding to the package

Adding a new model takes three steps:
1. `config/models.yaml` — add entry with `type`, `default_params`, `param_space`.
2. `src/containeer_optuna/models/classes.py` — map the name to the sklearn class.
3. `src/containeer_optuna/evaluation/model_cards.py` — add a `ModelCard` (pros/cons).
4. Regenerate model card docs (run the snippet in `docs/model_cards/`).

The `get_model` + `suggest_params` dispatcher handles the Optuna sampling
generically from `param_space` (types: `float`/`int`/`categorical` with
`low`/`high`/`log`/`choices`). No objective code changes needed.

## Roadmap

M0 (Foundation) is complete. Later milestones: M1 (regression maturity),
M2 (clustering maturity), M3 (t-SNE / dimensionality reduction), M4
(classification), M5 (statistics), M6–M7 (deep learning), M8 (NLP/AI),
M9 (productionization). See `docs/roadmap.md`.
