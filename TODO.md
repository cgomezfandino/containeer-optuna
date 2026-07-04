# TODO — containeer-optuna v1.0.0

## ✅ Completed (M0–M9, full roadmap)

### M0 — Foundation
- [x] Package importable from `src/` (config, data, models, pipelines, optimization, evaluation, utils, cli)
- [x] CLI: `run`, `list-models`, `list-datasets`, `list-experiments`, `describe`, `init`, `dashboard`
- [x] YAML-driven experiments validated by pydantic
- [x] Optuna integration with samplers (TPE, Random, CMA-ES, NSGA-II) and pruners
- [x] Model cards (pros/cons/when-to-use/assumptions)
- [x] Tests, docs (mkdocs-material), CI (GitHub Actions), pre-commit, LICENSE

### M1 — Regression Maturity
- [x] ElasticNet, DecisionTree, RandomForest, GradientBoosting, SVR
- [x] Pluggable regression metrics (r2/mse/rmse/mae)
- [x] Feature-set selection (Optuna searches over named feature subsets)
- [x] Bundled diabetes dataset (source: sklearn, no download)

### M2 — Clustering Maturity
- [x] HDBSCAN, Agglomerative, Spectral, Birch, OPTICS
- [x] Model-selection-as-categorical (`models: [...]` in YAML)
- [x] DBSCAN bug fix (clustering objective now uses `fit_predict` per fold)

### M3 — Dimensionality Reduction
- [x] t-SNE, TruncatedSVD, FactorAnalysis
- [x] Visualization: `plot_embedding_2d`, `plot_scree`, `runner.plot_best_embedding()`
- [x] t-SNE documented as visualization-only (no `transform()`, cannot be a Pipeline reducer)

### M4 — Classification
- [x] LogisticRegression, KNN, SVC, DecisionTreeClassifier
- [x] Classification metrics (accuracy, f1, f1_weighted, roc_auc, roc_auc_ovr)
- [x] StratifiedKFold auto-default for classification
- [x] 3 bundled datasets (breast_cancer, wine, iris_classification)
- [x] Model-selection extended to classification

### M5 — Statistics
- [x] Hypothesis tests (ttest, Mann-Whitney, ANOVA, Kruskal-Wallis)
- [x] Normality tests (Shapiro, D'Agostino, Anderson-Darling)
- [x] Correlation (Pearson, Spearman, Kendall, matrix)
- [x] Chi-square, descriptive statistics
- [x] Uniform `StatResult` return type
- [x] CLI `stats` subcommand group
- [x] Zero new dependencies (scipy.stats already required)

### M6 — Deep Learning Foundation
- [x] PyTorch MLP for tabular regression + classification
- [x] Custom DL objective with epoch pruning (trial.report + should_prune)
- [x] Lazy torch import; `[dl]` optional extra

### M7 — DL Advanced
- [x] CNN classifier (image classification, Conv2d → ReLU → MaxPool → FC)
- [x] RNN/GRU/LSTM classifier (sequence classification)
- [x] Backend abstraction (MLPBackend, CNNBackend, RNNBackend — shared training loop)
- [x] `source: torchvision` (MNIST, CIFAR10, FashionMNIST)

### M8 — NLP/AI
- [x] Transformer text classification (DistilBERT/BERT fine-tuning)
- [x] Dedicated `make_nlp_objective` (tokenization, AdamW, warmup scheduler, epoch pruning)
- [x] `source: huggingface` (AG_NEWS, IMDB, SST2)
- [x] `[nlp]` optional extra (transformers, datasets, tokenizers, accelerate)

### M9 — Productionization
- [x] Model serialization (`save_model: true` → `.joblib` via joblib)
- [x] `fit_best_pipeline()` method (reconstructs + fits best-trial sklearn pipeline)
- [x] Prediction saving (`save_predictions: true` → `.npy`)
- [x] MLflow tracking (optional `[mlflow]` extra)
- [x] `python -m containeer_optuna` support (`__main__.py`)
- [x] PyPI release metadata (`[project.urls]`, classifier bumped to Beta)
- [x] Publish workflow (auto-publish to PyPI on tag)
- [x] Version: **1.0.0**

---

## 📋 Current stats (v1.0.0)

| Metric | Value |
|--------|-------|
| Version | 1.0.0 |
| Models | 32 |
| Model cards | 32 |
| Tests | 192 |
| Public symbols | 57 |
| Subpackages | 10 (config, data, models+dl, pipelines, optimization, evaluation, statistics, utils, cli) |
| Optional extras | `[dev]`, `[dl]`, `[nlp]`, `[mlflow]` |
| Dataset sources | local, kaggle, sklearn, torchvision, huggingface |
| Paradigms | ML clásico, Estadística, Deep Learning (MLP/CNN/RNN), NLP (Transformers) |
| CI | Python 3.10/3.11/3.12 |
| PRs merged | 11 |

---

## 🔮 Future enhancements (post-v1.0.0, not committed)

### Short-term polish
- [ ] **Dask distributed studies** — Optuna already supports `n_jobs` + RDBMS storage; Dask adds a distributed scheduler. Low priority given native Optuna parallelism.
- [ ] **Cookiecutter template** — separate repo for scaffolding new experiments/projects.
- [ ] **ONNX export** — for sklearn models, `skl2onnx` conversion in addition to joblib.
- [ ] **HyperbandPruner defaults for DL** — auto-set `pruner: hyperband` for DL experiments if not specified.
- [ ] **Python 3.13** — add to CI matrix + classifiers once ecosystem catches up.

### Medium-term features
- [ ] **Gradient accumulation + fp16** for DL/NLP objectives (larger effective batch size on limited hardware).
- [ ] **HF Trainer API** integration as an alternative to the manual training loop in `make_nlp_objective`.
- [ ] **Embedding extraction** — standalone utility to extract embeddings from pre-trained models (no fine-tuning).
- [ ] **More NLP tasks** — NER (token classification), QA (question answering), text generation.
- [ ] **Time series** — forecasting models (ARIMA, Prophet, LSTM-forecasting) as a new task type.
- [ ] **Ensemble methods** — stacking/blendng of best models from model-selection studies.

### Infrastructure
- [ ] **GitHub Pages** deployment for mkdocs site (auto-deploy on push to main).
- [ ] **Benchmark suite** — automated performance regression tests on bundled datasets.
- [ ] **Type stubs** — ship `.pyi` files for better IDE autocomplete.
- [ ] **Interactive dashboard** — extend `optuna-dashboard` with custom visualizations from the framework.

---

## 🏗️ Architecture notes for future contributors

### Key design decisions (do NOT refactor without careful consideration)
1. **Clustering objective uses `fit_predict` per fold** (not `fit` + `predict`). This is critical: DBSCAN/HDBSCAN/Agglomerative/Spectral/OPTICS don't implement `predict()`.
2. **t-SNE cannot be a YAML Pipeline reducer** — sklearn Pipeline validates `transform()` on intermediate steps. t-SNE only works via `plot_best_embedding()` (manual step application).
3. **DL models bypass `cross_validate`** — the DL objective runs a manual training loop to enable epoch pruning. The DL backend abstraction (`MLPBackend`/`CNNBackend`/`RNNBackend`) shares the loop but isolates data prep + module construction.
4. **NLP models use a separate objective** (`make_nlp_objective`), NOT the DL backend. Transformers need tokenization + attention masks + AdamW, structurally different from tensor-in/logits-out.
5. **`"None"` sentinel in categorical param_spaces** is coerced to Python `None` by `get_model` (for optional params like `max_depth`). Don't remove this coercion.
6. **`random_state` is filtered** by `get_pipeline` to only kwargs the estimator accepts (KNN/DBSCAN/Agglomerative don't accept `random_state`).

### Adding a new model (4 steps)
1. `config/models.yaml` — add entry with `type`, `default_params`, `param_space`.
2. `src/containeer_optuna/models/classes.py` — map the name to the sklearn/PyTorch class.
3. `src/containeer_optuna/evaluation/model_cards.py` — add a `ModelCard` (pros/cons).
4. Add a test that it instantiates.

### Adding a new task type
1. Extend `TaskType` in `config.py`.
2. Add an objective factory in `optimization/objectives/`.
3. Add a dispatch branch in `runner._build_objective`.
4. Add CV strategy defaults in the `ExperimentConfig` model validator.
