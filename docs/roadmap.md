# Roadmap

The framework is built incrementally (Agile milestones). Each milestone ships
a working, tested, documented increment. M0–M5 are shipped (version 0.6.0);
27 models, 167 tests, 56 public symbols.

## M0 — Foundation ✅

Package importable from `src/`, CLI (`containeer-optuna`), YAML-driven
experiments, regression (Ridge/Lasso/OLS) + clustering (KMeans/DBSCAN/GMM) +
reducers (PCA/UMAP) + scalers (Standard/MinMax), Optuna integration with
samplers/pruners, model cards (pros/cons), mkdocs site.

## M1 — Regression maturity ✅

ElasticNet, DecisionTree, RandomForest, GradientBoosting, SVR. Pluggable
metrics (r2/mse/rmse/mae). Optuna feature-set selection. Model-selection-as-
categorical studies (one study across model families). Bundled diabetes
dataset (source: sklearn).

## M2 — Clustering maturity ✅

HDBSCAN, Agglomerative, Spectral, Birch, OPTICS. Robust noise handling
(DBSCAN fix — the clustering objective now uses `fit_predict` per fold).
Model-selection-as-categorical extended to clustering.

## M3 — Dimensionality reduction ✅

t-SNE, TruncatedSVD, FactorAnalysis. Visualization utilities (`plot_embedding_2d`,
`plot_scree`, `runner.plot_best_embedding()`). t-SNE documented as
visualization-only (no `transform()`, cannot be a YAML Pipeline reducer).

## M4 — Classification ✅

LogisticRegression, KNN, SVC, DecisionTreeClassifier. Classification metrics
(accuracy, f1, f1_weighted, roc_auc, roc_auc_ovr). StratifiedKFold default.
3 bundled classification datasets (breast_cancer, wine, iris_classification).
Model-selection extended to classification.

## M5 — Statistics ✅

`statistics/` subpackage with hypothesis tests (ttest, Mann-Whitney, ANOVA,
Kruskal-Wallis), normality (Shapiro, D'Agostino, Anderson-Darling),
correlation (Pearson, Spearman, Kendall, matrix), chi-square, descriptive
statistics — all powered by `scipy.stats` (zero new dependencies). Uniform
`StatResult` return type. `containeer-optuna stats` CLI subcommand group.

## M6 — Deep Learning foundation ✅

PyTorch + Optuna (MLP for tabular). `[dl]` extra. Device handling. Epoch
pruning (report intermediate values).

## M7 — DL advanced ✅

CNN (vision), RNN/Transformer (sequences).

## M8 — AI/NLP (planned)

HuggingFace fine-tuning with Optuna. Embeddings.

## M9 — Productionization (planned)

Model serialization (joblib/onnx). Optional MLflow tracking. Distributed
studies (Dask). PyPI release. Cookiecutter template.
