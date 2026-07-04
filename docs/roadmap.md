# Roadmap

The framework is built incrementally (Agile milestones). Each milestone ships
a working, tested, documented increment.

## M0 — Foundation ✅ (this release)

Package importable from `src/`, CLI (`containeer-optuna`), YAML-driven
experiments, regression (Ridge/Lasso/OLS) + clustering (KMeans/DBSCAN/GMM) +
reducers (PCA/UMAP) + scalers (Standard/MinMax), Optuna integration with
samplers/pruners, model cards (pros/cons), 57 tests, mkdocs site.

## M1 — Regression maturity ✅ (current release)

ElasticNet, DecisionTree, RandomForest, GradientBoosting, SVR. Pluggable
metrics (MSE/MAE/RMSE/MAPE). Optuna feature-set selection. Model-selection-as-
categorical studies (one study across model families).

## M2 — Clustering maturity ✅ (current release)

HDBSCAN, Agglomerative, Spectral, Birch, OPTICS. Robust noise handling.
Cluster stability metric.

## M3 — Dimensionality reduction (planned)

t-SNE (real perplexity — the original `perplexity.ipynb` is misnamed; it uses
UMAP, not t-SNE). TruncatedSVD, FactorAnalysis. Visualization utilities.

## M4 — Classification (planned)

New task type. LogReg / KNN / SVC / trees. Accuracy, F1, AUC. StratifiedKFold.

## M5 — Statistics (planned)

statsmodels integration. Hypothesis tests. GLM. ANOVA.

## M6 — Deep Learning foundation (planned)

PyTorch + Optuna (MLP for tabular). `[dl]` extra. Device handling. Epoch
pruning (report intermediate values).

## M7 — DL advanced (planned)

CNN (vision), RNN/Transformer (sequences).

## M8 — AI/NLP (planned)

HuggingFace fine-tuning with Optuna. Embeddings.

## M9 — Productionization (planned)

Model serialization (joblib/onnx). Optional MLflow tracking. Distributed
studies (Dask). PyPI release. Cookiecutter template.
