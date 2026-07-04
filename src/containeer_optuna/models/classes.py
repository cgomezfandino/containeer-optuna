"""Mappings from model registry names to sklearn estimator classes.

Adding a new model only requires: (1) a new entry in ``config/models.yaml`` and
(2) a new line here. The :func:`~containeer_optuna.models.get_model` factory
and the generic :func:`~containeer_optuna.models.suggest_params` dispatcher
handle everything else.
"""

from __future__ import annotations

from sklearn.cluster import (
    DBSCAN,
    OPTICS,
    AgglomerativeClustering,
    Birch,
    KMeans,
    SpectralClustering,
)
from sklearn.decomposition import PCA, FactorAnalysis, TruncatedSVD
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# Regression estimators ---------------------------------------------------
REGRESSION_MODELS: dict[str, type] = {
    "ridge": Ridge,
    "lasso": Lasso,
    "ols": LinearRegression,
    # M1 — Regression maturity
    "elasticnet": ElasticNet,
    "decision_tree": DecisionTreeRegressor,
    "random_forest": RandomForestRegressor,
    "gradient_boosting": GradientBoostingRegressor,
    "svr": SVR,
}

# Classification estimators (M4) -----------------------------------------
CLASSIFICATION_MODELS: dict[str, type] = {
    "logistic_regression": LogisticRegression,
    "knn": KNeighborsClassifier,
    "svc": SVC,
    "decision_tree_classifier": DecisionTreeClassifier,
}

# Clustering estimators ---------------------------------------------------
# HDBSCAN is optional (added to public sklearn.cluster in 1.3); the value can
# be None until the lazy import below succeeds.
CLUSTERING_MODELS: dict[str, type | None] = {
    "kmeans": KMeans,
    "dbscan": DBSCAN,
    "gmm": GaussianMixture,
    # M2 — Clustering maturity
    "agglomerative": AgglomerativeClustering,
    "spectral": SpectralClustering,
    "birch": Birch,
    "optics": OPTICS,
    "hdbscan": None,  # populated lazily below — sklearn >= 1.3 required
}


# HDBSCAN is imported lazily so the package works on older sklearn (< 1.3).
# Accessing get_model("hdbscan") will raise a clear ImportError when unavailable.
try:
    from sklearn.cluster import HDBSCAN  # type: ignore[attr-defined]

    CLUSTERING_MODELS["hdbscan"] = HDBSCAN
except ImportError:  # pragma: no cover
    pass

# Dimensionality reducers -------------------------------------------------
# UMAP is optional (umap-learn may be absent); the value can be None until
# the lazy import below succeeds. The M3 reducers (t-SNE, TruncatedSVD,
# FactorAnalysis) are core sklearn and always available.
REDUCER_MODELS: dict[str, type | None] = {
    "pca": PCA,
    "umap": None,  # populated lazily below — umap-learn is optional at import
    # M3 — Dimensionality reduction
    "tsne": TSNE,
    "truncated_svd": TruncatedSVD,
    "factor_analysis": FactorAnalysis,
}

# Scalers -----------------------------------------------------------------
SCALER_MODELS: dict[str, type] = {
    "standard_scaler": StandardScaler,
    "minmax_scaler": MinMaxScaler,
}


# UMAP is imported lazily so the package can be imported in environments
# without ``umap-learn`` (e.g. minimal CI). Accessing ``get_model("umap")``
# will raise a clear ImportError only when actually requested.
try:
    from umap import UMAP  # type: ignore[import-not-found]

    REDUCER_MODELS["umap"] = UMAP
except ImportError:  # pragma: no cover
    pass


# Deep learning models (M6) — PyTorch MLP for tabular data.
# torch is optional ([dl] extra); entries are None until the lazy import below.
DL_MODELS: dict[str, type | None] = {
    "mlp_regressor": None,
    "mlp_classifier": None,
    # M7 — DL advanced
    "cnn_classifier": None,
    "rnn_classifier": None,
    # M8 — NLP
    "transformer_classifier": None,
}


try:
    from .dl import MLP as _MLP

    DL_MODELS["mlp_regressor"] = _MLP
    DL_MODELS["mlp_classifier"] = _MLP
    DL_MODELS["cnn_classifier"] = _MLP  # stub class — DL objective uses the backend
    DL_MODELS["rnn_classifier"] = _MLP
    DL_MODELS["transformer_classifier"] = _MLP  # M8 — NLP objective uses it directly
except ImportError:  # pragma: no cover — torch not installed
    pass


# Final merged registry. Values are estimator classes (UMAP filled in lazily
# above when umap-learn is available; otherwise None).
MODEL_CLASSES: dict[str, type | None] = {
    **REGRESSION_MODELS,
    **CLASSIFICATION_MODELS,
    **CLUSTERING_MODELS,
    **REDUCER_MODELS,
    **SCALER_MODELS,
    **DL_MODELS,
}


__all__ = [
    "REGRESSION_MODELS",
    "CLASSIFICATION_MODELS",
    "CLUSTERING_MODELS",
    "REDUCER_MODELS",
    "SCALER_MODELS",
    "MODEL_CLASSES",
]
