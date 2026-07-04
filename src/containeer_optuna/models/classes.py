"""Mappings from model registry names to sklearn estimator classes.

Adding a new model only requires: (1) a new entry in ``config/models.yaml`` and
(2) a new line here. The :func:`~containeer_optuna.models.get_model` factory
and the generic :func:`~containeer_optuna.models.suggest_params` dispatcher
handle everything else.
"""

from __future__ import annotations

from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Regression estimators ---------------------------------------------------
REGRESSION_MODELS: dict[str, type] = {
    "ridge": Ridge,
    "lasso": Lasso,
    "ols": LinearRegression,
}

# Clustering estimators ---------------------------------------------------
CLUSTERING_MODELS: dict[str, type] = {
    "kmeans": KMeans,
    "dbscan": DBSCAN,
    "gmm": GaussianMixture,
}

# Dimensionality reducers -------------------------------------------------
# UMAP is optional (umap-learn may be absent); the value can be None until
# the lazy import below succeeds.
REDUCER_MODELS: dict[str, type | None] = {
    "pca": PCA,
    "umap": None,  # populated lazily below — umap-learn is optional at import
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


# Final merged registry. Values are estimator classes (UMAP filled in lazily
# above when umap-learn is available; otherwise None).
MODEL_CLASSES: dict[str, type | None] = {
    **REGRESSION_MODELS,
    **CLUSTERING_MODELS,
    **REDUCER_MODELS,
    **SCALER_MODELS,
}


__all__ = [
    "REGRESSION_MODELS",
    "CLUSTERING_MODELS",
    "REDUCER_MODELS",
    "SCALER_MODELS",
    "MODEL_CLASSES",
]
