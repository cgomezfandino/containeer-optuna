"""
Mirror script for notebooks/optuna/perplexity.ipynb

Auto-generated from the notebook code cells.
See the corresponding tutorial in docs/tutorials/ for context.
Original notebook: 0 markdown cells, 4 code cells.
"""

import optuna
import numpy as np
import pandas as pd

from sklearn.datasets import load_iris   # Example dataset; replace with your own
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.decomposition import PCA
import umap

from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


# -------------------------------
# Data Preparation
# -------------------------------
# Example dataset: Iris (replace with your dataset of choice)
X, _ = load_iris(return_X_y=True)

# %%

# -------------------------------
# Define Scalers
# -------------------------------
scalers = {
    "none": None,
    "standard": StandardScaler(),
    "minmax": MinMaxScaler()
}


# -------------------------------
# Define Clustering Models
# -------------------------------
def get_kmeans(trial):
    return KMeans(
        n_clusters=trial.suggest_int("kmeans_n_clusters", 2, 10),
        init=trial.suggest_categorical("kmeans_init", ["k-means++", "random"]),
        algorithm=trial.suggest_categorical("kmeans_algorithm", ["lloyd", "elkan"]),
        n_init=10,
        random_state=42
    )

def get_dbscan(trial):
    return DBSCAN(
        eps=trial.suggest_float("dbscan_eps", 0.1, 5.0, log=True),
        min_samples=trial.suggest_int("dbscan_min_samples", 3, 10)
    )

def get_gmm(trial):
    return GaussianMixture(
        n_components=trial.suggest_int("gmm_n_components", 2, 10),
        covariance_type=trial.suggest_categorical("gmm_covariance_type", ["full", "tied", "diag", "spherical"]),
        n_init=3,
        random_state=42
    )

clusterers = {
    "kmeans": get_kmeans,
    "dbscan": get_dbscan,
    "gmm": get_gmm
}


# -------------------------------
# Define Dimensionality Reduction
# -------------------------------
def get_pca(trial, n_features):
    return PCA(n_components=trial.suggest_int("pca_n_components", 2, min(n_features - 1, 15)))

def get_umap(trial):
    return umap.UMAP(
        n_neighbors=trial.suggest_int("umap_n_neighbors", 5, 15),
        min_dist=trial.suggest_float("umap_min_dist", 0.0, 1.0),
        n_components=trial.suggest_int("umap_n_components", 2, 10),
        random_state=42
    )

reducers = {
    "none": None,
    "pca": get_pca,
    "umap": get_umap
}


# -------------------------------
# Objective Function
# -------------------------------
def objective(trial):
    # Choose scaler
    scaler_name = trial.suggest_categorical("scaler", list(scalers.keys()))
    scaler = scalers[scaler_name]

    # Choose reducer
    reducer_name = trial.suggest_categorical("reducer", list(reducers.keys()))
    if reducer_name == "pca":
        reducer = get_pca(trial, X.shape[1])
    elif reducer_name == "umap":
        reducer = get_umap(trial)
    else:
        reducer = None

    # Choose clusterer
    clusterer_name = trial.suggest_categorical("clusterer", list(clusterers.keys()))
    clusterer = clusterers[clusterer_name](trial)

    # Build pipeline
    steps = []
    if scaler is not None:
        steps.append(("scaler", scaler))
    if reducer is not None:
        steps.append(("reducer", reducer))
    steps.append(("clusterer", clusterer))

    pipeline = Pipeline(steps)

    # Cross-validation for clustering stability
    kf = KFold(n_splits=3, shuffle=True, random_state=42)

    silhouette_scores, ch_scores, db_scores = [], [], []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]

        try:
            # Fit clustering model
            pipeline.fit(X_train)

            # Get labels (pipeline's last step)
            labels = pipeline["clusterer"].fit_predict(X_test)

            # Exclude invalid clustering (e.g. only one cluster found)
            if len(set(labels)) < 2:
                continue

            # Metrics
            sil = silhouette_score(X_test, labels)
            ch = calinski_harabasz_score(X_test, labels)
            db = davies_bouldin_score(X_test, labels)

            silhouette_scores.append(sil)
            ch_scores.append(ch)
            db_scores.append(db)

        except Exception as e:
            # Handle errors (e.g., poor hyperparameters)
            return -1.0

    # Mean scores across folds
    if len(silhouette_scores) == 0:
        return -1.0
    
    mean_silhouette = np.mean(silhouette_scores)
    mean_ch = np.mean(ch_scores)
    mean_db = np.mean(db_scores)

    # Track secondary metrics
    trial.set_user_attr("calinski_harabasz", mean_ch)
    trial.set_user_attr("davies_bouldin", mean_db)

    return mean_silhouette

# %%
study = optuna.create_study(
    direction="maximize",
    study_name="clustering_optimization",
    storage="sqlite:///clustering_optuna.db",
    load_if_exists=True
)

study.optimize(objective, n_trials=50)

# Results
print("\nBest trial:")
trial = study.best_trial
print(f"Silhouette: {trial.value}")
print("Hyperparameters:")
for k, v in trial.params.items():
    print(f"  {k}: {v}")
print("Secondary metrics:")
print("  Calinski-Harabasz:", trial.user_attrs["calinski_harabasz"])
print("  Davies-Bouldin:", trial.user_attrs["davies_bouldin"])

# %%
import optuna.visualization as vis

# After the study.optimize(...) call, run the following:

# 1. Plot optimization history: objective value (silhouette score) over trials
fig1 = vis.plot_optimization_history(study)
fig1.show()

# 2. Plot parameter importance: which hyperparameters affect silhouette the most
fig2 = vis.plot_param_importances(study)
fig2.show()

# 3. Parallel coordinate plot: explore interactions between hyperparameters and objective
fig3 = vis.plot_parallel_coordinate(study)
fig3.show()

# 4. Slice plot: effect of individual parameters on optimization
fig4 = vis.plot_slice(study)
fig4.show()

