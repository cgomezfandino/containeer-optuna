# Clustering pipeline tutorial

Reproduces `notebooks/optuna/perplexity.ipynb` (joint optimization of scaler →
reducer → clusterer) using the framework.

## Run

```bash
containeer-optuna run config/experiments/clustering_optimization.yaml --n-trials 30
```

## What it does

- Loads Iris (bundled, no download).
- Builds a `StandardScaler → PCA → KMeans` pipeline (all three tunable).
- KFold(3) over rows for cluster stability.
- Maximizes Silhouette; stores Calinski-Harabasz and Davies-Bouldin as
  `user_attrs`.
- Strips DBSCAN noise before scoring (the notebook did not).

## Bug fixes vs the notebook

1. The full pipeline is applied end-to-end on each fold (the notebook re-fit
   the clusterer directly on the test fold, bypassing scaler/reducer).
2. DBSCAN noise (`-1` labels) is stripped before computing metrics.

## Swap the clusterer

Edit `config/experiments/clustering_optimization.yaml`:

```yaml
model: dbscan   # or gmm
```

The framework's clustering objective handles noise stripping and degenerate
folds automatically.
