# Dimensionality reduction

The framework supports 5 reducers. This tutorial explains when to use each
and the important t-SNE caveat.

## The 5 reducers

| Reducer | Linear? | Has transform? | Use as YAML reducer? | Notes |
|---------|---------|----------------|----------------------|-------|
| `pca` | Yes | Yes | ✅ Yes | Default for dense data. |
| `truncated_svd` | Yes | Yes | ✅ Yes | For sparse data (TF-IDF). |
| `factor_analysis` | Yes | Yes | ✅ Yes | Generative latent factors. |
| `umap` | No | Yes | ✅ Yes | Non-linear, preserves neighborhoods. |
| `tsne` | No | **No** | ❌ **No** | Visualization only (via plot_best_embedding). |

## ⚠️ t-SNE caveat

t-SNE does NOT implement `transform()`. sklearn's `Pipeline` requires all
intermediate steps to implement `transform`, so **t-SNE cannot be used as a
YAML reducer** in an experiment pipeline. Use it via `plot_best_embedding()`,
which applies steps manually via `fit_transform` (bypassing the Pipeline
validation):

```python
runner = OptunaRunner(cfg)
runner.run(n_trials=20)
fig = runner.plot_best_embedding()  # works with t-SNE
```

For a YAML reducer that captures non-linear structure, use **UMAP** instead
(it has `transform` and works in pipelines).

## Choosing n_components

Use `plot_scree` to pick `n_components` for PCA/TruncatedSVD:

```python
from containeer_optuna import plot_scree
from sklearn.decomposition import PCA

pca = PCA().fit(X_scaled)
fig = plot_scree(pca.explained_variance_ratio_)
# Look for the elbow or the 95% cumulative threshold.
```

## Example: TruncatedSVD clustering

```bash
containeer-optuna run config/experiments/clustering_truncated_svd.yaml --n-trials 30
```

## Visualizing the best embedding

After any clustering study with a reducer, visualize the result:

```python
from containeer_optuna import load_config, OptunaRunner

cfg = load_config("config/experiments/clustering_optimization.yaml")
runner = OptunaRunner(cfg)
runner.run(n_trials=30)
fig = runner.plot_best_embedding()
fig.savefig("best_embedding.png", dpi=150)
```
