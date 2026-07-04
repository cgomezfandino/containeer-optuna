# Clustering models

## `dbscan`

**Density-based clustering; finds arbitrarily-shaped clusters and noise.**

**When to use:** Spatial/point-cloud data where clusters may be non-convex and you want outliers flagged as noise. No need to specify k in advance.

**Pros:**
- ✅ Does NOT require the number of clusters a priori.
- ✅ Discovers clusters of arbitrary shape.
- ✅ Robust to outliers (labels them -1).
- ✅ Well-suited to spatial / density-varying data.

**Cons:**
- ❌ Struggles when clusters have very different densities.
- ❌ Sensitive to eps and min_samples — needs careful tuning (Optuna helps).
- ❌ Distance metric and scaling strongly affect results.
- ❌ Noise points must be stripped before computing Silhouette/CH/DB.

**Assumptions:** Clusters are dense regions separated by sparse regions; Features scaled.

**Complexity:** O(n_samples * log n_samples) with a spatial index, O(n^2) worst case

**Key hyperparameters:** `eps`, `min_samples`

## `gmm`

**Gaussian Mixture Model — soft (probabilistic) clustering.**

**When to use:** When you want cluster membership probabilities (soft assignments), or clusters may be elliptical (covariance-aware). Useful for density estimation.

**Pros:**
- ✅ Soft assignments — gives P(cluster | point).
- ✅ Models elliptical clusters via per-component covariance.
- ✅ Provides a generative model (likelihood-based).
- ✅ covariance_type tunes cluster shape flexibility.

**Cons:**
- ❌ Requires n_components in advance.
- ❌ EM is sensitive to initialization — use n_init > 1.
- ❌ Can collapse a component onto a single point (singular covariance).
- ❌ More parameters → more data needed to fit reliably.

**Assumptions:** Each cluster ~ multivariate Gaussian; n_components known.

**Complexity:** O(n_components * n_features^2 * n_samples * n_iter)

**Key hyperparameters:** `n_components`, `covariance_type`, `n_init`

## `kmeans`

**Partitional clustering minimizing within-cluster variance (k groups).**

**When to use:** When clusters are expected to be roughly spherical, similarly sized, and you know (or can search) the number of clusters k.

**Pros:**
- ✅ Fast and scales well to large datasets.
- ✅ Simple to understand and interpret (centroid-based).
- ✅ Works well when cluster shapes are convex and isotropic.
- ✅ n_clusters is directly searchable via Optuna.

**Cons:**
- ❌ Requires the number of clusters k in advance.
- ❌ Assumes spherical, equal-variance clusters — fails on elongated/irregular shapes.
- ❌ Sensitive to initialization and outliers (use k-means++ init).
- ❌ Hard assignment only (no soft/probabilistic membership).

**Assumptions:** Globular clusters; Similar cluster sizes; Euclidean distance meaningful.

**Complexity:** O(n_samples * n_features * k * n_iter)

**Key hyperparameters:** `n_clusters`, `init`, `n_init`, `algorithm`
