# Clustering models

## `agglomerative`

**Hierarchical agglomerative clustering — builds a dendrogram bottom-up.**

**When to use:** When you want a cluster hierarchy (dendrogram) or when clusters may have arbitrary shapes linkable via a distance criterion. Good for small-to-medium datasets where the hierarchy itself is informative.

**Pros:**
- ✅ Produces a full hierarchy (dendrogram) — reveals structure at multiple resolutions.
- ✅ Flexible via linkage criterion (ward / complete / average / single).
- ✅ No need to specify the number of clusters at fit time (can cut the tree at any level).
- ✅ Works with any pairwise distance / affinity.

**Cons:**
- ❌ O(n^3) time and O(n^2) memory in the worst case — does NOT scale to large datasets.
- ❌ Irreversible: early merge mistakes cannot be undone.
- ❌ ward linkage requires Euclidean distance and roughly equal cluster sizes for best results.
- ❌ single linkage suffers from chaining; complete linkage is sensitive to outliers.
- ❌ Cannot predict labels for new unseen points (no predict method).

**Assumptions:** A meaningful pairwise distance exists; n_samples not huge (< ~10k).

**Complexity:** O(n_samples^3) full linkage; O(n_samples^2 * log n_samples) with efficient heap

**Key hyperparameters:** `n_clusters`, `linkage`, `affinity`

## `birch`

**Balanced Iterative Reducing and Clustering using Hierarchies — memory-efficient online clustering.**

**When to use:** Large datasets where memory matters: BIRCH builds a Clustering Feature Tree in a single pass and can cluster incrementally. A good scaler → BIRCH → final-clusterer pipeline scales to millions of rows.

**Pros:**
- ✅ Memory-efficient (summarizes data in a CF Tree).
- ✅ Single-pass — scales to very large datasets.
- ✅ Supports online/incremental updates (partial_fit).
- ✅ One of the few non-KMeans clusterers that implements predict (can label new points without refitting).

**Cons:**
- ❌ Sensitive to the threshold parameter (controls CF Tree granularity).
- ❌ Produces spherical clusters (inherits KMeans-like geometry).
- ❌ Order-dependent — the CF Tree depends on data arrival order.
- ❌ Struggles with high-dimensional data (curse of dimensionality).
- ❌ Less accurate than direct KMeans/HDBSCAN on small data.

**Assumptions:** Spherical clusters; Features scaled; n_features not too large.

**Complexity:** O(n_samples * n_features) single pass

**Key hyperparameters:** `threshold`, `n_clusters`, `branching_factor`

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

## `hdbscan`

**Hierarchical Density-Based Clustering — robust to varying cluster densities.**

**When to use:** When clusters have different densities (where DBSCAN struggles) and you want density-based clustering that auto-discovers cluster counts. Strong default for exploratory clustering of unknown structure.

**Pros:**
- ✅ Handles clusters of widely varying densities (DBSCAN's main weakness).
- ✅ No need to set the number of clusters; discovered automatically.
- ✅ Produces a cluster hierarchy (can extract flat labels at multiple granularities via cluster_selection_method).
- ✅ Robust to noise and outliers (labels them -1).
- ✅ Only two main hyperparameters (min_cluster_size, min_samples).

**Cons:**
- ❌ Requires sklearn >= 1.3 (not available on older installs).
- ❌ Sensitive to min_cluster_size — too high merges real clusters, too low creates spurious ones.
- ❌ Memory O(n^2) in the worst case for the mutual-reachability graph.
- ❌ Noise points must be stripped before computing Silhouette/CH/DB.
- ❌ Cannot predict labels for new unseen points (no predict method).

**Assumptions:** Clusters are dense regions separated by sparse regions; Features scaled.

**Complexity:** O(n_samples^2) worst case; O(n_samples * log n_samples) typically

**Key hyperparameters:** `min_cluster_size`, `min_samples`, `cluster_selection_method`

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

## `optics`

**Ordering Points To Identify Clustering Structure — DBSCAN-like, but with variable density.**

**When to use:** Spatial / point-cloud data where clusters have varying densities and you want a reachability plot for visual structure analysis. A more sophisticated alternative to DBSCAN.

**Pros:**
- ✅ Handles clusters of varying density better than DBSCAN.
- ✅ Produces a reachability plot — visualizes cluster hierarchy.
- ✅ Does not require a global eps (xi extracts clusters automatically).
- ✅ Robust to noise (labels outliers -1).

**Cons:**
- ❌ Slower than DBSCAN in practice (O(n log n) with index, O(n^2) worst).
- ❌ Sensitive to min_samples and xi — needs careful Optuna tuning.
- ❌ Reachability-plot interpretation is non-trivial.
- ❌ Cannot predict labels for new unseen points (no predict method).
- ❌ Noise points must be stripped before computing metrics.

**Assumptions:** Clusters are dense regions of varying density; Features scaled.

**Complexity:** O(n_samples * log n_samples) with a spatial index, O(n_samples^2) worst case

**Key hyperparameters:** `min_samples`, `xi`, `max_eps`

## `spectral`

**Graph-based clustering via eigen-decomposition of the similarity matrix.**

**When to use:** When clusters are non-convex / non-globular (e.g. concentric rings, two moons) and KMeans fails. Strong on small datasets where pairwise affinity is meaningful.

**Pros:**
- ✅ Detects non-convex cluster shapes that KMeans cannot.
- ✅ Mathematically grounded (graph Laplacian spectral decomposition).
- ✅ Works with any affinity (rbf, nearest_neighbors, or precomputed).
- ✅ Only needs the number of clusters (no density parameters).

**Cons:**
- ❌ O(n^3) eigen-decomposition — does NOT scale to large datasets.
- ❌ Sensitive to the affinity/gamma choice; needs careful tuning.
- ❌ Requires the number of clusters in advance.
- ❌ Numerically unstable when the similarity graph is disconnected.
- ❌ Cannot predict labels for new unseen points (no predict method).

**Assumptions:** A meaningful pairwise similarity (affinity) exists; n_samples not huge (< ~5k).

**Complexity:** O(n_samples^3) for the eigen-decomposition

**Key hyperparameters:** `n_clusters`, `affinity`, `gamma (for rbf)`
