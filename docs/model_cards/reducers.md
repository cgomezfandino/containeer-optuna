# Reducer models

## `pca`

**Principal Component Analysis — linear dimensionality reduction.**

**When to use:** Reduce dimensionality while preserving maximum variance; visualization; decorrelation of features; compression before clustering or regression.

**Pros:**
- ✅ Fast, deterministic, closed-form (SVD).
- ✅ Optimal linear reconstruction for a given target dimension.
- ✅ Decorrelates features (useful before linear models).
- ✅ No hyperparameters beyond n_components (easy to tune).

**Cons:**
- ❌ Linear only — misses non-linear structure (use UMAP/t-SNE instead).
- ❌ Components are global and hard to interpret.
- ❌ Variance preservation != class separability.
- ❌ Sensitive to feature scale — center/scale first.

**Assumptions:** Linear structure; Large-variance directions are meaningful; Features scaled.

**Complexity:** O(min(n_samples, n_features)^2 * n_features)

**Key hyperparameters:** `n_components`

## `umap`

**Uniform Manifold Approximation — non-linear dimensionality reduction.**

**When to use:** Visualization of high-dimensional data preserving local (and some global) structure; as a preprocessing step before clustering non-linear manifolds.

**Pros:**
- ✅ Preserves local neighborhood structure better than PCA.
- ✅ Faster than t-SNE on large datasets and generalizes to new points.
- ✅ Can be used as a feature extractor (transform).
- ✅ Scales to hundreds of thousands of samples.

**Cons:**
- ❌ Non-deterministic without a fixed seed (we set random_state=42).
- ❌ Heavy dependencies (numba) — adds install weight.
- ❌ n_neighbors/min_dist strongly affect the embedding shape.
- ❌ Not a density model — distances in the embedding are not directly meaningful.

**Assumptions:** Data lies on (or near) a locally-uniform Riemannian manifold.

**Complexity:** O(n_samples^1.14) approximately (NN-descent based)

**Key hyperparameters:** `n_neighbors`, `min_dist`, `n_components`
