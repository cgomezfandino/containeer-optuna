# Scaler models

## `minmax_scaler`

**Scale features to a fixed range, typically [0, 1].**

**When to use:** When you need bounded feature ranges (e.g. neural nets, image pixels), or when the data has a natural minimum/maximum. Also a reasonable alternative to StandardScaler for non-Gaussian distributions.

**Pros:**
- ✅ Produces bounded features — useful for downstream models sensitive to scale.
- ✅ Preserves the shape of the original distribution.
- ✅ Intuitive [0, 1] (or arbitrary) range.

**Cons:**
- ❌ Very sensitive to outliers (min/max are extreme statistics).
- ❌ Does not center to zero — some models (e.g. SVM with RBF) prefer standardization.
- ❌ New (unseen) data may fall outside [0, 1] and needs clipping.

**Assumptions:** Known / stable min and max in the training data.

**Complexity:** O(n_samples * n_features)

**Key hyperparameters:** `feature_range`

## `standard_scaler`

**Standardize features to zero mean and unit variance (z-scores).**

**When to use:** Default scaler for most distance- or regularization-sensitive models (Ridge, Lasso, KMeans, DBSCAN, SVM, neural nets).

**Pros:**
- ✅ Simple, stateless transform (no learnable shape parameter).
- ✅ Required for L1/L2-regularized and distance-based models.
- ✅ Preserves the shape of the distribution (just rescales).

**Cons:**
- ❌ Sensitive to outliers (mean and std are not robust).
- ❌ Does not bound the range — use MinMaxScaler if you need [0, 1].
- ❌ Assumes features are approximately Gaussian for z-score interpretation.

**Assumptions:** Features roughly Gaussian for meaningful z-scores.

**Complexity:** O(n_samples * n_features)

**Key hyperparameters:** `(none)`
