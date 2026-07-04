# Reducer models

## `factor_analysis`

**Factor Analysis — generative latent-variable linear reduction with Gaussian noise.**

**When to use:** When you believe observed features are driven by a small number of latent factors plus feature-specific Gaussian noise. Common in psychometrics, survey analysis, and sensor fusion. Works everywhere (has transform).

**Pros:**
- ✅ Generative model — estimates latent factors AND feature-specific noise (unlike PCA which assumes isotropic noise).
- ✅ Linear, has transform — works in regression and clustering pipelines.
- ✅ Produces interpretable factor loadings (feature → factor weights).
- ✅ Handles correlated features better than PCA in some settings.

**Cons:**
- ❌ Assumes Gaussian noise — fails on heavy-tailed or discrete data.
- ❌ EM algorithm can converge to local optima (multiple inits help).
- ❌ Slower than PCA (iterative EM vs single SVD).
- ❌ n_components selection is non-trivial (no clean variance-explained).
- ❌ Sensitive to feature scale — needs a scaler.

**Assumptions:** Observed features are linear combinations of latent factors + Gaussian noise; Features scaled.

**Complexity:** O(n_components^3 + n_samples * n_components^2) per EM iteration

**Key hyperparameters:** `n_components`, `max_iter`, `svd_method`

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

## `truncated_svd`

**Truncated Singular Value Decomposition (LSA) — linear reduction for sparse data.**

**When to use:** Dimensionality reduction on SPARSE data (TF-IDF, bag-of-words, one-hot matrices) where PCA (which centers data) would destroy sparsity. Works everywhere PCA does, plus sparse matrices.

**Pros:**
- ✅ Works on sparse matrices (PCA does not — PCA densifies).
- ✅ Linear, fast, deterministic, and reversible (has transform).
- ✅ Foundational for text/LSA (Latent Semantic Analysis).
- ✅ No hyperparameters beyond n_components (easy to tune).
- ✅ Works in both regression and clustering pipelines (has transform).

**Cons:**
- ❌ Linear only — misses non-linear structure (use t-SNE/UMAP for viz).
- ❌ Components (topics) are global and can be hard to interpret.
- ❌ Variance preservation != class separability.
- ❌ Randomized solver is approximate (set random_state for reproducibility).
- ❌ Less interpretable variance-explained than PCA on dense data.

**Assumptions:** Linear structure; Sparse input where relevant; n_components < min(n, p).

**Complexity:** O(n_components * n_samples * n_features) with randomized solver

**Key hyperparameters:** `n_components`, `n_iter (randomized solver)`

## `tsne`

**t-Distributed Stochastic Neighbor Embedding — non-linear visualization reduction.**

**When to use:** Visualizing high-dimensional data in 2D/3D via ``runner.plot_best_embedding()`` (which applies steps manually via fit_transform). NOT usable as a YAML reducer in an experiment pipeline: sklearn's Pipeline requires intermediate steps to implement transform(), which t-SNE lacks.

**Pros:**
- ✅ Reveals local structure and clusters that PCA/linear methods miss.
- ✅ Produces visually compelling 2D/3D embeddings.
- ✅ init='pca' gives a deterministic, well-conditioned start.
- ✅ Available via plot_best_embedding() which bypasses the Pipeline transform requirement by applying fit_transform directly.

**Cons:**
- ❌ NOT usable as a YAML reducer: sklearn Pipeline._validate_steps() requires transform() on intermediate steps, which t-SNE lacks. Use plot_best_embedding() (manual step application) instead, or use PCA / TruncatedSVD / FactorAnalysis as the YAML reducer.
- ❌ perplexity must be < n_samples.
- ❌ Stochastic: the embedding changes every fit unless random_state is fixed (we set random_state=42).
- ❌ Slow on large data (O(n^2) standard; O(n log n) with barnes_hut/fft).
- ❌ Global distances in the embedding are NOT meaningful — only local neighborhoods.
- ❌ n_components is typically 2 or 3 (visualization), not a search space.

**Assumptions:** Local neighborhoods are the meaningful structure; n_samples > perplexity; Used via plot_best_embedding (not as a YAML reducer).

**Complexity:** O(n_samples^2) exact; O(n_samples * log n_samples) with barnes_hut

**Key hyperparameters:** `perplexity`, `learning_rate`, `n_components`, `init`

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
