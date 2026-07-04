"""Model cards: when to use each algorithm, with pros and cons.

Each entry is a :class:`ModelCard` documenting a registered model. The cards
serve three purposes:

1. **In-code reference** — :func:`get_model_card` returns a card programmatically.
2. **CLI introspection** — ``containeer-optuna describe <model>`` prints a card.
3. **Documentation** — :func:`all_model_cards` is used to generate the
   ``model_cards/*.md`` pages of the docs site.

The cards intentionally cover both the registered M0 models and a few forward-
looking entries (RandomForest, HDBSCAN, t-SNE, LogisticRegression) so the docs
have a coherent narrative as later milestones land.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class ModelCard:
    """Structured documentation for a single model/algorithm.

    Attributes:
        name: Registry key (matches ``config/models.yaml``).
        kind: ``regression``, ``clustering``, ``reducer``, ``scaler``.
        summary: One-sentence description.
        when_to_use: Typical use cases / data regimes.
        pros: List of advantages.
        cons: List of limitations / caveats.
        assumptions: Modeling assumptions (linearity, cluster shape, etc.).
        complexity: Big-O or qualitative training/prediction cost.
        key_hyperparameters: Names of the most important tuning knobs.
        milestone: Which milestone (``M0``, ``M1``...) ships this model.
    """

    name: str
    kind: str
    summary: str
    when_to_use: str
    pros: list[str] = field(default_factory=list)
    cons: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    complexity: str = ""
    key_hyperparameters: list[str] = field(default_factory=list)
    milestone: str = "M0"


# --- M0: Regression ------------------------------------------------------

RIDGE_CARD = ModelCard(
    name="ridge",
    kind="regression",
    summary="L2-regularized linear regression (Tikhonov).",
    when_to_use=(
        "Linear regression with many correlated or redundant features and risk of "
        "overfitting / multicollinearity. Works well when n_samples >= n_features."
    ),
    pros=[
        "Closed-form, fast to train and predict.",
        "Reduces variance via shrinkage — improves generalization.",
        "Stable under multicollinearity (always has a unique solution).",
        "Interpretable coefficients (sign and magnitude).",
    ],
    cons=[
        "Assumes a linear relationship between features and target.",
        "Does not perform feature selection (coefficients shrink but stay non-zero).",
        "Sensitive to feature scale — needs a scaler.",
        "Outliers in features or target can dominate.",
    ],
    assumptions=["Linearity", "Homoscedastic, independent residuals", "Features scaled"],
    complexity="O(n_samples * n_features^2)",
    key_hyperparameters=["alpha (regularization strength)", "solver", "tol"],
)

LASSO_CARD = ModelCard(
    name="lasso",
    kind="regression",
    summary="L1-regularized linear regression with embedded feature selection.",
    when_to_use=(
        "High-dimensional data where you expect only a subset of features to "
        "matter. Lasso drives irrelevant coefficients exactly to zero."
    ),
    pros=[
        "Performs feature selection (sparse solutions).",
        "Interpretable: non-zero coefficients reveal the active features.",
        "Fast for moderate dimensions.",
    ],
    cons=[
        "Linear model — same expressiveness limits as OLS/Ridge.",
        "When features are correlated, tends to pick one and zero the rest "
        "(arbitrarily) — consider ElasticNet in that case.",
        "Sensitive to feature scale.",
        "Solution is not unique if n_features > n_samples.",
    ],
    assumptions=["Linearity", "Sparse true signal", "Features scaled"],
    complexity="O(n_samples * n_features) per coordinate-descent iteration",
    key_hyperparameters=["alpha", "max_iter", "tol"],
)

OLS_CARD = ModelCard(
    name="ols",
    kind="regression",
    summary="Ordinary Least Squares — unregularized linear regression.",
    when_to_use=(
        "Baseline regression, well-conditioned design matrix with n >> p and no "
        "strong multicollinearity. Useful as a reference point for Ridge/Lasso."
    ),
    pros=[
        "Simplest model — no hyperparameters to tune.",
        "Unbiased estimator with minimum variance among linear unbiased models "
        "(Gauss-Markov, under assumptions).",
        "Extremely fast — closed-form solution.",
    ],
    cons=[
        "Overfits badly when p approaches or exceeds n.",
        "No regularization — high variance under multicollinearity.",
        "Sensitive to outliers and feature scale.",
        "Cannot do feature selection.",
    ],
    assumptions=[
        "Linearity",
        "Independence of residuals",
        "Homoscedasticity",
        "No (or weak) multicollinearity",
    ],
    complexity="O(n_samples * n_features^2)",
    key_hyperparameters=["(none — fit_intercept is the only knob)"],
)


# --- M1: Regression maturity --------------------------------------------

ELASTICNET_CARD = ModelCard(
    name="elasticnet",
    kind="regression",
    summary="Linear regression with both L1 and L2 regularization (Lasso + Ridge blend).",
    when_to_use=(
        "When features are highly correlated and you want a balance between "
        "Ridge's stability and Lasso's sparsity. The l1_ratio knob lets Optuna "
        "discover the right blend automatically."
    ),
    pros=[
        "Best of both worlds: stability under multicollinearity (L2) + feature selection (L1).",
        "l1_ratio is a single knob interpolating Ridge (0.0) → Lasso (1.0); "
        "Optuna searches it for free.",
        "Closed-form-ish (coordinate descent), fast on moderate dimensions.",
        "More stable than pure Lasso when many features are correlated.",
    ],
    cons=[
        "Still a linear model — misses non-linear structure (use trees/SVR).",
        "Two hyperparameters to tune (alpha + l1_ratio) vs Ridge/Lasso's one.",
        "Sensitive to feature scale — needs a scaler.",
        "Coordinate descent can be slow to converge on wide datasets.",
    ],
    assumptions=["Linearity", "Features scaled", "Some features may be irrelevant"],
    complexity="O(n_samples * n_features) per coordinate-descent iteration",
    key_hyperparameters=["alpha", "l1_ratio", "max_iter"],
    milestone="M1",
)

DECISION_TREE_CARD = ModelCard(
    name="decision_tree",
    kind="regression",
    summary="Non-parametric model: recursive binary splits to minimize within-leaf variance.",
    when_to_use=(
        "When the target has non-linear, piecewise-constant relationships with "
        "features. As a building block for ensembles (RF / GBM) or an "
        "interpretable standalone model on small data."
    ),
    pros=[
        "Captures non-linearities and feature interactions natively — no "
        "feature engineering required.",
        "No scaling needed (splits are scale-invariant).",
        "Highly interpretable — the tree can be visualized.",
        "Handles mixed numeric/categorical (with encoding) and missing-ish data.",
    ],
    cons=[
        "Extremely high variance — small data changes produce very different "
        "trees (classic overfitter).",
        "Single trees are almost always dominated by their ensembles (RF, GBM).",
        "Extrapolation is flat (predicts the leaf mean) — fails outside the training range.",
        "Greedy splits → locally optimal, not globally.",
    ],
    assumptions=["Target is approximately piecewise-constant/smooth in feature space"],
    complexity="O(n_samples * n_features * log n_samples) to fit; O(depth) to predict",
    key_hyperparameters=["max_depth", "min_samples_split", "min_samples_leaf"],
    milestone="M1",
)

RANDOM_FOREST_CARD = ModelCard(
    name="random_forest",
    kind="regression",
    summary="Bagging ensemble of decision trees; averages many decorrelated trees.",
    when_to_use=(
        "Strong default for tabular regression: captures non-linearities and "
        "interactions with minimal tuning, robust to overfitting, and gives "
        "feature importances. Try this before gradient boosting on new data."
    ),
    pros=[
        "Excellent out-of-the-box accuracy on tabular data.",
        "Robust to overfitting via bagging + random feature subsets.",
        "No scaling needed; handles non-linearities and interactions natively.",
        "Parallelizable (n_jobs=-1) and gives feature importances.",
        "Few critical hyperparameters — works well with defaults.",
    ],
    cons=[
        "Large ensembles are memory-heavy and slower to predict than a single "
        "tree or linear model.",
        "Extrapolation is flat (averages leaf means) — fails outside training "
        "range; GBM is often better on trends.",
        "Less interpretable than a single tree.",
        "Generally slightly less accurate than well-tuned gradient boosting on structured data.",
    ],
    assumptions=["Rows are i.i.d.", "Target is smooth in feature space"],
    complexity="O(n_estimators * n_samples * n_features * log n_samples)",
    key_hyperparameters=["n_estimators", "max_depth", "min_samples_split", "max_features"],
    milestone="M1",
)

GRADIENT_BOOSTING_CARD = ModelCard(
    name="gradient_boosting",
    kind="regression",
    summary="Sequential ensemble of shallow trees; each tree fits the residual errors.",
    when_to_use=(
        "Top performer on most tabular regression benchmarks. Prefer over RF "
        "when you have enough data and tuning budget, and when the target has "
        "smooth trends (GBM extrapolates better than RF)."
    ),
    pros=[
        "Often the most accurate model on structured/tabular data.",
        "Handles non-linearities and interactions natively.",
        "Extrapolates trends better than RF (sequential residual fitting).",
        "Supports arbitrary differentiable losses (squared, absolute, huber, quantile).",
    ],
    cons=[
        "Sensitive to hyperparameters — needs careful Optuna tuning "
        "(learning_rate ↔ n_estimators trade-off).",
        "Sequential by default — cannot parallelize across trees (unlike RF).",
        "Prone to overfit if learning_rate is too high or n_estimators too large "
        "without early stopping.",
        "Slower to train than RF per estimator (but often fewer are needed).",
        "No scaling needed, but outliers in the target hurt (use huber/quantile loss).",
    ],
    assumptions=["Target is smooth in feature space", "Sufficient data to justify complexity"],
    complexity="O(n_estimators * n_samples * n_features * log n_samples)",
    key_hyperparameters=["learning_rate", "n_estimators", "max_depth", "subsample"],
    milestone="M1",
)

SVR_CARD = ModelCard(
    name="svr",
    kind="regression",
    summary="Support Vector Regression — fits a tube of width epsilon around the target.",
    when_to_use=(
        "Small-to-medium datasets with non-linear relationships, especially "
        "when you care about a margin of tolerance (epsilon-insensitive loss). "
        "Strong on high-dimensional data where n_features ≈ n_samples."
    ),
    pros=[
        "Flexible non-linear regression via the kernel trick (RBF/poly).",
        "epsilon-tube makes it robust to small target noise.",
        "Effective in high dimensions (kernel avoids the curse of dimensionality).",
        "Solution is sparse — only support vectors matter.",
    ],
    cons=[
        "Scaling is mandatory (distance-based kernel).",
        "Training time is roughly O(n_samples^2) to O(n_samples^3) — does NOT "
        "scale to large datasets (use RF/GBM or LinearSVR instead).",
        "Three hyperparameters (C, epsilon, kernel/gamma) interact and need "
        "careful Optuna tuning, all on log scale.",
        "Extrapolation is poor — predictions saturate outside the training range.",
        "No native feature importances.",
    ],
    assumptions=["Features scaled", "n_samples not huge (< ~10k)", "Kernel is appropriate"],
    complexity="O(n_samples^2) to O(n_samples^3) training; O(n_support_vectors * n_features) predict",
    key_hyperparameters=["C", "epsilon", "kernel", "gamma (RBF/poly)"],
    milestone="M1",
)


# --- M0: Clustering ------------------------------------------------------

KMEANS_CARD = ModelCard(
    name="kmeans",
    kind="clustering",
    summary="Partitional clustering minimizing within-cluster variance (k groups).",
    when_to_use=(
        "When clusters are expected to be roughly spherical, similarly sized, and "
        "you know (or can search) the number of clusters k."
    ),
    pros=[
        "Fast and scales well to large datasets.",
        "Simple to understand and interpret (centroid-based).",
        "Works well when cluster shapes are convex and isotropic.",
        "n_clusters is directly searchable via Optuna.",
    ],
    cons=[
        "Requires the number of clusters k in advance.",
        "Assumes spherical, equal-variance clusters — fails on elongated/irregular shapes.",
        "Sensitive to initialization and outliers (use k-means++ init).",
        "Hard assignment only (no soft/probabilistic membership).",
    ],
    assumptions=["Globular clusters", "Similar cluster sizes", "Euclidean distance meaningful"],
    complexity="O(n_samples * n_features * k * n_iter)",
    key_hyperparameters=["n_clusters", "init", "n_init", "algorithm"],
)

DBSCAN_CARD = ModelCard(
    name="dbscan",
    kind="clustering",
    summary="Density-based clustering; finds arbitrarily-shaped clusters and noise.",
    when_to_use=(
        "Spatial/point-cloud data where clusters may be non-convex and you want "
        "outliers flagged as noise. No need to specify k in advance."
    ),
    pros=[
        "Does NOT require the number of clusters a priori.",
        "Discovers clusters of arbitrary shape.",
        "Robust to outliers (labels them -1).",
        "Well-suited to spatial / density-varying data.",
    ],
    cons=[
        "Struggles when clusters have very different densities.",
        "Sensitive to eps and min_samples — needs careful tuning (Optuna helps).",
        "Distance metric and scaling strongly affect results.",
        "Noise points must be stripped before computing Silhouette/CH/DB.",
    ],
    assumptions=["Clusters are dense regions separated by sparse regions", "Features scaled"],
    complexity="O(n_samples * log n_samples) with a spatial index, O(n^2) worst case",
    key_hyperparameters=["eps", "min_samples"],
)

GMM_CARD = ModelCard(
    name="gmm",
    kind="clustering",
    summary="Gaussian Mixture Model — soft (probabilistic) clustering.",
    when_to_use=(
        "When you want cluster membership probabilities (soft assignments), or "
        "clusters may be elliptical (covariance-aware). Useful for density estimation."
    ),
    pros=[
        "Soft assignments — gives P(cluster | point).",
        "Models elliptical clusters via per-component covariance.",
        "Provides a generative model (likelihood-based).",
        "covariance_type tunes cluster shape flexibility.",
    ],
    cons=[
        "Requires n_components in advance.",
        "EM is sensitive to initialization — use n_init > 1.",
        "Can collapse a component onto a single point (singular covariance).",
        "More parameters → more data needed to fit reliably.",
    ],
    assumptions=["Each cluster ~ multivariate Gaussian", "n_components known"],
    complexity="O(n_components * n_features^2 * n_samples * n_iter)",
    key_hyperparameters=["n_components", "covariance_type", "n_init"],
)


# --- M2: Clustering maturity --------------------------------------------

HDBSCAN_CARD = ModelCard(
    name="hdbscan",
    kind="clustering",
    summary="Hierarchical Density-Based Clustering — robust to varying cluster densities.",
    when_to_use=(
        "When clusters have different densities (where DBSCAN struggles) and "
        "you want density-based clustering that auto-discovers cluster counts. "
        "Strong default for exploratory clustering of unknown structure."
    ),
    pros=[
        "Handles clusters of widely varying densities (DBSCAN's main weakness).",
        "No need to set the number of clusters; discovered automatically.",
        "Produces a cluster hierarchy (can extract flat labels at multiple "
        "granularities via cluster_selection_method).",
        "Robust to noise and outliers (labels them -1).",
        "Only two main hyperparameters (min_cluster_size, min_samples).",
    ],
    cons=[
        "Requires sklearn >= 1.3 (not available on older installs).",
        "Sensitive to min_cluster_size — too high merges real clusters, too "
        "low creates spurious ones.",
        "Memory O(n^2) in the worst case for the mutual-reachability graph.",
        "Noise points must be stripped before computing Silhouette/CH/DB.",
        "Cannot predict labels for new unseen points (no predict method).",
    ],
    assumptions=["Clusters are dense regions separated by sparse regions", "Features scaled"],
    complexity="O(n_samples^2) worst case; O(n_samples * log n_samples) typically",
    key_hyperparameters=["min_cluster_size", "min_samples", "cluster_selection_method"],
    milestone="M2",
)

AGGLOMERATIVE_CARD = ModelCard(
    name="agglomerative",
    kind="clustering",
    summary="Hierarchical agglomerative clustering — builds a dendrogram bottom-up.",
    when_to_use=(
        "When you want a cluster hierarchy (dendrogram) or when clusters may "
        "have arbitrary shapes linkable via a distance criterion. Good for "
        "small-to-medium datasets where the hierarchy itself is informative."
    ),
    pros=[
        "Produces a full hierarchy (dendrogram) — reveals structure at multiple resolutions.",
        "Flexible via linkage criterion (ward / complete / average / single).",
        "No need to specify the number of clusters at fit time (can cut the tree at any level).",
        "Works with any pairwise distance / affinity.",
    ],
    cons=[
        "O(n^3) time and O(n^2) memory in the worst case — does NOT scale to large datasets.",
        "Irreversible: early merge mistakes cannot be undone.",
        "ward linkage requires Euclidean distance and roughly equal cluster "
        "sizes for best results.",
        "single linkage suffers from chaining; complete linkage is sensitive to outliers.",
        "Cannot predict labels for new unseen points (no predict method).",
    ],
    assumptions=["A meaningful pairwise distance exists", "n_samples not huge (< ~10k)"],
    complexity="O(n_samples^3) full linkage; O(n_samples^2 * log n_samples) with efficient heap",
    key_hyperparameters=["n_clusters", "linkage", "affinity"],
    milestone="M2",
)

SPECTRAL_CARD = ModelCard(
    name="spectral",
    kind="clustering",
    summary="Graph-based clustering via eigen-decomposition of the similarity matrix.",
    when_to_use=(
        "When clusters are non-convex / non-globular (e.g. concentric rings, "
        "two moons) and KMeans fails. Strong on small datasets where pairwise "
        "affinity is meaningful."
    ),
    pros=[
        "Detects non-convex cluster shapes that KMeans cannot.",
        "Mathematically grounded (graph Laplacian spectral decomposition).",
        "Works with any affinity (rbf, nearest_neighbors, or precomputed).",
        "Only needs the number of clusters (no density parameters).",
    ],
    cons=[
        "O(n^3) eigen-decomposition — does NOT scale to large datasets.",
        "Sensitive to the affinity/gamma choice; needs careful tuning.",
        "Requires the number of clusters in advance.",
        "Numerically unstable when the similarity graph is disconnected.",
        "Cannot predict labels for new unseen points (no predict method).",
    ],
    assumptions=[
        "A meaningful pairwise similarity (affinity) exists",
        "n_samples not huge (< ~5k)",
    ],
    complexity="O(n_samples^3) for the eigen-decomposition",
    key_hyperparameters=["n_clusters", "affinity", "gamma (for rbf)"],
    milestone="M2",
)

BIRCH_CARD = ModelCard(
    name="birch",
    kind="clustering",
    summary="Balanced Iterative Reducing and Clustering using Hierarchies — memory-efficient online clustering.",
    when_to_use=(
        "Large datasets where memory matters: BIRCH builds a Clustering "
        "Feature Tree in a single pass and can cluster incrementally. A good "
        "scaler → BIRCH → final-clusterer pipeline scales to millions of rows."
    ),
    pros=[
        "Memory-efficient (summarizes data in a CF Tree).",
        "Single-pass — scales to very large datasets.",
        "Supports online/incremental updates (partial_fit).",
        "One of the few non-KMeans clusterers that implements predict (can "
        "label new points without refitting).",
    ],
    cons=[
        "Sensitive to the threshold parameter (controls CF Tree granularity).",
        "Produces spherical clusters (inherits KMeans-like geometry).",
        "Order-dependent — the CF Tree depends on data arrival order.",
        "Struggles with high-dimensional data (curse of dimensionality).",
        "Less accurate than direct KMeans/HDBSCAN on small data.",
    ],
    assumptions=["Spherical clusters", "Features scaled", "n_features not too large"],
    complexity="O(n_samples * n_features) single pass",
    key_hyperparameters=["threshold", "n_clusters", "branching_factor"],
    milestone="M2",
)

OPTICS_CARD = ModelCard(
    name="optics",
    kind="clustering",
    summary="Ordering Points To Identify Clustering Structure — DBSCAN-like, but with variable density.",
    when_to_use=(
        "Spatial / point-cloud data where clusters have varying densities and "
        "you want a reachability plot for visual structure analysis. A more "
        "sophisticated alternative to DBSCAN."
    ),
    pros=[
        "Handles clusters of varying density better than DBSCAN.",
        "Produces a reachability plot — visualizes cluster hierarchy.",
        "Does not require a global eps (xi extracts clusters automatically).",
        "Robust to noise (labels outliers -1).",
    ],
    cons=[
        "Slower than DBSCAN in practice (O(n log n) with index, O(n^2) worst).",
        "Sensitive to min_samples and xi — needs careful Optuna tuning.",
        "Reachability-plot interpretation is non-trivial.",
        "Cannot predict labels for new unseen points (no predict method).",
        "Noise points must be stripped before computing metrics.",
    ],
    assumptions=["Clusters are dense regions of varying density", "Features scaled"],
    complexity="O(n_samples * log n_samples) with a spatial index, O(n_samples^2) worst case",
    key_hyperparameters=["min_samples", "xi", "max_eps"],
    milestone="M2",
)


# --- M0: Reducers --------------------------------------------------------

PCA_CARD = ModelCard(
    name="pca",
    kind="reducer",
    summary="Principal Component Analysis — linear dimensionality reduction.",
    when_to_use=(
        "Reduce dimensionality while preserving maximum variance; visualization; "
        "decorrelation of features; compression before clustering or regression."
    ),
    pros=[
        "Fast, deterministic, closed-form (SVD).",
        "Optimal linear reconstruction for a given target dimension.",
        "Decorrelates features (useful before linear models).",
        "No hyperparameters beyond n_components (easy to tune).",
    ],
    cons=[
        "Linear only — misses non-linear structure (use UMAP/t-SNE instead).",
        "Components are global and hard to interpret.",
        "Variance preservation != class separability.",
        "Sensitive to feature scale — center/scale first.",
    ],
    assumptions=["Linear structure", "Large-variance directions are meaningful", "Features scaled"],
    complexity="O(min(n_samples, n_features)^2 * n_features)",
    key_hyperparameters=["n_components"],
)

UMAP_CARD = ModelCard(
    name="umap",
    kind="reducer",
    summary="Uniform Manifold Approximation — non-linear dimensionality reduction.",
    when_to_use=(
        "Visualization of high-dimensional data preserving local (and some global) "
        "structure; as a preprocessing step before clustering non-linear manifolds."
    ),
    pros=[
        "Preserves local neighborhood structure better than PCA.",
        "Faster than t-SNE on large datasets and generalizes to new points.",
        "Can be used as a feature extractor (transform).",
        "Scales to hundreds of thousands of samples.",
    ],
    cons=[
        "Non-deterministic without a fixed seed (we set random_state=42).",
        "Heavy dependencies (numba) — adds install weight.",
        "n_neighbors/min_dist strongly affect the embedding shape.",
        "Not a density model — distances in the embedding are not directly meaningful.",
    ],
    assumptions=["Data lies on (or near) a locally-uniform Riemannian manifold"],
    complexity="O(n_samples^1.14) approximately (NN-descent based)",
    key_hyperparameters=["n_neighbors", "min_dist", "n_components"],
)


# --- M3: Dimensionality reduction ---------------------------------------

TSNE_CARD = ModelCard(
    name="tsne",
    kind="reducer",
    summary="t-Distributed Stochastic Neighbor Embedding — non-linear visualization reduction.",
    when_to_use=(
        "Visualizing high-dimensional data in 2D/3D via "
        "``runner.plot_best_embedding()`` (which applies steps manually via "
        "fit_transform). NOT usable as a YAML reducer in an experiment "
        "pipeline: sklearn's Pipeline requires intermediate steps to "
        "implement transform(), which t-SNE lacks."
    ),
    pros=[
        "Reveals local structure and clusters that PCA/linear methods miss.",
        "Produces visually compelling 2D/3D embeddings.",
        "init='pca' gives a deterministic, well-conditioned start.",
        "Available via plot_best_embedding() which bypasses the Pipeline "
        "transform requirement by applying fit_transform directly.",
    ],
    cons=[
        "NOT usable as a YAML reducer: sklearn Pipeline._validate_steps() "
        "requires transform() on intermediate steps, which t-SNE lacks. "
        "Use plot_best_embedding() (manual step application) instead, or use "
        "PCA / TruncatedSVD / FactorAnalysis as the YAML reducer.",
        "perplexity must be < n_samples.",
        "Stochastic: the embedding changes every fit unless random_state is "
        "fixed (we set random_state=42).",
        "Slow on large data (O(n^2) standard; O(n log n) with barnes_hut/fft).",
        "Global distances in the embedding are NOT meaningful — only local neighborhoods.",
        "n_components is typically 2 or 3 (visualization), not a search space.",
    ],
    assumptions=[
        "Local neighborhoods are the meaningful structure",
        "n_samples > perplexity",
        "Used via plot_best_embedding (not as a YAML reducer)",
    ],
    complexity="O(n_samples^2) exact; O(n_samples * log n_samples) with barnes_hut",
    key_hyperparameters=["perplexity", "learning_rate", "n_components", "init"],
    milestone="M3",
)

TRUNCATED_SVD_CARD = ModelCard(
    name="truncated_svd",
    kind="reducer",
    summary="Truncated Singular Value Decomposition (LSA) — linear reduction for sparse data.",
    when_to_use=(
        "Dimensionality reduction on SPARSE data (TF-IDF, bag-of-words, "
        "one-hot matrices) where PCA (which centers data) would destroy "
        "sparsity. Works everywhere PCA does, plus sparse matrices."
    ),
    pros=[
        "Works on sparse matrices (PCA does not — PCA densifies).",
        "Linear, fast, deterministic, and reversible (has transform).",
        "Foundational for text/LSA (Latent Semantic Analysis).",
        "No hyperparameters beyond n_components (easy to tune).",
        "Works in both regression and clustering pipelines (has transform).",
    ],
    cons=[
        "Linear only — misses non-linear structure (use t-SNE/UMAP for viz).",
        "Components (topics) are global and can be hard to interpret.",
        "Variance preservation != class separability.",
        "Randomized solver is approximate (set random_state for reproducibility).",
        "Less interpretable variance-explained than PCA on dense data.",
    ],
    assumptions=["Linear structure", "Sparse input where relevant", "n_components < min(n, p)"],
    complexity="O(n_components * n_samples * n_features) with randomized solver",
    key_hyperparameters=["n_components", "n_iter (randomized solver)"],
    milestone="M3",
)

FACTOR_ANALYSIS_CARD = ModelCard(
    name="factor_analysis",
    kind="reducer",
    summary="Factor Analysis — generative latent-variable linear reduction with Gaussian noise.",
    when_to_use=(
        "When you believe observed features are driven by a small number of "
        "latent factors plus feature-specific Gaussian noise. Common in "
        "psychometrics, survey analysis, and sensor fusion. Works everywhere "
        "(has transform)."
    ),
    pros=[
        "Generative model — estimates latent factors AND feature-specific "
        "noise (unlike PCA which assumes isotropic noise).",
        "Linear, has transform — works in regression and clustering pipelines.",
        "Produces interpretable factor loadings (feature → factor weights).",
        "Handles correlated features better than PCA in some settings.",
    ],
    cons=[
        "Assumes Gaussian noise — fails on heavy-tailed or discrete data.",
        "EM algorithm can converge to local optima (multiple inits help).",
        "Slower than PCA (iterative EM vs single SVD).",
        "n_components selection is non-trivial (no clean variance-explained).",
        "Sensitive to feature scale — needs a scaler.",
    ],
    assumptions=[
        "Observed features are linear combinations of latent factors + Gaussian noise",
        "Features scaled",
    ],
    complexity="O(n_components^3 + n_samples * n_components^2) per EM iteration",
    key_hyperparameters=["n_components", "max_iter", "svd_method"],
    milestone="M3",
)


# --- M0: Scalers ---------------------------------------------------------

STANDARD_SCALER_CARD = ModelCard(
    name="standard_scaler",
    kind="scaler",
    summary="Standardize features to zero mean and unit variance (z-scores).",
    when_to_use=(
        "Default scaler for most distance- or regularization-sensitive models "
        "(Ridge, Lasso, KMeans, DBSCAN, SVM, neural nets)."
    ),
    pros=[
        "Simple, stateless transform (no learnable shape parameter).",
        "Required for L1/L2-regularized and distance-based models.",
        "Preserves the shape of the distribution (just rescales).",
    ],
    cons=[
        "Sensitive to outliers (mean and std are not robust).",
        "Does not bound the range — use MinMaxScaler if you need [0, 1].",
        "Assumes features are approximately Gaussian for z-score interpretation.",
    ],
    assumptions=["Features roughly Gaussian for meaningful z-scores"],
    complexity="O(n_samples * n_features)",
    key_hyperparameters=["(none)"],
)

MINMAX_SCALER_CARD = ModelCard(
    name="minmax_scaler",
    kind="scaler",
    summary="Scale features to a fixed range, typically [0, 1].",
    when_to_use=(
        "When you need bounded feature ranges (e.g. neural nets, image pixels), "
        "or when the data has a natural minimum/maximum. Also a reasonable "
        "alternative to StandardScaler for non-Gaussian distributions."
    ),
    pros=[
        "Produces bounded features — useful for downstream models sensitive to scale.",
        "Preserves the shape of the original distribution.",
        "Intuitive [0, 1] (or arbitrary) range.",
    ],
    cons=[
        "Very sensitive to outliers (min/max are extreme statistics).",
        "Does not center to zero — some models (e.g. SVM with RBF) prefer standardization.",
        "New (unseen) data may fall outside [0, 1] and needs clipping.",
    ],
    assumptions=["Known / stable min and max in the training data"],
    complexity="O(n_samples * n_features)",
    key_hyperparameters=["feature_range"],
)


# --- Registry ------------------------------------------------------------

_ALL_CARDS: dict[str, ModelCard] = {
    c.name: c
    for c in [
        RIDGE_CARD,
        LASSO_CARD,
        OLS_CARD,
        # M1 — Regression maturity
        ELASTICNET_CARD,
        DECISION_TREE_CARD,
        RANDOM_FOREST_CARD,
        GRADIENT_BOOSTING_CARD,
        SVR_CARD,
        KMEANS_CARD,
        DBSCAN_CARD,
        GMM_CARD,
        # M2 — Clustering maturity
        HDBSCAN_CARD,
        AGGLOMERATIVE_CARD,
        SPECTRAL_CARD,
        BIRCH_CARD,
        OPTICS_CARD,
        PCA_CARD,
        UMAP_CARD,
        # M3 — Dimensionality reduction
        TSNE_CARD,
        TRUNCATED_SVD_CARD,
        FACTOR_ANALYSIS_CARD,
        STANDARD_SCALER_CARD,
        MINMAX_SCALER_CARD,
    ]
}


def get_model_card(name: str) -> ModelCard | None:
    """Return the :class:`ModelCard` for ``name``, or ``None`` if unknown."""
    return _ALL_CARDS.get(name)


def all_model_cards() -> list[ModelCard]:
    """Return all registered model cards (sorted by kind then name)."""
    return sorted(_ALL_CARDS.values(), key=lambda c: (c.kind, c.name))


def card_to_dict(card: ModelCard) -> dict[str, object]:
    """Serialize a :class:`ModelCard` to a plain dict (for JSON output)."""
    return asdict(card)


__all__ = [
    "ModelCard",
    "get_model_card",
    "all_model_cards",
    "card_to_dict",
]
