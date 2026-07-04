# Classification models

## `decision_tree_classifier`

**Non-parametric classifier: recursive binary splits by Gini/entropy.**

**When to use:** When class boundaries are non-linear and piecewise-axis-aligned. As a building block for ensembles (RandomForest / GradientBoosting) or an interpretable standalone classifier on small data.

**Pros:**
- ✅ Captures non-linear, axis-aligned boundaries natively.
- ✅ Highly interpretable — the tree can be visualized.
- ✅ No scaling needed (splits are scale-invariant).
- ✅ Handles mixed numeric/categorical (with encoding).
- ✅ Fast to train and predict.

**Cons:**
- ❌ Extremely high variance — single trees overfit easily (use RF/GBM).
- ❌ Greedy splits → locally optimal, not globally.
- ❌ Extrapolation is flat (predicts the leaf class distribution).
- ❌ Sensitive to small data perturbations (unstable structure).
- ❌ Axis-aligned only — struggles with diagonal boundaries.

**Assumptions:** Decision boundaries are axis-aligned.

**Complexity:** O(n_samples * n_features * log n_samples) to fit; O(depth) to predict

**Key hyperparameters:** `max_depth`, `min_samples_split`, `criterion`

## `knn`

**K-Nearest Neighbors — classifies by majority vote among the k closest points.**

**When to use:** Small-to-medium datasets with non-linear boundaries where a simple, instance-based classifier suffices. Good baseline; no training phase (lazy learner) but slow to predict.

**Pros:**
- ✅ No training phase — predictions are immediate (lazy learner).
- ✅ Captures arbitrary non-linear boundaries naturally.
- ✅ Simple to explain: 'the k closest points vote'.
- ✅ weights='distance' handles uneven sampling density.

**Cons:**
- ❌ Slow to predict: O(n_samples * n_features) per query (needs a KD-tree or Ball-tree for scale).
- ❌ Sensitive to feature scale — distance-based, so a scaler is mandatory.
- ❌ Curse of dimensionality: distances become meaningless in high-d.
- ❌ k must be chosen carefully (too small → noise, too large → bias).
- ❌ No native feature importances.

**Assumptions:** A meaningful distance metric exists; Features scaled; n_features not too large.

**Complexity:** O(1) training; O(n_samples * n_features) prediction (or O(log n) with a tree index)

**Key hyperparameters:** `n_neighbors`, `weights`, `p (distance metric)`

## `logistic_regression`

**L2/L1-regularized linear classifier (logistic regression).**

**When to use:** Linearly-separable (or nearly so) classification; when you need interpretable coefficients and calibrated probabilities. Strong baseline for binary and multiclass with linear structure.

**Pros:**
- ✅ Fast to train and predict; produces calibrated probabilities.
- ✅ Interpretable coefficients (sign and magnitude per feature).
- ✅ L1 penalty (saga/liblinear solver) enables feature selection.
- ✅ Multiclass via softmax (multinomial) out of the box.
- ✅ Well-understood statistical foundations.

**Cons:**
- ❌ Linear decision boundary — fails on non-linear structure (use KNN/SVC/trees/ensembles).
- ❌ Sensitive to feature scale — needs a scaler.
- ❌ L1 + saga solver can be slow to converge on wide data.
- ❌ Multinomial + small-n per class can overfit.
- ❌ Outliers in features can dominate.

**Assumptions:** Linear log-odds relationship; Features scaled; i.i.d. samples.

**Complexity:** O(n_samples * n_features) per solver iteration

**Key hyperparameters:** `C`, `penalty`, `solver`, `max_iter`

## `mlp_classifier`

**Multi-Layer Perceptron (PyTorch) for tabular classification.**

**When to use:** When linear classifiers underfit and tree ensembles aren't ideal. The DL objective supports epoch pruning. Requires pip install containeer-optuna[dl].

**Pros:**
- ✅ Captures non-linear class boundaries.
- ✅ Epoch pruning cuts bad trials early.
- ✅ Softmax output gives calibrated-ish probabilities.
- ✅ Works with any classification metric (accuracy/f1/roc_auc).

**Cons:**
- ❌ Requires PyTorch (optional [dl] extra).
- ❌ Many hyperparameters — needs more trials than sklearn classifiers.
- ❌ Sensitive to feature scaling.
- ❌ Prone to overfitting on small datasets.
- ❌ Slower per trial than sklearn classifiers.

**Assumptions:** Features scaled; Sufficient data for the architecture size.

**Complexity:** O(epochs * n_samples * max(hidden_sizes)) per trial

**Key hyperparameters:** `hidden_layer_sizes`, `learning_rate`, `epochs`, `batch_size`, `dropout`

## `svc`

**Support Vector Classifier — max-margin classifier with the kernel trick.**

**When to use:** Small-to-medium datasets with non-linear boundaries. Strong on high-dimensional data (text, genomics) where the kernel avoids the curse of dimensionality. Set probability=True if you need probabilities or roc_auc.

**Pros:**
- ✅ Flexible non-linear boundaries via the kernel trick (RBF/poly).
- ✅ Max-margin → good generalization on small data.
- ✅ Effective in high dimensions (kernel distance).
- ✅ Solution is sparse — only support vectors matter.
- ✅ probability=True gives calibrated-ish probabilities.

**Cons:**
- ❌ Scaling is mandatory (distance-based kernel).
- ❌ Training time is O(n_samples^2) to O(n_samples^3) — does NOT scale to large datasets (use LinearSVC or trees/ensembles).
- ❌ Three hyperparameters (C, kernel, gamma) interact; need Optuna tuning on log scale.
- ❌ probability=True adds significant training cost (Platt scaling, 5-fold internal CV).
- ❌ No native feature importances.

**Assumptions:** Features scaled; n_samples not huge (< ~10k); Kernel is appropriate.

**Complexity:** O(n_samples^2) to O(n_samples^3) training; O(n_support_vectors * n_features) predict

**Key hyperparameters:** `C`, `kernel`, `gamma`, `probability`
