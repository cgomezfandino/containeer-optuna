# Regression models

## `decision_tree`

**Non-parametric model: recursive binary splits to minimize within-leaf variance.**

**When to use:** When the target has non-linear, piecewise-constant relationships with features. As a building block for ensembles (RF / GBM) or an interpretable standalone model on small data.

**Pros:**
- ✅ Captures non-linearities and feature interactions natively — no feature engineering required.
- ✅ No scaling needed (splits are scale-invariant).
- ✅ Highly interpretable — the tree can be visualized.
- ✅ Handles mixed numeric/categorical (with encoding) and missing-ish data.

**Cons:**
- ❌ Extremely high variance — small data changes produce very different trees (classic overfitter).
- ❌ Single trees are almost always dominated by their ensembles (RF, GBM).
- ❌ Extrapolation is flat (predicts the leaf mean) — fails outside the training range.
- ❌ Greedy splits → locally optimal, not globally.

**Assumptions:** Target is approximately piecewise-constant/smooth in feature space.

**Complexity:** O(n_samples * n_features * log n_samples) to fit; O(depth) to predict

**Key hyperparameters:** `max_depth`, `min_samples_split`, `min_samples_leaf`

## `elasticnet`

**Linear regression with both L1 and L2 regularization (Lasso + Ridge blend).**

**When to use:** When features are highly correlated and you want a balance between Ridge's stability and Lasso's sparsity. The l1_ratio knob lets Optuna discover the right blend automatically.

**Pros:**
- ✅ Best of both worlds: stability under multicollinearity (L2) + feature selection (L1).
- ✅ l1_ratio is a single knob interpolating Ridge (0.0) → Lasso (1.0); Optuna searches it for free.
- ✅ Closed-form-ish (coordinate descent), fast on moderate dimensions.
- ✅ More stable than pure Lasso when many features are correlated.

**Cons:**
- ❌ Still a linear model — misses non-linear structure (use trees/SVR).
- ❌ Two hyperparameters to tune (alpha + l1_ratio) vs Ridge/Lasso's one.
- ❌ Sensitive to feature scale — needs a scaler.
- ❌ Coordinate descent can be slow to converge on wide datasets.

**Assumptions:** Linearity; Features scaled; Some features may be irrelevant.

**Complexity:** O(n_samples * n_features) per coordinate-descent iteration

**Key hyperparameters:** `alpha`, `l1_ratio`, `max_iter`

## `gradient_boosting`

**Sequential ensemble of shallow trees; each tree fits the residual errors.**

**When to use:** Top performer on most tabular regression benchmarks. Prefer over RF when you have enough data and tuning budget, and when the target has smooth trends (GBM extrapolates better than RF).

**Pros:**
- ✅ Often the most accurate model on structured/tabular data.
- ✅ Handles non-linearities and interactions natively.
- ✅ Extrapolates trends better than RF (sequential residual fitting).
- ✅ Supports arbitrary differentiable losses (squared, absolute, huber, quantile).

**Cons:**
- ❌ Sensitive to hyperparameters — needs careful Optuna tuning (learning_rate ↔ n_estimators trade-off).
- ❌ Sequential by default — cannot parallelize across trees (unlike RF).
- ❌ Prone to overfit if learning_rate is too high or n_estimators too large without early stopping.
- ❌ Slower to train than RF per estimator (but often fewer are needed).
- ❌ No scaling needed, but outliers in the target hurt (use huber/quantile loss).

**Assumptions:** Target is smooth in feature space; Sufficient data to justify complexity.

**Complexity:** O(n_estimators * n_samples * n_features * log n_samples)

**Key hyperparameters:** `learning_rate`, `n_estimators`, `max_depth`, `subsample`

## `lasso`

**L1-regularized linear regression with embedded feature selection.**

**When to use:** High-dimensional data where you expect only a subset of features to matter. Lasso drives irrelevant coefficients exactly to zero.

**Pros:**
- ✅ Performs feature selection (sparse solutions).
- ✅ Interpretable: non-zero coefficients reveal the active features.
- ✅ Fast for moderate dimensions.

**Cons:**
- ❌ Linear model — same expressiveness limits as OLS/Ridge.
- ❌ When features are correlated, tends to pick one and zero the rest (arbitrarily) — consider ElasticNet in that case.
- ❌ Sensitive to feature scale.
- ❌ Solution is not unique if n_features > n_samples.

**Assumptions:** Linearity; Sparse true signal; Features scaled.

**Complexity:** O(n_samples * n_features) per coordinate-descent iteration

**Key hyperparameters:** `alpha`, `max_iter`, `tol`

## `ols`

**Ordinary Least Squares — unregularized linear regression.**

**When to use:** Baseline regression, well-conditioned design matrix with n >> p and no strong multicollinearity. Useful as a reference point for Ridge/Lasso.

**Pros:**
- ✅ Simplest model — no hyperparameters to tune.
- ✅ Unbiased estimator with minimum variance among linear unbiased models (Gauss-Markov, under assumptions).
- ✅ Extremely fast — closed-form solution.

**Cons:**
- ❌ Overfits badly when p approaches or exceeds n.
- ❌ No regularization — high variance under multicollinearity.
- ❌ Sensitive to outliers and feature scale.
- ❌ Cannot do feature selection.

**Assumptions:** Linearity; Independence of residuals; Homoscedasticity; No (or weak) multicollinearity.

**Complexity:** O(n_samples * n_features^2)

**Key hyperparameters:** `(none — fit_intercept is the only knob)`

## `random_forest`

**Bagging ensemble of decision trees; averages many decorrelated trees.**

**When to use:** Strong default for tabular regression: captures non-linearities and interactions with minimal tuning, robust to overfitting, and gives feature importances. Try this before gradient boosting on new data.

**Pros:**
- ✅ Excellent out-of-the-box accuracy on tabular data.
- ✅ Robust to overfitting via bagging + random feature subsets.
- ✅ No scaling needed; handles non-linearities and interactions natively.
- ✅ Parallelizable (n_jobs=-1) and gives feature importances.
- ✅ Few critical hyperparameters — works well with defaults.

**Cons:**
- ❌ Large ensembles are memory-heavy and slower to predict than a single tree or linear model.
- ❌ Extrapolation is flat (averages leaf means) — fails outside training range; GBM is often better on trends.
- ❌ Less interpretable than a single tree.
- ❌ Generally slightly less accurate than well-tuned gradient boosting on structured data.

**Assumptions:** Rows are i.i.d.; Target is smooth in feature space.

**Complexity:** O(n_estimators * n_samples * n_features * log n_samples)

**Key hyperparameters:** `n_estimators`, `max_depth`, `min_samples_split`, `max_features`

## `ridge`

**L2-regularized linear regression (Tikhonov).**

**When to use:** Linear regression with many correlated or redundant features and risk of overfitting / multicollinearity. Works well when n_samples >= n_features.

**Pros:**
- ✅ Closed-form, fast to train and predict.
- ✅ Reduces variance via shrinkage — improves generalization.
- ✅ Stable under multicollinearity (always has a unique solution).
- ✅ Interpretable coefficients (sign and magnitude).

**Cons:**
- ❌ Assumes a linear relationship between features and target.
- ❌ Does not perform feature selection (coefficients shrink but stay non-zero).
- ❌ Sensitive to feature scale — needs a scaler.
- ❌ Outliers in features or target can dominate.

**Assumptions:** Linearity; Homoscedastic, independent residuals; Features scaled.

**Complexity:** O(n_samples * n_features^2)

**Key hyperparameters:** `alpha (regularization strength)`, `solver`, `tol`

## `svr`

**Support Vector Regression — fits a tube of width epsilon around the target.**

**When to use:** Small-to-medium datasets with non-linear relationships, especially when you care about a margin of tolerance (epsilon-insensitive loss). Strong on high-dimensional data where n_features ≈ n_samples.

**Pros:**
- ✅ Flexible non-linear regression via the kernel trick (RBF/poly).
- ✅ epsilon-tube makes it robust to small target noise.
- ✅ Effective in high dimensions (kernel avoids the curse of dimensionality).
- ✅ Solution is sparse — only support vectors matter.

**Cons:**
- ❌ Scaling is mandatory (distance-based kernel).
- ❌ Training time is roughly O(n_samples^2) to O(n_samples^3) — does NOT scale to large datasets (use RF/GBM or LinearSVR instead).
- ❌ Three hyperparameters (C, epsilon, kernel/gamma) interact and need careful Optuna tuning, all on log scale.
- ❌ Extrapolation is poor — predictions saturate outside the training range.
- ❌ No native feature importances.

**Assumptions:** Features scaled; n_samples not huge (< ~10k); Kernel is appropriate.

**Complexity:** O(n_samples^2) to O(n_samples^3) training; O(n_support_vectors * n_features) predict

**Key hyperparameters:** `C`, `epsilon`, `kernel`, `gamma (RBF/poly)`
