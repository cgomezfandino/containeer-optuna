# Regression models

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
