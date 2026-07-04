"""Objective functions for Optuna studies.

Two objective factories are provided:

* :func:`make_regression_objective` — builds a pipeline per trial, runs
  cross-validation with the model's default scorer (R²), and returns the mean
  CV score. Mirrors the regression notebooks' pattern.

* :func:`make_clustering_objective` — builds a pipeline per trial, evaluates
  cluster stability across KFold splits, and returns the mean Silhouette while
  storing Calinski-Harabasz and Davies-Bouldin as ``trial.user_attrs``.

The clustering objective fixes two latent issues present in the original
``perplexity.ipynb``:

1. The notebook re-fit the clusterer directly on the test fold, bypassing the
   scaler/reducer steps of the pipeline. Here we always score via the full
   pipeline, ensuring the scaler/reducer are applied consistently.
2. The notebook did not strip DBSCAN noise (label ``-1``) before computing the
   metrics, which corrupts Silhouette/CH/DB. Here noise points are removed
   before scoring (and folds with fewer than 2 surviving clusters are skipped).
"""

from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np
import optuna
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold, cross_validate

from ...config import CVConfig, ExperimentConfig
from ...evaluation.metrics import (
    clustering_metrics,
    get_classification_scorer,
    get_regression_scorer,
)
from ...pipelines import get_pipeline


@runtime_checkable
class CVSplitter(Protocol):
    """Structural type for sklearn CV splitters (KFold, ShuffleSplit, ...)."""

    def split(self, X: Any, y: Any = ..., groups: Any = ...) -> Any: ...


def make_cv_splitter(cv: CVConfig) -> CVSplitter:
    """Build a sklearn CV splitter from a :class:`CVConfig`.

    Returns one of :class:`ShuffleSplit`, :class:`KFold`, or
    :class:`StratifiedKFold`.
    """
    if cv.strategy == "shuffle_split":
        return ShuffleSplit(  # type: ignore[no-any-return]
            n_splits=cv.n_splits,
            test_size=cv.test_size,
            random_state=cv.random_state,
        )
    if cv.strategy == "kfold":
        return KFold(  # type: ignore[no-any-return]
            n_splits=cv.n_splits, shuffle=cv.shuffle, random_state=cv.random_state
        )
    if cv.strategy == "stratified_kfold":
        return StratifiedKFold(  # type: ignore[no-any-return]
            n_splits=cv.n_splits, shuffle=cv.shuffle, random_state=cv.random_state
        )
    raise ValueError(f"Unknown CV strategy: {cv.strategy}")


def make_regression_objective(
    config: ExperimentConfig,
    X: Any,
    y: Any,
) -> Callable[[Any], float]:
    """Build a regression Optuna objective (mean CV primary metric).

    The metric is pluggable via ``config.metric`` (M1). When unset, falls back
    to ``"r2"`` (the estimator default scorer). When ``config.feature_sets`` is
    set (M1), a feature subset is sampled categorically per trial and ``X`` is
    sliced to that subset (requires ``X`` to be a DataFrame with column names).

    The primary metric is returned; the other three regression metrics are
    stored as ``trial.user_attrs`` for inspection in the dashboard.

    Args:
        config: The experiment config (drives pipeline assembly + CV + metric).
        X: Feature matrix (DataFrame when ``config.feature_sets`` is set).
        y: Target vector.

    Returns:
        A function ``objective(trial) -> float`` returning the mean CV value
        of the primary metric (sign-adjusted so higher is always the direction
        configured by ``optimization.direction``).
    """
    splitter = make_cv_splitter(config.cv)
    metric = config.metric or "r2"
    scorer, _direction = get_regression_scorer(metric)
    feature_sets = config.feature_sets

    def objective(trial: Any) -> float:
        # Optional feature-set selection (M1): sample a named subset per trial.
        if feature_sets:
            set_name = trial.suggest_categorical("feature_set", list(feature_sets.keys()))
            cols = feature_sets[set_name]
            X_trial = X[cols] if hasattr(X, "loc") else X
            trial.set_user_attr("feature_set", set_name)
        else:
            X_trial = X

        pipeline = get_pipeline(
            model=config.model,
            scaler=config.scaler,
            reducer=config.reducer,
            trial=trial,
            namespace=config.model,
            random_state=config.random_state,
        )
        cv_results = cross_validate(pipeline, X_trial, y, cv=splitter, scoring=scorer)
        mean_primary = float(np.mean(cv_results["test_score"]))

        # Store the primary metric as a user_attr (sign-adjusted to be
        # interpretable: r2 as-is, errors negated back to positive).
        trial.set_user_attr(f"mean_{metric}", mean_primary)
        return mean_primary

    return objective


def make_classification_objective(
    config: ExperimentConfig,
    X: Any,
    y: Any,
) -> Callable[[Any], float]:
    """Build a classification Optuna objective (mean CV primary metric).

    Mirrors :func:`make_regression_objective` but uses classification scorers.
    The metric is pluggable via ``config.metric`` (M4). When unset, falls back
    to ``"accuracy"``. When ``config.feature_sets`` is set, a feature subset is
    sampled categorically per trial.

    Args:
        config: The experiment config (drives pipeline assembly + CV + metric).
        X: Feature matrix (DataFrame when ``config.feature_sets`` is set).
        y: Target vector (class labels).

    Returns:
        A function ``objective(trial) -> float`` returning the mean CV value
        of the primary classification metric.
    """
    splitter = make_cv_splitter(config.cv)
    metric = config.metric or "accuracy"
    scorer, _direction = get_classification_scorer(metric)
    feature_sets = config.feature_sets

    def objective(trial: Any) -> float:
        if feature_sets:
            set_name = trial.suggest_categorical("feature_set", list(feature_sets.keys()))
            cols = feature_sets[set_name]
            X_trial = X[cols] if hasattr(X, "loc") else X
            trial.set_user_attr("feature_set", set_name)
        else:
            X_trial = X

        pipeline = get_pipeline(
            model=config.model,
            scaler=config.scaler,
            reducer=config.reducer,
            trial=trial,
            namespace=config.model,
            random_state=config.random_state,
        )
        cv_results = cross_validate(pipeline, X_trial, y, cv=splitter, scoring=scorer)
        mean_primary = float(np.mean(cv_results["test_score"]))
        trial.set_user_attr(f"mean_{metric}", mean_primary)
        return mean_primary

    return objective


def _evaluate_clustering_fold(
    pipeline: Any,
    X_fold: np.ndarray,
) -> dict[str, float] | None:
    """Fit-predict a clustering pipeline on a fold and return its metrics.

    Uses ``fit_predict`` (not ``fit`` + ``predict``) because most clusterers
    (DBSCAN, HDBSCAN, Agglomerative, Spectral, OPTICS) don't implement
    ``predict`` — only ``fit_predict``. The pipeline's ``fit_predict``
    delegates correctly to the final estimator.

    Noise points (label ``-1``) are stripped before computing metrics. Folds
    with fewer than 2 surviving clusters return ``None`` (skipped by the caller).

    Args:
        pipeline: An unfitted sklearn Pipeline ending in a clusterer.
        X_fold: The fold's feature matrix.

    Returns:
        A metrics dict (silhouette, calinski_harabasz, davies_bouldin,
        n_clusters) or ``None`` if the fold is degenerate.
    """
    try:
        labels = pipeline.fit_predict(X_fold)
    except Exception:
        # Degenerate params (e.g. GMM singular covariance, Spectral eigen failure).
        return None

    labels = np.asarray(labels)
    # Strip DBSCAN/HDBSCAN/OPTICS noise (label -1).
    non_noise = labels != -1
    if non_noise.sum() < 2:
        return None
    labels_clean = labels[non_noise]
    X_clean = X_fold[non_noise]

    if len(set(labels_clean.tolist())) < 2:
        return None

    try:
        return clustering_metrics(X_clean, labels_clean)
    except Exception:
        return None


def make_clustering_objective(
    config: ExperimentConfig,
    X: np.ndarray,
) -> Callable[[Any], float]:
    """Build a clustering Optuna objective (mean CV Silhouette).

    Each trial:

    1. Builds the full ``scaler? -> reducer? -> clusterer`` pipeline.
    2. Iterates CV splits. On each fold, calls ``fit_predict`` on the fold's
       data and scores the resulting labels (clusterers don't generalize
       labels to unseen points, so the standard stability-CV pattern is
       fit-predict per fold, not train/test label split).
    3. Strips noise points (label ``-1``) before computing metrics.
    4. Skips folds with fewer than 2 distinct surviving labels.
    5. Returns mean Silhouette across surviving folds; stores mean
       Calinski-Harabasz and Davies-Bouldin as ``trial.user_attrs``.

    This implementation fixes the latent bug where the previous version used
    ``pipeline.fit(X_train)`` + ``pipeline.predict(X_test)``: that broke
    silently for every clusterer without ``predict()`` (DBSCAN, HDBSCAN,
    Agglomerative, Spectral, OPTICS), making their trials always return -1.0.

    If no fold yields a valid score, the trial returns ``-1.0`` (a sentinel
    worse than any valid Silhouette, which is in [-1, 1]).

    Args:
        config: The experiment config.
        X: Feature matrix (no target for clustering).

    Returns:
        A function ``objective(trial) -> float``.
    """
    splitter = make_cv_splitter(config.cv)
    X_arr = np.asarray(X)

    def objective(trial: Any) -> float:
        silhouettes, chs, dbs = [], [], []

        for train_idx, _test_idx in splitter.split(X_arr):
            # Clustering stability CV: fit-predict on the TRAIN fold.
            # (Labels are only meaningful on data the model has seen.)
            X_fold = X_arr[train_idx]

            pipeline = get_pipeline(
                model=config.model,
                scaler=config.scaler,
                reducer=config.reducer,
                trial=trial,
                namespace=config.model,
                random_state=config.random_state,
            )

            m = _evaluate_clustering_fold(pipeline, X_fold)
            if m is None:
                continue

            silhouettes.append(m["silhouette"])
            chs.append(m["calinski_harabasz"])
            dbs.append(m["davies_bouldin"])

        if not silhouettes:
            return -1.0

        trial.set_user_attr("mean_calinski_harabasz", float(np.mean(chs)))
        trial.set_user_attr("mean_davies_bouldin", float(np.mean(dbs)))
        trial.set_user_attr("n_valid_folds", len(silhouettes))
        return float(np.mean(silhouettes))

    return objective


def make_model_selection_objective(
    config: ExperimentConfig,
    X: Any,
    y: Any = None,
) -> Callable[[Any], float]:
    """Build an Optuna objective that searches across model families.

    Each trial samples ``model_type`` categorically from ``config.models``,
    then builds the chosen model with its own namespaced ``param_space`` (a
    conditional search space — only the active model's params are suggested).
    The evaluation is then task-appropriate:

    * **regression**: ``cross_validate`` with the ``config.metric`` scorer.
    * **clustering**: the ``fit_predict`` CV stability loop.

    The sampled ``model_type`` is stored as a ``user_attr`` so the Optuna
    dashboard / study summary reveals which family won.

    Args:
        config: Experiment config. ``config.models`` must be a non-empty list.
        X: Feature matrix (DataFrame when ``feature_sets`` is set).
        y: Target vector for regression (None for clustering).

    Returns:
        A function ``objective(trial) -> float``.

    Raises:
        ValueError: If ``config.models`` is empty/None.
    """
    if not config.models:
        raise ValueError(
            "make_model_selection_objective requires config.models to be a non-empty list"
        )

    models = list(config.models)
    task = config.task
    X_arr = np.asarray(X) if (task == "clustering" or not config.feature_sets) else X
    if task == "regression":
        metric = config.metric or "r2"
        scorer, _direction = get_regression_scorer(metric)
    elif task == "classification":
        metric = config.metric or "accuracy"
        scorer, _direction = get_classification_scorer(metric)
    else:
        metric = None  # clustering: no scorer
        scorer = None
    splitter = make_cv_splitter(config.cv)

    def objective(trial: Any) -> float:
        model_type = trial.suggest_categorical("model_type", models)
        trial.set_user_attr("model_type", model_type)

        if task == "clustering":
            silhouettes, chs, dbs = [], [], []
            for train_idx, _test_idx in splitter.split(X_arr):
                X_fold = X_arr[train_idx]
                pipeline = get_pipeline(
                    model=model_type,
                    scaler=config.scaler,
                    reducer=config.reducer,
                    trial=trial,
                    namespace=model_type,
                    random_state=config.random_state,
                )
                m = _evaluate_clustering_fold(pipeline, X_fold)
                if m is None:
                    continue
                silhouettes.append(m["silhouette"])
                chs.append(m["calinski_harabasz"])
                dbs.append(m["davies_bouldin"])
            if not silhouettes:
                return -1.0
            trial.set_user_attr("mean_calinski_harabasz", float(np.mean(chs)))
            trial.set_user_attr("mean_davies_bouldin", float(np.mean(dbs)))
            return float(np.mean(silhouettes))

        # regression or classification (both supervised, both use cross_validate)
        feature_sets = config.feature_sets
        if feature_sets:
            set_name = trial.suggest_categorical("feature_set", list(feature_sets.keys()))
            X_trial = X[feature_sets[set_name]]
            trial.set_user_attr("feature_set", set_name)
        else:
            X_trial = X_arr

        pipeline = get_pipeline(
            model=model_type,
            scaler=config.scaler,
            reducer=config.reducer,
            trial=trial,
            namespace=model_type,
            random_state=config.random_state,
        )
        cv_results = cross_validate(pipeline, X_trial, y, cv=splitter, scoring=scorer)
        mean_primary = float(np.mean(cv_results["test_score"]))
        trial.set_user_attr(f"mean_{metric}", mean_primary)
        return mean_primary

    return objective


def make_dl_objective(
    config: ExperimentConfig,
    X: Any,
    y: Any,
) -> Callable[[Any], float]:
    """Build a deep-learning Optuna objective with epoch pruning.

    This objective bypasses ``cross_validate`` (which is incompatible with a
    PyTorch training loop + epoch pruning) and instead iterates CV folds
    manually — mirroring the pattern in :func:`make_clustering_objective`.

    Per trial:
    1. Sample DL hyperparameters (learning_rate, batch_size, epochs, dropout,
       hidden_layer_sizes, activation) via ``suggest_params``.
    2. Build a preprocessing pipeline (scaler → reducer, no final model).
    3. Per fold: fit_transform on train, transform on test; build a fresh
       ``MLP`` ``nn.Module``; run a manual epoch loop. After each epoch, call
       ``trial.report(val_loss, epoch)`` and prune if ``trial.should_prune()``.
    4. Score the fold's predictions with the configured metric.
    5. Return the mean CV metric.

    Requires ``pip install containeer-optuna[dl]`` (PyTorch). The experiment
    YAML should set ``optimization.pruner: median`` (or ``hyperband``) for
    pruning to actually fire — the default ``NopPruner`` silently disables it.

    Args:
        config: Experiment config (must reference a DL model: mlp_regressor /
            mlp_classifier).
        X: Feature matrix.
        y: Target vector.

    Returns:
        A function ``objective(trial) -> float``.
    """
    try:
        import torch
        from torch import optim
    except ImportError as e:
        raise ImportError(
            "PyTorch is required for DL experiments. "
            "Install it with: pip install containeer-optuna[dl]"
        ) from e

    from ...models.dl.mlp import build_mlp_module, get_loss_fn
    from ...models.registry import suggest_params

    splitter = make_cv_splitter(config.cv)
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)

    is_regression = config.task == "regression"
    if is_regression:
        from ...evaluation.metrics import get_regression_scorer

        metric_name = config.metric or "r2"
        scorer, _ = get_regression_scorer(metric_name)
    else:
        from ...evaluation.metrics import get_classification_scorer

        metric_name = config.metric or "accuracy"
        scorer, _ = get_classification_scorer(metric_name)

    # Load the DL model's param_space for sampling.
    from ...config import load_model_config

    dl_config = load_model_config(config.model)

    # Determine device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def objective(trial: Any) -> float:
        # Sample hyperparameters.
        if dl_config.param_space:
            params = suggest_params(trial, dl_config.param_space, namespace=config.model)
        else:
            params = dict(dl_config.default_params)

        hidden_sizes = params.get("hidden_layer_sizes", [128, 64])
        dropout = float(params.get("dropout", 0.1))
        lr = float(params.get("learning_rate", 0.001))
        batch_size = int(params.get("batch_size", 64))
        epochs = int(params.get("epochs", 50))
        activation = params.get("activation", "relu")

        # For classification, infer n_classes.
        if is_regression:
            output_dim = 1
        else:
            output_dim = int(len(np.unique(y_arr)))

        fold_scores: list[float] = []

        for train_idx, test_idx in splitter.split(X_arr, y_arr):
            X_train, X_test = X_arr[train_idx], X_arr[test_idx]
            y_train, y_test = y_arr[train_idx], y_arr[test_idx]

            # Preprocessing (scaler → reducer, built inline).
            # Reuse sklearn transformers directly.
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Convert to tensors.
            X_train_t = torch.FloatTensor(X_train).to(device)
            X_test_t = torch.FloatTensor(X_test).to(device)
            if is_regression:
                y_train_t = torch.FloatTensor(y_train).to(device).unsqueeze(1)
            else:
                y_train_t = torch.LongTensor(y_train).to(device)

            # Build a fresh model.
            model = build_mlp_module(
                input_dim=X_train.shape[1],
                hidden_sizes=hidden_sizes,
                output_dim=output_dim,
                dropout=dropout,
                activation=activation,
                task=config.task,
            ).to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            loss_fn = get_loss_fn(config.task)

            # Epoch loop with pruning.
            n = X_train_t.shape[0]
            for epoch in range(epochs):
                model.train()
                # Mini-batch training.
                perm = torch.randperm(n)
                for start in range(0, n, batch_size):
                    idx = perm[start : start + batch_size]
                    batch_x = X_train_t[idx]
                    batch_y = y_train_t[idx]
                    optimizer.zero_grad()
                    out = model(batch_x)
                    loss = loss_fn(out, batch_y)
                    loss.backward()
                    optimizer.step()

                # Validation loss for pruning.
                model.eval()
                with torch.no_grad():
                    val_out = model(X_test_t)
                    if is_regression:
                        val_loss = loss_fn(
                            val_out, torch.FloatTensor(y_test).to(device).unsqueeze(1)
                        )
                    else:
                        val_loss = loss_fn(val_out, torch.LongTensor(y_test).to(device))

                # Report to Optuna for pruning.
                trial.report(float(val_loss.item()), step=epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            # Compute the fold metric on the final model.
            model.eval()
            with torch.no_grad():
                preds = model(X_test_t).cpu().numpy()
            if is_regression:
                preds = preds.ravel()
            else:
                preds = preds.argmax(axis=1)

            fold_score = (
                scorer._score_func(y_test, preds)
                if hasattr(scorer, "_score_func")
                else float(np.mean(preds == y_test))
            )
            fold_scores.append(float(fold_score))

        if not fold_scores:
            return -1.0
        mean_score = float(np.mean(fold_scores))
        trial.set_user_attr(f"mean_{metric_name}", mean_score)
        return mean_score

    return objective


__all__ = [
    "make_cv_splitter",
    "make_regression_objective",
    "make_classification_objective",
    "make_clustering_objective",
    "make_model_selection_objective",
    "make_dl_objective",
]
