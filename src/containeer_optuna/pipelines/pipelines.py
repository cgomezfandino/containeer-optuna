"""Pipeline assembly for containeer-optuna.

A pipeline chains optional preprocessing steps (scaler, reducer) with a
mandatory final estimator (regressor or clusterer). Building pipelines from
named registry entries keeps YAML configs concise and prevents leakage
(scaling is fitted inside the CV loop, not globally beforehand).
"""

from __future__ import annotations

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from ..models import get_model


class BasePipeline:
    """Thin wrapper around :class:`sklearn.pipeline.Pipeline`.

    The wrapper exists so that the public API exposes a stable
    :class:`BasePipeline` / :func:`get_pipeline` pair, and so that clustering
    objectives can access the final estimator (for ``fit_predict``) by a
    well-known name (``"model"``).
    """

    #: Name used for the final estimator step. Objectives rely on this.
    FINAL_STEP = "model"

    def __init__(self, steps: list[tuple[str, BaseEstimator]]) -> None:
        self.steps = steps

    def build(self) -> Pipeline:
        """Return the underlying sklearn :class:`Pipeline`."""
        return Pipeline(self.steps)

    @property
    def final_estimator(self) -> BaseEstimator:
        """The final estimator of the pipeline (the model step)."""
        return self.steps[-1][1]


def get_pipeline(
    model: str,
    scaler: str | None = None,
    reducer: str | None = None,
    trial: object | None = None,
    namespace: str | None = None,
    base_dir: str | None = None,
    **model_overrides: object,
) -> Pipeline:
    """Assemble a sklearn :class:`Pipeline` from registry names.

    The order is always ``[scaler?] -> [reducer?] -> model``. All steps are
    instantiated via :func:`~containeer_optuna.models.get_model`, so they share
    the same YAML-driven defaults / param-space sampling.

    The reducer (if any) is sampled under its own namespace (e.g. ``pca_*``),
    and the final model under its own namespace (e.g. ``kmeans_*``), so the
    two search spaces never collide when both are tunable.

    Args:
        model: Final estimator registry name (e.g. ``"ridge"``, ``"kmeans"``).
        scaler: Optional scaler name (``"standard_scaler"``, ``"minmax_scaler"``).
        reducer: Optional reducer name (``"pca"``, ``"umap"``).
        trial: Optional Optuna trial for param sampling.
        namespace: Optional namespace override for the *model* params only.
            If None, defaults to the model name.
        base_dir: Optional base directory override.
        **model_overrides: Extra kwargs for the model constructor.

    Returns:
        A fresh, un-fitted sklearn :class:`Pipeline`.

    Raises:
        KeyError: If any name is not in the registry.
    """
    steps: list[tuple[str, BaseEstimator]] = []

    if scaler:
        steps.append(("scaler", get_model(scaler, trial=None, base_dir=base_dir)))
    if reducer:
        # Reducers use their own name as namespace — never the model's.
        steps.append(
            (
                "reducer",
                get_model(reducer, trial=trial, namespace=reducer, base_dir=base_dir),
            )
        )

    model_ns = namespace or model
    final = get_model(
        model,
        trial=trial,
        namespace=model_ns,
        base_dir=base_dir,
        **model_overrides,
    )
    steps.append((BasePipeline.FINAL_STEP, final))

    return Pipeline(steps)


__all__ = ["BasePipeline", "get_pipeline"]
