"""Model registry and factory for containeer-optuna.

This module provides:

* :class:`BaseModel` — abstract base for model entries.
* :func:`suggest_params` — a generic dispatcher that turns a YAML ``param_space``
  into ``trial.suggest_*`` calls. This is the bridge between the declarative
  config and Optuna.
* :func:`get_model` — builds a fresh sklearn estimator for a model name, either
  with its default params (when ``trial`` is None) or with Optuna-sampled params.

The registry is populated from ``config/models.yaml`` plus the Python class
mappings defined in :mod:`containeer_optuna.models.classes`. Adding a new model
only requires editing those two places — no changes to the objective code.

Parameter namespacing
---------------------
When multiple models are searched jointly (e.g. a clustering study that tries
KMeans/DBSCAN/GMM), Optuna requires unique parameter names across the whole
search space. Set ``namespace="<prefix>"`` to prepend e.g. ``kmeans_`` to every
suggested parameter, producing names like ``kmeans_n_clusters``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from sklearn.base import BaseEstimator

from ..config import ModelConfig, load_model_config
from .classes import MODEL_CLASSES

try:
    import optuna
    from optuna.trial import Trial
except ImportError:  # pragma: no cover — optuna is a hard dependency
    optuna = None  # type: ignore[assignment]
    Trial = Any  # type: ignore[misc,assignment]


class BaseModel(ABC):
    """Abstract base for model entries.

    A model entry pairs a sklearn estimator class with a :class:`ModelConfig`
    (default params + Optuna search space). Concrete behavior is implemented in
    :func:`get_model`; this class exists mainly to give the public API a stable
    type to reference (``BaseModel`` is re-exported from the package root).
    """

    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    @property
    def estimator_cls(self) -> type[BaseEstimator]:
        """The sklearn estimator class for this model."""
        cls = MODEL_CLASSES[self.config.name]
        if cls is None:
            raise ImportError(
                f"Model '{self.config.name}' requires an optional dependency that is not installed."
            )
        return cls

    @abstractmethod
    def build(self, trial: Any | None = None, namespace: str = "") -> BaseEstimator:
        """Return a fresh estimator instance.

        When ``trial`` is provided, hyperparameters are sampled from the
        configured ``param_space``; otherwise the ``default_params`` are used.
        """
        raise NotImplementedError


def suggest_params(
    trial: Any,
    param_space: Mapping[str, Mapping[str, Any]],
    namespace: str = "",
) -> dict[str, Any]:
    """Dispatch a YAML ``param_space`` block to ``trial.suggest_*`` calls.

    Each entry must have a ``type`` of ``float``, ``int`` or ``categorical``:

    * ``float`` → ``trial.suggest_float(name, low, high, log=log)``
    * ``int``   → ``trial.suggest_int(name, low, high, log=log)``
    * ``categorical`` → ``trial.suggest_categorical(name, choices)``

    Args:
        trial: An :class:`optuna.trial.Trial` (or any object with matching
            ``suggest_*`` methods).
        param_space: Mapping of param-name → spec.
        namespace: Optional prefix appended to each Optuna param name with ``_``.
            Useful for joint studies over multiple models.

    Returns:
        A dict of ``{param_name: sampled_value}`` with namespaces stripped —
        i.e. kwargs ready to pass to the estimator constructor.

    Raises:
        ValueError: If a param spec has an unknown ``type``.
    """
    params: dict[str, Any] = {}
    for raw_name, spec in param_space.items():
        kind = spec.get("type")
        optuna_name = f"{namespace}_{raw_name}" if namespace else raw_name

        if kind == "float":
            params[raw_name] = trial.suggest_float(
                optuna_name,
                float(spec["low"]),
                float(spec["high"]),
                log=bool(spec.get("log", False)),
            )
        elif kind == "int":
            params[raw_name] = trial.suggest_int(
                optuna_name,
                int(spec["low"]),
                int(spec["high"]),
                log=bool(spec.get("log", False)),
            )
        elif kind == "categorical":
            params[raw_name] = trial.suggest_categorical(
                optuna_name,
                list(spec["choices"]),
            )
        else:
            raise ValueError(
                f"Unknown param_space type '{kind}' for parameter '{raw_name}' "
                f"(expected 'float', 'int', or 'categorical')"
            )
    return params


def get_model(
    name: str,
    trial: Any | None = None,
    namespace: str = "",
    base_dir: str | None = None,
    **overrides: Any,
) -> BaseEstimator:
    """Build a fresh sklearn estimator for the named model.

    Args:
        name: Model registry key (e.g. ``"ridge"``, ``"kmeans"``, ``"pca"``).
        trial: Optional Optuna trial. When provided, hyperparameters listed in
            the model's ``param_space`` are sampled; otherwise ``default_params``
            are used.
        namespace: Optional prefix for Optuna parameter names (joint studies).
        base_dir: Optional base directory override for locating ``models.yaml``.
        **overrides: Extra constructor kwargs that win over both defaults and
            sampled values (e.g. ``random_state=42``).

    Returns:
        A fresh, un-fitted sklearn estimator instance.

    Raises:
        KeyError: If ``name`` is not in the class registry.
    """
    if name not in MODEL_CLASSES:
        raise KeyError(f"Model '{name}' is not registered. Known models: {sorted(MODEL_CLASSES)}")

    cls = MODEL_CLASSES[name]
    if cls is None:
        raise ImportError(
            f"Model '{name}' requires an optional dependency that is not installed "
            f"(e.g. umap-learn for UMAP). Install it and retry."
        )

    config = load_model_config(name, base_dir=base_dir)

    if trial is not None and config.param_space:
        params = suggest_params(trial, config.param_space, namespace=namespace)
    else:
        params = dict(config.default_params)

    params.update(overrides)
    return cls(**params)


__all__ = [
    "BaseModel",
    "suggest_params",
    "get_model",
]
