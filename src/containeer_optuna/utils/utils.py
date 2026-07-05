"""Utility helpers: reproducibility, logging, and IO.

These are small, dependency-light helpers used across the framework:

* :func:`seed_all` — set every relevant RNG seed for reproducible experiments.
* :func:`get_logger` — configure a stdlib logger with a Rich console handler
  (falls back to plain stderr if Rich is unavailable).
* :func:`ensure_dir`, :func:`save_json`, :func:`save_predictions` — small file
  helpers for persisting artifacts under the configured ``results_dir``.
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ..config import settings

_LOGGER_CONFIGURED = False


def seed_all(random_state: int = 42) -> None:
    """Seed Python, NumPy and the OS env for reproducible runs.

    Args:
        random_state: The seed to use across all RNGs.
    """
    os.environ["PYTHONHASHSEED"] = str(random_state)
    random.seed(random_state)
    np.random.seed(random_state)
    # Optuna samplers accept their own seed; we don't set it here to avoid
    # importing optuna at module-import time.


def get_logger(name: str = "containeer_optuna") -> logging.Logger:
    """Return a configured :class:`logging.Logger`.

    The first call configures the handler/formatter; subsequent calls just
    return ``logging.getLogger(name)``.
    """
    global _LOGGER_CONFIGURED
    logger = logging.getLogger(name)

    if not _LOGGER_CONFIGURED:
        level = getattr(logging, settings.log_level.upper(), logging.INFO)
        logger.setLevel(level)

        try:
            from rich.logging import RichHandler

            handler: logging.Handler = RichHandler(
                rich_tracebacks=True, show_path=False, log_time_format="[%X]"
            )
            formatter = logging.Formatter("%(message)s")
        except ImportError:  # pragma: no cover — rich is a hard dep, but be safe
            handler = logging.StreamHandler(sys.stderr)
            formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")

        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
        _LOGGER_CONFIGURED = True

    return logger


def ensure_dir(path: str | Path) -> Path:
    """Create ``path`` (and parents) if it does not exist; return as :class:`Path`."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def results_dir(base: str | Path | None = None) -> Path:
    """Return (and create) the configured results directory."""
    return ensure_dir(base if base else settings.results_dir)


def save_json(obj: Any, path: str | Path) -> Path:
    """Serialize ``obj`` to ``path`` as pretty-printed JSON.

    Numpy types are coerced to Python builtins. Parent directories are created.
    """
    p = Path(path)
    ensure_dir(p.parent)

    def _default(o: Any) -> Any:
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

    with open(p, "w") as f:
        json.dump(obj, f, indent=2, default=_default, sort_keys=False)
    return p


def save_predictions(
    predictions: Any,
    experiment_name: str,
    base: str | Path | None = None,
) -> Path:
    """Persist predictions to ``<results>/<experiment>_predictions.npy``.

    For clustering this is the label vector; for regression the predicted y.
    """
    base_dir = results_dir(base)
    path = base_dir / f"{experiment_name}_predictions.npy"
    np.save(path, np.asarray(predictions))
    return path


def save_model(
    model: Any,
    experiment_name: str,
    base: str | Path | None = None,
) -> Path:
    """Persist a fitted model to ``<results>/<experiment>_best_model.joblib``.

    Uses joblib (already a dependency) for serialization. Works with any
    pickle-compatible object (sklearn Pipeline, estimator, etc.).

    Args:
        model: The fitted model/pipeline to save.
        experiment_name: Used in the output filename.
        base: Optional override for the results directory.

    Returns:
        The path to the saved ``.joblib`` file.
    """
    import joblib

    base_dir = results_dir(base)
    path = base_dir / f"{experiment_name}_best_model.joblib"
    joblib.dump(model, path)
    return path


def save_model_onnx(
    model: Any,
    experiment_name: str,
    n_features: int,
    task: str = "regression",
    base: str | Path | None = None,
) -> Path:
    """Export a fitted sklearn Pipeline/estimator to ONNX format.

    Requires ``pip install containeer-optuna[onnx]`` (skl2onnx + onnx).

    Args:
        model: A fitted sklearn Pipeline or estimator.
        experiment_name: Used in the output filename.
        n_features: Number of input features (needed to define the ONNX input shape).
        task: ``"regression"``, ``"classification"``, or ``"clustering"``. Determines
            the ONNX output type.
        base: Optional override for the results directory.

    Returns:
        The path to the saved ``.onnx`` file.

    Raises:
        ImportError: If ``skl2onnx`` is not installed.
    """
    try:
        from skl2onnx import to_onnx
    except ImportError as e:
        raise ImportError(
            "skl2onnx is required for ONNX export. "
            "Install it with: pip install containeer-optuna[onnx]"
        ) from e

    base_dir = results_dir(base)
    path = base_dir / f"{experiment_name}_best_model.onnx"

    import numpy as np

    # Determine options from task.
    if task == "classification":
        options = {id(model): {"zipmap": False}}
    else:
        options = None

    onnx_model = to_onnx(model, np.zeros((1, n_features), dtype=np.float32), options=options)
    with open(path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    return path


def study_summary(study: Any) -> dict[str, Any]:
    """Return a JSON-serializable summary dict of an Optuna study.

    Captures best trial value/params/user_attrs plus trial count and direction.
    """
    best = study.best_trial if study.trials else None
    return {
        "study_name": study.study_name,
        "direction": str(study.direction),
        "n_trials": len(study.trials),
        "best_value": float(study.best_value) if study.trials else None,
        "best_params": dict(best.params) if best else {},
        "best_user_attrs": dict(best.user_attrs) if best and best.user_attrs else {},
    }


__all__ = [
    "seed_all",
    "get_logger",
    "ensure_dir",
    "results_dir",
    "save_json",
    "save_predictions",
    "study_summary",
]
