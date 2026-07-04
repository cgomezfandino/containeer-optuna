"""PyTorch MLP (Multi-Layer Perceptron) for tabular regression and classification.

This module defines a flexible :class:`MLP` ``nn.Module`` used by the DL
objective (:func:`~containeer_optuna.optimization.objectives.make_dl_objective`).
It is NOT a sklearn estimator — the DL objective drives the training loop
manually so that epoch-level Optuna pruning (``trial.report`` +
``trial.should_prune``) can be applied.

Architecture is configurable via:
* ``hidden_sizes`` — list of layer widths (e.g. ``[128, 64]``).
* ``dropout`` — dropout rate after each hidden layer.
* ``activation`` — ``"relu"``, ``"tanh"``, or ``"elu"``.
* ``output_dim`` — 1 for regression, ``n_classes`` for classification.
"""

from __future__ import annotations

from typing import Any


class MLP:  # type: ignore[no-untyped-def]
    """Deferred-import wrapper around ``torch.nn.Module``.

    This class is a namespace — the real module is built by
    :func:`build_mlp_module` which imports torch lazily. The wrapper exists
    so the registry (``MODEL_CLASSES``) has a class entry that can be lazily
    loaded.
    """


def build_mlp_module(
    input_dim: int,
    hidden_sizes: list[int],
    output_dim: int = 1,
    dropout: float = 0.1,
    activation: str = "relu",
    task: str = "regression",
) -> Any:
    """Build a PyTorch MLP ``nn.Module`` lazily.

    Args:
        input_dim: Number of input features.
        hidden_sizes: List of hidden layer widths.
        output_dim: 1 for regression, ``n_classes`` for classification.
        dropout: Dropout rate applied after each hidden layer.
        activation: ``"relu"`` / ``"tanh"`` / ``"elu"``.
        task: ``"regression"`` or ``"classification"``.

    Returns:
        A ``torch.nn.Module`` (not yet trained).

    Raises:
        ImportError: If ``torch`` is not installed (install with ``pip install
            containeer-optuna[dl]``).
    """
    try:
        import torch.nn as nn
    except ImportError as e:
        raise ImportError(
            "PyTorch is required for deep learning models. "
            "Install it with: pip install containeer-optuna[dl]"
        ) from e

    act_map = {"relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU}
    act_cls = act_map.get(activation, nn.ReLU)

    layers: list[nn.Module] = []
    prev = input_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(prev, h))
        layers.append(act_cls())
        layers.append(nn.Dropout(dropout))
        prev = h
    layers.append(nn.Linear(prev, output_dim))

    return nn.Sequential(*layers)


def get_loss_fn(task: str) -> Any:
    """Return the appropriate PyTorch loss function for the task.

    Args:
        task: ``"regression"`` or ``"classification"``.

    Returns:
        A ``torch.nn`` loss module (MSELoss for regression, CrossEntropyLoss
        for classification).
    """
    try:
        import torch.nn as nn
    except ImportError as e:
        raise ImportError(
            "PyTorch is required for deep learning models. "
            "Install it with: pip install containeer-optuna[dl]"
        ) from e

    if task == "regression":
        return nn.MSELoss()
    return nn.CrossEntropyLoss()


__all__ = ["MLP", "build_mlp_module", "get_loss_fn"]
