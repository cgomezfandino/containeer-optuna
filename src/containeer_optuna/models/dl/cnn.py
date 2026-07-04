"""PyTorch CNN (Convolutional Neural Network) for image classification.

Builds a configurable ConvNet: stacks of Conv2d → ReLU → MaxPool2d, then
Flatten → Linear → output. The number of conv channels, kernel size, and FC
layer sizes are all configurable — Optuna searches over them via the
``param_space`` in ``models.yaml``.
"""

from __future__ import annotations

from typing import Any


def build_cnn_module(
    input_shape: tuple[int, ...],
    conv_channels: list[int],
    kernel_size: int,
    fc_sizes: list[int],
    output_dim: int,
    dropout: float = 0.1,
) -> Any:
    """Build a PyTorch CNN ``nn.Module`` lazily.

    Args:
        input_shape: ``(C, H, W)`` of the input images (without batch dim).
        conv_channels: List of output channels per conv layer, e.g. ``[32, 64]``.
        kernel_size: Conv kernel size (odd, e.g. 3 or 5).
        fc_sizes: List of hidden widths for the FC block after flatten.
        output_dim: Number of classes.
        dropout: Dropout rate after each FC layer.

    Returns:
        A ``torch.nn.Module`` (not yet trained).

    Raises:
        ImportError: If ``torch`` is not installed.
    """
    try:
        import torch.nn as nn
    except ImportError as e:
        raise ImportError(
            "PyTorch is required for deep learning models. "
            "Install it with: pip install containeer-optuna[dl]"
        ) from e

    channels, height, width = input_shape
    padding = kernel_size // 2

    layers: list[nn.Module] = []
    prev_ch = channels
    h, w = height, width
    for ch in conv_channels:
        layers.append(nn.Conv2d(prev_ch, ch, kernel_size, padding=padding))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(2))
        prev_ch = ch
        h, w = h // 2, w // 2

    layers.append(nn.Flatten())

    flat_dim = prev_ch * h * w
    prev = flat_dim
    for fc in fc_sizes:
        layers.append(nn.Linear(prev, fc))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        prev = fc
    layers.append(nn.Linear(prev, output_dim))

    return nn.Sequential(*layers)


__all__ = ["build_cnn_module"]
