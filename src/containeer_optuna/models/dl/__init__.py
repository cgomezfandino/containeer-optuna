"""Deep learning models (PyTorch) — MLP for tabular data.

Public API:
    MLP, build_mlp_module, get_loss_fn
"""

from .mlp import MLP, build_mlp_module, get_loss_fn

__all__ = ["MLP", "build_mlp_module", "get_loss_fn"]
