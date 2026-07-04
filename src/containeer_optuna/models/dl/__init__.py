"""Deep learning models (PyTorch) — MLP, CNN, RNN/GRU for tabular/image/sequence data.

Public API:
    MLP, build_mlp_module, get_loss_fn
    build_cnn_module, build_rnn_module
    DL_BACKENDS, get_backend
"""

from .backends import DL_BACKENDS, CNNBackend, DLBackend, MLPBackend, RNNBackend, get_backend
from .cnn import build_cnn_module
from .mlp import MLP, build_mlp_module, get_loss_fn
from .rnn import build_rnn_module

__all__ = [
    "MLP",
    "build_mlp_module",
    "get_loss_fn",
    "build_cnn_module",
    "build_rnn_module",
    "DLBackend",
    "MLPBackend",
    "CNNBackend",
    "RNNBackend",
    "DL_BACKENDS",
    "get_backend",
]
