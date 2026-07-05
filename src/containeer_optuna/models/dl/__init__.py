"""Deep learning models (PyTorch) — MLP, CNN, RNN/GRU, Transformer, Embeddings.

Public API:
    MLP, build_mlp_module, get_loss_fn
    build_cnn_module, build_rnn_module
    build_transformer_module, build_tokenizer
    extract_embeddings, embedding_pipeline
    DL_BACKENDS, get_backend
"""

from .backends import DL_BACKENDS, CNNBackend, DLBackend, MLPBackend, RNNBackend, get_backend
from .cnn import build_cnn_module
from .embeddings import embedding_pipeline, extract_embeddings
from .mlp import MLP, build_mlp_module, get_loss_fn
from .rnn import build_rnn_module
from .transformer import build_tokenizer, build_transformer_module

__all__ = [
    "MLP",
    "build_mlp_module",
    "get_loss_fn",
    "build_cnn_module",
    "build_rnn_module",
    "build_transformer_module",
    "build_tokenizer",
    "extract_embeddings",
    "embedding_pipeline",
    "DLBackend",
    "MLPBackend",
    "CNNBackend",
    "RNNBackend",
    "DL_BACKENDS",
    "get_backend",
]
