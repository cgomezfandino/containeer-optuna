"""DL backends: pluggable architecture-specific modules for the DL objective.

Each backend provides two callables that ``make_dl_objective`` calls:

* ``build_module(params, input_shape, output_dim, task)`` — builds the nn.Module.
* ``prepare_fold(X_train, X_test, y_train, y_test, device)`` — normalizes and
  converts data to the right tensor shape.

This abstraction lets the shared training/pruning loop in
:func:`~containeer_optuna.optimization.objectives.make_dl_objective` work
across MLP (2D), CNN (4D), and RNN (3D) architectures without duplication.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class DLBackend(Protocol):
    """Protocol for pluggable DL architecture backends."""

    def build_module(
        self, params: dict[str, Any], input_shape: tuple[int, ...], output_dim: int, task: str
    ) -> Any: ...

    def prepare_fold(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        device: Any,
        task: str,
    ) -> Any: ...


class MLPBackend:
    """Backend for tabular MLP models (2D tensors, StandardScaler)."""

    def build_module(
        self, params: dict[str, Any], input_shape: tuple[int, ...], output_dim: int, task: str
    ) -> Any:
        from .mlp import build_mlp_module

        input_dim = input_shape[1] if len(input_shape) > 1 else input_shape[0]
        return build_mlp_module(
            input_dim=input_dim,
            hidden_sizes=params.get("hidden_layer_sizes", [128, 64]),
            output_dim=output_dim,
            dropout=float(params.get("dropout", 0.1)),
            activation=params.get("activation", "relu"),
            task=task,
        )

    def prepare_fold(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        device: Any,
        task: str,
    ) -> Any:
        import torch
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        X_train_t = torch.FloatTensor(X_train_s).to(device)
        X_test_t = torch.FloatTensor(X_test_s).to(device)
        if task == "regression":
            y_train_t = torch.FloatTensor(y_train).to(device).unsqueeze(1)
            y_test_t = torch.FloatTensor(y_test).to(device)
        else:
            y_train_t = torch.LongTensor(y_train).to(device)
            y_test_t = torch.LongTensor(y_test).to(device)

        return X_train_t, X_test_t, y_train_t, y_test_t


class CNNBackend:
    """Backend for CNN image classification (4D tensors, per-channel normalization)."""

    def build_module(
        self, params: dict[str, Any], input_shape: tuple[int, ...], output_dim: int, task: str
    ) -> Any:
        from .cnn import build_cnn_module

        # input_shape is (C, H, W) — passed directly.
        return build_cnn_module(
            input_shape=input_shape,
            conv_channels=params.get("conv_channels", [32, 64]),
            kernel_size=int(params.get("kernel_size", 3)),
            fc_sizes=params.get("fc_sizes", [128]),
            output_dim=output_dim,
            dropout=float(params.get("dropout", 0.1)),
        )

    def prepare_fold(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        device: Any,
        task: str,
    ) -> Any:
        import torch

        # Normalize to [0, 1] (assume pixel values in [0, 255]).
        if X_train.max() > 1.0:
            X_train = X_train.astype(np.float32) / 255.0
            X_test = X_test.astype(np.float32) / 255.0

        X_train_t = torch.FloatTensor(X_train).to(device)
        X_test_t = torch.FloatTensor(X_test).to(device)
        y_train_t = torch.LongTensor(y_train).to(device)
        y_test_t = torch.LongTensor(y_test).to(device)

        return X_train_t, X_test_t, y_train_t, y_test_t


class RNNBackend:
    """Backend for RNN/GRU/LSTM sequence classification (3D tensors)."""

    def build_module(
        self, params: dict[str, Any], input_shape: tuple[int, ...], output_dim: int, task: str
    ) -> Any:
        from .rnn import build_rnn_module

        # input_shape is (seq_len, input_size) for RNN.
        input_size = input_shape[-1] if len(input_shape) >= 2 else input_shape[0]
        return build_rnn_module(
            input_size=input_size,
            hidden_size=int(params.get("hidden_size", 64)),
            n_layers=int(params.get("n_layers", 1)),
            output_dim=output_dim,
            rnn_type=params.get("rnn_type", "lstm"),
            bidirectional=bool(params.get("bidirectional", False)),
            dropout=float(params.get("dropout", 0.1)),
        )

    def prepare_fold(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        device: Any,
        task: str,
    ) -> Any:
        import torch
        from sklearn.preprocessing import StandardScaler

        # Scale features per timestep (2D reshape → scale → back to 3D).
        n_train, seq_len, n_feat = X_train.shape
        scaler = StandardScaler()
        X_train_flat = scaler.fit_transform(X_train.reshape(-1, n_feat)).reshape(
            n_train, seq_len, n_feat
        )
        n_test = X_test.shape[0]
        X_test_flat = scaler.transform(X_test.reshape(-1, n_feat)).reshape(n_test, seq_len, n_feat)

        X_train_t = torch.FloatTensor(X_train_flat).to(device)
        X_test_t = torch.FloatTensor(X_test_flat).to(device)
        y_train_t = torch.LongTensor(y_train).to(device)
        y_test_t = torch.LongTensor(y_test).to(device)

        return X_train_t, X_test_t, y_train_t, y_test_t


# Registry: model name → backend instance.
DL_BACKENDS: dict[str, DLBackend] = {
    "mlp_regressor": MLPBackend(),
    "mlp_classifier": MLPBackend(),
    "cnn_classifier": CNNBackend(),
    "rnn_classifier": RNNBackend(),
}


def get_backend(model_name: str) -> DLBackend:
    """Return the DL backend for a model name.

    Raises:
        KeyError: If the model name is not a registered DL model.
    """
    if model_name not in DL_BACKENDS:
        raise KeyError(f"No DL backend for '{model_name}'. Known DL models: {sorted(DL_BACKENDS)}")
    return DL_BACKENDS[model_name]


__all__ = ["DLBackend", "MLPBackend", "CNNBackend", "RNNBackend", "DL_BACKENDS", "get_backend"]
