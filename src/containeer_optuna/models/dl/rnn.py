"""PyTorch RNN / GRU / LSTM for sequence classification.

Builds a recurrent network that processes a sequence of feature vectors and
classifies the entire sequence. The last hidden state is fed to a linear
classifier. Supports LSTM and GRU, configurable depth, hidden size, and
bidirectional mode.
"""

from __future__ import annotations

from typing import Any


def build_rnn_module(
    input_size: int,
    hidden_size: int,
    n_layers: int,
    output_dim: int,
    rnn_type: str = "lstm",
    bidirectional: bool = False,
    dropout: float = 0.1,
) -> Any:
    """Build a PyTorch RNN/GRU/LSTM ``nn.Module`` lazily.

    Args:
        input_size: Number of features per timestep.
        hidden_size: Hidden state dimension.
        n_layers: Number of stacked recurrent layers.
        output_dim: Number of classes.
        rnn_type: ``"lstm"`` or ``"gru"``.
        bidirectional: Use a bidirectional recurrent layer.
        dropout: Dropout between recurrent layers (only if ``n_layers > 1``).

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

    rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU

    class _RNNClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.rnn = rnn_cls(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=n_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if n_layers > 1 else 0.0,
            )
            factor = 2 if bidirectional else 1
            self.fc = nn.Linear(hidden_size * factor, output_dim)

        def forward(self, x: Any) -> Any:
            # x: (batch, seq_len, input_size)
            out, _ = self.rnn(x)
            # Take the last timestep's output.
            last = out[:, -1, :]
            return self.fc(last)

    return _RNNClassifier()


__all__ = ["build_rnn_module"]
