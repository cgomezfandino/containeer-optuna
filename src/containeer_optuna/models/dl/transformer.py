"""HuggingFace Transformer model builder for text classification (M8).

Wraps ``AutoModelForSequenceClassification`` for fine-tuning pre-trained
transformers (DistilBERT, BERT, etc.) on text classification tasks. This
module is consumed by :func:`~containeer_optuna.optimization.objectives.make_nlp_objective`
— NOT by the DL backend abstraction (transformers need tokenization +
attention masks, which don't fit the tensor-in/logits-out DL loop).
"""

from __future__ import annotations

from typing import Any


def build_transformer_module(model_name: str, num_labels: int) -> Any:
    """Build a HuggingFace sequence classification model lazily.

    Args:
        model_name: HuggingFace model name (e.g. ``"distilbert-base-uncased"``,
            ``"bert-base-uncased"``).
        num_labels: Number of classes.

    Returns:
        A ``transformers.AutoModelForSequenceClassification`` instance.

    Raises:
        ImportError: If ``transformers`` is not installed.
    """
    try:
        from transformers import AutoModelForSequenceClassification
    except ImportError as e:
        raise ImportError(
            "transformers is required for NLP models. "
            "Install it with: pip install containeer-optuna[nlp]"
        ) from e

    return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)


def build_tokenizer(model_name: str, max_seq_length: int = 128) -> Any:
    """Build a HuggingFace tokenizer lazily.

    Args:
        model_name: HuggingFace model name.
        max_seq_length: Maximum sequence length for padding/truncation.

    Returns:
        A ``transformers.AutoTokenizer`` instance.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "transformers is required for NLP models. "
            "Install it with: pip install containeer-optuna[nlp]"
        ) from e

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.model_max_length = max_seq_length
    return tokenizer


__all__ = ["build_transformer_module", "build_tokenizer"]
