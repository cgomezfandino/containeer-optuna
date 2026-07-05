"""Embedding extraction from pre-trained models (M8+ enhancement).

Provides utilities to extract dense vector embeddings from pre-trained
transformer models (DistilBERT, BERT, sentence-transformers) without
fine-tuning. The embeddings can then be used as features for sklearn models
(logistic_regression, random_forest, etc.) — a lightweight alternative to
full fine-tuning.

Requires ``pip install containeer-optuna[nlp]`` (transformers + torch).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def extract_embeddings(
    texts: list[str] | np.ndarray,
    model_name: str = "distilbert-base-uncased",
    batch_size: int = 32,
    max_seq_length: int = 128,
    pooling: str = "cls",
    device: str | None = None,
) -> np.ndarray:
    """Extract dense embeddings from a pre-trained transformer model.

    The model is used in inference mode (no training) — just the encoder
    output. The embeddings can be used as tabular features for sklearn models.

    Args:
        texts: List of raw text strings.
        model_name: HuggingFace model name (e.g. ``"distilbert-base-uncased"``).
        batch_size: Batch size for inference.
        max_seq_length: Maximum sequence length for tokenization.
        pooling: How to pool token embeddings into a single vector:
            ``"cls"`` (use [CLS] token), ``"mean"`` (average all tokens),
            or ``"max"`` (max-pool all tokens).
        device: ``"cuda"`` or ``"cpu"``. Auto-detected if None.

    Returns:
        A ``(n_samples, hidden_dim)`` numpy float32 array of embeddings.

    Raises:
        ImportError: If ``transformers`` or ``torch`` is not installed.
        ValueError: If ``pooling`` is not ``"cls"``, ``"mean"``, or ``"max"``.
    """
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "transformers + torch are required for embedding extraction. "
            "Install with: pip install containeer-optuna[nlp]"
        ) from e

    if pooling not in ("cls", "mean", "max"):
        raise ValueError(f"pooling must be 'cls', 'mean', or 'max'; got '{pooling}'")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    texts_list = list(texts) if isinstance(texts, np.ndarray) else texts
    all_embeddings: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(texts_list), batch_size):
            batch = texts_list[start : start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_seq_length,
                return_tensors="pt",
            ).to(device)

            outputs = model(**encoded)
            hidden = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)

            if pooling == "cls":
                pooled = hidden[:, 0, :]  # [CLS] token
            elif pooling == "mean":
                mask = encoded["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            else:  # max
                pooled = hidden.max(dim=1).values

            all_embeddings.append(pooled.cpu().numpy())

    return np.vstack(all_embeddings)


def embedding_pipeline(
    texts: list[str] | np.ndarray,
    labels: np.ndarray,
    model_name: str = "distilbert-base-uncased",
    classifier: str = "logistic_regression",
    **embed_kwargs: Any,
) -> tuple[np.ndarray, np.ndarray, Any]:
    """Extract embeddings and train a sklearn classifier on them.

    A convenience function that combines :func:`extract_embeddings` with
    ``get_model`` to create a lightweight NLP classifier without fine-tuning.

    Args:
        texts: Raw text strings.
        labels: Integer labels.
        model_name: Transformer model for embedding extraction.
        classifier: sklearn model name from the registry (e.g.
            ``"logistic_regression"``, ``"random_forest"``).
        **embed_kwargs: Passed to :func:`extract_embeddings`.

    Returns:
        A tuple ``(embeddings, labels, fitted_classifier)``.
    """
    from ..registry import get_model

    embeddings = extract_embeddings(texts, model_name=model_name, **embed_kwargs)
    clf = get_model(classifier)
    clf.fit(embeddings, labels)
    return embeddings, labels, clf


__all__ = ["extract_embeddings", "embedding_pipeline"]
