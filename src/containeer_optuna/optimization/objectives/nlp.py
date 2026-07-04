"""NLP objective: transformer fine-tuning with epoch pruning (M8).

A dedicated objective for text classification with HuggingFace transformers
(DistilBERT, BERT, etc.). Separate from ``make_dl_objective`` because
transformers need tokenization + attention masks + AdamW + warmup scheduler,
which don't fit the tensor-in/logits-out DL loop.

Per trial:
1. Sample hyperparameters (model_name, learning_rate, batch_size, epochs,
   weight_decay, warmup_ratio, max_seq_length).
2. Build the tokenizer + model once per trial.
3. Tokenize the full corpus once per trial.
4. Per fold: manual epoch loop with ``model(input_ids, attention_mask)`` forward,
   AdamW optimizer, linear warmup scheduler, ``trial.report`` + ``should_prune``.
5. Score with the classification metric, return mean CV metric.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import optuna

from ...config import ExperimentConfig
from ..objectives.factories import make_cv_splitter


def make_nlp_objective(
    config: ExperimentConfig,
    texts: Any,
    y: Any,
) -> Callable[[Any], float]:
    """Build an NLP Optuna objective for transformer fine-tuning with pruning.

    Args:
        config: Experiment config (must reference ``transformer_classifier``).
        texts: Array of raw text strings.
        y: Array of integer labels.

    Returns:
        A function ``objective(trial) -> float`` returning mean CV accuracy.
    """
    try:
        import torch
        from torch.optim import AdamW
    except ImportError as e:
        raise ImportError(
            "PyTorch + transformers are required for NLP experiments. "
            "Install with: pip install containeer-optuna[nlp]"
        ) from e

    from ...config import load_model_config
    from ...evaluation.metrics import get_classification_scorer
    from ...models.dl.transformer import build_tokenizer, build_transformer_module
    from ...models.registry import suggest_params

    splitter = make_cv_splitter(config.cv)
    texts_arr = np.asarray(texts)
    y_arr = np.asarray(y)

    metric_name = config.metric or "accuracy"
    scorer, _ = get_classification_scorer(metric_name)

    nlp_config = load_model_config(config.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_labels = int(len(np.unique(y_arr)))

    def objective(trial: Any) -> float:
        # Sample hyperparameters.
        if nlp_config.param_space:
            params = suggest_params(trial, nlp_config.param_space, namespace=config.model)
        else:
            params = dict(nlp_config.default_params)

        model_name = params.get("model_name", "distilbert-base-uncased")
        lr = float(params.get("learning_rate", 2e-5))
        batch_size = int(params.get("batch_size", 16))
        epochs = int(params.get("epochs", 3))
        weight_decay = float(params.get("weight_decay", 0.01))
        warmup_ratio = float(params.get("warmup_ratio", 0.1))
        max_seq_length = int(params.get("max_seq_length", 128))

        # Build tokenizer + tokenize all texts once per trial.
        tokenizer = build_tokenizer(model_name, max_seq_length)
        encoded = tokenizer(
            list(texts_arr),
            padding="max_length",
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt",
        )
        all_input_ids = encoded["input_ids"]
        all_attention_mask = encoded["attention_mask"]

        fold_scores: list[float] = []

        for train_idx, test_idx in splitter.split(texts_arr, y_arr):
            # Build a fresh model per fold.
            model = build_transformer_module(model_name, num_labels).to(device)

            train_ids = all_input_ids[train_idx].to(device)
            train_mask = all_attention_mask[train_idx].to(device)
            train_labels = torch.LongTensor(y_arr[train_idx]).to(device)
            test_ids = all_input_ids[test_idx].to(device)
            test_mask = all_attention_mask[test_idx].to(device)
            test_labels_np = y_arr[test_idx]

            optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

            # Linear warmup scheduler.
            total_steps = (len(train_idx) // batch_size + 1) * epochs
            warmup_steps = int(total_steps * warmup_ratio)
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, total_iters=max(warmup_steps, 1)
            )

            n = len(train_idx)

            for epoch in range(epochs):
                model.train()
                perm = torch.randperm(n)
                for start in range(0, n, batch_size):
                    idx = perm[start : start + batch_size]
                    optimizer.zero_grad()
                    out = model(
                        input_ids=train_ids[idx],
                        attention_mask=train_mask[idx],
                    )
                    loss = (
                        out.loss
                        if getattr(out, "loss", None) is not None
                        else torch.nn.functional.cross_entropy(out.logits, train_labels[idx])
                    )
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                # Validation loss for pruning.
                model.eval()
                with torch.no_grad():
                    val_out = model(input_ids=test_ids, attention_mask=test_mask)
                    val_logits = val_out.logits
                    val_loss = torch.nn.functional.cross_entropy(
                        val_logits, torch.LongTensor(test_labels_np).to(device)
                    )

                trial.report(float(val_loss.item()), step=epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            # Compute fold metric.
            model.eval()
            with torch.no_grad():
                preds = model(input_ids=test_ids, attention_mask=test_mask).logits
                preds = preds.argmax(dim=1).cpu().numpy()

            fold_score = (
                scorer._score_func(test_labels_np, preds)
                if hasattr(scorer, "_score_func")
                else float(np.mean(preds == test_labels_np))
            )
            fold_scores.append(float(fold_score))

        if not fold_scores:
            return -1.0
        mean_score = float(np.mean(fold_scores))
        trial.set_user_attr(f"mean_{metric_name}", mean_score)
        trial.set_user_attr("model_name", model_name)
        return mean_score

    return objective


__all__ = ["make_nlp_objective"]
