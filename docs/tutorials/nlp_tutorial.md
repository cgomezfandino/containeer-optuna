# NLP tutorial (M8)

Transformer-based text classification with HuggingFace + Optuna fine-tuning.

## Prerequisites

```bash
pip install containeer-optuna[nlp]
```

## Text classification with DistilBERT

```bash
containeer-optuna run config/experiments/agnews_transformer_classification.yaml --n-trials 20
```

The experiment YAML uses `pruner: median` for epoch pruning:

```yaml
model: transformer_classifier
pruner: median
```

## How it works

1. Per trial: Optuna samples `model_name` (DistilBERT vs BERT), `learning_rate`,
   `batch_size`, `epochs`, `weight_decay`, `warmup_ratio`, `max_seq_length`.
2. The tokenizer + model are built once per trial.
3. The full corpus is tokenized once (input_ids + attention_mask).
4. Per fold: manual epoch loop with `model(input_ids, attention_mask)` forward,
   AdamW optimizer, linear warmup scheduler, and `trial.report` / `should_prune`
   for epoch pruning.
5. Scored with the classification metric (accuracy, f1, etc.).

## Choosing between models

Optuna searches across model families:
- `distilbert-base-uncased`: smaller, faster, ~95% of BERT performance.
- `bert-base-uncased`: larger, slower, slightly better on some tasks.

The param-importances plot reveals whether model choice matters more than
learning_rate or batch_size.
