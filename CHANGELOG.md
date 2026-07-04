# Changelog

## [Unreleased] — M8 — NLP/AI (Transformers)

### Added
- **Transformer text classification** (`transformer_classifier`): DistilBERT/BERT
  fine-tuning for text classification. Per-trial: tokenization, AdamW optimizer,
  linear warmup scheduler, epoch pruning (trial.report + should_prune).
- **`make_nlp_objective`**: dedicated objective for transformer fine-tuning,
  separate from the DL objective (transformers need tokenization +
  attention_mask which don't fit the tensor-in/logits-out DL loop).
- **`models/dl/transformer.py`**: build_transformer_module + build_tokenizer
  (lazy import of transformers).
- **`[nlp]` optional extra**: transformers, datasets, tokenizers, accelerate.
- **`source: huggingface`**: new dataset source for text data (AG_NEWS, IMDB).
  Returns (texts, labels) numpy arrays.
- **Model card**: TRANSFORMER_CLASSIFIER_CARD (milestone M8).
- **Experiment**: agnews_transformer_classification.yaml.
- **5 new tests** (185 total): transformer module build, tokenizer, NLP e2e,
  card assertions. All guarded by pytest.importorskip("transformers").

### Design
- NLP uses a separate objective (make_nlp_objective), NOT the DL backend
  abstraction. Transformers need raw text → tokenizer → input_ids +
  attention_mask → model forward, which is structurally different from the
  tensor-in/logits-out DL loop.
- _NLP_MODELS set in runner.py routes to make_nlp_objective before _DL_MODELS.
- DatasetConfig.source widened to accept "huggingface".

## [0.6.0–0.7.0] — M5–M7

Statistics (M5) + DL MLP (M6) + DL CNN/RNN (M7). See PRs #6–#9.
