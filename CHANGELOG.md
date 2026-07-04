# Changelog

## [Unreleased] — M7 — DL Advanced (CNN + RNN)

### Added
- **CNN classifier** (`cnn_classifier`): PyTorch ConvNet for image
  classification (MNIST/CIFAR). Configurable conv channels, kernel size, FC
  layers. Per-channel [0,1] normalization.
- **RNN classifier** (`rnn_classifier`): PyTorch LSTM/GRU for sequence
  classification. Configurable hidden_size, n_layers, bidirectional, rnn_type.
- **Backend abstraction** (`models/dl/backends.py`): MLPBackend, CNNBackend,
  RNNBackend — each handles architecture-specific module construction and data
  preparation (tensor shaping + normalization). The shared training/pruning
  loop in make_dl_objective is now backend-agnostic.
- **source: torchvision**: new dataset source for image data (MNIST, CIFAR10,
  FashionMNIST). Returns (X, y) numpy arrays in NCHW format.
- **2 model cards** (CNN_CLASSIFIER_CARD, RNN_CLASSIFIER_CARD, milestone M7).
- **Experiments**: mnist_cnn_classification.yaml, mnist_rnn_classification.yaml.
- **6 new tests** (180 total): CNN/RNN module forward passes, CNN/RNN e2e,
  card assertions.

### Changed
- `make_dl_objective` refactored to accept a pluggable backend (MLP/CNN/RNN)
  instead of hardcoded MLP logic. The training loop (epoch + pruning + scoring)
  is shared; only build_module + prepare_fold differ.
- `_DL_MODELS` in runner.py expanded to include cnn_classifier, rnn_classifier.
- `DatasetConfig.source` widened to accept "torchvision".

## [0.6.0] — M5–M6

Statistics (M5) + Deep Learning MLP (M6). See PRs #6, #7, #8.
