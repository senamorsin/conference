"""Temporal 1D CNN with deltas + augmentation.

Improvements over ``src/words/sequence_model.TemporalWordCNN``:
- Input concatenates per-frame motion deltas (see ``reshape_with_deltas``).
- Training loop uses class-weighted CE, cosine LR schedule, early stopping on
  validation macro-F1, Gaussian feature jitter, and ±1 frame time jitter.
"""
from __future__ import annotations

from experiments.words.common import ExperimentContext, ExperimentResult, SkipRunner
from experiments.words._torch_common import TorchTrainConfig, train_sequence_model


NAME = "temporal_cnn_v2"
FAMILY = "neural"

HYPERPARAMETERS = {
    "hidden_dim": 192,
    "dropout": 0.3,
    "epochs": 120,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "noise_std": 0.01,
    "time_jitter_max": 1,
    "val_size": 0.2,
    "patience": 25,
}


def _build_model(input_dim: int, num_classes: int):
    try:
        from torch import nn
    except ImportError as exc:
        raise SkipRunner(f"torch not available: {exc}") from exc

    hidden = HYPERPARAMETERS["hidden_dim"]
    dropout = HYPERPARAMETERS["dropout"]

    class TemporalCNNV2(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv1d(input_dim, hidden, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
                nn.ReLU(),
            )
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden, num_classes),
            )

        def forward(self, inputs):
            x = inputs.transpose(1, 2)
            x = self.features(x)
            x = self.pool(x).squeeze(-1)
            return self.classifier(x)

    return TemporalCNNV2()


def run(context: ExperimentContext) -> ExperimentResult:
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise SkipRunner(f"torch not installed: {exc}") from exc

    train_config = TorchTrainConfig(
        epochs=HYPERPARAMETERS["epochs"],
        batch_size=HYPERPARAMETERS["batch_size"],
        learning_rate=HYPERPARAMETERS["learning_rate"],
        weight_decay=HYPERPARAMETERS["weight_decay"],
        val_size=HYPERPARAMETERS["val_size"],
        patience=HYPERPARAMETERS["patience"],
        noise_std=HYPERPARAMETERS["noise_std"],
        time_jitter_max=HYPERPARAMETERS["time_jitter_max"],
    )

    return train_sequence_model(
        name=NAME,
        family=FAMILY,
        context=context,
        build_model=_build_model,
        train_config=train_config,
        hyperparameters=HYPERPARAMETERS,
        notes="1D CNN with delta channels, cosine LR, early stop on val macro-F1, noise+time jitter.",
    )
