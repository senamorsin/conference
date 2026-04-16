"""Bidirectional GRU with additive attention pooling.

Small enough for CPU inference on T=12 sequences; uses the same delta-aware
input and training recipe as ``temporal_cnn_v2``.
"""
from __future__ import annotations

from experiments.words.common import ExperimentContext, ExperimentResult, SkipRunner
from experiments.words._torch_common import TorchTrainConfig, train_sequence_model


NAME = "bigru_attn"
FAMILY = "neural"

HYPERPARAMETERS = {
    "hidden_dim": 96,
    "num_layers": 1,
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
        import torch
        from torch import nn
    except ImportError as exc:
        raise SkipRunner(f"torch not available: {exc}") from exc

    hidden = HYPERPARAMETERS["hidden_dim"]
    layers = HYPERPARAMETERS["num_layers"]
    dropout = HYPERPARAMETERS["dropout"]

    class BiGRUWithAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_norm = nn.LayerNorm(input_dim)
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden,
                num_layers=layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if layers > 1 else 0.0,
            )
            self.attn = nn.Sequential(
                nn.Linear(2 * hidden, hidden),
                nn.Tanh(),
                nn.Linear(hidden, 1),
            )
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(2 * hidden, num_classes),
            )

        def forward(self, inputs):
            x = self.input_norm(inputs)
            outputs, _ = self.rnn(x)
            scores = self.attn(outputs)
            weights = torch.softmax(scores, dim=1)
            pooled = (outputs * weights).sum(dim=1)
            return self.classifier(pooled)

    return BiGRUWithAttention()


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
        notes="LayerNorm -> BiGRU -> additive attention pool -> dropout -> Linear, delta-aware input.",
    )
