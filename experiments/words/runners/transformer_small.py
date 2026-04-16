"""Minimal Transformer encoder for T=12 landmark sequences.

Kept tiny because the dataset has only ~250 training samples. Adds a learned
class token and a single projection head. Shares the delta-aware input and
training recipe with the other neural runners.
"""
from __future__ import annotations

from experiments.words.common import ExperimentContext, ExperimentResult, SkipRunner
from experiments.words._torch_common import TorchTrainConfig, train_sequence_model


NAME = "transformer_small"
FAMILY = "neural"

HYPERPARAMETERS = {
    "d_model": 96,
    "num_layers": 2,
    "num_heads": 4,
    "dim_feedforward": 192,
    "dropout": 0.3,
    "epochs": 120,
    "batch_size": 32,
    "learning_rate": 5e-4,
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

    d_model = HYPERPARAMETERS["d_model"]
    num_layers = HYPERPARAMETERS["num_layers"]
    num_heads = HYPERPARAMETERS["num_heads"]
    dim_ff = HYPERPARAMETERS["dim_feedforward"]
    dropout = HYPERPARAMETERS["dropout"]

    class TransformerSmall(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_proj = nn.Linear(input_dim, d_model)
            self.pos_embedding = nn.Parameter(torch.zeros(1, 32, d_model))
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=dim_ff,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.classifier = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Dropout(dropout),
                nn.Linear(d_model, num_classes),
            )
            nn.init.trunc_normal_(self.pos_embedding, std=0.02)
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        def forward(self, inputs):
            batch_size, seq_len, _ = inputs.shape
            x = self.input_proj(inputs)
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
            x = x + self.pos_embedding[:, : seq_len + 1, :]
            x = self.encoder(x)
            return self.classifier(x[:, 0, :])

    return TransformerSmall()


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
        notes="2-layer TransformerEncoder, learned pos + CLS token, delta-aware input.",
    )
