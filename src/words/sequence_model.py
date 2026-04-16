from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch
from torch import nn

from src.words.labels import WORD_FRAME_COLUMNS, WORD_FRAME_FEATURE_NAMES, WORD_LABELS, WORD_SEQUENCE_LENGTH


WORD_FRAME_FEATURE_DIM = len(WORD_FRAME_FEATURE_NAMES)
WORD_FRAME_VECTOR_DIM = len(WORD_FRAME_COLUMNS)


def reshape_flat_word_features(features: np.ndarray) -> np.ndarray:
    array = np.asarray(features, dtype=np.float32)
    if array.ndim == 1:
        if array.size < WORD_FRAME_VECTOR_DIM:
            raise ValueError(f"Expected at least {WORD_FRAME_VECTOR_DIM} flat frame features, got {array.size}")
        return array[:WORD_FRAME_VECTOR_DIM].reshape(WORD_SEQUENCE_LENGTH, WORD_FRAME_FEATURE_DIM)
    if array.ndim == 2:
        if array.shape[1] < WORD_FRAME_VECTOR_DIM:
            raise ValueError(f"Expected at least {WORD_FRAME_VECTOR_DIM} flat frame features, got {array.shape[1]}")
        return array[:, :WORD_FRAME_VECTOR_DIM].reshape(array.shape[0], WORD_SEQUENCE_LENGTH, WORD_FRAME_FEATURE_DIM)
    raise ValueError(f"Expected 1D or 2D flat features, got array with shape {array.shape}")


def ordered_word_labels(values: np.ndarray | list[str] | tuple[str, ...]) -> tuple[str, ...]:
    seen = {str(value) for value in values}
    ordered = [label for label in WORD_LABELS if label in seen]
    extras = sorted(seen.difference(ordered))
    return tuple(ordered + extras)


def encode_word_labels(values: np.ndarray | list[str] | tuple[str, ...], labels: tuple[str, ...]) -> np.ndarray:
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    return np.asarray([label_to_index[str(value)] for value in values], dtype=np.int64)


class TemporalWordCNN(nn.Module):
    def __init__(
        self,
        input_dim: int = WORD_FRAME_FEATURE_DIM,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        num_classes: int = len(WORD_LABELS),
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Input: [batch, time, features]
        x = inputs.transpose(1, 2)
        x = self.features(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)


def build_temporal_word_cnn(labels: tuple[str, ...], model_kwargs: dict[str, Any] | None = None) -> TemporalWordCNN:
    kwargs = dict(model_kwargs or {})
    kwargs["num_classes"] = len(labels)
    return TemporalWordCNN(**kwargs)


def load_temporal_word_cnn_artifact(
    model_path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[TemporalWordCNN, dict[str, Any]]:
    artifact_path = Path(model_path).expanduser().resolve()
    artifact = torch.load(artifact_path, map_location=device)
    labels = tuple(str(label) for label in artifact["labels"])
    model = build_temporal_word_cnn(labels, artifact.get("model_kwargs"))
    model.load_state_dict(artifact["state_dict"])
    model.to(device)
    model.eval()
    artifact["labels"] = labels
    return model, artifact


def export_temporal_word_cnn_onnx(
    model: TemporalWordCNN,
    export_path: str | Path,
    *,
    sequence_length: int = WORD_SEQUENCE_LENGTH,
    feature_dim: int = WORD_FRAME_FEATURE_DIM,
) -> Path:
    path = Path(export_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, sequence_length, feature_dim, dtype=torch.float32)
    model.eval()
    torch.onnx.export(
        model,
        dummy,
        path,
        input_names=["inputs"],
        output_names=["logits"],
        dynamic_axes={"inputs": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=17,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(path))
    return path
