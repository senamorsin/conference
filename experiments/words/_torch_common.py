"""Shared PyTorch utilities for sequence-model runners.

All torch runners reshape the flat ``(N, 1430)`` features into ``(N, T=12,
F=120)`` where F = 110 per-frame features + 10 per-frame deltas of the motion
sub-vector (padded with zeros at the first time step). They share a training
loop with validation-based early stopping on macro-F1, a cosine LR schedule,
class-weighted loss, and simple Gaussian-jitter + time-jitter augmentation.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

from experiments.words.common import (
    ExperimentContext,
    ExperimentResult,
    artifact_size_mb,
    compute_metrics,
    measure_inference_latency,
)
from src.words.labels import WORD_DELTA_FEATURE_NAMES, WORD_FRAME_COLUMNS, WORD_MOTION_FEATURE_NAMES
from src.words.sequence_model import (
    WORD_FRAME_FEATURE_DIM,
    WORD_SEQUENCE_LENGTH,
)


MOTION_FEATURE_DIM = len(WORD_MOTION_FEATURE_NAMES)
FRAME_VECTOR_DIM = len(WORD_FRAME_COLUMNS)
DELTA_VECTOR_DIM = len(WORD_DELTA_FEATURE_NAMES)
COMBINED_FEATURE_DIM = WORD_FRAME_FEATURE_DIM + MOTION_FEATURE_DIM  # 110 + 10 = 120


def reshape_with_deltas(flat_features: np.ndarray) -> np.ndarray:
    """Reshape ``(N, 1430)`` into ``(N, T=12, F=120)`` with per-frame deltas.

    The flat vector layout is (from ``src.words.labels``):
        frame features: T=12 blocks of WORD_FRAME_FEATURE_DIM=110 floats
        delta features: T-1=11 blocks of MOTION_FEATURE_DIM=10 floats

    We reshape the frame block into ``(N, 12, 110)`` and attach a zero-padded
    ``(N, 12, 10)`` delta channel whose row ``t`` holds the delta that
    connects frame ``t-1`` to frame ``t`` (row 0 is zeros).
    """
    arr = np.asarray(flat_features, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    frames = arr[:, :FRAME_VECTOR_DIM].reshape(
        arr.shape[0], WORD_SEQUENCE_LENGTH, WORD_FRAME_FEATURE_DIM
    )
    deltas_flat = arr[:, FRAME_VECTOR_DIM:FRAME_VECTOR_DIM + DELTA_VECTOR_DIM]
    deltas = deltas_flat.reshape(arr.shape[0], WORD_SEQUENCE_LENGTH - 1, MOTION_FEATURE_DIM)
    padded = np.zeros(
        (arr.shape[0], WORD_SEQUENCE_LENGTH, MOTION_FEATURE_DIM),
        dtype=np.float32,
    )
    padded[:, 1:, :] = deltas
    return np.concatenate([frames, padded], axis=-1).astype(np.float32)


@dataclass(slots=True)
class TorchTrainConfig:
    epochs: int = 80
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    val_size: float = 0.2
    patience: int = 20
    noise_std: float = 0.01
    time_jitter_max: int = 1
    mixup_alpha: float = 0.0
    device: str | None = None


class _TrainSequenceDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        noise_std: float,
        time_jitter_max: int,
        training: bool,
    ) -> None:
        self._X = np.asarray(X, dtype=np.float32)
        self._y = np.asarray(y, dtype=np.int64)
        self._noise_std = float(noise_std)
        self._time_jitter_max = int(time_jitter_max)
        self._training = bool(training)

    def __len__(self) -> int:
        return self._X.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self._X[idx]
        if self._training:
            if self._noise_std > 0:
                sample = sample + np.random.normal(
                    scale=self._noise_std, size=sample.shape
                ).astype(np.float32)
            if self._time_jitter_max > 0:
                shift = np.random.randint(-self._time_jitter_max, self._time_jitter_max + 1)
                if shift != 0:
                    sample = np.roll(sample, shift, axis=0)
                    # Zero-out the rolled-in region so we don't alias from the
                    # opposite end of the clip.
                    if shift > 0:
                        sample[:shift] = 0.0
                    else:
                        sample[shift:] = 0.0
        return torch.from_numpy(sample.copy()), torch.tensor(int(self._y[idx]))


def _pick_device(cfg: TorchTrainConfig) -> torch.device:
    if cfg.device:
        return torch.device(cfg.device)
    if torch.cuda.is_available():
        try:
            torch.cuda.current_device()
            return torch.device("cuda")
        except Exception:
            return torch.device("cpu")
    return torch.device("cpu")


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for inputs, labels in loader:
            logits = model(inputs.to(device))
            preds.append(torch.argmax(logits, dim=1).cpu().numpy())
            targets.append(labels.cpu().numpy())
    if not preds:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(targets), np.concatenate(preds)


def train_sequence_model(
    *,
    name: str,
    family: str,
    context: ExperimentContext,
    build_model: Callable[[int, int], nn.Module],
    train_config: TorchTrainConfig,
    hyperparameters: dict[str, Any],
    notes: str = "",
) -> ExperimentResult:
    device = _pick_device(train_config)

    labels = list(context.labels)
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    y_train_int = np.array([label_to_idx[str(v)] for v in context.y_train], dtype=np.int64)
    y_test_int = np.array([label_to_idx[str(v)] for v in context.y_test], dtype=np.int64)

    X_train_seq = reshape_with_deltas(context.X_train)
    X_test_seq = reshape_with_deltas(context.X_test)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_seq,
        y_train_int,
        test_size=float(train_config.val_size),
        random_state=int(context.seed),
        stratify=y_train_int,
    )

    train_ds = _TrainSequenceDataset(
        X_tr, y_tr,
        noise_std=train_config.noise_std,
        time_jitter_max=train_config.time_jitter_max,
        training=True,
    )
    val_ds = _TrainSequenceDataset(
        X_val, y_val,
        noise_std=0.0,
        time_jitter_max=0,
        training=False,
    )
    test_ds = _TrainSequenceDataset(
        X_test_seq, y_test_int,
        noise_std=0.0,
        time_jitter_max=0,
        training=False,
    )

    train_loader = DataLoader(train_ds, batch_size=train_config.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=train_config.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=train_config.batch_size, shuffle=False)

    model = build_model(COMBINED_FEATURE_DIM, len(labels)).to(device)

    class_counts = np.bincount(y_tr, minlength=len(labels)).astype(np.float64)
    class_weights = class_counts.sum() / np.maximum(class_counts, 1.0)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config.learning_rate),
        weight_decay=float(train_config.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, int(train_config.epochs))
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_val_f1 = -1.0
    best_epoch = 0
    epochs_since_improve = 0

    for epoch in range(1, int(train_config.epochs) + 1):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs.to(device))
            loss = criterion(logits, targets.to(device))
            loss.backward()
            optimizer.step()
        scheduler.step()

        y_val_true, y_val_pred = _evaluate(model, val_loader, device)
        if y_val_true.size == 0:
            continue
        val_f1 = float(f1_score(y_val_true, y_val_pred, average="macro", zero_division=0))
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= int(train_config.patience):
                break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    y_test_true, y_test_pred_int = _evaluate(model, test_loader, device)
    y_test_pred = np.array([labels[int(i)] for i in y_test_pred_int])
    y_test_true_str = np.array([labels[int(i)] for i in y_test_true])

    accuracy, f1_macro, f1_weighted, per_class_f1, support, cm = compute_metrics(
        y_test_true_str, y_test_pred, labels
    )

    cpu_model = copy.deepcopy(model).to("cpu").eval()

    def _predict_one(sample: np.ndarray) -> Any:
        seq = reshape_with_deltas(np.asarray(sample, dtype=np.float32))
        tensor = torch.from_numpy(np.ascontiguousarray(seq)).float()
        with torch.no_grad():
            return cpu_model(tensor)

    latency = measure_inference_latency(
        _predict_one,
        context.X_test[0],
        repeats=context.latency_repeats,
        warmup=context.latency_warmup,
    )
    size_mb = artifact_size_mb(cpu_model)

    hp = dict(hyperparameters)
    hp.update(
        device=str(device),
        best_epoch=best_epoch,
        best_val_f1_macro=round(best_val_f1, 4),
        epochs_run=epoch,
    )

    return ExperimentResult(
        name=name,
        family=family,
        accuracy=accuracy,
        f1_macro=f1_macro,
        f1_weighted=f1_weighted,
        per_class_f1=per_class_f1,
        support_per_class=support,
        confusion_matrix=cm,
        labels=labels,
        latency=latency,
        artifact_size_mb=size_mb,
        hyperparameters=hp,
        notes=notes,
        extras={"best_epoch": best_epoch, "best_val_f1_macro": best_val_f1},
    )


__all__ = [
    "COMBINED_FEATURE_DIM",
    "MOTION_FEATURE_DIM",
    "TorchTrainConfig",
    "reshape_with_deltas",
    "train_sequence_model",
]
