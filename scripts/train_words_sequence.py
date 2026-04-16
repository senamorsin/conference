from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from scripts.train_words import build_splits
from src.utils.config import load_yaml
from src.words.labels import WORD_FEATURE_COLUMNS
from src.words.sequence_model import (
    WORD_FRAME_FEATURE_DIM,
    WORD_SEQUENCE_LENGTH,
    encode_word_labels,
    export_temporal_word_cnn_onnx,
    ordered_word_labels,
    reshape_flat_word_features,
    build_temporal_word_cnn,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a temporal CNN whole-word prototype with PyTorch.")
    parser.add_argument(
        "--config",
        default="configs/model_words_sequence_combined.yaml",
        help="Path to the YAML config describing data sources and sequence-model outputs.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for inputs, labels in loader:
            logits = model(inputs.to(device))
            predictions.append(torch.argmax(logits, dim=1).cpu().numpy())
            targets.append(labels.cpu().numpy())
    y_pred = np.concatenate(predictions) if predictions else np.empty(0, dtype=np.int64)
    y_true = np.concatenate(targets) if targets else np.empty(0, dtype=np.int64)
    accuracy = float(accuracy_score(y_true, y_pred)) if y_true.size else 0.0
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if y_true.size else 0.0
    return accuracy, f1_macro, y_true, y_pred


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    training_config = config["training"]
    seed = int(training_config.get("random_state", 42))
    set_seed(seed)

    feature_columns = list(WORD_FEATURE_COLUMNS)
    X_train_flat, y_train_raw, X_test_flat, y_test_raw, split_strategy, source_desc = build_splits(
        config,
        feature_columns,
        training_config,
    )

    labels = ordered_word_labels(np.concatenate([y_train_raw, y_test_raw]))
    y_train = encode_word_labels(y_train_raw, labels)
    y_test = encode_word_labels(y_test_raw, labels)
    X_train = reshape_flat_word_features(X_train_flat)
    X_test = reshape_flat_word_features(X_test_flat)

    val_size = float(training_config.get("val_size", 0.15))
    X_train_main, X_val, y_train_main, y_val = train_test_split(
        X_train,
        y_train,
        test_size=val_size,
        random_state=seed,
        stratify=y_train,
    )

    train_dataset = TensorDataset(
        torch.from_numpy(np.array(X_train_main, copy=True)).float(),
        torch.from_numpy(np.array(y_train_main, copy=True)).long(),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(np.array(X_val, copy=True)).float(),
        torch.from_numpy(np.array(y_val, copy=True)).long(),
    )
    test_dataset = TensorDataset(
        torch.from_numpy(np.array(X_test, copy=True)).float(),
        torch.from_numpy(np.array(y_test, copy=True)).long(),
    )

    batch_size = int(training_config.get("batch_size", 32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_kwargs = {
        "input_dim": WORD_FRAME_FEATURE_DIM,
        "hidden_dim": int(training_config.get("hidden_dim", 128)),
        "dropout": float(training_config.get("dropout", 0.2)),
    }
    model = build_temporal_word_cnn(labels, model_kwargs).to(device)

    class_counts = np.bincount(y_train_main, minlength=len(labels))
    class_weights = class_counts.sum() / np.maximum(class_counts, 1)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config.get("learning_rate", 1e-3)),
        weight_decay=float(training_config.get("weight_decay", 1e-4)),
    )

    best_state = None
    best_val_accuracy = -1.0
    best_epoch = 0
    epochs = int(training_config.get("epochs", 30))
    for epoch in range(1, epochs + 1):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs.to(device))
            loss = criterion(logits, targets.to(device))
            loss.backward()
            optimizer.step()

        val_accuracy, _, _, _ = evaluate_model(model, val_loader, device)
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training did not produce any model state.")
    model.load_state_dict(best_state)

    test_accuracy, test_f1_macro, y_true, y_pred = evaluate_model(model, test_loader, device)
    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0,
        labels=list(range(len(labels))),
        target_names=list(labels),
    )
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))

    model_output_path = Path(training_config["model_output_path"]).expanduser().resolve()
    onnx_output_path = Path(training_config["onnx_output_path"]).expanduser().resolve()
    metrics_output_path = Path(training_config["metrics_output_path"]).expanduser().resolve()
    artifact = {
        "state_dict": model.state_dict(),
        "labels": labels,
        "sequence_length": WORD_SEQUENCE_LENGTH,
        "feature_dim": WORD_FRAME_FEATURE_DIM,
        "model_kwargs": model_kwargs,
        "source_features_csv": source_desc,
        "split_strategy": split_strategy,
        "dataset_name": config["dataset"]["name"],
    }
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, model_output_path)

    export_temporal_word_cnn_onnx(model.cpu(), onnx_output_path)
    model.to(device)

    rf_comparison = None
    rf_metrics_path = training_config.get("compare_against_metrics_path")
    if rf_metrics_path:
        comparison_path = Path(rf_metrics_path).expanduser().resolve()
        if comparison_path.exists():
            rf_metrics = json.loads(comparison_path.read_text(encoding="utf-8"))
            rf_accuracy = float(rf_metrics.get("accuracy", 0.0))
            rf_comparison = {
                "metrics_path": str(comparison_path),
                "rf_accuracy": rf_accuracy,
                "accuracy_delta": round(test_accuracy - rf_accuracy, 4),
            }

    metrics = {
        "dataset": config["dataset"]["name"],
        "device": str(device),
        "split_strategy": split_strategy,
        "train_samples": int(len(X_train_main)),
        "val_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
        "best_epoch": best_epoch,
        "best_val_accuracy": round(best_val_accuracy, 4),
        "accuracy": round(test_accuracy, 4),
        "f1_macro": round(test_f1_macro, 4),
        "labels": list(labels),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "model_output_path": str(model_output_path),
        "onnx_output_path": str(onnx_output_path),
        "rf_comparison": rf_comparison,
    }
    metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Device: {device}")
    print(f"Best val accuracy: {best_val_accuracy:.4f} (epoch {best_epoch})")
    print(f"Test accuracy: {test_accuracy:.4f}")
    if rf_comparison is not None:
        print(f"RF baseline accuracy: {rf_comparison['rf_accuracy']:.4f}")
        print(f"Accuracy delta vs RF: {rf_comparison['accuracy_delta']:+.4f}")
    print(f"Model saved to {model_output_path}")
    print(f"ONNX saved to {onnx_output_path}")
    print(f"Metrics saved to {metrics_output_path}")


if __name__ == "__main__":
    main()
