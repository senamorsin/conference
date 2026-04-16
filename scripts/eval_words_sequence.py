from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset

from scripts.train_words import build_splits
from src.utils.config import load_yaml
from src.words.labels import WORD_FEATURE_COLUMNS
from src.words.sequence_model import load_temporal_word_cnn_artifact, reshape_flat_word_features, encode_word_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained temporal CNN whole-word prototype.")
    parser.add_argument(
        "--config",
        default="configs/model_words_sequence_combined.yaml",
        help="Path to the YAML config used for sequence-model training.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write metrics JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    training_config = config["training"]
    feature_columns = list(WORD_FEATURE_COLUMNS)
    _, _, X_test_flat, y_test_raw, split_strategy, _ = build_splits(config, feature_columns, training_config)

    model_path = Path(training_config["model_output_path"]).expanduser().resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, artifact = load_temporal_word_cnn_artifact(model_path, device=device)
    labels = tuple(str(label) for label in artifact["labels"])

    X_test = reshape_flat_word_features(X_test_flat)
    y_test = encode_word_labels(y_test_raw, labels)
    dataset = TensorDataset(
        torch.from_numpy(np.array(X_test, copy=True)).float(),
        torch.from_numpy(np.array(y_test, copy=True)).long(),
    )
    loader = DataLoader(dataset, batch_size=int(training_config.get("batch_size", 32)), shuffle=False)

    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for inputs, target in loader:
            logits = model(inputs.to(device))
            predictions.append(torch.argmax(logits, dim=1).cpu().numpy())
            targets.append(target.cpu().numpy())

    y_pred = np.concatenate(predictions)
    y_true = np.concatenate(targets)
    accuracy = float(accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0,
        labels=list(range(len(labels))),
        target_names=list(labels),
    )
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))

    if args.output_json:
        out_path = Path(args.output_json).expanduser().resolve()
    else:
        base = Path(training_config["metrics_output_path"]).expanduser().resolve()
        out_path = base.with_name(base.stem + "_eval.json")

    result = {
        "dataset": config["dataset"]["name"],
        "model_path": str(model_path),
        "split_strategy": split_strategy,
        "device": str(device),
        "test_samples": int(len(X_test)),
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "labels": list(labels),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"Dataset: {config['dataset']['name']}")
    print(f"Split: {split_strategy}  |  test clips: {len(X_test)}")
    print(f"Accuracy:   {accuracy:.4f}")
    print(f"F1 macro:   {f1_macro:.4f}")
    print(f"F1 weighted:{f1_weighted:.4f}")
    print(f"Metrics JSON: {out_path}")


if __name__ == "__main__":
    main()
