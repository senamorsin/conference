from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

from src.words.labels import WORD_FEATURE_COLUMNS
from src.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the saved whole-word classifier on the held-out test split "
        "(same split logic as scripts/train_words.py)."
    )
    parser.add_argument(
        "--config",
        default="configs/model_words_msasl.yaml",
        help="YAML config with features.output_csv and training.model_output_path.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write metrics JSON (defaults to training.metrics_output_path with _eval suffix).",
    )
    return parser.parse_args()


def load_source_frames(sources: list[dict], feature_columns: list[str]) -> pd.DataFrame:
    """Load and concatenate feature CSVs from a list of source descriptors."""
    frames: list[pd.DataFrame] = []
    for source in sources:
        csv_path = Path(source["features_csv"]).expanduser().resolve()
        if not csv_path.exists():
            raise FileNotFoundError(f"Feature CSV not found: {csv_path}")
        frame = pd.read_csv(csv_path)
        allowed_splits = source.get("use_splits")
        if allowed_splits:
            frame = frame[frame["split"].isin(allowed_splits)].copy()
        frames.append(frame)
    if not frames:
        raise RuntimeError("No source frames loaded")
    return pd.concat(frames, ignore_index=True)


def build_test_split(config: dict, feature_columns: list[str], training_config: dict) -> tuple[pd.DataFrame, str]:
    """Return (test_frame, split_strategy) supporting both single-CSV and combined configs."""
    dataset_config = config["dataset"]

    if "sources" in dataset_config:
        test_sources = dataset_config.get("test_sources", [])
        if test_sources:
            return load_source_frames(test_sources, feature_columns), "combined_split"
        all_frame = load_source_frames(dataset_config["sources"], feature_columns)
        _, test_frame = train_test_split(
            all_frame,
            test_size=float(training_config.get("test_size", 0.2)),
            random_state=int(training_config.get("random_state", 42)),
            stratify=all_frame["label"].astype(str),
        )
        return test_frame, "stratified_split"

    features_csv = Path(config["features"]["output_csv"]).expanduser().resolve()
    if not features_csv.exists():
        raise FileNotFoundError(f"Word feature CSV was not found: {features_csv}")

    frame = pd.read_csv(features_csv)
    train_frame = frame[frame["split"].isin(["train", "val"])].copy()
    test_frame = frame[frame["split"] == "test"].copy()
    split_strategy = "official_split"
    if train_frame.empty:
        train_frame = frame.copy()
    if test_frame.empty:
        split_strategy = "stratified_split"
        _, test_frame = train_test_split(
            frame,
            test_size=float(training_config.get("test_size", 0.2)),
            random_state=int(training_config.get("random_state", 42)),
            stratify=frame["label"].astype(str),
        )
    return test_frame, split_strategy


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)

    training_config = config["training"]
    model_path = Path(training_config["model_output_path"]).expanduser().resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Trained model was not found: {model_path}. Run scripts/train_words.py first.")

    feature_columns = list(WORD_FEATURE_COLUMNS)
    test_frame, split_strategy = build_test_split(config, feature_columns, training_config)
    if test_frame.empty:
        raise RuntimeError("Test split is empty; cannot evaluate.")

    artifact = joblib.load(model_path)
    if not isinstance(artifact, dict):
        raise TypeError("Expected a dict joblib artifact from train_words.py")

    model = artifact["model"]
    labels = tuple(str(label) for label in artifact["labels"])
    expected_columns = list(artifact.get("feature_columns", feature_columns))
    missing = [column for column in expected_columns if column not in test_frame.columns]
    if missing:
        raise ValueError(f"Model feature columns missing from CSV (first few): {missing[:5]}")

    X_test = test_frame[expected_columns].to_numpy(dtype="float32")
    y_true = test_frame["label"].astype(str).to_numpy()

    selected_indices = artifact.get("selected_feature_indices")
    if selected_indices is not None:
        import numpy as np
        X_test = X_test[:, np.array(selected_indices, dtype=int)]

    y_pred = model.predict(X_test)

    accuracy = float(accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0, labels=list(labels)))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0, labels=list(labels)))
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0, labels=list(labels))
    matrix = confusion_matrix(y_true, y_pred, labels=list(labels))

    per_class_support = {
        label: int(matrix[i].sum()) for i, label in enumerate(labels)
    }

    result: dict[str, object] = {
        "dataset": config["dataset"]["name"],
        "model_path": str(model_path),
        "split_strategy": split_strategy,
        "test_samples": int(len(test_frame)),
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "labels": list(labels),
        "per_class_support": per_class_support,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }

    out_path: Path | None
    if args.output_json:
        out_path = Path(args.output_json).expanduser().resolve()
    else:
        base = Path(training_config["metrics_output_path"]).expanduser().resolve()
        out_path = base.with_name(base.stem + "_eval.json")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"Dataset: {config['dataset']['name']}")
    print(f"Split: {split_strategy}  |  test clips: {len(test_frame)}")
    print(f"Accuracy:   {accuracy:.4f}")
    print(f"F1 macro:   {f1_macro:.4f}")
    print(f"F1 weighted:{f1_weighted:.4f}")
    print(f"Metrics JSON: {out_path}")


if __name__ == "__main__":
    main()
