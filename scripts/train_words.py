from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from src.words.labels import WORD_FEATURE_COLUMNS
from src.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a word classifier from temporal landmark features.")
    parser.add_argument(
        "--config",
        default="configs/model_words_msasl.yaml",
        help="Path to the YAML config describing the feature CSV(s) and model output.",
    )
    return parser.parse_args()


def load_source_frames(
    sources: list[dict],
    feature_columns: list[str],
) -> pd.DataFrame:
    """Load and concatenate feature CSVs from a list of source descriptors."""
    frames: list[pd.DataFrame] = []
    for source in sources:
        csv_path = Path(source["features_csv"]).expanduser().resolve()
        if not csv_path.exists():
            raise FileNotFoundError(f"Feature CSV not found: {csv_path}")
        frame = pd.read_csv(csv_path)
        missing = [col for col in feature_columns if col not in frame.columns]
        if missing:
            raise ValueError(f"{csv_path} is missing feature columns: {missing[:5]}")
        allowed_splits = source.get("use_splits")
        if allowed_splits:
            frame = frame[frame["split"].isin(allowed_splits)].copy()
        frames.append(frame)
    if not frames:
        raise RuntimeError("No source frames loaded")
    return pd.concat(frames, ignore_index=True)


def build_splits(
    config: dict,
    feature_columns: list[str],
    training_config: dict,
) -> tuple:
    """Return (X_train, y_train, X_test, y_test, split_strategy, source_description)."""
    dataset_config = config["dataset"]

    if "sources" in dataset_config:
        train_frame = load_source_frames(dataset_config["sources"], feature_columns)
        test_sources = dataset_config.get("test_sources", [])
        if test_sources:
            test_frame = load_source_frames(test_sources, feature_columns)
        else:
            test_frame = pd.DataFrame()

        if test_frame.empty:
            X = train_frame[feature_columns].to_numpy(dtype="float32")
            y = train_frame["label"].astype(str).to_numpy()
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=float(training_config.get("test_size", 0.2)),
                random_state=int(training_config.get("random_state", 42)),
                stratify=y,
            )
            return X_train, y_train, X_test, y_test, "stratified_split", "combined"
        else:
            return (
                train_frame[feature_columns].to_numpy(dtype="float32"),
                train_frame["label"].astype(str).to_numpy(),
                test_frame[feature_columns].to_numpy(dtype="float32"),
                test_frame["label"].astype(str).to_numpy(),
                "combined_split",
                "combined",
            )

    features_csv = Path(config["features"]["output_csv"]).expanduser().resolve()
    if not features_csv.exists():
        raise FileNotFoundError(f"Word feature CSV was not found: {features_csv}")

    frame = pd.read_csv(features_csv)
    missing = [col for col in feature_columns if col not in frame.columns]
    if missing:
        raise ValueError(f"Expected {len(feature_columns)} word feature columns, found {len(feature_columns) - len(missing)}")

    train_frame = frame[frame["split"].isin(["train", "val"])].copy()
    test_frame = frame[frame["split"] == "test"].copy()
    split_strategy = "official_split"
    if train_frame.empty:
        train_frame = frame.copy()
    if test_frame.empty:
        split_strategy = "stratified_split"
        X = frame[feature_columns].to_numpy(dtype="float32")
        y = frame["label"].astype(str).to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=float(training_config.get("test_size", 0.2)),
            random_state=int(training_config.get("random_state", 42)),
            stratify=y,
        )
        return X_train, y_train, X_test, y_test, split_strategy, str(features_csv)
    return (
        train_frame[feature_columns].to_numpy(dtype="float32"),
        train_frame["label"].astype(str).to_numpy(),
        test_frame[feature_columns].to_numpy(dtype="float32"),
        test_frame["label"].astype(str).to_numpy(),
        split_strategy,
        str(features_csv),
    )


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)

    training_config = config["training"]
    model_output_path = Path(training_config["model_output_path"]).expanduser().resolve()
    metrics_output_path = Path(training_config["metrics_output_path"]).expanduser().resolve()

    feature_columns = list(WORD_FEATURE_COLUMNS)
    X_train, y_train, X_test, y_test, split_strategy, source_desc = build_splits(
        config, feature_columns, training_config,
    )

    top_k = training_config.get("select_top_k_features")
    selected_indices: np.ndarray | None = None
    if top_k is not None and int(top_k) > 0:
        k = min(int(top_k), X_train.shape[1])
        scout = RandomForestClassifier(
            n_estimators=max(100, int(training_config.get("n_estimators", 400)) // 2),
            random_state=int(training_config.get("random_state", 42)),
            class_weight=training_config.get("class_weight", "balanced_subsample"),
            n_jobs=-1,
        )
        scout.fit(X_train, y_train)
        importances = scout.feature_importances_
        selected_indices = np.argsort(importances)[-k:]
        selected_indices.sort()
        X_train = X_train[:, selected_indices]
        X_test = X_test[:, selected_indices]
        selected_names = [feature_columns[i] for i in selected_indices]
        print(f"Feature selection: kept {k}/{len(feature_columns)} features by importance")
    else:
        selected_names = feature_columns

    model = RandomForestClassifier(
        n_estimators=int(training_config.get("n_estimators", 400)),
        random_state=int(training_config.get("random_state", 42)),
        class_weight=training_config.get("class_weight", "balanced_subsample"),
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    artifact = {
        "model": model,
        "labels": tuple(str(label) for label in model.classes_),
        "feature_columns": feature_columns,
        "feature_dim": len(feature_columns),
        "selected_feature_names": selected_names,
        "selected_feature_indices": selected_indices.tolist() if selected_indices is not None else None,
        "dataset_name": config["dataset"]["name"],
        "sequence_length": int(config["features"].get("sequence_length", 12)),
        "source_features_csv": source_desc,
        "split_strategy": split_strategy,
    }
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_output_path)

    metrics = {
        "accuracy": accuracy,
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "labels": artifact["labels"],
        "classification_report": report,
        "model_output_path": str(model_output_path),
        "split_strategy": split_strategy,
    }
    metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Model saved to {model_output_path}")
    print(f"Metrics saved to {metrics_output_path}")


if __name__ == "__main__":
    main()
