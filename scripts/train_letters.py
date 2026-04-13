from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.letters.labels import FEATURE_COLUMNS
from src.utils.config import load_yaml
from src.utils.hardware import has_nvidia_gpu


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a baseline ASL letter classifier from landmark features.")
    parser.add_argument(
        "--config",
        default="configs/model_letters.yaml",
        help="Path to the YAML config describing the feature CSV and model output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)

    features_csv = Path(config["features"]["output_csv"]).expanduser().resolve()
    training_config = config["training"]
    model_output_path = Path(training_config["model_output_path"]).expanduser().resolve()
    metrics_output_path = Path(training_config["metrics_output_path"]).expanduser().resolve()

    if not features_csv.exists():
        raise FileNotFoundError(f"Feature CSV was not found: {features_csv}. Run scripts/prepare_asl_alphabet.py first.")

    frame = pd.read_csv(features_csv)
    feature_columns = [column for column in FEATURE_COLUMNS if column in frame.columns]
    if len(feature_columns) != len(FEATURE_COLUMNS):
        raise ValueError(f"Expected {len(FEATURE_COLUMNS)} feature columns, found {len(feature_columns)}")

    label_counts = frame["label"].astype(str).value_counts()
    rare_labels = sorted(label for label, count in label_counts.items() if int(count) < 2)
    if rare_labels:
        frame = frame[~frame["label"].isin(rare_labels)].copy()

    if frame.empty:
        raise RuntimeError("No training rows remain after filtering classes with fewer than 2 samples.")

    X = frame[feature_columns].to_numpy(dtype="float32")
    y = frame["label"].astype(str).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(training_config.get("test_size", 0.2)),
        random_state=int(training_config.get("random_state", 42)),
        stratify=y,
    )

    backend_name, resolved_device, model = build_training_backend(training_config)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    artifact = {
        "model": model,
        "labels": tuple(str(label) for label in model.classes_),
        "feature_columns": feature_columns,
        "feature_dim": len(feature_columns),
        "dataset_name": config["dataset"]["name"],
        "source_features_csv": str(features_csv),
        "backend": backend_name,
        "device": resolved_device,
    }
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_output_path)

    metrics = {
        "accuracy": accuracy,
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "labels": artifact["labels"],
        "filtered_rare_labels": rare_labels,
        "classification_report": report,
        "model_output_path": str(model_output_path),
        "backend": backend_name,
        "device": resolved_device,
    }
    metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Backend: {backend_name} ({resolved_device})")
    if rare_labels:
        print(f"Filtered rare labels: {', '.join(rare_labels)}")
    print(f"Model saved to {model_output_path}")
    print(f"Metrics saved to {metrics_output_path}")


def build_training_backend(training_config: dict[str, Any]) -> tuple[str, str, Any]:
    backend = str(training_config.get("backend", "auto")).lower()
    if backend not in {"auto", "random_forest", "xgboost"}:
        raise ValueError(f"Unsupported training backend: {backend}")

    preferred_device = str(training_config.get("device", "auto")).lower()
    if preferred_device not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unsupported training device: {preferred_device}")

    random_state = int(training_config.get("random_state", 42))

    if backend in {"auto", "xgboost"}:
        xgb_classifier = build_xgboost_classifier(training_config, preferred_device, random_state)
        if xgb_classifier is not None:
            return "xgboost", xgb_classifier["device"], xgb_classifier["model"]
        if backend == "xgboost":
            raise RuntimeError(
                "XGBoost backend was requested, but xgboost is not installed or a compatible device was not available."
            )

    model = RandomForestClassifier(
        n_estimators=int(training_config.get("n_estimators", 300)),
        random_state=random_state,
        class_weight=training_config.get("class_weight", "balanced_subsample"),
        n_jobs=-1,
    )
    return "random_forest", "cpu", model


def build_xgboost_classifier(
    training_config: dict[str, Any],
    preferred_device: str,
    random_state: int,
) -> dict[str, Any] | None:
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return None

    resolved_device = resolve_training_device(preferred_device)
    model = XGBClassifier(
        n_estimators=int(training_config.get("n_estimators", 300)),
        random_state=random_state,
        max_depth=int(training_config.get("max_depth", 10)),
        learning_rate=float(training_config.get("learning_rate", 0.1)),
        subsample=float(training_config.get("subsample", 0.9)),
        colsample_bytree=float(training_config.get("colsample_bytree", 0.9)),
        eval_metric="mlogloss",
        tree_method="hist",
        device=resolved_device,
    )
    return {
        "device": resolved_device,
        "model": model,
    }


def resolve_training_device(preferred_device: str) -> str:
    if preferred_device == "cpu":
        return "cpu"
    if preferred_device == "cuda":
        return "cuda"
    return "cuda" if has_nvidia_gpu() else "cpu"


if __name__ == "__main__":
    main()
