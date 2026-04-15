from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.letters.disambiguation import SPECIALIZED_GROUPS
from src.letters.labels import FEATURE_COLUMNS
from src.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train specialized letter disambiguators for confused ASL groups.")
    parser.add_argument(
        "--config",
        default="configs/disambiguators.yaml",
        help="Path to YAML config describing the feature CSV and output paths.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)

    features_csv = Path(config["features"]["input_csv"]).expanduser().resolve()
    output_dir = Path(config["training"]["output_dir"]).expanduser().resolve()
    metrics_dir = Path(config["training"]["metrics_dir"]).expanduser().resolve()
    random_state = int(config["training"].get("random_state", 42))
    n_estimators = int(config["training"].get("n_estimators", 300))
    test_size = float(config["training"].get("test_size", 0.2))

    if not features_csv.exists():
        raise FileNotFoundError(f"Feature CSV was not found: {features_csv}")

    frame = pd.read_csv(features_csv)
    feature_columns = [column for column in FEATURE_COLUMNS if column in frame.columns]
    if len(feature_columns) != len(FEATURE_COLUMNS):
        raise ValueError(f"Expected {len(FEATURE_COLUMNS)} feature columns, found {len(feature_columns)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    for group_name, labels in SPECIALIZED_GROUPS.items():
        group_frame = frame[frame["label"].astype(str).isin(labels)].copy()
        if group_frame.empty:
            raise RuntimeError(f"No rows found for disambiguator group {group_name}: {labels}")

        X = group_frame[feature_columns].to_numpy(dtype="float32")
        y = group_frame["label"].astype(str).to_numpy()

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight="balanced_subsample",
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
            "group_name": group_name,
            "source_features_csv": str(features_csv),
        }
        model_output_path = output_dir / f"{group_name}.joblib"
        metrics_output_path = metrics_dir / f"{group_name}.json"
        joblib.dump(artifact, model_output_path)
        metrics_output_path.write_text(
            json.dumps(
                {
                    "group_name": group_name,
                    "labels": list(labels),
                    "accuracy": accuracy,
                    "train_samples": int(len(X_train)),
                    "test_samples": int(len(X_test)),
                    "classification_report": report,
                    "model_output_path": str(model_output_path),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        print(f"{group_name}: accuracy={accuracy:.4f}")
        print(f"Saved model to {model_output_path}")
        print(f"Saved metrics to {metrics_output_path}")


if __name__ == "__main__":
    main()
