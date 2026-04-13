from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.letters.labels import FEATURE_COLUMNS
from src.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained ASL letter model on an external landmark feature CSV."
    )
    parser.add_argument(
        "--config",
        default="configs/model_letters_cross_dataset.yaml",
        help="Path to the YAML config describing the external dataset features and output metrics path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)

    dataset_config = config["dataset"]
    features_csv = Path(config["features"]["output_csv"]).expanduser().resolve()
    evaluation_config = config["evaluation"]
    model_path = Path(evaluation_config["model_path"]).expanduser().resolve()
    metrics_output_path = Path(evaluation_config["metrics_output_path"]).expanduser().resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact was not found: {model_path}")
    if not features_csv.exists():
        raise FileNotFoundError(
            f"Feature CSV was not found: {features_csv}. Run scripts/prepare_asl_alphabet.py with the external config first."
        )

    artifact = joblib.load(model_path)
    model = artifact["model"]
    model_labels = {str(label) for label in artifact["labels"]}
    expected_feature_columns = artifact.get("feature_columns", list(FEATURE_COLUMNS))

    frame = pd.read_csv(features_csv)
    feature_columns = [column for column in expected_feature_columns if column in frame.columns]
    if len(feature_columns) != len(expected_feature_columns):
        raise ValueError(
            f"Expected {len(expected_feature_columns)} feature columns from the model artifact, "
            f"found {len(feature_columns)} in {features_csv}"
        )

    total_rows = int(len(frame))
    total_label_counts = {
        str(label): int(count)
        for label, count in frame["label"].astype(str).value_counts().sort_index().items()
    }

    supported_frame = frame[frame["label"].astype(str).isin(model_labels)].copy()
    unsupported_rows = frame[~frame["label"].astype(str).isin(model_labels)].copy()
    unsupported_label_counts = {
        str(label): int(count)
        for label, count in unsupported_rows["label"].astype(str).value_counts().sort_index().items()
    }

    if supported_frame.empty:
        raise RuntimeError(
            "No rows in the evaluation CSV match the trained model labels. "
            "Check that the external dataset contains overlapping classes."
        )

    X_eval = supported_frame[feature_columns].to_numpy(dtype="float32")
    y_true = supported_frame["label"].astype(str).to_numpy()
    y_pred = model.predict(X_eval)

    accuracy = float(accuracy_score(y_true, y_pred))
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    metrics = {
        "dataset_name": dataset_config["name"],
        "features_csv": str(features_csv),
        "model_path": str(model_path),
        "model_dataset_name": artifact.get("dataset_name"),
        "accuracy": accuracy,
        "evaluated_samples": int(len(supported_frame)),
        "total_samples_in_csv": total_rows,
        "excluded_unsupported_samples": int(len(unsupported_rows)),
        "supported_labels": sorted(model_labels),
        "all_label_counts": total_label_counts,
        "excluded_unsupported_label_counts": unsupported_label_counts,
        "classification_report": report,
    }
    metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Evaluated samples: {len(supported_frame)} / {total_rows}")
    if unsupported_label_counts:
        excluded = ", ".join(f"{label}={count}" for label, count in unsupported_label_counts.items())
        print(f"Excluded unsupported labels: {excluded}")
    print(f"Metrics saved to {metrics_output_path}")


if __name__ == "__main__":
    main()
