from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.letters.labels import FEATURE_COLUMNS
from src.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate confusion matrix artifacts for an ASL letter model on a feature CSV."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config describing the model, feature CSV, and outputs.",
    )
    return parser.parse_args()


def build_top_confusions(matrix: np.ndarray, labels: list[str], top_k: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for true_index, true_label in enumerate(labels):
        support = int(matrix[true_index].sum())
        if support == 0:
            continue

        for pred_index, pred_label in enumerate(labels):
            if true_index == pred_index:
                continue

            count = int(matrix[true_index, pred_index])
            if count == 0:
                continue

            rows.append(
                {
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "count": count,
                    "share_of_true_label": count / support,
                }
            )

    rows.sort(key=lambda row: (row["count"], row["share_of_true_label"]), reverse=True)
    return rows[:top_k]


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)

    analysis_config = config["analysis"]
    model_path = Path(analysis_config["model_path"]).expanduser().resolve()
    features_csv = Path(analysis_config["features_csv"]).expanduser().resolve()
    csv_output_path = Path(analysis_config["csv_output_path"]).expanduser().resolve()
    normalized_csv_output_path = Path(analysis_config["normalized_csv_output_path"]).expanduser().resolve()
    top_confusions_output_path = Path(analysis_config["top_confusions_output_path"]).expanduser().resolve()
    plot_output_path = Path(analysis_config["plot_output_path"]).expanduser().resolve()
    top_k = int(analysis_config.get("top_k", 10))

    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact was not found: {model_path}")
    if not features_csv.exists():
        raise FileNotFoundError(f"Feature CSV was not found: {features_csv}")

    artifact = joblib.load(model_path)
    model = artifact["model"]
    labels = [str(label) for label in artifact["labels"]]
    expected_feature_columns = artifact.get("feature_columns", list(FEATURE_COLUMNS))

    frame = pd.read_csv(features_csv)
    frame = frame[frame["label"].astype(str).isin(labels)].copy()
    if frame.empty:
        raise RuntimeError("No rows in the feature CSV match the model labels.")

    feature_columns = [column for column in expected_feature_columns if column in frame.columns]
    if len(feature_columns) != len(expected_feature_columns):
        raise ValueError(
            f"Expected {len(expected_feature_columns)} feature columns from the model artifact, "
            f"found {len(feature_columns)} in {features_csv}"
        )

    X_eval = frame[feature_columns].to_numpy(dtype="float32")
    y_true = frame["label"].astype(str).to_numpy()
    y_pred = model.predict(X_eval)

    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    normalized = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")

    csv_output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(csv_output_path)
    pd.DataFrame(normalized, index=labels, columns=labels).to_csv(normalized_csv_output_path)

    top_confusions = build_top_confusions(matrix, labels, top_k=top_k)
    top_confusions_output_path.write_text(json.dumps(top_confusions, indent=2), encoding="utf-8")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; saved CSV and JSON artifacts only.")
    else:
        figure, axis = plt.subplots(figsize=(12, 10))
        display = ConfusionMatrixDisplay(confusion_matrix=normalized, display_labels=labels)
        display.plot(
            ax=axis,
            cmap="Blues",
            colorbar=False,
            values_format=".2f",
            xticks_rotation=45,
        )
        axis.set_title(str(analysis_config.get("title", "Confusion Matrix")))
        figure.tight_layout()
        plot_output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(plot_output_path, dpi=200)
        plt.close(figure)

    print(f"Saved confusion matrix CSV to {csv_output_path}")
    print(f"Saved normalized confusion matrix CSV to {normalized_csv_output_path}")
    print(f"Saved top confusions JSON to {top_confusions_output_path}")
    print(f"Top {len(top_confusions)} confusions:")
    for row in top_confusions:
        print(
            f"  {row['true_label']} -> {row['pred_label']}: "
            f"{row['count']} ({row['share_of_true_label']:.2%})"
        )


if __name__ == "__main__":
    main()
