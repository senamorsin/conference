from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.letters.classifier import LetterPrediction, SklearnLetterClassifier
from src.letters.disambiguation import load_specialized_disambiguator
from src.letters.labels import FEATURE_COLUMNS
from src.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate confusion matrix artifacts for the runtime pipeline: base classifier + specialized disambiguators."
    )
    parser.add_argument("--config", required=True, help="Path to the YAML config describing model, features CSV, and outputs.")
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
    disambiguator_dir = Path(analysis_config["disambiguator_dir"]).expanduser().resolve()
    features_csv = Path(analysis_config["features_csv"]).expanduser().resolve()
    csv_output_path = Path(analysis_config["csv_output_path"]).expanduser().resolve()
    normalized_csv_output_path = Path(analysis_config["normalized_csv_output_path"]).expanduser().resolve()
    top_confusions_output_path = Path(analysis_config["top_confusions_output_path"]).expanduser().resolve()
    plot_output_path = Path(analysis_config["plot_output_path"]).expanduser().resolve()
    summary_output_path = Path(analysis_config["summary_output_path"]).expanduser().resolve()
    top_k = int(analysis_config.get("top_k", 12))

    artifact = joblib.load(model_path)
    model = SklearnLetterClassifier(
        model=artifact["model"],
        labels=tuple(str(label) for label in artifact["labels"]),
        feature_dim=artifact.get("feature_dim"),
    )
    disambiguator = load_specialized_disambiguator(disambiguator_dir)
    labels = [str(label) for label in artifact["labels"]]
    expected_feature_columns = artifact.get("feature_columns", list(FEATURE_COLUMNS))

    frame = pd.read_csv(features_csv)
    frame = frame[frame["label"].astype(str).isin(labels)].copy()
    feature_columns = [column for column in expected_feature_columns if column in frame.columns]
    if len(feature_columns) != len(expected_feature_columns):
        raise ValueError(
            f"Expected {len(expected_feature_columns)} feature columns from the model artifact, "
            f"found {len(feature_columns)} in {features_csv}"
        )

    X_eval = frame[feature_columns].to_numpy(dtype="float32")
    y_true = frame["label"].astype(str).to_numpy()

    y_pred: list[str] = []
    resolver_counts: dict[str, int] = {}
    changes = 0
    for features in X_eval:
        base_prediction = model.predict(features)
        resolved = disambiguator.resolve(features, base_prediction)
        y_pred.append(resolved.prediction.label)
        if resolved.resolver_name:
            resolver_counts[resolved.resolver_name] = resolver_counts.get(resolved.resolver_name, 0) + 1
        if resolved.prediction.label != base_prediction.label:
            changes += 1

    y_pred_arr = np.array(y_pred, dtype=str)
    accuracy = float(accuracy_score(y_true, y_pred_arr))
    matrix = confusion_matrix(y_true, y_pred_arr, labels=labels)
    normalized = confusion_matrix(y_true, y_pred_arr, labels=labels, normalize="true")

    csv_output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(csv_output_path)
    pd.DataFrame(normalized, index=labels, columns=labels).to_csv(normalized_csv_output_path)

    top_confusions = build_top_confusions(matrix, labels, top_k=top_k)
    top_confusions_output_path.write_text(json.dumps(top_confusions, indent=2), encoding="utf-8")
    summary_output_path.write_text(
        json.dumps(
            {
                "accuracy": accuracy,
                "evaluated_samples": int(len(y_true)),
                "resolver_counts": resolver_counts,
                "resolved_prediction_changes": changes,
                "model_path": str(model_path),
                "disambiguator_dir": str(disambiguator_dir),
                "features_csv": str(features_csv),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; saved CSV and JSON artifacts only.")
    else:
        figure, axis = plt.subplots(figsize=(12, 10))
        display = ConfusionMatrixDisplay(confusion_matrix=normalized, display_labels=labels)
        display.plot(ax=axis, cmap="Blues", colorbar=False, values_format=".2f", xticks_rotation=45)
        axis.set_title(str(analysis_config.get("title", "Runtime Confusion Matrix")))
        figure.tight_layout()
        plot_output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(plot_output_path, dpi=200)
        plt.close(figure)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Resolver counts: {resolver_counts}")
    print(f"Resolved prediction changes: {changes}")
    print(f"Top {len(top_confusions)} confusions:")
    for row in top_confusions:
        print(
            f"  {row['true_label']} -> {row['pred_label']}: "
            f"{row['count']} ({row['share_of_true_label']:.2%})"
        )


if __name__ == "__main__":
    main()
