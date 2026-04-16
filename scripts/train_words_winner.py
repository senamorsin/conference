"""Produce the production whole-word classifier artifact.

Trains the leaderboard winner (currently ``gbdt_xgb``) using the same data,
split, and feature-selection recipe as the experiments harness, and saves a
joblib dict with the exact schema that ``src.words.classifier.load_word_classifier``
expects. The resulting artifact is a drop-in replacement for the RF artifact
emitted by ``scripts/train_words.py``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import accuracy_score, classification_report, f1_score  # noqa: E402

from scripts.train_words import build_splits  # noqa: E402
from src.utils.config import load_yaml  # noqa: E402
from src.words.labels import WORD_FEATURE_COLUMNS  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs" / "model_words_combined.yaml"
DEFAULT_OUTPUT_PATH = ROOT / "models" / "words" / "combined_words_xgb.joblib"
DEFAULT_METRICS_PATH = ROOT / "reports" / "words" / "combined_words_xgb_metrics.json"


WINNER_HYPERPARAMETERS = {
    "select_top_k_features": 300,
    "scout_n_estimators": 200,
    "n_estimators": 400,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "tree_method": "hist",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--metrics", default=str(DEFAULT_METRICS_PATH))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    import xgboost as xgb

    args = parse_args()
    config = load_yaml(args.config)
    training_config = dict(config.get("training", {}))
    training_config.setdefault("random_state", args.seed)

    feature_columns = list(WORD_FEATURE_COLUMNS)
    X_train, y_train, X_test, y_test, split_strategy, source_desc = build_splits(
        config, feature_columns, training_config,
    )
    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)

    scout = RandomForestClassifier(
        n_estimators=WINNER_HYPERPARAMETERS["scout_n_estimators"],
        random_state=args.seed,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    scout.fit(X_train, y_train)
    k = min(int(WINNER_HYPERPARAMETERS["select_top_k_features"]), X_train.shape[1])
    selected = np.sort(np.argsort(scout.feature_importances_)[-k:]).astype(int)
    X_train_sel = X_train[:, selected]
    X_test_sel = X_test[:, selected]
    selected_names = [feature_columns[i] for i in selected]
    print(f"Feature selection: kept {k}/{len(feature_columns)} features by importance")

    labels = tuple(sorted({str(v) for v in np.concatenate([y_train, y_test])}))
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    y_train_int = np.asarray([label_to_idx[str(v)] for v in y_train], dtype=np.int64)
    y_test_int = np.asarray([label_to_idx[str(v)] for v in y_test], dtype=np.int64)

    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=len(labels),
        n_estimators=WINNER_HYPERPARAMETERS["n_estimators"],
        max_depth=WINNER_HYPERPARAMETERS["max_depth"],
        learning_rate=WINNER_HYPERPARAMETERS["learning_rate"],
        subsample=WINNER_HYPERPARAMETERS["subsample"],
        colsample_bytree=WINNER_HYPERPARAMETERS["colsample_bytree"],
        reg_lambda=WINNER_HYPERPARAMETERS["reg_lambda"],
        tree_method=WINNER_HYPERPARAMETERS["tree_method"],
        random_state=args.seed,
        n_jobs=-1,
        eval_metric="mlogloss",
    )
    model.fit(X_train_sel, y_train_int)

    y_pred_int = model.predict(X_test_sel)
    y_pred = np.asarray([labels[int(i)] for i in y_pred_int])
    y_true = np.asarray([labels[int(i)] for i in y_test_int])
    accuracy = float(accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0))
    f1_weighted = float(
        f1_score(y_true, y_pred, labels=list(labels), average="weighted", zero_division=0)
    )
    report = classification_report(
        y_true,
        y_pred,
        labels=list(labels),
        target_names=list(labels),
        output_dict=True,
        zero_division=0,
    )

    artifact = {
        "model": model,
        "labels": labels,
        "feature_columns": feature_columns,
        "feature_dim": len(feature_columns),
        "selected_feature_names": selected_names,
        "selected_feature_indices": selected.tolist(),
        "dataset_name": config.get("dataset", {}).get("name", "combined"),
        "sequence_length": int(config.get("features", {}).get("sequence_length", 12)),
        "source_features_csv": source_desc,
        "split_strategy": split_strategy,
        "algorithm": "xgboost",
        "algorithm_version": xgb.__version__,
        "hyperparameters": WINNER_HYPERPARAMETERS,
    }
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)

    metrics = {
        "algorithm": "xgboost",
        "algorithm_version": xgb.__version__,
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "train_samples": int(len(X_train_sel)),
        "test_samples": int(len(X_test_sel)),
        "labels": list(labels),
        "classification_report": report,
        "model_output_path": str(output_path),
        "split_strategy": split_strategy,
        "source_description": str(source_desc),
        "hyperparameters": WINNER_HYPERPARAMETERS,
    }
    metrics_path = Path(args.metrics).expanduser().resolve()
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {f1_macro:.4f}")
    print(f"Model saved to {output_path}")
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
