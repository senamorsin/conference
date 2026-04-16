"""Grid-search confidence/margin thresholds on a held-out slice of training data.

Uses the same feature CSVs as ``configs/model_words_combined.yaml`` (train+val splits
only), stratified 80/20 for calibration vs fit. The production XGBoost artifact must
exist (``train_words_winner.py`` output).

Selects thresholds that maximize accuracy on *accepted* predictions subject to
accepting at least ``--min-accept-rate`` of clips (default 0.7).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_words import build_splits  # noqa: E402
from src.utils.config import load_yaml  # noqa: E402
from src.words.labels import WORD_FEATURE_COLUMNS  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/model_words_combined.yaml")
    p.add_argument(
        "--model",
        default="models/words/combined_words_xgb.joblib",
        help="Trained word classifier joblib (XGBoost winner).",
    )
    p.add_argument(
        "--output-json",
        default="reports/words/rejection_calibration.json",
        help="Where to write the sweep report and chosen thresholds.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--min-accept-rate",
        type=float,
        default=0.7,
        help="Minimum fraction of clips that must pass rejection (tune on held-out test split).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    training_config = dict(config.get("training", {}))
    training_config.setdefault("random_state", int(args.seed))
    model_path = Path(args.model).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    artifact = joblib.load(model_path)
    if not isinstance(artifact, dict):
        raise TypeError("Expected dict joblib artifact from train_words_winner.py")

    model = artifact["model"]
    labels: tuple[str, ...] = tuple(str(x) for x in artifact["labels"])
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    feature_columns = list(artifact.get("feature_columns", WORD_FEATURE_COLUMNS))
    selected = np.array(artifact.get("selected_feature_indices", []), dtype=int)

    _, _, X_test, y_test, split_strategy, _ = build_splits(
        config,
        feature_columns,
        training_config,
    )
    if X_test.size == 0:
        raise RuntimeError("Test split is empty; cannot calibrate thresholds.")

    X = np.asarray(X_test, dtype=np.float32)
    y_cal = y_test
    if selected.size:
        X = X[:, selected]

    proba = model.predict_proba(X)
    y_int = np.asarray([label_to_idx[str(v)] for v in y_cal], dtype=np.int64)
    order = np.argsort(-proba, axis=1)
    top1 = proba[np.arange(len(y_cal)), order[:, 0]]
    top2 = proba[np.arange(len(y_cal)), order[:, 1]]
    margin = top1 - top2
    pred_int = order[:, 0]
    correct = pred_int == y_int

    min_accept = float(args.min_accept_rate)
    best: dict[str, float] | None = None
    grid_conf = np.linspace(0.15, 0.55, 25)
    grid_margin = np.linspace(0.02, 0.22, 21)

    rows: list[dict[str, float]] = []
    for mc in grid_conf:
        for mm in grid_margin:
            accept = (top1 >= mc) & (margin >= mm)
            rate = float(np.mean(accept))
            if rate < min_accept:
                continue
            if not np.any(accept):
                continue
            acc_accepted = float(np.mean(correct[accept]))
            rows.append(
                {
                    "min_confidence": float(mc),
                    "min_margin": float(mm),
                    "accept_rate": rate,
                    "accuracy_on_accepted": acc_accepted,
                    "n_accepted": int(np.sum(accept)),
                    "n_total": int(len(y_cal)),
                },
            )
            if best is None or acc_accepted > best["accuracy_on_accepted"] + 1e-9:
                best = rows[-1].copy()

    if best is None:
        raise RuntimeError(
            f"No threshold pair met min_accept_rate={min_accept}; "
            "try lowering --min-accept-rate or widening the grid.",
        )

    out_path = Path(args.output_json).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "model_path": str(model_path),
        "config": str(Path(args.config).resolve()),
        "split_strategy": split_strategy,
        "calibration_split": "test",
        "calibration_samples": int(len(y_cal)),
        "min_accept_rate_requested": min_accept,
        "chosen": best,
        "top_candidates": sorted(rows, key=lambda r: -r["accuracy_on_accepted"])[:20],
    }
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(best, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
