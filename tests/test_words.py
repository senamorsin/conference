from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.words.classifier import (
    DEFAULT_WORD_MODEL_PATH,
    WordPrediction,
    load_word_classifier,
)
from src.words.labels import WORD_FEATURE_COLUMNS, display_word_label, normalize_msasl_word_label
from src.words.service import REJECTION_THRESHOLDS, _prediction_status, _top_prediction_margin


def test_normalize_msasl_word_label_maps_dataset_aliases() -> None:
    assert normalize_msasl_word_label("thanks") == "THANK_YOU"
    assert normalize_msasl_word_label("thank you") == "THANK_YOU"
    assert normalize_msasl_word_label("toilet") == "BATHROOM"
    assert normalize_msasl_word_label("book") == "BOOK"
    assert normalize_msasl_word_label("unknown-label") is None


def test_display_word_label_replaces_underscores() -> None:
    assert display_word_label("THANK_YOU") == "THANK YOU"


def test_top_prediction_margin_uses_top_two_scores() -> None:
    margin = _top_prediction_margin([("HELP", 0.48), ("HELLO", 0.43)])
    assert margin == 0.05


def test_prediction_status_rejects_ambiguous_top_two() -> None:
    prediction = SimpleNamespace(
        confidence=0.50,
        top_predictions=[("HELP", 0.50), ("HELLO", 0.48)],
    )

    status, accepted_prediction, rejection_reason, margin = _prediction_status(
        prediction,
        min_confidence=REJECTION_THRESHOLDS["min_confidence"],
        min_margin=REJECTION_THRESHOLDS["min_margin"],
    )

    assert status == "rejected_ambiguous"
    assert accepted_prediction is False
    assert rejection_reason == "top_predictions_too_close"
    assert margin == 0.02


def test_default_word_model_artifact_predicts_valid_prediction() -> None:
    """Loading the default on-disk artifact must expose ``WordPrediction``
    with non-empty ``top_predictions`` so the service-layer confidence and
    rejection logic always has data to work with.
    """
    artifact_path = Path(DEFAULT_WORD_MODEL_PATH)
    if not artifact_path.exists():
        pytest.skip(f"default word model artifact missing: {artifact_path}")

    classifier = load_word_classifier(artifact_path)
    rng = np.random.default_rng(0)
    features = rng.normal(size=len(WORD_FEATURE_COLUMNS)).astype(np.float32)

    prediction = classifier.predict(features)

    assert isinstance(prediction, WordPrediction)
    assert isinstance(prediction.label, str) and prediction.label
    assert 0.0 <= prediction.confidence <= 1.0
    assert len(prediction.top_predictions) >= 1
    assert all(
        isinstance(label, str) and 0.0 <= float(conf) <= 1.0
        for label, conf in prediction.top_predictions
    )
