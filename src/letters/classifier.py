from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol

import joblib
import numpy as np

from src.letters.labels import DEFAULT_TRAINING_LABELS, STATIC_ASL_LETTERS


DEFAULT_LETTER_MODEL_PATH: Final[Path] = Path(__file__).resolve().parents[2] / "models" / "letters" / "asl_letters_random_forest.joblib"


@dataclass(slots=True)
class LetterPrediction:
    label: str
    confidence: float


class LetterClassifierProtocol(Protocol):
    def predict(self, features: np.ndarray) -> LetterPrediction:
        ...


class DummyLetterClassifier:
    """Placeholder classifier used until a real ASL model is trained."""

    def predict(self, features: np.ndarray) -> LetterPrediction:
        feature_mean = float(np.mean(np.abs(features)))
        bucket = int(feature_mean * 1000) % len(STATIC_ASL_LETTERS)
        confidence = min(0.95, 0.35 + feature_mean * 3.0)
        return LetterPrediction(label=STATIC_ASL_LETTERS[bucket], confidence=confidence)


class SklearnLetterClassifier:
    def __init__(self, model: Any, labels: tuple[str, ...]) -> None:
        self._model = model
        self._labels = labels

    def predict(self, features: np.ndarray) -> LetterPrediction:
        probabilities = self._model.predict_proba(features.reshape(1, -1))[0]
        best_index = int(np.argmax(probabilities))
        return LetterPrediction(
            label=self._labels[best_index],
            confidence=float(probabilities[best_index]),
        )


def load_letter_classifier(model_path: str | Path = DEFAULT_LETTER_MODEL_PATH) -> SklearnLetterClassifier:
    artifact_path = Path(model_path).expanduser().resolve()
    artifact = joblib.load(artifact_path)

    if isinstance(artifact, dict):
        model = artifact["model"]
        labels = tuple(str(label) for label in artifact["labels"])
    else:
        model = artifact
        labels = DEFAULT_TRAINING_LABELS

    return SklearnLetterClassifier(model=model, labels=labels)


def build_default_letter_classifier(model_path: str | Path = DEFAULT_LETTER_MODEL_PATH) -> LetterClassifierProtocol:
    artifact_path = Path(model_path).expanduser().resolve()
    if artifact_path.exists():
        return load_letter_classifier(artifact_path)
    return DummyLetterClassifier()
