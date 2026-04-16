from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol

import joblib
import numpy as np

from src.words.labels import WORD_FEATURE_COLUMNS, WORD_LABELS


_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
_MODELS_DIR: Final[Path] = _PROJECT_ROOT / "models" / "words"

# Preferred artifact is the leaderboard winner; see ``experiments/words``.
DEFAULT_WORD_MODEL_PATH: Final[Path] = _MODELS_DIR / "combined_words_xgb.joblib"

# Legacy artifacts we still fall back to so the app keeps working with older
# on-disk models if the winner hasn't been trained yet.
_FALLBACK_MODEL_PATHS: Final[tuple[Path, ...]] = (
    _MODELS_DIR / "combined_words_random_forest.joblib",
    _MODELS_DIR / "msasl_words_random_forest.joblib",
)


@dataclass(slots=True)
class WordPrediction:
    label: str
    confidence: float
    top_predictions: list[tuple[str, float]] = ()

    def __post_init__(self) -> None:
        if not self.top_predictions:
            self.top_predictions = [(self.label, self.confidence)]


class WordClassifierProtocol(Protocol):
    def predict(self, features: np.ndarray) -> WordPrediction:
        ...


class DummyWordClassifier:
    def predict(self, features: np.ndarray) -> WordPrediction:
        feature_mean = float(np.mean(np.abs(features)))
        bucket = int(feature_mean * 1000) % len(WORD_LABELS)
        confidence = min(0.55, 0.25 + feature_mean)
        return WordPrediction(label=WORD_LABELS[bucket], confidence=confidence)


class SklearnWordClassifier:
    def __init__(
        self,
        model: Any,
        labels: tuple[str, ...],
        feature_dim: int | None = None,
        selected_feature_indices: list[int] | None = None,
    ) -> None:
        self._model = model
        self._labels = labels
        self._feature_dim = feature_dim
        self._selected_indices = (
            np.array(selected_feature_indices, dtype=int) if selected_feature_indices else None
        )

    def predict(self, features: np.ndarray) -> WordPrediction:
        aligned = self._align_features(features).reshape(1, -1)
        if self._selected_indices is not None:
            aligned = aligned[:, self._selected_indices]
        probabilities = self._model.predict_proba(aligned)[0]
        ranked = sorted(enumerate(probabilities), key=lambda pair: pair[1], reverse=True)
        best_index = ranked[0][0]
        top_predictions = [
            (self._labels[idx], float(prob))
            for idx, prob in ranked[:2]
        ]
        return WordPrediction(
            label=self._labels[best_index],
            confidence=float(probabilities[best_index]),
            top_predictions=top_predictions,
        )

    def _align_features(self, features: np.ndarray) -> np.ndarray:
        if self._feature_dim is None:
            return features
        if features.size < self._feature_dim:
            raise ValueError(f"Received {features.size} word features, expected at least {self._feature_dim}")
        if features.size == self._feature_dim:
            return features
        return features[: self._feature_dim]


def load_word_classifier(model_path: str | Path = DEFAULT_WORD_MODEL_PATH) -> SklearnWordClassifier:
    artifact_path = Path(model_path).expanduser().resolve()
    artifact = joblib.load(artifact_path)

    if isinstance(artifact, dict):
        model = artifact["model"]
        labels = tuple(str(label) for label in artifact["labels"])
        feature_dim = artifact.get("feature_dim")
        selected_indices = artifact.get("selected_feature_indices")
    else:
        model = artifact
        labels = WORD_LABELS
        feature_dim = len(WORD_FEATURE_COLUMNS)
        selected_indices = None

    return SklearnWordClassifier(
        model=model, labels=labels, feature_dim=feature_dim,
        selected_feature_indices=selected_indices,
    )


def _candidate_model_paths(primary: str | Path | None) -> list[Path]:
    """Build an ordered list of model paths to try when constructing the
    default classifier. The ``WORD_MODEL_PATH`` env var, if set, overrides
    everything. Otherwise we try ``primary`` (defaulting to the leaderboard
    winner) and fall back to the legacy RF artifacts.
    """
    env_override = os.getenv("WORD_MODEL_PATH")
    ordered: list[Path] = []
    if env_override:
        ordered.append(Path(env_override).expanduser().resolve())
    if primary is not None:
        ordered.append(Path(primary).expanduser().resolve())
    for fallback in _FALLBACK_MODEL_PATHS:
        ordered.append(fallback)
    # Deduplicate while preserving order.
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in ordered:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


def build_default_word_classifier(model_path: str | Path | None = DEFAULT_WORD_MODEL_PATH) -> WordClassifierProtocol:
    for candidate in _candidate_model_paths(model_path):
        if candidate.exists():
            return load_word_classifier(candidate)
    return DummyWordClassifier()
