from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import joblib
import numpy as np

from src.letters.classifier import LetterPrediction, SklearnLetterClassifier


SPECIALIZED_GROUPS: Final[dict[str, tuple[str, ...]]] = {
    "mn": ("M", "N"),
    "cdo": ("C", "D", "O"),
}
DEFAULT_DISAMBIGUATOR_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "models" / "letters" / "disambiguators"


@dataclass(slots=True)
class DisambiguationResult:
    prediction: LetterPrediction
    resolver_name: str | None = None


class SpecializedLetterDisambiguator:
    def __init__(self, classifiers: dict[str, SklearnLetterClassifier]) -> None:
        self._classifiers = classifiers

    def resolve(self, features: np.ndarray, base_prediction: LetterPrediction) -> DisambiguationResult:
        group_name = self._group_for_label(base_prediction.label)
        if group_name is None:
            return DisambiguationResult(prediction=base_prediction)

        classifier = self._classifiers.get(group_name)
        if classifier is None:
            return DisambiguationResult(prediction=base_prediction)

        refined = classifier.predict(features)
        return DisambiguationResult(prediction=refined, resolver_name=group_name)

    @staticmethod
    def _group_for_label(label: str) -> str | None:
        for group_name, labels in SPECIALIZED_GROUPS.items():
            if label in labels:
                return group_name
        return None


def load_specialized_disambiguator(model_dir: str | Path = DEFAULT_DISAMBIGUATOR_DIR) -> SpecializedLetterDisambiguator:
    base_dir = Path(model_dir).expanduser().resolve()
    classifiers: dict[str, SklearnLetterClassifier] = {}

    for group_name in SPECIALIZED_GROUPS:
        artifact_path = base_dir / f"{group_name}.joblib"
        if not artifact_path.exists():
            continue

        artifact = joblib.load(artifact_path)
        classifiers[group_name] = SklearnLetterClassifier(
            model=artifact["model"],
            labels=tuple(str(label) for label in artifact["labels"]),
            feature_dim=artifact.get("feature_dim"),
        )

    return SpecializedLetterDisambiguator(classifiers=classifiers)
