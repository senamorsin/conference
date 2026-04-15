from __future__ import annotations

from typing import Final

from src.landmarks.geometry_features import ENGINEERED_FEATURE_NAMES


STATIC_ASL_LETTERS: Final[tuple[str, ...]] = tuple("ABCDEFGHIKLMNOPQRSTUVWXY")
MOTION_ASL_LETTERS: Final[tuple[str, ...]] = ("J", "Z")
CONTROL_LABELS: Final[tuple[str, ...]] = ("NOTHING", "SPACE", "DELETE")
DEFAULT_TRAINING_LABELS: Final[tuple[str, ...]] = STATIC_ASL_LETTERS + ("NOTHING",)
LANDMARK_FEATURE_COLUMNS: Final[tuple[str, ...]] = tuple(f"f{index}" for index in range(63))
FEATURE_COLUMNS: Final[tuple[str, ...]] = LANDMARK_FEATURE_COLUMNS + ENGINEERED_FEATURE_NAMES

_DATASET_LABEL_ALIASES: Final[dict[str, str]] = {
    **{letter.lower(): letter for letter in STATIC_ASL_LETTERS + MOTION_ASL_LETTERS},
    "nothing": "NOTHING",
    "space": "SPACE",
    "delete": "DELETE",
    "del": "DELETE",
}


def normalize_dataset_label(raw_label: str) -> str | None:
    key = raw_label.strip().lower()
    if not key:
        return None
    return _DATASET_LABEL_ALIASES.get(key)
