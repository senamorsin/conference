import numpy as np

from src.landmarks.normalization import normalize_hand_landmarks
from src.postprocess.dictionary import DictionaryCorrector


def test_normalize_hand_landmarks_returns_expected_shape() -> None:
    landmarks = np.zeros((21, 3), dtype=np.float32)
    landmarks[1] = [1.0, 0.0, 0.0]

    normalized = normalize_hand_landmarks(landmarks)

    assert normalized.shape == (21, 3)
    assert normalized.dtype == np.float32


def test_dictionary_corrector_returns_exact_match() -> None:
    corrector = DictionaryCorrector(["HELP", "HELLO"])

    suggestion = corrector.correct("help")

    assert suggestion.normalized_word == "HELP"
    assert suggestion.status == "exact"
    assert suggestion.score == 1.0


def test_dictionary_corrector_returns_unknown_for_far_word() -> None:
    corrector = DictionaryCorrector(["HELLO", "PLEASE"])

    suggestion = corrector.correct("XYZ")

    assert suggestion.normalized_word == "XYZ"
    assert suggestion.status == "unknown"
