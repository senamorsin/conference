from pathlib import Path

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier

from src.letters.classifier import LetterPrediction, build_default_letter_classifier
from src.letters.decoder import LetterDecoder
from src.letters.word_builder import LetterWordBuilder
from src.letters.labels import DEFAULT_TRAINING_LABELS, normalize_dataset_label
from src.postprocess.dictionary import DictionaryCorrector


def test_decoder_accepts_stable_letter_once() -> None:
    decoder = LetterDecoder(stable_steps=3, accept_threshold=0.6, blank_steps_to_reset=2)

    for _ in range(4):
        decoder.step(LetterPrediction(label="A", confidence=0.9))

    assert decoder.accepted_letters == "A"


def test_decoder_requires_blank_frames_before_reaccepting_letter() -> None:
    decoder = LetterDecoder(stable_steps=2, accept_threshold=0.6, blank_steps_to_reset=2)

    decoder.step(LetterPrediction(label="B", confidence=0.9))
    decoder.step(LetterPrediction(label="B", confidence=0.9))
    decoder.step(LetterPrediction(label="B", confidence=0.9))
    decoder.step(None)
    decoder.step(None)
    decoder.step(LetterPrediction(label="B", confidence=0.9))
    decoder.step(LetterPrediction(label="B", confidence=0.9))

    assert decoder.accepted_letters == "BB"


def test_normalize_dataset_label_supports_common_aliases() -> None:
    assert normalize_dataset_label("a") == "A"
    assert normalize_dataset_label("del") == "DELETE"
    assert normalize_dataset_label("nothing") == "NOTHING"
    assert normalize_dataset_label(" ") is None


def test_build_default_letter_classifier_loads_joblib_artifact(tmp_path: Path) -> None:
    X = np.zeros((4, 63), dtype=np.float32)
    y = np.array(["A", "A", "NOTHING", "NOTHING"])
    model = DummyClassifier(strategy="prior")
    model.fit(X, y)

    artifact_path = tmp_path / "letters.joblib"
    joblib.dump({"model": model, "labels": tuple(model.classes_)}, artifact_path)

    classifier = build_default_letter_classifier(artifact_path)
    prediction = classifier.predict(np.zeros(63, dtype=np.float32))

    assert prediction.label in {"A", "NOTHING"}
    assert 0.0 <= prediction.confidence <= 1.0


def test_build_default_letter_classifier_falls_back_to_dummy_when_model_missing(tmp_path: Path) -> None:
    classifier = build_default_letter_classifier(tmp_path / "missing.joblib")
    prediction = classifier.predict(np.zeros(63, dtype=np.float32))

    assert prediction.label in DEFAULT_TRAINING_LABELS


def test_word_builder_finalizes_corrected_word_after_pause() -> None:
    builder = LetterWordBuilder(
        corrector=DictionaryCorrector(["HELLO"]),
        finalize_blank_steps=3,
    )

    assert builder.maybe_finalize("HELLO", blank_steps=2) is None

    finalized = builder.maybe_finalize("HELLP", blank_steps=3)

    assert finalized is not None
    assert finalized.raw_word == "HELLP"
    assert finalized.final_word == "HELLO"
    assert finalized.status == "corrected"
    assert builder.history == ("HELLO",)
