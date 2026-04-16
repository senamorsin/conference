import numpy as np

from src.app.state import AppPipelineState
from src.landmarks.mediapipe_extractor import LandmarkExtractionResult
from src.letters.classifier import LetterPrediction
from src.letters.decoder import LetterDecoder
from src.letters.disambiguation import DisambiguationResult
from src.letters.word_builder import LetterWordBuilder
from src.postprocess.dictionary import DictionaryCorrector


class StubExtractor:
    feature_dim = 80

    def extract_from_image_bytes(self, image_bytes: bytes) -> LandmarkExtractionResult:
        return LandmarkExtractionResult(
            landmarks_detected=True,
            features=np.zeros(self.feature_dim, dtype=np.float32),
            landmarks=[],
            hands_landmarks=[],
            detected_hands=1,
            primary_hand_index=0,
        )

    def close(self) -> None:
        return None


class StubClassifier:
    def predict(self, features: np.ndarray) -> LetterPrediction:
        return LetterPrediction(label="SPACE", confidence=0.95)


class StubDisambiguator:
    def resolve(self, features: np.ndarray, prediction: LetterPrediction) -> DisambiguationResult:
        return DisambiguationResult(prediction=prediction, resolver_name=None)


def test_space_prediction_finalizes_current_word_immediately() -> None:
    state = AppPipelineState(
        mode="letters",
        extractor=StubExtractor(),
        classifier=StubClassifier(),
        disambiguator=StubDisambiguator(),
        decoder=LetterDecoder(stable_steps=2, accept_threshold=0.6, blank_steps_to_reset=2),
        word_builder=LetterWordBuilder(
            corrector=DictionaryCorrector(["HI"]),
            finalize_blank_steps=7,
        ),
    )
    state.decoder._accepted = ["H", "I"]

    result = state.process_image_bytes(b"ignored", filename="frame.jpg")

    assert result["status"] == "word_finalized"
    assert result["final_word"] == "HI"
    assert result["accepted_letters"] == ""
