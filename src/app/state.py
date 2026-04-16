from __future__ import annotations

from dataclasses import dataclass, field
import os
import time

from src.landmarks.mediapipe_extractor import MediaPipeHandExtractor
from src.letters.classifier import LetterClassifierProtocol, build_default_letter_classifier
from src.letters.disambiguation import SpecializedLetterDisambiguator, load_specialized_disambiguator
from src.letters.decoder import LetterDecoder
from src.letters.word_builder import LetterWordBuilder
from src.postprocess.dictionary import DictionaryCorrector


@dataclass(slots=True)
class AppPipelineState:
    mode: str
    extractor: MediaPipeHandExtractor
    classifier: LetterClassifierProtocol
    disambiguator: SpecializedLetterDisambiguator
    decoder: LetterDecoder
    word_builder: LetterWordBuilder
    last_status: str = field(default="idle")
    blank_started_at: float | None = field(default=None)

    def process_image_bytes(self, image_bytes: bytes, filename: str) -> dict[str, object]:
        extraction = self.extractor.extract_from_image_bytes(image_bytes)

        if not extraction.landmarks_detected:
            if self.blank_started_at is None:
                self.blank_started_at = time.monotonic()
            self.decoder.step(None)
            finalized = self._maybe_finalize_word(blank_duration_seconds=time.monotonic() - self.blank_started_at)
            self.last_status = "word_finalized" if finalized is not None else "no_hand_detected"
            return self._build_response(
                filename=filename,
                landmarks_detected=False,
                confidence=0.0,
                detected_hands=0,
                primary_hand_index=None,
                hand_landmarks=[],
                hands_landmarks=[],
                resolver_name=None,
            )

        prediction = self.classifier.predict(extraction.features)
        resolution = self.disambiguator.resolve(extraction.features, prediction)
        prediction = resolution.prediction
        if prediction.label == "NOTHING":
            if self.blank_started_at is None:
                self.blank_started_at = time.monotonic()
            self.decoder.step(None)
            finalized = self._maybe_finalize_word(blank_duration_seconds=time.monotonic() - self.blank_started_at)
            self.last_status = "word_finalized" if finalized is not None else "blank_gesture"
            return self._build_response(
                filename=filename,
                landmarks_detected=True,
                confidence=prediction.confidence,
                detected_hands=extraction.detected_hands,
                primary_hand_index=extraction.primary_hand_index,
                hand_landmarks=extraction.landmarks,
                hands_landmarks=extraction.hands_landmarks,
                resolver_name=resolution.resolver_name,
            )

        if prediction.label == "SPACE":
            finalized = self.word_builder.finalize_now(self.decoder.accepted_letters)
            self.decoder.reset()
            self.blank_started_at = None
            self.last_status = "word_finalized" if finalized is not None else "space_detected"
            return self._build_response(
                filename=filename,
                landmarks_detected=True,
                confidence=prediction.confidence,
                detected_hands=extraction.detected_hands,
                primary_hand_index=extraction.primary_hand_index,
                hand_landmarks=extraction.landmarks,
                hands_landmarks=extraction.hands_landmarks,
                resolver_name=resolution.resolver_name,
            )

        self.blank_started_at = None
        self.decoder.step(prediction)
        self.last_status = "letter_updated"
        return self._build_response(
            filename=filename,
            landmarks_detected=True,
            confidence=self.decoder.current_confidence,
            detected_hands=extraction.detected_hands,
            primary_hand_index=extraction.primary_hand_index,
            hand_landmarks=extraction.landmarks,
            hands_landmarks=extraction.hands_landmarks,
            resolver_name=resolution.resolver_name,
        )

    def reset(self) -> None:
        self.decoder.reset()
        self.word_builder.reset()
        self.blank_started_at = None
        self.last_status = "idle"

    def delete_last_letter(self) -> None:
        self.decoder.delete_last()
        self.blank_started_at = None
        self.last_status = "letter_deleted"

    def close(self) -> None:
        self.extractor.close()

    def _maybe_finalize_word(self, blank_duration_seconds: float | None = None):
        finalized = self.word_builder.maybe_finalize(
            accepted_letters=self.decoder.accepted_letters,
            blank_steps=self.decoder.blank_steps,
            blank_duration_seconds=blank_duration_seconds,
        )
        if finalized is not None:
            self.decoder.reset()
            self.blank_started_at = None
        return finalized

    def _build_response(
        self,
        filename: str,
        landmarks_detected: bool,
        confidence: float,
        detected_hands: int,
        primary_hand_index: int | None,
        hand_landmarks: list[list[float]],
        hands_landmarks: list[list[list[float]]],
        resolver_name: str | None,
    ) -> dict[str, object]:
        latest = self.word_builder.latest
        return {
            "mode": self.mode,
            "status": self.last_status,
            "current_letter": self.decoder.current_letter,
            "confidence": confidence,
            "resolver_name": resolver_name,
            "accepted_letters": self.decoder.accepted_letters,
            "feature_dim": self.extractor.feature_dim,
            "landmarks_detected": landmarks_detected,
            "frame_source": filename,
            "final_word_raw": latest.raw_word if latest else None,
            "final_word": latest.final_word if latest else None,
            "final_word_status": latest.status if latest else None,
            "final_word_score": latest.score if latest else 0.0,
            "word_history": list(self.word_builder.history),
            "detected_hands": detected_hands,
            "primary_hand_index": primary_hand_index,
            "hand_landmarks": hand_landmarks,
            "hands_landmarks": hands_landmarks,
        }


def create_app_state() -> AppPipelineState:
    extractor = MediaPipeHandExtractor(
        max_num_hands=2,
        delegate=os.getenv("MP_DELEGATE", "auto"),
    )
    classifier = build_default_letter_classifier()
    disambiguator = load_specialized_disambiguator()
    decoder = LetterDecoder(
        stable_steps=3,
        accept_threshold=0.6,
        blank_steps_to_reset=5,
        repeat_hold_steps=4,
    )
    word_builder = LetterWordBuilder(
        corrector=DictionaryCorrector.from_path(),
        finalize_blank_seconds=3.0,
        finalize_blank_steps=24,
    )
    return AppPipelineState(
        mode="letters",
        extractor=extractor,
        classifier=classifier,
        disambiguator=disambiguator,
        decoder=decoder,
        word_builder=word_builder,
    )
