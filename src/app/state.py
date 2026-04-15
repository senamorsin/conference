from __future__ import annotations

from dataclasses import dataclass, field

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

    def process_image_bytes(self, image_bytes: bytes, filename: str) -> dict[str, object]:
        extraction = self.extractor.extract_from_image_bytes(image_bytes)

        if not extraction.landmarks_detected:
            self.decoder.step(None)
            finalized = self._maybe_finalize_word()
            self.last_status = "word_finalized" if finalized is not None else "no_hand_detected"
            return self._build_response(
                filename=filename,
                landmarks_detected=False,
                confidence=0.0,
                hand_landmarks=[],
            )

        prediction = self.classifier.predict(extraction.features)
        resolution = self.disambiguator.resolve(extraction.features, prediction)
        prediction = resolution.prediction
        if prediction.label == "NOTHING":
            self.decoder.step(None)
            finalized = self._maybe_finalize_word()
            self.last_status = "word_finalized" if finalized is not None else "blank_gesture"
            return self._build_response(
                filename=filename,
                landmarks_detected=True,
                confidence=prediction.confidence,
                hand_landmarks=extraction.landmarks,
                resolver_name=resolution.resolver_name,
            )

        self.decoder.step(prediction)
        self.last_status = "letter_updated"
        return self._build_response(
            filename=filename,
            landmarks_detected=True,
            confidence=self.decoder.current_confidence,
            hand_landmarks=extraction.landmarks,
            resolver_name=resolution.resolver_name,
        )

    def reset(self) -> None:
        self.decoder.reset()
        self.word_builder.reset()
        self.last_status = "idle"

    def close(self) -> None:
        self.extractor.close()

    def _maybe_finalize_word(self):
        finalized = self.word_builder.maybe_finalize(
            accepted_letters=self.decoder.accepted_letters,
            blank_steps=self.decoder.blank_steps,
        )
        if finalized is not None:
            self.decoder.reset()
        return finalized

    def _build_response(
        self,
        filename: str,
        landmarks_detected: bool,
        confidence: float,
        hand_landmarks: list[list[float]],
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
            "hand_landmarks": hand_landmarks,
        }


def create_app_state() -> AppPipelineState:
    extractor = MediaPipeHandExtractor(max_num_hands=1)
    classifier = build_default_letter_classifier()
    disambiguator = load_specialized_disambiguator()
    decoder = LetterDecoder(stable_steps=3, accept_threshold=0.6, blank_steps_to_reset=5)
    word_builder = LetterWordBuilder(
        corrector=DictionaryCorrector.from_path(),
        finalize_blank_steps=7,
    )
    return AppPipelineState(
        mode="letters",
        extractor=extractor,
        classifier=classifier,
        disambiguator=disambiguator,
        decoder=decoder,
        word_builder=word_builder,
    )
