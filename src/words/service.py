from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from src.landmarks.mediapipe_extractor import MediaPipeHandExtractor
from src.words.classifier import WordClassifierProtocol, build_default_word_classifier
from src.words.features import (
    FrameRecord,
    build_feature_vector_from_records,
    extract_frame_records_from_video,
    extract_word_features_from_video,
    resample_records_to_sequence,
)
from src.words.labels import WORD_FEATURE_COLUMNS, WORD_SEQUENCE_LENGTH, display_word_label
from src.words.segmenter import (
    SigningSegment,
    detect_signing_segments,
    fallback_whole_clip_segment,
)


CONFIDENCE_THRESHOLDS: dict[str, float] = {
    "high": 0.60,
    "medium": 0.35,
}
# Defaults tuned via ``scripts/calibrate_word_rejection.py`` (held-out test split, 50-class model).
REJECTION_THRESHOLDS: dict[str, float] = {
    "min_confidence": 0.43,
    "min_margin": 0.03,
}

# Default landmark sample rate for sequence analysis. 10 fps gives the segmenter
# enough temporal resolution to catch ~0.3s pauses while keeping MediaPipe calls
# bounded (~ duration * 10). Tuned for the in-browser MediaRecorder clips the
# UI produces, which run ~15s max.
SEQUENCE_SAMPLE_FPS: float = 10.0


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip()


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    return float(str(raw).strip())


def _env_optional_positive_int(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return None
    value = int(str(raw).strip())
    return value if value > 0 else None


def _classify_confidence(confidence: float) -> str:
    if confidence >= CONFIDENCE_THRESHOLDS["high"]:
        return "high"
    if confidence >= CONFIDENCE_THRESHOLDS["medium"]:
        return "medium"
    return "low"


def _top_prediction_margin(top_predictions: list[tuple[str, float]]) -> float:
    if len(top_predictions) < 2:
        return 1.0
    return round(max(0.0, float(top_predictions[0][1]) - float(top_predictions[1][1])), 4)


def _prediction_status(
    prediction: object,
    *,
    min_confidence: float,
    min_margin: float,
) -> tuple[str, bool, str | None, float]:
    top_predictions = getattr(prediction, "top_predictions", [])
    confidence = float(getattr(prediction, "confidence", 0.0))
    margin = _top_prediction_margin(top_predictions)
    if confidence < min_confidence:
        return "rejected_low_confidence", False, "below_confidence_threshold", margin
    if margin < min_margin:
        return "rejected_ambiguous", False, "top_predictions_too_close", margin
    return "word_predicted", True, None, margin


@dataclass(slots=True)
class WordRecognitionService:
    extractor: MediaPipeHandExtractor
    classifier: WordClassifierProtocol
    sequence_length: int = WORD_SEQUENCE_LENGTH
    # Sequence pipeline (multi-word clip). ``legacy`` = seek-per-sample (original).
    sequence_frame_extract: str = "fast"
    sequence_sample_fps: float = SEQUENCE_SAMPLE_FPS
    sequence_max_input_side: int | None = None
    segment_min_pause_s: float = 0.3
    segment_min_segment_s: float = 0.4
    segment_motion_threshold: float = 0.008
    segment_lead_in_s: float = 0.15
    segment_lead_out_s: float = 0.15
    rejection_min_confidence: float = REJECTION_THRESHOLDS["min_confidence"]
    rejection_min_margin: float = REJECTION_THRESHOLDS["min_margin"]

    @property
    def feature_dim(self) -> int:
        return len(WORD_FEATURE_COLUMNS)

    def predict_from_video_bytes(self, video_bytes: bytes, filename: str) -> dict[str, object]:
        suffix = Path(filename).suffix or ".mp4"
        with NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            handle.write(video_bytes)
            temp_path = Path(handle.name)

        try:
            extraction = extract_word_features_from_video(
                video_path=temp_path,
                extractor=self.extractor,
                sequence_length=self.sequence_length,
            )
        finally:
            temp_path.unlink(missing_ok=True)

        if extraction.detected_steps == 0:
            return {
                "status": "no_hand_detected",
                "accepted_prediction": False,
                "predicted_word": None,
                "predicted_word_display": None,
                "confidence": 0.0,
                "confidence_level": "none",
                "rejection_reason": "no_hand_detected",
                "top_prediction_margin": 0.0,
                "top_predictions": [],
                "detected_steps": 0,
                "sampled_steps": extraction.sampled_steps,
                "frame_source": filename,
            }

        prediction = self.classifier.predict(extraction.features)
        confidence_level = _classify_confidence(prediction.confidence)
        status, accepted_prediction, rejection_reason, top_prediction_margin = _prediction_status(
            prediction,
            min_confidence=self.rejection_min_confidence,
            min_margin=self.rejection_min_margin,
        )
        top_preds = [
            {"label": display_word_label(label) or label, "confidence": round(conf, 4)}
            for label, conf in prediction.top_predictions
        ]
        return {
            "status": status,
            "accepted_prediction": accepted_prediction,
            "predicted_word": prediction.label,
            "predicted_word_display": display_word_label(prediction.label),
            "confidence": prediction.confidence,
            "confidence_level": confidence_level,
            "rejection_reason": rejection_reason,
            "top_prediction_margin": round(top_prediction_margin, 4),
            "top_predictions": top_preds,
            "detected_steps": extraction.detected_steps,
            "sampled_steps": extraction.sampled_steps,
            "frame_source": filename,
        }

    def predict_sequence_from_video_bytes(
        self,
        video_bytes: bytes,
        filename: str,
        *,
        sample_fps: float | None = None,
    ) -> dict[str, object]:
        """Classify a multi-word clip by segmenting signing bursts and running
        the single-word classifier on each one. Assumes brief pauses between
        signs; falls back to a single whole-clip prediction when no clear
        segmentation emerges.
        """

        fps = float(sample_fps) if sample_fps is not None else float(self.sequence_sample_fps)
        suffix = Path(filename).suffix or ".mp4"
        with NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            handle.write(video_bytes)
            temp_path = Path(handle.name)

        try:
            records, duration_seconds = extract_frame_records_from_video(
                video_path=temp_path,
                extractor=self.extractor,
                target_fps=fps,
                extract_mode=self.sequence_frame_extract,
                max_input_side=self.sequence_max_input_side,
            )
        finally:
            temp_path.unlink(missing_ok=True)

        if not records:
            return self._empty_sequence_response(
                filename=filename,
                duration_seconds=0.0,
                sample_fps=fps,
            )

        raw_segments = detect_signing_segments(
            records,
            min_pause_s=self.segment_min_pause_s,
            min_segment_s=self.segment_min_segment_s,
            motion_threshold=self.segment_motion_threshold,
            lead_in_s=self.segment_lead_in_s,
            lead_out_s=self.segment_lead_out_s,
        )
        if not raw_segments:
            fallback = fallback_whole_clip_segment(records)
            if fallback is not None:
                raw_segments = [fallback]

        if not raw_segments:
            return self._empty_sequence_response(
                filename=filename,
                duration_seconds=duration_seconds,
                sample_fps=fps,
            )

        segment_payloads: list[dict[str, object]] = []
        accepted_words: list[str] = []
        for segment in raw_segments:
            segment_payload = self._classify_segment(records=records, segment=segment)
            segment_payloads.append(segment_payload)
            if segment_payload["accepted_prediction"] and segment_payload["predicted_word_display"]:
                accepted_words.append(str(segment_payload["predicted_word_display"]))

        transcript = " ".join(accepted_words)
        status = "sequence_predicted" if accepted_words else "sequence_inconclusive"
        return {
            "status": status,
            "transcript": transcript,
            "total_segments": len(segment_payloads),
            "accepted_segments": len(accepted_words),
            "segments": segment_payloads,
            "duration_seconds": round(float(duration_seconds), 3),
            "sample_fps": fps,
            "frame_source": filename,
            "frame_extract": self.sequence_frame_extract,
            "max_input_side": self.sequence_max_input_side,
        }

    def _classify_segment(
        self,
        records: list[FrameRecord],
        segment: SigningSegment,
    ) -> dict[str, object]:
        picks = resample_records_to_sequence(
            records=records,
            start_idx=segment.start_idx,
            end_idx=segment.end_idx,
            sequence_length=self.sequence_length,
        )
        features, detected_steps = build_feature_vector_from_records(picks, sequence_length=self.sequence_length)

        if detected_steps == 0:
            return {
                "status": "no_hand_detected",
                "accepted_prediction": False,
                "predicted_word": None,
                "predicted_word_display": None,
                "confidence": 0.0,
                "confidence_level": "none",
                "rejection_reason": "no_hand_detected",
                "top_prediction_margin": 0.0,
                "top_predictions": [],
                "detected_steps": 0,
                "sampled_steps": self.sequence_length,
                "start_time": round(segment.start_time_s, 3),
                "end_time": round(segment.end_time_s, 3),
            }

        prediction = self.classifier.predict(features)
        confidence_level = _classify_confidence(prediction.confidence)
        status, accepted_prediction, rejection_reason, margin = _prediction_status(
            prediction,
            min_confidence=self.rejection_min_confidence,
            min_margin=self.rejection_min_margin,
        )
        top_preds = [
            {"label": display_word_label(label) or label, "confidence": round(conf, 4)}
            for label, conf in prediction.top_predictions
        ]
        return {
            "status": status,
            "accepted_prediction": accepted_prediction,
            "predicted_word": prediction.label,
            "predicted_word_display": display_word_label(prediction.label),
            "confidence": round(float(prediction.confidence), 4),
            "confidence_level": confidence_level,
            "rejection_reason": rejection_reason,
            "top_prediction_margin": round(margin, 4),
            "top_predictions": top_preds,
            "detected_steps": detected_steps,
            "sampled_steps": self.sequence_length,
            "start_time": round(segment.start_time_s, 3),
            "end_time": round(segment.end_time_s, 3),
        }

    def _empty_sequence_response(
        self,
        filename: str,
        duration_seconds: float,
        *,
        sample_fps: float | None = None,
    ) -> dict[str, object]:
        fps = float(sample_fps) if sample_fps is not None else float(self.sequence_sample_fps)
        return {
            "status": "no_hand_detected",
            "transcript": "",
            "total_segments": 0,
            "accepted_segments": 0,
            "segments": [],
            "duration_seconds": round(float(duration_seconds), 3),
            "sample_fps": fps,
            "frame_source": filename,
            "frame_extract": self.sequence_frame_extract,
            "max_input_side": self.sequence_max_input_side,
        }

    def close(self) -> None:
        self.extractor.close()


def create_default_word_service() -> WordRecognitionService:
    extractor = MediaPipeHandExtractor(
        max_num_hands=2,
        delegate=os.getenv("MP_DELEGATE", "auto"),
    )
    classifier = build_default_word_classifier()
    sequence_extract = _env_str("SEQUENCE_FRAME_EXTRACT", "fast").lower()
    sample_fps = _env_float("SEQUENCE_SAMPLE_FPS", SEQUENCE_SAMPLE_FPS)
    max_side = _env_optional_positive_int("SEQUENCE_MAX_INPUT_SIDE")
    return WordRecognitionService(
        extractor=extractor,
        classifier=classifier,
        sequence_frame_extract=sequence_extract,
        sequence_sample_fps=sample_fps,
        sequence_max_input_side=max_side,
        segment_min_pause_s=_env_float("SEQUENCE_SEGMENT_MIN_PAUSE_S", 0.3),
        segment_min_segment_s=_env_float("SEQUENCE_SEGMENT_MIN_SEGMENT_S", 0.4),
        segment_motion_threshold=_env_float("SEQUENCE_SEGMENT_MOTION_THRESHOLD", 0.008),
        segment_lead_in_s=_env_float("SEQUENCE_SEGMENT_LEAD_IN_S", 0.15),
        segment_lead_out_s=_env_float("SEQUENCE_SEGMENT_LEAD_OUT_S", 0.15),
        rejection_min_confidence=_env_float(
            "WORD_REJECTION_MIN_CONFIDENCE",
            REJECTION_THRESHOLDS["min_confidence"],
        ),
        rejection_min_margin=_env_float(
            "WORD_REJECTION_MIN_MARGIN",
            REJECTION_THRESHOLDS["min_margin"],
        ),
    )
