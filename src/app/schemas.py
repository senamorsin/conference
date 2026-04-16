from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    mode: str
    accepted_letters: str
    feature_dim: int
    word_feature_dim: int | None = None
    word_vocab_size: int = 0
    word_vocab: list[str] = []
    final_word: str | None = None
    final_word_status: str | None = None
    word_history: list[str] = []
    sequence_frame_extract: str = "fast"
    sequence_sample_fps: float = 10.0
    sequence_max_input_side: int | None = None


class PredictResponse(BaseModel):
    mode: str
    status: str
    current_letter: str | None
    confidence: float
    resolver_name: str | None = None
    accepted_letters: str
    feature_dim: int
    landmarks_detected: bool
    frame_source: str
    final_word_raw: str | None = None
    final_word: str | None = None
    final_word_status: str | None = None
    final_word_score: float = 0.0
    word_history: list[str] = []
    detected_hands: int = 0
    primary_hand_index: int | None = None
    hand_landmarks: list[list[float]] = []
    hands_landmarks: list[list[list[float]]] = []


class ResetResponse(BaseModel):
    status: str
    accepted_letters: str
    word_history: list[str] = []


class WordPredictResponse(BaseModel):
    status: str
    accepted_prediction: bool = False
    predicted_word: str | None = None
    predicted_word_display: str | None = None
    confidence: float
    confidence_level: str = "none"
    rejection_reason: str | None = None
    top_prediction_margin: float = 0.0
    top_predictions: list[dict[str, object]] = []
    detected_steps: int
    sampled_steps: int
    frame_source: str


class WordSequenceSegment(BaseModel):
    status: str
    accepted_prediction: bool = False
    predicted_word: str | None = None
    predicted_word_display: str | None = None
    confidence: float = 0.0
    confidence_level: str = "none"
    rejection_reason: str | None = None
    top_prediction_margin: float = 0.0
    top_predictions: list[dict[str, object]] = []
    detected_steps: int = 0
    sampled_steps: int = 0
    start_time: float = 0.0
    end_time: float = 0.0


class WordSequenceResponse(BaseModel):
    status: str
    transcript: str = ""
    total_segments: int = 0
    accepted_segments: int = 0
    segments: list[WordSequenceSegment] = []
    duration_seconds: float = 0.0
    sample_fps: float = 0.0
    frame_source: str
    frame_extract: str = ""
    max_input_side: int | None = None


class TTSRequest(BaseModel):
    text: str | None = None
