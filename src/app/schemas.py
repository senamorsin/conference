from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    mode: str
    accepted_letters: str
    feature_dim: int
    final_word: str | None = None
    final_word_status: str | None = None
    word_history: list[str] = []


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
    hand_landmarks: list[list[float]] = []


class ResetResponse(BaseModel):
    status: str
    accepted_letters: str
    word_history: list[str] = []


class TTSRequest(BaseModel):
    text: str | None = None
