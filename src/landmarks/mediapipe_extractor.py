from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Final

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarker, HandLandmarkerOptions

from src.landmarks.geometry_features import ENGINEERED_FEATURE_NAMES, compute_engineered_hand_features
from src.landmarks.normalization import normalize_hand_landmarks


HAND_LANDMARK_COUNT: Final[int] = 21
LANDMARK_FEATURE_DIM: Final[int] = HAND_LANDMARK_COUNT * 3
DEFAULT_MODEL_PATH: Final[Path] = Path(__file__).resolve().parents[2] / "models" / "mediapipe" / "hand_landmarker.task"


@dataclass(slots=True)
class LandmarkExtractionResult:
    landmarks_detected: bool
    features: np.ndarray
    landmarks: list[list[float]]
    hands_landmarks: list[list[list[float]]]
    detected_hands: int
    primary_hand_index: int | None


class MediaPipeHandExtractor:
    def __init__(
        self,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.35,
        min_presence_confidence: float = 0.35,
        min_tracking_confidence: float = 0.35,
        model_asset_path: str | Path = DEFAULT_MODEL_PATH,
        delegate: str = "cpu",
    ) -> None:
        self.feature_dim = LANDMARK_FEATURE_DIM + len(ENGINEERED_FEATURE_NAMES)
        self._timestamp_ms = 0
        self._timestamp_lock = threading.Lock()
        model_path = Path(model_asset_path).expanduser().resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Hand Landmarker model asset was not found: {model_path}")

        self.delegate = self._resolve_delegate(delegate)
        self._landmarker = self._create_landmarker(
            model_path=model_path,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            delegate=self.delegate,
        )

    def extract_from_image_bytes(self, image_bytes: bytes) -> LandmarkExtractionResult:
        image_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
        bgr_frame = cv2.imdecode(image_buffer, cv2.IMREAD_COLOR)
        if bgr_frame is None:
            raise ValueError("Unable to decode the uploaded image")

        return self.extract_from_bgr_frame(bgr_frame)

    def extract_from_bgr_frame(self, bgr_frame: np.ndarray) -> LandmarkExtractionResult:
        if bgr_frame is None or bgr_frame.size == 0:
            raise ValueError("Received an empty frame for landmark extraction")

        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._landmarker.detect_for_video(mp_image, self._next_timestamp_ms())

        if not result.hand_landmarks:
            return LandmarkExtractionResult(
                landmarks_detected=False,
                features=np.zeros(self.feature_dim, dtype=np.float32),
                landmarks=[],
                hands_landmarks=[],
                detected_hands=0,
                primary_hand_index=None,
            )

        hands_raw = [
            np.array([[lm.x, lm.y, lm.z] for lm in hand], dtype=np.float32)
            for hand in result.hand_landmarks
        ]
        primary_hand_index = max(range(len(hands_raw)), key=lambda index: self._hand_extent_score(hands_raw[index]))
        raw = hands_raw[primary_hand_index]
        normalized = normalize_hand_landmarks(raw)
        engineered = compute_engineered_hand_features(normalized)
        features = np.concatenate([normalized.reshape(-1), engineered], dtype=np.float32)
        return LandmarkExtractionResult(
            landmarks_detected=True,
            features=features,
            landmarks=raw.tolist(),
            hands_landmarks=[hand.tolist() for hand in hands_raw],
            detected_hands=len(hands_raw),
            primary_hand_index=primary_hand_index,
        )

    def close(self) -> None:
        self._landmarker.close()

    @staticmethod
    def _resolve_delegate(delegate: str) -> str:
        normalized = delegate.strip().lower()
        if normalized not in {"cpu", "gpu", "auto"}:
            raise ValueError(f"Unsupported MediaPipe delegate: {delegate}")
        return normalized

    def _create_landmarker(
        self,
        model_path: Path,
        max_num_hands: int,
        min_detection_confidence: float,
        min_presence_confidence: float,
        min_tracking_confidence: float,
        delegate: str,
    ) -> HandLandmarker:
        delegates_to_try = ["gpu", "cpu"] if delegate == "auto" else [delegate]
        last_error: Exception | None = None

        for delegate_name in delegates_to_try:
            try:
                options = HandLandmarkerOptions(
                    base_options=BaseOptions(
                        model_asset_path=str(model_path),
                        delegate=self._to_mediapipe_delegate(delegate_name),
                    ),
                    running_mode=VisionTaskRunningMode.VIDEO,
                    num_hands=max_num_hands,
                    min_hand_detection_confidence=min_detection_confidence,
                    min_hand_presence_confidence=min_presence_confidence,
                    min_tracking_confidence=min_tracking_confidence,
                )
                self.delegate = delegate_name
                return HandLandmarker.create_from_options(options)
            except Exception as exc:  # pragma: no cover - depends on host GPU stack
                last_error = exc
                if delegate != "auto":
                    raise

        if last_error is not None:
            raise RuntimeError("Failed to initialize MediaPipe HandLandmarker on both GPU and CPU delegates") from last_error
        raise RuntimeError("Failed to initialize MediaPipe HandLandmarker")

    @staticmethod
    def _to_mediapipe_delegate(delegate: str) -> BaseOptions.Delegate:
        if delegate == "gpu":
            return BaseOptions.Delegate.GPU
        return BaseOptions.Delegate.CPU

    @staticmethod
    def _hand_extent_score(landmarks: np.ndarray) -> float:
        xs = landmarks[:, 0]
        ys = landmarks[:, 1]
        return float((xs.max() - xs.min()) * (ys.max() - ys.min()))

    def _next_timestamp_ms(self) -> int:
        with self._timestamp_lock:
            self._timestamp_ms += 33
            return self._timestamp_ms
