from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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


class MediaPipeHandExtractor:
    def __init__(
        self,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.5,
        model_asset_path: str | Path = DEFAULT_MODEL_PATH,
        delegate: str = "cpu",
    ) -> None:
        self.feature_dim = LANDMARK_FEATURE_DIM + len(ENGINEERED_FEATURE_NAMES)
        model_path = Path(model_asset_path).expanduser().resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Hand Landmarker model asset was not found: {model_path}")

        self.delegate = self._resolve_delegate(delegate)
        self._landmarker = self._create_landmarker(
            model_path=model_path,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            delegate=self.delegate,
        )

    def extract_from_image_bytes(self, image_bytes: bytes) -> LandmarkExtractionResult:
        image_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
        bgr_frame = cv2.imdecode(image_buffer, cv2.IMREAD_COLOR)
        if bgr_frame is None:
            raise ValueError("Unable to decode the uploaded image")

        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._landmarker.detect(mp_image)

        if not result.hand_landmarks:
            return LandmarkExtractionResult(
                landmarks_detected=False,
                features=np.zeros(self.feature_dim, dtype=np.float32),
                landmarks=[],
            )

        first_hand = result.hand_landmarks[0]
        raw = np.array([[lm.x, lm.y, lm.z] for lm in first_hand], dtype=np.float32)
        normalized = normalize_hand_landmarks(raw)
        engineered = compute_engineered_hand_features(normalized)
        features = np.concatenate([normalized.reshape(-1), engineered], dtype=np.float32)
        return LandmarkExtractionResult(
            landmarks_detected=True,
            features=features,
            landmarks=raw.tolist(),
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
                    running_mode=VisionTaskRunningMode.IMAGE,
                    num_hands=max_num_hands,
                    min_hand_detection_confidence=min_detection_confidence,
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
