from __future__ import annotations

import numpy as np


def normalize_hand_landmarks(landmarks: np.ndarray) -> np.ndarray:
    if landmarks.shape != (21, 3):
        raise ValueError(f"Expected hand landmarks of shape (21, 3), got {landmarks.shape}")

    wrist = landmarks[0]
    centered = landmarks - wrist
    scale = float(np.max(np.linalg.norm(centered[:, :2], axis=1)))
    if scale <= 1e-6:
        return centered.astype(np.float32)
    normalized = centered / scale
    return normalized.astype(np.float32)
