from __future__ import annotations

import math
from typing import Final

import numpy as np


ENGINEERED_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "palm_width",
    "thumb_tip_to_index_tip",
    "thumb_tip_to_middle_tip",
    "thumb_tip_to_index_mcp",
    "thumb_tip_to_middle_mcp",
    "thumb_tip_to_ring_mcp",
    "thumb_tip_to_pinky_mcp",
    "thumb_tip_to_palm_center",
    "thumb_tip_to_ring_tip",
    "thumb_tip_to_pinky_tip",
    "index_middle_tip_gap",
    "middle_ring_tip_gap",
    "index_middle_mcp_gap",
    "middle_ring_mcp_gap",
    "index_tip_to_palm_center",
    "index_tip_to_wrist",
    "middle_tip_to_wrist",
    "thumb_tip_to_wrist",
    "ring_tip_to_wrist",
    "pinky_tip_to_wrist",
    "index_cross_middle",
    "thumb_to_middle_ring_line",
    "thumb_to_index_middle_line",
    "index_pip_angle",
    "thumb_mcp_angle",
    "thumb_ip_angle",
    "ring_dip_angle",
    "pinky_dip_angle",
    "middle_dip_angle",
    "ring_pip_angle",
    "thumb_tip_x",
    "thumb_tip_y",
    "index_tip_x",
    "middle_tip_x",
    "ring_tip_x",
    "pinky_tip_x",
    "ring_tip_y",
)


def compute_engineered_hand_features(landmarks: np.ndarray) -> np.ndarray:
    if landmarks.shape != (21, 3):
        raise ValueError(f"Expected hand landmarks of shape (21, 3), got {landmarks.shape}")

    wrist = landmarks[0]
    thumb_cmc, thumb_mcp, thumb_ip, thumb_tip = landmarks[1], landmarks[2], landmarks[3], landmarks[4]
    index_mcp, index_pip, index_dip, index_tip = landmarks[5], landmarks[6], landmarks[7], landmarks[8]
    middle_mcp, middle_pip, middle_dip, middle_tip = landmarks[9], landmarks[10], landmarks[11], landmarks[12]
    ring_mcp, ring_pip, ring_dip, ring_tip = landmarks[13], landmarks[14], landmarks[15], landmarks[16]
    pinky_mcp, pinky_pip, pinky_dip, pinky_tip = landmarks[17], landmarks[18], landmarks[19], landmarks[20]
    palm_center = (index_mcp + middle_mcp + ring_mcp + pinky_mcp) / 4.0
    palm_width = _distance(index_mcp, pinky_mcp)

    features = np.array(
        [
            palm_width,
            _distance(thumb_tip, index_tip),
            _distance(thumb_tip, middle_tip),
            _distance(thumb_tip, index_mcp),
            _distance(thumb_tip, middle_mcp),
            _distance(thumb_tip, ring_mcp),
            _distance(thumb_tip, pinky_mcp),
            _distance(thumb_tip, palm_center),
            _distance(thumb_tip, ring_tip),
            _distance(thumb_tip, pinky_tip),
            _distance(index_tip, middle_tip),
            _distance(middle_tip, ring_tip),
            _distance(index_mcp, middle_mcp),
            _distance(middle_mcp, ring_mcp),
            _distance(index_tip, palm_center),
            _distance(index_tip, wrist),
            _distance(middle_tip, wrist),
            _distance(thumb_tip, wrist),
            _distance(ring_tip, wrist),
            _distance(pinky_tip, wrist),
            _crossing_score(index_tip, middle_tip, index_pip, middle_pip),
            _distance_to_segment(thumb_tip, middle_mcp, ring_mcp),
            _distance_to_segment(thumb_tip, index_mcp, middle_mcp),
            _angle(index_mcp, index_pip, index_dip),
            _angle(thumb_cmc, thumb_mcp, thumb_ip),
            _angle(thumb_mcp, thumb_ip, thumb_tip),
            _angle(ring_pip, ring_dip, ring_tip),
            _angle(pinky_pip, pinky_dip, pinky_tip),
            _angle(middle_pip, middle_dip, middle_tip),
            _angle(ring_mcp, ring_pip, ring_dip),
            float(thumb_tip[0]),
            float(thumb_tip[1]),
            float(index_tip[0]),
            float(middle_tip[0]),
            float(ring_tip[0]),
            float(pinky_tip[0]),
            float(ring_tip[1]),
        ],
        dtype=np.float32,
    )
    return features


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    denominator = float(np.linalg.norm(ba) * np.linalg.norm(bc))
    if denominator <= 1e-8:
        return 0.0
    cosine = float(np.clip(np.dot(ba, bc) / denominator, -1.0, 1.0))
    return float(math.degrees(math.acos(cosine)))


def _crossing_score(first_tip: np.ndarray, second_tip: np.ndarray, first_mid: np.ndarray, second_mid: np.ndarray) -> float:
    first_delta = float(first_tip[0] - second_tip[0])
    second_delta = float(first_mid[0] - second_mid[0])
    return first_delta * second_delta


def _distance_to_segment(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    segment = end - start
    denominator = float(np.dot(segment, segment))
    if denominator <= 1e-8:
        return _distance(point, start)
    t = float(np.clip(np.dot(point - start, segment) / denominator, 0.0, 1.0))
    projection = start + t * segment
    return _distance(point, projection)
