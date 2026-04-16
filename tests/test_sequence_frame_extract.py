"""Sequence frame extraction: legacy (seek) vs fast (linear decode) agreement on timeline shape."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.landmarks.mediapipe_extractor import LandmarkExtractionResult
from src.words.features import (
    extract_frame_records_from_video,
    extract_frame_records_from_video_fast,
    extract_frame_records_from_video_legacy,
)


class _NoHandExtractor:
    """Avoids MediaPipe; returns no landmarks for every frame."""

    feature_dim = 63

    def extract_from_bgr_frame(self, bgr_frame: np.ndarray) -> LandmarkExtractionResult:
        return LandmarkExtractionResult(
            landmarks_detected=False,
            features=np.zeros(self.feature_dim, dtype=np.float32),
            landmarks=[],
            hands_landmarks=[],
            detected_hands=0,
            primary_hand_index=None,
        )


def _write_test_avi(path: Path, *, fps: float = 10.0, num_frames: int = 50, size: int = 64) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (size, size))
    if not writer.isOpened():
        pytest.skip("OpenCV VideoWriter unavailable for this fourcc on this runner")
    try:
        frame = np.zeros((size, size, 3), dtype=np.uint8)
        for _ in range(num_frames):
            writer.write(frame)
    finally:
        writer.release()


def test_legacy_and_fast_produce_matching_timelines_and_counts(tmp_path: Path) -> None:
    video_path = tmp_path / "test.avi"
    fps = 10.0
    num_frames = 50
    _write_test_avi(video_path, fps=fps, num_frames=num_frames)

    ext = _NoHandExtractor()
    target_fps = 10.0

    legacy_records, legacy_dur = extract_frame_records_from_video_legacy(
        video_path, ext, target_fps=target_fps,
    )
    fast_records, fast_dur = extract_frame_records_from_video_fast(
        video_path, ext, target_fps=target_fps,
    )

    assert abs(legacy_dur - fast_dur) < 0.15
    assert len(legacy_records) == len(fast_records)
    for leg, fst in zip(legacy_records, fast_records, strict=True):
        assert abs(leg.timestamp_s - fst.timestamp_s) < 1e-5


def test_extract_frame_records_dispatch_aliases(tmp_path: Path) -> None:
    video_path = tmp_path / "t2.avi"
    _write_test_avi(video_path, num_frames=30)

    ext = _NoHandExtractor()
    r_legacy, _ = extract_frame_records_from_video(video_path, ext, 10.0, extract_mode="legacy")
    r_seek, _ = extract_frame_records_from_video(video_path, ext, 10.0, extract_mode="seek")
    r_fast, _ = extract_frame_records_from_video(video_path, ext, 10.0, extract_mode="fast")
    r_lin, _ = extract_frame_records_from_video(video_path, ext, 10.0, extract_mode="linear")

    assert len(r_legacy) == len(r_seek) == len(r_fast) == len(r_lin)


def test_unknown_extract_mode_raises(tmp_path: Path) -> None:
    video_path = tmp_path / "t3.avi"
    _write_test_avi(video_path, num_frames=10)
    ext = _NoHandExtractor()
    with pytest.raises(ValueError, match="Unknown extract_mode"):
        extract_frame_records_from_video(video_path, ext, 10.0, extract_mode="other")
