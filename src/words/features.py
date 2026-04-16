from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from src.landmarks.mediapipe_extractor import MediaPipeHandExtractor
from src.words.labels import (
    WORD_FEATURE_COLUMNS,
    WORD_FRAME_FEATURE_NAMES,
    WORD_MOTION_FEATURE_NAMES,
)


WORD_FRAME_FEATURE_DIM: int = len(WORD_FRAME_FEATURE_NAMES)
_MOTION_DIM: int = len(WORD_MOTION_FEATURE_NAMES)
_BBOX_CENTER_X_OFFSET: int = WORD_MOTION_FEATURE_NAMES.index("bbox_center_x")
_BBOX_CENTER_Y_OFFSET: int = WORD_MOTION_FEATURE_NAMES.index("bbox_center_y")


@dataclass(slots=True)
class FrameRecord:
    """Per-frame landmark summary used by both single-word and sequence paths.

    ``frame_features`` is the concatenation of ``[normalized_landmarks,
    engineered_hand_features, motion_anchor_features]`` in the same order the
    classifier was trained on. Its tail holds the motion anchors so
    ``bbox_center`` can be read without re-running MediaPipe.
    """

    timestamp_s: float
    detected: bool
    frame_features: np.ndarray

    @property
    def bbox_center(self) -> tuple[float, float]:
        motion = self.frame_features[-_MOTION_DIM:]
        return float(motion[_BBOX_CENTER_X_OFFSET]), float(motion[_BBOX_CENTER_Y_OFFSET])


@dataclass(slots=True)
class WordFeatureExtractionResult:
    features: np.ndarray
    detected_steps: int
    sampled_steps: int


def extract_word_features_from_video(
    video_path: str | Path,
    extractor: MediaPipeHandExtractor,
    sequence_length: int,
    start_time: float | None = None,
    end_time: float | None = None,
) -> WordFeatureExtractionResult:
    """Extract the flat word-feature vector for a single isolated clip.

    Samples ``sequence_length`` evenly-spaced timestamps inside
    ``[start_time, end_time]`` (defaults to the whole clip), runs MediaPipe
    once per timestamp, and assembles the standard frame + inter-frame-delta
    feature vector.
    """

    path = Path(video_path).expanduser().resolve()
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video for word feature extraction: {path}")

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        duration_seconds = frame_count / fps if fps > 0 and frame_count > 0 else 0.0

        clip_start = max(0.0, float(start_time or 0.0))
        clip_end = float(end_time) if end_time is not None else duration_seconds
        if duration_seconds > 0.0:
            clip_end = min(clip_end, duration_seconds)
        if clip_end <= clip_start:
            clip_end = clip_start + max(0.5, duration_seconds or 0.5)

        records: list[FrameRecord] = []
        for timestamp in sample_timestamps(clip_start=clip_start, clip_end=clip_end, sequence_length=sequence_length):
            records.append(_extract_frame_record_at(capture, extractor, timestamp))
    finally:
        capture.release()

    if len(records) != sequence_length:
        raise RuntimeError(f"Expected {sequence_length} sampled feature steps, got {len(records)}")

    features, detected_steps = build_feature_vector_from_records(records, sequence_length=sequence_length)
    return WordFeatureExtractionResult(
        features=features,
        detected_steps=detected_steps,
        sampled_steps=sequence_length,
    )


def build_feature_vector_from_records(
    records: list[FrameRecord],
    sequence_length: int,
) -> tuple[np.ndarray, int]:
    """Assemble the flat classifier feature vector from pre-extracted records.

    The records must be ordered by time and ``len(records) == sequence_length``.
    Undetected frames contribute zero-valued feature rows, matching how
    ``extract_word_features_from_video`` has always handled missing frames.
    Returns ``(flat_features, detected_steps)``.
    """

    if len(records) != sequence_length:
        raise RuntimeError(f"Expected {sequence_length} records, got {len(records)}")

    step_features: list[np.ndarray] = []
    detected_steps = 0
    for record in records:
        if record.detected:
            detected_steps += 1
            step_features.append(record.frame_features.astype(np.float32, copy=False))
        else:
            step_features.append(np.zeros(WORD_FRAME_FEATURE_DIM, dtype=np.float32))

    flattened_frames = np.concatenate(step_features, dtype=np.float32)
    deltas = compute_inter_frame_deltas(step_features)
    flattened = np.concatenate([flattened_frames, deltas], dtype=np.float32)
    expected_dim = len(WORD_FEATURE_COLUMNS)
    if flattened.size != expected_dim:
        raise RuntimeError(f"Expected {expected_dim} word features, got {flattened.size}")
    return flattened, detected_steps


def resample_records_to_sequence(
    records: list[FrameRecord],
    start_idx: int,
    end_idx: int,
    sequence_length: int,
) -> list[FrameRecord]:
    """Downsample a slice of records to ``sequence_length`` evenly-spaced picks.

    The returned records carry their original timestamps (not synthetic ones),
    which keeps confidence/latency telemetry interpretable. Used by the
    multi-word sequence pipeline where the classifier still expects a fixed
    number of steps per segment.
    """

    if end_idx < start_idx:
        raise ValueError(f"end_idx ({end_idx}) must be >= start_idx ({start_idx})")
    span = end_idx - start_idx + 1
    if span == sequence_length:
        return records[start_idx : end_idx + 1]

    picks: list[FrameRecord] = []
    for step in range(sequence_length):
        # Half-pixel style sampling so we always land strictly inside the slice.
        fractional_index = (step + 0.5) * span / sequence_length
        absolute_idx = start_idx + min(int(fractional_index), span - 1)
        picks.append(records[absolute_idx])
    return picks


def compute_inter_frame_deltas(step_features: list[np.ndarray]) -> np.ndarray:
    """Compute velocity (first-order delta) of motion-anchor features between consecutive frames."""
    motion_dim = len(WORD_MOTION_FEATURE_NAMES)
    deltas: list[np.ndarray] = []
    for i in range(len(step_features) - 1):
        prev_motion = step_features[i][-motion_dim:]
        curr_motion = step_features[i + 1][-motion_dim:]
        deltas.append(curr_motion - prev_motion)
    return np.concatenate(deltas, dtype=np.float32)


def sample_timestamps(clip_start: float, clip_end: float, sequence_length: int) -> list[float]:
    span = max(clip_end - clip_start, 0.5)
    return [
        clip_start + span * ((index + 0.5) / sequence_length)
        for index in range(sequence_length)
    ]


def compute_motion_anchor_features(raw_landmarks: list[list[float]]) -> np.ndarray:
    points = np.asarray(raw_landmarks, dtype=np.float32)
    if points.shape != (21, 3):
        raise ValueError(f"Expected (21, 3) hand landmarks, got {points.shape}")

    xs = points[:, 0]
    ys = points[:, 1]
    wrist = points[0]
    thumb_tip = points[4]
    index_tip = points[8]
    bbox_center_x = float((xs.min() + xs.max()) * 0.5)
    bbox_center_y = float((ys.min() + ys.max()) * 0.5)
    bbox_width = float(xs.max() - xs.min())
    bbox_height = float(ys.max() - ys.min())

    return np.array(
        [
            wrist[0],
            wrist[1],
            thumb_tip[0],
            thumb_tip[1],
            index_tip[0],
            index_tip[1],
            bbox_center_x,
            bbox_center_y,
            bbox_width,
            bbox_height,
        ],
        dtype=np.float32,
    )


def _maybe_resize_bgr(frame: np.ndarray, max_side: int) -> np.ndarray:
    """Shrink so longest edge is at most ``max_side`` (aspect preserved)."""
    if max_side <= 0:
        return frame
    h, w = frame.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return frame
    scale = max_side / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _frame_record_from_bgr(
    frame: np.ndarray,
    extractor: MediaPipeHandExtractor,
    timestamp_s: float,
    *,
    max_input_side: int | None = None,
) -> FrameRecord:
    if frame is None or frame.size == 0:
        return FrameRecord(
            timestamp_s=timestamp_s,
            detected=False,
            frame_features=np.zeros(WORD_FRAME_FEATURE_DIM, dtype=np.float32),
        )

    bgr = frame
    if max_input_side is not None and max_input_side > 0:
        bgr = _maybe_resize_bgr(bgr, max_input_side)

    extraction = extractor.extract_from_bgr_frame(bgr)
    if not extraction.landmarks_detected:
        return FrameRecord(
            timestamp_s=timestamp_s,
            detected=False,
            frame_features=np.zeros(WORD_FRAME_FEATURE_DIM, dtype=np.float32),
        )

    motion = compute_motion_anchor_features(extraction.landmarks)
    frame_features = np.concatenate([extraction.features, motion], dtype=np.float32)
    return FrameRecord(timestamp_s=timestamp_s, detected=True, frame_features=frame_features)


def _extract_frame_record_at(
    capture: cv2.VideoCapture,
    extractor: MediaPipeHandExtractor,
    timestamp_s: float,
    *,
    max_input_side: int | None = None,
) -> FrameRecord:
    capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_s * 1000.0)
    ok, frame = capture.read()
    if not ok or frame is None:
        return FrameRecord(
            timestamp_s=timestamp_s,
            detected=False,
            frame_features=np.zeros(WORD_FRAME_FEATURE_DIM, dtype=np.float32),
        )

    return _frame_record_from_bgr(frame, extractor, timestamp_s, max_input_side=max_input_side)


def _probe_video_duration(capture: cv2.VideoCapture) -> tuple[float, float]:
    """Return ``(duration_seconds, declared_fps)``."""

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    duration_seconds = frame_count / fps if fps > 0 and frame_count > 0 else 0.0
    if duration_seconds <= 0.0:
        capture.set(cv2.CAP_PROP_POS_AVI_RATIO, 1.0)
        duration_seconds = float(capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
        capture.set(cv2.CAP_PROP_POS_MSEC, 0.0)
    return duration_seconds, fps


def extract_frame_records_from_video_legacy(
    video_path: str | Path,
    extractor: MediaPipeHandExtractor,
    target_fps: float = 10.0,
    *,
    max_input_side: int | None = None,
) -> tuple[list[FrameRecord], float]:
    """Original seek-per-timestamp sampling (reference implementation).

    One ``CAP_PROP_POS_MSEC`` seek per sample. Slower on some containers (e.g.
    browser WebM) but matches historical training-serving behavior when
    ``max_input_side`` is unset.
    """

    path = Path(video_path).expanduser().resolve()
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video for word sequence extraction: {path}")

    try:
        duration_seconds, _fps = _probe_video_duration(capture)
        if duration_seconds <= 0.0:
            return [], 0.0

        step = 1.0 / max(target_fps, 1.0)
        records: list[FrameRecord] = []
        timestamp = 0.0
        while timestamp < duration_seconds:
            records.append(
                _extract_frame_record_at(
                    capture,
                    extractor,
                    timestamp,
                    max_input_side=max_input_side,
                ),
            )
            timestamp += step
        return records, duration_seconds
    finally:
        capture.release()


def _sequence_sample_times(duration_seconds: float, target_fps: float) -> list[float]:
    """Match ``extract_frame_records_from_video_legacy`` sample count and timestamps."""

    step = 1.0 / max(target_fps, 1.0)
    times: list[float] = []
    timestamp = 0.0
    while timestamp < duration_seconds:
        times.append(timestamp)
        timestamp += step
    return times


def extract_frame_records_from_video_fast(
    video_path: str | Path,
    extractor: MediaPipeHandExtractor,
    target_fps: float = 10.0,
    *,
    max_input_side: int | None = None,
) -> tuple[list[FrameRecord], float]:
    """Linear decode with subsampling (usually faster than repeated seeks)."""

    path = Path(video_path).expanduser().resolve()
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video for word sequence extraction: {path}")

    try:
        duration_seconds, declared_fps = _probe_video_duration(capture)
        if duration_seconds <= 0.0:
            return [], 0.0

        native_fps = declared_fps if declared_fps > 1e-6 else 30.0
        sample_times = _sequence_sample_times(duration_seconds, target_fps)
        if not sample_times:
            return [], duration_seconds

        records: list[FrameRecord] = []
        next_idx = 0
        frame_index = 0

        while next_idx < len(sample_times):
            ok, frame = capture.read()
            if not ok or frame is None:
                break

            t_msec = float(capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
            if t_msec > 0.0:
                t = t_msec / 1000.0
            else:
                t = frame_index / native_fps

            while next_idx < len(sample_times) and t + 1e-6 >= sample_times[next_idx]:
                records.append(
                    _frame_record_from_bgr(
                        frame,
                        extractor,
                        sample_times[next_idx],
                        max_input_side=max_input_side,
                    ),
                )
                next_idx += 1

            frame_index += 1

        while next_idx < len(sample_times):
            target_t = sample_times[next_idx]
            records.append(
                FrameRecord(
                    timestamp_s=target_t,
                    detected=False,
                    frame_features=np.zeros(WORD_FRAME_FEATURE_DIM, dtype=np.float32),
                ),
            )
            next_idx += 1

        return records, duration_seconds
    finally:
        capture.release()


def extract_frame_records_from_video(
    video_path: str | Path,
    extractor: MediaPipeHandExtractor,
    target_fps: float = 10.0,
    *,
    extract_mode: Literal["legacy", "fast", "seek", "linear"] | str = "legacy",
    max_input_side: int | None = None,
) -> tuple[list[FrameRecord], float]:
    """Sample a video at ``target_fps`` and return per-frame records + duration.

    Used by the multi-word sequence pipeline so the segmenter and per-segment
    classifier share a single MediaPipe pass.

    ``extract_mode``:
    - ``legacy`` / ``seek`` — seek to each sample time (original behavior).
    - ``fast`` / ``linear`` — decode frames sequentially and subsample.
    """

    mode = (extract_mode or "legacy").strip().lower()
    if mode in ("legacy", "seek"):
        return extract_frame_records_from_video_legacy(
            video_path,
            extractor,
            target_fps,
            max_input_side=max_input_side,
        )
    if mode in ("fast", "linear"):
        return extract_frame_records_from_video_fast(
            video_path,
            extractor,
            target_fps,
            max_input_side=max_input_side,
        )
    raise ValueError(
        f"Unknown extract_mode {extract_mode!r}; expected legacy|seek|fast|linear",
    )
