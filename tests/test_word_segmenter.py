from __future__ import annotations

import numpy as np

from src.words.features import FrameRecord, WORD_FRAME_FEATURE_DIM
from src.words.labels import WORD_MOTION_FEATURE_NAMES
from src.words.segmenter import detect_signing_segments, fallback_whole_clip_segment


_MOTION_DIM = len(WORD_MOTION_FEATURE_NAMES)
_BBOX_CENTER_X_MOTION_INDEX = WORD_MOTION_FEATURE_NAMES.index("bbox_center_x")
_BBOX_CENTER_Y_MOTION_INDEX = WORD_MOTION_FEATURE_NAMES.index("bbox_center_y")


def _make_record(timestamp_s: float, *, detected: bool, bbox_center: tuple[float, float] = (0.5, 0.5)) -> FrameRecord:
    features = np.zeros(WORD_FRAME_FEATURE_DIM, dtype=np.float32)
    if detected:
        features[-_MOTION_DIM + _BBOX_CENTER_X_MOTION_INDEX] = float(bbox_center[0])
        features[-_MOTION_DIM + _BBOX_CENTER_Y_MOTION_INDEX] = float(bbox_center[1])
    return FrameRecord(timestamp_s=timestamp_s, detected=detected, frame_features=features)


def _moving_segment(start_time: float, duration_s: float, *, step_s: float = 0.1, dx_per_step: float = 0.02) -> list[FrameRecord]:
    """Build a moving-hand sequence that will exceed the motion threshold."""
    records: list[FrameRecord] = []
    t = start_time
    x = 0.3
    end = start_time + duration_s
    while t < end - 1e-6:
        records.append(_make_record(t, detected=True, bbox_center=(x, 0.5)))
        x += dx_per_step
        t += step_s
    return records


def _silent_gap(start_time: float, duration_s: float, *, step_s: float = 0.1) -> list[FrameRecord]:
    records: list[FrameRecord] = []
    t = start_time
    end = start_time + duration_s
    while t < end - 1e-6:
        records.append(_make_record(t, detected=False))
        t += step_s
    return records


def test_detect_segments_returns_empty_for_empty_input() -> None:
    assert detect_signing_segments([]) == []


def test_detect_segments_returns_empty_when_no_hand_is_ever_detected() -> None:
    records = _silent_gap(0.0, 2.0)
    assert detect_signing_segments(records) == []


def test_single_gesture_yields_one_segment() -> None:
    records = _silent_gap(0.0, 0.3) + _moving_segment(0.3, 1.2) + _silent_gap(1.5, 0.5)
    segments = detect_signing_segments(records)
    assert len(segments) == 1
    segment = segments[0]
    # Lead-in/out expand the segment a bit; the gesture body (0.3 -> 1.5) must
    # be fully contained.
    assert segment.start_time_s <= 0.3 + 1e-6
    assert segment.end_time_s >= 1.4 - 1e-6


def test_two_gestures_separated_by_long_pause_yield_two_segments() -> None:
    records = (
        _silent_gap(0.0, 0.2)
        + _moving_segment(0.2, 1.0)
        + _silent_gap(1.2, 0.8)  # pause well above min_pause_s
        + _moving_segment(2.0, 1.0)
        + _silent_gap(3.0, 0.3)
    )
    segments = detect_signing_segments(records)
    assert len(segments) == 2
    first, second = segments
    assert first.end_time_s < second.start_time_s


def test_short_pause_below_threshold_keeps_gestures_merged() -> None:
    records = (
        _moving_segment(0.0, 0.8)
        + _silent_gap(0.8, 0.15)  # pause shorter than min_pause_s default 0.3
        + _moving_segment(0.95, 0.8)
    )
    segments = detect_signing_segments(records)
    assert len(segments) == 1


def test_tiny_spurious_motion_below_min_segment_is_dropped() -> None:
    records = (
        _silent_gap(0.0, 0.5)
        + _moving_segment(0.5, 0.2)  # too short, should be dropped
        + _silent_gap(0.7, 0.5)
    )
    segments = detect_signing_segments(records)
    assert segments == []


def test_fallback_whole_clip_covers_detected_range() -> None:
    records = _silent_gap(0.0, 0.3) + _moving_segment(0.3, 0.3, dx_per_step=0.001) + _silent_gap(0.6, 0.3)
    # Near-stationary hand is below threshold, so detect_signing_segments
    # may return no segments even though the hand was present.
    assert detect_signing_segments(records) == []
    fallback = fallback_whole_clip_segment(records)
    assert fallback is not None
    assert fallback.start_time_s >= 0.3 - 1e-6
    assert fallback.end_time_s <= 0.6 + 1e-6


def test_fallback_returns_none_when_nothing_detected() -> None:
    assert fallback_whole_clip_segment(_silent_gap(0.0, 1.0)) is None


def test_lead_in_and_lead_out_expand_segment_bounds() -> None:
    records = (
        _silent_gap(0.0, 1.0)
        + _moving_segment(1.0, 1.0)
        + _silent_gap(2.0, 1.0)
    )
    # All silent frames are undetected, so lead-in/out expansion walks into
    # undetected territory but still grants a wider time window.
    segments = detect_signing_segments(records, lead_in_s=0.3, lead_out_s=0.3)
    assert len(segments) == 1
    segment = segments[0]
    assert segment.start_time_s < 1.0
    assert segment.end_time_s > 2.0 - 1e-6
