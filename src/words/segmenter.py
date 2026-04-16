"""Heuristic segmentation of a time-ordered list of FrameRecords into signing
segments separated by pauses. The current implementation assumes the signer
pauses briefly between signs (hand leaves frame or goes still). The pause-based
strategy can later be swapped for a sliding-window / CTC-style approach without
touching the service layer: callers only depend on ``detect_signing_segments``
returning a list of ``SigningSegment``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.words.features import FrameRecord


@dataclass(slots=True, frozen=True)
class SigningSegment:
    start_idx: int
    end_idx: int  # inclusive
    start_time_s: float
    end_time_s: float

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_time_s - self.start_time_s)


def detect_signing_segments(
    records: list[FrameRecord],
    *,
    min_pause_s: float = 0.3,
    min_segment_s: float = 0.4,
    motion_threshold: float = 0.008,
    lead_in_s: float = 0.15,
    lead_out_s: float = 0.15,
) -> list[SigningSegment]:
    """Split a records timeline into signing segments.

    Segmentation rules:

    1. A frame is ``active`` if the hand was detected AND the short-window
       smoothed bbox-center velocity is above ``motion_threshold``. Detected
       frames immediately adjacent to active frames are also promoted to
       active, so we don't lose the first/last frame of a gesture where
       velocity is naturally zero.
    2. Consecutive active frames form a raw segment. Adjacent raw segments
       whose inter-segment gap is shorter than ``min_pause_s`` are merged
       back together.
    3. Segments shorter than ``min_segment_s`` are dropped as spurious blips.
    4. Remaining segments are padded by ``lead_in_s`` / ``lead_out_s`` to
       include the start/end of the gesture that the motion threshold trims.
    """

    n = len(records)
    if n == 0:
        return []

    motion = _compute_motion_signal(records)
    smoothed = _rolling_max(motion, radius=2)

    active = np.array(
        [record.detected and smoothed[i] >= motion_threshold for i, record in enumerate(records)],
        dtype=bool,
    )
    active = _promote_adjacent_detected(active, records)

    raw_runs = _find_runs(active)
    merged = _merge_short_gaps(raw_runs, records, min_pause_s=min_pause_s)
    kept = [
        (start, end)
        for start, end in merged
        if records[end].timestamp_s - records[start].timestamp_s >= min_segment_s
    ]

    segments: list[SigningSegment] = []
    for start, end in kept:
        start_expanded = _expand_back(records, start, lead_s=lead_in_s)
        end_expanded = _expand_forward(records, end, lead_s=lead_out_s)
        segments.append(
            SigningSegment(
                start_idx=start_expanded,
                end_idx=end_expanded,
                start_time_s=records[start_expanded].timestamp_s,
                end_time_s=records[end_expanded].timestamp_s,
            )
        )
    return segments


def fallback_whole_clip_segment(records: list[FrameRecord]) -> SigningSegment | None:
    """Treat the whole clip as one segment, spanning the detected range.

    Used by the service as a safety net when the pause-based segmenter finds
    no segments but frames were detected: we'd rather return a single-word
    prediction over the full clip than refuse.
    """

    detected_indices = [i for i, record in enumerate(records) if record.detected]
    if not detected_indices:
        return None
    start = detected_indices[0]
    end = detected_indices[-1]
    return SigningSegment(
        start_idx=start,
        end_idx=end,
        start_time_s=records[start].timestamp_s,
        end_time_s=records[end].timestamp_s,
    )


def _compute_motion_signal(records: list[FrameRecord]) -> np.ndarray:
    n = len(records)
    motion = np.zeros(n, dtype=np.float32)
    last_center: tuple[float, float] | None = None
    for i, record in enumerate(records):
        if not record.detected:
            last_center = None
            continue
        center = record.bbox_center
        if last_center is not None:
            dx = center[0] - last_center[0]
            dy = center[1] - last_center[1]
            motion[i] = float(np.hypot(dx, dy))
        last_center = center
    return motion


def _rolling_max(signal: np.ndarray, radius: int) -> np.ndarray:
    n = len(signal)
    if n == 0 or radius <= 0:
        return signal.copy()
    out = np.zeros(n, dtype=signal.dtype)
    for i in range(n):
        lo = max(0, i - radius)
        hi = min(n, i + radius + 1)
        out[i] = signal[lo:hi].max()
    return out


def _promote_adjacent_detected(active: np.ndarray, records: list[FrameRecord]) -> np.ndarray:
    """Flip detected-but-inactive frames to active when they sit next to an
    active frame on either side. Two passes so effects propagate one step.
    """

    if not active.any():
        return active
    promoted = active.copy()
    n = len(records)
    for i in range(n - 2, -1, -1):
        if promoted[i + 1] and records[i].detected and not promoted[i]:
            promoted[i] = True
    for i in range(1, n):
        if promoted[i - 1] and records[i].detected and not promoted[i]:
            promoted[i] = True
    return promoted


def _find_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    n = len(mask)
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        start = i
        while i < n and mask[i]:
            i += 1
        runs.append((start, i - 1))
    return runs


def _merge_short_gaps(
    runs: list[tuple[int, int]],
    records: list[FrameRecord],
    *,
    min_pause_s: float,
) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for run in runs:
        if merged:
            prev_end_time = records[merged[-1][1]].timestamp_s
            this_start_time = records[run[0]].timestamp_s
            if (this_start_time - prev_end_time) < min_pause_s:
                merged[-1] = (merged[-1][0], run[1])
                continue
        merged.append(run)
    return merged


def _expand_back(records: list[FrameRecord], start_idx: int, lead_s: float) -> int:
    if lead_s <= 0.0 or start_idx == 0:
        return start_idx
    target_time = records[start_idx].timestamp_s - lead_s
    idx = start_idx
    while idx > 0 and records[idx - 1].timestamp_s >= target_time:
        idx -= 1
    return idx


def _expand_forward(records: list[FrameRecord], end_idx: int, lead_s: float) -> int:
    if lead_s <= 0.0:
        return end_idx
    n = len(records)
    if end_idx >= n - 1:
        return end_idx
    target_time = records[end_idx].timestamp_s + lead_s
    idx = end_idx
    while idx < n - 1 and records[idx + 1].timestamp_s <= target_time:
        idx += 1
    return idx
