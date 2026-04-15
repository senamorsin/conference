from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.letters.labels import FEATURE_COLUMNS
from src.utils.config import load_yaml


FINGERS = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze interpretable landmark geometry for confused ASL letter pairs."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML config describing the model, features CSV, and output path.",
    )
    return parser.parse_args()


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    denominator = float(np.linalg.norm(ba) * np.linalg.norm(bc))
    if denominator <= 1e-8:
        return 0.0
    cosine = float(np.clip(np.dot(ba, bc) / denominator, -1.0, 1.0))
    return float(math.degrees(math.acos(cosine)))


def distance_to_segment(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    segment = end - start
    denominator = float(np.dot(segment, segment))
    if denominator <= 1e-8:
        return euclidean(point, start)
    t = float(np.clip(np.dot(point - start, segment) / denominator, 0.0, 1.0))
    projection = start + t * segment
    return euclidean(point, projection)


def crossing_score(first_a: np.ndarray, first_b: np.ndarray, second_a: np.ndarray, second_b: np.ndarray) -> float:
    first_delta = first_a[0] - first_b[0]
    second_delta = second_a[0] - second_b[0]
    return float(first_delta * second_delta)


def build_geometry_features(landmarks: np.ndarray) -> dict[str, float]:
    wrist = landmarks[0]
    thumb_cmc, thumb_mcp, thumb_ip, thumb_tip = landmarks[1], landmarks[2], landmarks[3], landmarks[4]
    index_mcp, index_pip, index_dip, index_tip = landmarks[5], landmarks[6], landmarks[7], landmarks[8]
    middle_mcp, middle_pip, middle_dip, middle_tip = landmarks[9], landmarks[10], landmarks[11], landmarks[12]
    ring_mcp, ring_pip, ring_dip, ring_tip = landmarks[13], landmarks[14], landmarks[15], landmarks[16]
    pinky_mcp, pinky_pip, pinky_dip, pinky_tip = landmarks[17], landmarks[18], landmarks[19], landmarks[20]

    palm_width = max(euclidean(index_mcp, pinky_mcp), 1e-8)
    palm_center = (index_mcp + middle_mcp + ring_mcp + pinky_mcp) / 4.0

    features: dict[str, float] = {
        "palm_width": palm_width,
        "thumb_tip_to_index_tip": euclidean(thumb_tip, index_tip),
        "thumb_tip_to_middle_tip": euclidean(thumb_tip, middle_tip),
        "thumb_tip_to_ring_tip": euclidean(thumb_tip, ring_tip),
        "thumb_tip_to_pinky_tip": euclidean(thumb_tip, pinky_tip),
        "thumb_tip_to_index_mcp": euclidean(thumb_tip, index_mcp),
        "thumb_tip_to_middle_mcp": euclidean(thumb_tip, middle_mcp),
        "thumb_tip_to_ring_mcp": euclidean(thumb_tip, ring_mcp),
        "thumb_tip_to_pinky_mcp": euclidean(thumb_tip, pinky_mcp),
        "thumb_tip_to_palm_center": euclidean(thumb_tip, palm_center),
        "thumb_to_index_gap_line": distance_to_segment(thumb_tip, index_mcp, middle_mcp),
        "thumb_to_middle_ring_line": distance_to_segment(thumb_tip, middle_mcp, ring_mcp),
        "index_middle_tip_gap": euclidean(index_tip, middle_tip),
        "middle_ring_tip_gap": euclidean(middle_tip, ring_tip),
        "ring_pinky_tip_gap": euclidean(ring_tip, pinky_tip),
        "index_middle_mcp_gap": euclidean(index_mcp, middle_mcp),
        "middle_ring_mcp_gap": euclidean(middle_mcp, ring_mcp),
        "ring_pinky_mcp_gap": euclidean(ring_mcp, pinky_mcp),
        "index_cross_middle": crossing_score(index_tip, middle_tip, index_pip, middle_pip),
        "middle_cross_ring": crossing_score(middle_tip, ring_tip, middle_pip, ring_pip),
        "index_tip_to_wrist": euclidean(index_tip, wrist),
        "middle_tip_to_wrist": euclidean(middle_tip, wrist),
        "ring_tip_to_wrist": euclidean(ring_tip, wrist),
        "pinky_tip_to_wrist": euclidean(pinky_tip, wrist),
        "thumb_tip_to_wrist": euclidean(thumb_tip, wrist),
        "index_tip_y": float(index_tip[1]),
        "middle_tip_y": float(middle_tip[1]),
        "ring_tip_y": float(ring_tip[1]),
        "pinky_tip_y": float(pinky_tip[1]),
        "thumb_tip_y": float(thumb_tip[1]),
        "index_tip_x": float(index_tip[0]),
        "middle_tip_x": float(middle_tip[0]),
        "ring_tip_x": float(ring_tip[0]),
        "pinky_tip_x": float(pinky_tip[0]),
        "thumb_tip_x": float(thumb_tip[0]),
        "index_pip_angle": angle(index_mcp, index_pip, index_dip),
        "index_dip_angle": angle(index_pip, index_dip, index_tip),
        "middle_pip_angle": angle(middle_mcp, middle_pip, middle_dip),
        "middle_dip_angle": angle(middle_pip, middle_dip, middle_tip),
        "ring_pip_angle": angle(ring_mcp, ring_pip, ring_dip),
        "ring_dip_angle": angle(ring_pip, ring_dip, ring_tip),
        "pinky_pip_angle": angle(pinky_mcp, pinky_pip, pinky_dip),
        "pinky_dip_angle": angle(pinky_pip, pinky_dip, pinky_tip),
        "thumb_mcp_angle": angle(thumb_cmc, thumb_mcp, thumb_ip),
        "thumb_ip_angle": angle(thumb_mcp, thumb_ip, thumb_tip),
    }

    for finger_name, (mcp_idx, pip_idx, dip_idx, tip_idx) in FINGERS.items():
        mcp = landmarks[mcp_idx]
        tip = landmarks[tip_idx]
        features[f"{finger_name}_extension"] = euclidean(tip, mcp)
        features[f"{finger_name}_tip_to_palm_center"] = euclidean(tip, palm_center)

    return features


def effect_size(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return 0.0
    mean_diff = float(np.mean(a) - np.mean(b))
    pooled = float(np.sqrt((np.var(a) + np.var(b)) / 2.0))
    if pooled <= 1e-8:
        return 0.0
    return mean_diff / pooled


def build_pair_summary(frame: pd.DataFrame, label_a: str, label_b: str, top_k: int) -> dict[str, object]:
    pair_frame = frame[frame["label"].isin([label_a, label_b])].copy()
    feature_columns = [
        column for column in pair_frame.columns
        if column not in {"label", "source_path", "pred_label", "is_correct"}
    ]

    rows_a = pair_frame[pair_frame["label"] == label_a]
    rows_b = pair_frame[pair_frame["label"] == label_b]

    ranked: list[dict[str, object]] = []
    for feature_name in feature_columns:
        values_a = rows_a[feature_name].to_numpy(dtype=float)
        values_b = rows_b[feature_name].to_numpy(dtype=float)
        d = effect_size(values_a, values_b)
        ranked.append(
            {
                "feature": feature_name,
                "effect_size": d,
                "abs_effect_size": abs(d),
                f"{label_a}_mean": float(np.mean(values_a)),
                f"{label_b}_mean": float(np.mean(values_b)),
            }
        )

    ranked.sort(key=lambda row: row["abs_effect_size"], reverse=True)

    confusions_ab = rows_a[rows_a["pred_label"] == label_b]["source_path"].astype(str).tolist()
    confusions_ba = rows_b[rows_b["pred_label"] == label_a]["source_path"].astype(str).tolist()

    return {
        "pair": [label_a, label_b],
        "samples": {
            label_a: int(len(rows_a)),
            label_b: int(len(rows_b)),
        },
        "confusions": {
            f"{label_a}_to_{label_b}": {
                "count": len(confusions_ab),
                "sample_paths": confusions_ab[:5],
            },
            f"{label_b}_to_{label_a}": {
                "count": len(confusions_ba),
                "sample_paths": confusions_ba[:5],
            },
        },
        "top_features": ranked[:top_k],
    }


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)

    analysis_config = config["analysis"]
    model_path = Path(analysis_config["model_path"]).expanduser().resolve()
    features_csv = Path(analysis_config["features_csv"]).expanduser().resolve()
    output_path = Path(analysis_config["output_path"]).expanduser().resolve()
    top_k = int(analysis_config.get("top_k", 8))
    pairs = [tuple(pair) for pair in analysis_config["pairs"]]

    artifact = joblib.load(model_path)
    model = artifact["model"]
    labels = {str(label) for label in artifact["labels"]}
    expected_feature_columns = artifact.get("feature_columns", list(FEATURE_COLUMNS))

    frame = pd.read_csv(features_csv)
    frame = frame[frame["label"].astype(str).isin(labels)].copy()
    feature_columns = [column for column in expected_feature_columns if column in frame.columns]
    if len(feature_columns) != len(expected_feature_columns):
        raise ValueError(
            f"Expected {len(expected_feature_columns)} feature columns from the model artifact, "
            f"found {len(feature_columns)} in {features_csv}"
        )

    X_eval = frame[feature_columns].to_numpy(dtype="float32")
    frame["pred_label"] = model.predict(X_eval).astype(str)
    frame["is_correct"] = frame["label"].astype(str) == frame["pred_label"].astype(str)

    landmarks = frame[feature_columns].to_numpy(dtype="float32").reshape((-1, 21, 3))
    geometry_rows = [build_geometry_features(points) for points in landmarks]
    geometry_frame = pd.DataFrame(geometry_rows)

    analysis_frame = pd.concat(
        [
            frame[["label", "source_path", "pred_label", "is_correct"]].reset_index(drop=True),
            geometry_frame.reset_index(drop=True),
        ],
        axis=1,
    )

    summaries = [
        build_pair_summary(analysis_frame, label_a=label_a, label_b=label_b, top_k=top_k)
        for label_a, label_b in pairs
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    print(f"Saved pair analysis to {output_path}")
    for summary in summaries:
        label_a, label_b = summary["pair"]
        first_feature = summary["top_features"][0]
        print(
            f"{label_a}/{label_b}: top feature {first_feature['feature']} "
            f"(effect={first_feature['effect_size']:.2f})"
        )


if __name__ == "__main__":
    main()
