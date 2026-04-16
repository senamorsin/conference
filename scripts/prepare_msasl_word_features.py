from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.landmarks.mediapipe_extractor import MediaPipeHandExtractor
from src.words.features import extract_word_features_from_video
from src.words.labels import WORD_FEATURE_COLUMNS, WORD_SEQUENCE_LENGTH
from src.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract temporal landmark features for a word-recognition dataset.")
    parser.add_argument(
        "--config",
        default="configs/model_words_msasl.yaml",
        help="Path to the YAML config describing the manifest and output CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    dataset_config = config["dataset"]
    download_config = config["download"]
    features_config = config["features"]

    manifest_path = Path(dataset_config["manifest_output_path"]).expanduser().resolve()
    raw_video_dir = Path(download_config["raw_video_dir"]).expanduser().resolve()
    output_csv = Path(features_config["output_csv"]).expanduser().resolve()
    summary_path = output_csv.with_suffix(".summary.json")
    sequence_length = int(features_config.get("sequence_length", WORD_SEQUENCE_LENGTH))
    delegate = str(features_config.get("delegate", "cpu"))

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest was not found: {manifest_path}. Run build_msasl_word_manifest.py first.")

    rows = list(csv.DictReader(manifest_path.open("r", encoding="utf-8")))
    feature_rows: list[dict[str, object]] = []
    saved_counter: Counter[str] = Counter()
    skipped_counter: Counter[str] = Counter()
    extractor = MediaPipeHandExtractor(
        max_num_hands=2,
        delegate=delegate,
    )

    print(f"Using MediaPipe delegate: {extractor.delegate}")

    try:
        for row in rows:
            video_path = find_existing_video(raw_video_dir, row)
            if video_path is None:
                skipped_counter["missing_video"] += 1
                continue

            extraction = extract_word_features_from_video(
                video_path=video_path,
                extractor=extractor,
                sequence_length=sequence_length,
                start_time=float(row["start_time"]),
                end_time=float(row["end_time"]),
            )
            if extraction.detected_steps == 0:
                skipped_counter["no_hand_detected"] += 1
                continue

            feature_row = {
                "split": str(row["split"]),
                "label": str(row["label"]),
                "video_id": str(row["video_id"]),
                "source_path": str(video_path.relative_to(raw_video_dir)),
                "start_time": float(row["start_time"]),
                "end_time": float(row["end_time"]),
                "detected_steps": extraction.detected_steps,
                "sampled_steps": extraction.sampled_steps,
            }
            feature_row.update(dict(zip(WORD_FEATURE_COLUMNS, extraction.features.tolist(), strict=True)))
            feature_rows.append(feature_row)
            saved_counter[str(row["label"])] += 1
    finally:
        extractor.close()

    if not feature_rows:
        raise RuntimeError("No word feature rows were extracted. Check downloads and MediaPipe detection.")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(feature_rows).to_csv(output_csv, index=False)

    summary = {
        "manifest_path": str(manifest_path),
        "raw_video_dir": str(raw_video_dir),
        "output_csv": str(output_csv),
        "delegate": extractor.delegate,
        "sequence_length": sequence_length,
        "rows_written": len(feature_rows),
        "saved_per_label": dict(saved_counter),
        "skipped": dict(skipped_counter),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {len(feature_rows)} word-feature rows to {output_csv}")
    print(f"Summary saved to {summary_path}")


def find_existing_video(raw_video_dir: Path, row: dict) -> Path | None:
    source_path = row.get("source_path", "")
    if source_path:
        candidate = raw_video_dir / Path(source_path).name
        if candidate.exists():
            return candidate

    video_id = str(row.get("video_id", ""))
    if video_id:
        matches = sorted(
            p for p in raw_video_dir.glob(f"{video_id}.*")
            if p.suffix not in (".part", ".ytdl")
        )
        if matches:
            return matches[0]

    return None


if __name__ == "__main__":
    main()
