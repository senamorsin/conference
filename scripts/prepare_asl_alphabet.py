from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.landmarks.mediapipe_extractor import MediaPipeHandExtractor
from src.letters.labels import FEATURE_COLUMNS, normalize_dataset_label
from src.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract MediaPipe hand landmarks from the ASL Alphabet dataset.")
    parser.add_argument(
        "--config",
        default="configs/model_letters.yaml",
        help="Path to the YAML config describing the dataset and output paths.",
    )
    return parser.parse_args()


def resolve_train_dir(root_dir: Path, train_subdir: str) -> Path:
    candidates = [root_dir / train_subdir, root_dir]
    for candidate in candidates:
        if not candidate.exists() or not candidate.is_dir():
            continue

        current = candidate
        while True:
            child_dirs = sorted(path for path in current.iterdir() if path.is_dir())
            if not child_dirs:
                break

            if any(normalize_dataset_label(path.name) is not None for path in child_dirs):
                return current

            if len(child_dirs) == 1:
                current = child_dirs[0]
                continue

            break

    raise FileNotFoundError(f"Dataset directory was not found: {candidate}")


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)

    dataset_config = config["dataset"]
    features_config = config["features"]

    accepted_labels = {str(label).upper() for label in dataset_config["accepted_labels"]}
    dataset_root = Path(dataset_config["root_dir"]).expanduser().resolve()
    train_dir = resolve_train_dir(dataset_root, str(dataset_config.get("train_subdir", "asl_alphabet_train")))
    output_csv = Path(features_config["output_csv"]).expanduser().resolve()
    summary_path = output_csv.with_suffix(".summary.json")
    max_images_per_class = features_config.get("max_images_per_class")
    min_detection_confidence = float(features_config.get("min_detection_confidence", 0.5))
    delegate = str(features_config.get("delegate", "cpu"))

    rows: list[dict[str, object]] = []
    saved_counter: Counter[str] = Counter()
    skipped_counter: Counter[str] = Counter()
    extractor = MediaPipeHandExtractor(
        min_detection_confidence=min_detection_confidence,
        delegate=delegate,
    )

    print(f"Using MediaPipe delegate: {extractor.delegate}")

    try:
        for label_dir in sorted(train_dir.iterdir()):
            if not label_dir.is_dir():
                continue

            normalized_label = normalize_dataset_label(label_dir.name)
            if normalized_label is None or normalized_label not in accepted_labels:
                skipped_counter["unsupported_label"] += 1
                continue

            image_paths = sorted(
                path for path in label_dir.iterdir()
                if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
            )
            if max_images_per_class:
                image_paths = image_paths[: int(max_images_per_class)]

            print(f"Processing {normalized_label}: {len(image_paths)} images")

            for image_path in image_paths:
                extraction = extractor.extract_from_image_bytes(image_path.read_bytes())
                if not extraction.landmarks_detected:
                    skipped_counter["no_hand_detected"] += 1
                    continue

                row = {
                    "label": normalized_label,
                    "source_path": str(image_path.relative_to(train_dir)),
                }
                row.update(dict(zip(FEATURE_COLUMNS, extraction.features.tolist(), strict=True)))
                rows.append(row)
                saved_counter[normalized_label] += 1
    finally:
        extractor.close()

    if not rows:
        raise RuntimeError("No feature rows were extracted. Check the dataset path and whether MediaPipe can detect hands.")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)

    summary = {
        "dataset_name": dataset_config["name"],
        "dataset_root": str(dataset_root),
        "train_dir": str(train_dir),
        "output_csv": str(output_csv),
        "delegate": extractor.delegate,
        "accepted_labels": sorted(accepted_labels),
        "rows_written": len(rows),
        "saved_per_label": dict(saved_counter),
        "skipped": dict(skipped_counter),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {len(rows)} feature rows to {output_csv}")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
