from __future__ import annotations

import argparse
import csv
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract only the overlapping ASL Citizen word subset from the zip archive.")
    parser.add_argument(
        "--config",
        default="configs/model_words_asl_citizen.yaml",
        help="Path to the YAML config describing the archive, manifest, and raw-video directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    dataset_config = config["dataset"]
    download_config = config["download"]

    archive_path = Path(dataset_config["archive_path"]).expanduser().resolve()
    manifest_path = Path(dataset_config["manifest_output_path"]).expanduser().resolve()
    raw_video_dir = Path(download_config["raw_video_dir"]).expanduser().resolve()
    summary_path = raw_video_dir.parent / "extract.summary.json"

    if not archive_path.exists():
        raise FileNotFoundError(f"ASL Citizen archive was not found: {archive_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest was not found: {manifest_path}. Run build_asl_citizen_word_manifest.py first.")

    rows = list(csv.DictReader(manifest_path.open("r", encoding="utf-8")))
    members = sorted({str(row["source_member"]) for row in rows})
    raw_video_dir.mkdir(parents=True, exist_ok=True)

    summary_counter: Counter[str] = Counter()
    with zipfile.ZipFile(archive_path) as zf:
        for member in members:
            output_path = raw_video_dir / Path(member).name
            if output_path.exists():
                summary_counter["already_present"] += 1
                continue

            try:
                with zf.open(member) as source, output_path.open("wb") as target:
                    target.write(source.read())
            except KeyError:
                summary_counter["missing_in_archive"] += 1
                continue

            summary_counter["extracted"] += 1

    summary = {
        "archive_path": str(archive_path),
        "manifest_path": str(manifest_path),
        "raw_video_dir": str(raw_video_dir),
        "requested_members": len(members),
        "results": dict(summary_counter),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Extraction summary saved to {summary_path}")
    for key, value in sorted(summary_counter.items()):
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
