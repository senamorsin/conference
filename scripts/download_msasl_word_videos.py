from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the MS-ASL videos required for the 20-word word-mode subset.")
    parser.add_argument(
        "--config",
        default="configs/model_words_msasl.yaml",
        help="Path to the YAML config describing the manifest and raw-video directory.",
    )
    return parser.parse_args()


def main() -> None:
    try:
        from yt_dlp import YoutubeDL
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "yt-dlp is required for downloading MS-ASL videos. Run this script via "
            "`uv run --with yt-dlp python scripts/download_msasl_word_videos.py ...`."
        ) from exc

    args = parse_args()
    config = load_yaml(args.config)
    dataset_config = config["dataset"]
    download_config = config["download"]

    manifest_path = Path(dataset_config["manifest_output_path"]).expanduser().resolve()
    raw_video_dir = Path(download_config["raw_video_dir"]).expanduser().resolve()
    summary_path = raw_video_dir / "download.summary.json"
    max_downloads = download_config.get("max_downloads")

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest was not found: {manifest_path}. Run build_msasl_word_manifest.py first.")

    raw_video_dir.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(manifest_path.open("r", encoding="utf-8")))
    unique_videos: dict[str, str] = {}
    usage_counter: Counter[str] = Counter()
    for row in rows:
        video_id = str(row["video_id"])
        unique_videos.setdefault(video_id, str(row["url"]))
        usage_counter[video_id] += 1

    queue = sorted(unique_videos.items(), key=lambda item: (-usage_counter[item[0]], item[0]))
    if max_downloads is not None:
        queue = queue[: int(max_downloads)]

    summary_counter: Counter[str] = Counter()
    ydl = YoutubeDL(
        {
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "retries": 2,
            "socket_timeout": 20,
            "outtmpl": str(raw_video_dir / "%(id)s.%(ext)s"),
            "format": "best[height<=480]/best",
        }
    )

    for video_id, url in queue:
        if find_existing_video(raw_video_dir, video_id) is not None:
            summary_counter["already_present"] += 1
            continue

        try:
            ydl.download([url])
        except Exception:
            summary_counter["download_failed"] += 1
            continue

        if find_existing_video(raw_video_dir, video_id) is None:
            summary_counter["download_missing_after_success"] += 1
            continue

        summary_counter["downloaded"] += 1

    summary = {
        "manifest_path": str(manifest_path),
        "raw_video_dir": str(raw_video_dir),
        "unique_videos_considered": len(queue),
        "results": dict(summary_counter),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Download summary saved to {summary_path}")
    for key, value in sorted(summary_counter.items()):
        print(f"{key}: {value}")


def find_existing_video(raw_video_dir: Path, video_id: str) -> Path | None:
    matches = sorted(raw_video_dir.glob(f"{video_id}.*"))
    return matches[0] if matches else None


if __name__ == "__main__":
    main()
