from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.words.labels import WORD_LABELS, normalize_msasl_word_label
from src.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a balanced MS-ASL manifest for word-mode training (see WORD_LABELS).")
    parser.add_argument(
        "--config",
        default="configs/model_words_msasl.yaml",
        help="Path to the YAML config describing the dataset roots and output manifest.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)

    dataset_config = config["dataset"]
    root_dir = Path(dataset_config["root_dir"]).expanduser().resolve()
    manifest_path = Path(dataset_config["manifest_output_path"]).expanduser().resolve()
    summary_path = manifest_path.with_suffix(".summary.json")
    max_per_split = {
        split_name: int(value)
        for split_name, value in dict(dataset_config.get("max_samples_per_split", {})).items()
    }

    rows: list[dict[str, object]] = []
    per_split_counter: dict[str, Counter[str]] = defaultdict(Counter)
    skipped_counter: Counter[str] = Counter()

    for split_name, msasl_filename in (
        ("train", "MSASL_train.json"),
        ("val", "MSASL_val.json"),
        ("test", "MSASL_test.json"),
    ):
        split_candidates: list[dict[str, object]] = []
        for sample in json.loads((root_dir / msasl_filename).read_text(encoding="utf-8")):
            canonical_label = normalize_msasl_word_label(str(sample["text"]))
            if canonical_label is None:
                skipped_counter["unsupported_label"] += 1
                continue

            url = normalize_youtube_url(str(sample["url"]))
            video_id = extract_youtube_id(url)
            if video_id is None:
                skipped_counter["invalid_youtube_url"] += 1
                continue

            split_candidates.append(
                {
                    "split": split_name,
                    "label": canonical_label,
                    "raw_label": str(sample["text"]).strip(),
                    "video_id": video_id,
                    "url": url,
                    "start_time": float(sample["start_time"]),
                    "end_time": float(sample["end_time"]),
                    "signer_id": int(sample["signer_id"]),
                    "fps": float(sample.get("fps", 0.0) or 0.0),
                    "width": int(sample.get("width", 0) or 0),
                    "height": int(sample.get("height", 0) or 0),
                }
            )

        selected_rows = select_rows_for_split(
            split_candidates=split_candidates,
            split_name=split_name,
            label_limit=max_per_split.get(split_name),
        )
        rows.extend(selected_rows)
        per_split_counter[split_name].update(str(row["label"]) for row in selected_rows)

        label_limit = max_per_split.get(split_name)
        if label_limit is not None:
            missing = sum(max(0, label_limit - per_split_counter[split_name][label]) for label in WORD_LABELS)
            if missing:
                skipped_counter["limited_by_split_cap"] += missing

    if not rows:
        raise RuntimeError("No MS-ASL word rows were selected. Check the dataset path and label alias mapping.")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "dataset_root": str(root_dir),
        "manifest_output_path": str(manifest_path),
        "rows_written": len(rows),
        "unique_videos": len({str(row["video_id"]) for row in rows}),
        "counts_per_split": {
            split_name: dict(counter)
            for split_name, counter in per_split_counter.items()
        },
        "skipped": dict(skipped_counter),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {len(rows)} word-manifest rows to {manifest_path}")
    print(f"Summary saved to {summary_path}")


def normalize_youtube_url(url: str) -> str:
    normalized = url.strip()
    if normalized.startswith("www."):
        normalized = f"https://{normalized}"
    if normalized.startswith("http://"):
        normalized = "https://" + normalized.removeprefix("http://")
    return normalized


def extract_youtube_id(url: str) -> str | None:
    if "v=" not in url:
        return None
    query = url.split("v=", 1)[1]
    return query.split("&", 1)[0].strip() or None


def select_rows_for_split(
    split_candidates: list[dict[str, object]],
    split_name: str,
    label_limit: int | None,
) -> list[dict[str, object]]:
    if label_limit is None:
        return split_candidates

    remaining = Counter({label: label_limit for label in WORD_LABELS})
    by_video: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in split_candidates:
        by_video[str(row["video_id"])].append(row)

    selected: list[dict[str, object]] = []
    while any(count > 0 for count in remaining.values()):
        best_video_id = None
        best_video_score = 0
        best_video_rows: list[dict[str, object]] = []

        for video_id, rows in by_video.items():
            useful_rows = [row for row in rows if remaining[str(row["label"])] > 0]
            if not useful_rows:
                continue

            score = sum(min(remaining[label], count) for label, count in Counter(str(row["label"]) for row in useful_rows).items())
            if score > best_video_score:
                best_video_id = video_id
                best_video_score = score
                best_video_rows = useful_rows

        if best_video_id is None:
            break

        by_video.pop(best_video_id, None)
        per_label_rows: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in sorted(best_video_rows, key=lambda item: float(item["start_time"])):
            per_label_rows[str(row["label"])].append(row)

        for label, rows in per_label_rows.items():
            take = min(remaining[label], len(rows))
            if take <= 0:
                continue
            selected.extend(rows[:take])
            remaining[label] -= take

    return sorted(selected, key=lambda item: (str(item["label"]), str(item["video_id"]), float(item["start_time"])))


if __name__ == "__main__":
    main()
