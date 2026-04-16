from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.words.labels import normalize_asl_citizen_word_label
from src.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an ASL Citizen word manifest for the overlapping word subset.")
    parser.add_argument(
        "--config",
        default="configs/model_words_asl_citizen.yaml",
        help="Path to the YAML config describing the ASL Citizen archive and outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    dataset_config = config["dataset"]

    archive_path = Path(dataset_config["archive_path"]).expanduser().resolve()
    manifest_path = Path(dataset_config["manifest_output_path"]).expanduser().resolve()
    summary_path = manifest_path.with_suffix(".summary.json")
    max_per_split = {
        split_name: int(value)
        for split_name, value in dict(dataset_config.get("max_samples_per_split", {})).items()
    }

    if not archive_path.exists():
        raise FileNotFoundError(f"ASL Citizen archive was not found: {archive_path}")

    rows: list[dict[str, object]] = []
    per_split_counter: dict[str, Counter[str]] = defaultdict(Counter)
    skipped_counter: Counter[str] = Counter()

    with zipfile.ZipFile(archive_path) as zf:
        for split_name in ("train", "val", "test"):
            member = f"ASL_Citizen/splits/{split_name}.csv"
            limit = max_per_split.get(split_name)
            with zf.open(member) as handle:
                reader = csv.DictReader(io.TextIOWrapper(handle, encoding="utf-8"))
                for source_row in reader:
                    canonical_label = normalize_asl_citizen_word_label(str(source_row["Gloss"]))
                    if canonical_label is None:
                        skipped_counter["unsupported_label"] += 1
                        continue

                    if limit is not None and per_split_counter[split_name][canonical_label] >= limit:
                        skipped_counter["limited_by_split_cap"] += 1
                        continue

                    video_file = str(source_row["Video file"]).strip()
                    if not video_file:
                        skipped_counter["missing_video_file"] += 1
                        continue

                    rows.append(
                        {
                            "split": split_name,
                            "label": canonical_label,
                            "raw_label": str(source_row["Gloss"]).strip(),
                            "video_id": Path(video_file).stem,
                            "source_member": f"ASL_Citizen/videos/{video_file}",
                            "source_path": video_file,
                            "start_time": 0.0,
                            "end_time": 999999.0,
                            "participant_id": str(source_row.get("Participant ID", "")).strip(),
                            "asl_lex_code": str(source_row.get("ASL-LEX Code", "")).strip(),
                        }
                    )
                    per_split_counter[split_name][canonical_label] += 1

    if not rows:
        raise RuntimeError("No overlapping ASL Citizen rows were selected.")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "archive_path": str(archive_path),
        "manifest_output_path": str(manifest_path),
        "rows_written": len(rows),
        "unique_videos": len({str(row['video_id']) for row in rows}),
        "counts_per_split": {
            split_name: dict(counter)
            for split_name, counter in per_split_counter.items()
        },
        "skipped": dict(skipped_counter),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {len(rows)} ASL Citizen rows to {manifest_path}")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
