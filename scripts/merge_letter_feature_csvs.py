from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge multiple ASL landmark feature CSV files into one training dataset."
    )
    parser.add_argument(
        "--config",
        default="configs/model_letters_combined.yaml",
        help="Path to the YAML config describing input CSV files and the merged output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)

    merge_config = config["merge"]
    input_csvs = [Path(path).expanduser().resolve() for path in merge_config["input_csvs"]]
    output_csv = Path(config["features"]["output_csv"]).expanduser().resolve()
    summary_path = output_csv.with_suffix(".summary.json")

    frames: list[pd.DataFrame] = []
    rows_per_source: dict[str, int] = {}

    for input_csv in input_csvs:
        if not input_csv.exists():
            raise FileNotFoundError(f"Input feature CSV was not found: {input_csv}")

        frame = pd.read_csv(input_csv)
        frame["source_dataset_csv"] = str(input_csv)
        frames.append(frame)
        rows_per_source[str(input_csv)] = int(len(frame))

    if not frames:
        raise RuntimeError("No input feature CSVs were provided.")

    merged = pd.concat(frames, ignore_index=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)

    summary = {
        "output_csv": str(output_csv),
        "total_rows": int(len(merged)),
        "input_csvs": [str(path) for path in input_csvs],
        "rows_per_source": rows_per_source,
        "label_counts": {
            str(label): int(count)
            for label, count in merged["label"].astype(str).value_counts().sort_index().items()
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Merged {len(input_csvs)} CSV files into {output_csv}")
    print(f"Total rows: {len(merged)}")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
