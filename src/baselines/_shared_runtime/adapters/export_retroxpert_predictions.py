#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import pandas as pd


INDEX_PATTERN = re.compile(r"rxn_data_(\d+)\.pkl$")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored-predictions-csv", type=Path, required=True)
    parser.add_argument("--test-data-dir", type=Path, required=True)
    parser.add_argument("--beam-size", type=int, default=10)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.scored_predictions_csv)
    data_files = sorted(args.test_data_dir.glob("rxn_data_*.pkl"), key=lambda path: path.name)
    if len(frame) != len(data_files):
        raise ValueError(f"length mismatch: predictions={len(frame)}, test files={len(data_files)}")

    records = []
    for (_, row), data_file in zip(frame.iterrows(), data_files):
        match = INDEX_PATTERN.search(data_file.name)
        if match is None:
            raise ValueError(f"unexpected test data filename: {data_file.name}")
        native_rank = int(row["rank"])
        records.append(
            {
                "source_row": int(match.group(1)),
                "predictions": [
                    str(row[f"prediction_{rank}"])
                    for rank in range(1, args.beam_size + 1)
                ],
                "native_supported": True,
                "native_hit_rank": native_rank if native_rank > 0 else None,
            }
        )

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8", newline="\n") as handle:
        for record in sorted(records, key=lambda item: item["source_row"]):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"exported {len(records)} RetroXpert prediction rows")


if __name__ == "__main__":
    main()
