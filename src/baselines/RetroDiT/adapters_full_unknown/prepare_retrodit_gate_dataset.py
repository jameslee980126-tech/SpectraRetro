"""Create a deterministic, class-balanced RetroDiT end-to-end gate dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def balanced_rows(path: Path, limit: int) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    classes = sorted({row["class"] for row in rows}, key=lambda value: int(value))
    quota, remainder = divmod(limit, len(classes))
    limits = {
        label: quota + int(index < remainder)
        for index, label in enumerate(classes)
    }
    selected: list[dict[str, str]] = []
    counts: Counter[str] = Counter()
    for row in rows:
        label = row["class"]
        if counts[label] >= limits[label]:
            continue
        selected.append(row)
        counts[label] += 1
        if len(selected) == limit:
            break
    if len(selected) != limit:
        raise ValueError(f"{path} supplied only {len(selected)} / {limit} rows")
    return fieldnames, selected


def write_split(source: Path, destination: Path, limit: int) -> dict[str, object]:
    fieldnames, rows = balanced_rows(source, limit)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    return {
        "source": str(source.resolve()),
        "destination": str(destination.resolve()),
        "rows": len(rows),
        "class_counts": dict(sorted(Counter(row["class"] for row in rows).items())),
        "ids": [row["id"] for row in rows],
        "sha256": digest,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=1200)
    parser.add_argument("--valid-size", type=int, default=300)
    parser.add_argument("--test-size", type=int, default=200)
    args = parser.parse_args()

    raw_output = args.output_root / "raw"
    specifications = (
        ("train", "raw_train.csv", args.train_size),
        ("valid", "raw_val.csv", args.valid_size),
        ("test", "raw_test.csv", args.test_size),
    )
    manifest = {
        split: write_split(
            args.source_root / filename,
            raw_output / filename,
            size,
        )
        for split, filename, size in specifications
    }
    manifest_path = args.output_root / "gate_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value["class_counts"] for key, value in manifest.items()}))
    print(manifest_path)


if __name__ == "__main__":
    main()
