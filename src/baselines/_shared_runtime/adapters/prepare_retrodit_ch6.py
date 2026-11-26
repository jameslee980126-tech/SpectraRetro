#!/usr/bin/env python3
"""Create a frozen CH6 split in the filename/layout expected by RetroDiT."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


REACTION_COLUMN = "reactants>reagents>production"
OUTPUT_NAMES = {
    "train": "uspto50k_train.csv",
    "val": "uspto50k_val.csv",
    "test": "uspto50k_test.csv",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert(source: Path, target: Path) -> dict:
    with source.open("r", encoding="utf-8", newline="") as input_handle:
        reader = csv.DictReader(input_handle)
        required = {"class", REACTION_COLUMN}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{source} is missing columns: {sorted(missing)}")
        rows = list(reader)

    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(reader.fieldnames or [])
    with target.open("w", encoding="utf-8", newline="") as output_handle:
        writer = csv.DictWriter(output_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    classes = Counter(int(row["class"]) for row in rows)
    malformed = sum(row[REACTION_COLUMN].count(">") != 2 for row in rows)
    if malformed:
        raise ValueError(f"{source} contains {malformed} malformed reaction SMILES rows")
    return {
        "source": str(source.resolve()),
        "source_sha256": sha256(source),
        "output": str(target.resolve()),
        "output_sha256": sha256(target),
        "rows": len(rows),
        "class_counts": dict(sorted(classes.items())),
        "reaction_class_retained_as_metadata_only": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    split_sources = {"train": args.train, "val": args.val, "test": args.test}
    raw_root = args.output_root / "raw"
    manifest = {
        "model": "RetroDiT",
        "information_regime": "unknown reaction class",
        "class_conditioning": False,
        "note": (
            "The released USPTO loader requires a class column to build metadata, "
            "but configs/data/uspto_50k.yaml sets label.as_input=false and "
            "label.as_target=false."
        ),
        "splits": {},
    }
    for split, source in split_sources.items():
        manifest["splits"][split] = convert(source, raw_root / OUTPUT_NAMES[split])

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

