#!/usr/bin/env python3
"""Adapt the frozen CH6 split to RetroDiT's native USPTO-50K CSV schema.

This adapter changes only filenames and column names. Atom-mapped reaction
SMILES, split membership, reaction classes, and row order are preserved.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


REACTION_COLUMN = "reactants>reagents>production"
OUTPUT_NAMES = {
    "train": "raw_train.csv",
    "valid": "raw_val.csv",
    "test": "raw_test.csv",
}
EXPECTED_ROWS = {"train": 74146, "valid": 9395, "test": 9250}
EXPECTED_SHA256 = {
    "train": "957e27d354d6b82f6bc1eab1fda8f5ec365517bede2e4d31112c3677ca50f648",
    "valid": "9731bd8cfa1e6865deea5b5498184a823e37908c570769d8d75a4b6152860503",
    "test": "9fd0e110a733462561be5e1c3be01b45e7b4f2a9c2488d7a7646dc50f9a00de4",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert(split: str, source: Path, target: Path) -> dict[str, object]:
    source_digest = sha256(source)
    if source_digest != EXPECTED_SHA256[split]:
        raise ValueError(
            f"{source} SHA-256 mismatch: {source_digest} != {EXPECTED_SHA256[split]}"
        )

    with source.open("r", encoding="utf-8", newline="") as input_handle:
        reader = csv.DictReader(input_handle)
        required = {"class", "rxn_smiles"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{source} is missing columns: {sorted(missing)}")
        rows = list(reader)

    if len(rows) != EXPECTED_ROWS[split]:
        raise ValueError(f"{source} has {len(rows)} rows, expected {EXPECTED_ROWS[split]}")

    target.parent.mkdir(parents=True, exist_ok=True)
    output_fields = ["id", "class", REACTION_COLUMN, "original_class"]
    class_counts: Counter[int] = Counter()
    with target.open("w", encoding="utf-8", newline="") as output_handle:
        writer = csv.DictWriter(output_handle, fieldnames=output_fields, lineterminator="\n")
        writer.writeheader()
        for index, row in enumerate(rows):
            reaction = row["rxn_smiles"]
            if reaction.count(">") != 2:
                raise ValueError(f"{source}:{index + 2} is not a three-part reaction SMILES")
            reaction_class = int(row["class"])
            if reaction_class not in range(1, 7):
                raise ValueError(f"{source}:{index + 2} has unexpected class {reaction_class}")
            class_counts[reaction_class] += 1
            writer.writerow(
                {
                    "id": f"ch6_{split}_{index:06d}",
                    "class": reaction_class,
                    REACTION_COLUMN: reaction,
                    "original_class": row.get("original_class", row["class"]),
                }
            )

    return {
        "source": str(source.resolve()),
        "source_sha256": source_digest,
        "output": str(target.resolve()),
        "output_sha256": sha256(target),
        "rows": len(rows),
        "class_counts": dict(sorted(class_counts.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--valid", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    sources = {"train": args.train, "valid": args.valid, "test": args.test}
    raw_root = args.output_root / "raw"
    manifest: dict[str, object] = {
        "model": "RetroDiT",
        "information_regime": "known reaction class",
        "class_conditioning": {
            "author_interface": "data.cond_cfg.label.as_input",
            "as_input": True,
            "as_target": False,
            "k_class": 6,
        },
        "adaptation": "filename and column-schema translation only",
        "splits": {},
    }
    for split, source in sources.items():
        manifest["splits"][split] = convert(
            split, source, raw_root / OUTPUT_NAMES[split]
        )

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
