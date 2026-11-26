"""Create an immutable-schema adapter for RetroDiT's official RC predictor.

The study's frozen CH6 split stores the mapped reaction in ``rxn_smiles``.
The newly released official RetroDiT RC package instead requires exactly an
``id,class,reactants>reagents>production`` CSV.  This script only renames that
field and adds a deterministic split-local ID; it never changes SMILES, class
labels, or the frozen source files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


SPLITS = ("train", "eval", "test")
REQUIRED_COLUMNS = {"class", "rxn_smiles", "original_class"}
OUTPUT_COLUMNS = ("id", "class", "reactants>reagents>production", "original_class")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adapt_split(source: Path, destination: Path, split: str) -> dict[str, object]:
    with source.open("r", encoding="utf-8", newline="") as reader_handle:
        reader = csv.DictReader(reader_handle)
        if reader.fieldnames is None or not REQUIRED_COLUMNS.issubset(reader.fieldnames):
            raise ValueError(
                f"{source} must contain {sorted(REQUIRED_COLUMNS)}; "
                f"found {reader.fieldnames}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8", newline="") as writer_handle:
            writer = csv.DictWriter(writer_handle, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()
            rows = 0
            for rows, row in enumerate(reader, start=1):
                reaction = row["rxn_smiles"]
                if reaction.count(">") != 2:
                    raise ValueError(f"{source}:{rows + 1} is not a three-part reaction")
                writer.writerow(
                    {
                        "id": f"ch6_{split}_{rows - 1:06d}",
                        "class": row["class"],
                        "reactants>reagents>production": reaction,
                        "original_class": row["original_class"],
                    }
                )
    return {
        "source": str(source),
        "source_sha256": sha256(source),
        "output": str(destination),
        "output_sha256": sha256(destination),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    manifest = {"adapter": Path(__file__).name, "splits": {}}
    for split in SPLITS:
        source = args.stage1_dir / f"selected_{split}.csv"
        if split == "eval":
            output_name = "valid.csv"
        else:
            output_name = f"{split}.csv"
        manifest["splits"][split] = adapt_split(source, args.output_dir / output_name, split)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
