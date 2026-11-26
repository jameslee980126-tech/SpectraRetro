import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


SPLIT_FILES = {
    "train": "selected_train.csv",
    "valid": "selected_eval.csv",
    "test": "selected_test.csv",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def to_mapped_reaction(reaction: str) -> str:
    reactants, _, product = reaction.split(">", 2)
    return f"{reactants}>>{product}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"dataset": "CH6_69pct", "splits": {}}

    for split, filename in SPLIT_FILES.items():
        source = args.source_dir / filename
        source_frame = pd.read_csv(source)
        frame = source_frame if args.limit is None else source_frame.head(args.limit).copy()
        converted = pd.DataFrame(
            {
                "id": frame.index.astype(int),
                "class": frame["class"].astype(int),
                "mapped_rxn": frame["rxn_smiles"].map(to_mapped_reaction),
                "source_row": frame.index.astype(int),
            }
        )
        output = args.output_dir / f"canonicalized_{split}.csv"
        converted.to_csv(output, index=False)
        manifest["splits"][split] = {
            "source": str(source.resolve()),
            "source_rows": int(len(source_frame)),
            "source_sha256": sha256(source),
            "converted_rows": int(len(converted)),
            "converted_sha256": sha256(output),
        }

    manifest_path = args.output_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(manifest_path)


if __name__ == "__main__":
    main()
