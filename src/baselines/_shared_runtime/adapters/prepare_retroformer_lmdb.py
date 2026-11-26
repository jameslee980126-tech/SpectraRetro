#!/usr/bin/env python3
import argparse
import hashlib
import json
import sys
from pathlib import Path


EXPECTED_RAW_SHA256 = {
    "train": "e6e1dd1aba23a21b4910206d476f05cac292178b22bffa22d0b658335c26ac84",
    "val": "5fc173ede332b1ec5208061139390ab78c886c8f7757ae7f464e611ecfc0f936",
    "test": "26144b6547679b9de21bb0f8becffdbb421be3320018451476d700ad241ca940",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retroformer-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--intermediate-dir", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument(
        "--known-class",
        choices=("True", "False"),
        default="True",
        help="whether the native RetroFormer input includes the reaction-class token",
    )
    args = parser.parse_args()
    known_class = args.known_class == "True"

    if not (args.intermediate_dir / "vocab_share.pk").is_file():
        raise FileNotFoundError("train-only shared vocabulary is missing")
    for split, expected in EXPECTED_RAW_SHA256.items():
        path = args.data_dir / f"raw_{split}.csv"
        if sha256(path) != expected:
            raise ValueError(f"Retroformer raw_{split}.csv hash mismatch")
        cooked_path = args.data_dir / f"cooked_{split}.lmdb"
        if cooked_path.exists() and not args.output_manifest.exists():
            raise RuntimeError(f"unverified partial LMDB already exists: {cooked_path}")

    sys.path.insert(0, str(args.retroformer_root.resolve()))
    from retroformer.dataset import RetroDataset

    counts = {}
    for split in ("train", "val", "test"):
        dataset = RetroDataset(
            mode=split,
            data_folder=str(args.data_dir),
            intermediate_folder=str(args.intermediate_dir),
            known_class=known_class,
            shared_vocab=True,
            augment=split == "train",
        )
        counts[split] = len(dataset)
        dataset.env.close()

    payload = {
        "model": "Retroformer",
        "known_class": known_class,
        "vocabulary_scope": "frozen_training_split_only",
        "raw_sha256": EXPECTED_RAW_SHA256,
        "lmdb_rows": counts,
        "external_source_changed": False,
    }
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
