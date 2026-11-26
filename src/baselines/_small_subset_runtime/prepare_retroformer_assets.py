#!/usr/bin/env python3
"""Build a train-only Retroformer vocabulary and exact-row LMDB datasets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import sys
from pathlib import Path

from rdkit import Chem, RDLogger


EXPECTED_ROWS = {"train": 18370, "val": 2400, "test": 2400}
REACTION_COLUMN = "reactants>reagents>production"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def build_train_vocab(retroformer_root: Path, train_path: Path, intermediate_dir: Path):
    sys.path.insert(0, str(retroformer_root.resolve()))
    from retroformer.utils.smiles_utils import (
        canonical_smiles_with_am,
        clear_map_number,
        remove_am_without_canonical,
        smi_tokenizer,
    )

    tokens = set()
    usable = 0
    with train_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != EXPECTED_ROWS["train"]:
        raise ValueError(f"training row count mismatch: {len(rows)}")
    for source_row, row in enumerate(rows):
        reactants, _, product = row[REACTION_COLUMN].split(">")
        canonical_product = clear_map_number(product)
        canonical_reactants_with_maps = canonical_smiles_with_am(reactants)
        canonical_reactants = remove_am_without_canonical(canonical_reactants_with_maps)
        if Chem.MolFromSmiles(canonical_product) is None or Chem.MolFromSmiles(canonical_reactants) is None:
            raise ValueError(f"training row {source_row} failed Retroformer canonicalization")
        tokens.update(smi_tokenizer(canonical_product))
        tokens.update(smi_tokenizer(canonical_reactants))
        usable += 1
    tokens.update(f"<RX_{reaction_class}>" for reaction_class in range(1, 11))
    tokens.add("<UNK>")
    vocabulary = ["<unk>", "<pad>", "<sos>", "<eos>"] + sorted(tokens)
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = intermediate_dir / "vocab_share.pk"
    with vocab_path.open("wb") as handle:
        pickle.dump([vocabulary, vocabulary], handle)
    return {
        "scope": "frozen training split only",
        "training_rows": len(rows),
        "usable_training_rows": usable,
        "tokens": len(vocabulary),
        "sha256": sha256(vocab_path),
    }


def build_lmdb(retroformer_root: Path, data_dir: Path, intermediate_dir: Path):
    sys.path.insert(0, str(retroformer_root.resolve()))
    from retroformer.dataset import RetroDataset

    counts = {}
    for split in ("train", "val", "test"):
        raw_path = data_dir / f"raw_{split}.csv"
        with raw_path.open("r", encoding="utf-8", newline="") as handle:
            raw_rows = sum(1 for _ in csv.DictReader(handle))
        if raw_rows != EXPECTED_ROWS[split]:
            raise ValueError(f"{split} raw row count mismatch: {raw_rows}")
        dataset = RetroDataset(
            mode=split,
            data_folder=str(data_dir),
            intermediate_folder=str(intermediate_dir),
            known_class=False,
            shared_vocab=True,
            augment=split == "train",
        )
        counts[split] = len(dataset)
        dataset.env.close()
        if counts[split] != raw_rows:
            raise RuntimeError(f"{split} LMDB silently dropped rows: raw={raw_rows}, lmdb={counts[split]}")
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retroformer-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--intermediate-dir", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()
    RDLogger.DisableLog("rdApp.*")

    train_path = args.data_dir / "raw_train.csv"
    manifest = {
        "model": "Retroformer",
        "source_commit": "4e94540e06f5a0ed52ac7b7c4acdd98e92aeda56",
        "known_class": False,
        "shared_vocabulary": True,
        "raw_sha256": {
            split: sha256(args.data_dir / f"raw_{split}.csv")
            for split in ("train", "val", "test")
        },
    }
    manifest["vocabulary"] = build_train_vocab(args.retroformer_root, train_path, args.intermediate_dir)
    manifest["lmdb_rows"] = build_lmdb(args.retroformer_root, args.data_dir, args.intermediate_dir)
    manifest["status"] = "passed"
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
