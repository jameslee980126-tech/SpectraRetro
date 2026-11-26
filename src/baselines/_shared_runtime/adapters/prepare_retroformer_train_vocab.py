#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import pickle
import sys
from pathlib import Path

from rdkit import Chem, RDLogger


EXPECTED_TRAIN_SHA256 = "957e27d354d6b82f6bc1eab1fda8f5ec365517bede2e4d31112c3677ca50f648"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retroformer-root", type=Path, required=True)
    parser.add_argument("--selected-train", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    train_hash = sha256(args.selected_train)
    if train_hash != EXPECTED_TRAIN_SHA256:
        raise ValueError("selected_train.csv SHA-256 does not match the frozen split")

    sys.path.insert(0, str(args.retroformer_root.resolve()))
    from retroformer.utils.smiles_utils import (
        canonical_smiles_with_am,
        clear_map_number,
        remove_am_without_canonical,
        smi_tokenizer,
    )

    RDLogger.DisableLog("rdApp.*")
    tokens = set()
    total_rows = 0
    usable_rows = 0
    with args.selected_train.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            total_rows += 1
            parts = row["rxn_smiles"].split(">")
            if len(parts) != 3 or not parts[0] or not parts[2]:
                continue
            reactants = parts[0]
            product = parts[2]
            canonical_product = clear_map_number(product)
            canonical_reactants_with_maps = canonical_smiles_with_am(reactants)
            canonical_reactants = remove_am_without_canonical(canonical_reactants_with_maps)
            if (
                Chem.MolFromSmiles(canonical_product) is None
                or Chem.MolFromSmiles(canonical_reactants) is None
            ):
                continue
            tokens.update(smi_tokenizer(canonical_product))
            tokens.update(smi_tokenizer(canonical_reactants))
            usable_rows += 1

    tokens.update(f"<RX_{reaction_class}>" for reaction_class in range(1, 11))
    tokens.add("<UNK>")
    vocabulary = ["<unk>", "<pad>", "<sos>", "<eos>"] + sorted(tokens)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = args.output_dir / "vocab_share.pk"
    with vocab_path.open("wb") as handle:
        pickle.dump([vocabulary, vocabulary], handle)

    manifest = {
        "model": "Retroformer",
        "source_commit": "4e94540e06f5a0ed52ac7b7c4acdd98e92aeda56",
        "vocabulary_scope": "frozen_training_split_only",
        "selected_train_sha256": train_hash,
        "total_training_rows": total_rows,
        "usable_training_rows": usable_rows,
        "shared_vocabulary_tokens": len(vocabulary),
        "vocab_sha256": sha256(vocab_path),
    }
    manifest_path = args.output_dir / "vocab_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
