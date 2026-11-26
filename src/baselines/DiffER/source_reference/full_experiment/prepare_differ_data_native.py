#!/usr/bin/env python3
"""Build deterministic 20x root-aligned SMILES data for the DiffER single model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from rdkit import Chem, RDLogger


EXPECTED = {
    "train": (18370, "3CF1513137EE73D497AF8A9BB2B49B87D47510CF534178424DBD902F60E50222"),
    "valid": (2400, "08F0A52E76B1F30DA1449828A54DCA11D2EA3EEFCC401B73C776202181DA3B63"),
    "test": (2400, "D7F01A2D78656C37A698C4F9F9E64DEC8F9EF3DEEACAA0CA48BDCEFC8D77B614"),
}
DIFFER_SPLIT = {"train": "train", "valid": "val", "test": "test"}
_ROOT_MODULE = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def canonical(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"invalid SMILES: {smiles}")
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    fragments = Chem.GetMolFrags(molecule, asMols=True, sanitizeFrags=True)
    return ".".join(
        sorted(
            Chem.MolToSmiles(fragment, canonical=True, isomericSmiles=True)
            for fragment in fragments
        )
    )


def load_root_module(source: str):
    global _ROOT_MODULE
    if _ROOT_MODULE is None:
        source_path = Path(source)
        spec = importlib.util.spec_from_file_location("frozen_root_aligned_smiles", source_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        _ROOT_MODULE = module
    return _ROOT_MODULE


def process_row(task):
    source_path, seed, split, source_row, reaction, augmentation = task
    random.seed(seed)
    reactants, _, product = reaction.split(">")
    module = load_root_module(source_path)
    result = module.get_retro_rsmiles(
        {"product": product, "reactant": reactants, "augmentation": augmentation}
    )
    if result.get("status") != 0:
        return {"split": split, "source_row": source_row, "status": result.get("status")}
    if len(result["src_data"]) != augmentation or len(result["tgt_data"]) != augmentation:
        return {"split": split, "source_row": source_row, "status": "wrong_augmentation_count"}
    expected_product = canonical(product)
    expected_reactants = canonical(reactants)
    for aug_index, (src, tgt) in enumerate(zip(result["src_data"], result["tgt_data"])):
        if canonical(src.replace(" ", "")) != expected_product:
            return {"split": split, "source_row": source_row, "status": f"product_roundtrip_{aug_index}"}
        if canonical(tgt.replace(" ", "")) != expected_reactants:
            return {"split": split, "source_row": source_row, "status": f"reactant_roundtrip_{aug_index}"}
    return {
        "split": split,
        "source_row": source_row,
        "status": 0,
        "src": result["src_data"],
        "tgt": result["tgt_data"],
    }


def read_reactions(path: Path, split: str):
    expected_rows, expected_hash = EXPECTED[split]
    actual_hash = sha256(path)
    if actual_hash != expected_hash:
        raise ValueError(f"{split} SHA-256 mismatch: {actual_hash}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected_rows:
        raise ValueError(f"{split} row count mismatch: {len(rows)}")
    return rows, actual_hash


def write_split(source_path: Path, rows, split: str, output_root: Path, augmentation: int, seed: int, workers: int):
    differ_split = DIFFER_SPLIT[split]
    target_dir = output_root / differ_split
    target_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        (
            str(source_path.resolve()),
            seed * 1_000_003 + {"train": 0, "valid": 1, "test": 2}[split] * 100_003 + row_id,
            split,
            row_id,
            row["rxn_smiles"],
            augmentation,
        )
        for row_id, row in enumerate(rows)
    ]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(process_row, tasks, chunksize=16))

    failures = [result for result in results if result["status"] != 0]
    if failures:
        failure_path = target_dir / "preprocess_failures.json"
        failure_path.write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")
        raise RuntimeError(f"{split}: {len(failures)} root-alignment failures; see {failure_path}")

    src_path = target_dir / f"src-{differ_split}.txt"
    tgt_path = target_dir / f"tgt-{differ_split}.txt"
    index_path = target_dir / f"row-index-{differ_split}.jsonl"
    with src_path.open("w", encoding="utf-8", newline="\n") as src_handle, tgt_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as tgt_handle, index_path.open("w", encoding="utf-8", newline="\n") as index_handle:
        line_index = 0
        for result in results:
            for aug_index, (src, tgt) in enumerate(zip(result["src"], result["tgt"])):
                src_handle.write(src + "\n")
                tgt_handle.write(tgt + "\n")
                index_handle.write(
                    json.dumps(
                        {
                            "line_index": line_index,
                            "source_row": result["source_row"],
                            "augmentation_index": aug_index,
                        }
                    )
                    + "\n"
                )
                line_index += 1
    expected_lines = len(rows) * augmentation
    if line_index != expected_lines:
        raise RuntimeError(f"{split}: wrote {line_index} lines, expected {expected_lines}")
    return {
        "source_rows": len(rows),
        "augmentation": augmentation,
        "output_lines": line_index,
        "src_sha256": sha256(src_path),
        "tgt_sha256": sha256(tgt_path),
        "row_index_sha256": sha256(index_path),
        "roundtrip_identity": "passed",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--root-aligned-source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--augmentation", type=int, default=20)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.augmentation <= 0:
        raise ValueError("augmentation must be positive")
    RDLogger.DisableLog("rdApp.*")

    manifest = {
        "model": "DiffER single",
        "preprocessing": "Root-aligned SMILES, frozen upstream implementation",
        "root_aligned_commit": "0468429c4c2d51e74a7f68ea57014d760d28028e",
        "seed": args.seed,
        "splits": {},
    }
    for split in ("train", "valid", "test"):
        rows, source_hash = read_reactions(args.data_root / f"{split}.csv", split)
        stats = write_split(
            args.root_aligned_source,
            rows,
            split,
            args.output_root,
            args.augmentation,
            args.seed,
            args.workers,
        )
        stats["source_sha256"] = source_hash
        manifest["splits"][split] = stats

    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
