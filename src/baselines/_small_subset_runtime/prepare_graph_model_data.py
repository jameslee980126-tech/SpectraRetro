#!/usr/bin/env python3
"""Prepare frozen raw inputs for RetroDiT, RetroSynFlow and Retroformer."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from rdkit import Chem, RDLogger


EXPECTED = {
    "train": (18370, "3CF1513137EE73D497AF8A9BB2B49B87D47510CF534178424DBD902F60E50222"),
    "valid": (2400, "08F0A52E76B1F30DA1449828A54DCA11D2EA3EEFCC401B73C776202181DA3B63"),
    "test": (2400, "D7F01A2D78656C37A698C4F9F9E64DEC8F9EF3DEEACAA0CA48BDCEFC8D77B614"),
}
MODEL_SPLIT = {"train": "train", "valid": "val", "test": "test"}
REACTION_COLUMN = "reactants>reagents>production"


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


def assign_unique_reactant_maps(reactants: str, product: str):
    reactant_mol = Chem.MolFromSmiles(reactants)
    product_mol = Chem.MolFromSmiles(product)
    if reactant_mol is None or product_mol is None:
        raise ValueError("RDKit failed to parse a reaction")
    used = [atom.GetAtomMapNum() for atom in reactant_mol.GetAtoms() if atom.GetAtomMapNum() > 0]
    used.extend(atom.GetAtomMapNum() for atom in product_mol.GetAtoms() if atom.GetAtomMapNum() > 0)
    next_map = max(used, default=0) + 1
    assigned = 0
    for atom in reactant_mol.GetAtoms():
        if atom.GetAtomMapNum() <= 0:
            atom.SetAtomMapNum(next_map)
            next_map += 1
            assigned += 1
    mapped = Chem.MolToSmiles(reactant_mol, canonical=True, isomericSmiles=True)
    mapped_mol = Chem.MolFromSmiles(mapped)
    maps = [atom.GetAtomMapNum() for atom in mapped_mol.GetAtoms()]
    if min(maps, default=0) <= 0 or len(maps) != len(set(maps)):
        raise ValueError("RetroSynFlow reactant atom maps are not positive and unique")
    product_maps = {atom.GetAtomMapNum() for atom in product_mol.GetAtoms() if atom.GetAtomMapNum() > 0}
    if not product_maps.issubset(set(maps)):
        raise ValueError("product maps are not contained in remapped reactants")
    if canonical(mapped) != canonical(reactants):
        raise ValueError("assigning missing maps changed reactant identity")
    return mapped, assigned


def read_rows(path: Path, split: str):
    expected_rows, expected_hash = EXPECTED[split]
    actual_hash = sha256(path)
    if actual_hash != expected_hash:
        raise ValueError(f"{split} SHA-256 mismatch: {actual_hash}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected_rows:
        raise ValueError(f"{split} row count mismatch: {len(rows)}")
    return rows, actual_hash


def write_csv(path: Path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prepare_split(rows, split: str, output_root: Path):
    suffix = MODEL_SPLIT[split]
    retrodit_rows = []
    retroformer_rows = []
    retrosynflow_rows = []
    assigned_atoms = 0
    roundtrip_rows = 0

    for source_row, row in enumerate(rows):
        parts = row["rxn_smiles"].split(">")
        if len(parts) != 3 or not parts[0] or not parts[2]:
            raise ValueError(f"{split} row {source_row}: malformed reaction")
        reactants, reagents, product = parts
        expected_reactants = canonical(reactants)
        expected_product = canonical(product)

        retrodit_reaction = f"{reactants}>{reagents}>{product}"
        retrodit_rows.append(
            {
                "class": row["class"],
                REACTION_COLUMN: retrodit_reaction,
                "source_row": source_row,
            }
        )

        retroformer_reaction = f"{reactants}>>{product}"
        retroformer_rows.append(
            {
                "id": source_row,
                "class": row["class"],
                REACTION_COLUMN: retroformer_reaction,
                "source_row": source_row,
            }
        )

        mapped_reactants, assigned = assign_unique_reactant_maps(reactants, product)
        assigned_atoms += assigned
        retrosynflow_reaction = f"{mapped_reactants}>>{product}"
        retrosynflow_rows.append(
            {
                "class": row["class"],
                REACTION_COLUMN: retrosynflow_reaction,
                "source_row": source_row,
            }
        )

        for reaction in (retrodit_reaction, retroformer_reaction, retrosynflow_reaction):
            out_reactants, _, out_product = reaction.split(">")
            if canonical(out_reactants) != expected_reactants or canonical(out_product) != expected_product:
                raise ValueError(f"{split} row {source_row}: adapter round-trip mismatch")
        roundtrip_rows += 1

    paths = {
        "retrodit": output_root / "retrodit" / "raw" / f"uspto50k_{suffix}.csv",
        "retroformer": output_root / "retroformer" / f"raw_{suffix}.csv",
        "retrosynflow": output_root / "retrosynflow" / "raw" / f"uspto50k_{suffix}.csv",
    }
    write_csv(paths["retrodit"], ["class", REACTION_COLUMN, "source_row"], retrodit_rows)
    write_csv(paths["retroformer"], ["id", "class", REACTION_COLUMN, "source_row"], retroformer_rows)
    write_csv(paths["retrosynflow"], ["class", REACTION_COLUMN, "source_row"], retrosynflow_rows)
    return {
        "rows": len(rows),
        "roundtrip_rows": roundtrip_rows,
        "retrosynflow_new_atom_maps_assigned": assigned_atoms,
        "outputs": {
            model: {"path": path.relative_to(output_root).as_posix(), "sha256": sha256(path)}
            for model, path in paths.items()
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    RDLogger.DisableLog("rdApp.*")

    manifest = {
        "information_regime": "unknown reaction class; class retained as metadata only",
        "retrosynflow_compatibility": {
            "missing_reactant_atom_maps": "assign deterministic positive maps above the reaction maximum",
            "max_dummy_nodes": 30,
            "explicit_H": "add train-observed H to graph vocabulary",
        },
        "splits": {},
    }
    for split in ("train", "valid", "test"):
        rows, source_hash = read_rows(args.data_root / f"{split}.csv", split)
        stats = prepare_split(rows, split, args.output_root)
        stats["source_sha256"] = source_hash
        manifest["splits"][split] = stats

    manifest_path = args.output_root / "graph_data_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
