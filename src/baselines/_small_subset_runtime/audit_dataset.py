#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from rdkit import Chem, RDLogger


EXPECTED = {
    "train": (18370, "3CF1513137EE73D497AF8A9BB2B49B87D47510CF534178424DBD902F60E50222"),
    "valid": (2400, "08F0A52E76B1F30DA1449828A54DCA11D2EA3EEFCC401B73C776202181DA3B63"),
    "test": (2400, "D7F01A2D78656C37A698C4F9F9E64DEC8F9EF3DEEACAA0CA48BDCEFC8D77B614"),
}
RETROSYNFLOW_ELEMENTS = {
    "N", "C", "O", "S", "Cl", "F", "B", "Br", "P", "Si", "I", "Sn", "Mg", "Cu", "Zn", "Se", "*"
}
SMILES_TOKEN_PATTERN = re.compile(
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>|\*|\$|%[0-9]{2}|[0-9])"
)


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
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def token_count(smiles: str) -> int:
    tokens = SMILES_TOKEN_PATTERN.findall(smiles)
    if "".join(tokens) != smiles:
        raise ValueError(f"DiffER tokenizer cannot round-trip SMILES: {smiles}")
    return len(tokens)


def read_split(path: Path, name: str):
    expected_rows, expected_sha = EXPECTED[name]
    observed_sha = sha256(path)
    if observed_sha != expected_sha:
        raise ValueError(f"{name} SHA-256 mismatch: {observed_sha}")

    rows = []
    elements = Counter()
    missing_map_atoms = 0
    unsupported = Counter()
    added_atom_counts = []
    max_component_count = 0
    max_smiles_chars = {"reactants": 0, "product": 0}
    max_map_free_tokens = {"reactants": 0, "product": 0}
    over_254_tokens = {"reactants": 0, "product": 0}
    no_product_map_components = 0
    rows_with_no_product_map_component = 0

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"class", "rxn_smiles"}.issubset(reader.fieldnames or []):
            raise ValueError(f"{name} columns are incomplete")
        for row_id, row in enumerate(reader):
            parts = row["rxn_smiles"].split(">")
            if len(parts) != 3:
                raise ValueError(f"{name} row {row_id}: reaction does not have three fields")
            reactants, _, product = parts
            reactant_mol = Chem.MolFromSmiles(reactants)
            product_mol = Chem.MolFromSmiles(product)
            if reactant_mol is None or product_mol is None:
                raise ValueError(f"{name} row {row_id}: RDKit parse failed")
            for molecule in (reactant_mol, product_mol):
                for atom in molecule.GetAtoms():
                    symbol = atom.GetSymbol()
                    elements[symbol] += 1
                    if symbol not in RETROSYNFLOW_ELEMENTS:
                        unsupported[symbol] += 1
                    if atom.GetAtomMapNum() <= 0:
                        missing_map_atoms += 1
            product_maps = {a.GetAtomMapNum() for a in product_mol.GetAtoms() if a.GetAtomMapNum() > 0}
            reactant_maps = {a.GetAtomMapNum() for a in reactant_mol.GetAtoms() if a.GetAtomMapNum() > 0}
            if not product_maps.issubset(reactant_maps):
                raise ValueError(f"{name} row {row_id}: product atom maps are not contained in reactants")
            added_atom_counts.append(max(0, reactant_mol.GetNumAtoms() - product_mol.GetNumAtoms()))
            max_component_count = max(max_component_count, len([x for x in reactants.split(".") if x]))
            max_smiles_chars["reactants"] = max(max_smiles_chars["reactants"], len(reactants))
            max_smiles_chars["product"] = max(max_smiles_chars["product"], len(product))
            map_free_reactants = canonical(reactants)
            map_free_product = canonical(product)
            token_lengths = {
                "reactants": token_count(map_free_reactants),
                "product": token_count(map_free_product),
            }
            for side, length in token_lengths.items():
                max_map_free_tokens[side] = max(max_map_free_tokens[side], length)
                over_254_tokens[side] += int(length > 254)
            row_no_product_map_components = 0
            for component in reactants.split("."):
                component_mol = Chem.MolFromSmiles(component)
                component_maps = {
                    atom.GetAtomMapNum()
                    for atom in component_mol.GetAtoms()
                    if atom.GetAtomMapNum() > 0
                }
                if not (component_maps & product_maps):
                    row_no_product_map_components += 1
            no_product_map_components += row_no_product_map_components
            rows_with_no_product_map_component += int(row_no_product_map_components > 0)
            rows.append(
                {
                    "row": row_id,
                    "product": map_free_product,
                    "reactants": map_free_reactants,
                }
            )

    if len(rows) != expected_rows:
        raise ValueError(f"{name} row count {len(rows)} != {expected_rows}")
    return rows, {
        "rows": len(rows),
        "unique_products": len({row["product"] for row in rows}),
        "elements": dict(sorted(elements.items())),
        "retrosynflow_unsupported_element_occurrences": dict(sorted(unsupported.items())),
        "unmapped_atom_occurrences": missing_map_atoms,
        "added_atoms": {
            "max": max(added_atom_counts),
            "over_10_rows": sum(value > 10 for value in added_atom_counts),
            "over_20_rows": sum(value > 20 for value in added_atom_counts),
            "over_30_rows": sum(value > 30 for value in added_atom_counts),
        },
        "max_reactant_components": max_component_count,
        "max_smiles_characters": max_smiles_chars,
        "differ_map_free_token_lengths": {
            "max": max_map_free_tokens,
            "rows_over_254_excluding_bos_eos": over_254_tokens,
        },
        "reactant_components_without_product_atom_map": {
            "components": no_product_map_components,
            "rows": rows_with_no_product_map_component,
        },
        "sha256": observed_sha,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--test-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    RDLogger.DisableLog("rdApp.*")

    all_rows = {}
    report = {"splits": {}}
    for split in ("train", "valid", "test"):
        rows, stats = read_split(args.data_root / f"{split}.csv", split)
        all_rows[split] = rows
        report["splits"][split] = stats

    product_sets = {split: {row["product"] for row in rows} for split, rows in all_rows.items()}
    overlaps = {
        "train_valid": len(product_sets["train"] & product_sets["valid"]),
        "train_test": len(product_sets["train"] & product_sets["test"]),
        "valid_test": len(product_sets["valid"] & product_sets["test"]),
    }
    if any(overlaps.values()):
        raise ValueError(f"canonical product leakage detected: {overlaps}")
    report["canonical_product_overlap"] = overlaps

    index_sha = sha256(args.test_index)
    if index_sha != "B525DA479CD8208366CD43F3D76AFCCFCADF42B385BD8F63230A9E4D77A8C810":
        raise ValueError("test_index SHA-256 mismatch")
    groups = Counter()
    with args.test_index.open("r", encoding="utf-8", newline="") as handle:
        index_rows = list(csv.DictReader(handle))
    if len(index_rows) != EXPECTED["test"][0]:
        raise ValueError("test_index row count mismatch")
    for row in index_rows:
        groups[row["similarity_group"]] += 1
    if dict(groups) != {"low": 800, "medium": 800, "high": 800}:
        raise ValueError(f"similarity group mismatch: {dict(groups)}")
    report["test_similarity_groups"] = dict(groups)
    report["test_index_sha256"] = index_sha
    report["status"] = "passed"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
