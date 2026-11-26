"""Portable integrity and chemistry checks for the packaged CHRetro subset."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

from rdkit import Chem, rdBase, RDLogger

import similarity as sim


ROOT = Path(__file__).resolve().parent
FROZEN = ROOT / "frozen"
FULL = ROOT.parent / "full_CHRetro_92791"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def group_for(value, rule):
    if value < rule["low"]:
        return "low"
    if value < rule["high"]:
        return "medium"
    return "high"


def main():
    RDLogger.DisableLog("rdApp.warning")
    manifest = load_json(FROZEN / "manifest.json")
    rule = load_json(FROZEN / "thresholds.json")["rule"]

    assert sha256(ROOT / "selection_plan.json") == manifest["selection_plan_sha256"]
    assert sha256(ROOT / "build_dataset.py") == manifest["build_script_sha256"]
    for name, expected in manifest["files"].items():
        assert sha256(FROZEN / name) == expected, name

    product_keys = {}
    total_rows = 0
    recomputed = 0
    for split in ("train", "valid", "test"):
        full_rows = list(csv.DictReader((FULL / f"{split}.csv").open(encoding="utf-8-sig")))
        rows = list(csv.DictReader((FROZEN / f"{split}.csv").open(encoding="utf-8-sig")))
        index = list(csv.DictReader((FROZEN / f"{split}_index.csv").open(encoding="utf-8-sig")))
        design = load_json(FROZEN / f"{split}_design.json")
        assert len(rows) == len(index) == len(design) == manifest["counts"][split]

        selected_by_product = defaultdict(set)
        parent_by_product = defaultdict(set)
        for item in design:
            parent_by_product[item["product_group"]].add(item["sample_id"])

        sample_for_recompute = set()
        for label in ("low", "medium", "high"):
            candidates = [item for item in design if item["similarity_group"] == label]
            candidates.sort(key=lambda item: hashlib.sha256(("verify|" + item["sample_id"]).encode()).hexdigest())
            sample_for_recompute.update(item["sample_id"] for item in candidates[:8])

        group_cells = defaultdict(Counter)
        for row, idx, meta in zip(rows, index, design):
            source_row = int(idx["source_row"])
            assert row == full_rows[source_row]
            assert idx["sample_id"] == meta["sample_id"] == f"{split}:{source_row:06d}"
            assert idx["class"] == meta["class"] == row["class"]

            reactants, _, product = row["rxn_smiles"].split(">")
            reactant_mol = Chem.MolFromSmiles(reactants)
            product_mol = Chem.MolFromSmiles(product)
            assert reactant_mol is not None and product_mol is not None

            product_maps = [atom.GetAtomMapNum() for atom in product_mol.GetAtoms()]
            assert 0 not in product_maps and len(product_maps) == len(set(product_maps))
            reactant_bonds = {
                tuple(sorted((bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum())))
                for bond in reactant_mol.GetBonds()
            }
            added = [
                tuple(sorted((bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum())))
                for bond in product_mol.GetBonds()
                if tuple(sorted((bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum())))
                not in reactant_bonds
            ]
            assert len(added) == 1 and list(added[0]) == meta["gold_atom_maps"]
            assert product_mol.GetNumHeavyAtoms() == meta["heavy_atoms"]
            assert product_mol.GetNumAtoms() + product_mol.GetNumBonds() == meta["candidate_count"]

            copy = Chem.Mol(product_mol)
            for atom in copy.GetAtoms():
                atom.SetAtomMapNum(0)
            Chem.RemoveStereochemistry(copy)
            product_key = Chem.MolToSmiles(copy, canonical=True, isomericSmiles=False)
            assert product_key == idx["product_group"] == meta["product_group"]
            selected_by_product[product_key].add(idx["sample_id"])

            score = meta["radii"]["2"]["difficulty"]
            label = group_for(score, rule)
            assert label == idx["similarity_group"] == meta["similarity_group"]
            atom_bin = 0 if meta["heavy_atoms"] <= 15 else 1 if meta["heavy_atoms"] <= 20 else 2 if meta["heavy_atoms"] <= 30 else 3
            candidate_bin = 0 if meta["candidate_count"] <= 30 else 1 if meta["candidate_count"] <= 45 else 2 if meta["candidate_count"] <= 65 else 3
            group_cells[label][(meta["class"], atom_bin, candidate_bin)] += 1

            if meta["sample_id"] in sample_for_recompute:
                bonds = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in product_mol.GetBonds()]
                gold = next(
                    i for i, (u, v) in enumerate(bonds)
                    if sorted((product_maps[u], product_maps[v])) == list(added[0])
                )
                rooted_gold = sim.rooted_product_key(product_mol, bonds[gold])
                graph = sim.molecular_graph(product_mol)
                _, gold_fp = sim.local_environment(graph, bonds[gold], 2)
                competitors = [
                    i for i, bond in enumerate(bonds)
                    if i != gold and sim.rooted_product_key(product_mol, bond) != rooted_gold
                ]
                measured = max(
                    sim.similarity(gold_fp, sim.local_environment(graph, bonds[i], 2)[1])
                    for i in competitors
                )
                assert abs(measured - score) < 1e-12
                recomputed += 1

        # Every selected product group is complete within the selected design.
        assert selected_by_product == parent_by_product
        if split in ("valid", "test"):
            assert group_cells["low"] == group_cells["medium"] == group_cells["high"]
            assert sum(group_cells["low"].values()) == 800
        product_keys[split] = set(selected_by_product)
        total_rows += len(rows)
        print(f"VERIFIED {split}: {len(rows)} rows, {len(selected_by_product)} products")

    assert not (product_keys["train"] & product_keys["valid"])
    assert not (product_keys["train"] & product_keys["test"])
    assert not (product_keys["valid"] & product_keys["test"])
    print(f"ALL_CHECKS_PASSED: {total_rows} rows; {recomputed} similarities recomputed; RDKit {rdBase.rdkitVersion}")


if __name__ == "__main__":
    main()
