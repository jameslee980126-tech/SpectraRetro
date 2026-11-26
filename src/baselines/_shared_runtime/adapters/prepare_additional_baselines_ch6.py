#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
from pathlib import Path

from rdkit import Chem


DATASET_NAME = "ch_activation_6class_strict3d_fixed_no_halogen_donor_drop_orig2_20260606"
EXPECTED_SHA256 = {
    "selected_train.csv": "957e27d354d6b82f6bc1eab1fda8f5ec365517bede2e4d31112c3677ca50f648",
    "selected_eval.csv": "9731bd8cfa1e6865deea5b5498184a823e37908c570769d8d75a4b6152860503",
    "selected_test.csv": "9fd0e110a733462561be5e1c3be01b45e7b4f2a9c2488d7a7646dc50f9a00de4",
}
SPLITS = {
    "train": ("selected_train.csv", "train", "train"),
    "valid": ("selected_eval.csv", "eval", "val"),
    "test": ("selected_test.csv", "test", "test"),
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_reaction(value):
    parts = str(value).split(">")
    if len(parts) != 3 or not parts[0] or not parts[2]:
        raise ValueError(f"invalid reaction SMILES: {value}")
    return parts[0], parts[1], parts[2]


def prepare_retrosim_reaction(reactants, product):
    product_molecule = Chem.MolFromSmiles(product)
    if product_molecule is None:
        raise ValueError(f"invalid RetroSim product SMILES: {product}")

    product_maps = {
        atom.GetAtomMapNum()
        for atom in product_molecule.GetAtoms()
        if atom.GetAtomMapNum() > 0
    }
    contributing_reactants = []
    for component in reactants.split("."):
        molecule = Chem.MolFromSmiles(component)
        if molecule is None:
            raise ValueError(f"invalid RetroSim reactant SMILES: {component}")
        contributes = False
        for atom in molecule.GetAtoms():
            atom_map = atom.GetAtomMapNum()
            if atom_map in product_maps:
                contributes = True
            elif atom_map > 0:
                atom.SetAtomMapNum(0)
        if contributes:
            contributing_reactants.append(Chem.MolToSmiles(molecule, True))

    if not contributing_reactants:
        raise ValueError(f"RetroSim reaction has no product-contributing reactants: {reactants}")

    mapped_product = Chem.MolToSmiles(product_molecule, True)
    for atom in product_molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    map_free_product = Chem.MolToSmiles(product_molecule, True)
    mapped_reaction = f"{'.'.join(contributing_reactants)}>>{mapped_product}"
    return mapped_reaction, map_free_product


def read_split(path, limit):
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"class", "rxn_smiles"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"missing columns in {path}: {sorted(required)}")
        for source_row, row in enumerate(reader):
            reaction_class = int(row["class"])
            if reaction_class not in range(1, 7):
                raise ValueError(f"unexpected class {reaction_class} at row {source_row}")
            reactants, reagents, product = parse_reaction(row["rxn_smiles"])
            rows.append(
                {
                    "source_row": source_row,
                    "class": reaction_class,
                    "original_class": row.get("original_class", reaction_class),
                    "reaction": row["rxn_smiles"],
                    "reactants": reactants,
                    "reagents": reagents,
                    "product": product,
                    "mapped_reaction": f"{reactants}>>{product}",
                }
            )
            if limit is not None and len(rows) >= limit:
                break
    return rows


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def register_output(manifest, root, path, rows):
    relative = path.relative_to(root).as_posix()
    manifest["outputs"][relative] = {"rows": rows, "sha256": sha256(path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": DATASET_NAME,
        "limit": args.limit,
        "split_policy": "frozen_source_rows",
        "vocabulary_and_template_policy": "train_only",
        "retrosim_policy": {
            "reaction_format": "mapped_product_contributing_reactants>>mapped_product",
            "retrieval_product": "atom_map_free_canonical_smiles",
            "reference": "official proc_data.ipynb and test_similarity.py semantics",
        },
        "source_sha256": {},
        "rows": {},
        "outputs": {},
    }

    for split, (filename, graphretro_name, retroformer_name) in SPLITS.items():
        source_path = args.source_dir / filename
        actual_hash = sha256(source_path)
        if actual_hash != EXPECTED_SHA256[filename]:
            raise ValueError(f"SHA256 mismatch for {source_path}: {actual_hash}")
        records = read_split(source_path, args.limit)
        manifest["source_sha256"][filename] = actual_hash
        manifest["rows"][split] = len(records)

        common_path = args.output_dir / "common" / f"{split}.jsonl"
        write_jsonl(
            common_path,
            [
                {
                    "source_row": row["source_row"],
                    "class": row["class"],
                    "rxn_smiles": row["reaction"],
                    "mapped_rxn": row["mapped_reaction"],
                }
                for row in records
            ],
        )
        register_output(manifest, args.output_dir, common_path, len(records))

        graph2edits_path = args.output_dir / "graph2edits" / f"canonicalized_{split}.csv"
        write_csv(
            graph2edits_path,
            ["id", "class", "mapped_rxn", "source_row"],
            [
                {
                    "id": row["source_row"],
                    "class": row["class"],
                    "mapped_rxn": row["mapped_reaction"],
                    "source_row": row["source_row"],
                }
                for row in records
            ],
        )
        register_output(manifest, args.output_dir, graph2edits_path, len(records))

        graphretro_path = args.output_dir / "graphretro" / f"{graphretro_name}.csv"
        graphretro_rows = [
            {
                "id": row["source_row"],
                "class": row["class"],
                "reactants>reagents>production": row["mapped_reaction"],
            }
            for row in records
        ]
        write_csv(
            graphretro_path,
            ["id", "class", "reactants>reagents>production"],
            graphretro_rows,
        )
        register_output(manifest, args.output_dir, graphretro_path, len(records))

        retroxpert_path = (
            args.output_dir / "retroxpert" / "canonicalized_csv" / f"{split}.csv"
        )
        write_csv(
            retroxpert_path,
            ["id", "class", "rxn_smiles", "source_row"],
            [
                {
                    "id": row["source_row"],
                    "class": row["class"],
                    "rxn_smiles": row["mapped_reaction"],
                    "source_row": row["source_row"],
                }
                for row in records
            ],
        )
        register_output(manifest, args.output_dir, retroxpert_path, len(records))

        gln_path = args.output_dir / "gln" / DATASET_NAME / f"raw_{retroformer_name}.csv"
        write_csv(
            gln_path,
            ["id", "class", "reactants>reagents>production"],
            [
                {
                    "id": row["source_row"],
                    "class": row["class"],
                    "reactants>reagents>production": row["reaction"],
                }
                for row in records
            ],
        )
        register_output(manifest, args.output_dir, gln_path, len(records))

        retroformer_path = args.output_dir / "retroformer" / f"raw_{retroformer_name}.csv"
        write_csv(
            retroformer_path,
            ["id", "class", "reactants>reagents>production", "source_row"],
            [
                {
                    "id": row["source_row"],
                    "class": row["class"],
                    "reactants>reagents>production": row["mapped_reaction"],
                    "source_row": row["source_row"],
                }
                for row in records
            ],
        )
        register_output(manifest, args.output_dir, retroformer_path, len(records))

        retrosim_path = args.output_dir / "retrosim" / f"{split}.csv"
        retrosim_label = "val" if split == "valid" else split
        retrosim_rows = []
        for row in records:
            retrosim_reaction, retrosim_product = prepare_retrosim_reaction(
                row["reactants"], row["product"]
            )
            retrosim_rows.append(
                {
                    "source_row": row["source_row"],
                    "class": row["class"],
                    "rxn_smiles": retrosim_reaction,
                    "prod_smiles": retrosim_product,
                    "dataset": retrosim_label,
                }
            )
        write_csv(
            retrosim_path,
            ["source_row", "class", "rxn_smiles", "prod_smiles", "dataset"],
            retrosim_rows,
        )
        register_output(manifest, args.output_dir, retrosim_path, len(records))

    retrosim_combined = args.output_dir / "retrosim" / "data_processed.csv"
    combined_rows = []
    for split in ("train", "valid", "test"):
        with (args.output_dir / "retrosim" / f"{split}.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            combined_rows.extend(csv.DictReader(handle))
    write_csv(
        retrosim_combined,
        ["source_row", "class", "rxn_smiles", "prod_smiles", "dataset"],
        combined_rows,
    )
    register_output(manifest, args.output_dir, retrosim_combined, len(combined_rows))

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
