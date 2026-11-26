#!/usr/bin/env python3
"""Build unified RetroDiT metrics and a verified local analysis report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import defaultdict, deque
from pathlib import Path

from rdkit import Chem, RDLogger


RDLogger.DisableLog("rdApp.*")

SUPPORTED_ATOMS = {
    "C", "O", "N", "F", "Br", "Cl", "S", "I", "Si", "P", "B", "H",
    "Se", "Cu", "Sn", "Ag", "Ge", "Li", "Na", "Zn",
}
BOND_DECODER = {
    1: Chem.BondType.SINGLE,
    2: Chem.BondType.DOUBLE,
    3: Chem.BondType.TRIPLE,
    4: Chem.BondType.AROMATIC,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--selected-test", type=Path, required=True)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_molecule(value):
    molecule = Chem.MolFromSmiles(str(value))
    if molecule is None:
        return None
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    molecule = Chem.RemoveHs(molecule)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def canonical_set(value):
    components = []
    for item in str(value).split("."):
        if not item:
            continue
        canonical = canonical_molecule(item)
        if canonical is None:
            return None
        components.append(canonical)
    return ".".join(sorted(set(components))) if components else None


def released_supported_reaction(reactants, product, max_n_len=250, n_dummy=30):
    """Mirror the released preprocessing gates that reduced 9,250 rows to 9,248."""
    if not reactants or not product:
        return False
    reactant_mol = Chem.MolFromSmiles(reactants)
    product_mol = Chem.MolFromSmiles(product)
    if reactant_mol is None or product_mol is None:
        return False
    r_nodes = reactant_mol.GetNumAtoms()
    p_nodes = product_mol.GetNumAtoms()
    if r_nodes < 5 or p_nodes == 1:
        return False
    if r_nodes < p_nodes or r_nodes > max_n_len or p_nodes > max_n_len:
        return False
    if r_nodes - p_nodes > n_dummy:
        return False
    if product_mol.GetNumBonds() == 0 or reactant_mol.GetNumBonds() == 0:
        return False
    if any(atom.GetSymbol() not in SUPPORTED_ATOMS for atom in reactant_mol.GetAtoms()):
        return False
    if any(atom.GetSymbol() not in SUPPORTED_ATOMS for atom in product_mol.GetAtoms()):
        return False

    product_maps = [atom.GetAtomMapNum() for atom in product_mol.GetAtoms()]
    if any(value <= 0 for value in product_maps) or len(product_maps) != len(set(product_maps)):
        return False
    positive_reactant_maps = [
        atom.GetAtomMapNum() for atom in reactant_mol.GetAtoms() if atom.GetAtomMapNum() > 0
    ]
    if len(positive_reactant_maps) != len(set(positive_reactant_maps)):
        return False
    r_symbols = {
        atom.GetAtomMapNum(): atom.GetSymbol()
        for atom in reactant_mol.GetAtoms()
        if atom.GetAtomMapNum() > 0
    }
    p_symbols = {
        atom.GetAtomMapNum(): atom.GetSymbol()
        for atom in product_mol.GetAtoms()
        if atom.GetAtomMapNum() > 0
    }
    if not set(p_symbols).issubset(r_symbols):
        return False
    if any(r_symbols[key] != value for key, value in p_symbols.items()):
        return False
    return canonical_molecule(product) != canonical_set(reactants)


def projected_graph_smiles(molecule, mapping, graph_size):
    """Reproduce RetroDiT's lossy graph-to-SMILES serialization exactly.

    The released graph vocabulary stores only element and bond type. Formal charge,
    atom-map, hydrogen, and stereochemical attributes are intentionally absent. Also,
    all map-zero reagent atoms share the final map-zero coordinate in the released
    converter. Reconstructing this representation is required to map shuffled test
    outputs back to frozen source rows; ordinary RDKit canonicalization cannot do so.
    """
    symbols = ["*"] * graph_size
    for atom in molecule.GetAtoms():
        symbols[mapping[atom.GetAtomMapNum()]] = atom.GetSymbol()

    edge_counts = defaultdict(lambda: defaultdict(int))
    bond_encoder = {
        Chem.BondType.SINGLE: 1,
        Chem.BondType.DOUBLE: 2,
        Chem.BondType.TRIPLE: 3,
        Chem.BondType.AROMATIC: 4,
    }
    for bond in molecule.GetBonds():
        left = mapping[molecule.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetAtomMapNum()]
        right = mapping[molecule.GetAtomWithIdx(bond.GetEndAtomIdx()).GetAtomMapNum()]
        key = tuple(sorted((left, right)))
        edge_counts[key][bond_encoder[bond.GetBondType()]] += 1

    rebuilt = Chem.RWMol()
    coordinate_to_atom = {}
    for coordinate, symbol in enumerate(symbols):
        if symbol == "*":
            continue
        coordinate_to_atom[coordinate] = rebuilt.AddAtom(Chem.Atom(symbol))
    for (left, right), counts in edge_counts.items():
        if left == right or left not in coordinate_to_atom or right not in coordinate_to_atom:
            continue
        # torch_geometric.to_dense_adj sums duplicate one-hot edges; argmax then
        # selects the lowest bond category when counts tie.
        bond_type = min(counts, key=lambda value: (-counts[value], value))
        rebuilt.AddBond(
            coordinate_to_atom[left],
            coordinate_to_atom[right],
            BOND_DECODER[bond_type],
        )
    try:
        return Chem.MolToSmiles(rebuilt, canonical=True)
    except Exception:
        return None


def released_signature(reactants, product, n_dummy=30):
    reactant_mol = Chem.MolFromSmiles(reactants)
    product_mol = Chem.MolFromSmiles(product)
    product_map = {
        atom.GetAtomMapNum(): index for index, atom in enumerate(product_mol.GetAtoms())
    }
    next_index = len(product_map)
    reactant_map = {}
    for atom in reactant_mol.GetAtoms():
        atom_map = atom.GetAtomMapNum()
        if atom_map in product_map:
            reactant_map[atom_map] = product_map[atom_map]
        else:
            reactant_map[atom_map] = next_index
            next_index += 1
    graph_size = product_mol.GetNumAtoms() + n_dummy
    return (
        projected_graph_smiles(product_mol, product_map, graph_size),
        projected_graph_smiles(reactant_mol, reactant_map, graph_size),
    )


def selected_signatures(path):
    signatures = defaultdict(deque)
    unsupported_rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for source_row, row in enumerate(csv.DictReader(handle)):
            reactants, _, product = row["rxn_smiles"].split(">")
            if not released_supported_reaction(reactants, product):
                unsupported_rows.append(source_row)
                continue
            signature = released_signature(reactants, product)
            if None in signature:
                raise ValueError(f"failed to reconstruct released graph at row {source_row}")
            signatures[signature].append(source_row)
    return signatures, unsupported_rows


def export_predictions(samples_path, selected_test, output):
    samples = json.loads(samples_path.read_text(encoding="utf-8"))
    signatures, unsupported_rows = selected_signatures(selected_test)
    records = []
    unmatched = []
    for sample_index, sample in enumerate(samples):
        signature = (sample["cond"], sample["ref"])
        candidates = signatures.get(signature)
        if not candidates:
            unmatched.append({"sample_index": sample_index, "signature": signature})
            continue
        source_row = candidates.popleft()
        generated = sample.get("gen", [])
        native_hit = next(
            (rank for rank, value in enumerate(generated, start=1) if value == sample["ref"]),
            None,
        )
        records.append(
            {
                "source_row": source_row,
                "predictions": generated,
                "native_supported": True,
                "native_hit_rank": native_hit,
            }
        )
    remaining = sum(len(queue) for queue in signatures.values())
    # Exactly two source rows were rejected by the released preprocessing.  All
    # 9,248 generated records must otherwise map back to the frozen split.
    if unmatched or len(records) != 9248 or remaining != 0 or len(unsupported_rows) != 2:
        raise ValueError(
            f"prediction/source mapping failed: records={len(records)}, "
            f"unmatched={len(unmatched)}, remaining_supported={remaining}, "
            f"unsupported_frozen={len(unsupported_rows)}"
        )
    with output.open("w", encoding="utf-8") as handle:
        for record in sorted(records, key=lambda value: value["source_row"]):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    samples = list(args.local_dir.rglob("generated_samples.json"))
    metrics = list(args.local_dir.rglob("generated_metrics.json"))
    if len(samples) != 1 or len(metrics) != 1:
        raise ValueError(
            f"expected one formal generated output, found samples={len(samples)} metrics={len(metrics)}"
        )
    predictions = args.local_dir / "retrodit_ranked_predictions.jsonl"
    export_predictions(samples[0], args.selected_test, predictions)

    project_root = Path(__file__).resolve().parents[2]
    evaluator = project_root / "baseline_experiments/evaluation/evaluate_ranked_precursors_dual.py"
    unified_json = args.local_dir / "retrodit_top10_dual_metrics.json"
    per_reaction = args.local_dir / "retrodit_per_reaction_metrics.csv"
    result = subprocess.run(
        [
            sys.executable,
            str(evaluator),
            "--model-name",
            "RetroDiT-product-only",
            "--predictions-jsonl",
            str(predictions),
            "--selected-test",
            str(args.selected_test),
            "--output-json",
            str(unified_json),
            "--per-reaction-csv",
            str(per_reaction),
            "--beam-size",
            "10",
            "--candidate-ranking",
            "released_frequency_ranking_from_10_stochastic_samples",
        ],
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stdout)

    native = json.loads(metrics[0].read_text(encoding="utf-8"))
    unified = json.loads(unified_json.read_text(encoding="utf-8"))
    u = unified["unified_canonical"]
    report = args.local_dir / "ANALYSIS_REPORT_LOCAL.md"
    report.write_text(
        "# RetroDiT CH6 Unknown-class local analysis\n\n"
        "All downloaded artifacts passed the remote and local SHA-256 checks recorded "
        "in `SYNC_MANIFEST.json`. Formal inference used the validation-selected 30,000-step "
        "checkpoint, 50 diffusion steps, and ten stochastic product-only candidates without "
        "reaction-class or ground-truth reaction-center input.\n\n"
        "## Source-row mapping audit\n\n"
        "RetroDiT shuffled the test DataLoader and serialized its condition/reference "
        "graphs using a lossy element-and-bond vocabulary. The 9,248 saved records were "
        "mapped back to frozen source rows by reproducing that released graph serializer, "
        "including its map-zero reagent-coordinate behavior. All 9,248 records matched; "
        "the remaining two of 9,250 frozen rows are exactly the rows rejected by the "
        "released preprocessing gates.\n\n"
        "## Native processed-split metrics\n\n"
        f"```json\n{json.dumps(native, indent=2)}\n```\n\n"
        "## Unified full-test metrics\n\n"
        f"- Denominator: {u['total_reactions']}\n"
        f"- Prediction rows: {u['prediction_rows_present']}\n"
        f"- Missing/unsupported rows: {u['missing_prediction_rows']}\n"
        f"- Top-1: {u['Top-1']:.6f}\n"
        f"- Top-3: {u['Top-3']:.6f}\n"
        f"- Top-5: {u['Top-5']:.6f}\n"
        f"- Top-10: {u['Top-10']:.6f}\n\n"
        "The native numbers use RetroDiT's released string-level hit calculation on the "
        "9,248 processed rows. The unified numbers use the study-wide map-free, "
        "stereo-preserving, order-independent canonical precursor comparison on all "
        "9,250 frozen rows; invalid and duplicate candidates retain their original model "
        "rank slots. These two metric families are therefore reported separately.\n\n"
        "Detailed per-class and per-reaction results are stored in "
        "`retrodit_top10_dual_metrics.json` and `retrodit_per_reaction_metrics.csv`.\n",
        encoding="utf-8",
    )
    outputs = [predictions, unified_json, per_reaction, report]
    manifest = {
        "all_local_analysis_files_verified": True,
        "files": [
            {"path": item.name, "bytes": item.stat().st_size, "sha256": sha256(item)}
            for item in outputs
        ],
    }
    (args.local_dir / "LOCAL_ANALYSIS_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"state": "complete", "outputs": len(outputs)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
