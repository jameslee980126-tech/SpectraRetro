"""Diagnose RDKit rooted-SMILES failures in a processed RetroDiT RC split."""

from __future__ import annotations

import argparse

import torch
from rdkit import Chem

try:
    from torchdrug_pred.chem import get_canonical_map_order
except ImportError:
    get_canonical_map_order = None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("processed_split")
    parser.add_argument("--max-failures", type=int, default=20)
    args = parser.parse_args()

    payload = torch.load(args.processed_split, map_location="cpu", weights_only=False)
    failures = []
    helper_failures = []
    alignment_failures = []
    disconnected = 0
    for sample_index, ((_, product_graph), reaction) in enumerate(
        zip(payload["data"], payload["smiles_list"])
    ):
        product_smiles = reaction.split(">", 2)[2]
        molecule = Chem.MolFromSmiles(product_smiles)
        if molecule is None:
            failures.append((sample_index, "parse", product_smiles))
            continue
        if "." in product_smiles:
            disconnected += 1
        raw_maps = [atom.GetAtomMapNum() for atom in molecule.GetAtoms()]
        graph_maps = [int(value) for value in product_graph.atom_map.tolist()]
        if sorted(raw_maps) != sorted(graph_maps):
            alignment_failures.append(
                (sample_index, raw_maps, graph_maps, product_smiles)
            )
        for root_index in range(molecule.GetNumAtoms()):
            try:
                Chem.MolToSmiles(
                    molecule,
                    isomericSmiles=True,
                    rootedAtAtom=root_index,
                    canonical=True,
                )
            except Exception as error:  # RDKit raises RuntimeError here.
                failures.append(
                    (sample_index, root_index, molecule.GetNumAtoms(), str(error), product_smiles)
                )
                break
        if get_canonical_map_order is not None and "." in product_smiles:
            for root_index in range(molecule.GetNumAtoms()):
                try:
                    order = get_canonical_map_order(product_smiles, root=root_index)
                    if order is None or len(order) != molecule.GetNumAtoms():
                        raise ValueError(
                            f"invalid order length: {None if order is None else len(order)}"
                        )
                except Exception as error:
                    helper_failures.append(
                        (
                            sample_index,
                            root_index,
                            molecule.GetNumAtoms(),
                            str(error),
                            product_smiles,
                        )
                    )
                    break
        if len(failures) >= args.max_failures:
            break

    print(f"samples={len(payload['data'])}")
    print(f"disconnected_products={disconnected}")
    print(f"alignment_failures={len(alignment_failures)}")
    for failure in alignment_failures[: args.max_failures]:
        print("ALIGNMENT_FAILURE", repr(failure))
    print(f"rooted_smiles_failures={len(failures)}")
    for failure in failures:
        print("ROOT_FAILURE", repr(failure))
    if get_canonical_map_order is not None:
        print(f"package_helper_failures={len(helper_failures)}")
        for failure in helper_failures[: args.max_failures]:
            print("HELPER_FAILURE", repr(failure))


if __name__ == "__main__":
    main()
