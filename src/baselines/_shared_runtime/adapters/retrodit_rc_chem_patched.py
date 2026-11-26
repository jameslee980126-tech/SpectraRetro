from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import Atom, Mol


def clear_map_canonical_smiles(
    smiles: str,
    canonical: bool = True,
    root: int = -1,
) -> str | None:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    for atom in molecule.GetAtoms():
        if atom.HasProp("molAtomMapNumber"):
            atom.ClearProp("molAtomMapNumber")
    fragments = Chem.GetMolFrags(molecule, asMols=False, sanitizeFrags=False)
    if root >= 0 and len(fragments) > 1:
        # RDKit 2022.09 applies ``rootedAtAtom`` independently while writing
        # disconnected fragments and can treat a global atom index as a local
        # one, raising ``Range Error``. Root only the containing fragment and
        # keep the remaining canonical fragments after it. The resulting atom
        # traversal is still a complete, deterministic permutation.
        if root >= molecule.GetNumAtoms():
            raise IndexError("root atom index exceeds molecule atom count")
        rooted_fragment = next(
            fragment for fragment in fragments if root in fragment
        )
        rooted_smiles = Chem.MolFragmentToSmiles(
            molecule,
            atomsToUse=list(rooted_fragment),
            isomericSmiles=True,
            rootedAtAtom=root,
            canonical=canonical,
        )
        other_smiles = sorted(
            Chem.MolFragmentToSmiles(
                molecule,
                atomsToUse=list(fragment),
                isomericSmiles=True,
                canonical=canonical,
            )
            for fragment in fragments
            if fragment != rooted_fragment
        )
        result = ".".join([rooted_smiles, *other_smiles])
    else:
        result = Chem.MolToSmiles(
            molecule,
            isomericSmiles=True,
            rootedAtAtom=root,
            canonical=canonical,
        )
    if canonical and root < 0:
        canonical_molecule = Chem.MolFromSmiles(result)
        if canonical_molecule is None:
            return None
        result = Chem.MolToSmiles(
            canonical_molecule,
            isomericSmiles=True,
            canonical=True,
        )
    return result


def get_canonical_map_order(smiles: str, root: int = -1) -> list[int] | None:
    mapped_molecule = Chem.MolFromSmiles(smiles)
    if mapped_molecule is None:
        return None
    canonical_smiles = clear_map_canonical_smiles(smiles, root=root)
    if canonical_smiles is None:
        return None
    canonical_molecule = Chem.MolFromSmiles(canonical_smiles)
    if canonical_molecule is None:
        return None
    canonical_to_mapped = mapped_molecule.GetSubstructMatch(canonical_molecule)
    correctly_mapped = [
        canonical_molecule.GetAtomWithIdx(index).GetSymbol()
        == mapped_molecule.GetAtomWithIdx(mapped_index).GetSymbol()
        for index, mapped_index in enumerate(canonical_to_mapped)
    ]
    atom_count = canonical_molecule.GetNumAtoms()
    if np.sum(correctly_mapped) < atom_count or len(canonical_to_mapped) < atom_count:
        canonical_to_mapped = [0] * atom_count
        mapped_to_canonical = canonical_molecule.GetSubstructMatch(mapped_molecule)
        if len(mapped_to_canonical) != atom_count:
            return None
        for index, mapped_index in enumerate(mapped_to_canonical):
            canonical_to_mapped[mapped_index] = index
    atom_maps = [atom.GetAtomMapNum() for atom in mapped_molecule.GetAtoms()]
    final_map = [atom_maps[canonical_to_mapped[index]] for index in range(atom_count)]
    if not final_map:
        return None
    max_atom_map = max(final_map)
    for index, atom_map in enumerate(final_map):
        if atom_map == 0:
            max_atom_map += 1
            final_map[index] = max_atom_map
    return final_map


def find_permutation(original_order: list[int], new_order: list[int]) -> list[int]:
    return [original_order.index(atom_map) for atom_map in new_order]


def sanity_check(
    product_smiles: str,
    reactant_smiles: str,
    max_atoms: int,
    max_atom_difference: int,
) -> tuple[Mol, Mol] | None:
    if not product_smiles or not reactant_smiles:
        return None
    product = Chem.MolFromSmiles(product_smiles)
    reactant = Chem.MolFromSmiles(reactant_smiles)
    if product is None or reactant is None:
        return None
    reactant_atoms = reactant.GetNumAtoms()
    product_atoms = product.GetNumAtoms()
    if reactant_atoms < 5 or product_atoms == 1:
        return None
    canonical_product = clear_map_canonical_smiles(product_smiles)
    canonical_reactant = clear_map_canonical_smiles(reactant_smiles)
    if canonical_product is None or canonical_reactant is None:
        return None
    if canonical_product == canonical_reactant:
        return None
    if not all(atom.HasProp("molAtomMapNumber") for atom in product.GetAtoms()):
        return None
    product_maps = [
        atom.GetAtomMapNum() for atom in product.GetAtoms() if atom.GetAtomMapNum() > 0
    ]
    if len(product_maps) != len(set(product_maps)) or len(product_maps) != product_atoms:
        return None
    reactant_maps = [
        atom.GetAtomMapNum() for atom in reactant.GetAtoms() if atom.GetAtomMapNum() > 0
    ]
    if len(reactant_maps) != len(set(reactant_maps)):
        return None
    reactant_by_map = {
        atom.GetAtomMapNum(): atom
        for atom in reactant.GetAtoms()
        if atom.GetAtomMapNum() > 0
    }
    product_by_map = {
        atom.GetAtomMapNum(): atom
        for atom in product.GetAtoms()
        if atom.GetAtomMapNum() > 0
    }
    for atom_map in reactant_by_map.keys() & product_by_map.keys():
        if reactant_by_map[atom_map].GetSymbol() != product_by_map[atom_map].GetSymbol():
            return None
    if product_by_map.keys() - reactant_by_map.keys():
        return None
    if reactant_atoms < product_atoms:
        return None
    if reactant_atoms > max_atoms or product_atoms > max_atoms:
        return None
    if reactant_atoms - product_atoms > max_atom_difference:
        return None
    if product.GetNumBonds() == 0 or reactant.GetNumBonds() == 0:
        return None
    return product, reactant


def _atom_properties(atom: Atom) -> tuple[int, int, bool, str, str | None]:
    chirality = (
        str(atom.GetChiralTag())
        if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
        else None
    )
    return (
        atom.GetFormalCharge(),
        atom.GetTotalNumHs(),
        atom.GetIsAromatic(),
        str(atom.GetHybridization()),
        chirality,
    )


def get_reaction_center(
    reaction_smiles: str,
) -> dict[str, list[list[int]] | bool] | None:
    try:
        reactants, _, products = reaction_smiles.split(">")
    except ValueError:
        return None
    reactant_molecules = [
        Chem.MolFromSmiles(part) for part in reactants.split(".") if part
    ]
    product_molecules = [
        Chem.MolFromSmiles(part) for part in products.split(".") if part
    ]
    if not reactant_molecules or not product_molecules:
        return None
    if any(molecule is None for molecule in reactant_molecules + product_molecules):
        return None
    product_map_to_index: dict[int, int] = {}
    product_offset = 0
    for molecule in product_molecules:
        if molecule is None:
            continue
        for atom in molecule.GetAtoms():
            atom_map = atom.GetAtomMapNum()
            if atom_map:
                product_map_to_index[atom_map] = product_offset + atom.GetIdx()
        product_offset += molecule.GetNumAtoms()
    product_atom_maps = set(product_map_to_index)
    reactant_bonds: dict[tuple[int, int], float] = {}
    for molecule in reactant_molecules:
        if molecule is None:
            continue
        for bond in molecule.GetBonds():
            atom_maps = (
                bond.GetBeginAtom().GetAtomMapNum(),
                bond.GetEndAtom().GetAtomMapNum(),
            )
            if all(atom_maps):
                reactant_bonds[tuple(sorted(atom_maps))] = bond.GetBondTypeAsDouble()
    product_bonds: dict[tuple[int, int], float] = {}
    for molecule in product_molecules:
        if molecule is None:
            continue
        for bond in molecule.GetBonds():
            atom_maps = (
                bond.GetBeginAtom().GetAtomMapNum(),
                bond.GetEndAtom().GetAtomMapNum(),
            )
            if all(atom_maps):
                product_bonds[tuple(sorted(atom_maps))] = bond.GetBondTypeAsDouble()
    formed_bonds = [
        [product_map_to_index[first], product_map_to_index[second]]
        for first, second in set(product_bonds) - set(reactant_bonds)
        if first in product_atom_maps and second in product_atom_maps
    ]
    broken_bonds: list[list[int]] = []
    for first, second in set(reactant_bonds) - set(product_bonds):
        indices = [
            product_map_to_index[atom_map]
            for atom_map in (first, second)
            if atom_map in product_map_to_index
        ]
        if indices:
            broken_bonds.append(indices)
    bond_order_changes = [
        [product_map_to_index[first], product_map_to_index[second]]
        for first, second in set(reactant_bonds) & set(product_bonds)
        if reactant_bonds[(first, second)] != product_bonds[(first, second)]
        and first in product_atom_maps
        and second in product_atom_maps
    ]
    reactant_properties: dict[int, tuple[int, int, bool, str, str | None]] = {}
    for molecule in reactant_molecules:
        if molecule is None:
            continue
        for atom in molecule.GetAtoms():
            atom_map = atom.GetAtomMapNum()
            if atom_map:
                reactant_properties[atom_map] = _atom_properties(atom)
    product_properties: dict[int, tuple[int, int, bool, str, str | None]] = {}
    for molecule in product_molecules:
        if molecule is None:
            continue
        for atom in molecule.GetAtoms():
            atom_map = atom.GetAtomMapNum()
            if atom_map:
                product_properties[atom_map] = _atom_properties(atom)
    charge_changes: list[list[int]] = []
    h_count_changes: list[list[int]] = []
    aromatic_changes: list[list[int]] = []
    hybridization_changes: list[list[int]] = []
    chirality_changes: list[list[int]] = []
    targets = (
        charge_changes,
        h_count_changes,
        aromatic_changes,
        hybridization_changes,
        chirality_changes,
    )
    for atom_map in product_atom_maps:
        index = product_map_to_index[atom_map]
        product_values = product_properties[atom_map]
        if atom_map not in reactant_properties:
            if product_values[4] is not None:
                chirality_changes.append([index])
            continue
        reactant_values = reactant_properties[atom_map]
        for position, target in enumerate(targets):
            if reactant_values[position] != product_values[position]:
                target.append([index])
    result: dict[str, list[list[int]] | bool] = {
        "formed_bonds": formed_bonds,
        "broken_bonds": broken_bonds,
        "bond_order_changes": bond_order_changes,
        "charge_changes": charge_changes,
        "h_count_changes": h_count_changes,
        "chirality_changes": chirality_changes,
        "aromatic_changes": aromatic_changes,
        "hybridization_changes": hybridization_changes,
    }
    result["has_changes"] = any(result.values())
    return result


def get_positive_center_atom_maps(
    reaction_smiles: str,
    include_symmetric: bool = True,
) -> list[tuple[int, ...]]:
    """Return every valid edge / node center as product atom-map tuples.

    Bond formation and bond-order changes are edge candidates. Broken bonds are
    represented by both surviving endpoint atoms because the product has no
    corresponding edge. Atom-property changes are node candidates. When
    requested, exact product-graph automorphisms are included as equivalent
    positives so they are not trained as negatives.
    """
    center = get_reaction_center(reaction_smiles)
    if center is None or not center["has_changes"]:
        return []
    try:
        _, _, product_smiles = reaction_smiles.split(">")
    except ValueError:
        return []
    product = Chem.MolFromSmiles(product_smiles)
    if product is None:
        return []
    atom_maps = [atom.GetAtomMapNum() for atom in product.GetAtoms()]

    positives: list[tuple[int, ...]] = []

    def add(indices: list[int]) -> None:
        if not indices or any(index >= len(atom_maps) for index in indices):
            return
        maps = tuple(atom_maps[index] for index in indices)
        if all(atom_map > 0 for atom_map in maps) and maps not in positives:
            positives.append(maps)

    for key in ("formed_bonds", "bond_order_changes"):
        candidates = center[key]
        if isinstance(candidates, list):
            for candidate in candidates:
                if len(candidate) == 2:
                    add(candidate)
    candidates = center["broken_bonds"]
    if isinstance(candidates, list):
        for candidate in candidates:
            for index in candidate:
                add([index])
    for key in (
        "charge_changes",
        "h_count_changes",
        "hybridization_changes",
        "aromatic_changes",
        "chirality_changes",
    ):
        candidates = center[key]
        if isinstance(candidates, list):
            for candidate in candidates:
                for index in candidate:
                    add([index])

    if not include_symmetric or not positives:
        return positives

    symmetry_molecule = Chem.Mol(product)
    for atom in symmetry_molecule.GetAtoms():
        if atom.HasProp("molAtomMapNumber"):
            atom.ClearProp("molAtomMapNumber")
    symmetry_ranks = list(Chem.CanonicalRankAtoms(symmetry_molecule, breakTies=False))
    map_to_index = {atom_map: index for index, atom_map in enumerate(atom_maps)}
    originals = list(positives)
    for positive in originals:
        indices = [map_to_index[atom_map] for atom_map in positive]
        if len(indices) == 1:
            target_rank = symmetry_ranks[indices[0]]
            for index, rank in enumerate(symmetry_ranks):
                if rank == target_rank:
                    add([index])
        elif len(indices) == 2:
            target_ranks = sorted(symmetry_ranks[index] for index in indices)
            for bond in product.GetBonds():
                bond_indices = [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
                bond_ranks = sorted(symmetry_ranks[index] for index in bond_indices)
                if bond_ranks == target_ranks:
                    add(bond_indices)
    return positives
