"""Audit whether the legacy drawing-derived planar dual is chemically stable.

This is a diagnostic, not a training script.  It compares the faces produced by
``Molecule.planar_face1`` before and after atom renumbering, contrasts those
faces with RDKit's chemical ring basis, and detects the closure-vertex bias in
the legacy face mean.
"""

import argparse
import random

from rdkit import Chem

from torchdrug import data


DEFAULT_SMILES = (
    "CCO",
    "c1ccccc1",
    "c1ccc2ccccc2c1",
    "C1CC2CCC1C2",
    "CC1=CC=CC=C1C(=O)N",
)


def torchdrug_molecule(molecule):
    return data.Molecule.from_molecule(
        molecule,
        atom_feature="center_identification",
        bond_feature="default",
        with_hydrogen=False,
        kekulize=True,
    ).directed()


def open_face(face):
    """Remove the two closure vertices appended by ``planar_face1``."""
    face = [int(atom) for atom in face]
    if len(face) >= 3 and face[-2:] == face[:2]:
        return face[:-2]
    raise RuntimeError(f"Malformed planar face closure: {face}")


def face_signature(faces, new_to_old=None):
    signature = []
    for face in faces:
        atoms = open_face(face)
        if new_to_old is not None:
            atoms = [new_to_old[atom] for atom in atoms]
        signature.append(tuple(sorted(atoms)))
    return tuple(sorted(signature))


def ring_signature(molecule, new_to_old=None):
    rings = []
    for ring in Chem.GetSymmSSSR(molecule):
        atoms = [int(atom) for atom in ring]
        if new_to_old is not None:
            atoms = [new_to_old[atom] for atom in atoms]
        rings.append(tuple(sorted(atoms)))
    return tuple(sorted(rings))


def packed_dual(graph):
    """Exercise the same PackedGraph boundary used by Stage-1."""
    return data.Molecule.pack([graph]).undirected().dual_graph()


def closure_bias_count(faces):
    count = 0
    for face in faces:
        closed = [int(atom) for atom in face]
        opened = open_face(face)
        # Atom indices are used as a deterministic scalar probe.  If the two
        # means differ, the duplicated closure atoms necessarily bias a generic
        # atom-feature mean as well.
        closed_mean = sum(closed) / len(closed)
        open_mean = sum(opened) / len(opened)
        count += abs(closed_mean - open_mean) > 1e-12
    return count


def audit_smiles(smiles, trials, seed):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    base = torchdrug_molecule(molecule)
    base_faces = base.planar_face1()
    base_signature = face_signature(base_faces)
    base_dual = packed_dual(base)

    rng = random.Random(seed)
    mismatched_faces = 0
    mismatched_shapes = 0
    mismatched_rings = 0
    for _ in range(trials):
        order = list(range(molecule.GetNumAtoms()))
        rng.shuffle(order)
        renumbered = Chem.RenumberAtoms(molecule, order)
        graph = torchdrug_molecule(renumbered)
        faces = graph.planar_face1()
        mapped_signature = face_signature(faces, new_to_old=order)
        mismatched_faces += mapped_signature != base_signature
        mismatched_rings += ring_signature(renumbered, new_to_old=order) != ring_signature(molecule)
        dual = packed_dual(graph)
        mismatched_shapes += (
            int(dual.num_node) != int(base_dual.num_node)
            or int(dual.num_edge) != int(base_dual.num_edge)
        )

    num_atom = molecule.GetNumAtoms()
    num_bond = molecule.GetNumBonds()
    cycle_rank = num_bond - num_atom + 1
    return {
        "smiles": smiles,
        "num_atom": num_atom,
        "num_bond": num_bond,
        "cycle_rank": cycle_rank,
        "planar_faces": len(base_faces),
        "chemical_rings": len(ring_signature(molecule)),
        "closure_biased_faces": closure_bias_count(base_faces),
        "renumber_face_mismatches": mismatched_faces,
        "renumber_shape_mismatches": mismatched_shapes,
        "renumber_ring_mismatches": mismatched_rings,
        "trials": trials,
        "planar_signature": base_signature,
        "ring_signature": ring_signature(molecule),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles", nargs="*", default=DEFAULT_SMILES)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.trials <= 0:
        raise ValueError("--trials must be positive")

    failed = False
    for smiles in args.smiles:
        result = audit_smiles(smiles, args.trials, args.seed)
        print(result)
        failed |= result["renumber_face_mismatches"] > 0
        failed |= result["renumber_shape_mismatches"] > 0
    if failed:
        raise SystemExit("FAIL: legacy planar dual is not atom-renumbering invariant")
    print("PASS: no atom-renumbering mismatch was found in this probe set")


if __name__ == "__main__":
    main()
