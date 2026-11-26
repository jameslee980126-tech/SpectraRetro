"""Convert generated XYZ to a molecular graph with RDKit bond perception."""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import rdDetermineBonds


ELEMENTS = "C N O S B Br Cl P I F Mg Si Sn Zn Cu Se H Li Na Cr Mn Ge Ru Ag Te Pt".split()


def decode_one(item: dict) -> dict:
    coords = item["coords"]
    kinds = item["elements"]
    if not coords or len(coords) != len(kinds):
        return {"smiles": "MISSING", "bond_perception": "failed"}
    xyz = str(len(kinds)) + "\n\n" + "".join(
        f"{ELEMENTS[int(kind)]} {point[0]:.9f} {point[1]:.9f} {point[2]:.9f}\n"
        for point, kind in zip(coords, kinds))
    molecule = Chem.MolFromXYZBlock(xyz)
    if molecule is None:
        return {"smiles": "MISSING", "bond_perception": "failed"}
    mode = "rdkit_bond_orders"
    try:
        rdDetermineBonds.DetermineBonds(molecule, charge=0, allowChargedFragments=False)
    except Exception:
        mode = "rdkit_distance_connectivity_only"
        try:
            molecule = Chem.MolFromXYZBlock(xyz)
            rdDetermineBonds.DetermineConnectivity(molecule)
        except Exception:
            return {"smiles": "MISSING", "bond_perception": "failed"}
    try:
        known_atoms = int(item.get("known_atoms", 0))
        known_bonds = item.get("known_bonds", [])
        if known_atoms:
            editable = Chem.RWMol(molecule)
            for bond in list(editable.GetBonds()):
                a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                if a < known_atoms and b < known_atoms:
                    editable.RemoveBond(a, b)
            mapping = {1.0: Chem.BondType.SINGLE, 2.0: Chem.BondType.DOUBLE,
                       3.0: Chem.BondType.TRIPLE, 1.5: Chem.BondType.AROMATIC}
            for a, b, order in known_bonds:
                editable.AddBond(int(a), int(b), mapping[float(order)])
            molecule = editable.GetMol()
            mode += "+known_fragment_bonds"
        try:
            Chem.SanitizeMol(molecule)
        except Exception:
            pass
        return {"smiles": Chem.MolToSmiles(molecule, isomericSmiles=True),
                "bond_perception": mode}
    except Exception:
        return {"smiles": "MISSING", "bond_perception": "failed"}


def decode_batch(items: list[dict]) -> list[dict]:
    return [decode_one(item) for item in items]
