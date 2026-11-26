"""Historical six-class contract. IDs are hypotheses, never dataset lookups."""
import os
from pathlib import Path

ROOT = Path(os.environ.get("SPECTRARETRO_MODEL_WORKSPACE", str(Path(__file__).resolve().parents[2]))).resolve()
CLASSES = {1: "C-H halogenation", 2: "C-H alkenylation/olefination", 3: "C-H alkylation",
           4: "C-H amination/amidation", 5: "C-H arylation", 6: "C-H acylation/carbonylation"}
WEIGHTS = {
    "stage1": ("code/known_class/stage1/model/ch_activation_6class_final_stage1_learnedgate_512x3_wd0_20260706.pth",
               "26ced66b6f554462f7f8836408fcb4918a49cd7ca3b1d27b0e9175194b9169ea"),
    "diffusion": ("code/known_class/stage2/checkpoints/ch_activation_6class_final_diffusion_20260706/last.ckpt",
                  "49af2d21aafcc323d04f384cf637ebbf03aee3030766589c4ffb3c424d29608e"),
    "size": ("code/known_class/stage2/models/ch_activation_6class_final_size_20260706/last.ckpt",
             "f409c0cf86cc56493949785bce2a805c0d9e8d8d89e0a5796e1e913abf72bdea"),
}


def checked_class(value):
    if type(value) is not int or value not in CLASSES:
        raise ValueError("Class hypothesis must be an integer in 1..6")
    return value


def molecule_context(smiles):
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid product")
    atoms = [{"id": a.GetIdx(), "element": a.GetSymbol(), "aromatic": a.GetIsAromatic(),
              "hydrogens": a.GetTotalNumHs()} for a in mol.GetAtoms()]
    bonds = [{"atoms": [b.GetBeginAtomIdx(), b.GetEndAtomIdx()], "type": str(b.GetBondType()),
              "in_ring": b.IsInRing()} for b in mol.GetBonds()]
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    return {"mapped_smiles": Chem.MolToSmiles(mol), "mapping": "atom map = zero-based id + 1",
            "atoms": atoms, "bonds": bonds}
