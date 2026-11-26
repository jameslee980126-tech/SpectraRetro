"""CPU Stage1 worker: a caller-supplied class hypothesis, not a reference label."""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(os.environ.get("SPECTRARETRO_MODEL_WORKSPACE", str(Path(__file__).resolve().parents[2]))).resolve()
os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(ROOT / "agentic_retrosynthesis/.torch_extensions_known"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code/known_class/stage1"))
from agentic_retrosynthesis.known_contract import WEIGHTS, checked_class
import torch
from rdkit import Chem
from torchdrug import data, models, tasks


def main():
    inp, out = map(Path, sys.argv[1:3])
    request = json.loads(inp.read_text(encoding="utf-8"))
    cls = checked_class(request["class_id"])
    torch.set_num_threads(2)
    path, expected = WEIGHTS["stage1"]
    assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected
    mol = Chem.MolFromSmiles(request["product"])
    for i, atom in enumerate(mol.GetAtoms()):
        atom.SetAtomMapNum(i + 1)
    graph = data.Molecule.from_molecule(mol, atom_feature="center_identification", bond_feature="default", kekulize=True)
    # Synthetic schema initialization only. No train/test reaction or reference precursor is read.
    schema = graph.clone()
    with schema.node():
        schema.node_label = torch.zeros(schema.num_node)
        schema.node_label[0] = 1
    with schema.edge():
        schema.edge_label = torch.zeros(schema.num_edge)
    payload = torch.load(ROOT / path, map_location="cpu")
    state = payload["model"]
    batch_norm = any("batch_norm" in key for key in state)
    options = dict(input_dim=graph.node_feature.shape[-1], hidden_dims=[512]*3,
                   batch_norm=batch_norm, short_cut=True, num_relation=int(graph.num_relation), concat_hidden=False)
    task = tasks.CenterIdentification(models.RGCN(**options), models.RGCN(**options),
        feature=("graph", "atom", "bond", "reaction"), retro_mtgr_edge_features=True,
        retro_mtgr_prior_gate="learned", aee_edge_features=True, aee_prior_gate="learned")
    task.preprocess([{"graph": (schema, schema), "reaction": i} for i in range(6)], [], [])
    task.load_state_dict(state, strict=True)
    task.eval().cpu()
    with torch.no_grad():
        probabilities = task.predict({"graph": (None, data.Molecule.pack([graph])),
                                      "reaction": torch.tensor([cls - 1])}).softmax(0).tolist()
    candidates = []
    for i, (a, b, _) in enumerate(graph.directed().edge_list.tolist()):
        bond = mol.GetBondBetweenAtoms(a, b)
        candidates.append(dict(kind="bond", atom_indices=[a, b], in_ring=bond.IsInRing(),
                               bond_type=str(bond.GetBondType()), probability=probabilities[i]))
    offset = len(candidates)
    for i in range(mol.GetNumAtoms()):
        candidates.append(dict(kind="atom", atom_indices=[i], probability=probabilities[offset+i]))
    candidates.sort(key=lambda c: -c["probability"])
    for rank, candidate in enumerate(candidates, 1):
        candidate["rank"] = rank
    out.write_text(json.dumps({"checkpoint_sha256": expected, "class_id": cls,
        "class_source": "caller_hypothesis", "nodes": [{"smiles": request["product"], "candidates": candidates}]}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
