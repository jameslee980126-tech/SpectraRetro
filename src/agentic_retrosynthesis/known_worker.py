"""Isolated historical known-class runtime; no dataset/reference loading."""
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(os.environ.get("SPECTRARETRO_MODEL_WORKSPACE", str(Path(__file__).resolve().parents[2]))).resolve()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code/known_class/stage2"))
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from src import const
from src.datasets import collate, collate_with_fragment_edges
from src.lightning import DDPM
from src.linker_size_lightning import SizeClassifier
from agentic_retrosynthesis.known_contract import CLASSES, WEIGHTS, checked_class
from agentic_retrosynthesis.product_backend import ProductOnlyBackend, PROBE

DECODER = Path(os.environ.get("SPECTRARETRO_DECODER_PYTHON", sys.executable))
S1 = Path(os.environ.get("SPECTRARETRO_STAGE1_PYTHON", sys.executable))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def stable_seed(value):
    return int(hashlib.sha256(value.encode()).hexdigest()[:8], 16) % 2147483647


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def embed_product(smiles):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed, params.numThreads = stable_seed(smiles), 1
    if AllChem.EmbedMolecule(mol, params) != 0:
        raise ValueError("Product embedding failed")
    status = (AllChem.MMFFOptimizeMolecule(mol, maxIters=500) if AllChem.MMFFHasAllMoleculeParams(mol)
              else AllChem.UFFOptimizeMolecule(mol, maxIters=500))
    return Chem.RemoveHs(mol), status


def fragments(mol, candidate):
    if candidate["kind"] != "bond":
        return None, "atom_center_not_supported"
    if candidate["in_ring"]:
        return None, "ring_opening_not_supported"
    if candidate["bond_type"] != "SINGLE":
        return None, "non_single_bond_not_supported"
    a, b = candidate["atom_indices"]
    rw = Chem.RWMol(mol)
    rw.RemoveBond(a, b)
    groups = Chem.GetMolFrags(rw.GetMol(), asMols=False)
    if len(groups) != 2:
        return None, "not_two_components"
    if min(map(len, groups)) < 2:
        return None, "single_atom_fragment_not_supported"
    xyz, records, info = mol.GetConformer().GetPositions(), [], []
    for j, group in enumerate(groups):
        group = list(group)
        n = len(group)
        elements = [mol.GetAtomWithIdx(i).GetSymbol() for i in group]
        if any(e not in const.ATOM2IDX for e in elements):
            return None, "element_not_in_vocabulary"
        anchor = group.index(a if a in group else b)
        anchors = torch.zeros(n, device="cuda")
        anchors[anchor] = 1
        records.append(dict(uuid=str(j), name="product_only_known_hypothesis", num_atoms=n,
            positions=torch.tensor(xyz[group], dtype=torch.float32, device="cuda"),
            one_hot=torch.tensor(np.eye(26, dtype=np.float32)[[const.ATOM2IDX[e] for e in elements]], device="cuda"),
            charges=torch.tensor([const.CHARGES[e] for e in elements], dtype=torch.float32, device="cuda"),
            anchors=anchors, fragment_mask=torch.ones(n, device="cuda"), linker_mask=torch.zeros(n, device="cuda")))
        info.append(dict(product_atom_indices=group, elements=elements, anchor_local_index=anchor))
    return (records, info), None


class KnownRuntime(ProductOnlyBackend):
    candidate_source = "known_class_product_only_edm"

    def __init__(self, output, class_id, **kwargs):
        self.class_id = checked_class(class_id)
        super().__init__(output, **kwargs)

    def conditioning_metadata(self):
        return dict(reaction_class_input=True, class_id=self.class_id,
                    class_label=CLASSES[self.class_id], class_source="caller_hypothesis")

    def condition_records(self, records):
        for rec in records:
            condition = torch.zeros((rec["num_atoms"], 6), device=rec["one_hot"].device)
            condition[:, self.class_id - 1] = 1
            rec["one_hot"] = torch.cat((rec["one_hot"], condition), dim=-1)
        return records

    def score_product(self, product, folder):
        write(folder / "stage1_inputs.json", dict(product=product, class_id=self.class_id))
        with (folder / "stage1.log").open("w", encoding="utf-8") as log:
            subprocess.run([str(S1), str(Path(__file__).resolve().parent / "score_known_product.py"),
                str(folder / "stage1_inputs.json"), str(folder / "stage1_scores.json")],
                check=True, timeout=180, stdout=log, stderr=log, cwd=ROOT)
        return json.loads((folder / "stage1_scores.json").read_text(encoding="utf-8"))

    def _load(self):
        if self.loaded:
            return
        if not torch.cuda.is_available():
            raise RuntimeError("Known-class runtime requires CUDA")
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.weights = {}
        for name, (relative, expected) in WEIGHTS.items():
            if sha(ROOT / relative) != expected:
                raise RuntimeError("Historical checkpoint hash mismatch: " + name)
            self.weights[name] = expected
        for name, constructor, attr in (("diffusion", DDPM, "model"), ("size", SizeClassifier, "size_model")):
            ckpt = torch.load(ROOT / WEIGHTS[name][0], map_location="cpu", weights_only=False)
            params = dict(ckpt["hyper_parameters"], torch_device="cuda")
            if params["in_node_nf"] != 32:
                raise ValueError("Expected 26 element + 6 class features")
            model = constructor(**params).cuda().eval()
            model.load_state_dict(ckpt["state_dict"], strict=True)
            setattr(self, attr, model)
        if self.model.bond_aware or self.model.diffusion_process != "edm":
            raise ValueError("Only historical non-bond-aware EDM is permitted")
        self.model.edm.T = self.steps
        self.helper_hashes = {str(p.relative_to(ROOT)): sha(p) for p in (
            Path(__file__), Path(__file__).resolve().parent / "score_known_product.py",
            ROOT / "agentic_retrosynthesis/product_backend.py", PROBE / "decode_candidates.py",
            *sorted((ROOT / "code/known_class/stage2/src").glob("*.py")))}
        self.r, self.loaded = sys.modules[__name__], True


def main():
    inp, out = map(Path, sys.argv[1:3])
    request = json.loads(inp.read_text(encoding="utf-8"))
    backend = KnownRuntime(out.parent / "evidence", request["class_id"],
        steps=request.get("steps", 100), max_centers=request.get("max_centers", 10))
    if request["action"] == "describe":
        value = backend.describe(request["product"])
    elif request["action"] == "complete":
        value = backend.complete(request["product"], request["center_id"], request.get("sizes"), request["draws"])
    else:
        raise ValueError("Unknown worker action")
    write(out, value)


if __name__ == "__main__":
    main()
