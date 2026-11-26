"""Read-only Stage1 centre ranking for the frozen external literature cases.

This is deliberately Stage1-only. The unknown-class prior arm is an interim
validation-selected checkpoint saved before the planned server continuation,
so outputs are labelled preliminary and must not be presented as final paired
model results.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from rdkit import Chem, rdBase


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from spectraretro.reaction_center import training as common
from spectraretro.reaction_center.features import ROOT, SCHEMA, attach_product_prior
from spectraretro.reaction_center.unknown_class.prior_model import build_task
from torchdrug import data

WORKSPACE = ROOT.parents[1]
MANIFEST = WORKSPACE / "output" / "literature_ch_activation_cases_20260920" / "case_manifest.json"
OUTPUT = WORKSPACE / "output" / "literature_ch_activation_cases_20260920" / "stage1_preliminary_results.json"


def build_product(case: dict) -> dict:
    mol = Chem.MolFromSmiles(case["product_smiles"])
    if mol is None:
        raise ValueError(case["product_smiles"])
    for idx, atom in enumerate(mol.GetAtoms()):
        atom.SetAtomMapNum(idx + 1)
    graph = data.Molecule.from_molecule(
        mol,
        atom_feature="center_identification",
        bond_feature="default",
        kekulize=True,
    )
    with graph.edge():
        graph.edge_label = torch.zeros(int(graph.num_edge), dtype=torch.long)
    with graph.node():
        graph.node_label = torch.zeros(int(graph.num_node), dtype=torch.long)

    a, b = case["literature_new_bond_atom_indices"]
    wanted_maps = tuple(sorted((a + 1, b + 1)))
    hits = []
    for edge_idx, (u, v, _) in enumerate(graph.edge_list.tolist()):
        maps = tuple(sorted((int(graph.atom_map[u]), int(graph.atom_map[v]))))
        if maps == wanted_maps:
            hits.append(edge_idx)
    if len(hits) != 2:
        raise RuntimeError(f"Expected two directed copies of centre bond {wanted_maps}; found {hits}")
    canonical_hit = next(idx for idx in hits if int(graph.edge_list[idx, 0]) < int(graph.edge_list[idx, 1]))
    graph.edge_label[canonical_hit] = 1
    reason = attach_product_prior(graph)
    if reason:
        raise RuntimeError(f"Chemical descriptors unavailable for {case['case_id']}: {reason}")
    return {"graph": (graph.clone(), graph), "reaction": 0, "sample id": 0}


def candidate_keys(graph) -> list[dict]:
    canonical = graph.directed()
    keys = []
    for u, v, _ in canonical.edge_list.tolist():
        a, b = int(graph.atom_map[u]), int(graph.atom_map[v])
        keys.append(
            {
                "type": "bond",
                "atom_maps": sorted((a, b)),
                "elements": [graph.to_molecule().GetAtomWithIdx(u).GetSymbol(), graph.to_molecule().GetAtomWithIdx(v).GetSymbol()],
            }
        )
    for idx, atom_map in enumerate(graph.atom_map.tolist()):
        keys.append(
            {
                "type": "atom",
                "atom_maps": [int(atom_map)],
                "elements": [graph.to_molecule().GetAtomWithIdx(idx).GetSymbol()],
            }
        )
    return keys


@torch.no_grad()
def score(task, row: dict) -> dict:
    task.eval()
    batch = next(iter(data.DataLoader([row], 1, shuffle=False)))
    logits = task.predict(batch).cpu()
    labels, sizes = task.target(batch)
    count = int(sizes[0])
    logits = logits[:count]
    labels = labels[:count].cpu()
    probabilities = logits.softmax(0)
    target = int(labels.argmax())
    order = sorted(range(count), key=lambda idx: (-float(logits[idx]), idx))
    rank = order.index(target) + 1
    keys = candidate_keys(row["graph"][1])
    if len(keys) != count:
        raise RuntimeError(f"Candidate key mismatch: {len(keys)} vs {count}")
    generator = torch.Generator().manual_seed(0)
    draws = torch.multinomial(probabilities, 300, replacement=True, generator=generator)
    top = []
    for candidate_idx in order[:10]:
        item = dict(keys[candidate_idx])
        item.update(
            candidate_index=candidate_idx,
            probability=float(probabilities[candidate_idx]),
            is_literature_center=candidate_idx == target,
        )
        top.append(item)
    return {
        "candidate_count": count,
        "literature_center_candidate_index": target,
        "literature_center_rank": rank,
        "literature_center_probability": float(probabilities[target]),
        "literature_center_draws_out_of_300": int((draws == target).sum()),
        "top10": top,
    }


def load_arm(arm: str, train, valid):
    common.seed_all(0)
    task = build_task(train, valid, prior=arm == "prior", width=512, num_layers=3)
    checkpoint_path = ROOT / "runs" / f"stage1_unknown_{arm}" / "best.pth"
    checkpoint = common.torch_load(checkpoint_path)
    task.load_state_dict(checkpoint["model"], strict=True)
    return task, checkpoint_path, checkpoint


def load_frozen_cached_dataset(csv_path: Path, sources: dict):
    """Reuse the training-time frozen cache without rebuilding or altering it."""
    identity = {
        "csv_sha256": common.sha(csv_path),
        "sources": sources,
        "schema": SCHEMA,
        "rdkit": rdBase.rdkitVersion,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "atom_feature": "center_identification",
        "bond_feature": "default",
        "with_hydrogen": False,
        "kekulize": True,
        "official_atom_vocab": True,
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path = ROOT / "runtime" / "stage1_cache" / f"{key}.pth"
    if not path.exists():
        raise FileNotFoundError(f"Frozen Stage1 cache not found: {path}")
    saved = common.torch_load(path)
    if saved["identity"] != identity:
        raise RuntimeError(f"Frozen cache identity mismatch: {path}")
    return saved["dataset"]


def main() -> None:
    checkpoint_contract = common.torch_load(
        ROOT / "runs" / "stage1_unknown_baseline" / "best.pth"
    )["contract"]
    sources = checkpoint_contract["sources"]
    train = load_frozen_cached_dataset(ROOT / "data" / "stage1" / "train.csv", sources)
    valid = load_frozen_cached_dataset(ROOT / "data" / "stage1" / "valid.csv", sources)
    baseline, baseline_path, baseline_checkpoint = load_arm("baseline", train, valid)
    prior, prior_path, prior_checkpoint = load_arm("prior", train, valid)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    cases = []
    for case in manifest["cases"]:
        row = build_product(case)
        cases.append(
            {
                "case_id": case["case_id"],
                "product_smiles": case["product_smiles"],
                "literature_new_bond_atom_indices": case["literature_new_bond_atom_indices"],
                "baseline": score(baseline, row),
                "prior": score(prior, row),
            }
        )

    result = {
        "status": "preliminary_stage1_only_nonfinal",
        "scope": "Product-only unknown-class Stage1 reaction-centre ranking; no Stage2 precursor generation was run.",
        "warning": "The prior checkpoint is the interim validation-selected epoch saved before planned server continuation. Do not report this as the final paired model result.",
        "sampling": "300 categorical draws from the product-only centre distribution, seed 0",
        "baseline_checkpoint": {
            "path": str(baseline_path.relative_to(WORKSPACE)),
            "epoch": baseline_checkpoint["epoch"],
            "best_epoch": baseline_checkpoint["best_epoch"],
            "best_validation_top1": baseline_checkpoint["best"],
            "sha256": common.sha(baseline_path),
        },
        "prior_checkpoint": {
            "path": str(prior_path.relative_to(WORKSPACE)),
            "epoch": prior_checkpoint["epoch"],
            "best_epoch": prior_checkpoint["best_epoch"],
            "best_validation_top1": prior_checkpoint["best"],
            "sha256": common.sha(prior_path),
        },
        "cases": cases,
    }
    common.save_json(OUTPUT, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
