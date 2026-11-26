"""Attach source-audited observed-graph features without changing base tensors."""
import ast
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from rdkit import Chem

import bootstrap  # noqa: F401
from src import const
from src.datasets import ZincDataset, collate, collate_with_fragment_edges
from prior_features import FEATURE_DIM, extract_synthon_features


def parse_anchors(value):
    text = str(value).strip()
    if text.startswith(("[", "(")):
        return [int(x) for x in ast.literal_eval(text)]
    return [int(float(text))]


def align_synthon_graph(fragments_smiles, coordinate_molecule):
    """Use the table's synthon chemistry, with a bijective identity alignment."""
    graph = Chem.MolFromSmiles(fragments_smiles, sanitize=False)
    if graph is None:
        raise ValueError("Cannot read observed synthon SMILES")
    source_ids = [atom.GetAtomMapNum() for atom in graph.GetAtoms()]
    target_ids = [atom.GetAtomMapNum() for atom in coordinate_molecule.GetAtoms()]
    if (0 in source_ids or 0 in target_ids or len(set(source_ids)) != len(source_ids)
            or len(set(target_ids)) != len(target_ids) or set(source_ids) != set(target_ids)):
        raise ValueError("Observed synthon and coordinate SDF need a unique, complete atom-map bijection")
    lookup = {identity: i for i, identity in enumerate(source_ids)}
    aligned = Chem.RenumberAtoms(graph, [lookup[identity] for identity in target_ids])
    if [a.GetAtomicNum() for a in aligned.GetAtoms()] != [a.GetAtomicNum() for a in coordinate_molecule.GetAtoms()]:
        raise ValueError("Atom-map alignment changed element identity")
    return aligned


class PriorDataset(Dataset):
    def __init__(self, data_path, prefix, device="cpu"):
        self.base = ZincDataset(data_path, prefix, device)
        with open(Path(data_path) / (prefix + "_table.csv"), encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        supplier = Chem.SDMolSupplier(str(Path(data_path) / (prefix + "_frag.sdf")), sanitize=False, removeHs=False)
        if len(rows) != len(self.base) or len(supplier) != len(rows):
            raise ValueError("Fragment/table/cache row counts disagree")
        self.priors = []
        self.unavailable = []
        audit_path = Path(data_path) / (prefix + "_condition_alignment.json")
        audit = {}
        if audit_path.exists():
            records = json.loads(audit_path.read_text(encoding="utf-8"))
            audit = {str(entry["uuid"]): entry for entry in records}
            if len(audit) != len(rows) or set(audit) != {r["uuid"] for r in rows}:
                raise ValueError("Condition alignment audit does not match table UUIDs")
        for index, row in enumerate(rows):
            item, mol = self.base[index], supplier[index]
            if str(item["uuid"]) != row["uuid"]:
                raise ValueError("Fragment table UUID order differs from cached dataset")
            mask = item["fragment_mask"].detach().cpu().numpy().astype(bool)
            n_frag = int(mask.sum())
            if mol is None or mol.GetNumAtoms() != n_frag or not np.array_equal(mask, np.arange(len(mask)) < n_frag):
                raise ValueError("Known atoms must form the ordered fragment prefix")
            expected = [const.ATOM2IDX[a.GetSymbol()] for a in mol.GetAtoms()]
            actual = item["one_hot"][:n_frag].argmax(-1).detach().cpu().tolist()
            if expected != actual:
                raise ValueError("Fragment atom identity order differs from cache")
            positions = item["positions"][:n_frag].detach().cpu().numpy()
            if not np.allclose(mol.GetConformer().GetPositions(), positions, atol=1e-4, rtol=0):
                raise ValueError("Fragment atom coordinate order differs from cache")
            anchors = parse_anchors(row["anchors"])
            cache_anchors = item["anchors"][:n_frag].nonzero().flatten().detach().cpu().tolist()
            if sorted(set(anchors)) != sorted(cache_anchors):
                raise ValueError("Fragment anchor indices differ from cache")
            prior = torch.zeros(len(mask), FEATURE_DIM, dtype=item["positions"].dtype, device=device)
            audit_entry = audit.get(row["uuid"])
            if audit_entry and audit_entry["condition_smiles_sha256"] != hashlib.sha256(row["fragments"].encode()).hexdigest():
                raise ValueError("Condition alignment audit belongs to different synthon chemistry")
            try:
                synthon = align_synthon_graph(row["fragments"], mol)
            except ValueError as error:
                if not audit_entry or audit_entry["available"]:
                    raise
                self.unavailable.append({"uuid": row["uuid"], "reason": audit_entry["reason"], "error": str(error)})
            else:
                if audit_entry and not audit_entry["available"]:
                    raise ValueError("Current condition unexpectedly disagrees with its unavailability audit")
                prior[:n_frag] = torch.as_tensor(extract_synthon_features(synthon, anchors), device=device)
            self.priors.append(prior)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        return dict(self.base[index], chemistry_prior=self.priors[index])


def prior_collate(batch, fragment_edges=False):
    base_items = [{k: v for k, v in item.items() if k != "chemistry_prior"} for item in batch]
    result = (collate_with_fragment_edges if fragment_edges else collate)(base_items)
    result["chemistry_prior"] = torch.nn.utils.rnn.pad_sequence(
        [item["chemistry_prior"] for item in batch], batch_first=True, padding_value=0)
    return result


def prior_collate_fragment_edges(batch):
    return prior_collate(batch, fragment_edges=True)
