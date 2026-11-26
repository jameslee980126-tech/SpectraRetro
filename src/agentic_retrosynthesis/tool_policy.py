"""Attach deterministic tool observations to model candidates for route decisions.

This adapter does not change the molecular predictor or its generation frequencies.
Observations are also available to the external search policy and in exported routes.
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

from .planner import canonical_smiles


@lru_cache(maxsize=8192)
def molecule_observation(smiles):
    normalized = canonical_smiles(smiles)
    mol = Chem.MolFromSmiles(normalized)
    return {"smiles": normalized, "molecular_weight": round(Descriptors.MolWt(mol), 3),
            "heavy_atoms": mol.GetNumHeavyAtoms(), "rings": rdMolDescriptors.CalcNumRings(mol),
            "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
            "formal_charge": Chem.GetFormalCharge(mol),
            "radical_electrons": sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms()),
            "ch_sites": sum(a.GetAtomicNum() == 6 and a.GetTotalNumHs() > 0 for a in mol.GetAtoms())}


class ToolEnrichedModel:
    """Run structural validation and stock lookup before a search expansion."""
    def __init__(self, model, stock):
        self.model, self.stock = model, stock
        self.observations = []

    def predict(self, product, top_k):
        raw = self.model.predict(product, top_k)
        candidates = []
        for candidate in raw:
            item = deepcopy(candidate)
            pieces = item.get("precursors", [])
            if isinstance(pieces, str):
                pieces = pieces.split(".")
            try:
                molecules = [molecule_observation(p) for s in pieces for p in s.split(".")]
            except (ValueError, TypeError, AttributeError):
                self.observations.append({"event": "tool_rejected", "product": product,
                                          "reason": "invalid_precursor_structure"})
                continue
            evidence = {"structure_status": "valid", "source": "RDKit and configured terminal inventory",
                        "precursors": [{**m, "in_terminal_inventory": m["smiles"] in self.stock}
                                       for m in molecules],
                        "unresolved_precursors": sum(m["smiles"] not in self.stock for m in molecules),
                        "forward_validation": "not_checked"}
            metadata = item.get("metadata")
            item["metadata"] = {**(metadata if isinstance(metadata, dict) else {}), "tool_evidence": evidence}
            candidates.append(item)
        self.observations.append({"event": "chemical_tool_screen", "product": product,
                                  "input_candidates": len(raw), "valid_candidates": len(candidates),
                                  "tools": ["structure_validation", "molecular_properties", "terminal_inventory"]})
        return candidates

