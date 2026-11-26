"""Deterministic screening, not an experimental feasibility oracle."""
from __future__ import annotations

from dataclasses import dataclass, field
from rdkit import Chem
from .planner import canonical_smiles


@dataclass
class RouteConstraints:
    forbidden_smiles: list[str] = field(default_factory=list)
    unavailable_stock: list[str] = field(default_factory=list)
    require_forward_check: bool = False

    def __post_init__(self):
        self.forbidden_smiles = sorted({canonical_smiles(s) for s in self.forbidden_smiles})
        self.unavailable_stock = sorted({canonical_smiles(s) for s in self.unavailable_stock})

    def effective_stock(self, stock):
        return {canonical_smiles(s) for s in stock} - set(self.unavailable_stock) - set(self.forbidden_smiles)

    def check(self, product, candidate):
        reasons = []
        for smiles in [product] + list(candidate["precursors"]):
            if smiles in self.forbidden_smiles:
                reasons.append("forbidden_molecule:" + smiles)
            mol = Chem.MolFromSmiles(smiles)
            if mol is None or any(a.GetAtomicNum() == 0 or a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
                reasons.append("invalid_or_radical_molecule")
        # A verification result must be attached by a configured tool, not an LLM.
        status = candidate.get("metadata", {}).get("forward_check", {}).get("status", "not_checked")
        if status == "rejected" or (self.require_forward_check and status != "passed"):
            reasons.append("forward_check:" + status)
        return reasons
