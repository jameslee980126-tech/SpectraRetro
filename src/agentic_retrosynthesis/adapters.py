"""Model adapters; a deployed endpoint must accept only the product SMILES."""

from __future__ import annotations

import json
import csv
import hashlib
import re
from collections import Counter
from pathlib import Path
from urllib.request import Request, urlopen

from .planner import Planner, canonical_smiles


_UNSATURATED_ATOM = re.compile(r"\[(C|N|O|c|n|o|S|s|P)\]")


def fill_openbabel_hydrogens(smiles: str) -> str:
    """Planner-only heuristic for exact neutral bracket atoms lacking H counts."""
    repaired = _UNSATURATED_ATOM.sub(r"\1", smiles)
    return canonical_smiles(repaired)


class HTTPModel:
    def __init__(self, url: str, token: str | None = None, timeout: int = 120):
        self.url, self.token, self.timeout = url, token, timeout

    def predict(self, product: str, top_k: int) -> list[dict]:
        body = json.dumps({"product": product, "top_k": top_k}).encode()
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        with urlopen(Request(self.url, data=body, headers=headers, method="POST"),
                     timeout=self.timeout) as response:
            result = json.load(response)
        if not isinstance(result, dict) or not isinstance(result.get("candidates"), list):
            raise ValueError("Model endpoint must return {'candidates': [...]} ")
        return result["candidates"][:top_k]


class JSONModel:
    """Offline interface fixture; its predictions are not claimed as live model output."""

    def __init__(self, path: Path):
        self.path = path
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("Predictions file must map product SMILES to candidate lists")
        self.data = {canonical_smiles(product): candidates for product, candidates in raw.items()}

    def predict(self, product: str, top_k: int) -> list[dict]:
        return self.data.get(product, [])[:top_k]


class GDiffRetroSnapshotModel:
    """Look up 300-draw 3D diffusion results for products in a saved test run.

    This adapter reads candidates for the prepared test synthons. Optional Stage 1
    counts limit the number of draws per product under the evaluation protocol.
    """

    def __init__(self, merged_result: Path, stage1_counts: Path | None = None,
                 repair_openbabel: bool = False):
        self.path = merged_result.resolve()
        self.source_sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.repair_openbabel = repair_openbabel
        freeze = self.path.parent.parent / "evaluation_freeze_1500.json"
        self.weight_provenance = {}
        if freeze.is_file():
            record = json.loads(freeze.read_text(encoding="utf-8"))
            arms = record.get("arms", {})
            self.weight_provenance = {
                "edm_3d_best_epoch": arms.get("edm_3d", {}).get("best_epoch"),
                "edm_3d_best_sha256": arms.get("edm_3d", {}).get("best_sha256"),
                "size_best_epoch": arms.get("size", {}).get("best_epoch"),
                "size_best_sha256": arms.get("size", {}).get("best_sha256"),
            }
        self.counts = None
        self.stage1_sha256 = None
        if stage1_counts:
            import pickle
            self.stage1_sha256 = hashlib.sha256(stage1_counts.read_bytes()).hexdigest()
            self.counts = pickle.loads(stage1_counts.read_bytes())
        self.index: dict[str, list[tuple[str, list[str]]]] = {}
        with self.path.open(encoding="utf-8", newline="") as handle:
            for row in csv.reader(handle, delimiter="\t"):
                if len(row) != 303:
                    raise ValueError("Expected 826-row merged_result.csv with 300 draws, reference, product")
                product = row[302]
                normalized = canonical_smiles(product)
                draws = row[1:301]
                if self.counts is not None:
                    count = int(self.counts[product])
                    if not 0 <= count <= 300:
                        raise ValueError("Stage1 count must be between 0 and 300")
                    draws = draws[:count]
                self.index.setdefault(normalized, []).append((product, draws))

    @property
    def available_products(self) -> set[str]:
        return set(self.index)

    def gate_status(self, product: str) -> dict:
        """Return the preserved Stage1 hit-count gate for one product."""
        entries = self.index.get(canonical_smiles(product), [])
        if not entries:
            return {"available": False, "hits": 0, "samples": 300, "misses": 300}
        if self.counts is None:
            return {"available": False, "hits": 300, "samples": 300, "misses": 0}
        values = {int(self.counts[raw_product]) for raw_product, _ in entries}
        if len(values) != 1:
            raise ValueError("Inconsistent Stage1 counts for a normalized product")
        hits = values.pop()
        return {"available": True, "hits": hits, "samples": 300,
                "misses": 300 - hits, "passed": hits > 0,
                "policy": "use the recorded Stage 1 count as the per-product draw limit"}

    def predict(self, product: str, top_k: int) -> list[dict]:
        entries = self.index.get(canonical_smiles(product), [])
        if not entries:
            return []
        gate = self.gate_status(product)
        draws = [draw for _, group in entries for draw in group]
        frequencies = Counter(draws)
        # Counter preserves first draw order on equal frequencies, matching the
        # original process_row implementation's stable frequency sort.
        ranked = sorted(frequencies.items(), key=lambda pair: pair[1], reverse=True)
        candidates = []
        for smiles, frequency in ranked:
            if not smiles or smiles == "MISSING":
                continue
            raw_components = smiles.split(".")
            if self.repair_openbabel:
                try:
                    components = [fill_openbabel_hydrogens(s) for s in raw_components]
                except ValueError:
                    continue
            else:
                components = raw_components
            candidates.append({
                "precursors": components,
                "score": frequency / max(len(draws), 1),
                "source": ("spectraretro_stage1_gated_3d_snapshot" if self.counts is not None
                           else "spectraretro_stage2_3d_snapshot"),
                "metadata": {"draw_frequency": frequency, "draws": len(draws),
                             "conditioning": "ground_truth_center_reference_derived",
                             "stage1_gate": self.counts is not None,
                             "stage1_hits": gate["hits"],
                             "stage1_samples": gate["samples"],
                             "stage1_misses": gate["misses"],
                             "stage1_policy": gate.get("policy"),
                             "stage1_gate_sha256": self.stage1_sha256,
                             "raw_precursors": raw_components,
                             "planner_postprocess": "neutral_bracket_implicit_h_fill" if self.repair_openbabel else "none",
                             "frozen_weights": self.weight_provenance,
                             "source_file": str(self.path),
                             "source_sha256": self.source_sha256},
            })
            if len(candidates) >= top_k:
                break
        return candidates


class SpectraRetroSnapshotModel(GDiffRetroSnapshotModel):
    """Look up saved SpectraRetro candidates with optional Stage 1 evaluation counts."""


class CascadeModel:
    """Use preserved GDiffRetro draws first, then a clearly labeled fallback."""

    def __init__(self, primary, fallback):
        self.primary, self.fallback = primary, fallback

    def predict(self, product: str, top_k: int) -> list[dict]:
        primary = self.primary.predict(product, top_k)
        if Planner._validate_candidates(canonical_smiles(product), primary):
            return primary
        return self.fallback.predict(product, top_k)
