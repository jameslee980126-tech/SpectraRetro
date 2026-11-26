"""Search saved precursor candidates to build routes for the pharmaceutical cases.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import time

from agentic_retrosynthesis.planner import Planner, canonical_smiles
from agentic_retrosynthesis.constraints import RouteConstraints
from .route_diagram import chemistry_review


ROOT = Path(os.environ.get("SPECTRARETRO_MODEL_WORKSPACE", str(Path(__file__).resolve().parents[2]))).resolve()
BASE = ROOT / "experiments/case_partition_completion_20261013"
BANK = BASE / "fixed_core_completion_20261013/connected_routes_cross_draw"
PINNED = {
    "edm": "7d43cdf2b3f9b730198ae059745d23fbe41413ea2c6fdb17292ac7ad4c31ba56",
    "size": "67303500b8b73d2dae49f186d96bad15219a49d38179513de01f33f52ecfff17",
}
CASES = {
    "lapatinib": ("case_l", "Lapatinib", "lapatinib.json", "LAP-", ["拉帕替尼"]),
    "nemiralisib": ("case_n", "Nemiralisib / GSK2269557", "nemiralisib.json", "NEMI-", ["GSK2269557"]),
    "osimertinib_3r": ("case_o", "Osimertinib active derivative 3r", "osimertinib_3r.json", "OSI-", ["osimertinib", "奥希替尼", "3r"]),
    "tafamidis": ("case_t", "Tafamidis", "tafamidis.json", "TAF-", ["塔法米迪", "氯苯唑酸"]),
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class FixedCoreCaseService:
    def __init__(self):
        receipts = {Path(r["path"]).resolve(): r["sha256"]
                    for r in read(BANK / "artifact_sha256.json")}
        # The independent combination audit was committed after the bank manifest.
        receipts[(BANK / "combination_audit.json").resolve()] = "82e45788c382bed4a2ba6f167463eaf066752200a78a5bac65236e2056828bcf"
        for name in ("candidate_pools.json", "input_product_mapping.json", "protocol.json", "combination_audit.json"):
            path = BANK / name
            if sha(path) != receipts.get(path.resolve()):
                raise RuntimeError("Case candidate integrity check failed: " + name)
        self.audit = read(BANK / "combination_audit.json")
        required = ("all_candidate_molecules_from_actual_model_banks",
                    "all_combination_counts_and_scores_reconstructed",
                    "no_reference_filter_for_combinations", "stock_not_used_for_generation")
        if self.audit.get("state") != "complete" or not all(self.audit.get(k) for k in required):
            raise RuntimeError("Case candidate combination audit is incomplete")
        self.weights = read(BASE / "selected_model_manifest.json")
        for component, digest in PINNED.items():
            entry = self.weights[component]
            if entry["sha256"] != digest or sha(entry["path"]) != digest:
                raise RuntimeError("Latest case weight integrity check failed: " + component)
        for path, digest in self.audit["effective_generation_banks"].items():
            if sha(path) != digest:
                raise RuntimeError("Generated completion bank integrity check failed")
        self.pools = read(BANK / "candidate_pools.json")
        self.mapping = read(BANK / "input_product_mapping.json")
        self.requests = read(BASE / "whole_routes/requests.json")
        self.pool_sha = receipts[(BANK / "candidate_pools.json").resolve()]
        self._catalogue = []
        for case_id, (job, name, _, prefix, aliases) in CASES.items():
            rows = [r for r in self.mapping if r["step_id"].startswith(prefix)]
            self._catalogue.append({"case_id": case_id, "title": name,
                "product": canonical_smiles(self.requests[job]["target"]),
                "provided_step_templates": len(rows), "aliases": aliases,
                "mode": "fixed_core_latest_candidates", "weights_sha256": dict(PINNED),
                "conditioning": "intermediates and attachment sites recorded for each case step",
                "generation": "saved latest-weight draws; cross-draw fragment assembly",
                "case_training_contact": True,
                "scope": "Free-base Lapatinib; derivative 3r, not Osimertinib API; literature terminal boundaries"})

    def catalogue(self):
        return copy.deepcopy(self._catalogue)

    def find_case(self, value):
        if not isinstance(value, str):
            return None
        value = value.strip()
        for row in self._catalogue:
            if value.casefold() in {s.casefold() for s in [row["case_id"], row["title"], *row["aliases"]]}:
                return row["case_id"]
        # Names above are handled before SMILES parsing to avoid RDKit error noise.
        if not any(c in value for c in "()[]=123456789"):
            return None
        try:
            product = canonical_smiles(value)
        except ValueError:
            return None
        return next((r["case_id"] for r in self._catalogue if r["product"] == product), None)

    def predict(self, product, top_k):
        candidates = self.pools.get(canonical_smiles(product), {}).get("candidates", [])
        output = []
        for rank, candidate in enumerate(candidates[:top_k], 1):
            metadata = candidate["metadata"]
            support = [{"input_template": s["input_template"], "fragments": [
                {"uuid": f["uuid"], "generated_smiles": f["smiles"],
                 "matching_draw_count": len(f["draws"])} for f in s["fragments"]]}
                for s in metadata["support"]]
            output.append({"precursors": list(candidate["precursors"]), "score": candidate["score"],
                "source": "latest_case_fixed_core_generated_bank", "metadata": {
                    "given_known_core_condition": True, "case_training_contact": True,
                    "inventory_completion_used": False, "reference_unknown_structure_input": False,
                    "cross_draw_assembly": True, "new_sampling": False,
                    "candidate_bank_rank": rank, "candidate_bank_sha256": self.pool_sha,
                    "weights_sha256": dict(PINNED),
                    "empirical_fragment_frequency_product": metadata["empirical_fragment_frequency_product"],
                    "combination_slots": self.pools[canonical_smiles(product)]["paired_draw_slots"],
                    "score_definition": "product of fragment frequencies; not calibrated confidence",
                    "support": support,
                    "effective_generation_banks": self.audit["effective_generation_banks"]}})
        return output

    def _annotate(self, case_id, result):
        # Full literature precursors are read AFTER search for annotation only.
        refs = read(ROOT / "output/four_case_stepwise_neighbors_20261009/routes" / CASES[case_id][2])
        edges = {}
        for ref in refs["steps"]:
            key = (canonical_smiles(ref["product"]["smiles"]),
                   tuple(sorted(canonical_smiles(r["smiles"]) for r in ref["reactants"])))
            edges[key] = {"step_id": ref["step_id"], "reaction_family": ref.get("reaction_family"),
                          "source_url": ref.get("source_url"), "source_locator": ref.get("source_locator")}
        for route in result["routes"]:
            for step in route["steps"]:
                ref = edges.get((step["product"], tuple(sorted(step["precursors"]))))
                step["metadata"]["literature_match"] = ref
                step["metadata"]["chemistry_review"] = chemistry_review(step)
            route["all_edges_documented"] = all(s["metadata"]["literature_match"] for s in route["steps"])
            route["experimental_feasibility_verified"] = False
        result["documented_route_count"] = sum(r["all_edges_documented"] for r in result["routes"])

    def plan(self, request):
        case_id = self.find_case(request.get("case_id") or request.get("product", ""))
        if case_id is None:
            raise ValueError("Choose one of the four supported pharmaceutical cases")
        started = time.monotonic()
        job, name, _, _, _ = CASES[case_id]
        source = self.requests[job]
        budgets = {"max_depth": (12, 12), "top_k": (10000, 10000),
                   "max_routes": (3, 30), "max_model_calls": (300, 300), "max_states": (10000, 10000)}
        settings = {}
        for key, (default, ceiling) in budgets.items():
            value = request.get(key, default)
            if type(value) is not int or not 1 <= value <= ceiling:
                raise ValueError(key + " is outside the case search budget")
            settings[key] = value
        raw_constraints = request.get("constraints", {})
        if not isinstance(raw_constraints, dict) or set(raw_constraints) - {"forbidden_smiles", "unavailable_stock"}:
            raise ValueError("Unsupported case route constraints")
        for value in raw_constraints.values():
            if not isinstance(value, list) or len(value) > 100 or not all(isinstance(s, str) for s in value):
                raise ValueError("Molecule constraints must be lists of at most 100 SMILES")
        constraints = RouteConstraints(**raw_constraints) if raw_constraints else None
        result = Planner(self, set(source["stock"]), None, max_depth=settings["max_depth"],
            max_model_calls=settings["max_model_calls"], expansion_top_k=settings["top_k"],
            max_routes=settings["max_routes"], max_states=settings["max_states"],
            cost_policy="rank", constraints=constraints).plan(source["target"]).to_dict()
        self._annotate(case_id, result)
        result.update({"mode": "fixed_core_latest_candidates", "case_id": case_id, "case_name": name,
            "seconds": round(time.monotonic() - started, 3), "search_settings": settings,
            "one_step_candidates": self.predict(source["target"], min(settings["top_k"], 300)),
            "model_calls_definition": "candidate-bank queries, not GPU inference calls",
            "llm_policy": "model_order", "llm_calls": 0, "new_sampling": False, "new_training": False,
            "sampling_steps": 100, "base_draws_per_fragment": 30, "blocker_draws_per_fragment": 300,
            "provided_step_templates": next(r["provided_step_templates"] for r in self._catalogue if r["case_id"] == case_id),
            "weights_sha256": dict(PINNED), "candidate_bank_sha256": self.pool_sha,
            "stock_source": source["stock_source"], "terminal_set": "Case-specific literature starting-material boundary",
            "scope": "The case records define the intermediates and attachment sites used for generation. The case models had training exposure to these examples. Results describe a computational route search.",
            "generation": "Reused latest-weight generated banks; complete precursor sets assembled across draws",
            "reference_precursors_used_for_search": False, "stock_used_for_candidate_generation": False,
            "constraints": raw_constraints})
        preferred_steps = 9 if case_id == "nemiralisib" else None
        result["primary_route_index"] = next((i for i, route in enumerate(result["routes"])
            if route["all_edges_documented"] and (preferred_steps is None or len(route["steps"]) == preferred_steps)),
            next((i for i, route in enumerate(result["routes"]) if route["all_edges_documented"]), 0))
        return result
