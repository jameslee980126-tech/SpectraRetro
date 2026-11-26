"""Budgeted center/size exploration. No reference inputs or model retraining."""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from .planner import Planner, canonical_smiles


@dataclass
class GenerationBudget:
    max_calls: int = 24
    max_paired_draws: int = 120
    used_calls: int = 0
    used_paired_draws: int = 0

    def __post_init__(self):
        if min(self.max_calls, self.max_paired_draws) < 1:
            raise ValueError("Generation limits must be positive")

    def reserve(self, requested):
        if self.used_calls >= self.max_calls:
            return 0
        count = min(requested, self.max_paired_draws - self.used_paired_draws)
        if count <= 0:
            return 0
        self.used_calls += 1
        self.used_paired_draws += count  # Failed slots are never refunded.
        return count

    def available(self):
        return self.used_calls < self.max_calls and self.used_paired_draws < self.max_paired_draws


def public_centers(raw):
    """Whitelist product-only fields before sending anything to an LLM."""
    result = []
    for center in raw:
        cid = center.get("center_id")
        allowed = center.get("allowed_sizes", [])
        fragments = center.get("fragments", [])
        if (type(cid) is not int or any(type(n) is not int or n < 0 or n > 64 for n in allowed)
                or not allowed or len(fragments) != 2):
            continue
        cleaned = []
        for frag in fragments:
            probs = {str(n): float(frag.get("size_probabilities", {}).get(str(n), 0)) for n in allowed}
            if any(not math.isfinite(p) or p < 0 or p > 1 for p in probs.values()):
                raise ValueError("Invalid size probabilities")
            cleaned.append({"atom_indices": frag.get("atom_indices", []),
                            "elements": frag.get("elements", []),
                            "anchor_atom": frag.get("anchor_atom"), "size_probabilities": probs})
        metadata = {key: center[key] for key in ("class_id", "class_label", "class_source", "reaction_class_input") if key in center}
        result.append({"center_id": cid, "bond_atoms": center["bond_atoms"],
                       "center_probability": center.get("center_probability"),
                       "allowed_sizes": sorted(set(allowed)), "fragments": cleaned, **metadata})
    return result


class ControlledModel:
    """backend.describe(product), backend.complete(product, center, sizes, draws).

    sizes=None requests learned-size sampling; [n0,n1] requests a hypothesis.
    All alternatives are exploratory, not official benchmark predictions.
    """
    def __init__(self, backend, policy="enumerate", advisor=None, budget=None,
                 draws_per_call=5, max_centers=3, alternatives=4):
        if policy not in ("baseline", "enumerate", "llm"):
            raise ValueError("policy must be baseline, enumerate or llm")
        if min(draws_per_call, max_centers, alternatives) < 1:
            raise ValueError("Positive generation budgets required")
        if policy == "llm" and advisor is None:
            raise ValueError("LLM policy requires a configured endpoint and model")
        self.backend, self.policy, self.advisor = backend, policy, advisor
        self.budget = budget or GenerationBudget()
        self.draws, self.max_centers, self.alternatives = draws_per_call, max_centers, alternatives
        self.context, self.candidates, self.attempted = {}, {}, set()
        self.trace = []

    def _run(self, product, proposal):
        center, sizes = proposal["center_id"], proposal.get("sizes")
        key = (product, center, None if sizes is None else tuple(sizes))
        if key in self.attempted:
            return []
        count = self.budget.reserve(self.draws)
        if not count:
            return []
        self.attempted.add(key)
        event = {"event": "generation", "product": product, "center_id": center,
                 "sizes": sizes, "size_source": proposal["source"], "paired_draws": count}
        if proposal.get("reason"):
            reason = str(proposal["reason"])[:400]
            secret = getattr(self.advisor, "api_key", None)
            event["hypothesis_note"] = reason.replace(secret, "[redacted]") if secret else reason
        try:
            raw = self.backend.complete(product, center, sizes, count)
            valid = Planner._validate_candidates(product, raw)
            for c in valid:
                c["metadata"] = dict(c["metadata"], size_source=proposal["source"],
                                     requested_sizes=sizes, center_id=center, reference_inputs=False)
            event.update(status="completed", valid_candidates=len(valid))
        except Exception as exc:
            valid = []
            event.update(status="failed", error_type=type(exc).__name__)
        self.trace.append(event)
        return valid

    def _alternatives(self, product, feedback):
        centers = self.context[product]
        proposals = []
        # LLM is restricted to class/search decisions. Sizes come only from the
        # learned distribution, including deterministic finite enumeration.
        # Round-robin over centers; even highly peaked distributions get alternate sizes.
        grids = []
        for c in centers:
            ps = [f["size_probabilities"] for f in c["fragments"]]
            pairs = sorted(itertools.product(c["allowed_sizes"], repeat=2),
                           key=lambda pair: (-ps[0][str(pair[0])] * ps[1][str(pair[1])], sum(pair), pair))
            grids.append([{"center_id": c["center_id"], "sizes": list(pair),
                           "source": "size_distribution_enumeration"} for pair in pairs])
        for row in itertools.zip_longest(*grids):
            proposals.extend(p for p in row if p is not None)
        selected, seen = [], set()
        for p in proposals:
            key = (product, p["center_id"], tuple(p["sizes"]))
            if key not in self.attempted and key not in seen:
                seen.add(key)
                selected.append(p)
            if len(selected) >= self.alternatives:
                break
        return selected

    def _merge(self, product, new):
        existing = self.candidates.setdefault(product, [])
        seen = {tuple(c["precursors"]) for c in existing}
        for c in new:
            if tuple(c["precursors"]) not in seen:
                existing.append(c)
                seen.add(tuple(c["precursors"]))
        return existing

    def predict(self, product, top_k):
        product = canonical_smiles(product)
        if product not in self.context:
            if not self.budget.available():
                self.trace.append({"event": "generation_budget_exhausted", "product": product})
                return []
            self.context[product] = public_centers(self.backend.describe(product))[:self.max_centers]
            generated = []
            for c in self.context[product]:
                generated += self._run(product, {"center_id": c["center_id"], "sizes": None,
                                                  "source": "learned_distribution"})
            self._merge(product, generated)
            if self.policy != "baseline":
                self.recover(product, top_k, [{"event": "initial_exploration",
                                             "valid_candidates": len(generated)}])
        # Interleave attempt families so a prolific first attempt cannot hide all alternatives.
        groups = {}
        for c in self.candidates.get(product, []):
            m = c["metadata"]
            key = (m["center_id"], None if m["requested_sizes"] is None else tuple(m["requested_sizes"]))
            groups.setdefault(key, []).append(c)
        return [c for row in itertools.zip_longest(*groups.values()) for c in row if c is not None][:top_k]

    def recover(self, product, top_k, feedback):
        if (not self.budget.available() or self.policy == "baseline"
                or product not in self.context or not self.context[product]):
            return []
        generated = []
        observations = feedback + [e for e in self.trace if e.get("product") == product][-8:]
        for p in self._alternatives(product, observations):
            generated += self._run(product, p)
        self._merge(product, generated)
        return generated[:top_k]
