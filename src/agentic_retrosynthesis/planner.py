"""Bounded AND-OR route search around a product-only one-step predictor.

An entire precursor set is one AND node: every precursor must be in stock or
expanded before the route is considered solved.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import heapq
import math
from typing import Any, Protocol


def canonical_smiles(smiles: str) -> str:
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise RuntimeError("RDKit is required to validate molecules; install rdkit first") from exc
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or not mol.GetNumAtoms():
        raise ValueError(f"Invalid molecule SMILES: {smiles!r}")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


class OneStepModel(Protocol):
    def predict(self, product: str, top_k: int) -> list[dict[str, Any]]: ...


class Ranker(Protocol):
    def rank(self, product: str, candidates: list[dict[str, Any]], stock: set[str]) -> list[int]: ...
    def choose_frontier(self, frontier: tuple[str, ...], steps: tuple[Step, ...]) -> int: ...


@dataclass
class Step:
    product: str
    precursors: list[str]
    model_score: float | None
    model_rank: int
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PlanningResult:
    target: str
    solved: bool
    routes: list[dict[str, Any]]
    model_calls: int
    expanded_products: list[str]
    candidate_hit_products: list[str]
    empty_candidate_products: list[str]
    stock_size: int
    errors: list[str]
    trace: list[dict[str, Any]] = field(default_factory=list)
    termination: str = "search_exhausted"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(order=True)
class _State:
    priority: float
    serial: int
    open_molecules: tuple[str, ...] = field(compare=False)
    steps: tuple[Step, ...] = field(compare=False)
    score: float = field(compare=False)


class Planner:
    def __init__(self, model: OneStepModel, stock: set[str], ranker: Ranker | None = None,
                 max_depth: int = 4, max_model_calls: int = 50, expansion_top_k: int = 10,
                 max_routes: int = 5, max_states: int = 2000, *, constraints=None,
                 cost_policy: str = "model_score", recovery_rounds: int = 0):
        if min(max_depth, max_model_calls, expansion_top_k, max_routes, max_states) < 1:
            raise ValueError("All search budgets must be positive")
        self.model = model
        self.stock = ({canonical_smiles(s) for s in stock} if constraints is None
                      else constraints.effective_stock(stock))
        if cost_policy not in ("model_score", "rank") or not 0 <= recovery_rounds <= 5:
            raise ValueError("Invalid cost policy or recovery budget")
        self.constraints, self.cost_policy = constraints, cost_policy
        self.recovery_rounds = recovery_rounds
        self.ranker = ranker
        self.max_depth = max_depth
        self.max_model_calls = max_model_calls
        self.expansion_top_k = expansion_top_k
        self.max_routes = max_routes
        self.max_states = max_states

    def plan(self, target: str) -> PlanningResult:
        target = canonical_smiles(target)
        if self.constraints and target in self.constraints.forbidden_smiles:
            raise ValueError("Target is forbidden by route constraints")
        if target in self.stock:
            return PlanningResult(target, True, [{"steps": [], "score": 0.0,
                "starting_materials": [target]}], 0, [], [], [], len(self.stock), [],
                termination="target_in_stock")
        heap: list[_State] = [_State(0.0, 0, (target,), (), 0.0)]
        cache: dict[str, list[dict[str, Any]]] = {}
        errors: list[str] = []
        routes: list[dict[str, Any]] = []
        serial = 0
        trace = []
        recovery = 0
        visited: set[tuple[tuple[str, ...], tuple[tuple[str, tuple[str, ...]], ...]]] = set()
        while len(routes) < self.max_routes:
            if not heap:
                if routes or recovery >= self.recovery_rounds or serial >= self.max_states or not hasattr(self.model, "recover"):
                    break
                recovery += 1
                changed = False
                for product in list(cache):
                    try:
                        extra = self.model.recover(product, self.expansion_top_k,
                            [{"event": "route_search_exhausted", "round": recovery,
                              "unresolved_products": sorted(p for p, c in cache.items() if not c)}])
                        combined = self._validate_candidates(product, cache[product] + extra)
                        if len(combined) > len(cache[product]):
                            changed = True
                        cache[product] = combined
                    except Exception as exc:
                        errors.append(f"Recovery failed ({type(exc).__name__})")
                trace.append({"event": "recovery", "round": recovery, "new_candidates": changed})
                if not changed:
                    continue
                visited.clear()
                heapq.heappush(heap, _State(0.0, serial, (target,), (), 0.0))
            state = heapq.heappop(heap)
            if not state.open_molecules:
                leaves = sorted({p for step in state.steps for p in step.precursors if p in self.stock})
                routes.append({"steps": [asdict(step) for step in state.steps],
                               "score": round(state.score, 6), "starting_materials": leaves})
                continue
            # Drain terminal states even after the generated-state budget is reached.
            if serial >= self.max_states:
                continue
            if len(state.steps) >= self.max_depth:
                continue
            frontier_index = 0
            if self.ranker and len(state.open_molecules) > 1:
                try:
                    chosen = self.ranker.choose_frontier(state.open_molecules, state.steps)
                    if type(chosen) is int and 0 <= chosen < len(state.open_molecules):
                        frontier_index = chosen
                    else:
                        errors.append("LLM frontier choice was invalid; first open molecule used")
                except Exception as exc:
                    errors.append(f"LLM frontier choice failed ({type(exc).__name__}); first open molecule used")
            product = state.open_molecules[frontier_index]
            if product not in cache:
                if len(cache) >= self.max_model_calls:
                    continue
                try:
                    raw = self.model.predict(product, self.expansion_top_k)
                    cache[product] = self._validate_candidates(product, raw)
                except Exception as exc:
                    errors.append(f"{product}: predictor failed ({type(exc).__name__})")
                    cache[product] = []
                trace.append({"event": "expand", "product": product, "candidates": len(cache[product])})
            candidates = cache[product]
            order = list(range(len(candidates)))
            if self.ranker and len(order) > 1:
                try:
                    proposal = self.ranker.rank(product, candidates, self.stock)
                    if (isinstance(proposal, list) and all(type(i) is int for i in proposal)
                            and sorted(proposal) == order):
                        order = proposal
                    else:
                        errors.append(f"{product}: LLM ranking was invalid; model order used")
                except Exception as exc:
                    errors.append(f"{product}: LLM ranking failed ({type(exc).__name__}); model order used")
            for preference, index in enumerate(order):
                candidate = candidates[index]
                precursors = candidate["precursors"]
                if self.constraints:
                    reasons = self.constraints.check(product, candidate)
                    if reasons:
                        trace.append({"event": "constraint_rejected", "product": product,
                                      "precursors": list(precursors), "reasons": reasons})
                        continue
                if self._creates_cycle(state.steps, product, precursors):
                    continue
                expanded = {step.product for step in state.steps}
                remaining = tuple(p for p in precursors if p not in self.stock and p not in expanded)
                other_open = state.open_molecules[:frontier_index] + state.open_molecules[frontier_index + 1:]
                new_open = tuple(dict.fromkeys(other_open + remaining))
                step = Step(product, list(precursors), candidate["score"], candidate["rank"],
                            candidate["source"], candidate["metadata"])
                new_steps = state.steps + (step,)
                key = (new_open, tuple((s.product, tuple(s.precursors)) for s in new_steps))
                if key in visited:
                    continue
                visited.add(key)
                serial += 1
                score = state.score + self._step_cost(
                    candidate["score"] if self.cost_policy == "model_score" else None, preference)
                heapq.heappush(heap, _State(score + 0.1 * len(new_open), serial,
                                            new_open, new_steps, score))
                if serial >= self.max_states:
                    break
        return PlanningResult(target, bool(routes), routes, len(cache), sorted(cache),
                              sorted(p for p, candidates in cache.items() if candidates),
                              sorted(p for p, candidates in cache.items() if not candidates),
                              len(self.stock), errors, trace,
                              "routes_found" if routes else ("state_budget" if serial >= self.max_states
                              else "call_budget_or_search_exhausted"))

    @staticmethod
    def _step_cost(score: float | None, preference: int) -> float:
        return 1.0 + (0.0 if score is None else -math.log(max(score, 1e-6))) + preference

    @staticmethod
    def _creates_cycle(steps: tuple[Step, ...], product: str, precursors: tuple[str, ...]) -> bool:
        children = {step.product: step.precursors for step in steps}
        children[product] = precursors
        for precursor in precursors:
            stack = [precursor]
            seen = set()
            while stack:
                node = stack.pop()
                if node == product:
                    return True
                if node not in seen:
                    seen.add(node)
                    stack.extend(children.get(node, []))
        return False

    @staticmethod
    def _validate_candidates(product: str, raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not isinstance(raw, list):
            raise ValueError("Model response must be a list of candidates")
        valid = []
        seen = set()
        for rank, item in enumerate(raw, 1):
            if not isinstance(item, dict):
                continue
            precursors = item.get("precursors")
            if isinstance(precursors, tuple):
                precursors = list(precursors)
            if isinstance(precursors, str):
                precursors = precursors.split(".")
            if not isinstance(precursors, list) or not precursors:
                continue
            if not all(isinstance(s, str) and s for s in precursors):
                continue
            try:
                molecules = tuple(sorted(canonical_smiles(component) for s in precursors
                                         for component in s.split(".")))
            except (ValueError, TypeError):
                continue
            if product in molecules or molecules in seen:
                continue
            score = item.get("score")
            if score is not None and (not isinstance(score, (int, float)) or
                                      not math.isfinite(score) or not 0 <= score <= 1):
                continue
            seen.add(molecules)
            valid.append({"precursors": molecules, "score": score, "rank": rank,
                          "source": str(item.get("source", "one_step_model")),
                          "metadata": item.get("metadata", {}) if isinstance(item.get("metadata", {}), dict) else {}})
        return valid
