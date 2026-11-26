"""Isolated model-first planning over complete generated candidate pools.

This module does not change the existing industrial/snapshot entry points.
Generation uses the unknown-class backend with learned sizes only. Pagination
changes search priority, never the stored fusion ranking or generation budget.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

from .constraints import RouteConstraints
from .controlled import ControlledModel, GenerationBudget
from .planner import Planner, canonical_smiles


SCHEMA = "gdiffretro_generated_pool_v1"
POOL_ORDER = "existing_baseline_attempt_interleave; per_attempt_fusion_metadata_unchanged"
LIMITS = {
    "max_steps": (4, 1, 12), "max_products": (5, 1, 30),
    "max_generation_calls": (12, 1, 100), "max_paired_draws": (60, 1, 600),
    "draws_per_call": (5, 1, 300), "sampling_steps": (100, 1, 100),
    "max_centers": (3, 1, 10), "max_routes": (3, 1, 20),
    "max_states": (1000, 1, 10000), "page_size": (20, 1, 100),
    "max_llm_calls": (12, 1, 100),
}


def _digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


class PoolSourceError(ValueError):
    """An input is not an unknown-class, product-only model observation."""


def _check_conditioning(record):
    if (record.get("reaction_class_input") or record.get("reference_inputs")
            or record.get("class_id") is not None or record.get("class_label") is not None
            or record.get("class_source") not in (None, "none")):
        raise PoolSourceError("Only unknown-class product-only generated pools are supported")
    if any(record.get(key) is not None for key in
           ("reference_precursors", "true_size", "true_center", "reference_coordinates")):
        raise PoolSourceError("Reference-derived fields are not permitted")


def _check_sources(candidates):
    for candidate in candidates:
        metadata = candidate.get("metadata", {})
        _check_conditioning(candidate)
        _check_conditioning(metadata)
        if str(candidate.get("source", "")).startswith("llm_"):
            raise PoolSourceError("LLM-generated molecules are not permitted in this mode")


class _SourceCheckedBackend:
    """Check original provenance before ControlledModel normalizes metadata."""

    def __init__(self, backend):
        self.backend = backend

    def describe(self, product):
        centers = self.backend.describe(product)
        for center in centers:
            _check_conditioning(center)
            for fragment in center.get("fragments", []):
                _check_conditioning(fragment)
        return centers

    def complete(self, product, center_id, sizes, draws):
        candidates = self.backend.complete(product, center_id, sizes, draws)
        _check_sources(candidates)
        return candidates


def _pool_record(product, candidates, status):
    _check_sources(candidates)
    payload = {"product": product, "candidates": copy.deepcopy(candidates)}
    return dict(payload, pool_sha256=_digest(payload), status=status,
                candidate_count=len(candidates), pool_order=POOL_ORDER)


def _bank(pools):
    return {"schema": SCHEMA, "model_mode": "unknown_class", "reference_inputs": False,
            "pool_order": POOL_ORDER, "candidate_pools": copy.deepcopy(pools),
            "pool_bank_sha256": _digest({p: record["pool_sha256"] for p, record in sorted(pools.items())})}


class GeneratedPoolModel:
    """Generate once per visited molecule; all later reads use an immutable pool.

    The existing ControlledModel baseline interleaves generation attempts. The
    maximum number of unique pairs cannot exceed the total paired-draw budget,
    so requesting that bound avoids the legacy planner's early top-k cut.
    No recover method is exposed: fetching more candidates never samples again.
    """

    def __init__(self, backend, budget, draws_per_call=5, max_centers=3):
        self.budget = budget
        self.controlled = ControlledModel(_SourceCheckedBackend(backend), policy="baseline", budget=budget,
                                          draws_per_call=draws_per_call, max_centers=max_centers)
        self.pools = {}
        self.visited_products = []
        self.uncovered_products = []

    def predict(self, product, top_k):
        # top_k remains in the existing planner protocol; it is not a pool limit.
        product = canonical_smiles(product)
        if product not in self.visited_products:
            self.visited_products.append(product)
        if product not in self.pools:
            available = self.budget.available()
            candidates = self.controlled.predict(product, self.budget.max_paired_draws)
            status = "generated" if candidates else ("empty_generated_pool" if available else "generation_budget_exhausted")
            self.pools[product] = _pool_record(product, candidates, status)
        return copy.deepcopy(self.pools[product]["candidates"])

    def export_bank(self):
        record = _bank(self.pools)
        record.update(generation_budget=asdict(self.budget), generation_trace=copy.deepcopy(self.controlled.trace))
        return record


class ReplayPoolModel:
    """Read a fixed product-only bank without generating uncovered molecules."""

    def __init__(self, path):
        self.path = Path(path).resolve()
        raw = self.path.read_bytes()
        self.source_sha256 = hashlib.sha256(raw).hexdigest()
        self.bank = json.loads(raw)
        if (self.bank.get("schema") != SCHEMA or self.bank.get("model_mode") != "unknown_class"
                or self.bank.get("reference_inputs") is not False):
            raise ValueError("Unsupported pool snapshot or conditioning")
        self.pools = self.bank.get("candidate_pools")
        if not isinstance(self.pools, dict):
            raise ValueError("Pool snapshot must contain candidate_pools")
        for product, record in self.pools.items():
            if canonical_smiles(product) != product or record.get("product") != product:
                raise ValueError("Noncanonical or mismatched pool product")
            candidates = record.get("candidates")
            if not isinstance(candidates, list):
                raise ValueError("Pool candidates must be a list")
            _check_sources(candidates)
            payload = {"product": product, "candidates": candidates}
            if _digest(payload) != record.get("pool_sha256"):
                raise ValueError("Candidate pool hash mismatch")
            if record.get("candidate_count") != len(candidates):
                raise ValueError("Candidate count mismatch")
        if _bank(self.pools)["pool_bank_sha256"] != self.bank.get("pool_bank_sha256"):
            raise ValueError("Pool bank hash mismatch")
        self.visited_products = []
        self.uncovered_products = []

    def predict(self, product, top_k):
        product = canonical_smiles(product)
        if product not in self.visited_products:
            self.visited_products.append(product)
        if product not in self.pools:
            if product not in self.uncovered_products:
                self.uncovered_products.append(product)
            return []
        return copy.deepcopy(self.pools[product]["candidates"])

    def export_bank(self):
        # Keep the complete input bank, including products not visited by this arm.
        return copy.deepcopy(self.bank)


class PagedSearchRanker:
    """Page-local LLM ordering with deterministic round-robin search priority.

    Every candidate is retained, even when LLM calls fail or their budget runs
    out. These preferences never replace the stored model/fusion rank.
    """

    def __init__(self, advisor, page_size=20, max_calls=12):
        if advisor is not None and not callable(getattr(advisor, "rank", None)):
            raise ValueError("A supplied advisor must support ranking")
        if min(page_size, max_calls) < 1:
            raise ValueError("Positive page and LLM budgets required")
        self.advisor, self.page_size, self.max_calls = advisor, page_size, max_calls
        self.request_attempts = 0
        self.ranking_audit, self.frontier_audit = [], []
        self._rank_cache = {}

    def _call(self, method, *args):
        if self.advisor is None:
            return None, "not_called_static_page", None
        if self.request_attempts >= self.max_calls:
            return None, "not_called_llm_budget", None
        self.request_attempts += 1
        try:
            return getattr(self.advisor, method)(*args), "returned", None
        except Exception as exc:
            return None, "call_failed", type(exc).__name__

    def rank(self, product, candidates, stock):
        key = _digest({"product": product, "candidates": candidates, "stock": sorted(stock)})
        if key in self._rank_cache:
            return list(self._rank_cache[key])
        pages, page_audit = [], []
        for start in range(0, len(candidates), self.page_size):
            batch = candidates[start:start + self.page_size]
            default = list(range(len(batch)))
            if self.advisor is None:
                proposal, status, error = default, "not_called_static_page", None
            elif len(batch) == 1:
                proposal, status, error = default, "single_candidate_no_call", None
            else:
                proposal, status, error = self._call("rank", product, copy.deepcopy(batch), set(stock))
            valid = (isinstance(proposal, list) and all(type(i) is int for i in proposal)
                     and sorted(proposal) == default)
            if status == "returned":
                status = "accepted" if valid else "invalid_order_fallback"
            local = proposal if valid else default
            global_ids = [start + i for i in local]
            pages.append(global_ids)
            page_audit.append({"page": len(pages) - 1, "candidate_ids": list(range(start, start + len(batch))),
                               "search_order_ids": global_ids, "status": status, "error_type": error})
        order = [page[i] for i in range(max(map(len, pages), default=0)) for page in pages if i < len(page)]
        assert sorted(order) == list(range(len(candidates)))
        self._rank_cache[key] = list(order)
        self.ranking_audit.append({"product": product, "input_sha256": key, "candidate_count": len(candidates),
                                   "page_size": self.page_size, "pages": page_audit, "search_order_ids": order,
                                   "page_order_source": "model_order" if self.advisor is None else "llm_with_local_fallback",
                                   "merge_rule": "page_local_orders_round_robin_not_global_llm_ranking"})
        return order

    def choose_frontier(self, frontier, steps):
        selected, status, error = self._call("choose_frontier", frontier, steps)
        valid = type(selected) is int and 0 <= selected < len(frontier)
        if status == "returned":
            status = "accepted" if valid else "invalid_frontier_fallback"
        index = selected if valid else 0
        self.frontier_audit.append({"frontier": list(frontier), "selected": index,
                                    "status": status, "error_type": error})
        return index


def validate_request(request):
    allowed = {"target", "stock", "stock_source", "constraints", "strategy", "budgets", "case_name", "model_mode"}
    if not isinstance(request, dict) or set(request) - allowed:
        raise ValueError("Unsupported request fields; no reference inputs or generation fallbacks")
    if request.get("model_mode", "unknown_class") != "unknown_class":
        raise ValueError("This entry point supports unknown_class only")
    strategy = request.get("strategy", "model_order")
    if strategy not in ("model_order", "paged_model_order", "llm_pages"):
        raise ValueError("strategy must be model_order, paged_model_order or llm_pages")
    target = canonical_smiles(request.get("target", ""))
    from rdkit import Chem
    mol = Chem.MolFromSmiles(target)
    if len(Chem.GetMolFrags(mol)) != 1 or mol.GetNumHeavyAtoms() > 150:
        raise ValueError("Provide one connected target with at most 150 heavy atoms")
    stock = request.get("stock")
    if not isinstance(stock, list) or len(stock) > 10000 or not all(isinstance(s, str) for s in stock):
        raise ValueError("Provide an explicit stock SMILES list (maximum 10000)")
    stock = sorted({canonical_smiles(s) for s in stock})
    if not isinstance(request.get("stock_source"), str) or not request["stock_source"].strip():
        raise ValueError("State the source of stock, including whether it is simulated")
    raw = request.get("constraints", {})
    if not isinstance(raw, dict) or set(raw) - {"forbidden_smiles", "unavailable_stock", "require_forward_check"}:
        raise ValueError("Unsupported route constraints")
    for name in ("forbidden_smiles", "unavailable_stock"):
        if not isinstance(raw.get(name, []), list) or not all(isinstance(s, str) for s in raw.get(name, [])):
            raise ValueError("Constraint molecules must be SMILES lists")
    if type(raw.get("require_forward_check", False)) is not bool:
        raise ValueError("require_forward_check must be boolean")
    constraints = RouteConstraints(**raw)
    supplied = request.get("budgets", {})
    if not isinstance(supplied, dict) or set(supplied) - set(LIMITS):
        raise ValueError("Unsupported budgets; page_size is not a permanent top-k cutoff")
    budgets = {}
    for name, (default, lo, hi) in LIMITS.items():
        value = supplied.get(name, default)
        if type(value) is not int or not lo <= value <= hi:
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
        budgets[name] = value
    return target, stock, constraints, strategy, budgets


def run(request, output, *, backend=None, advisor=None, snapshot=None):
    target, stock, constraints, strategy, b = validate_request(request)
    if snapshot is not None and backend is not None:
        raise ValueError("Snapshot replay cannot use a generation backend")
    if strategy == "llm_pages" and advisor is None:
        from .industrial import load_llm
        advisor = load_llm(b["max_llm_calls"])
        if advisor is None:
            raise ValueError("Configure an LLM for llm_pages, or select model_order")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if snapshot is not None:
        model = ReplayPoolModel(snapshot)
    else:
        if backend is None:
            from .product_backend import ProductOnlyBackend
            backend = ProductOnlyBackend(output / "model_evidence", b["sampling_steps"], 10)
        model = GeneratedPoolModel(backend, GenerationBudget(b["max_generation_calls"], b["max_paired_draws"]),
                                   b["draws_per_call"], b["max_centers"])
    ranker = (PagedSearchRanker(advisor if strategy == "llm_pages" else None, b["page_size"], b["max_llm_calls"])
              if strategy in ("paged_model_order", "llm_pages") else None)
    _write(output / "request.json", request)
    started = time.monotonic()
    try:
        result = Planner(model, set(stock), ranker, max_depth=b["max_steps"], max_model_calls=b["max_products"],
                         expansion_top_k=b["page_size"], max_routes=b["max_routes"], max_states=b["max_states"],
                         constraints=constraints, cost_policy="rank", recovery_rounds=0).plan(target).to_dict()
    finally:
        bank = model.export_bank()
        _write(output / "candidate_pools.json", bank)
    result.update(mode="fixed_generated_pool_replay" if snapshot is not None else "model_first_generated_pool",
                  strategy=strategy, case_name=request.get("case_name", ""), model_mode="unknown_class",
                  reaction_class_input=False, reference_inputs=False, llm_generation_fallback=False,
                  llm_size_prediction=False, size_policy="learned_sampling_only_no_enumeration",
                  stock_source=request["stock_source"], constraints=asdict(constraints), budgets=b,
                  pool_bank_sha256=bank["pool_bank_sha256"], candidate_pools_file=str((output / "candidate_pools.json").resolve()),
                  candidate_pool_counts={p: v["candidate_count"] for p, v in bank["candidate_pools"].items()},
                  visited_pool_products=model.visited_products, uncovered_products=model.uncovered_products,
                  uncovered_product_query_count=len(model.uncovered_products),
                  planner_product_queries=result["model_calls"],
                  model_calls=0 if snapshot is not None else model.budget.used_calls,
                  model_calls_definition="backend.complete invocation attempts, including failures; zero during snapshot replay",
                  generation_budget_used={"used_calls": 0, "used_paired_draws": 0} if snapshot is not None else asdict(model.budget),
                  generation_trace=[] if snapshot is not None else model.controlled.trace,
                  ranking_audit=ranker.ranking_audit if ranker else [], frontier_audit=ranker.frontier_audit if ranker else [],
                  llm_request_attempts=ranker.request_attempts if ranker else 0,
                  llm_model=getattr(advisor, "model", None) if strategy == "llm_pages" else None,
                  llm_usage=copy.deepcopy(getattr(advisor, "usage", {})) if strategy == "llm_pages" else {},
                  seconds=round(time.monotonic() - started, 3), forward_validation="not_configured",
                  industrial_validation=False, model_ranking_preserved=True,
                  search_order_definition=({"llm_pages": "page-local LLM orders then round-robin",
                                            "paged_model_order": "unchanged local page orders then identical round-robin"}
                                           .get(strategy, POOL_ORDER)),
                  model_score_definition="frozen uncalibrated ranking utility; not reaction feasibility probability",
                  cost_definition="step count plus search preference; not probability, feasibility or money",
                  scope=("Fixed available candidate-graph replay; missing nodes remain uncovered; not end-to-end accuracy"
                         if snapshot is not None else "Product-only generated candidate search; stock closure is not experimental validation"))
    if snapshot is not None:
        result.update(snapshot_source=str(model.path), snapshot_source_sha256=model.source_sha256)
    _write(output / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--snapshot", type=Path, help="Read-only fixed generated pool replay; never runs generation")
    parser.add_argument("--check", action="store_true", help="Validate request/snapshot only; no GPU, LLM or credential loading")
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8-sig"))
    target, stock, _, strategy, budgets = validate_request(request)
    if args.check:
        bank = ReplayPoolModel(args.snapshot).export_bank() if args.snapshot is not None else None
        print(json.dumps({"request_valid": True, "target": target, "stock_size": len(stock), "strategy": strategy,
                          "budgets": budgets, "snapshot_validated": bank is not None,
                          "pool_bank_sha256": bank["pool_bank_sha256"] if bank else None,
                          "network_called": False, "gpu_loaded": False}, ensure_ascii=False))
        return
    if args.output is None:
        parser.error("--output must name a new directory")
    result = run(request, args.output, snapshot=args.snapshot)
    print(json.dumps({"solved": result["solved"], "routes": len(result["routes"]),
                      "pool_bank_sha256": result["pool_bank_sha256"], "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
