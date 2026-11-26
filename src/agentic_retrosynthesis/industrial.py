"""Run the constrained product-only prototype; no training or reference lookup."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

from .constraints import RouteConstraints
from .controlled import ControlledModel, GenerationBudget
from .llm import OpenAICompatibleRanker
from .planner import Planner, canonical_smiles


def load_llm(max_calls=12):
    # Reuse the LLM connection configured for CHRetroAgent. Keys are environment-only for new settings.
    from webapp.llm_config import configured_llm
    ranker, _ = configured_llm()
    if ranker:
        ranker.max_calls = max_calls
    return ranker


def validate_request(request):
    allowed = {"target", "stock", "stock_source", "constraints", "policy", "budgets", "case_name", "model_mode"}
    if not isinstance(request, dict) or set(request) - allowed:
        raise ValueError("Unsupported request fields; reference structures/counts are not accepted")
    target = canonical_smiles(request.get("target", ""))
    from rdkit import Chem
    target_mol = Chem.MolFromSmiles(target)
    if len(Chem.GetMolFrags(target_mol)) != 1 or target_mol.GetNumHeavyAtoms() > 150:
        raise ValueError("This prototype supports one connected target with at most 150 heavy atoms")
    stock = request.get("stock")
    if not isinstance(stock, list) or len(stock) > 10000 or not all(isinstance(s, str) for s in stock):
        raise ValueError("Provide an explicit stock SMILES list (maximum 10000)")
    stock = sorted({canonical_smiles(s) for s in stock})
    if not isinstance(request.get("stock_source"), str) or not request["stock_source"].strip():
        raise ValueError("State stock_source, including whether inventory is simulated")
    constraints = request.get("constraints", {})
    if not isinstance(constraints, dict) or set(constraints) - {"forbidden_smiles", "unavailable_stock", "require_forward_check"}:
        raise ValueError("Only explicit molecule/stock constraints are currently supported")
    for name in ("forbidden_smiles", "unavailable_stock"):
        if not isinstance(constraints.get(name, []), list):
            raise ValueError("Constraint molecules must be lists")
    if type(constraints.get("require_forward_check", False)) is not bool:
        raise ValueError("require_forward_check must be a boolean")
    constraints = RouteConstraints(**constraints)
    policy = request.get("policy", "enumerate")
    if policy not in ("baseline", "enumerate", "llm"):
        raise ValueError("Unknown exploration policy")
    mode = request.get("model_mode", "unknown_class")
    if mode not in ("unknown_class", "known_class_agent"):
        raise ValueError("Unknown model_mode")
    if mode == "known_class_agent" and policy != "llm":
        raise ValueError("known_class_agent requires policy=llm for class hypotheses")
    limits = {"max_steps": (4, 1, 12), "max_products": (5, 1, 30),
              "max_generation_calls": (12, 1, 100), "max_paired_draws": (60, 1, 600),
              "draws_per_call": (5, 1, 30), "sampling_steps": (100, 1, 100),
              "max_centers": (3, 1, 10), "alternatives": (3, 1, 10),
              "top_k": (20, 1, 100), "max_routes": (3, 1, 20),
              "max_states": (1000, 1, 10000), "recovery_rounds": (1, 0, 3),
              "max_llm_calls": (12, 1, 100), "max_class_hypotheses": (2, 1, 3)}
    supplied = request.get("budgets", {})
    if not isinstance(supplied, dict) or set(supplied) - set(limits):
        raise ValueError("Unsupported budget field")
    budgets = {}
    for key, (default, lo, hi) in limits.items():
        n = supplied.get(key, default)
        if type(n) is not int or not lo <= n <= hi:
            raise ValueError(f"{key} must be an integer in {lo}..{hi}")
        budgets[key] = n
    return target, stock, constraints, policy, budgets


def run(request, output, backend=None, advisor=None):
    target, stock, constraints, policy, b = validate_request(request)
    if policy == "llm" and advisor is None:
        advisor = load_llm(b["max_llm_calls"])
        if advisor is None:
            raise ValueError("Configure LLM_URL, LLM_MODEL and LLM_API_KEY first; or use enumerate")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if backend is None:
        if request.get("model_mode") == "known_class_agent":
            from .known_backend import AgentKnownClassBackend
            backend = AgentKnownClassBackend(output / "model_evidence", advisor, b["sampling_steps"],
                                             10, b["max_class_hypotheses"])
        else:
            from .product_backend import ProductOnlyBackend
            backend = ProductOnlyBackend(output / "model_evidence", b["sampling_steps"], 10)
    budget = GenerationBudget(b["max_generation_calls"], b["max_paired_draws"])
    model = ControlledModel(backend, policy, advisor, budget, b["draws_per_call"],
                            b["max_centers"], b["alternatives"])
    (output / "request.json").write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
    started = time.monotonic()
    result = Planner(model, set(stock), advisor if policy == "llm" else None,
                     max_depth=b["max_steps"], max_model_calls=b["max_products"],
                     expansion_top_k=b["top_k"], max_routes=b["max_routes"], max_states=b["max_states"],
                     constraints=constraints, cost_policy="rank", recovery_rounds=b["recovery_rounds"]).plan(target).to_dict()
    result.update(mode="constrained_product_only_prototype", policy=policy,
                  size_policy="learned_sampling_then_model_distribution_enumeration" if policy != "baseline" else "learned_sampling",
                  llm_size_prediction=False,
                  case_name=request.get("case_name", ""), stock_source=request["stock_source"],
                  constraints=asdict(constraints), budgets=b, generation_budget=asdict(budget),
                  generation_trace=model.trace, seconds=round(time.monotonic() - started, 3),
                  llm_calls=advisor.calls if advisor else 0, llm_usage=advisor.usage if advisor else {},
                  llm_model=advisor.model if advisor else None,
                  llm_policy="openai_compatible" if policy == "llm" else "none",
                  terminal_set="explicit_user_inventory", reference_inputs=False,
                  model_mode=request.get("model_mode", "unknown_class"),
                  reaction_class_input=request.get("model_mode") == "known_class_agent",
                  class_source="agent_hypothesis" if request.get("model_mode") == "known_class_agent" else "none",
                  class_selection_trace=getattr(backend, "trace", []),
                  one_step_candidates=model.candidates.get(target, []),
                  scope="Stock closure under explicit graph constraints, not proven synthesis feasibility",
                  forward_validation="not_configured", industrial_validation=False,
                  cost_definition="step_count_plus_candidate_order; not probability or monetary cost")
    (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true", help="Validate only; no network, GPU or credentials output")
    parser.add_argument("--check-llm", action="store_true", help="Explicitly send one tiny connectivity request")
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8-sig"))
    target, stock, constraints, policy, budgets = validate_request(request)
    if args.check or args.check_llm:
        llm = load_llm(budgets["max_llm_calls"])
        status = {"request_valid": True, "target": target, "stock_size": len(stock),
                  "policy": policy, "llm_configured": llm is not None,
                  "key_present": bool(llm and llm.api_key), "network_called": False}
        if args.check_llm:
            if llm is None:
                parser.error("Configure LLM_URL and LLM_MODEL first")
            llm.choose_frontier(("CC", "O"), ())
            status.update(network_called=True, structured_response_valid=True)
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return
    if args.output is None:
        parser.error("--output must be a new directory")
    result = run(request, args.output)
    print(json.dumps({"solved": result["solved"], "routes": len(result["routes"]),
                      "output": str(args.output.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
