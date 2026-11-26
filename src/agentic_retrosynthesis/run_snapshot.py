"""Plan routes using the saved 3D diffusion candidates for the test products."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .adapters import CascadeModel, GDiffRetroSnapshotModel, HTTPModel
from .llm import OpenAICompatibleRanker, OpenAICompatibleStepFallback
from .planner import Planner, canonical_smiles


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan routes using the saved 3D diffusion candidates")
    parser.add_argument("target", help="Product SMILES; must occur in the preserved test snapshot")
    parser.add_argument("--stock", type=Path, help="Optional one-SMILES-per-line stock file")
    parser.add_argument("--llm-url", help="OpenAI-compatible chat completions URL")
    parser.add_argument("--llm-model")
    parser.add_argument("--llm-step-fallback", action="store_true")
    parser.add_argument("--fallback-model-url")
    parser.add_argument("--stage1-gate", action="store_true",
                        help="Limit samples for each test product using its recorded Stage 1 evaluation count")
    parser.add_argument("--raw-openbabel", action="store_true",
                        help="Keep the original radical-like decoded atom notation")
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--max-model-calls", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-routes", type=int, default=5)
    parser.add_argument("--max-states", type=int, default=2000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.llm_step_fallback and args.fallback_model_url:
        parser.error("Choose one fallback for uncached intermediates")
    if args.llm_url and not args.llm_model:
        parser.error("LLM use requires --llm-model")
    if args.llm_step_fallback and not args.llm_url:
        parser.error("--llm-step-fallback requires --llm-url")
    workspace = Path(os.environ.get("SPECTRARETRO_MODEL_WORKSPACE", str(Path(__file__).resolve().parents[2]))).resolve()
    backup = json.loads((workspace / "experiments/stage2_geometry20k_20260915/server_results/backup_in_progress.json")
                        .read_text(encoding="utf-8"))
    if backup.get("state") != "verified":
        parser.error("GDiffRetro backup has not passed verification")
    snapshot = Path(backup["local_snapshot"])
    merged = snapshot / "server_run/evaluation_edm_3d/merged_result.csv"
    counts = snapshot / "stage1_unknown_test_300.pkl" if args.stage1_gate else None
    model = GDiffRetroSnapshotModel(merged, counts, repair_openbabel=not args.raw_openbabel)
    target = canonical_smiles(args.target)
    if target not in model.available_products:
        parser.error("Target is not among the 826 preserved products; use the live --model-url interface")
    ranker = (OpenAICompatibleRanker(args.llm_url, args.llm_model, os.getenv("LLM_API_KEY"))
              if args.llm_url else None)
    if args.llm_step_fallback:
        model = CascadeModel(model, OpenAICompatibleStepFallback(
            args.llm_url, args.llm_model, os.getenv("LLM_API_KEY")))
    elif args.fallback_model_url:
        model = CascadeModel(model, HTTPModel(args.fallback_model_url, os.getenv("RETRO_MODEL_TOKEN")))
    default_stock = ("geometry20k_reference_terminal_set.smi" if args.raw_openbabel
                     else "geometry20k_hydrogen_filled_terminal_set.smi")
    stock_path = args.stock or workspace / "agentic_retrosynthesis/data" / default_stock
    stock = {line.strip() for line in stock_path.read_text(encoding="utf-8").splitlines()
             if line.strip() and not line.lstrip().startswith("#")}
    result = Planner(model, stock, ranker, max_depth=args.max_depth,
                     max_model_calls=args.max_model_calls,
                     expansion_top_k=args.top_k, max_routes=args.max_routes,
                     max_states=args.max_states).plan(target).to_dict()
    result["dataset_terminal_set"] = args.stock is None
    result["planner_postprocess"] = "none" if args.raw_openbabel else "neutral_bracket_implicit_h_fill"
    result["stock_source"] = str(stock_path.resolve())
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(args.output.resolve())
    else:
        print(rendered)


if __name__ == "__main__":
    main()
