"""Command-line entry point for route planning."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .adapters import CascadeModel, GDiffRetroSnapshotModel, HTTPModel, JSONModel
from .llm import OpenAICompatibleRanker, OpenAICompatibleStepFallback
from .planner import Planner


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan routes from a prediction service or saved precursor candidates")
    parser.add_argument("target", help="Target molecule SMILES")
    parser.add_argument("--stock", type=Path, required=True, help="One purchasable SMILES per line")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model-url", help="POST endpoint accepting product and top_k")
    source.add_argument("--predictions", type=Path, help="Offline interface fixture JSON")
    source.add_argument("--gdiffretro-merged", type=Path,
                        help="Route candidates from a 3D diffusion test run (merged_result.csv)")
    parser.add_argument("--stage1-counts", type=Path,
                        help="Pickle of per-product Stage 1 evaluation counts for limiting draws from --gdiffretro-merged")
    parser.add_argument("--repair-openbabel", action="store_true",
                        help="Planner-only implicit-H fill for unsaturated decoded atom notation")
    parser.add_argument("--model-token-env", default="RETRO_MODEL_TOKEN")
    parser.add_argument("--llm-url", help="OpenAI-compatible /chat/completions endpoint")
    parser.add_argument("--llm-model")
    parser.add_argument("--llm-step-fallback", action="store_true",
                        help="Generate clearly labeled LLM precursor proposals for uncached intermediates")
    parser.add_argument("--fallback-model-url", help="Product-only predictor for uncached intermediates")
    parser.add_argument("--llm-key-env", default="LLM_API_KEY")
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--max-model-calls", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-routes", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    stock = {line.strip() for line in args.stock.read_text(encoding="utf-8").splitlines()
             if line.strip() and not line.lstrip().startswith("#")}
    if args.model_url:
        model = HTTPModel(args.model_url, os.getenv(args.model_token_env))
    elif args.gdiffretro_merged:
        model = GDiffRetroSnapshotModel(args.gdiffretro_merged, args.stage1_counts,
                                        repair_openbabel=args.repair_openbabel)
    else:
        model = JSONModel(args.predictions)
    ranker = None
    if args.llm_url:
        if not args.llm_model:
            parser.error("--llm-url requires --llm-model")
        ranker = OpenAICompatibleRanker(args.llm_url, args.llm_model, os.getenv(args.llm_key_env))
    if args.llm_step_fallback and not ranker:
        parser.error("--llm-step-fallback requires --llm-url and --llm-model")
    if args.llm_step_fallback and args.fallback_model_url:
        parser.error("Choose one fallback model for uncached intermediates")
    if args.gdiffretro_merged and args.llm_step_fallback:
        model = CascadeModel(model, OpenAICompatibleStepFallback(
            args.llm_url, args.llm_model, os.getenv(args.llm_key_env)))
    elif args.gdiffretro_merged and args.fallback_model_url:
        model = CascadeModel(model, HTTPModel(args.fallback_model_url,
                                               os.getenv(args.model_token_env)))
    result = Planner(model, stock, ranker, args.max_depth, args.max_model_calls,
                     args.top_k, args.max_routes).plan(args.target).to_dict()
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
