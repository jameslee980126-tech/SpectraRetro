"""Dependency-free localhost HTTP tool server for the GDiffRetro route agent."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path

from .adapters import CascadeModel, GDiffRetroSnapshotModel, HTTPModel
from .llm import OpenAICompatibleRanker, OpenAICompatibleStepFallback
from .planner import Planner, canonical_smiles


def make_handler(model, stock: set[str], ranker=None):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload: dict):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"status": "ready", "stock_size": len(stock)})
            else:
                self._send(404, {"error": "Unknown endpoint"})

        def do_POST(self):
            if self.path not in ("/predict", "/plan"):
                self._send(404, {"error": "Unknown endpoint"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 65536:
                    raise ValueError("JSON request must be between 1 and 65536 bytes")
                request = json.loads(self.rfile.read(size))
                product = request.get("product") if self.path == "/predict" else request.get("target")
                if not isinstance(product, str):
                    raise ValueError("Specify a product or target SMILES string")
                normalized = canonical_smiles(product)
                top_k = int(request.get("top_k", 10))
                if not 1 <= top_k <= 300:
                    raise ValueError("top_k must be between 1 and 300")
                if self.path == "/predict":
                    self._send(200, {"product": normalized,
                                     "candidates": model.predict(normalized, top_k)})
                    return
                depth = int(request.get("max_depth", 4))
                calls = int(request.get("max_model_calls", 50))
                routes = int(request.get("max_routes", 5))
                states = int(request.get("max_states", 2000))
                if not (1 <= depth <= 20 and 1 <= calls <= 500 and
                        1 <= routes <= 100 and 1 <= states <= 50000):
                    raise ValueError("Search budget outside supported range")
                result = Planner(model, stock, ranker, max_depth=depth,
                                 max_model_calls=calls, expansion_top_k=top_k,
                                 max_routes=routes, max_states=states).plan(normalized)
                self._send(200, result.to_dict())
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                self._send(400, {"error": str(exc)})
            except Exception as exc:
                self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

        def log_message(self, *args):
            pass

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve GDiffRetro snapshot prediction and route planning")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8108)
    parser.add_argument("--stock", type=Path)
    parser.add_argument("--llm-url")
    parser.add_argument("--llm-model")
    parser.add_argument("--llm-step-fallback", action="store_true")
    parser.add_argument("--fallback-model-url")
    args = parser.parse_args()
    if args.llm_step_fallback and args.fallback_model_url:
        parser.error("Choose one fallback")
    if args.llm_url and not args.llm_model:
        parser.error("LLM use requires --llm-model")
    if args.llm_step_fallback and not args.llm_url:
        parser.error("--llm-step-fallback requires --llm-url")
    workspace = Path(os.environ.get("SPECTRARETRO_MODEL_WORKSPACE", str(Path(__file__).resolve().parents[2]))).resolve()
    backup = json.loads((workspace / "experiments/stage2_geometry20k_20260915/server_results/backup_in_progress.json")
                        .read_text(encoding="utf-8"))
    if backup.get("state") != "verified":
        parser.error("The 3D EDM backup is not verified")
    snapshot = Path(backup["local_snapshot"])
    model = GDiffRetroSnapshotModel(snapshot / "server_run/evaluation_edm_3d/merged_result.csv",
                                    repair_openbabel=True)
    ranker = (OpenAICompatibleRanker(args.llm_url, args.llm_model, os.getenv("LLM_API_KEY"))
              if args.llm_url else None)
    if args.llm_step_fallback:
        model = CascadeModel(model, OpenAICompatibleStepFallback(
            args.llm_url, args.llm_model, os.getenv("LLM_API_KEY")))
    elif args.fallback_model_url:
        model = CascadeModel(model, HTTPModel(args.fallback_model_url, os.getenv("RETRO_MODEL_TOKEN")))
    stock_path = args.stock or workspace / "agentic_retrosynthesis/data/geometry20k_hydrogen_filled_terminal_set.smi"
    stock = {line.strip() for line in stock_path.read_text(encoding="utf-8").splitlines()
             if line.strip() and not line.startswith("#")}
    server = ThreadingHTTPServer((args.host, args.port), make_handler(model, stock, ranker))
    print(f"GDiffRetro agent ready at http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
