"""Render and serve CHRetroAgent routes using CPU-only RDKit."""
from __future__ import annotations

import argparse
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agentic_retrosynthesis.adapters import JSONModel
from agentic_retrosynthesis.planner import Planner
from webapp.route_diagram import route_checks, route_svg


def example_result():
    examples = ROOT / "src/agentic_retrosynthesis/examples"
    model = JSONModel(examples / "predictions.json")
    stock = set((examples / "stock.smi").read_text(encoding="utf-8").splitlines())
    result = Planner(model, stock, max_depth=4, max_model_calls=10).plan("CCOC(C)=O").to_dict()
    result["case_name"] = "Offline interface example"
    result["scope"] = "Synthetic interface fixture; not a learned-model result or chemical validation."
    return result


def render(report, index):
    svg = route_svg(report, route_index=index).decode("utf-8")
    if svg.startswith("<?xml"):
        svg = svg.split("?>", 1)[1]
    checks = route_checks(report, route_index=index)
    title = escape(str(report.get("case_name") or report.get("target") or "Route"))
    scope = escape(str(report.get("scope", "Saved molecular route; scores are ranking utilities.")))
    checks_text = escape(json.dumps(checks, ensure_ascii=False, indent=2))
    return ("<!doctype html><html lang='zh-CN'><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{title} · CHRetroAgent</title><style>body{{font:16px/1.6 system-ui,sans-serif;"
            "max-width:1400px;margin:32px auto;padding:0 24px;color:#17242a;background:#f7f9fa}}"
            "section{background:white;border:1px solid #dbe2e5;border-radius:12px;padding:20px;margin:20px 0}"
            ".diagram{overflow:auto}.diagram svg{height:auto;max-width:100%;min-width:620px}"
            "pre{white-space:pre-wrap;font-size:13px}button{padding:8px 16px;cursor:pointer}</style>"
            f"<header>CHRetroAgent</header><h1>{title}</h1><p>{scope}</p>"
            "<p>所有前体分支均显示。T 为配置的终端边界；Rank 为候选排名。</p>"
            "<button onclick='saveSvg()'>导出 SVG</button>"
            f"<section class='diagram'>{svg}</section><section><h2>结构检查</h2><pre>{checks_text}</pre></section>"
            "<script>function saveSvg(){const svg=document.querySelector('.diagram svg').outerHTML;"
            "const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([svg],{type:'image/svg+xml'}));"
            "a.download='route.svg';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}</script></html>")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, help="Planning-result JSON with target and routes fields")
    parser.add_argument("--route", type=int, default=0, help="Zero-based route index")
    parser.add_argument("--export", type=Path, help="Export standalone HTML or SVG and exit")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8120)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8-sig")) if args.report else example_result()
    if not isinstance(report, dict) or not report.get("routes"):
        parser.error("A report must contain at least one complete route")
    if args.export:
        args.export.parent.mkdir(parents=True, exist_ok=True)
        data = (route_svg(report, args.route) if args.export.suffix.lower() == ".svg"
                else render(report, args.route).encode("utf-8"))
        args.export.write_bytes(data)
        print(str(args.export.resolve()))
        return
    data = render(report, args.route).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path not in ("/", "/index.html"):
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *unused):
            pass

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"CHRetroAgent route viewer: http://{args.host}:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
