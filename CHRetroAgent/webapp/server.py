"""CHRetroAgent browser API for route planning with SpectraRetro and LLM tools."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from copy import copy
import json
import mimetypes
import os
from pathlib import Path
import subprocess
import threading
import time
from urllib.parse import parse_qs, urlparse
from urllib.parse import urlsplit
import uuid

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

from agentic_retrosynthesis.adapters import CascadeModel, SpectraRetroSnapshotModel
from agentic_retrosynthesis.llm import OpenAICompatibleRanker
from agentic_retrosynthesis.planner import Planner, canonical_smiles
from agentic_retrosynthesis.constraints import RouteConstraints
from agentic_retrosynthesis.tool_policy import ToolEnrichedModel, molecule_observation
from agentic_retrosynthesis.chem_tools import ChemistryToolkit
from .llm_config import configured_llm
from .storage import WorkspaceStore
from .agent_runtime import AgentRuntime
from .secret_store import save_api_key, load_api_key, credential_status
from .fixed_core_cases import FixedCoreCaseService
from .route_diagram import route_svg, route_checks
from .supplier_quotes import LiveSupplierQuotes
from .route_assessment import assess_complete_routes


ROOT = Path(os.environ.get("SPECTRARETRO_MODEL_WORKSPACE", str(Path(__file__).resolve().parents[2]))).resolve()
STATIC = Path(__file__).resolve().parent / "static"
EXPERIMENT = ROOT / "experiments/stage2_geometry20k_20260915"
BACKUP_FILE = EXPERIMENT / "server_results/backup_in_progress.json"
BACKUP = json.loads(BACKUP_FILE.read_text(encoding="utf-8")) if BACKUP_FILE.is_file() else {"state": "resources_required", "local_snapshot": str(EXPERIMENT / "snapshot")}
SNAPSHOT = Path(BACKUP["local_snapshot"])
MODEL_FILE = SNAPSHOT / "server_run/evaluation_edm_3d/merged_result.csv"
STAGE1_FILE = ROOT / "webapp/data/stage1_fspec_best_300.pkl"
STAGE1_MANIFEST_FILE = ROOT / "webapp/data/stage1_fspec_best_300.manifest.json"
STAGE1_MANIFEST = json.loads(STAGE1_MANIFEST_FILE.read_text(encoding="utf-8")) if STAGE1_MANIFEST_FILE.is_file() else {}
SHOWCASE_FILE = ROOT / "webapp/data/showcases.json"
STOCK_FILE = ROOT / "agentic_retrosynthesis/data/geometry20k_hydrogen_filled_expandable_stock.smi"
LIVE_PYTHON = Path(os.environ.get("SPECTRARETRO_PYTHON", __import__("sys").executable))


def load_showcases() -> list[dict]:
    rows = json.loads(SHOWCASE_FILE.read_text(encoding="utf-8")) if SHOWCASE_FILE.is_file() else []
    successful = [row for row in rows if row.get("route_count", 0) > 0]
    labels = [
        ("Halogenated aromatic aldehyde", "A 300/300 center-hit case with broad candidate coverage and convergent routes."),
        ("Nitrogen heterocycle ketone", "A heterocyclic target with complete Stage1 gating and Stage2 3D completion."),
        ("Fused lactam scaffold", "High-frequency precursor candidates for a more complex fused scaffold."),
        ("Fluorinated indole aldehyde", "A model case combining a fused heterocycle with halogen substitution."),
    ]
    return [{**row, "title": labels[i][0], "description": labels[i][1]}
            for i, row in enumerate(successful[:4])]


SHOWCASES = load_showcases()
EXAMPLES = [item["product"] for item in SHOWCASES[:3]]


class AppState:
    def __init__(self, data_dir=None):
        if BACKUP.get("state") != "verified":
            raise RuntimeError("CHRetroAgent requires the model workspace for live planning. See CHRetroAgent/README.md or run CHRetroAgent/preview.py to view a saved route.")
        self.model = SpectraRetroSnapshotModel(MODEL_FILE, stage1_counts=STAGE1_FILE,
                                               repair_openbabel=True)
        self.fixed_core_cases = FixedCoreCaseService()
        self.ranker, self.llm_fallback = configured_llm()
        self.planning_model = (CascadeModel(self.model, self.llm_fallback)
                               if self.llm_fallback else self.model)
        self.stock = {canonical_smiles(line) for line in STOCK_FILE.read_text(encoding="utf-8").splitlines()
                      if line and not line.startswith("#")}
        self.jobs: dict[str, dict] = {}
        self.analyses: dict[str, dict] = {}
        self.lock = threading.Lock()
        self.gpu = threading.Semaphore(1)
        self.data_dir = Path(data_dir or (ROOT / "webapp/data/workspace"))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.store = WorkspaceStore(self.data_dir / "workspace.sqlite3")
        self.toolkit = ChemistryToolkit(self.model, self.stock,
                                       inventory_path=self.data_dir / "inventory.json",
                                       evidence_path=self.data_dir / "condition_evidence.json")
        self.agent = AgentRuntime(self, self.toolkit, self.store)
        self.supplier_quotes = LiveSupplierQuotes(self.data_dir)
        self.chat_jobs = {}
        self.chat_slots = threading.BoundedSemaphore(3)

    def tool_catalog(self):
        records = {item["name"]: item for item in self.toolkit.catalog()}
        return [{**records.get(spec["function"]["name"], {"status": "ready", "category": "Workspace"}),
                 **spec["function"]} for spec in self.agent.schemas()]

    def add_chat_job(self, request):
        session_id = str(request.get("session_id", ""))
        if not self.agent.get_session(session_id):
            raise ValueError("Conversation not found")
        message = request.get("message")
        if not isinstance(message, str) or not message.strip() or len(message) > 12000:
            raise ValueError("Enter a message of 1 to 12000 characters")
        context = request.get("context", {})
        if not isinstance(context, dict):
            raise ValueError("Conversation context must be an object")
        with self.lock:
            if any(job["session_id"] == session_id and job["status"] == "running"
                   for job in self.chat_jobs.values()):
                raise ValueError("Wait for the current reply in this conversation")
            if not self.chat_slots.acquire(blocking=False):
                raise ValueError("All assistant slots are busy; try again shortly")
            job_id = uuid.uuid4().hex
            self.chat_jobs[job_id] = {"id": job_id, "session_id": session_id,
                                      "status": "running", "created": time.time()}
            completed = sorted((j for j in self.chat_jobs.values() if j["status"] != "running"),
                               key=lambda j: j["created"])
            for old in completed[:-100]:
                self.chat_jobs.pop(old["id"], None)
        def work():
            try:
                result = self.agent.chat(session_id, message, context)
                with self.lock:
                    self.chat_jobs[job_id].update(status="complete", result=result)
            except Exception as exc:
                with self.lock:
                    self.chat_jobs[job_id].update(status="error", error=f"Assistant task failed ({type(exc).__name__})")
            finally:
                self.chat_slots.release()
        threading.Thread(target=work, daemon=True).start()
        return job_id

    def configure_llm(self, request: dict) -> dict:
        url = str(request.get("url", "")).strip()
        model = str(request.get("model", "")).strip()
        api_key = str(request.get("api_key", "")).strip()
        step_fallback = bool(request.get("step_fallback", False))
        parsed = urlsplit(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("The LLM endpoint must be a valid HTTP or HTTPS URL")
        if not model:
            raise ValueError("Enter an LLM model name")
        key = api_key or load_api_key(url)
        if not key and os.getenv("LLM_URL") == url:
            key = os.getenv("LLM_API_KEY") or None
        probe = OpenAICompatibleRanker(url, model, key, timeout=45)
        try:
            probe.choose_frontier(("CC", "O"), ())
        except Exception as exc:
            raise ValueError(f"LLM connection or structured-output validation failed ({type(exc).__name__})") from None
        if api_key:
            save_api_key(api_key, url)
        target = Path(__file__).resolve().parent / "llm_settings.local.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps({"url": url, "model": model,
                                         "step_fallback": step_fallback,
                                         "timeout": 60, "max_tokens": 2048},
                                        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, target)
        os.environ["LLM_URL"] = url
        os.environ["LLM_MODEL"] = model
        if key:
            os.environ["LLM_API_KEY"] = key
        else:
            os.environ.pop("LLM_API_KEY", None)
        ranker, fallback = configured_llm()
        with self.lock:
            self.ranker, self.llm_fallback = ranker, fallback
            self.planning_model = CascadeModel(self.model, fallback) if fallback else self.model
        return {"llm_ready": True, "model": model, "step_fallback": step_fallback,
                "connection_tested": True, "key_storage": credential_status(url).get("storage")}

    def llm_status(self):
        provider = self.ranker
        status = credential_status(provider.url if provider else None)
        return {"configured": bool(provider), "model": provider.model if provider else None,
                "url": provider.url if provider else None,
                "provider": "DeepSeek" if provider and urlsplit(provider.url).hostname == "api.deepseek.com" else "External LLM",
                "credential_saved": bool(status.get("available")),
                "key_available": bool(provider and provider.api_key),
                "credential_management": "server",
                "step_fallback": bool(self.llm_fallback)}

    def add_industrial_job(self, request):
        from agentic_retrosynthesis.industrial import validate_request
        target, _, _, policy, _ = validate_request(request)
        if policy == "llm" and self.ranker is None:
            raise ValueError("Configure the external LLM first, or select size enumeration")
        job_id = uuid.uuid4().hex
        with self.lock:
            self.jobs[job_id] = {"id": job_id, "status": "queued", "stage": "queued",
                                 "fraction": 0, "product": target, "created": time.time(),
                                 "request": request, "mode": "constrained_product_only_prototype"}
        threading.Thread(target=self._run_industrial_job, args=(job_id,), daemon=True).start()
        return job_id

    def _run_industrial_job(self, job_id):
        with self.gpu:
            self._update(job_id, status="running", stage="product_only_search")
            folder = ROOT / "agentic_retrosynthesis/runs" / ("industrial_" + job_id)
            request_path = ROOT / "agentic_retrosynthesis/runs" / (job_id + "_request.json")
            request_path.parent.mkdir(exist_ok=True)
            request_path.write_text(json.dumps(self.jobs[job_id]["request"], ensure_ascii=False), encoding="utf-8")
            try:
                with request_path.with_suffix(".log").open("w", encoding="utf-8") as log:
                    subprocess.run([str(LIVE_PYTHON), "-m", "agentic_retrosynthesis.industrial",
                                    "--request", str(request_path), "--output", str(folder)],
                                   cwd=ROOT, stdout=log, stderr=log, check=True, timeout=3600)
                result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
                analysis_id = self.save_analysis(result)
                self._update(job_id, status="complete", stage="complete", fraction=1,
                             result=result, analysis_id=analysis_id, completed=time.time())
            except Exception as exc:
                self._update(job_id, status="error", stage="error",
                             error=f"Product-only task failed ({type(exc).__name__}); inspect local task log",
                             completed=time.time())

    def add_job(self, request: dict) -> str:
        target = canonical_smiles(request.get("product", ""))
        if target not in self.model.available_products:
            raise ValueError("This product is not in the current model catalogue. Select a case or catalogue target.")
        draws = int(request.get("draws", 300))
        steps = int(request.get("steps", 100))
        top_k = int(request.get("top_k", 10))
        max_calls = int(request.get("max_model_calls", 3))
        max_depth = int(request.get("max_depth", 3))
        if not (1 <= draws <= 300 and 1 <= steps <= 100 and
                1 <= top_k <= 300 and 1 <= max_calls <= 5 and 1 <= max_depth <= 6):
            raise ValueError("The sampling or search budget is outside the allowed range")
        job_id = uuid.uuid4().hex
        normalized = {"product": target, "draws": draws, "steps": steps,
                      "top_k": top_k, "max_model_calls": max_calls,
                      "max_depth": max_depth,
                      "seed": int(request.get("seed", 0))}
        preferences = request.get("route_preferences", "")
        assessment_enabled = request.get("assess_routes", True)
        if not isinstance(preferences, str) or len(preferences) > 2000 or type(assessment_enabled) is not bool:
            raise ValueError("Invalid route assessment preferences or switch")
        normalized.update(route_preferences=preferences, assess_routes=assessment_enabled,
                          use_llm=False)
        with self.lock:
            self.jobs[job_id] = {"id": job_id, "status": "queued", "stage": "queued",
                                 "fraction": 0, "product": target, "created": time.time(),
                                 "request": normalized}
        threading.Thread(target=self._run_job, args=(job_id,), daemon=True).start()
        return job_id

    def _update(self, job_id: str, **values):
        with self.lock:
            self.jobs[job_id].update(values)

    def save_analysis(self, result: dict) -> str:
        return self.store.put_analysis(result)

    def evaluate_routes(self, result, request=None):
        request = request or {}
        return assess_complete_routes(result, self.ranker,
            preferences=request.get("route_preferences", ""),
            enabled=request.get("assess_routes", True))

    def plan_fixed_core(self, request):
        return self.evaluate_routes(self.fixed_core_cases.plan(request), request)

    def get_analysis(self, analysis_id: str) -> dict | None:
        result = self.store.get_analysis(analysis_id)
        return ({"id": analysis_id, "created": result.get("created"), "result": result}
                if result is not None else None)

    def _run_job(self, job_id: str):
        with self.gpu:
            self._update(job_id, status="running", stage="loading_model")
            request = self.jobs[job_id]["request"]
            try:
                process = subprocess.Popen(
                    [str(LIVE_PYTHON), "-m", "webapp.model_runner"],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding="utf-8", cwd=ROOT)
                assert process.stdin and process.stdout
                process.stdin.write(json.dumps(request) + "\n")
                process.stdin.close()
                result = None
                error = None
                log_tail = ""
                for line in process.stdout:
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        log_tail = (log_tail + line)[-600:]
                        continue
                    if event.get("type") == "progress":
                        details = {key: value for key, value in event.items() if key != "type"}
                        self._update(job_id, **details)
                    elif event.get("type") == "result":
                        result = event["result"]
                    elif event.get("type") == "error":
                        error = event.get("error")
                code = process.wait()
                if code != 0 or result is None:
                    raise RuntimeError(error or log_tail or "The model task returned no result")
                self._update(job_id, stage="assessing_routes")
                result = self.evaluate_routes(result, request)
                analysis_id = self.save_analysis(result)
                self._update(job_id, status="complete", stage="complete", fraction=1,
                             result=result, analysis_id=analysis_id, completed=time.time())
            except Exception as exc:
                self._update(job_id, status="error", stage="error", error=str(exc),
                             completed=time.time())

    def get_job(self, job_id: str) -> dict | None:
        with self.lock:
            record = self.jobs.get(job_id)
            return dict(record) if record else None

    def quick_plan(self, request: dict) -> dict:
        case_id = self.fixed_core_cases.find_case(request.get("case_id") or request.get("product", ""))
        if request.get("mode") == "fixed_core_latest_candidates" or (case_id and request.get("mode") not in ("history", "live")):
            case_request = dict(request, case_id=case_id)
            # Defaults in the general planner (4-6 steps) cannot accommodate these routes.
            if request.get("mode") != "fixed_core_latest_candidates":
                for key in ("max_depth", "top_k", "max_model_calls", "max_states"):
                    case_request.pop(key, None)
            return self.plan_fixed_core(case_request)
        target = canonical_smiles(request.get("product", ""))
        if target not in self.model.available_products:
            raise ValueError("This product is not in the current model catalogue. Select a case or catalogue target.")
        top_k = int(request.get("top_k", 100))
        if not 1 <= top_k <= 300:
            raise ValueError("The candidate count must be between 1 and 300")
        max_depth = int(request.get("max_depth", 4))
        max_routes = int(request.get("max_routes", 20))
        if not 1 <= max_depth <= 6 or not 1 <= max_routes <= 30:
            raise ValueError("The route depth or route count is outside the allowed range")
        max_calls = int(request.get("max_model_calls", 50))
        llm_calls = int(request.get("max_llm_calls", 12))
        if not 1 <= max_calls <= 100 or not 1 <= llm_calls <= 24:
            raise ValueError("Model calls must be 1 to 100 and LLM calls 1 to 24")
        raw_constraints = request.get("constraints", {})
        if not isinstance(raw_constraints, dict) or set(raw_constraints) - {"forbidden_smiles", "unavailable_stock"}:
            raise ValueError("Only forbidden_smiles and unavailable_stock constraints are supported here")
        for value in raw_constraints.values():
            if not isinstance(value, list) or len(value) > 100 or not all(isinstance(s, str) for s in value):
                raise ValueError("Molecule constraints must be lists of at most 100 SMILES")
        constraints = RouteConstraints(**raw_constraints) if raw_constraints else None
        stock = constraints.effective_stock(self.stock) if constraints else self.stock
        use_llm = bool(request.get("use_llm", False))
        fallback = copy(self.llm_fallback) if self.llm_fallback and use_llm else None
        fallback_budget = min(3, llm_calls) if fallback else 0
        if fallback:
            fallback.calls, fallback.max_calls = 0, fallback_budget
            fallback.usage = {"prompt_tokens": 0, "completion_tokens": 0}
        planning_model = CascadeModel(self.model, fallback) if fallback else self.model
        model = ToolEnrichedModel(planning_model, stock)
        ranker = copy(self.ranker) if self.ranker and use_llm else None
        if ranker:
            ranker.calls, ranker.max_calls = 0, llm_calls - fallback_budget
            ranker.usage = {"prompt_tokens": 0, "completion_tokens": 0}
        started = time.time()
        result = Planner(model, stock, ranker,
                         max_depth=max_depth, max_model_calls=max_calls,
                         expansion_top_k=top_k, max_routes=max_routes,
                         max_states=5000, constraints=constraints).plan(target).to_dict()
        result.update({"mode": "preserved_model_draws", "seconds": round(time.time() - started, 2),
                       "one_step_candidates": self.model.predict(target, top_k),
                       "draws_per_product": 300, "sampling_steps": 100,
                       "llm_policy": "openai_compatible" if ranker else "model_frequency",
                       "llm_calls": (ranker.calls if ranker else 0) + (fallback.calls if fallback else 0),
                       "tool_observations": model.observations,
                       "target_properties": molecule_observation(target),
                       "search_settings": {"top_k": top_k, "max_depth": max_depth,
                                           "max_routes": max_routes, "max_model_calls": max_calls,
                                           "max_llm_calls": llm_calls, "constraints": raw_constraints},
                       "llm_step_fallback": bool(fallback),
                       "stage1": {**self.model.gate_status(target),
                                  "stage2_draws": self.model.gate_status(target)["hits"],
                                  "requested_draws": 300,
                                  "gate_sha256": self.model.stage1_sha256},
                       "conditions": "spectraretro_3d_conditioning",
                       "terminal_set": "configured_starting_material_set"})
        return self.evaluate_routes(result, request)


def molecule_svg(smiles: str, highlight_atoms: list[int] | None = None,
                 add_indices: bool = False) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Unable to draw this molecule")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    drawer = rdMolDraw2D.MolDraw2DSVG(420, 260)
    options = drawer.drawOptions()
    options.clearBackground = False
    options.bondLineWidth = 2.4
    options.addAtomIndices = add_indices
    valid_atoms = [index for index in (highlight_atoms or [])
                   if 0 <= index < mol.GetNumAtoms()]
    drawer.DrawMolecule(mol, highlightAtoms=valid_atoms)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def ch_activation_analysis(smiles: str) -> dict:
    """Canonical-indexed structural C-H inventory shared with Agent tools."""
    return ChemistryToolkit().execute("analyze_ch_sites", {"smiles": smiles})


def make_handler(state: AppState):
    class Handler(BaseHTTPRequestHandler):
        def respond(self, code: int, data, content_type="application/json; charset=utf-8"):
            body = (json.dumps(data, ensure_ascii=False).encode("utf-8")
                    if isinstance(data, (dict, list)) else data)
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store" if self.path.startswith("/api/") else "no-cache")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/api/llm/status":
                    self.respond(200, state.llm_status())
                    return
                if parsed.path == "/api/tools":
                    self.respond(200, {"tools": state.tool_catalog()})
                    return
                if parsed.path == "/api/inventory":
                    rows = state.toolkit.inventory()
                    self.respond(200, {"rows": rows, "count": len(rows)})
                    return
                if parsed.path == "/api/condition-evidence":
                    rows = state.toolkit.condition_evidence()
                    self.respond(200, {"rows": rows, "count": len(rows)})
                    return
                if parsed.path == "/api/history":
                    self.respond(200, {"analyses": state.store.list_analyses(limit=100)})
                    return
                if parsed.path == "/api/sessions":
                    self.respond(200, {"sessions": state.agent.list_sessions()})
                    return
                if parsed.path.startswith("/api/sessions/"):
                    session = state.agent.get_session(parsed.path.rsplit("/", 1)[-1])
                    self.respond(200, session) if session else self.respond(404, {"error": "Conversation not found"})
                    return
                if parsed.path.startswith("/api/chat/jobs/"):
                    with state.lock:
                        job = state.chat_jobs.get(parsed.path.rsplit("/", 1)[-1])
                        job = dict(job) if job else None
                    self.respond(200, job) if job else self.respond(404, {"error": "Assistant task not found"})
                    return
                if parsed.path.startswith("/api/reports/"):
                    analysis_id = parsed.path.rsplit("/", 1)[-1]
                    kind = parse_qs(parsed.query).get("format", ["json"])[0]
                    report = state.agent.execute_tool("export_report", {"analysis_id": analysis_id, "format": kind})
                    if "content" not in report:
                        self.respond(404, report)
                    else:
                        body = report["content"].encode("utf-8")
                        self.send_response(200)
                        self.send_header("Content-Type", report["mime_type"])
                        self.send_header("Content-Disposition", 'attachment; filename="' + report["filename"] + '"')
                        self.send_header("Content-Length", str(len(body)))
                        self.end_headers()
                        self.wfile.write(body)
                    return
                if parsed.path == "/api/reviews":
                    analysis_id = parse_qs(parsed.query).get("analysis_id", [None])[0]
                    self.respond(200, {"reviews": state.store.list_reviews(analysis_id)})
                    return
                if parsed.path == "/api/meta":
                    self.respond(200, {"name": "CHRetroAgent", "products": len(state.model.available_products),
                                       "model": "SpectraRetro 3D · Stage1 + Stage2", "examples": EXAMPLES,
                                       "showcases": SHOWCASES,
                                       "live_ready": LIVE_PYTHON.is_file(),
                                       "stage1_ready": STAGE1_FILE.is_file(),
                                       "stage1_model": STAGE1_MANIFEST["identity"],
                                       "stage1_official_top1": STAGE1_MANIFEST["official_test_top1"],
                                       "stage1_catalogue_top1": STAGE1_MANIFEST["catalogue_deterministic_metrics"]["top-1 accuracy"],
                                       "candidate_budget": 300,
                                       "llm_ready": bool(state.ranker),
                                       "llm_model": state.ranker.model if state.ranker else None,
                                       "llm_step_fallback": bool(state.llm_fallback),
                                       "whole_route_assessment_ready": bool(state.ranker),
                                       "fixed_core_cases": state.fixed_core_cases.catalogue(),
                                       "tool_count": len(state.tool_catalog()),
                                       "persistent_workspace": True,
                                       "terminal_set": "Configured starting-material set"})
                    return
                if parsed.path == "/api/fixed-core/cases":
                    self.respond(200, {"cases": state.fixed_core_cases.catalogue()})
                    return
                if parsed.path == "/api/catalog":
                    params = parse_qs(parsed.query)
                    query = params.get("query", [""])[0].strip().lower()
                    limit = max(1, min(int(params.get("limit", ["60"])[0]), 120))
                    offset = max(0, int(params.get("offset", ["0"])[0]))
                    products = sorted(state.model.available_products)
                    if query:
                        products = [product for product in products if query in product.lower()]
                    rows = []
                    for product in products[offset:offset + limit]:
                        gate = state.model.gate_status(product)
                        rows.append({"product": product,
                                     "stage1_hits": gate.get("hits", 0),
                                     "stage1_samples": gate.get("samples", 300),
                                     "stage1_passed": bool(gate.get("passed")),
                                     "stage2_budget": gate.get("hits", 0)})
                    self.respond(200, {"total": len(products), "offset": offset,
                                       "limit": limit, "items": rows})
                    return
                if parsed.path.startswith("/api/jobs/"):
                    job = state.get_job(parsed.path.rsplit("/", 1)[-1])
                    self.respond(200, job) if job else self.respond(404, {"error": "Task not found"})
                    return
                if parsed.path.startswith("/api/analyses/"):
                    analysis = state.get_analysis(parsed.path.rsplit("/", 1)[-1])
                    if analysis and not analysis["result"].get("conversation_session_id"):
                        analysis["result"]["conversation_session_id"] = state.store.session_for_analysis(parsed.path.rsplit("/", 1)[-1])
                    self.respond(200, analysis) if analysis else self.respond(404, {"error": "Report not found"})
                    return
                if parsed.path == "/api/route-checks":
                    params = parse_qs(parsed.query)
                    analysis = state.get_analysis(params.get("analysis_id", [""])[0])
                    if analysis is None:
                        self.respond(404, {"error": "Analysis not found"})
                        return
                    self.respond(200, route_checks(analysis["result"], int(params.get("route_index", ["0"])[0])))
                    return
                if parsed.path == "/api/route-suppliers":
                    params = parse_qs(parsed.query)
                    analysis = state.get_analysis(params.get("analysis_id", [""])[0])
                    if analysis is None:
                        self.respond(404, {"error": "Analysis not found"})
                        return
                    self.respond(200, state.supplier_quotes.route_quotes(
                        analysis["result"], int(params.get("route_index", ["0"])[0]),
                        refresh=params.get("refresh", ["0"])[0] == "1"))
                    return
                if parsed.path == "/api/molecule.svg":
                    params = parse_qs(parsed.query)
                    smiles = params.get("smiles", [""])[0]
                    highlight = [int(value) for value in params.get("highlight", [""])[0].split(",")
                                 if value.strip().isdigit()]
                    add_indices = params.get("indices", ["0"])[0] == "1"
                    self.respond(200, molecule_svg(smiles, highlight, add_indices).encode("utf-8"),
                                 "image/svg+xml; charset=utf-8")
                    return
                if parsed.path == "/api/route-diagram.svg":
                    params = parse_qs(parsed.query)
                    identifier = params.get("analysis_id", [""])[0]
                    analysis = state.get_analysis(identifier)
                    if analysis is None:
                        self.respond(404, {"error": "Analysis not found"})
                        return
                    route_index = int(params.get("route_index", ["0"])[0])
                    self.respond(200, route_svg(analysis["result"], route_index), "image/svg+xml; charset=utf-8")
                    return
                filename = "index.html" if parsed.path == "/" else parsed.path.lstrip("/")
                target = (STATIC / filename).resolve()
                if not target.is_file() or STATIC.resolve() not in target.parents:
                    self.respond(404, {"error": "Page not found"})
                    return
                kind = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
                self.respond(200, target.read_bytes(), kind)
            except ValueError as exc:
                self.respond(400, {"error": str(exc)})
            except Exception as exc:
                self.respond(500, {"error": f"{type(exc).__name__}: {exc}"})

        def do_POST(self):
            parsed = urlparse(self.path)
            if parsed.path not in ("/api/jobs", "/api/plan", "/api/analyses",
                                   "/api/ch-analysis", "/api/llm/configure", "/api/industrial/jobs",
                                   "/api/tools/execute", "/api/sessions", "/api/chat",
                                   "/api/compare", "/api/reviews", "/api/inventory", "/api/condition-evidence",
                                   "/api/fixed-core/plan"):
                self.respond(404, {"error": "Endpoint not found"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                body_limit = 2_000_000 if parsed.path in ("/api/inventory", "/api/condition-evidence") else 65536
                if not 0 < size <= body_limit:
                    raise ValueError("Invalid request length")
                request = json.loads(self.rfile.read(size))
                if not isinstance(request, dict):
                    raise ValueError("Request must be a JSON object")
                if parsed.path == "/api/tools/execute":
                    self.respond(200, {"result": state.agent.execute_tool(request.get("name", ""), request.get("arguments", {}))})
                elif parsed.path == "/api/sessions":
                    self.respond(201, state.agent.create_session(request.get("title", "New conversation")))
                elif parsed.path == "/api/chat":
                    self.respond(202, {"job_id": state.add_chat_job(request)})
                elif parsed.path == "/api/compare":
                    self.respond(200, state.agent.execute_tool("compare_routes", request))
                elif parsed.path == "/api/reviews":
                    self.respond(200, state.agent.execute_tool("request_human_review", request))
                elif parsed.path == "/api/inventory":
                    self.respond(200, state.toolkit.set_inventory(request.get("rows", [])))
                elif parsed.path == "/api/condition-evidence":
                    self.respond(200, state.toolkit.set_condition_evidence(request.get("rows", [])))
                elif parsed.path == "/api/industrial/jobs":
                    self.respond(202, {"job_id": state.add_industrial_job(request)})
                elif parsed.path == "/api/jobs":
                    job_id = state.add_job(request)
                    self.respond(202, {"job_id": job_id})
                elif parsed.path == "/api/analyses":
                    result = state.quick_plan(request)
                    self.respond(200, {"analysis_id": state.save_analysis(result),
                                       "target": result.get("target")})
                elif parsed.path == "/api/fixed-core/plan":
                    result = state.plan_fixed_core(request)
                    self.respond(200, {"analysis_id": state.save_analysis(result), "target": result["target"],
                                       "solved": result["solved"], "route_count": len(result["routes"]),
                                       "mode": result["mode"]})
                elif parsed.path == "/api/ch-analysis":
                    self.respond(200, state.toolkit.execute("analyze_ch_sites", {"smiles": str(request.get("smiles", ""))}))
                elif parsed.path == "/api/llm/configure":
                    self.respond(200, state.configure_llm(request))
                else:
                    self.respond(200, state.quick_plan(request))
            except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                self.respond(400, {"error": str(exc)})
            except Exception as exc:
                self.respond(500, {"error": f"{type(exc).__name__}: {exc}"})

        def log_message(self, *args):
            pass

    return Handler


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Launch the CHRetroAgent browser application")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8120)
    args = parser.parse_args()
    state = AppState()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    print(f"CHRetroAgent is ready: http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
