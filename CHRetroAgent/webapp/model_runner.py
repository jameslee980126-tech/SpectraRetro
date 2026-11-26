"""Run SpectraRetro for CHRetroAgent using the selected product and its prepared test inputs."""

from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
import pickle
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from rdkit import Chem

from agentic_retrosynthesis.adapters import CascadeModel, SpectraRetroSnapshotModel, fill_openbabel_hydrogens
from agentic_retrosynthesis.planner import Planner, canonical_smiles
from .decode_rdkit import decode_batch
from .llm_config import configured_llm


ROOT = Path(os.environ.get("SPECTRARETRO_MODEL_WORKSPACE", str(Path(__file__).resolve().parents[2]))).resolve()
EXPERIMENT = ROOT / "experiments/stage2_geometry20k_20260915"
sys.path.insert(0, str(EXPERIMENT))
from runtime_common import collate, create, records, seed, setup, to_device  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LiveSpectraRetro:
    def __init__(self, draws_per_product: int = 300, steps: int = 100, seed_value: int = 0,
                 progress=None):
        if not 1 <= draws_per_product <= 300 or not 1 <= steps <= 100:
            raise ValueError("draws must be 1..300 and sampling steps 1..100")
        self.draws_per_product = draws_per_product
        self.steps = steps
        self.seed_value = seed_value
        self.progress = progress or (lambda **_kwargs: None)
        setup(4)
        seed(seed_value)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        backup = json.loads((EXPERIMENT / "server_results/backup_in_progress.json")
                            .read_text(encoding="utf-8"))
        if backup.get("state") != "verified":
            raise RuntimeError("Frozen 3D EDM backup is not verified")
        self.snapshot = Path(backup["local_snapshot"])
        self.checkpoint = self.snapshot / "server_run/edm_3d/best.ckpt"
        self.size_draws_path = self.snapshot / "server_run/shared_sizes_test.npz"
        frozen = json.loads((self.snapshot / "server_run/evaluation_freeze_1500.json")
                            .read_text(encoding="utf-8"))
        self.checkpoint_sha256 = sha(self.checkpoint)
        if self.checkpoint_sha256 != frozen["arms"]["edm_3d"]["best_sha256"]:
            raise RuntimeError("3D EDM checkpoint differs from frozen evaluation")
        self.size_sha256 = sha(self.size_draws_path)
        self.stage1_path = ROOT / "webapp/data/stage1_fspec_best_300.pkl"
        self.stage1_manifest_path = ROOT / "webapp/data/stage1_fspec_best_300.manifest.json"
        self.stage1_manifest = json.loads(self.stage1_manifest_path.read_text(encoding="utf-8"))
        self.stage1_sha256 = sha(self.stage1_path)
        self.stage1_counts = pickle.loads(self.stage1_path.read_bytes())
        manifest = json.loads((self.snapshot / "server_run/shared_sizes_manifest.json")
                              .read_text(encoding="utf-8"))
        if self.size_sha256 != manifest["draws_sha256"]:
            raise RuntimeError("Learned-size draws differ from frozen evaluation")
        self.progress(stage="loading_conditions", fraction=0.05)
        self.data = records(3, "test")
        self.by_uuid = {int(record["uuid"]): record for record in self.data}
        with np.load(self.size_draws_path) as sizes:
            uuids = sizes["uuid"]
            matrix = sizes["sizes"]
            self.sizes = {int(uuid): matrix[i].copy() for i, uuid in enumerate(uuids)}
        self.products: dict[str, list[int]] = {}
        self.stage1_by_product: dict[str, int] = {}
        self.frag_bonds: dict[int, list[list[float]]] = {}
        self.rerank_observations = {}
        table = EXPERIMENT / "data_3d/geometry20k_test_table.csv"
        fragments = Chem.SDMolSupplier(str(EXPERIMENT / "data_3d/geometry20k_test_frag.sdf"),
                                        sanitize=False)
        with table.open(encoding="utf-8", newline="") as handle:
            for index, row in enumerate(csv.DictReader(handle)):
                normalized = canonical_smiles(row["product"])
                uuid = int(row["uuid"])
                self.products.setdefault(normalized, []).append(uuid)
                hits = int(self.stage1_counts[row["product"]])
                previous = self.stage1_by_product.setdefault(normalized, hits)
                if previous != hits:
                    raise RuntimeError(f"Inconsistent Stage1 gate for product {normalized}")
                fragment = fragments[index]
                if fragment is None:
                    raise RuntimeError(f"Unreadable stored synthon for uuid {uuid}")
                from spectraretro.fusion_reranker.integration import observed_synthon
                anchor_indices = [int(float(v)) for v in str(row['anchors']).replace(',', ';').split(';') if v.strip()]
                self.rerank_observations[uuid] = observed_synthon(fragment, anchor_indices)
                known_count = int(self.by_uuid[uuid]["fragment_mask"].sum())
                if fragment.GetNumAtoms() != known_count:
                    raise RuntimeError(f"Synthon atom count differs from cached condition for uuid {uuid}")
                self.frag_bonds[uuid] = [[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(),
                                          bond.GetBondTypeAsDouble()]
                                         for bond in fragment.GetBonds()]
        self.progress(stage="loading_model", fraction=0.12)
        self.model = create(3, self.device)
        state = torch.load(self.checkpoint, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["state_dict"])
        self.model.eval()
        self.model.edm.T = steps
        self.progress(stage="ready", fraction=0.2)

    def predict(self, product: str, top_k: int) -> list[dict]:
        product = canonical_smiles(product)
        uuids = self.products.get(product)
        if not uuids:
            return []
        stage1_hits = self.stage1_by_product[product]
        stage2_draws = min(self.draws_per_product, stage1_hits)
        self.progress(stage="stage1_center", fraction=0.2, current_product=product,
                      stage1_hits=stage1_hits, stage1_samples=300,
                      stage2_draws=stage2_draws,
                      missing_draws=self.draws_per_product - stage2_draws)
        if stage2_draws == 0:
            self.progress(stage="decoded", fraction=0.88, current_product=product)
            return []
        conditions = [self.by_uuid[uuid] for uuid in uuids]
        batch = to_device(collate(conditions), self.device)
        outputs = [[] for _ in uuids]
        coords_by_draw = []
        decoder_by_draw = []
        valid_draws = 0
        for draw in range(stage2_draws):
            self.progress(stage="sampling", fraction=0.24 + 0.61 * draw / stage2_draws,
                          current_product=product, draw=draw + 1, draws=stage2_draws,
                          stage1_hits=stage1_hits, stage1_samples=300)
            draw_sizes = torch.tensor([int(self.sizes[uuid][draw]) for uuid in uuids],
                                      device=self.device, dtype=torch.long)
            try:
                with torch.no_grad():
                    chain, mask = self.model.sample_chain(
                        batch, sample_fn=lambda _: draw_sizes, keep_frames=1)
                if not torch.isfinite(chain).all():
                    raise FloatingPointError("Non-finite model output")
                points = chain[0, ..., :3].detach().cpu().numpy()
                elements = chain[0, ..., 3:].argmax(-1).detach().cpu().numpy()
                lengths = mask.sum((1, 2)).int().detach().cpu().numpy()
                jobs = [{"coords": points[j, :length].tolist(),
                         "elements": elements[j, :length].tolist(),
                         "known_atoms": int(conditions[j]["fragment_mask"].sum()),
                         "known_bonds": self.frag_bonds[uuids[j]]}
                        for j, length in enumerate(lengths)]
                decoded = decode_batch(jobs)
                if len(decoded) != len(uuids):
                    raise RuntimeError("RDKit returned wrong number of fragments")
                for j, value in enumerate(decoded):
                    outputs[j].append(value["smiles"])
                coords_by_draw.append(jobs)
                decoder_by_draw.append([entry["bond_perception"] for entry in decoded])
                valid_draws += 1
            except Exception as exc:
                for row in outputs:
                    row.append("MISSING")
                coords_by_draw.append([])
                decoder_by_draw.append(["failed"] * len(uuids))
                self.progress(stage="sampling", fraction=0.24 + 0.61 * (draw + 1) / stage2_draws,
                              current_product=product, last_error=f"{type(exc).__name__}: {exc}")
        for _ in range(self.draws_per_product - stage2_draws):
            for row in outputs:
                row.append("MISSING")
            coords_by_draw.append([])
            decoder_by_draw.append(["stage1_truncated"] * len(uuids))
        combined = [".".join(outputs[j][draw] for j in range(len(uuids)))
                    for draw in range(self.draws_per_product)]
        first_draw = {}
        for draw, raw in enumerate(combined):
            first_draw.setdefault(raw, draw)
        frequency = Counter(combined)
        from spectraretro.fusion_reranker.integration import rank_generated
        fused = rank_generated(product, [self.rerank_observations[uid] for uid in uuids], outputs)
        ranked = [(c['candidate'], c['frequency']) for c in fused['candidates']]
        fused_by_raw = {c['candidate']: c for c in fused['candidates']}
        candidates = []
        for raw, count in ranked:
            if "MISSING" in raw or not raw.strip():
                continue
            components = raw.split(".")
            try:
                precursors = [fill_openbabel_hydrogens(component) for component in components]
            except ValueError:
                continue
            candidates.append({
                "precursors": precursors,
                "score": fused_by_raw[raw]['score'],
                "source": "spectraretro_live_stage1_gated_3d",
                "metadata": {"raw_precursors": components, "draw_frequency": count,
                             "ranking_version": fused['ranking_version'],
                             "score_definition": fused['score_definition'],
                             "fusion_score": fused_by_raw[raw]['fusion_score'],
                             "fusion_rank": fused_by_raw[raw]['rank'],
                             "frequency_score": count / self.draws_per_product,
                             "draws": self.draws_per_product, "valid_draws": valid_draws,
                             "sampling_steps": self.steps, "uuid_group": uuids,
                             "stage1_hits": stage1_hits, "stage1_samples": 300,
                             "stage1_misses": 300 - stage1_hits,
                             "stage1_gate_sha256": self.stage1_sha256,
                             "stage1_policy": "use the recorded Stage 1 count as the per-product draw limit",
                             "conditioning": "stored_synthon_anchor_reference_derived_3d",
                             "size_policy": "frozen_learned_categorical_draws",
                             "checkpoint_sha256": self.checkpoint_sha256,
                             "size_draws_sha256": self.size_sha256,
                             "bond_perception": decoder_by_draw[first_draw[raw]],
                             "geometry": [{"coords": job["coords"],
                                           "elements": job["elements"],
                                           "known_atoms": job["known_atoms"]}
                                          for job in coords_by_draw[first_draw[raw]]],
                             "planner_postprocess": "neutral_bracket_implicit_h_fill"},
            })
            if len(candidates) >= top_k:
                break
        self.progress(stage="decoded", fraction=0.88, current_product=product)
        return candidates

def run_job(request: dict, progress=None) -> dict:
    started = time.time()
    target = canonical_smiles(request["product"])
    draws = int(request.get("draws", 300))
    steps = int(request.get("steps", 100))
    predictor = LiveSpectraRetro(draws, steps, int(request.get("seed", 0)), progress)
    if target not in predictor.products:
        raise ValueError("This target is not in the 826-product SpectraRetro condition catalog")
    stock_file = ROOT / "agentic_retrosynthesis/data/geometry20k_hydrogen_filled_expandable_stock.smi"
    stock = {line for line in stock_file.read_text(encoding="utf-8").splitlines()
             if line and not line.startswith("#")}
    snapshot = SpectraRetroSnapshotModel(
        predictor.snapshot / "server_run/evaluation_edm_3d/merged_result.csv",
        stage1_counts=predictor.stage1_path, repair_openbabel=True)
    top_k = int(request.get("top_k", 10))
    initial_candidates = predictor.predict(target, top_k)
    memo = {target: initial_candidates}

    class LiveThenHistory:
        def predict(self, product, top_k):
            normalized = canonical_smiles(product)
            if normalized in memo:
                return memo[normalized][:top_k]
            if normalized in predictor.products:
                candidates = predictor.predict(normalized, top_k)
                memo[normalized] = candidates
                return candidates
            return snapshot.predict(product, top_k)

    max_calls = int(request.get("max_model_calls", 3))
    max_depth = int(request.get("max_depth", 3))
    ranker, llm_fallback = configured_llm()
    planning_model = LiveThenHistory()
    if llm_fallback:
        planning_model = CascadeModel(planning_model, llm_fallback)
    planner = Planner(planning_model, stock, ranker, max_depth=max_depth,
                      max_model_calls=max_calls, expansion_top_k=top_k,
                      max_routes=10, max_states=1500)
    result = planner.plan(target).to_dict()
    stage1_hits = predictor.stage1_by_product[target]
    result.update({"mode": "live_model", "seconds": round(time.time() - started, 2),
                   "one_step_candidates": initial_candidates,
                   "draws_per_product": draws, "sampling_steps": steps,
                   "conditions": "stored_reference_derived_synthon_anchor_3d",
                   "stage1": {"available": True, "hits": stage1_hits, "samples": 300,
                              "misses": 300 - stage1_hits, "passed": stage1_hits > 0,
                              "stage2_draws": min(draws, stage1_hits),
                              "requested_draws": draws,
                              "policy": "use the recorded Stage 1 count as the per-product draw limit",
                              "gate_sha256": predictor.stage1_sha256,
                              "model": predictor.stage1_manifest["identity"],
                              "official_test_top1": predictor.stage1_manifest["official_test_top1"]},
                   "terminal_set": "reference_derived_excluding_expandable_products",
                   "llm_policy": "openai_compatible" if ranker else "model_frequency",
                   "llm_step_fallback": bool(llm_fallback),
                   "checkpoint_sha256": predictor.checkpoint_sha256,
                   "size_draws_sha256": predictor.size_sha256})
    if progress:
        progress(stage="complete", fraction=1.0)
    return result


def main():
    request = json.load(sys.stdin)
    def progress(**details):
        print(json.dumps({"type": "progress", **details}, ensure_ascii=False), flush=True)
    try:
        result = run_job(request, progress)
        print(json.dumps({"type": "result", "result": result}, ensure_ascii=False), flush=True)
    except Exception as exc:
        print(json.dumps({"type": "error", "error": f"{type(exc).__name__}: {exc}"},
                         ensure_ascii=False), flush=True)
        raise


if __name__ == "__main__":
    main()
