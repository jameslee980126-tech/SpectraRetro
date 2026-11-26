"""Load the one-step predictor and coordinate decoder on demand for route planning."""
from __future__ import annotations

from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

from .planner import canonical_smiles

ROOT = Path(os.environ.get("SPECTRARETRO_MODEL_WORKSPACE", str(Path(__file__).resolve().parents[2]))).resolve()
EXP = ROOT / "experiments/stage2_geometry20k_20260915"
PROBE = EXP / "testset_literature_shortlist_20260921/paper_redesign/recursive_probe"


class ProductOnlyBackend:
    candidate_source = "spectraretro_product_only_edm"

    def conditioning_metadata(self):
        return {"reaction_class_input": False}

    def condition_records(self, records):
        return records

    def score_product(self, product, folder):
        r = self.r
        r.write(folder / "stage1_inputs.json", [{"id": folder.name, "smiles": product}])
        with (folder / "stage1.log").open("w", encoding="utf-8") as log:
            subprocess.run([str(r.S1), str(PROBE / "score_product.py"),
                            str(folder / "stage1_inputs.json"), str(folder / "stage1_scores.json")],
                           stdout=log, stderr=log, check=True, timeout=180, cwd=ROOT)
        return json.loads((folder / "stage1_scores.json").read_text())

    def __init__(self, output: Path, steps=100, max_centers=10):
        if not 1 <= steps <= 100 or not 1 <= max_centers <= 10:
            raise ValueError("Supported budgets: steps 1..100, centers 1..10")
        self.output = output.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.steps, self.max_centers = steps, max_centers
        self.prepared = {}
        self.loaded = False
        self.calls = 0

    def _load(self):
        if self.loaded:
            return
        spec = importlib.util.spec_from_file_location("spectra_product_helpers", PROBE / "run_recursive.py")
        r = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(r)
        if not r.torch.cuda.is_available():
            raise RuntimeError("Product-only EDM currently requires a CUDA GPU")
        freeze = json.loads((EXP / "server_results/evaluation_freeze_1500.json").read_text())
        self.weights = {"diffusion": r.sha(r.CHECKPOINT), "size": r.sha(r.SIZE_CHECKPOINT)}
        if (self.weights["diffusion"] != freeze["arms"]["edm_3d"]["best_sha256"]
                or self.weights["size"] != freeze["arms"]["size"]["best_sha256"]):
            raise RuntimeError("Frozen checkpoint hash mismatch")
        self.helper_hashes = {name: r.sha(PROBE / name) for name in
                              ("run_recursive.py", "score_product.py", "decode_candidates.py")}
        r.setup(2)
        self.model = r.create(3, r.torch.device("cuda"))
        self.model.load_state_dict(r.torch.load(r.CHECKPOINT, map_location="cpu", weights_only=False)["state_dict"], strict=True)
        self.model.eval()
        self.model.edm.T = self.steps
        self.size_model = r.create(3, r.torch.device("cuda"), size=True)
        self.size_model.load_state_dict(r.torch.load(r.SIZE_CHECKPOINT, map_location="cpu", weights_only=False)["state_dict"], strict=True)
        self.size_model.eval()
        self.r, self.loaded = r, True

    def describe(self, product):
        product = canonical_smiles(product)
        if product in self.prepared:
            return self.prepared[product]["public"]
        self._load()
        r = self.r
        import hashlib
        folder = self.output / hashlib.sha256(product.encode()).hexdigest()[:16]
        folder.mkdir(exist_ok=True)
        # Use an explicit timeout; Stage1 independently checks its frozen hash.
        scored = self.score_product(product, folder)
        self.weights["stage1"] = scored["checkpoint_sha256"]
        mol, optimization = r.embed_product(product)
        r.Chem.MolToMolFile(mol, str(folder / "product.mol"))
        supported, public, unsupported = {}, [], []
        for c in scored["nodes"][0]["candidates"][:self.max_centers]:
            built, reason = r.fragments(mol, c)
            if reason:
                unsupported.append({"center_id": c["rank"], "reason": reason})
                continue
            records, info = built
            records = self.condition_records(records)
            allowed = sorted(int(n) for n in self.size_model.linker_id2size)
            fragments = []
            for rec, desc in zip(records, info):
                with r.torch.no_grad():
                    logits, _ = self.size_model.forward(r.collate_with_fragment_edges([rec]), return_loss=False)
                p = logits.softmax(-1)[0].cpu().tolist()
                fragments.append({"atom_indices": desc["product_atom_indices"],
                                  "elements": desc["elements"],
                                  "anchor_atom": desc["product_atom_indices"][desc["anchor_local_index"]],
                                  "size_probabilities": {str(self.size_model.linker_id2size[i]): v for i, v in enumerate(p)}})
            item = {"center_id": c["rank"], "bond_atoms": c["atom_indices"],
                    "center_probability": c["probability"], "allowed_sizes": allowed, "fragments": fragments,
                    **self.conditioning_metadata()}
            public.append(item)
            supported[c["rank"]] = (records, item)
        self.prepared[product] = {"folder": folder, "public": public, "supported": supported}
        r.write(folder / "context.json", {"product": product, "centers": public, "unsupported": unsupported,
                "reference_inputs": False, **self.conditioning_metadata(), "weights": self.weights,
                "helper_hashes": self.helper_hashes, "optimization_status": optimization})
        return public

    def complete(self, product, center_id, sizes, draws):
        product = canonical_smiles(product)
        self.describe(product)
        r = self.r
        records, c = self.prepared[product]["supported"][center_id]
        if sizes is not None and (len(sizes) != 2 or any(type(n) is not int or n not in c["allowed_sizes"] for n in sizes)):
            raise ValueError("Unsupported size hypothesis")
        if not 1 <= draws <= 300:
            raise ValueError("Invalid draw budget")
        self.calls += 1
        folder = self.prepared[product]["folder"] / f"generation_{self.calls:04d}"
        folder.mkdir(exist_ok=False)
        seed = r.stable_seed(product + "|" + str(center_id) + "|" + json.dumps(sizes))
        r.seed(seed)
        sampled = []
        for frag in c["fragments"]:
            if sizes is None:
                probabilities = r.torch.tensor([frag["size_probabilities"][str(n)] for n in c["allowed_sizes"]])
                ids = r.torch.multinomial(probabilities, draws, replacement=True).tolist()
                sampled.append([c["allowed_sizes"][i] for i in ids])
            else:
                sampled.append([sizes[len(sampled)]] * draws)
        slots, xyz, types, lengths = {}, [], [], []
        jobs = [(d, j, sampled[j][d]) for d in range(draws) for j in range(2)]
        from src.utils import FoundNaNException
        for start in range(0, len(jobs), 12):
            chunk = jobs[start:start + 12]
            batch = r.collate([records[j] for d, j, n in chunk])
            counts = r.torch.tensor([n for d, j, n in chunk], device="cuda", dtype=r.torch.long)
            try:
                with r.torch.no_grad():
                    chain, mask = self.model.sample_chain(batch, sample_fn=lambda _: counts, keep_frames=1)
                if not r.torch.isfinite(chain).all():
                    raise FloatingPointError("Nonfinite diffusion output")
            except (FoundNaNException, FloatingPointError):
                for d, j, n in chunk:
                    slots[(d, j)] = None
                continue
            array = chain[0].detach().cpu().numpy()
            for k, ((d, j, n), count) in enumerate(zip(chunk, mask.sum((1, 2)).int().cpu().tolist())):
                slots[(d, j)] = len(lengths)
                xyz.append(array[k, :count, :3])
                types.append(array[k, :count, 3:].argmax(-1).astype(r.np.uint8))
                lengths.append(count)
        decoded = []
        if lengths:
            raw_path = folder / "raw_candidates.npz"
            r.np.savez_compressed(raw_path, xyz=r.np.concatenate(xyz), element=r.np.concatenate(types), length=r.np.array(lengths))
            with (folder / "decoder.log").open("w", encoding="utf-8") as log:
                subprocess.run([str(r.DECODER), str(PROBE / "decode_candidates.py"), str(raw_path),
                                str(folder / "decoded.json")], check=True, stdout=log, stderr=log,
                               timeout=180, cwd=ROOT)
            decoded = json.loads((folder / "decoded.json").read_text())["rows"]
        frequencies, details = Counter(), []
        for d in range(draws):
            pair = [decoded[slots[(d, j)]] if slots[(d, j)] is not None else
                    {"smiles": None, "reason": "numerical_failure"} for j in range(2)]
            details.append({"draw": d, "sizes": [sampled[j][d] for j in range(2)], "fragments": pair})
            if all(v["smiles"] for v in pair):
                frequencies[tuple(sorted(v["smiles"] for v in pair))] += 1
        metadata = {"center_id": center_id, "center_probability": c["center_probability"],
                    "draws": draws, "reference_inputs": False, **self.conditioning_metadata(),
                    "conditioning": "current_product_ETKDG_fragment_coordinates", "weights": self.weights,
                    "score_definition": "within_attempt_pair_frequency_not_reaction_feasibility",
                    "sampling_steps": self.steps, "seed": seed, "evidence_directory": str(folder),
                    "forward_check": {"status": "not_checked"}, "experimental_validation": False}
        from spectraretro.fusion_reranker.integration import product_fragments, rank_generated
        observations = product_fragments(product, c['bond_atoms'], c['fragments'])
        generated = [[d['fragments'][j]['smiles'] or 'MISSING' for d in details] for j in range(2)]
        condition = self.conditioning_metadata()
        ranked = rank_generated(product, observations, generated,
            class_id=condition.get('class_id') if condition.get('reaction_class_input') else None)
        metadata.update(ranking_version=ranked['ranking_version'],
                        score_definition=ranked['score_definition'], candidate_pool=ranked['candidate_pool'])
        candidates, seen = [], set()
        for candidate in ranked['candidates']:
            if 'MISSING' in candidate['candidate']:
                continue
            pair = tuple(sorted(candidate['parts']))
            if pair in seen:
                continue
            seen.add(pair)
            candidates.append({'precursors': list(pair), 'score': candidate['score'],
                'source': self.candidate_source, 'metadata': dict(metadata,
                    draw_frequency=frequencies[pair], frequency_score=frequencies[pair]/draws,
                    fusion_score=candidate['fusion_score'], fusion_rank=candidate['rank'])})
        r.write(folder / "result.json", {"product": product, "metadata": metadata,
                "requested_sizes": sizes, "draws": details, "candidates": candidates})
        return candidates
