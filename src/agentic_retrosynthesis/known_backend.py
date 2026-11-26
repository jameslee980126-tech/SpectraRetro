"""Agent class hypotheses dispatch isolated, product-only known-class inference."""
import itertools
import json
import os
from pathlib import Path
import subprocess

from .known_contract import CLASSES, checked_class
from .planner import canonical_smiles

ROOT = Path(os.environ.get("SPECTRARETRO_MODEL_WORKSPACE", str(Path(__file__).resolve().parents[2]))).resolve()


class AgentKnownClassBackend:
    def __init__(self, output, advisor, steps=100, max_centers=10, max_classes=2):
        if advisor is None or not callable(getattr(advisor, "propose_classes", None)):
            raise ValueError("Known-class Agent mode requires a class proposal advisor")
        if type(max_classes) is not int or not 1 <= max_classes <= 3:
            raise ValueError("max_classes must be 1..3")
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.advisor, self.steps, self.max_centers, self.max_classes = advisor, steps, max_centers, max_classes
        self.prepared, self.dispatch, self.trace = {}, {}, []
        self.calls = 0

    def _save(self):
        (self.output / "class_trace.json").write_text(json.dumps(self.trace, indent=2, ensure_ascii=False), encoding="utf-8")

    def _worker(self, request):
        self.calls += 1
        folder = self.output / f"worker_{self.calls:04d}"
        folder.mkdir(exist_ok=False)
        inp, out = folder / "input.json", folder / "output.json"
        inp.write_text(json.dumps(dict(request, steps=self.steps, max_centers=self.max_centers)), encoding="utf-8")
        with (folder / "worker.log").open("w", encoding="utf-8") as log:
            subprocess.run([os.environ.get("SPECTRARETRO_PYTHON", __import__("sys").executable), str(Path(__file__).resolve().parent / "known_worker.py"),
                str(inp), str(out)], check=True, timeout=600, stdout=log, stderr=log, cwd=ROOT)
        return json.loads(out.read_text(encoding="utf-8"))

    def describe(self, product):
        product = canonical_smiles(product)
        if product in self.prepared:
            return self.prepared[product]
        event = dict(product=product, class_source="agent_hypothesis", reference_inputs=False)
        self.trace.append(event)
        try:
            proposals = self.advisor.propose_classes(product, self.max_classes)
            if not isinstance(proposals, list) or len(proposals) > self.max_classes:
                raise ValueError("Invalid class proposal list")
            seen, accepted = set(), []
            for p in proposals:
                cls = checked_class(p["class_id"])
                if cls in seen:
                    raise ValueError("Duplicate class hypothesis")
                seen.add(cls)
                reason = str(p.get("reason", ""))[:400]
                secret = getattr(self.advisor, "api_key", None)
                accepted.append(dict(class_id=cls, class_label=CLASSES[cls],
                                     reason=reason.replace(secret, "[redacted]") if secret else reason))
            event.update(status="proposed" if accepted else "no_supported_hypothesis", hypotheses=accepted)
        except Exception as exc:
            event.update(status="proposal_failed", error_type=type(exc).__name__)
            self.prepared[product] = []
            self._save()
            return []  # No true-label or forced-class fallback.
        groups = []
        for p in accepted:
            cls = p["class_id"]
            try:
                centers = self._worker(dict(action="describe", product=product, class_id=cls))
                mapped = []
                for center in centers:
                    local = center["center_id"]
                    if type(local) is not int or not 1 <= local <= self.max_centers:
                        raise ValueError("Invalid local center ID")
                    cid = cls * 10000 + local
                    self.dispatch[product, cid] = cls, local
                    mapped.append(dict(center, center_id=cid, class_id=cls, class_label=CLASSES[cls],
                                       class_source="agent_hypothesis", reaction_class_input=True))
                groups.append(mapped)
                p.update(status="centers_scored", supported_centers=len(mapped))
            except Exception as exc:
                p.update(status="model_failed", error_type=type(exc).__name__)
        # Give each proposed class a slot before second centers of the first class.
        result = [c for row in itertools.zip_longest(*groups) for c in row if c is not None]
        self.prepared[product] = result
        self._save()
        return result

    def complete(self, product, center_id, sizes, draws):
        product = canonical_smiles(product)
        self.describe(product)
        cls, local = self.dispatch[product, center_id]
        candidates = self._worker(dict(action="complete", product=product, class_id=cls,
                                       center_id=local, sizes=sizes, draws=draws))
        for candidate in candidates:
            candidate["source"] = "known_class_product_only_edm"
            candidate["metadata"].update(center_id=center_id, local_center_id=local, class_id=cls,
                class_label=CLASSES[cls], class_source="agent_hypothesis", reaction_class_input=True)
        return candidates
