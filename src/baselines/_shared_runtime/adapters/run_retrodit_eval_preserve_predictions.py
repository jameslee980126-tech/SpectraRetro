#!/usr/bin/env python3
"""Run released RetroDiT evaluation while preserving raw ranked candidates.

The released distributed result store intentionally drops the ``gen`` field
when serializing per-rank samples.  That is sufficient for its native metric,
but prevents the study-wide canonical precursor evaluator from scoring the
same generation run.  This repository-external entrypoint changes only result
serialization: model loading, diffusion sampling, candidate order, and native
metric computation remain in the released implementation.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch

# Import the released evaluation entrypoint first to preserve its package
# initialization order (the author repository otherwise exposes a circular
# import between src.models and sampling_store).
from src.eval import main
import src.models  # noqa: F401,E402
from src.utils.sampling_store import ResultStore


def _save_results_with_raw_predictions(self, to_log, write_path):
    with open(write_path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(to_log) + "\n")
        for sample in self.samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")


def _get_samples_with_raw_predictions(self):
    return list(self.samples)


ResultStore.save_results = _save_results_with_raw_predictions
ResultStore.get_samples = _get_samples_with_raw_predictions
torch.set_float32_matmul_precision("high")

event_log = os.environ.get("RETRODIT_COMPAT_EVENT_LOG")
if event_log:
    path = Path(event_log)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "event": "evaluation_serialization_compatibility",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "raw_ranked_predictions_preserved": True,
                    "model_changed": False,
                    "sampling_changed": False,
                    "ranking_changed": False,
                    "native_metric_changed": False,
                    "float32_matmul_precision": "high",
                }
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
