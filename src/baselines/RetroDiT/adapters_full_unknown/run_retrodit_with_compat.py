#!/usr/bin/env python3
"""Launch the released RetroDiT entrypoint with narrow runtime compatibility fixes.

The ICML 2026 source returns human-readable ``"hits/total=value"`` strings
from validation and then passes them to Lightning's numeric logger.  Lightning
2.5 correctly rejects those strings.  For the single-GPU validation path used
here, request the already-supported numeric return representation instead.
No model, optimizer, loss, data, ranking, or sampling operation is changed.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch

# Import the released entrypoint first so its normal module initialization order
# is preserved; importing ResultStore directly triggers the repository's
# src.models <-> sampling_store circular import.
from src.train import main
# Preserve the package initialization order used by Hydra when it resolves the
# released ``src.models.RetroGenerativeModule`` target.  Importing
# sampling_store before src.models makes sampling_store import forward_syn
# while src.models.__init__ is still incomplete, exposing an author-side
# circular import.
import src.models  # noqa: F401,E402
from src.utils.sampling_store import ResultStore


_released_eval = ResultStore.eval


def _numeric_eval_for_lightning(self, samples=None, rank_temp_file=None, is_str_form=True):
    if rank_temp_file is None:
        is_str_form = False
    return _released_eval(
        self,
        samples=samples,
        rank_temp_file=rank_temp_file,
        is_str_form=is_str_form,
    )


ResultStore.eval = _numeric_eval_for_lightning
torch.set_float32_matmul_precision("high")

event_log = os.environ.get("RETRODIT_COMPAT_EVENT_LOG")
if event_log:
    path = Path(event_log)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "event": "runtime_compatibility",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "validation_metric_strings_converted_to_released_numeric_form": True,
                    "single_gpu_only": True,
                    "model_changed": False,
                    "optimizer_changed": False,
                    "loss_changed": False,
                    "sampling_changed": False,
                    "ranking_changed": False,
                    "float32_matmul_precision": "high",
                }
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
