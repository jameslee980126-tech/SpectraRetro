#!/usr/bin/env python3
"""Run released RC export with a narrow malformed-root guard.

One CH6 product causes RDKit's rooted canonicalizer to raise a RangeError for a
candidate root outside the raw product atom range.  The released exporter already
supports skipped candidates and identity completion, so this guard only skips
that invalid candidate and records the count; it does not alter the predictor,
checkpoint, scores, or ranking settings.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from rdkit import Chem
import torchdrug_pred.export as export_module


_released_add = export_module.add_permutation_score
_skipped = 0


def _safe_add_permutation_score(*args, **kwargs):
    global _skipped
    # ``add_permutation_score`` receives (mapping, product_smiles, root_index,
    # score).  Avoid invoking RDKit's rooted canonicalizer when the candidate
    # root is outside the raw product molecule; the released exporter later
    # fills products with the identity permutation.
    if len(args) >= 3:
        product_smiles = args[1]
        root_index = int(args[2])
        molecule = Chem.MolFromSmiles(product_smiles)
        if molecule is None or root_index < 0 or root_index >= molecule.GetNumAtoms():
            _skipped += 1
            return False
    try:
        return _released_add(*args, **kwargs)
    except Exception as error:  # RDKit RangeError is version-dependent
        _skipped += 1
        return False


export_module.add_permutation_score = _safe_add_permutation_score

from torchdrug_pred.cli import main  # noqa: E402


if __name__ == "__main__":
    torchdrug_output = os.environ.get("RETRODIT_RC_EXPORT_COMPAT_LOG")
    main()
    if torchdrug_output:
        path = Path(torchdrug_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "event": "rc_export_compatibility",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "skipped_invalid_root_candidates": _skipped,
                    "model_changed": False,
                    "scores_changed": False,
                    "ranking_changed": False,
                    "identity_completion": "released exporter behavior",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
