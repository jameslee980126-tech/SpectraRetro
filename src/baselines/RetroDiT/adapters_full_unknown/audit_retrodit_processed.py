#!/usr/bin/env python3
"""Audit RetroDiT's serialized PyG datasets without changing them."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch


def raw_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.reader(handle)) - 1


def summarize_counts(values: np.ndarray) -> dict[str, float | int]:
    return {
        "min": int(values.min()),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": int(values.max()),
        "mean": float(values.mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    processed_root = args.data_root / "processed" / "ndummy_30_maxlen_250_cano"
    result: dict[str, object] = {
        "data_root": str(args.data_root.resolve()),
        "processed_root": str(processed_root.resolve()),
        "max_n_len": 250,
        "n_dummy": 30,
        "splits": {},
    }

    for split in ("train", "val", "test"):
        data, slices = torch.load(processed_root / f"{split}.pt", weights_only=False)
        first_key = next(iter(slices))
        processed_rows = int(slices[first_key].numel() - 1)
        node_slice_key = (
            "x"
            if "x" in slices
            else "r_x"
            if "r_x" in slices
            else "r_X"
            if "r_X" in slices
            else None
        )
        if node_slice_key is None:
            raise RuntimeError(
                f"The {split} cache has no node-feature slices; keys={list(slices)}"
            )
        node_counts = torch.diff(slices[node_slice_key]).cpu().numpy()
        raw_path = args.data_root / "raw" / f"uspto50k_{split}.csv"
        original_rows = raw_rows(raw_path)
        permutation_path = args.data_root / f"permutations_{split}.json"
        with permutation_path.open("r", encoding="utf-8") as handle:
            permutations = json.load(handle)
        permutation_lengths = {
            key: len(value) for key, value in permutations.items()
        }
        result["splits"][split] = {
            "raw_rows": original_rows,
            "processed_rows": processed_rows,
            "dropped_rows": original_rows - processed_rows,
            "retention_rate": processed_rows / original_rows,
            "permutation_types": sorted(permutations),
            "permutation_rows_by_type": permutation_lengths,
            "permutation_matches_raw_rows": all(
                length == original_rows for length in permutation_lengths.values()
            ),
            "node_counts": summarize_counts(node_counts),
            "cache_bytes": (processed_root / f"{split}.pt").stat().st_size,
            "data_keys": sorted(data.keys()),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
