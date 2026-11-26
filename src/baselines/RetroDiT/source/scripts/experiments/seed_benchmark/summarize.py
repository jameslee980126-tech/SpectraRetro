#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 42])
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    per_seed: dict[str, dict[str, float]] = {}
    for seed in args.seeds:
        path = args.benchmark_root / "retro" / f"seed_{seed}" / "sample" / "generated_metrics.json"
        with path.open() as handle:
            metrics = json.load(handle)
        per_seed[str(seed)] = {
            key: float(value)
            for key, value in metrics.items()
            if key.startswith("deduplicate_top_") and key.endswith("_accuracy")
        }

    metric_names = sorted(
        set.intersection(*(set(metrics) for metrics in per_seed.values())),
        key=lambda name: int(name.split("_")[2]),
    )
    summary = {
        "seeds": args.seeds,
        "std_definition": "sample standard deviation (n-1 denominator)",
        "per_seed": per_seed,
        "aggregate": {
            name: {
                "mean": statistics.mean(per_seed[str(seed)][name] for seed in args.seeds),
                "std": statistics.stdev(per_seed[str(seed)][name] for seed in args.seeds),
            }
            for name in metric_names
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
