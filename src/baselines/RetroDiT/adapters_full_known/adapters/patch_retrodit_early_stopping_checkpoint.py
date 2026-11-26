#!/usr/bin/env python3
"""Copy a Lightning checkpoint while changing only EarlyStopping patience."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--patience", type=int, required=True)
    args = parser.parse_args()

    if args.patience < 1:
        raise ValueError("patience must be positive")

    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)
    changed = []
    for key, state in checkpoint.get("callbacks", {}).items():
        if "EarlyStopping" not in str(key):
            continue
        previous = state.get("patience")
        state["patience"] = args.patience
        changed.append((str(key), previous, state.get("wait_count")))

    if not changed:
        raise RuntimeError("EarlyStopping callback state was not found")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, args.output)

    print(f"input={args.input}")
    print(f"output={args.output}")
    print(f"global_step={checkpoint.get('global_step')} epoch={checkpoint.get('epoch')}")
    for key, previous, wait_count in changed:
        print(
            f"callback={key} patience={previous}->{args.patience} "
            f"wait_count={wait_count}"
        )


if __name__ == "__main__":
    main()
