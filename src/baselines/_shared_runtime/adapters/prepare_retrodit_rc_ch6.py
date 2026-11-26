#!/usr/bin/env python3
"""Preprocess CH6 data with RetroDiT's released reaction-center dataset.

The released USPTO-50K loader hard-codes an atom-count difference of 10.  CH6
contains valid frozen reactions with differences up to 18, so this adapter only
widens dataset-shape limits.  Reaction-center labels and all model behavior are
still computed by the released ``USPTOReactionCenter`` implementation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from torchdrug_pred.dataset import USPTOReactionCenter


class CH6ReactionCenter(USPTOReactionCenter):
    def __init__(
        self,
        path: str | Path,
        *,
        verbose: int,
        objective: str,
        max_atoms: int,
        max_atom_difference: int,
    ) -> None:
        if objective not in {
            "single_positive",
            "multi_positive",
            "multi_positive_margin",
        }:
            raise ValueError(f"Unsupported objective: {objective}")
        self.path = str(path)
        self.dataset = "50k"
        self.objective = objective
        self.max_atoms = max_atoms
        self.max_atom_difference = max_atom_difference
        self._load_csv(path, verbose)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--objective", default="single_positive")
    parser.add_argument("--max-atoms", type=int, default=250)
    parser.add_argument("--max-atom-difference", type=int, default=30)
    args = parser.parse_args()

    dataset = CH6ReactionCenter(
        args.input_csv,
        verbose=args.verbose,
        objective=args.objective,
        max_atoms=args.max_atoms,
        max_atom_difference=args.max_atom_difference,
    )
    dataset.save_processed(args.output_path)
    print(
        f"saved={args.output_path} retained={len(dataset)} "
        f"valid_rate={dataset.valid_rate:.6f} max_atoms={dataset.max_atoms} "
        f"max_atom_difference={dataset.max_atom_difference}"
    )


if __name__ == "__main__":
    main()
