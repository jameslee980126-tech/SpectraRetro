"""Budgeted training wrapper for RetroDiT's official RC predictor.

It calls the author's released TorchDrug model unchanged, but evaluates the
released root-ranking selection metric after every epoch and stops after a
configurable number of non-improving epochs.  Test data is not evaluated here;
use the official exporter with the saved ``best.pt`` only after selection.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def better(
    candidate: tuple[float, float, float],
    incumbent: tuple[float, float, float],
    min_delta: float,
) -> bool:
    if candidate[0] > incumbent[0] + min_delta:
        return True
    return abs(candidate[0] - incumbent[0]) <= min_delta and candidate[1:] > incumbent[1:]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--valid-path", required=True)
    parser.add_argument("--test-path", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset", choices=("50k", "full"), default="50k")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256] * 6)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-worker", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--selection-center-top-k", type=int, default=5)
    parser.add_argument("--selection-root-limit", type=int, default=5)
    parser.add_argument("--selection-temperature", type=float, default=1.0)
    parser.add_argument("--objective", default="single_positive")
    parser.add_argument("--verbose", type=int, default=1)
    args = parser.parse_args()

    if args.epochs <= 0 or args.patience <= 0 or args.min_delta < 0:
        raise ValueError("epochs/patience must be positive and min-delta non-negative")

    from torchdrug_pred.io import write_json
    from torchdrug_pred.training import (
        _root_selection_key,
        _selection_accuracy,
        build_solver,
        evaluate_candidate_ranking,
        evaluate_solver,
        load_datasets,
        set_seed,
    )

    set_seed(args.seed)
    train_set, valid_set, test_set = load_datasets(
        args.train_path, args.valid_path, args.test_path, args.dataset, args.verbose, args.objective
    )
    solver = build_solver(
        train_set, valid_set, test_set, args.hidden_dims, args.learning_rate,
        args.batch_size, args.num_worker, args.gpu, objective=args.objective,
    )

    output_dir = args.output_dir
    checkpoint_dir = output_dir / "checkpoints"
    metrics_dir = output_dir / "metrics"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["output_dir"] = str(output_dir)
    write_json(output_dir / "config.json", config)

    best_key = (float("-inf"), float("-inf"), float("-inf"))
    best_epoch = 0
    best_validation_accuracy = float("-inf")
    wait = 0
    history: list[dict[str, object]] = []
    for epoch in range(1, args.epochs + 1):
        solver.train(num_epoch=1)
        validation = evaluate_solver(solver, "valid")
        ranking = evaluate_candidate_ranking(
            solver, valid_set, args.batch_size, args.num_worker,
            args.selection_center_top_k, args.selection_root_limit, args.selection_temperature,
        )
        selection_key = _root_selection_key(ranking)
        record: dict[str, object] = {
            "epoch": epoch,
            "validation": validation,
            "native_validation_accuracy": _selection_accuracy(validation),
            "candidate_ranking": ranking,
            "selection_key": list(selection_key),
        }
        solver.save(checkpoint_dir / "last.pt")
        if better(selection_key, best_key, args.min_delta):
            best_key = selection_key
            best_epoch = epoch
            best_validation_accuracy = float(record["native_validation_accuracy"])
            wait = 0
            solver.save(checkpoint_dir / "best.pt")
            write_json(metrics_dir / "validation_best.json", record)
            record["improved"] = True
        else:
            wait += 1
            record["improved"] = False
            record["epochs_without_improvement"] = wait
        history.append(record)
        write_json(metrics_dir / "history.json", history)
        if wait >= args.patience:
            history[-1]["early_stopped"] = True
            write_json(metrics_dir / "history.json", history)
            break

    summary = {
        "max_epochs": args.epochs,
        "epochs_ran": len(history),
        "early_stopped": len(history) < args.epochs,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_validation_accuracy,
        "selection_metric": ["root_hit_at_5", "root_mrr", "root_hit_at_3"],
        "best_selection_key": list(best_key),
        "best_checkpoint": str(checkpoint_dir / "best.pt"),
        "last_checkpoint": str(checkpoint_dir / "last.pt"),
    }
    write_json(metrics_dir / "training_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
