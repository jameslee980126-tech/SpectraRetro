from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def _add_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", choices=("50k", "full"), default="50k")
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--valid-path", required=True)
    parser.add_argument("--test-path", required=True)
    parser.add_argument("--verbose", type=int, default=1)


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[256, 256, 256, 256, 256, 256],
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-worker", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--short-cut", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--batch-norm", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--concat-hidden", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--readout", choices=("sum", "mean"), default="sum")
    parser.add_argument(
        "--objective",
        choices=("single_positive", "multi_positive", "multi_positive_margin"),
        default="single_positive",
    )
    parser.add_argument("--ranking-loss-weight", type=float, default=0.0)
    parser.add_argument("--ranking-margin", type=float, default=1.0)


def _add_export_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--center-top-k", type=int, nargs="+", default=[3, 5])
    parser.add_argument("--strict-root-top-k", type=int, nargs="+", default=[3, 5])
    parser.add_argument("--reference-root-limit", type=int, default=10)
    parser.add_argument(
        "--export-policies",
        choices=("reference", "strict_root"),
        nargs="+",
        default=["reference", "strict_root"],
    )
    parser.add_argument("--temperature", type=float, nargs="+", default=[1.0])
    parser.add_argument(
        "--root-aggregation",
        choices=(
            "current",
            "split_edge",
            "reciprocal_rank",
            "probability_rank_hybrid",
        ),
        nargs="+",
        default=["current"],
    )
    parser.add_argument("--prediction-prefix", default="torchdrug_rgcn")
    parser.add_argument(
        "--export-splits",
        choices=("valid", "test"),
        nargs="+",
        default=["valid"],
    )


def _add_selection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ranking-eval-interval", type=int, default=1)
    parser.add_argument("--selection-center-top-k", type=int, default=5)
    parser.add_argument("--selection-root-limit", type=int, default=5)
    parser.add_argument("--selection-temperature", type=float, default=1.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="torchdrug-pred",
        description="Native TorchDrug reaction-center prediction for Retro",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess = subparsers.add_parser("preprocess")
    preprocess.add_argument("--input-csv", required=True)
    preprocess.add_argument("--output-path", required=True)
    preprocess.add_argument("--dataset", choices=("50k", "full"), default="50k")
    preprocess.add_argument("--verbose", type=int, default=1)
    preprocess.add_argument(
        "--objective",
        choices=("single_positive", "multi_positive", "multi_positive_margin"),
        default="single_positive",
    )

    train = subparsers.add_parser("train")
    _add_dataset_arguments(train)
    _add_model_arguments(train)
    _add_selection_arguments(train)
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--output-dir", required=True)

    evaluate = subparsers.add_parser("evaluate")
    _add_dataset_arguments(evaluate)
    _add_model_arguments(evaluate)
    evaluate.add_argument(
        "--evaluation-splits",
        choices=("valid", "test"),
        nargs="+",
        default=["valid"],
    )
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--output-dir", required=True)

    export = subparsers.add_parser("export")
    _add_dataset_arguments(export)
    _add_model_arguments(export)
    _add_export_arguments(export)
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--output-dir", required=True)

    pipeline = subparsers.add_parser("train_eval_export")
    _add_dataset_arguments(pipeline)
    _add_model_arguments(pipeline)
    _add_export_arguments(pipeline)
    _add_selection_arguments(pipeline)
    pipeline.add_argument("--epochs", type=int, default=20)
    pipeline.add_argument("--output-dir", required=True)
    return parser


def _load_solver(args: argparse.Namespace) -> tuple[Any, Any, Any, Any]:
    from torchdrug_pred.training import build_solver, load_datasets, set_seed

    set_seed(args.seed)
    train_set, valid_set, test_set = load_datasets(
        args.train_path,
        args.valid_path,
        args.test_path,
        args.dataset,
        args.verbose,
        args.objective,
    )
    solver = build_solver(
        train_set,
        valid_set,
        test_set,
        args.hidden_dims,
        args.learning_rate,
        args.batch_size,
        args.num_worker,
        args.gpu,
        args.short_cut,
        args.batch_norm,
        args.dropout,
        args.concat_hidden,
        args.readout,
        args.objective,
        args.ranking_loss_weight,
        args.ranking_margin,
    )
    return solver, train_set, valid_set, test_set


def _config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: value
        for key, value in vars(args).items()
        if key not in {"command"}
    }


def _preprocess(args: argparse.Namespace) -> None:
    from torchdrug_pred.dataset import USPTOReactionCenter

    dataset = USPTOReactionCenter(
        args.input_csv,
        dataset=args.dataset,
        verbose=args.verbose,
        objective=args.objective,
    )
    dataset.save_processed(args.output_path)
    print(
        f"saved={args.output_path} retained={len(dataset)} "
        f"valid_rate={dataset.valid_rate:.6f}"
    )


def _train(args: argparse.Namespace) -> None:
    from torchdrug_pred.training import train_with_validation

    solver, _, valid_set, _ = _load_solver(args)
    summary = train_with_validation(
        solver,
        args.epochs,
        args.output_dir,
        _config(args),
        valid_set,
        args.ranking_eval_interval,
        args.selection_center_top_k,
        args.selection_root_limit,
        args.selection_temperature,
    )
    print(
        f"best_epoch={summary['best_epoch']} "
        f"best_validation_accuracy={summary['best_validation_accuracy']:.6f}"
    )


def _evaluate(args: argparse.Namespace) -> None:
    from torchdrug_pred.io import write_json
    from torchdrug_pred.training import evaluate_solver

    solver, _, _, _ = _load_solver(args)
    solver.load(args.checkpoint)
    output_dir = Path(args.output_dir) / "metrics"
    for split in args.evaluation_splits:
        metrics = evaluate_solver(solver, split)
        filename = "validation.json" if split == "valid" else "test.json"
        write_json(output_dir / filename, metrics)
    print(f"saved_metrics={output_dir}")


def _export_loaded_splits(
    args: argparse.Namespace,
    solver: Any,
    valid_set: Any,
    test_set: Any,
) -> dict[str, Any]:
    from torchdrug_pred.export import collect_predictions, export_prediction_sets
    from torchdrug_pred.io import write_json

    maximum_center_top_k = max(args.center_top_k)
    datasets = {"valid": valid_set, "test": test_set}
    root_manifest: dict[str, Any] = {}
    for split in args.export_splits:
        dataset = datasets[split]
        predictions = collect_predictions(
            solver.model,
            dataset,
            maximum_center_top_k,
            args.batch_size,
            args.num_worker,
        )
        split_output = Path(args.output_dir) / "predictions" / split
        manifest = export_prediction_sets(
            predictions,
            dataset.smiles_list,
            split_output,
            args.prediction_prefix,
            args.center_top_k,
            args.strict_root_top_k,
            args.reference_root_limit,
            args.temperature,
            split=split,
            metrics_dir=Path(args.output_dir) / "metrics" / "predictions" / split,
            aggregation_methods=args.root_aggregation,
            export_policies=args.export_policies,
        )
        root_manifest[split] = {
            "split": split,
            "index_path": str(split_output / "index.json"),
            "retained_samples": len(dataset),
            "prediction_sets": len(manifest),
        }
    write_json(Path(args.output_dir) / "predictions" / "index.json", root_manifest)
    return root_manifest


def _export(args: argparse.Namespace, checkpoint: str | None = None) -> None:
    from torchdrug_pred.io import write_json

    solver, _, valid_set, test_set = _load_solver(args)
    checkpoint_path = args.checkpoint if checkpoint is None else checkpoint
    export_config = _config(args)
    export_config["checkpoint"] = checkpoint_path
    write_json(Path(args.output_dir) / "config.json", export_config)
    solver.load(checkpoint_path)
    manifest = _export_loaded_splits(args, solver, valid_set, test_set)
    print(
        f"exported_splits={len(manifest)} "
        f"exported_prediction_sets={sum(v['prediction_sets'] for v in manifest.values())}"
    )


def _train_eval_export(args: argparse.Namespace) -> None:
    from torchdrug_pred.io import write_json
    from torchdrug_pred.training import evaluate_solver, train_with_validation

    solver, _, valid_set, test_set = _load_solver(args)
    summary = train_with_validation(
        solver,
        args.epochs,
        args.output_dir,
        _config(args),
        valid_set,
        args.ranking_eval_interval,
        args.selection_center_top_k,
        args.selection_root_limit,
        args.selection_temperature,
    )
    solver.load(summary["best_checkpoint"])
    metrics_dir = Path(args.output_dir) / "metrics"
    for split in args.export_splits:
        filename = "validation.json" if split == "valid" else "test.json"
        write_json(metrics_dir / filename, evaluate_solver(solver, split))
    manifest = _export_loaded_splits(args, solver, valid_set, test_set)
    print(
        f"best_epoch={summary['best_epoch']} "
        f"exported_splits={len(manifest)} "
        f"exported_prediction_sets={sum(v['prediction_sets'] for v in manifest.values())}"
    )


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "preprocess":
        _preprocess(args)
    elif args.command == "train":
        _train(args)
    elif args.command == "evaluate":
        _evaluate(args)
    elif args.command == "export":
        _export(args)
    elif args.command == "train_eval_export":
        _train_eval_export(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
