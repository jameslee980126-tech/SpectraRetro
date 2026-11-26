from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from torchdrug_pred.dataset import USPTOReactionCenter
from torchdrug_pred.io import write_json


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_datasets(
    train_path: str,
    valid_path: str,
    test_path: str,
    dataset: str,
    verbose: int,
    objective: str = "single_positive",
) -> tuple[USPTOReactionCenter, USPTOReactionCenter, USPTOReactionCenter]:
    train_set = USPTOReactionCenter(
        train_path, dataset=dataset, verbose=verbose, objective=objective
    )
    valid_set = USPTOReactionCenter(
        valid_path, dataset=dataset, verbose=verbose, objective=objective
    )
    test_set = USPTOReactionCenter(
        test_path, dataset=dataset, verbose=verbose, objective=objective
    )
    return train_set, valid_set, test_set


def build_solver(
    train_set: USPTOReactionCenter,
    valid_set: USPTOReactionCenter,
    test_set: USPTOReactionCenter,
    hidden_dims: Sequence[int],
    learning_rate: float,
    batch_size: int,
    num_worker: int,
    gpu: int,
    short_cut: bool = False,
    batch_norm: bool = False,
    dropout: float = 0.0,
    concat_hidden: bool = True,
    readout: str = "sum",
    objective: str = "single_positive",
    ranking_loss_weight: float = 0.0,
    ranking_margin: float = 1.0,
) -> Any:
    from torchdrug import core, tasks

    from torchdrug_pred.model import DropoutRGCN

    model = DropoutRGCN(
        input_dim=train_set.node_feature_dim,
        hidden_dims=list(hidden_dims),
        num_relation=train_set.num_bond_type,
        short_cut=short_cut,
        batch_norm=batch_norm,
        dropout=dropout,
        concat_hidden=concat_hidden,
        readout=readout,
    )
    if objective == "single_positive":
        task = tasks.CenterIdentification(
            model,
            feature=("graph", "atom", "bond"),
        )
    elif objective in {"multi_positive", "multi_positive_margin"}:
        from torchdrug_pred.task import build_multi_positive_task

        task = build_multi_positive_task(
            model,
            ranking_loss_weight=(
                ranking_loss_weight
                if objective == "multi_positive_margin"
                else 0.0
            ),
            ranking_margin=ranking_margin,
        )
    else:
        raise ValueError(f"Unsupported objective: {objective}")
    optimizer = torch.optim.Adam(task.parameters(), lr=learning_rate)
    gpus = [gpu] if gpu >= 0 and torch.cuda.is_available() else None
    return core.Engine(
        task,
        train_set,
        valid_set,
        test_set,
        optimizer,
        gpus=gpus,
        batch_size=batch_size,
        num_worker=num_worker,
    )


def to_jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def evaluate_solver(solver: Any, split: str) -> dict[str, Any]:
    return to_jsonable(solver.evaluate(split))


def _selection_accuracy(metrics: dict[str, Any]) -> float:
    exact_keys = ("accuracy", "center accuracy")
    for key in exact_keys:
        if key in metrics:
            return float(metrics[key])
    candidates = [
        float(value)
        for key, value in metrics.items()
        if "accuracy" in key.lower() and isinstance(value, (int, float))
    ]
    if not candidates:
        raise KeyError(f"No accuracy metric found in {sorted(metrics)}")
    return max(candidates)


def _root_selection_key(metrics: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(metrics["root_hit_at_5"]),
        float(metrics["root_mrr"]),
        float(metrics["root_hit_at_3"]),
    )


def evaluate_candidate_ranking(
    solver: Any,
    dataset: USPTOReactionCenter,
    batch_size: int,
    num_worker: int,
    center_top_k: int = 5,
    root_limit: int = 5,
    temperature: float = 1.0,
) -> dict[str, Any]:
    from torchdrug_pred.export import calculate_ranking_metrics, collect_predictions

    predictions = collect_predictions(
        solver.model,
        dataset,
        center_top_k,
        batch_size,
        num_worker,
    )
    metrics = calculate_ranking_metrics(
        predictions,
        center_top_k,
        temperature,
        root_limit,
    )
    return dict(metrics)


def train_with_validation(
    solver: Any,
    epochs: int,
    output_dir: str | Path,
    config: dict[str, Any],
    validation_set: USPTOReactionCenter | None = None,
    ranking_eval_interval: int = 1,
    selection_center_top_k: int = 5,
    selection_root_limit: int = 5,
    selection_temperature: float = 1.0,
) -> dict[str, Any]:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    experiment_dir = Path(output_dir)
    checkpoint_dir = experiment_dir / "checkpoints"
    metrics_dir = experiment_dir / "metrics"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    write_json(experiment_dir / "config.json", config)
    if validation_set is None:
        raise ValueError("validation_set is required for root-metric selection")
    if ranking_eval_interval <= 0:
        raise ValueError("ranking_eval_interval must be positive")
    best_key = (float("-inf"), float("-inf"), float("-inf"))
    best_accuracy = float("-inf")
    best_ranking: dict[str, Any] | None = None
    best_epoch = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        solver.train(num_epoch=1)
        validation = evaluate_solver(solver, "valid")
        accuracy = _selection_accuracy(validation)
        record = {
            "epoch": epoch,
            "validation": validation,
            "native_validation_accuracy": accuracy,
        }
        evaluate_ranking = (
            epoch == 1
            or epoch % ranking_eval_interval == 0
            or epoch == epochs
        )
        if evaluate_ranking:
            ranking = evaluate_candidate_ranking(
                solver,
                validation_set,
                int(config["batch_size"]),
                int(config["num_worker"]),
                selection_center_top_k,
                selection_root_limit,
                selection_temperature,
            )
            selection_key = _root_selection_key(ranking)
            record["candidate_ranking"] = ranking
            record["selection_key"] = list(selection_key)
        history.append(record)
        write_json(metrics_dir / "history.json", history)
        solver.save(checkpoint_dir / "last.pt")
        if evaluate_ranking and selection_key > best_key:
            best_key = selection_key
            best_accuracy = accuracy
            best_ranking = ranking
            best_epoch = epoch
            solver.save(checkpoint_dir / "best.pt")
            write_json(metrics_dir / "validation_best.json", record)
    summary = {
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_accuracy,
        "selection_metric": ["root_hit_at_5", "root_mrr", "root_hit_at_3"],
        "best_selection_key": list(best_key),
        "best_validation_root_metrics": best_ranking,
        "ranking_eval_interval": ranking_eval_interval,
        "best_checkpoint": str(checkpoint_dir / "best.pt"),
        "last_checkpoint": str(checkpoint_dir / "last.pt"),
    }
    write_json(metrics_dir / "training_summary.json", summary)
    return summary
