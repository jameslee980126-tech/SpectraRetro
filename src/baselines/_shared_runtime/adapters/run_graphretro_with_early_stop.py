#!/usr/bin/env python3
import argparse
import json
import math
import runpy
import sys
from pathlib import Path

import torch

from seq_graph_retro.models.core_edits import single_edit as single_edit_module
from seq_graph_retro.models.trainer import Trainer
from seq_graph_retro.utils import metrics as graphretro_metrics


def append_event(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=json_scalar, sort_keys=True) + "\n")


def json_scalar(value):
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def metric_gain(metric_name, previous, current):
    if previous is None or not math.isfinite(previous):
        return math.inf
    if "loss" in metric_name:
        return previous - current
    return current - previous


def scheduler_learning_rates(scheduler):
    optimizer = getattr(scheduler, "optimizer", None)
    param_groups = getattr(optimizer, "param_groups", [])
    return [float(group["lr"]) for group in param_groups]


def get_accuracy_edits_compat(edit_logits, labels):
    accuracy = torch.tensor(0.0)
    for logits, label in zip(edit_logits, labels):
        target = label.item() if label.numel() == 1 else torch.argmax(label).item()
        if torch.argmax(logits).item() == target:
            accuracy += 1.0
    return accuracy / len(labels)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--event-log", type=Path, required=True)
    parser.add_argument("--warmup-validations", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=0.003)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--require-lr-reduction", action="store_true")
    args, runner_args = parser.parse_known_args()

    graphretro_metrics.get_accuracy_edits = get_accuracy_edits_compat
    single_edit_module.get_accuracy_edits = get_accuracy_edits_compat
    append_event(
        args.event_log,
        {
            "event": "runtime_compatibility",
            "detail": (
                "Preserve the released scalar-or-argmax accuracy logic without "
                "depending on the exception type raised by Tensor.item()."
            ),
            "model_source_changed": False,
        },
    )

    original_train_epochs = Trainer.train_epochs

    def train_epochs_with_early_stop(self, train_data, eval_data, epochs=10, **kwargs):
        stalled = 0
        significant_best = self.best_metric
        initial_learning_rates = scheduler_learning_rates(self.scheduler)
        append_event(
            args.event_log,
            {
                "event": "early_stop_monitor_started",
                "maximum_epochs": epochs,
                "metric": self.eval_metric,
                "warmup_validations": args.warmup_validations,
                "min_delta": args.min_delta,
                "patience": args.patience,
                "require_lr_reduction": args.require_lr_reduction,
                "initial_learning_rates": initial_learning_rates,
            },
        )
        for epoch_index in range(1, epochs + 1):
            original_train_epochs(self, train_data, eval_data, epochs=1, **kwargs)
            current_best = self.best_metric
            gain = metric_gain(self.eval_metric, significant_best, current_best)
            improved = gain >= args.min_delta
            if improved:
                significant_best = current_best
                stalled = 0
            current_learning_rates = scheduler_learning_rates(self.scheduler)
            lr_reduced = any(
                current < initial
                for current, initial in zip(current_learning_rates, initial_learning_rates)
            )
            early_stop_eligible = (
                epoch_index > args.warmup_validations
                and (not args.require_lr_reduction or lr_reduced)
            )
            if early_stop_eligible and not improved:
                stalled += 1
            append_event(
                args.event_log,
                {
                    "event": "validation_checked",
                    "epoch": epoch_index,
                    "metric": self.eval_metric,
                    "best_metric": current_best,
                    "significant_best": significant_best,
                    "gain": gain,
                    "improved": improved,
                    "stalled": stalled,
                    "learning_rates": current_learning_rates,
                    "lr_reduced": lr_reduced,
                    "early_stop_eligible": early_stop_eligible,
                },
            )
            if early_stop_eligible and stalled >= args.patience:
                append_event(
                    args.event_log,
                    {
                        "event": "early_stop",
                        "epoch": epoch_index,
                        "selected_checkpoint": "best_model.pt",
                        "best_metric": current_best,
                    },
                )
                return

    Trainer.train_epochs = train_epochs_with_early_stop
    sys.argv = [str(args.runner), *runner_args]
    runpy.run_path(str(args.runner), run_name="__main__")


if __name__ == "__main__":
    main()
