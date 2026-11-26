#!/usr/bin/env python3
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def write_event(path, payload):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def read_top1(summary_path):
    with summary_path.open(encoding="utf-8") as handle:
        handle.readline()
        line = handle.readline().strip()
    if not line.startswith("top 1:"):
        raise ValueError(f"invalid GLN validation summary: {summary_path}")
    return float(line.rsplit(maxsplit=1)[-1])


def stop_process(process):
    if process.poll() is not None:
        return process.returncode
    os.killpg(process.pid, signal.SIGTERM)
    os.killpg(process.pid, signal.SIGCONT)
    try:
        return process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        return process.wait(timeout=30)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--python-exe", required=True)
    parser.add_argument("--validation-script", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--event-log", type=Path, required=True)
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument("--warmup-validations", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=0.003)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--max-checkpoints", type=int, default=10)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    args, runner_args = parser.parse_known_args()
    if runner_args and runner_args[0] == "--":
        runner_args = runner_args[1:]

    args.event_log.parent.mkdir(parents=True, exist_ok=True)
    args.selection_json.parent.mkdir(parents=True, exist_ok=True)
    write_event(
        args.event_log,
        {
            "event": "early_stop_monitor_started",
            "metric": "validation_native_top1",
            "warmup_validations": args.warmup_validations,
            "min_delta": args.min_delta,
            "patience": args.patience,
            "model_source_changed": False,
        },
    )

    workdir = args.runner.parents[2]
    process = subprocess.Popen(
        [args.python_exe, str(args.runner), *runner_args],
        cwd=workdir,
        start_new_session=True,
    )
    best_metric = None
    best_index = None
    significant_best = None
    stalled = 0
    validation_count = 0
    checkpoint_index = 0
    early_stopped = False
    process_paused = False
    exit_wait_started = None

    try:
        while checkpoint_index < args.max_checkpoints:
            checkpoint = args.checkpoint_dir / f"model-{checkpoint_index}.dump"
            model_path = checkpoint / "model.dump"
            args_path = checkpoint / "args.pkl"
            summary_path = args.checkpoint_dir / f"val-{checkpoint_index}.summary"
            if model_path.is_file() and args_path.is_file():
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGSTOP)
                    process_paused = True
                    time.sleep(1)
                subprocess.run(
                    ["bash", str(args.validation_script), str(checkpoint_index)],
                    cwd=workdir,
                    check=True,
                )
                metric = read_top1(summary_path)
                validation_count += 1
                if best_metric is None or metric > best_metric:
                    best_metric = metric
                    best_index = checkpoint_index
                gain = metric if significant_best is None else metric - significant_best
                improved = significant_best is None or gain >= args.min_delta
                if improved:
                    significant_best = metric
                eligible = validation_count > args.warmup_validations
                stalled = 0 if improved or not eligible else stalled + 1
                payload = {
                    "event": "validation_checked",
                    "checkpoint_index": checkpoint_index,
                    "validation_count": validation_count,
                    "metric": "validation_native_top1",
                    "value": metric,
                    "best_metric": best_metric,
                    "best_checkpoint_index": best_index,
                    "significant_best": significant_best,
                    "gain": gain,
                    "improved": improved,
                    "early_stop_eligible": eligible,
                    "stalled": stalled,
                }
                write_event(args.event_log, payload)
                args.selection_json.write_text(
                    json.dumps(
                        {
                            "checkpoint_index": best_index,
                            "validation_top1": best_metric,
                            "significant_best": significant_best,
                            "validation_count": validation_count,
                            "stalled": stalled,
                            "early_stopped": False,
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                checkpoint_index += 1
                if eligible and stalled >= args.patience:
                    early_stopped = True
                    write_event(
                        args.event_log,
                        {
                            "event": "early_stop",
                            "checkpoint_index": checkpoint_index - 1,
                            "best_checkpoint_index": best_index,
                            "best_metric": best_metric,
                        },
                    )
                    stop_process(process)
                    process_paused = False
                    break
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGCONT)
                    process_paused = False
                exit_wait_started = None
                continue

            returncode = process.poll()
            if returncode is not None:
                if returncode != 0:
                    raise RuntimeError(f"GLN training exited with status {returncode}")
                if exit_wait_started is None:
                    exit_wait_started = time.monotonic()
                elif time.monotonic() - exit_wait_started >= max(10.0, args.poll_seconds * 2):
                    break
            time.sleep(args.poll_seconds)
    finally:
        if process_paused and process.poll() is None:
            os.killpg(process.pid, signal.SIGCONT)
        if process.poll() is None and not early_stopped:
            stop_process(process)

    if best_index is None or best_metric is None:
        raise RuntimeError("GLN training produced no validated checkpoints")
    args.selection_json.write_text(
        json.dumps(
            {
                "checkpoint_index": best_index,
                "validation_top1": best_metric,
                "significant_best": significant_best,
                "validation_count": validation_count,
                "stalled": stalled,
                "early_stopped": early_stopped,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_event(
        args.event_log,
        {
            "event": "training_complete",
            "best_checkpoint_index": best_index,
            "best_metric": best_metric,
            "early_stopped": early_stopped,
        },
    )


if __name__ == "__main__":
    main()
