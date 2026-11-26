#!/usr/bin/env python3
"""Run one Retroformer seed with frozen validation-only early stopping."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "third_party" / "benchmark_20260919" / "Retroformer"
STEP_RE = re.compile(r"Checkpoint saved to (.*?/model_(\d+)\.pt)")
METRIC_RE = re.compile(r"Validation accuracy: ([0-9.]+) - ([0-9.]+) - ([0-9.]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--intermediate-dir", type=Path, required=True)
    parser.add_argument("--min-delta", type=float, default=0.005)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--min-validations", type=int, default=5)
    parser.add_argument("--max-step", type=int, default=100000)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    return parser.parse_args()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def completed_events(path: Path) -> list[tuple[Path, int, float]]:
    text = path.read_text(errors="replace") if path.exists() else ""
    checkpoints = [(Path(name), int(step)) for name, step in STEP_RE.findall(text)]
    token_metrics = [float(token) for _, _, token in METRIC_RE.findall(text)]
    return [(checkpoint, step, metric) for (checkpoint, step), metric in zip(checkpoints, token_metrics)]


def stable_checkpoint(path: Path) -> dict:
    for _ in range(40):
        if path.exists() and path.stat().st_size > 0:
            first = path.stat().st_size
            time.sleep(2)
            if path.exists() and path.stat().st_size == first:
                payload = torch.load(path, map_location="cpu", weights_only=False)
                if not isinstance(payload, dict) or not {"model", "optim", "step"}.issubset(payload):
                    raise RuntimeError(f"invalid Retroformer checkpoint: {path}")
                return payload
        time.sleep(1)
    raise TimeoutError(f"checkpoint did not stabilize: {path}")


def replace_hardlink(source: Path, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    os.link(source, temporary)
    os.replace(temporary, destination)


def terminate_group(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=20)


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.data_dir = args.data_dir.resolve()
    args.intermediate_dir = args.intermediate_dir.resolve()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_base = args.output_dir / "checkpoint"
    checkpoint_actual = args.output_dir / "checkpoint_untyped"
    log_path = args.output_dir / "upstream.log"

    command = [
        sys.executable, "-m", "retroformer.train",
        "--encoder_num_layers", "8", "--decoder_num_layers", "8", "--heads", "8",
        "--max_step", str(args.max_step), "--batch_size_trn", "4", "--batch_size_val", "4",
        "--batch_size_token", "4096", "--save_per_step", "2500", "--val_per_step", "2500",
        "--report_per_step", "200", "--device", "cuda", "--known_class", "False",
        "--shared_vocab", "True", "--shared_encoder", "False",
        "--data_dir", str(args.data_dir), "--intermediate_dir", str(args.intermediate_dir),
        "--checkpoint_dir", str(checkpoint_base), "--seed", str(args.seed),
    ]
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(PROJECT), PYTHONHASHSEED=str(args.seed), PYTHONUNBUFFERED="1")
    with log_path.open("ab", buffering=0) as handle:
        process = subprocess.Popen(
            command, cwd=PROJECT, env=environment, stdin=subprocess.DEVNULL,
            stdout=handle, stderr=subprocess.STDOUT, start_new_session=True,
        )
    (args.output_dir / "training.pid").write_text(f"{process.pid}\n")
    state = {
        "seed": args.seed,
        "pid": process.pid,
        "status": "running",
        "test_data_used": False,
        "policy": {
            "metric": "validation_token_accuracy", "mode": "max",
            "min_delta": args.min_delta, "patience": args.patience,
            "min_validations": args.min_validations, "hard_max_step": args.max_step,
        },
    }
    atomic_json(args.output_dir / "state.json", state)
    event_count = 0
    meaningful_best = float("-inf")
    absolute_best = float("-inf")
    meaningful_best_step = None
    absolute_best_step = None
    bad_events = 0
    retained_sources: set[Path] = set()
    started = time.time()
    stop_reason = "upstream_finished"

    try:
        while True:
            events = completed_events(log_path)
            while event_count < len(events):
                checkpoint, step, metric = events[event_count]
                payload = stable_checkpoint(checkpoint)
                if int(payload["step"]) != step:
                    raise RuntimeError(f"checkpoint step mismatch: expected {step}, got {payload['step']}")

                last_path = args.output_dir / "last.pt"
                replace_hardlink(checkpoint, last_path)
                if metric > absolute_best:
                    absolute_best = metric
                    absolute_best_step = step
                    replace_hardlink(checkpoint, args.output_dir / "best.pt")
                if metric >= meaningful_best + args.min_delta:
                    meaningful_best = metric
                    meaningful_best_step = step
                    bad_events = 0
                else:
                    bad_events += 1
                event_count += 1
                state.update(
                    completed_validations=event_count, latest_step=step, latest_metric=metric,
                    latest_checkpoint_sha256=sha256(checkpoint), absolute_best=absolute_best,
                    absolute_best_step=absolute_best_step, meaningful_best=meaningful_best,
                    meaningful_best_step=meaningful_best_step, bad_events=bad_events,
                    elapsed_seconds=time.time() - started,
                )
                atomic_json(args.output_dir / "state.json", state)
                print(json.dumps({"event": "validation", **state}, sort_keys=True), flush=True)

                # best.pt and last.pt are hard links, so old numbered files are
                # redundant. Keep at most the just-published source until the
                # next event, bounding checkpoint disk use without losing state.
                retained_sources.add(checkpoint)
                for old in list(retained_sources):
                    if old != checkpoint and old.exists():
                        old.unlink()
                        retained_sources.remove(old)

                if event_count >= args.min_validations and bad_events >= args.patience:
                    stop_reason = "validation_plateau"
                    terminate_group(process)
                    break
            if stop_reason == "validation_plateau":
                break
            return_code = process.poll()
            if return_code is not None:
                if return_code != 0:
                    raise RuntimeError(f"upstream Retroformer exited with code {return_code}")
                break
            time.sleep(args.poll_seconds)
    except BaseException:
        terminate_group(process)
        state.update(status="failed", elapsed_seconds=time.time() - started)
        atomic_json(args.output_dir / "state.json", state)
        raise

    if absolute_best_step is None:
        raise RuntimeError("Retroformer produced no completed validation/checkpoint pair")
    summary = {
        "seed": args.seed, "status": "complete", "stop_reason": stop_reason,
        "completed_validations": event_count, "latest_step": state["latest_step"],
        "absolute_best_metric": absolute_best, "absolute_best_step": absolute_best_step,
        "meaningful_best_metric": meaningful_best, "meaningful_best_step": meaningful_best_step,
        "best_checkpoint": str(args.output_dir / "best.pt"),
        "best_checkpoint_sha256": sha256(args.output_dir / "best.pt"),
        "last_checkpoint": str(args.output_dir / "last.pt"),
        "last_checkpoint_sha256": sha256(args.output_dir / "last.pt"),
        "test_data_used": False, "elapsed_seconds": time.time() - started,
    }
    state.update(summary)
    atomic_json(args.output_dir / "state.json", state)
    atomic_json(args.output_dir / "training_complete.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
