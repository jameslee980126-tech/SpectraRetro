#!/usr/bin/env python3
"""Run one frozen DiffER seed with validation-only early stopping.

The upstream training entrypoint remains responsible for model construction,
data loading, objective, optimization, and validation.  This process launches
it with a fixed RNG seed, snapshots its otherwise-overwritten model after each
completed validation event, and stops the isolated child process group only
after the frozen plateau rule is met.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
DIFFER_ROOT = ROOT / "third_party" / "benchmark_20260919" / "DiffER"
METRIC_RE = re.compile(r"Epoch - (\d+)\.[^|]*\|.*?mol_accuracy - ([0-9.eE+-]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--min-delta", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--min-validations", type=int, default=5)
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


def replace_hardlink(source: Path, destination: Path) -> None:
    """Atomically retain an identical checkpoint without duplicating its data."""
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    os.link(source, temporary)
    os.replace(temporary, destination)


def metrics(path: Path) -> list[tuple[int, float]]:
    text = path.read_text(errors="replace") if path.exists() else ""
    return [(int(epoch) + 1, float(metric)) for epoch, metric in METRIC_RE.findall(text)]


def wait_for_new_model(path: Path, previous_mtime_ns: int, process: subprocess.Popen) -> int:
    deadline = time.time() + 180
    while time.time() < deadline:
        if path.exists():
            stat = path.stat()
            if stat.st_size > 0 and stat.st_mtime_ns > previous_mtime_ns:
                first = stat.st_size
                time.sleep(2)
                second_stat = path.stat()
                if second_stat.st_size == first and second_stat.st_mtime_ns == stat.st_mtime_ns:
                    return stat.st_mtime_ns
        if process.poll() is not None:
            raise RuntimeError("DiffER exited before publishing the checkpoint for a completed validation")
        time.sleep(1)
    raise TimeoutError("DiffER checkpoint did not update within 180 seconds of validation")


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
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(range(10, 18)))
    args.config = args.config.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for directory in ("models", "metrics", "losses"):
        (DIFFER_ROOT / "out" / directory).mkdir(parents=True, exist_ok=True)
    model_path = DIFFER_ROOT / "out" / "models" / f"{args.name}.pkl"
    metric_path = DIFFER_ROOT / "out" / "metrics" / f"{args.name}_metrics_log.txt"
    loss_path = DIFFER_ROOT / "out" / "losses" / f"{args.name}_losses.npy"
    for path in (model_path, metric_path, loss_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing upstream artifact: {path}")

    code = (
        "import random,runpy,numpy as np,torch; "
        f"random.seed({args.seed}); np.random.seed({args.seed}); "
        f"torch.manual_seed({args.seed}); torch.cuda.manual_seed_all({args.seed}); "
        "runpy.run_path('train.py',run_name='__main__')"
    )
    command = [
        sys.executable,
        "-c",
        code,
        "--name",
        args.name,
        "--config_path",
        str(args.config),
    ]
    environment = os.environ.copy()
    environment.update(PYTHONHASHSEED=str(args.seed), PYTHONUNBUFFERED="1")
    process = subprocess.Popen(
        command,
        cwd=DIFFER_ROOT,
        env=environment,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    (args.output_dir / "training.pid").write_text(f"{process.pid}\n")

    state = {
        "seed": args.seed,
        "name": args.name,
        "pid": process.pid,
        "status": "running",
        "test_data_used": False,
        "policy": {
            "metric": "mol_accuracy",
            "mode": "max",
            "min_delta": args.min_delta,
            "patience": args.patience,
            "min_validations": args.min_validations,
        },
    }
    atomic_json(args.output_dir / "state.json", state)
    event_count = 0
    # ``min_delta`` governs plateau detection only.  Checkpoint selection uses
    # the numerically best validation metric even when the gain is smaller
    # than ``min_delta``.
    plateau_reference_metric = float("-inf")
    best_metric = float("-inf")
    best_epoch = None
    bad_events = 0
    model_mtime_ns = 0
    stop_reason = "upstream_finished"
    started = time.time()

    try:
        while True:
            observed = metrics(metric_path)
            while event_count < len(observed):
                epoch, metric = observed[event_count]
                model_mtime_ns = wait_for_new_model(model_path, model_mtime_ns, process)
                snapshot = args.output_dir / f"epoch_{epoch:03d}.pkl"
                temporary = snapshot.with_suffix(".pkl.tmp")
                shutil.copy2(model_path, temporary)
                os.replace(temporary, snapshot)
                payload = torch.load(snapshot, map_location="cpu", weights_only=True)
                if not isinstance(payload, dict) or not payload:
                    raise RuntimeError(f"invalid DiffER state dict: {snapshot}")

                if metric > best_metric:
                    best_metric = metric
                    best_epoch = epoch
                    best_path = args.output_dir / "best.pkl"
                    replace_hardlink(snapshot, best_path)
                if metric >= plateau_reference_metric + args.min_delta:
                    plateau_reference_metric = metric
                    bad_events = 0
                else:
                    bad_events += 1
                event_count += 1
                state.update(
                    completed_validations=event_count,
                    latest_epoch=epoch,
                    latest_metric=metric,
                    latest_snapshot=str(snapshot),
                    latest_snapshot_sha256=sha256(snapshot),
                    best_metric=best_metric,
                    best_epoch=best_epoch,
                    early_stop_reference_metric=plateau_reference_metric,
                    bad_events=bad_events,
                    elapsed_seconds=time.time() - started,
                )
                atomic_json(args.output_dir / "state.json", state)
                print(json.dumps({"event": "validation", **state}, sort_keys=True), flush=True)

                # best.pkl keeps its own hard link to the selected inode.  Only
                # the latest numbered snapshot is needed to construct last.pkl,
                # so older numbered copies are redundant and can be unlinked.
                for old_snapshot in args.output_dir.glob("epoch_*.pkl"):
                    if old_snapshot != snapshot:
                        old_snapshot.unlink()

                if event_count >= args.min_validations and bad_events >= args.patience:
                    stop_reason = "validation_plateau"
                    terminate_group(process)
                    break
            if stop_reason == "validation_plateau":
                break
            return_code = process.poll()
            if return_code is not None:
                if return_code != 0:
                    raise RuntimeError(f"upstream DiffER exited with code {return_code}")
                break
            time.sleep(args.poll_seconds)
    except BaseException:
        terminate_group(process)
        state.update(status="failed", elapsed_seconds=time.time() - started)
        atomic_json(args.output_dir / "state.json", state)
        raise

    if best_epoch is None:
        raise RuntimeError("DiffER produced no completed validation/checkpoint pair")
    last_path = args.output_dir / "last.pkl"
    last_source = args.output_dir / f"epoch_{state['latest_epoch']:03d}.pkl"
    replace_hardlink(last_source, last_path)
    last_source.unlink()
    best_hash = sha256(args.output_dir / "best.pkl")
    last_hash = sha256(last_path)
    summary = {
        "seed": args.seed,
        "status": "complete",
        "stop_reason": stop_reason,
        "completed_epochs": state["latest_epoch"],
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "checkpoint_selection": "absolute_best_validation_mol_accuracy",
        "early_stop_reference_metric": plateau_reference_metric,
        "best_checkpoint": str(args.output_dir / "best.pkl"),
        "best_checkpoint_sha256": best_hash,
        "last_checkpoint": str(last_path),
        "last_checkpoint_sha256": last_hash,
        "test_data_used": False,
        "elapsed_seconds": time.time() - started,
    }
    state.update(latest_snapshot=str(last_path), latest_snapshot_sha256=last_hash)
    state.update(summary)
    atomic_json(args.output_dir / "state.json", state)
    atomic_json(args.output_dir / "training_complete.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
