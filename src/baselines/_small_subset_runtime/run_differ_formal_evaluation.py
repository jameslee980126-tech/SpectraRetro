#!/usr/bin/env python3
"""Run and verify one frozen DiffER seed's complete formal test workflow.

Generation is delegated unchanged to the released ``eval.py`` entrypoint.  This
wrapper supplies the frozen best checkpoint, records process and artifact
provenance, converts the native sample JSON without changing its first raw
slots, and invokes the common 2,400-reaction Top-k evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).resolve().parent
DIFFER_ROOT = ROOT / "third_party" / "benchmark_20260919" / "DiffER"
FROZEN = ROOT / "experiments" / "stage1_similarity_rebalanced_20260912" / "frozen"
EXPECTED_TEST_SHA256 = "D7F01A2D78656C37A698C4F9F9E64DEC8F9EF3DEEACAA0CA48BDCEFC8D77B614"
EXPECTED_INDEX_SHA256 = "B525DA479CD8208366CD43F3D76AFCCFCADF42B385BD8F63230A9E4D77A8C810"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--training-complete", type=Path)
    parser.add_argument("--config", type=Path, default=EXPERIMENT / "differ_formal.yaml")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--native-batch-size",
        type=int,
        default=64,
        help="Inference micro-batch before the released evaluator expands each item into 20 samples.",
    )
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.native_batch_size < 1:
        parser.error("--native-batch-size must be positive")
    return args


def verify_inputs(args: argparse.Namespace) -> dict:
    seed_dir = EXPERIMENT / "formal" / "differ" / f"seed{args.seed}"
    checkpoint = (args.checkpoint or seed_dir / "best.pkl").resolve()
    training_complete = (args.training_complete or seed_dir / "training_complete.json").resolve()
    config = args.config.resolve()
    required = {
        "checkpoint": checkpoint,
        "training_complete": training_complete,
        "config": config,
        "upstream_eval": DIFFER_ROOT / "eval.py",
        "converter": EXPERIMENT / "convert_differ_predictions.py",
        "evaluator": EXPERIMENT / "evaluate_full_retro_topk.py",
        "test_csv": FROZEN / "test.csv",
        "test_index": FROZEN / "test_index.csv",
    }
    missing = [f"{name}={path}" for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing formal evaluation inputs: " + ", ".join(missing))
    if sha256(required["test_csv"]).upper() != EXPECTED_TEST_SHA256:
        raise RuntimeError("frozen test.csv SHA-256 mismatch")
    if sha256(required["test_index"]).upper() != EXPECTED_INDEX_SHA256:
        raise RuntimeError("frozen test_index.csv SHA-256 mismatch")

    summary = json.loads(training_complete.read_text(encoding="utf-8"))
    if summary.get("status") != "complete" or int(summary.get("seed", -1)) != args.seed:
        raise RuntimeError("training completion evidence is absent or belongs to another seed")
    checkpoint_hash = sha256(checkpoint)
    if checkpoint_hash != str(summary.get("best_checkpoint_sha256", "")).lower():
        raise RuntimeError("best checkpoint hash does not match training completion evidence")
    if bool(summary.get("test_data_used", True)):
        raise RuntimeError("training completion evidence indicates test data use")
    return {
        "seed": args.seed,
        "status": "passed",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "training_complete": str(training_complete),
        "training_complete_sha256": sha256(training_complete),
        "config": str(config),
        "config_sha256": sha256(config),
        "test_csv": str(required["test_csv"]),
        "test_csv_sha256": EXPECTED_TEST_SHA256.lower(),
        "test_index": str(required["test_index"]),
        "test_index_sha256": EXPECTED_INDEX_SHA256.lower(),
        "test_data_used_for_checkpoint_or_protocol_selection": False,
    }


def run_checked(command: list[str], *, cwd: Path, environment: dict, log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
        (log_path.parent / (log_path.stem + ".pid")).write_text(f"{process.pid}\n")
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"command failed with exit code {return_code}: {command}")


def main() -> None:
    args = parse_args()
    evidence = verify_inputs(args)
    if args.preflight_only:
        print(json.dumps(evidence, sort_keys=True))
        return

    output_dir = (args.output_dir or EXPERIMENT / "formal" / "evaluation" / "differ" / f"seed{args.seed}").resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty formal output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "preflight.json", evidence)

    native_name = f"full_retro_differ_seed{args.seed}_test"
    native_samples = DIFFER_ROOT / "out" / "samples" / f"{native_name}_samples.json"
    if native_samples.exists():
        raise FileExistsError(f"refusing to overwrite upstream native samples: {native_samples}")
    native_samples.parent.mkdir(parents=True, exist_ok=True)

    environment = os.environ.copy()
    environment.update(
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        PYTHONHASHSEED=str(args.seed),
        PYTHONUNBUFFERED="1",
    )
    started = time.time()
    state = {
        **evidence,
        "status": "generating",
        "native_name": native_name,
        "native_sample_budget": {
            "augmentations": 20,
            "samples_per_input": 20,
            "inference_micro_batch_size": args.native_batch_size,
        },
        "started_at_unix": started,
    }
    atomic_json(output_dir / "state.json", state)

    generation_command = [
        sys.executable,
        "eval.py",
        "--name",
        native_name,
        "--config_path",
        evidence["config"],
        "--load",
        evidence["checkpoint"],
        "--num_samples",
        "20",
        "--batch_size",
        str(args.native_batch_size),
        "--test",
    ]
    run_checked(
        generation_command,
        cwd=DIFFER_ROOT,
        environment=environment,
        log_path=output_dir / "generation.log",
    )
    if not native_samples.is_file() or native_samples.stat().st_size == 0:
        raise RuntimeError("released DiffER evaluator did not publish native samples")

    predictions = output_dir / "predictions.jsonl"
    conversion_manifest = output_dir / "conversion_manifest.json"
    run_checked(
        [
            sys.executable,
            str(EXPERIMENT / "convert_differ_predictions.py"),
            "--native-samples-json",
            str(native_samples),
            "--test-csv",
            evidence["test_csv"],
            "--output-jsonl",
            str(predictions),
            "--manifest-json",
            str(conversion_manifest),
            "--expected-rows",
            "2400",
            "--beam-size",
            "10",
            "--augmentations",
            "20",
            "--samples-per-input",
            "20",
        ],
        cwd=EXPERIMENT,
        environment=environment,
        log_path=output_dir / "conversion.log",
    )

    metrics = output_dir / "metrics.json"
    per_reaction = output_dir / "per_reaction.csv"
    run_checked(
        [
            sys.executable,
            str(EXPERIMENT / "evaluate_full_retro_topk.py"),
            "--model-name",
            f"DiffER single seed {args.seed}",
            "--predictions-jsonl",
            str(predictions),
            "--test-csv",
            evidence["test_csv"],
            "--test-index",
            evidence["test_index"],
            "--expected-test-sha256",
            EXPECTED_TEST_SHA256,
            "--expected-index-sha256",
            EXPECTED_INDEX_SHA256,
            "--expected-rows",
            "2400",
            "--output-json",
            str(metrics),
            "--per-reaction-csv",
            str(per_reaction),
            "--beam-size",
            "10",
            "--candidate-ranking",
            "first_raw_sampling_slots_in_released_eval_json_order",
        ],
        cwd=EXPERIMENT,
        environment=environment,
        log_path=output_dir / "scoring.log",
    )
    parsed_metrics = json.loads(metrics.read_text(encoding="utf-8"))
    unified = parsed_metrics.get("unified", {})
    if unified.get("total_reactions") != 2400 or unified.get("prediction_rows_present") != 2400:
        raise RuntimeError("formal metrics do not cover all 2,400 frozen test reactions")
    conversion = json.loads(conversion_manifest.read_text(encoding="utf-8"))
    if conversion.get("status") != "passed" or conversion.get("source_rows") != 2400:
        raise RuntimeError("DiffER conversion verification failed")

    complete = {
        **evidence,
        "status": "complete",
        "native_samples": str(native_samples),
        "native_samples_sha256": sha256(native_samples),
        "predictions_jsonl": str(predictions),
        "predictions_jsonl_sha256": sha256(predictions),
        "conversion_manifest": str(conversion_manifest),
        "conversion_manifest_sha256": sha256(conversion_manifest),
        "metrics": str(metrics),
        "metrics_sha256": sha256(metrics),
        "per_reaction": str(per_reaction),
        "per_reaction_sha256": sha256(per_reaction),
        "elapsed_seconds": time.time() - started,
        "test_labels_used_only_for_post_generation_scoring": True,
    }
    atomic_json(output_dir / "state.json", complete)
    atomic_json(output_dir / "EVALUATION_COMPLETE.json", complete)
    print(json.dumps(complete, sort_keys=True))


if __name__ == "__main__":
    main()
