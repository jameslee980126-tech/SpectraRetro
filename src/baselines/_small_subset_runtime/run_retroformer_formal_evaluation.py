#!/usr/bin/env python3
"""Run and verify one Retroformer seed's complete formal test workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).resolve().parent
PROJECT = ROOT / "third_party" / "benchmark_20260919" / "Retroformer"
PREPARED = EXPERIMENT / "prepared" / "retroformer"
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


def link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite staged input: {destination}")
    os.link(source, destination)


def stage_compatible_vocabulary(
    base_vocabulary: Path, destination: Path, checkpoint_vocabulary_size: int
) -> dict:
    with base_vocabulary.open("rb") as handle:
        source_vocabulary, target_vocabulary = pickle.load(handle)
    if source_vocabulary != target_vocabulary:
        raise RuntimeError("formal Retroformer run requires the frozen shared vocabulary")
    if len(target_vocabulary) > checkpoint_vocabulary_size:
        raise RuntimeError("checkpoint vocabulary is smaller than the frozen LMDB vocabulary")
    padding_count = checkpoint_vocabulary_size - len(target_vocabulary)
    if padding_count != 2 or list(target_vocabulary[-2:]) != ["o", "s"]:
        raise RuntimeError("unexpected Retroformer checkpoint/base-vocabulary compatibility shape")
    # The training-time 87-entry vocabulary decoded its two never-targeted tail
    # classes as ``o`` and ``s``.  Keep those legal aliases at indices 85/86 so
    # an occasional unused-class beam token remains tokenisable, while the
    # actually trained LMDB IDs 0..84 retain their frozen semantics.
    padding_tokens = ["o", "s"]
    staged_source = list(source_vocabulary) + padding_tokens
    staged_target = list(target_vocabulary) + padding_tokens
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump((staged_source, staged_target), handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, destination)
    return {
        "base_vocabulary_size": len(target_vocabulary),
        "checkpoint_vocabulary_size": checkpoint_vocabulary_size,
        "padding_tokens": padding_tokens,
        "staged_vocabulary": str(destination),
        "staged_vocabulary_sha256": sha256(destination),
        "token_id_contract": "frozen LMDB token IDs 0..84 are unchanged; checkpoint-only IDs 85/86 retain their original legal o/s decode aliases",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--training-complete", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def verify_inputs(args: argparse.Namespace) -> dict:
    seed_dir = EXPERIMENT / "formal" / "retroformer" / f"seed{args.seed}"
    checkpoint = (args.checkpoint or seed_dir / "best.pt").resolve()
    training_complete = (args.training_complete or seed_dir / "training_complete.json").resolve()
    required = {
        "checkpoint": checkpoint,
        "training_complete": training_complete,
        "upstream_translate": PROJECT / "retroformer" / "translate.py",
        "data_test": PREPARED / "cooked_test.lmdb" / "data.mdb",
        "vocabulary": PREPARED / "intermediate" / "vocab_share.pk",
        "converter": EXPERIMENT / "convert_retroformer_predictions.py",
        "reference_contract": EXPERIMENT / "retroformer_reference_contract.json",
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
    reference_contract = json.loads(required["reference_contract"].read_text(encoding="utf-8"))
    if reference_contract.get("frozen_test_csv_sha256", "").lower() != EXPECTED_TEST_SHA256.lower():
        raise RuntimeError("Retroformer reference contract has the wrong frozen test hash")
    if reference_contract.get("retroformer_test_lmdb_data_sha256", "").lower() != sha256(required["data_test"]):
        raise RuntimeError("Retroformer reference contract has the wrong test LMDB hash")
    mismatch_rows = reference_contract.get("expected_mismatch_source_rows", [])
    if len(mismatch_rows) != 49 or len(set(mismatch_rows)) != 49:
        raise RuntimeError("Retroformer reference contract mismatch rows are invalid")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not {"model", "optim", "step"}.issubset(payload):
        raise RuntimeError("invalid full Retroformer checkpoint")
    expected_step = int(summary["absolute_best_step"])
    if int(payload["step"]) != expected_step:
        raise RuntimeError(f"best checkpoint step {payload['step']} != summary {expected_step}")
    with required["vocabulary"].open("rb") as handle:
        source_vocabulary, target_vocabulary = pickle.load(handle)
    generator_weight = payload["model"].get("generator.0.weight")
    if generator_weight is None:
        raise RuntimeError("checkpoint has no Retroformer generator weight")
    checkpoint_vocabulary_size = int(generator_weight.shape[0])
    if source_vocabulary != target_vocabulary:
        raise RuntimeError("prepared Retroformer vocabulary is not shared")
    if checkpoint_vocabulary_size < len(target_vocabulary):
        raise RuntimeError("checkpoint output dimension is smaller than the frozen vocabulary")
    return {
        "seed": args.seed,
        "status": "passed",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_step": expected_step,
        "training_complete": str(training_complete),
        "training_complete_sha256": sha256(training_complete),
        "data_dir": str(PREPARED.resolve()),
        "test_lmdb": str((PREPARED / "cooked_test.lmdb").resolve()),
        "test_lmdb_data_sha256": sha256(required["data_test"]),
        "reference_contract": str(required["reference_contract"].resolve()),
        "reference_contract_sha256": sha256(required["reference_contract"]),
        "vocabulary": str(required["vocabulary"].resolve()),
        "vocabulary_sha256": sha256(required["vocabulary"]),
        "source_vocabulary_size": len(source_vocabulary),
        "target_vocabulary_size": len(target_vocabulary),
        "checkpoint_vocabulary_size": checkpoint_vocabulary_size,
        "test_csv": str(required["test_csv"].resolve()),
        "test_csv_sha256": EXPECTED_TEST_SHA256.lower(),
        "test_index": str(required["test_index"].resolve()),
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

    output_dir = (args.output_dir or EXPERIMENT / "formal" / "evaluation" / "retroformer" / f"seed{args.seed}").resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty formal output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "preflight.json", evidence)

    work = output_dir / "work"
    checkpoint_dir = work / "checkpoint_untyped"
    intermediate_dir = work / "intermediate"
    result_dir = work / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
    link(Path(evidence["checkpoint"]), checkpoint_dir / "best.pt")
    staged_vocabulary = stage_compatible_vocabulary(
        Path(evidence["vocabulary"]),
        intermediate_dir / "vocab_share.pk",
        int(evidence["checkpoint_vocabulary_size"]),
    )
    data_alias = PROJECT / ".formal_eval_data_20260919"
    if data_alias.exists() or data_alias.is_symlink():
        if data_alias.resolve() != PREPARED.resolve():
            raise RuntimeError(f"neutral data alias points to the wrong directory: {data_alias}")
    else:
        data_alias.symlink_to(PREPARED.resolve(), target_is_directory=True)
    native_pickle = result_dir / "vanilla_bs_top10_generation_untyped.pk"

    environment = os.environ.copy()
    environment.update(
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        PYTHONHASHSEED=str(args.seed),
        PYTHONUNBUFFERED="1",
        PYTHONPATH=str(PROJECT),
    )
    started = time.time()
    state = {
        **evidence,
        **staged_vocabulary,
        "neutral_data_alias": str(data_alias),
        "neutral_data_alias_target": str(data_alias.resolve()),
        "status": "generating",
        "decoder": "vanilla_beam10_unknown_class_no_template",
        "started_at_unix": started,
    }
    atomic_json(output_dir / "state.json", state)
    run_checked(
        [
            sys.executable,
            "-m",
            "retroformer.translate",
            "--batch_size_val",
            "8",
            "--shared_vocab",
            "True",
            "--shared_encoder",
            "False",
            "--data_dir",
            data_alias.name,
            "--intermediate_dir",
            str(intermediate_dir),
            "--checkpoint_dir",
            str(work / "checkpoint"),
            "--checkpoint",
            "best.pt",
            "--known_class",
            "False",
            "--beam_size",
            "10",
            "--stepwise",
            "False",
            "--use_template",
            "False",
        ],
        cwd=PROJECT,
        environment=environment,
        log_path=output_dir / "generation.log",
    )
    if not native_pickle.is_file() or native_pickle.stat().st_size == 0:
        raise RuntimeError("Retroformer translator did not publish its native beam pickle")

    predictions = output_dir / "predictions.jsonl"
    conversion_manifest = output_dir / "conversion_manifest.json"
    run_checked(
        [
            sys.executable,
            str(EXPERIMENT / "convert_retroformer_predictions.py"),
            "--native-pickle",
            str(native_pickle),
            "--test-csv",
            evidence["test_csv"],
            "--test-lmdb",
            evidence["test_lmdb"],
            "--expected-reference-mismatch-json",
            evidence["reference_contract"],
            "--output-jsonl",
            str(predictions),
            "--manifest-json",
            str(conversion_manifest),
            "--expected-rows",
            "2400",
            "--beam-size",
            "10",
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
            f"Retroformer seed {args.seed}",
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
            "retroformer_native_length_normalized_beam_order",
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
        raise RuntimeError("Retroformer conversion verification failed")

    complete = {
        **evidence,
        "status": "complete",
        "native_pickle": str(native_pickle),
        "native_pickle_sha256": sha256(native_pickle),
        "predictions_jsonl": str(predictions),
        "predictions_jsonl_sha256": sha256(predictions),
        "conversion_manifest": str(conversion_manifest),
        "conversion_manifest_sha256": sha256(conversion_manifest),
        "metrics": str(metrics),
        "metrics_sha256": sha256(metrics),
        "per_reaction": str(per_reaction),
        "per_reaction_sha256": sha256(per_reaction),
        "elapsed_seconds": time.time() - started,
        "test_labels_used_only_for_post_generation_scoring_and_order_audit": True,
    }
    atomic_json(output_dir / "state.json", complete)
    atomic_json(output_dir / "EVALUATION_COMPLETE.json", complete)
    print(json.dumps(complete, sort_keys=True))


if __name__ == "__main__":
    main()
