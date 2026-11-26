#!/usr/bin/env python3
"""Run and verify one RetroDiT seed's complete formal test workflow."""

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
PROJECT = ROOT / "third_party" / "benchmark_20260919" / "RetroDiT"
FROZEN = ROOT / "experiments" / "stage1_similarity_rebalanced_20260912" / "frozen"
DATA = EXPERIMENT / "formal" / "retrodit" / "seed0_data"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--training-complete", type=Path)
    parser.add_argument("--rc-complete", type=Path)
    parser.add_argument("--mapping-json", type=Path, default=EXPERIMENT / "formal" / "retrodit" / "test_mapping.json")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume-postprocessing", action="store_true")
    parser.add_argument(
        "--unified-ranking",
        choices=("raw_first", "native_frequency"),
        default="raw_first",
    )
    return parser.parse_args()


def verify_inputs(args: argparse.Namespace) -> dict:
    training_dir = EXPERIMENT / "formal" / "retrodit" / f"seed{args.seed}_training"
    training_complete = (args.training_complete or training_dir / "training_complete.json").resolve()
    if not training_complete.is_file():
        raise FileNotFoundError(training_complete)
    training = json.loads(training_complete.read_text(encoding="utf-8"))
    checkpoint = (args.checkpoint or Path(training.get("best_checkpoint", ""))).resolve()
    rc_complete = (
        args.rc_complete
        or EXPERIMENT / "formal" / "retrodit_rc" / f"seed{args.seed}" / "complete.json"
    ).resolve()
    required = {
        "checkpoint": checkpoint,
        "training_complete": training_complete,
        "rc_complete": rc_complete,
        "mapping_json": args.mapping_json.resolve(),
        "upstream_eval": PROJECT / "src" / "eval.py",
        "converter": EXPERIMENT / "convert_retrodit_predictions.py",
        "input_sequence_verifier": EXPERIMENT / "verify_retrodit_generated_input_sequence.py",
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
    if training.get("status") != "complete" or int(training.get("seed", -1)) != args.seed:
        raise RuntimeError("RetroDiT training completion evidence is absent or belongs to another seed")
    if bool(training.get("test_data_used", True)):
        raise RuntimeError("training completion evidence indicates test data use")
    if sha256(checkpoint) != str(training.get("best_checkpoint_sha256", "")).lower():
        raise RuntimeError("RetroDiT best checkpoint hash mismatch")

    rc = json.loads(rc_complete.read_text(encoding="utf-8"))
    if rc.get("status") != "complete" or int(rc.get("seed", -1)) != args.seed:
        raise RuntimeError("RetroDiT reaction-center completion evidence is invalid")
    permutations = Path(rc["test_permutations"]).resolve()
    if not permutations.is_file() or sha256(permutations) != str(rc["test_permutations_sha256"]).lower():
        raise RuntimeError("RetroDiT reaction-center permutation hash mismatch")

    mapping = json.loads(required["mapping_json"].read_text(encoding="utf-8"))
    if (
        mapping.get("status") != "passed"
        or mapping.get("source_rows") != 2400
        or mapping.get("processed_rows") != 2398
        or mapping.get("unsupported_source_rows") != [860, 1517]
    ):
        raise RuntimeError("RetroDiT frozen test mapping evidence is invalid")
    for path_key, hash_key in (
        ("raw_test_csv", "raw_test_csv_sha256"),
        ("processed_test", "processed_test_sha256"),
        ("rc_verification", "rc_verification_sha256"),
    ):
        path = Path(mapping[path_key])
        if not path.is_file() or sha256(path) != mapping[hash_key]:
            raise RuntimeError(f"RetroDiT mapping input changed: {path_key}")

    return {
        "status": "passed",
        "seed": args.seed,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "training_complete": str(training_complete),
        "training_complete_sha256": sha256(training_complete),
        "rc_complete": str(rc_complete),
        "rc_complete_sha256": sha256(rc_complete),
        "test_permutations": str(permutations),
        "test_permutations_sha256": sha256(permutations),
        "mapping_json": str(required["mapping_json"]),
        "mapping_json_sha256": sha256(required["mapping_json"]),
        "test_csv": str(required["test_csv"]),
        "test_csv_sha256": EXPECTED_TEST_SHA256.lower(),
        "test_index": str(required["test_index"]),
        "test_index_sha256": EXPECTED_INDEX_SHA256.lower(),
        "test_data_used_for_checkpoint_or_protocol_selection": False,
    }


def main() -> None:
    args = parse_args()
    evidence = verify_inputs(args)
    if args.preflight_only:
        print(json.dumps(evidence, sort_keys=True))
        return

    output_dir = (
        args.output_dir
        or EXPERIMENT / "formal" / "evaluation" / "retrodit" / f"seed{args.seed}"
    ).resolve()
    resuming = args.resume_postprocessing
    if output_dir.exists() and any(output_dir.iterdir()) and not resuming:
        raise FileExistsError(f"refusing to overwrite non-empty formal output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if resuming:
        prior_preflight = json.loads((output_dir / "preflight.json").read_text(encoding="utf-8"))
        if prior_preflight != evidence:
            raise RuntimeError("resume preflight differs from the original formal preflight")
        if (output_dir / "EVALUATION_COMPLETE.json").exists():
            raise FileExistsError("formal evaluation is already complete")
    else:
        atomic_json(output_dir / "preflight.json", evidence)

    native_hydra = output_dir / "native_hydra"
    native_output = output_dir / "native_output"
    if resuming:
        if not native_hydra.is_dir() or not native_output.is_dir():
            raise FileNotFoundError("resume requires the original native output directories")
    else:
        native_hydra.mkdir()
        native_output.mkdir()
    environment = os.environ.copy()
    environment.update(
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        PROJECT_ROOT=str(PROJECT),
        PYTHONPATH=str(PROJECT),
        PYTHONHASHSEED=str(args.seed),
        PYTHONUNBUFFERED="1",
    )
    started = time.time()
    if resuming:
        prior_state = json.loads((output_dir / "state.json").read_text(encoding="utf-8"))
        started = float(prior_state.get("started_at_unix", started))
        state = {
            **evidence,
            "status": "resuming_postprocessing",
            "started_at_unix": started,
            "unified_ranking": args.unified_ranking,
        }
        atomic_json(output_dir / "state.json", state)
    else:
        state = {
            **evidence,
            "status": "generating",
            "started_at_unix": started,
            "unified_ranking": args.unified_ranking,
        }
        atomic_json(output_dir / "state.json", state)
    command = [
        sys.executable,
        "-m", "src.eval",
        "experiment=retro",
        "data=uspto_50k",
        "+model/sde=dfm",
        "+model/net=graph_dit/8m",
        "data.n_dummy=30",
        "data.batch_size=64",
        "data.num_workers=0",
        f"data.perm_types={evidence['test_permutations']}",
        f"seed={args.seed}",
        f"task_name=retrodit_formal_eval_seed{args.seed}",
        f"paths.data_dir={DATA}",
        f"paths.output_dir={native_output}",
        f"hydra.run.dir={native_hydra}",
        "data.forward_syn_model=null",
        "trainer.accelerator=gpu",
        "trainer.devices=1",
        "trainer.strategy=auto",
        "trainer.precision=bf16-mixed",
        "trainer.enable_progress_bar=false",
        "model.test_top_ks=[1,3,5,10,20,50,100]",
        f"ckpt_path={evidence['checkpoint']}",
        "~logger",
    ]
    if not resuming:
        run_checked(command, cwd=PROJECT, environment=environment, log_path=output_dir / "generation.log")
    native_samples = native_hydra / "generated_samples.json"
    native_metrics = native_hydra / "generated_metrics.json"
    native_complete = native_output / "EVALUATION_COMPLETE"
    for artifact in (native_samples, native_metrics, native_complete):
        if not artifact.is_file():
            raise FileNotFoundError(artifact)

    input_sequence_audit = output_dir / "input_sequence_audit.json"
    run_checked(
        [
            sys.executable,
            str(EXPERIMENT / "verify_retrodit_generated_input_sequence.py"),
            "--native-samples-json", str(native_samples),
            "--mapping-json", evidence["mapping_json"],
            "--dataset-root", str(DATA / "uspto_50k"),
            "--output-json", str(input_sequence_audit),
            "--expected-rows", "2398",
            "--batch-size", "64",
        ],
        cwd=PROJECT,
        environment=environment,
        log_path=output_dir / "input_sequence_audit.log",
    )

    predictions = output_dir / "predictions.jsonl"
    conversion_manifest = output_dir / "conversion_manifest.json"
    run_checked(
        [
            sys.executable,
            str(EXPERIMENT / "convert_retrodit_predictions.py"),
            "--native-samples-json", str(native_samples),
            "--mapping-json", evidence["mapping_json"],
            "--test-csv", evidence["test_csv"],
            "--output-jsonl", str(predictions),
            "--manifest-json", str(conversion_manifest),
            "--input-sequence-audit", str(input_sequence_audit),
            "--expected-rows", "2400",
            "--expected-generated-rows", "2398",
            "--samples-per-product", "100",
            "--raw-beam-size", "10",
            "--unified-ranking", args.unified_ranking,
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
            "--model-name", f"RetroDiT seed {args.seed}",
            "--predictions-jsonl", str(predictions),
            "--test-csv", evidence["test_csv"],
            "--test-index", evidence["test_index"],
            "--expected-test-sha256", EXPECTED_TEST_SHA256,
            "--expected-index-sha256", EXPECTED_INDEX_SHA256,
            "--expected-rows", "2400",
            "--output-json", str(metrics),
            "--per-reaction-csv", str(per_reaction),
            "--beam-size", "10",
            "--candidate-ranking",
            (
                "released_null_filter_then_frequency_counter"
                if args.unified_ranking == "native_frequency"
                else "first_raw_retrodit_generation_slots"
            ),
        ],
        cwd=EXPERIMENT,
        environment=environment,
        log_path=output_dir / "scoring.log",
    )
    parsed = json.loads(metrics.read_text(encoding="utf-8"))
    unified = parsed.get("unified", {})
    conversion = json.loads(conversion_manifest.read_text(encoding="utf-8"))
    if unified.get("total_reactions") != 2400 or unified.get("prediction_rows_present") != 2400:
        raise RuntimeError("formal metrics do not cover all 2,400 frozen test reactions")
    if conversion.get("status") != "passed" or conversion.get("source_rows") != 2400:
        raise RuntimeError("RetroDiT conversion verification failed")

    complete = {
        **evidence,
        "status": "complete",
        "native_samples": str(native_samples),
        "native_samples_sha256": sha256(native_samples),
        "native_metrics": str(native_metrics),
        "native_metrics_sha256": sha256(native_metrics),
        "input_sequence_audit": str(input_sequence_audit),
        "input_sequence_audit_sha256": sha256(input_sequence_audit),
        "predictions_jsonl": str(predictions),
        "predictions_jsonl_sha256": sha256(predictions),
        "conversion_manifest": str(conversion_manifest),
        "conversion_manifest_sha256": sha256(conversion_manifest),
        "metrics": str(metrics),
        "metrics_sha256": sha256(metrics),
        "per_reaction": str(per_reaction),
        "per_reaction_sha256": sha256(per_reaction),
        "unified_ranking": args.unified_ranking,
        "unified_candidate_order": conversion["unified_candidate_order"],
        "elapsed_seconds": time.time() - started,
        "test_labels_used_only_for_post_generation_scoring_and_order_audit": True,
    }
    atomic_json(output_dir / "state.json", complete)
    atomic_json(output_dir / "EVALUATION_COMPLETE.json", complete)
    print(json.dumps(complete, sort_keys=True))


if __name__ == "__main__":
    main()
