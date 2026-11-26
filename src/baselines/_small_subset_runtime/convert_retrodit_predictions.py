#!/usr/bin/env python3
"""Convert RetroDiT samples to the frozen 2,400-row JSONL contract."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from rdkit import RDLogger

from evaluate_full_retro_topk import canonicalize_precursor_set


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


def read_frozen(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    frozen = []
    for source_row, row in enumerate(rows):
        parts = row["rxn_smiles"].split(">")
        if len(parts) != 3:
            raise ValueError(f"invalid frozen reaction at row {source_row}")
        target = canonicalize_precursor_set(parts[0])
        product = canonicalize_precursor_set(parts[2])
        if target is None or product is None:
            raise ValueError(f"invalid frozen structures at row {source_row}")
        frozen.append({"target": target, "product": product})
    return frozen


def native_hit_rank(raw_generations: list, reference: str | None) -> int | None:
    if reference is None:
        return None
    # Upstream ResultStore removes nulls and then uses Counter.most_common().
    # Counter preserves first occurrence for ties on supported Python versions.
    valid = [value for value in raw_generations if value is not None]
    ranked = [value for value, _ in Counter(valid).most_common()]
    return next((rank for rank, value in enumerate(ranked, start=1) if value == reference), None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-samples-json", type=Path, required=True)
    parser.add_argument("--mapping-json", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--input-sequence-audit", type=Path)
    parser.add_argument("--expected-rows", type=int, default=2400)
    parser.add_argument("--expected-generated-rows", type=int, default=2398)
    parser.add_argument("--samples-per-product", type=int, default=100)
    parser.add_argument("--raw-beam-size", type=int, default=10)
    parser.add_argument(
        "--unified-ranking",
        choices=("raw_first", "native_frequency"),
        default="raw_first",
    )
    args = parser.parse_args()
    if args.raw_beam_size > args.samples_per_product or args.raw_beam_size < 10:
        raise ValueError("raw beam size must be in [10, samples-per-product]")
    RDLogger.DisableLog("rdApp.*")

    samples = json.loads(args.native_samples_json.read_text(encoding="utf-8"))
    mapping_evidence = json.loads(args.mapping_json.read_text(encoding="utf-8"))
    if not isinstance(samples, list):
        raise TypeError("RetroDiT generated samples must be a JSON list")
    if mapping_evidence.get("status") != "passed":
        raise ValueError("RetroDiT test mapping verification did not pass")
    mapping = [int(value) for value in mapping_evidence["processed_index_to_source_row"]]
    if len(samples) != args.expected_generated_rows or len(mapping) != len(samples):
        raise ValueError(
            f"generated/mapping lengths are {len(samples)}/{len(mapping)}, expected {args.expected_generated_rows}"
        )
    if len(set(mapping)) != len(mapping) or any(not 0 <= row < args.expected_rows for row in mapping):
        raise ValueError("mapping contains duplicate or out-of-range source rows")

    frozen = read_frozen(args.test_csv)
    if len(frozen) != args.expected_rows:
        raise ValueError(f"frozen test row count {len(frozen)} != {args.expected_rows}")

    generated = {}
    reference_mismatches = []
    condition_mismatches = []
    native_hit_histogram = Counter()
    for processed_index, (source_row, sample) in enumerate(zip(mapping, samples)):
        if not isinstance(sample, dict):
            raise TypeError(f"generated sample {processed_index} is not an object")
        if "gen" not in sample:
            raise ValueError(
                "RetroDiT output omitted raw `gen` slots; frequency counts cannot reproduce unified raw order"
            )
        raw_generations = sample["gen"]
        if not isinstance(raw_generations, list) or len(raw_generations) != args.samples_per_product:
            raise ValueError(
                f"source row {source_row} has {len(raw_generations) if isinstance(raw_generations, list) else 'non-list'} "
                f"raw generations, expected {args.samples_per_product}"
            )
        condition = canonicalize_precursor_set(sample.get("cond"))
        reference = sample.get("ref")
        canonical_reference = canonicalize_precursor_set(reference)
        if condition != frozen[source_row]["product"]:
            condition_mismatches.append(source_row)
        if canonical_reference != frozen[source_row]["target"]:
            reference_mismatches.append(source_row)
        hit_rank = native_hit_rank(raw_generations, reference)
        if hit_rank is not None:
            native_hit_histogram[hit_rank] += 1
        if args.unified_ranking == "native_frequency":
            valid = [value for value in raw_generations if value is not None]
            unified_predictions = [
                value for value, _ in Counter(valid).most_common(args.raw_beam_size)
            ]
        else:
            unified_predictions = raw_generations[: args.raw_beam_size]
        generated[source_row] = {
            "predictions": unified_predictions,
            "native_supported": True,
            "native_hit_rank": hit_rank,
        }
    input_sequence_audit = None
    if args.input_sequence_audit is not None:
        input_sequence_audit = json.loads(args.input_sequence_audit.read_text(encoding="utf-8"))
        if (
            input_sequence_audit.get("status") != "passed"
            or input_sequence_audit.get("rows") != len(samples)
            or input_sequence_audit.get("pair_exact_rows") != len(samples)
            or input_sequence_audit.get("native_samples_sha256") != sha256(args.native_samples_json)
            or input_sequence_audit.get("mapping_json_sha256") != sha256(args.mapping_json)
        ):
            raise ValueError("RetroDiT generated-input sequence audit is invalid or stale")
    if condition_mismatches and input_sequence_audit is None:
        raise ValueError(f"generated condition/source-row mismatches: {condition_mismatches[:20]}")

    missing_rows = sorted(set(range(args.expected_rows)) - set(generated))
    expected_missing = sorted(int(value) for value in mapping_evidence["unsupported_source_rows"])
    if missing_rows != expected_missing:
        raise ValueError(f"mapping complement {missing_rows} != verified unsupported rows {expected_missing}")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_jsonl.with_suffix(args.output_jsonl.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for source_row in range(args.expected_rows):
            record = generated.get(source_row)
            if record is None:
                record = {"predictions": [], "native_supported": False, "native_hit_rank": None}
            handle.write(json.dumps({"source_row": source_row, **record}, ensure_ascii=False) + "\n")
    os.replace(temporary, args.output_jsonl)

    manifest = {
        "status": "passed",
        "model": "RetroDiT",
        "source_rows": args.expected_rows,
        "generated_source_rows": len(generated),
        "missing_source_rows": missing_rows,
        "samples_per_product": args.samples_per_product,
        "raw_beam_size": args.raw_beam_size,
        "unified_candidate_order": (
            "released_null_filter_then_frequency_counter"
            if args.unified_ranking == "native_frequency"
            else "first_raw_retrodit_generation_slots"
        ),
        "validity_filtering_before_unified_ranking": args.unified_ranking == "native_frequency",
        "deduplication_before_unified_ranking": args.unified_ranking == "native_frequency",
        "native_ranking": "released_null_filter_then_frequency_counter",
        "source_row_identity_check": "verified_processed_test_order_plus_exact_generated_input_sequence_replay",
        "generated_condition_identity_check": "diagnostic_only_after_exact_processed_graph_decode_replay",
        "generated_condition_canonical_match_rows": len(generated) - len(condition_mismatches),
        "generated_condition_canonical_mismatch_rows": condition_mismatches,
        "native_reference_canonical_match_rows": len(generated) - len(reference_mismatches),
        "native_reference_mismatch_rows": reference_mismatches,
        "native_reference_note": "diagnostic_only; unified scoring uses the frozen target indexed by source_row",
        "native_hit_rank_histogram": {str(k): v for k, v in sorted(native_hit_histogram.items())},
        "test_labels_used_for_selection": False,
        "candidate_order_or_content_modified_using_test_labels": False,
        "native_samples_json": str(args.native_samples_json.resolve()),
        "native_samples_sha256": sha256(args.native_samples_json),
        "mapping_json": str(args.mapping_json.resolve()),
        "mapping_json_sha256": sha256(args.mapping_json),
        "input_sequence_audit": str(args.input_sequence_audit.resolve()) if args.input_sequence_audit else None,
        "input_sequence_audit_sha256": sha256(args.input_sequence_audit) if args.input_sequence_audit else None,
        "test_csv": str(args.test_csv.resolve()),
        "test_csv_sha256": sha256(args.test_csv),
        "predictions_jsonl": str(args.output_jsonl.resolve()),
        "predictions_jsonl_sha256": sha256(args.output_jsonl),
    }
    atomic_json(args.manifest_json, manifest)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
