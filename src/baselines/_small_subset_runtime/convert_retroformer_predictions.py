#!/usr/bin/env python3
"""Convert Retroformer's native beam pickle into the frozen JSONL contract."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
from collections import Counter
from pathlib import Path

from rdkit import Chem, RDLogger

from evaluate_full_retro_topk import canonicalize_precursor_set


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def native_canonicalize(smiles: str) -> str:
    """Match Retroformer's native ``canonical_smiles`` implementation."""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return smiles
    canonical = Chem.MolToSmiles(molecule)
    if "." in canonical:
        components = canonical.split(".")
        canonical = ".".join(sorted(components, key=lambda value: (len(value), value)))
    return canonical


def frozen_targets(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    targets = []
    for row_number, row in enumerate(rows):
        parts = row["rxn_smiles"].split(">")
        if len(parts) != 3:
            raise ValueError(f"invalid frozen reaction at row {row_number}")
        target = canonicalize_precursor_set(parts[0])
        if target is None:
            raise ValueError(f"invalid frozen reactants at row {row_number}")
        targets.append(target)
    return targets


def native_source_rows(lmdb_directory: Path | None, expected_rows: int) -> list[int]:
    if lmdb_directory is None:
        return list(range(expected_rows))
    import lmdb

    environment = lmdb.open(
        str(lmdb_directory),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
    )
    try:
        with environment.begin(write=False) as transaction:
            keys = list(transaction.cursor().iternext(values=False))
    finally:
        environment.close()
    rows = [int(key.decode("utf-8").split(" ", 1)[0]) for key in keys]
    if len(rows) != expected_rows:
        raise ValueError(f"LMDB row count {len(rows)} != expected {expected_rows}")
    if sorted(rows) != list(range(expected_rows)):
        raise ValueError("LMDB source rows are not a permutation of the frozen row range")
    return rows


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-pickle", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--test-lmdb", type=Path)
    parser.add_argument("--expected-reference-mismatch-json", type=Path)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=2400)
    parser.add_argument("--beam-size", type=int, default=10)
    args = parser.parse_args()

    RDLogger.DisableLog("rdApp.*")
    with args.native_pickle.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, tuple) or len(payload) != 2:
        raise ValueError("Retroformer native output must be (ground_truths, generations)")
    ground_truths, generations = payload
    if len(ground_truths) != args.expected_rows or len(generations) != args.expected_rows:
        raise ValueError(
            f"native row count mismatch: targets={len(ground_truths)}, "
            f"generations={len(generations)}, expected={args.expected_rows}"
        )
    targets = frozen_targets(args.test_csv)
    if len(targets) != args.expected_rows:
        raise ValueError(f"frozen test row count {len(targets)} != {args.expected_rows}")
    source_rows = native_source_rows(args.test_lmdb, args.expected_rows)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_jsonl.with_suffix(args.output_jsonl.suffix + ".tmp")
    candidate_counts = Counter()
    reference_mismatches = []
    native_hits = Counter()
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for native_row, (source_row, ground_truth, raw_candidates) in enumerate(
            zip(source_rows, ground_truths, generations)
        ):
            candidates = [str(value) for value in list(raw_candidates)[: args.beam_size]]
            candidate_counts[len(candidates)] += 1
            canonical_ground_truth = canonicalize_precursor_set(str(ground_truth))
            if canonical_ground_truth != targets[source_row]:
                reference_mismatches.append({"native_row": native_row, "source_row": source_row})
            native_target = native_canonicalize(str(ground_truth))
            native_hit_rank = next(
                (
                    rank
                    for rank, candidate in enumerate(candidates, start=1)
                    if native_canonicalize(candidate) == native_target
                ),
                None,
            )
            if native_hit_rank is not None:
                native_hits[native_hit_rank] += 1
            record = {
                "source_row": source_row,
                "predictions": candidates,
                "native_supported": True,
                "native_hit_rank": native_hit_rank,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    mismatch_contract = None
    if args.expected_reference_mismatch_json is not None:
        mismatch_contract = json.loads(args.expected_reference_mismatch_json.read_text(encoding="utf-8"))
        expected_mismatches = set(mismatch_contract["expected_mismatch_source_rows"])
        actual_mismatches = {item["source_row"] for item in reference_mismatches}
        if actual_mismatches != expected_mismatches:
            temporary.unlink(missing_ok=True)
            raise ValueError(
                "Retroformer preprocessing-reference mismatch set differs from the frozen contract: "
                f"missing={sorted(expected_mismatches - actual_mismatches)[:20]}, "
                f"unexpected={sorted(actual_mismatches - expected_mismatches)[:20]}"
            )
    elif reference_mismatches:
        temporary.unlink(missing_ok=True)
        raise ValueError(
            f"Retroformer output order/reference mismatch at {len(reference_mismatches)} rows; "
            f"first rows={reference_mismatches[:20]}"
        )
    os.replace(temporary, args.output_jsonl)

    manifest = {
        "status": "passed",
        "model": "Retroformer",
        "source_rows": args.expected_rows,
        "beam_size": args.beam_size,
        "candidate_count_histogram": {str(key): value for key, value in sorted(candidate_counts.items())},
        "reference_order_check": "passed_via_lmdb_source_row_mapping",
        "native_reference_match_count": args.expected_rows - len(reference_mismatches),
        "native_reference_mismatch_count": len(reference_mismatches),
        "native_reference_mismatch_source_rows": sorted(
            item["source_row"] for item in reference_mismatches
        ),
        "reference_mismatch_contract": (
            str(args.expected_reference_mismatch_json.resolve())
            if args.expected_reference_mismatch_json is not None
            else None
        ),
        "reference_mismatch_contract_sha256": (
            sha256(args.expected_reference_mismatch_json)
            if args.expected_reference_mismatch_json is not None
            else None
        ),
        "native_row_mapping": "lmdb_key_source_row" if args.test_lmdb else "identity",
        "native_source_rows_sha256": hashlib.sha256(
            json.dumps(source_rows, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "test_labels_used_for_selection": False,
        "test_labels_used_post_generation_for_order_and_native_metric_audit": True,
        "candidate_order_or_content_modified_using_test_labels": False,
        "native_pickle": str(args.native_pickle.resolve()),
        "native_pickle_sha256": sha256(args.native_pickle),
        "test_csv": str(args.test_csv.resolve()),
        "test_csv_sha256": sha256(args.test_csv),
        "predictions_jsonl": str(args.output_jsonl.resolve()),
        "predictions_jsonl_sha256": sha256(args.output_jsonl),
        "native_hit_rank_histogram": {str(key): value for key, value in sorted(native_hits.items())},
    }
    args.manifest_json.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.manifest_json, manifest)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
