#!/usr/bin/env python3
"""Convert one DiffER checkpoint's native samples to the frozen JSONL contract.

The unified result preserves the first raw sampling slots (after removing the
model's ``?`` padding token) without validity filtering or deduplication.  The
author dashboard's plurality ranking is computed separately as a supplemental
native metric and never replaces the unified raw-slot ordering.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from rdkit import Chem, RDLogger

from evaluate_full_retro_topk import canonicalize_precursor_set


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def differ_canonicalize(smiles: str) -> str | None:
    """Match the released DiffER dashboard's SMILES cleanup."""
    molecule = Chem.MolFromSmiles(str(smiles).replace("?", ""))
    if molecule is None:
        return None
    return Chem.MolToSmiles(molecule)


def product_identity(smiles: str) -> str | None:
    """Reproduce the data-preparation identity key for source products.

    Frozen products still contain atom maps whereas DiffER's prepared inputs
    do not.  Clearing maps before fragment-wise canonicalization is exactly the
    rule used by ``prepare_differ_data.py`` and avoids map-dependent canonical
    ring/stereochemistry spellings without changing molecular identity.
    """
    molecule = Chem.MolFromSmiles(str(smiles).replace("?", ""))
    if molecule is None:
        return None
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    fragments = Chem.GetMolFrags(molecule, asMols=True, sanitizeFrags=True)
    return ".".join(
        sorted(
            Chem.MolToSmiles(fragment, canonical=True, isomericSmiles=True)
            for fragment in fragments
        )
    )


def read_frozen_test(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = []
    for source_row, row in enumerate(rows):
        parts = row["rxn_smiles"].split(">")
        if len(parts) != 3:
            raise ValueError(f"invalid frozen reaction at row {source_row}")
        product = product_identity(parts[2])
        target = canonicalize_precursor_set(parts[0])
        if product is None or target is None:
            raise ValueError(f"invalid frozen reaction structures at row {source_row}")
        result.append({"product": product, "target": target})
    return result


def native_plurality_ranking(samples: list[str], product: str, samples_per_input: int) -> list[str]:
    """Reproduce the released dashboard's frequency/rank aggregation order."""
    plurality: dict[str, int] = defaultdict(int)
    by_input: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for sample_index, raw_sample in enumerate(samples):
        sample = differ_canonicalize(raw_sample)
        if sample is None or sample == product:
            continue
        plurality[sample] += 1
        by_input[sample_index // samples_per_input][sample] += 1

    ranked_choice: dict[str, int] = {}
    input_ranks: dict[int, dict[str, int]] = {}
    for input_index, counts in by_input.items():
        input_ranks[input_index] = {}
        current_rating = float("inf")
        current_rank = 0
        for sample, rating in sorted(counts.items(), key=lambda item: item[1]):
            ranked_choice[sample] = 0
            if rating < current_rating:
                current_rating = rating
                current_rank += 1
            input_ranks[input_index][sample] = current_rank
    for sample in ranked_choice:
        for ranks in input_ranks.values():
            if sample in ranks:
                ranked_choice[sample] += ranks[sample]
    ranking_values = {
        sample: (plurality[sample], ranked_choice[sample]) for sample in ranked_choice
    }
    return sorted(ranking_values, key=lambda sample: ranking_values[sample], reverse=True)


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-samples-json", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=2400)
    parser.add_argument("--beam-size", type=int, default=10)
    parser.add_argument("--augmentations", type=int, default=20)
    parser.add_argument("--samples-per-input", type=int, default=20)
    args = parser.parse_args()
    if min(args.beam_size, args.augmentations, args.samples_per_input) <= 0:
        raise ValueError("beam, augmentation, and sample counts must be positive")
    native_budget = args.augmentations * args.samples_per_input

    RDLogger.DisableLog("rdApp.*")
    native = json.loads(args.native_samples_json.read_text(encoding="utf-8"))
    if not isinstance(native, dict):
        raise ValueError("DiffER native samples must be a JSON object")
    grouped: dict[str, list[str]] = defaultdict(list)
    for raw_product, record in native.items():
        product = product_identity(raw_product)
        if product is None:
            raise ValueError(f"invalid DiffER source product: {raw_product}")
        samples = record.get("samples")
        if not isinstance(samples, list):
            raise ValueError(f"missing sample list for source: {raw_product}")
        grouped[product].extend(str(sample) for sample in samples)

    frozen = read_frozen_test(args.test_csv)
    if len(frozen) != args.expected_rows:
        raise ValueError(f"frozen test row count {len(frozen)} != {args.expected_rows}")
    missing_products = sorted({row["product"] for row in frozen if row["product"] not in grouped})
    short_products = sorted(
        {row["product"] for row in frozen if len(grouped.get(row["product"], [])) < native_budget}
    )
    if missing_products or short_products:
        raise ValueError(
            f"DiffER coverage failed: missing={missing_products[:10]}, short={short_products[:10]}"
        )

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_jsonl.with_suffix(args.output_jsonl.suffix + ".tmp")
    raw_count_histogram = Counter()
    native_hit_histogram = Counter()
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for source_row, row in enumerate(frozen):
            # Cap before either unified or native reporting so products with
            # multiple frozen references never receive extra sampling budget.
            samples = grouped[row["product"]][:native_budget]
            raw_predictions = [sample.replace("?", "") for sample in samples[: args.beam_size]]
            raw_count_histogram[len(raw_predictions)] += 1
            native_ranking = native_plurality_ranking(
                samples, row["product"], args.samples_per_input
            )
            native_hit_rank = next(
                (
                    rank
                    for rank, prediction in enumerate(native_ranking, start=1)
                    if canonicalize_precursor_set(prediction) == row["target"]
                ),
                None,
            )
            if native_hit_rank is not None:
                native_hit_histogram[native_hit_rank] += 1
            handle.write(
                json.dumps(
                    {
                        "source_row": source_row,
                        "predictions": raw_predictions,
                        "native_supported": True,
                        "native_hit_rank": native_hit_rank,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    os.replace(temporary, args.output_jsonl)

    manifest = {
        "status": "passed",
        "model": "DiffER single checkpoint",
        "source_rows": args.expected_rows,
        "unified_candidate_order": "first_raw_sampling_slots_in_released_eval_json_order",
        "padding_cleanup": "remove_question_mark_token_only",
        "validity_filtering_before_unified_ranking": False,
        "deduplication_before_unified_ranking": False,
        "beam_size": args.beam_size,
        "augmentations": args.augmentations,
        "samples_per_input": args.samples_per_input,
        "fixed_native_sample_budget_per_product": native_budget,
        "native_ranking": "released_dashboard_plurality_then_per_input_rank",
        "test_labels_used_for_selection": False,
        "test_labels_used_post_generation_for_native_metric": True,
        "candidate_order_or_content_modified_using_test_labels": False,
        "raw_prediction_count_histogram": {
            str(key): value for key, value in sorted(raw_count_histogram.items())
        },
        "native_hit_rank_histogram": {
            str(key): value for key, value in sorted(native_hit_histogram.items())
        },
        "native_samples_json": str(args.native_samples_json.resolve()),
        "native_samples_sha256": sha256(args.native_samples_json),
        "test_csv": str(args.test_csv.resolve()),
        "test_csv_sha256": sha256(args.test_csv),
        "predictions_jsonl": str(args.output_jsonl.resolve()),
        "predictions_jsonl_sha256": sha256(args.output_jsonl),
    }
    args.manifest_json.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.manifest_json, manifest)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
