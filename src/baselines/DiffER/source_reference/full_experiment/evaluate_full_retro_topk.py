#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from rdkit import Chem, RDLogger


TOP_K = (1, 3, 5, 10)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def canonicalize_molecule(smiles):
    try:
        molecule = Chem.MolFromSmiles(str(smiles))
        if molecule is None:
            return None
        for atom in molecule.GetAtoms():
            atom.SetAtomMapNum(0)
        molecule = Chem.RemoveHs(molecule)
        return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def canonicalize_precursor_set(value):
    components = []
    for smiles in str(value).split("."):
        smiles = smiles.strip()
        if not smiles:
            continue
        canonical = canonicalize_molecule(smiles)
        if canonical is None:
            return None
        components.append(canonical)
    if not components:
        return None
    return ".".join(sorted(set(components)))


def ranked_canonical_predictions(predictions):
    ranked = []
    seen = set()
    invalid = 0
    duplicates = 0
    for raw_rank, prediction in enumerate(predictions, start=1):
        canonical = canonicalize_precursor_set(prediction)
        if canonical is None:
            invalid += 1
            continue
        if canonical in seen:
            duplicates += 1
            continue
        seen.add(canonical)
        ranked.append((raw_rank, canonical))
    return ranked, invalid, duplicates


def read_test(test_path, index_path):
    with test_path.open("r", encoding="utf-8", newline="") as handle:
        test_rows = list(csv.DictReader(handle))
    with index_path.open("r", encoding="utf-8", newline="") as handle:
        index_rows = list(csv.DictReader(handle))
    if len(test_rows) != len(index_rows):
        raise ValueError("test and test_index row counts differ")
    rows = []
    for local_row, (test_row, index_row) in enumerate(zip(test_rows, index_rows)):
        parts = test_row["rxn_smiles"].split(">")
        if len(parts) != 3:
            raise ValueError(f"invalid reaction at test row {local_row}")
        target = canonicalize_precursor_set(parts[0])
        if target is None:
            raise ValueError(f"invalid target at test row {local_row}")
        rows.append(
            {
                "source_row": local_row,
                "source_parent_row": int(index_row["source_row"]),
                "sample_id": index_row["sample_id"],
                "class": str(int(test_row["class"])),
                "similarity_group": index_row["similarity_group"],
                "product_group": index_row["product_group"],
                "canonical_target": target,
            }
        )
    return rows


def read_predictions(path, total_rows):
    predictions = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            source_row = int(record["source_row"])
            if source_row < 0 or source_row >= total_rows:
                raise ValueError(f"out-of-range source_row at predictions line {line_number}")
            if source_row in predictions:
                raise ValueError(f"duplicate source_row {source_row}")
            candidates = record.get("predictions", [])
            if not isinstance(candidates, list):
                raise ValueError(f"predictions must be a list at line {line_number}")
            native_hit_rank = record.get("native_hit_rank")
            if native_hit_rank is not None:
                native_hit_rank = int(native_hit_rank)
            predictions[source_row] = {
                "predictions": [str(value) for value in candidates],
                "native_supported": bool(record.get("native_supported", True)),
                "native_hit_rank": native_hit_rank,
            }
    return predictions


def rates(hits, total):
    return {f"Top-{k}": (hits[k] / total if total else None) for k in TOP_K}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--predictions-jsonl", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--test-index", type=Path, required=True)
    parser.add_argument("--expected-test-sha256")
    parser.add_argument("--expected-index-sha256")
    parser.add_argument("--expected-rows", type=int, default=2400)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--per-reaction-csv", type=Path, required=True)
    parser.add_argument("--beam-size", type=int, default=10)
    parser.add_argument("--candidate-ranking", default="model_native_order")
    args = parser.parse_args()
    if args.beam_size < max(TOP_K):
        raise ValueError("beam size must be at least 10")
    if args.expected_test_sha256 and sha256(args.test_csv) != args.expected_test_sha256.upper():
        raise ValueError("test CSV SHA-256 mismatch")
    if args.expected_index_sha256 and sha256(args.test_index) != args.expected_index_sha256.upper():
        raise ValueError("test index SHA-256 mismatch")

    RDLogger.DisableLog("rdApp.*")
    test_rows = read_test(args.test_csv, args.test_index)
    if len(test_rows) != args.expected_rows:
        raise ValueError(f"test row count {len(test_rows)} != {args.expected_rows}")
    prediction_rows = read_predictions(args.predictions_jsonl, len(test_rows))

    hits = Counter()
    group_hits = defaultdict(Counter)
    group_totals = Counter()
    class_hits = defaultdict(Counter)
    class_totals = Counter()
    native_hits = Counter()
    native_total = 0
    invalid_total = 0
    duplicate_total = 0
    missing_total = 0
    per_reaction = []

    for row in test_rows:
        source_row = row["source_row"]
        group = row["similarity_group"]
        reaction_class = row["class"]
        group_totals[group] += 1
        class_totals[reaction_class] += 1
        record = prediction_rows.get(source_row)
        if record is None:
            missing_total += 1
            candidates = []
            native_supported = False
            native_hit_rank = None
        else:
            candidates = record["predictions"][: args.beam_size]
            native_supported = record["native_supported"]
            native_hit_rank = record["native_hit_rank"]
        ranked, invalid, duplicates = ranked_canonical_predictions(candidates)
        invalid_total += invalid
        duplicate_total += duplicates
        hit_rank = next(
            (raw_rank for raw_rank, canonical in ranked if canonical == row["canonical_target"]),
            None,
        )
        for k in TOP_K:
            if hit_rank is not None and hit_rank <= k:
                hits[k] += 1
                group_hits[group][k] += 1
                class_hits[reaction_class][k] += 1
        if native_supported:
            native_total += 1
            for k in TOP_K:
                if native_hit_rank is not None and native_hit_rank <= k:
                    native_hits[k] += 1
        per_reaction.append(
            {
                **row,
                "prediction_row_present": record is not None,
                "raw_prediction_count": len(candidates),
                "valid_unique_prediction_count": len(ranked),
                "invalid_prediction_count": invalid,
                "duplicate_prediction_count": duplicates,
                "canonical_match_rank": hit_rank or "",
                "native_supported": native_supported,
                "native_hit_rank": native_hit_rank or "",
            }
        )

    metrics = {
        "model": args.model_name,
        "top_k": list(TOP_K),
        "artifacts": {
            "predictions": str(args.predictions_jsonl.resolve()),
            "predictions_sha256": sha256(args.predictions_jsonl),
            "test_csv_sha256": sha256(args.test_csv),
            "test_index_sha256": sha256(args.test_index),
        },
        "unified": {
            "total_reactions": len(test_rows),
            "prediction_rows_present": len(prediction_rows),
            "missing_prediction_rows": missing_total,
            "invalid_prediction_occurrences": invalid_total,
            "duplicate_prediction_occurrences": duplicate_total,
            "rank_slot_policy": "invalid_and_duplicate_candidates_consume_original_rank",
            "canonicalization": "map_free_stereo_preserving_order_independent_component_set",
            **rates(hits, len(test_rows)),
            "by_similarity_group": {
                group: {"total": group_totals[group], **rates(group_hits[group], group_totals[group])}
                for group in ("low", "medium", "high")
            },
            "by_class": {
                cls: {"total": class_totals[cls], **rates(class_hits[cls], class_totals[cls])}
                for cls in sorted(class_totals, key=int)
            },
        },
        "native": {"supported_rows": native_total, **rates(native_hits, native_total)},
        "candidate_ranking": args.candidate_ranking,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.per_reaction_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with args.per_reaction_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_reaction[0]))
        writer.writeheader()
        writer.writerows(per_reaction)
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
