#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from rdkit import Chem, RDLogger


EXPECTED_TEST_SHA256 = "9FD0E110A733462561BE5E1C3BE01B45E7B4F2A9C2488D7A7646DC50F9A00DE4"
EXPECTED_TEST_ROWS = 9250
EXPECTED_CLASS_TOTALS = {"1": 2438, "2": 1371, "3": 1882, "4": 549, "5": 1574, "6": 1436}
TOP_K = (1, 3, 5, 10)


def sha256(path):
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


def read_test(path):
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"class", "rxn_smiles"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"selected test lacks columns: {sorted(required)}")
        for source_row, row in enumerate(reader):
            reaction_parts = row["rxn_smiles"].split(">")
            if len(reaction_parts) != 3:
                raise ValueError(f"invalid reaction at source row {source_row}")
            target = canonicalize_precursor_set(reaction_parts[0])
            if target is None:
                raise ValueError(f"invalid target at source row {source_row}")
            rows.append(
                {
                    "source_row": source_row,
                    "class": str(int(row["class"])),
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
                raise ValueError(f"out-of-range source row {source_row} at line {line_number}")
            if source_row in predictions:
                raise ValueError(f"duplicate source row {source_row}")
            candidates = record.get("predictions", [])
            if not isinstance(candidates, list):
                raise ValueError(f"predictions must be a list at line {line_number}")
            native_supported = record.get("native_supported", True)
            if not isinstance(native_supported, bool):
                raise ValueError(f"native_supported must be boolean at line {line_number}")
            native_hit_rank = record.get("native_hit_rank")
            if native_hit_rank is not None:
                if isinstance(native_hit_rank, bool):
                    raise ValueError(f"native_hit_rank must be an integer at line {line_number}")
                native_hit_rank = int(native_hit_rank)
                if native_hit_rank < 1 or native_hit_rank > len(candidates):
                    raise ValueError(
                        f"native_hit_rank is outside the saved prediction list at line {line_number}"
                    )
            predictions[source_row] = {
                "predictions": [str(value) for value in candidates],
                "native_supported": native_supported,
                "native_hit_rank": native_hit_rank,
            }
    return predictions


def rates(hits, total):
    if total == 0:
        return {f"Top-{k}": None for k in TOP_K}
    return {f"Top-{k}": hits[k] / total for k in TOP_K}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--predictions-jsonl", type=Path, required=True)
    parser.add_argument("--selected-test", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--per-reaction-csv", type=Path, required=True)
    parser.add_argument("--beam-size", type=int, default=10)
    parser.add_argument("--candidate-ranking", default="model_native_order")
    args = parser.parse_args()

    if args.beam_size < max(TOP_K):
        raise ValueError(f"--beam-size must be at least {max(TOP_K)}")
    if sha256(args.selected_test) != EXPECTED_TEST_SHA256:
        raise ValueError("selected_test.csv SHA-256 does not match the frozen split")

    RDLogger.DisableLog("rdApp.*")
    test_rows = read_test(args.selected_test)
    if len(test_rows) != EXPECTED_TEST_ROWS:
        raise ValueError(f"frozen test row count is {len(test_rows)}, expected {EXPECTED_TEST_ROWS}")
    observed_class_totals = Counter(row["class"] for row in test_rows)
    if dict(observed_class_totals) != EXPECTED_CLASS_TOTALS:
        raise ValueError(
            f"frozen test class totals differ: {dict(observed_class_totals)}"
        )
    prediction_rows = read_predictions(args.predictions_jsonl, len(test_rows))

    unified_hits = Counter()
    class_hits = defaultdict(Counter)
    class_totals = Counter()
    native_hits = Counter()
    native_total = 0
    native_rank_rows = 0
    invalid_predictions = 0
    duplicate_predictions = 0
    missing_rows = 0
    per_reaction = []

    for test_row in test_rows:
        source_row = test_row["source_row"]
        reaction_class = test_row["class"]
        class_totals[reaction_class] += 1
        record = prediction_rows.get(source_row)
        if record is None:
            missing_rows += 1
            candidates = []
            native_supported = False
            native_hit_rank = None
        else:
            candidates = record["predictions"][: args.beam_size]
            native_supported = record["native_supported"]
            native_hit_rank = record["native_hit_rank"]

        ranked, invalid_count, duplicate_count = ranked_canonical_predictions(candidates)
        invalid_predictions += invalid_count
        duplicate_predictions += duplicate_count
        canonical_hit_rank = next(
            (
                raw_rank
                for raw_rank, canonical in ranked
                if canonical == test_row["canonical_target"]
            ),
            None,
        )

        for k in TOP_K:
            if canonical_hit_rank is not None and canonical_hit_rank <= k:
                unified_hits[k] += 1
                class_hits[reaction_class][k] += 1

        if native_supported:
            native_total += 1
            if native_hit_rank is not None:
                native_rank_rows += 1
                for k in TOP_K:
                    if native_hit_rank <= k:
                        native_hits[k] += 1

        per_reaction.append(
            {
                "source_row": source_row,
                "class": reaction_class,
                "prediction_row_present": record is not None,
                "raw_prediction_count": len(candidates),
                "valid_unique_canonical_predictions": len(ranked),
                "invalid_prediction_count": invalid_count,
                "duplicate_canonical_prediction_count": duplicate_count,
                "canonical_target": test_row["canonical_target"],
                "canonical_top1_prediction": next(
                    (canonical for raw_rank, canonical in ranked if raw_rank == 1), ""
                ),
                "canonical_match_rank": canonical_hit_rank or "",
                "native_supported": native_supported,
                "native_hit_rank": native_hit_rank or "",
            }
        )

    class_metrics = {
        reaction_class: {
            "total_reactions": class_totals[reaction_class],
            **rates(class_hits[reaction_class], class_totals[reaction_class]),
        }
        for reaction_class in sorted(class_totals, key=int)
    }
    metrics = {
        "model": args.model_name,
        "beam_size": args.beam_size,
        "artifacts": {
            "predictions_jsonl": str(args.predictions_jsonl.resolve()),
            "predictions_sha256": sha256(args.predictions_jsonl),
            "selected_test": str(args.selected_test.resolve()),
            "selected_test_sha256": sha256(args.selected_test),
        },
        "unified_canonical": {
            "denominator": "all_frozen_test_rows",
            "total_reactions": len(test_rows),
            "prediction_rows_present": len(prediction_rows),
            "missing_prediction_rows": missing_rows,
            "invalid_prediction_occurrences": invalid_predictions,
            "duplicate_prediction_occurrences": duplicate_predictions,
            "candidate_ranking": args.candidate_ranking,
            "rank_slot_policy": "invalid_and_duplicate_candidates_consume_original_model_rank",
            "canonical_deduplication": "first_occurrence_retained_without_rank_compaction",
            "precursor_comparison": "map_free_stereo_preserving_order_independent_set",
            **rates(unified_hits, len(test_rows)),
            "by_class": class_metrics,
        },
        "native_from_export": {
            "available": native_total > 0,
            "supported_rows": native_total,
            "rows_with_native_hit_rank": native_rank_rows,
            **(rates(native_hits, native_total) if native_total > 0 else {}),
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.per_reaction_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    with args.per_reaction_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_reaction[0].keys())
        writer.writeheader()
        writer.writerows(per_reaction)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
