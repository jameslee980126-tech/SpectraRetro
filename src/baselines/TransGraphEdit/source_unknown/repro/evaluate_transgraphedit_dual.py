import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import pandas as pd
from rdkit import Chem, RDLogger


EXPECTED_TEST_SHA256 = "9FD0E110A733462561BE5E1C3BE01B45E7B4F2A9C2488D7A7646DC50F9A00DE4"
TOP_K = (1, 3, 5, 10)
HEADER_PATTERN = re.compile(r"^\((\d+)\)\s+(\S+)\s*$")
PREDICTION_PATTERN = re.compile(
    r"^(\d+)\s+prediction_is_correct:(True|False)\s+"
    r"probability:(\S+)\s+(\S+)(?:\s+.*)?$"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Summarize a Graph2Edits-family native Beam-k result and evaluate the same "
            "saved predictions with the reaction-row canonical exact-match protocol "
            "adapted from the GDiffRetro legacy-78 evaluation family."
        )
    )
    parser.add_argument("--model-name", default="TransGraphEdit")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--selected-test", type=Path, required=True)
    parser.add_argument("--native-test-file", type=Path, required=True)
    parser.add_argument("--beam-size", type=int, default=10)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--per-reaction-csv", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--allow-short-beams", action="store_true")
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def mapped_reaction(reaction):
    reactants, _, product = reaction.split(">", 2)
    return f"{reactants}>>{product}"


def source_row_from_rxn_id(rxn_id):
    """Accept both released integer IDs and frozen CH6 adapter IDs."""
    value = str(rxn_id)
    if value.startswith("ch6_test_"):
        value = value.removeprefix("ch6_test_")
    return int(value)


def canonicalize_molecule(smiles):
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        mol = Chem.RemoveHs(mol)
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
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


def canonicalize_reaction(reaction):
    parts = str(reaction).split(">>")
    if len(parts) != 2:
        return None
    reactants = canonicalize_precursor_set(parts[0])
    products = canonicalize_precursor_set(parts[1])
    if reactants is None or products is None:
        return None
    return reactants, products


def finalize_block(blocks, block, beam_size, allow_partial, allow_short_beams):
    if block is None:
        return
    native_index = block["native_index"]
    if native_index in blocks:
        raise ValueError(f"Duplicate prediction block for native index {native_index}")
    beam_indices = [prediction["beam_index"] for prediction in block["predictions"]]
    if beam_indices != list(range(len(beam_indices))):
        raise ValueError(
            f"Non-contiguous beam indices for native index {native_index}: {beam_indices}"
        )
    if len(beam_indices) > beam_size:
        raise ValueError(
            f"Native index {native_index} has {len(beam_indices)} predictions; "
            f"maximum is {beam_size}"
        )
    if len(beam_indices) < beam_size and not (allow_partial or allow_short_beams):
        raise ValueError(
            f"Native index {native_index} has {len(beam_indices)} predictions; "
            f"expected {beam_size}"
        )
    blocks[native_index] = block


def parse_predictions(path, beam_size, allow_partial, allow_short_beams):
    blocks = {}
    block = None
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\r\n")
            header_match = HEADER_PATTERN.match(line)
            if header_match:
                finalize_block(
                    blocks, block, beam_size, allow_partial, allow_short_beams
                )
                block = {
                    "native_index": int(header_match.group(1)),
                    "reaction": header_match.group(2),
                    "predictions": [],
                }
                continue

            prediction_match = PREDICTION_PATTERN.match(line)
            if prediction_match:
                if block is None:
                    raise ValueError(
                        f"Prediction before the first reaction header at line {line_number}"
                    )
                block["predictions"].append(
                    {
                        "beam_index": int(prediction_match.group(1)),
                        "native_correct": prediction_match.group(2) == "True",
                        "probability": float(prediction_match.group(3)),
                        "smiles": prediction_match.group(4),
                    }
                )
    finalize_block(blocks, block, beam_size, allow_partial, allow_short_beams)
    return blocks


def load_native_test(repo_root, native_test_file, source_frame):
    sys.path.insert(0, str(repo_root.resolve()))
    native_items = joblib.load(native_test_file)
    source_to_native = {}
    reaction_mismatches = []
    for native_index, item in enumerate(native_items):
        source_row = source_row_from_rxn_id(item.rxn_id)
        if source_row < 0 or source_row >= len(source_frame):
            raise ValueError(f"Out-of-range rxn_id {source_row} at native index {native_index}")
        if source_row in source_to_native:
            raise ValueError(f"Duplicate native rxn_id {source_row}")
        source_to_native[source_row] = native_index
        expected_reaction = canonicalize_reaction(
            mapped_reaction(str(source_frame.iloc[source_row]["rxn_smiles"]))
        )
        native_reaction = canonicalize_reaction(str(item.rxn_smi))
        if expected_reaction is None or native_reaction != expected_reaction:
            reaction_mismatches.append(native_index)
    return native_items, source_to_native, reaction_mismatches


def ranked_canonical_predictions(predictions):
    ranked = []
    seen = set()
    invalid = 0
    duplicates = 0
    for raw_rank, prediction in enumerate(predictions, start=1):
        canonical = canonicalize_precursor_set(prediction["smiles"])
        if canonical is None:
            invalid += 1
            continue
        if canonical in seen:
            duplicates += 1
            continue
        seen.add(canonical)
        ranked.append((raw_rank, canonical))
    return ranked, invalid, duplicates


def hit_rank(ranked, target):
    return next((raw_rank for raw_rank, canonical in ranked if canonical == target), None)


def rates(hits, total):
    return {f"Top-{k}": hits[k] / total for k in TOP_K}


def class_metrics(class_hits, class_totals):
    metrics = {}
    for reaction_class in sorted(class_totals, key=lambda value: int(value)):
        metrics[reaction_class] = {
            "total_reactions": class_totals[reaction_class],
            **rates(class_hits[reaction_class], class_totals[reaction_class]),
        }
    return metrics


def main():
    args = parse_args()
    if args.beam_size < max(TOP_K):
        raise ValueError(f"--beam-size must be at least {max(TOP_K)}")

    RDLogger.DisableLog("rdApp.*")
    actual_test_hash = sha256(args.selected_test)
    if actual_test_hash != EXPECTED_TEST_SHA256:
        raise ValueError(
            f"Unexpected selected_test.csv SHA-256: {actual_test_hash}; "
            f"expected {EXPECTED_TEST_SHA256}"
        )

    source_frame = pd.read_csv(args.selected_test)
    required_columns = {"rxn_smiles", "class"}
    missing_columns = required_columns.difference(source_frame.columns)
    if missing_columns:
        raise ValueError(f"selected_test.csv lacks columns: {sorted(missing_columns)}")

    native_items, source_to_native, native_reaction_mismatches = load_native_test(
        args.repo_root, args.native_test_file, source_frame
    )
    blocks = parse_predictions(
        args.predictions,
        args.beam_size,
        args.allow_partial,
        args.allow_short_beams,
    )

    expected_native_indices = set(range(len(native_items)))
    unexpected_indices = sorted(set(blocks).difference(expected_native_indices))
    if unexpected_indices:
        raise ValueError(f"Unexpected native prediction indices: {unexpected_indices[:10]}")
    missing_native_indices = sorted(expected_native_indices.difference(blocks))
    if missing_native_indices and not args.allow_partial:
        raise ValueError(
            f"Missing {len(missing_native_indices)} native prediction blocks; "
            f"first indices: {missing_native_indices[:10]}"
        )
    short_beam_blocks = {
        native_index: args.beam_size - len(block["predictions"])
        for native_index, block in blocks.items()
        if len(block["predictions"]) < args.beam_size
    }

    header_reaction_mismatches = []
    native_hits = Counter()
    unified_hits = Counter()
    unified_class_hits = defaultdict(Counter)
    class_totals = Counter()
    unsupported_by_class = Counter()
    invalid_targets = []
    invalid_prediction_occurrences = 0
    duplicate_prediction_occurrences = 0
    rows = []

    for source_row, source_record in source_frame.iterrows():
        reaction_class = str(int(source_record["class"]))
        class_totals[reaction_class] += 1
        target_reactants = str(source_record["rxn_smiles"]).split(">", 1)[0]
        canonical_target = canonicalize_precursor_set(target_reactants)
        if canonical_target is None:
            invalid_targets.append(int(source_row))

        native_index = source_to_native.get(int(source_row))
        supported = native_index is not None
        if not supported:
            unsupported_by_class[reaction_class] += 1
        block = blocks.get(native_index) if supported else None
        predictions = block["predictions"] if block is not None else []

        if block is not None and block["reaction"] != str(native_items[native_index].rxn_smi):
            header_reaction_mismatches.append(native_index)

        ranked, invalid_count, duplicate_count = ranked_canonical_predictions(predictions)
        invalid_prediction_occurrences += invalid_count
        duplicate_prediction_occurrences += duplicate_count
        matched_rank = hit_rank(ranked, canonical_target) if canonical_target else None

        for k in TOP_K:
            if matched_rank is not None and matched_rank <= k:
                unified_hits[k] += 1
                unified_class_hits[reaction_class][k] += 1

        if block is not None:
            native_flags = [prediction["native_correct"] for prediction in predictions]
            for k in TOP_K:
                if any(native_flags[:k]):
                    native_hits[k] += 1

        rows.append(
            {
                "source_row": int(source_row),
                "native_index": "" if native_index is None else native_index,
                "class": reaction_class,
                "native_preprocessing_supported": supported,
                "prediction_block_present": block is not None,
                "prediction_count": len(predictions),
                "valid_unique_canonical_predictions": len(ranked),
                "invalid_prediction_count": invalid_count,
                "duplicate_canonical_prediction_count": duplicate_count,
                "canonical_target": canonical_target or "",
                "canonical_top1_prediction": next(
                    (canonical for raw_rank, canonical in ranked if raw_rank == 1), ""
                ),
                "canonical_match_rank": "" if matched_rank is None else matched_rank,
            }
        )

    if invalid_targets:
        raise ValueError(
            f"Invalid reference reactants for {len(invalid_targets)} rows: {invalid_targets[:10]}"
        )
    if native_reaction_mismatches:
        raise ValueError(
            f"Native test objects differ from frozen source rows at "
            f"{len(native_reaction_mismatches)} indices"
        )
    if header_reaction_mismatches:
        raise ValueError(
            f"Prediction headers differ from native test objects at "
            f"{len(header_reaction_mismatches)} indices"
        )

    native_total = len(native_items)
    source_total = len(source_frame)
    unsupported_source_rows = sorted(set(range(source_total)).difference(source_to_native))
    metrics = {
        "model": args.model_name,
        "beam_size": args.beam_size,
        "partial_evaluation": bool(args.allow_partial and missing_native_indices),
        "artifacts": {
            "predictions": str(args.predictions.resolve()),
            "selected_test": str(args.selected_test.resolve()),
            "selected_test_sha256": actual_test_hash,
            "native_test_file": str(args.native_test_file.resolve()),
        },
        "native_original": {
            "evaluation_semantics": f"original_{args.model_name}_prediction_is_correct_flags",
            "denominator": "native_preprocessing_successes_only",
            "total_reactions": native_total,
            "prediction_blocks_present": len(blocks),
            "missing_prediction_blocks": len(missing_native_indices),
            "short_prediction_blocks": len(short_beam_blocks),
            "missing_beam_candidates": sum(short_beam_blocks.values()),
            "short_beam_policy": "unfilled ranks receive no hit credit",
            "precursor_order": "ignored_by_python_set",
            "duplicate_precursors": "collapsed_as_set",
            "candidate_ranking": "original_beam_order",
            "metrics": rates(native_hits, native_total),
            "hits": {f"Top-{k}": native_hits[k] for k in TOP_K},
        },
        "legacy78_adapted_canonical": {
            "evaluation_semantics": "reaction_instance_canonical_precursor_set_exact_match",
            "reference_family": {
                "result": "GDiffRetro old-flow Stage1 gate-on Top-1 78.02%",
                "implementation": "code/known_class/stage2/fast_merge_eval.py",
                "preserved_principles": [
                    "full precursor identity exact match",
                    "cumulative Top-k hit reporting",
                    "invalid predictions never count as correct",
                ],
            },
            "model_aware_adaptations": {
                "evaluation_unit": (
                    "9250 frozen reaction rows; product-group collapsing is not transferred "
                    f"because {args.model_name} emits one ranked list per reaction"
                ),
                "stage1_gate": (
                    f"not applicable; {args.model_name} is single-stage and has no comparable "
                    "post-hoc center-rank gate"
                ),
                "oracle_component_merge": (
                    "not applicable; predictions are complete precursor sets"
                ),
                "candidate_ranking": (
                    "native beam order and original rank slots retained; canonical duplicates "
                    "keep their first occurrence without compacting later candidates"
                ),
                "native_preprocessing_failures": "counted as failures in the full denominator",
            },
            "canonicalization": {
                "toolkit": f"RDKit {Chem.rdBase.rdkitVersion}",
                "graph_sanitization": "enabled",
                "atom_mapping": "removed",
                "explicit_hydrogens": "removed",
                "stereochemistry": "preserved",
                "precursor_order": "ignored_by_sorting",
                "duplicate_precursors": "collapsed_as_set",
                "invalid_candidates": "no_hit_credit_original_rank_slot_preserved",
                "duplicate_beam_candidates": (
                    "first_canonical_occurrence_retained_later_occurrences_consume_rank_slots"
                ),
            },
            "denominator": "all_frozen_test_reaction_rows",
            "total_reactions": source_total,
            "native_preprocessing_supported": native_total,
            "native_preprocessing_failures": len(unsupported_source_rows),
            "unsupported_source_rows": unsupported_source_rows,
            "unsupported_by_class": dict(sorted(unsupported_by_class.items())),
            "prediction_blocks_present": len(blocks),
            "missing_prediction_blocks": len(missing_native_indices),
            "short_prediction_blocks": len(short_beam_blocks),
            "missing_beam_candidates": sum(short_beam_blocks.values()),
            "short_beam_policy": "unfilled ranks receive no hit credit",
            "invalid_prediction_occurrences": invalid_prediction_occurrences,
            "duplicate_canonical_prediction_occurrences": duplicate_prediction_occurrences,
            "metrics": rates(unified_hits, source_total),
            "hits": {f"Top-{k}": unified_hits[k] for k in TOP_K},
            "per_class": class_metrics(unified_class_hits, class_totals),
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.per_reaction_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(args.per_reaction_csv, index=False)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
