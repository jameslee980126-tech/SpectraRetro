#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem


EXPECTED_MANIFEST_SHA256 = "9fd0e110a733462561be5e1c3be01b45e7b4f2a9c2488d7a7646dc50f9a00de4"
EXPECTED_DATA_SHA256 = "e789078a3982050c2f9ae972856c1b11a6da85c505cbac15b02e0c149db543ad"
EXPECTED_SPLIT_COUNTS = {"train": 74146, "val": 9395, "test": 9250}
TOP_K = (1, 3, 5, 10, 20, 50)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    return AllChem.GetMorganFingerprint(molecule, 2, useFeatures=True)


def native_precursor_target(reaction):
    molecule = Chem.MolFromSmiles(reaction.split(">", 1)[0])
    if molecule is None:
        return None
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    smiles = Chem.MolToSmiles(molecule, True)
    molecule = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(molecule, True) if molecule is not None else None


def load_rows(path):
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"source_row", "class", "rxn_smiles", "prod_smiles", "dataset"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"missing columns in {path}: {sorted(required)}")
        for row in reader:
            record = dict(row)
            record["source_row"] = int(record["source_row"])
            record["class"] = int(record["class"])
            record["prod_fp"] = fingerprint(record["prod_smiles"])
            rows.append(record)
    return rows


def prepare_precedent(row, process_an_example):
    prepared = dict(row)
    prepared["react_fp"] = fingerprint(row["rxn_smiles"].split(">", 1)[0])
    try:
        retro_canonical = process_an_example(row["rxn_smiles"], super_general=True)
    except Exception:
        retro_canonical = None
    prepared["template"] = (
        "(" + retro_canonical.replace(">>", ")>>") if retro_canonical is not None else None
    )
    return prepared


def run_one(test_row, candidate_training, rdchiral_classes, max_precedents, beam_size):
    rdchiral_reaction, rdchiral_reactants, rdchiral_run = rdchiral_classes
    if test_row["prod_fp"] is None:
        return {
            "source_row": test_row["source_row"],
            "predictions": [],
            "native_supported": False,
            "native_hit_rank": None,
        }

    similarities = DataStructs.BulkTanimotoSimilarity(
        test_row["prod_fp"], [row["prod_fp"] for row in candidate_training]
    )
    ordering = np.argsort(similarities)[::-1][:max_precedents]
    product = rdchiral_reactants(test_row["prod_smiles"])
    scores = {}

    for position in ordering:
        precedent = candidate_training[int(position)]
        template = precedent["template"]
        reactant_fingerprint = precedent["react_fp"]
        if template is None or reactant_fingerprint is None:
            continue
        try:
            outcomes = rdchiral_run(
                rdchiral_reaction(template), product, combine_enantiomers=False
            )
        except Exception:
            continue
        for precursors in outcomes:
            precursor_fingerprint = fingerprint(precursors)
            if precursor_fingerprint is None:
                continue
            precursor_similarity = DataStructs.TanimotoSimilarity(
                precursor_fingerprint, reactant_fingerprint
            )
            score = float(precursor_similarity) * float(similarities[int(position)])
            scores[precursors] = max(scores.get(precursors, float("-inf")), score)

    ranked = [
        precursors
        for precursors, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)[
            :beam_size
        ]
    ]
    target = native_precursor_target(test_row["rxn_smiles"])
    try:
        native_hit_rank = ranked.index(target) + 1
    except ValueError:
        native_hit_rank = None
    return {
        "source_row": test_row["source_row"],
        "predictions": ranked,
        "native_supported": True,
        "native_hit_rank": native_hit_rank,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrosim-root", type=Path, required=True)
    parser.add_argument("--rdchiral-root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--selected-test", type=Path, required=True)
    parser.add_argument("--predictions-jsonl", type=Path, required=True)
    parser.add_argument("--native-metrics", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--max-precedents", type=int, default=100)
    parser.add_argument("--beam-size", type=int, default=50)
    args = parser.parse_args()

    if sha256(args.selected_test) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("selected_test.csv SHA-256 does not match the frozen split")
    data_hash = sha256(args.data)
    if data_hash != EXPECTED_DATA_SHA256:
        raise ValueError("RetroSim data_processed.csv SHA-256 does not match the frozen adapter")
    retrosim_root = args.retrosim_root.resolve()
    rdchiral_root = args.rdchiral_root.resolve()
    sys.path.insert(0, str(retrosim_root))
    sys.path.insert(0, str(rdchiral_root))
    import rdchiral

    imported_rdchiral = Path(rdchiral.__file__).resolve()
    if rdchiral_root not in imported_rdchiral.parents:
        raise ValueError(f"unexpected rdchiral runtime: {imported_rdchiral}")
    from rdchiral.main import rdchiralReaction, rdchiralReactants, rdchiralRun
    from retrosim.utils.generate_retro_templates import process_an_example

    RDLogger.DisableLog("rdApp.*")
    rows = load_rows(args.data)
    split_counts = Counter(row["dataset"] for row in rows)
    if dict(split_counts) != EXPECTED_SPLIT_COUNTS:
        raise ValueError(f"Unexpected RetroSim split counts: {dict(split_counts)}")
    for split, expected_count in EXPECTED_SPLIT_COUNTS.items():
        source_rows = sorted(row["source_row"] for row in rows if row["dataset"] == split)
        if source_rows != list(range(expected_count)):
            raise ValueError(f"RetroSim {split} source rows do not cover the frozen split")
    training_rows = [row for row in rows if row["dataset"] == "train"]
    test_rows = sorted(
        (row for row in rows if row["dataset"] == "test"), key=lambda row: row["source_row"]
    )
    if [row["source_row"] for row in test_rows] != list(range(len(test_rows))):
        raise ValueError("RetroSim test source rows are not contiguous frozen row indices")

    print(f"precomputing {len(training_rows)} training templates and reactant fingerprints")
    prepared_training = Parallel(n_jobs=args.jobs, verbose=5, prefer="threads")(
        delayed(prepare_precedent)(row, process_an_example) for row in training_rows
    )
    training_by_class = defaultdict(list)
    for row in prepared_training:
        if row["prod_fp"] is not None:
            training_by_class[row["class"]].append(row)

    results = Parallel(n_jobs=args.jobs, verbose=5, prefer="threads")(
        delayed(run_one)(
            test_row,
            training_by_class[test_row["class"]],
            (rdchiralReaction, rdchiralReactants, rdchiralRun),
            args.max_precedents,
            args.beam_size,
        )
        for test_row in test_rows
    )

    args.predictions_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.predictions_jsonl.open("w", encoding="utf-8", newline="\n") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    hits = Counter()
    class_hits = defaultdict(Counter)
    class_totals = Counter()
    supported = 0
    for test_row, result in zip(test_rows, results):
        reaction_class = str(test_row["class"])
        class_totals[reaction_class] += 1
        if result["native_supported"]:
            supported += 1
        hit_rank = result["native_hit_rank"]
        for k in TOP_K:
            if hit_rank is not None and hit_rank <= k:
                hits[k] += 1
                class_hits[reaction_class][k] += 1

    metrics = {
        "model": "RetroSim",
        "configuration": {
            "reaction_class": "known",
            "fingerprint": "Morgan2Feat",
            "similarity": "Tanimoto",
            "max_precedents": args.max_precedents,
            "candidate_limit": args.beam_size,
            "knowledge_base": "frozen_training_split_only",
            "rdchiral_runtime": "official_1.1.0_python3_compatibility_release",
        },
        "input_audit": {
            "data_sha256": data_hash,
            "selected_test_sha256": EXPECTED_MANIFEST_SHA256,
            "split_counts": EXPECTED_SPLIT_COUNTS,
            "test_set_used_for_training_or_selection": False,
        },
        "total_test_rows": len(test_rows),
        "supported_test_rows": supported,
        "metrics_all_frozen_rows": {
            f"Top-{k}": hits[k] / len(test_rows) for k in TOP_K
        },
        "by_class": {
            reaction_class: {
                "total": class_totals[reaction_class],
                **{
                    f"Top-{k}": class_hits[reaction_class][k]
                    / class_totals[reaction_class]
                    for k in TOP_K
                },
            }
            for reaction_class in sorted(class_totals, key=int)
        },
    }
    args.native_metrics.parent.mkdir(parents=True, exist_ok=True)
    args.native_metrics.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
