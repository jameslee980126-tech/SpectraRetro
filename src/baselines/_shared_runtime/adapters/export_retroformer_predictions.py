#!/usr/bin/env python3
import argparse
import json
import pickle
import sys
from pathlib import Path

import lmdb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retroformer-root", type=Path, required=True)
    parser.add_argument("--test-lmdb", type=Path, required=True)
    parser.add_argument("--generation-pickle", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.retroformer_root.resolve()))
    from retroformer.utils.smiles_utils import canonical_smiles

    with args.generation_pickle.open("rb") as handle:
        ground_truths, generations = pickle.load(handle)

    environment = lmdb.open(
        str(args.test_lmdb),
        max_readers=1,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        subdir=args.test_lmdb.is_dir(),
    )
    with environment.begin(write=False) as transaction:
        keys = list(transaction.cursor().iternext(values=False))
    environment.close()

    if len(keys) != len(ground_truths) or len(keys) != len(generations):
        raise ValueError(
            f"length mismatch: keys={len(keys)}, ground_truths={len(ground_truths)}, "
            f"generations={len(generations)}"
        )

    records = []
    for key, ground_truth, candidates in zip(keys, ground_truths, generations):
        source_row = int(key.decode("utf-8").split(" ", 1)[0])
        predictions = [str(candidate) for candidate in candidates]
        native_target = canonical_smiles(str(ground_truth))
        native_candidates = [canonical_smiles(candidate) for candidate in predictions]
        try:
            native_hit_rank = native_candidates.index(native_target) + 1
        except ValueError:
            native_hit_rank = None
        records.append(
            {
                "source_row": source_row,
                "predictions": predictions,
                "native_supported": True,
                "native_hit_rank": native_hit_rank,
            }
        )

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8", newline="\n") as handle:
        for record in sorted(records, key=lambda item: item["source_row"]):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"exported {len(records)} Retroformer prediction rows")


if __name__ == "__main__":
    main()
