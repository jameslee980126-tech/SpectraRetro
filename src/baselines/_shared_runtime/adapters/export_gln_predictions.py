#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

from rdkit import Chem


VAL_PATTERN = re.compile(r"val-(\d+)\.summary$")


def canonicalize(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return ""
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(molecule, isomericSmiles=True)


def top1_from_summary(path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("top 1:"):
            return float(line.split(":", 1)[1].strip())
    raise ValueError(f"missing Top-1 in {path}")


def select_checkpoint(results_dir):
    candidates = []
    for path in results_dir.glob("val-*.summary"):
        match = VAL_PATTERN.match(path.name)
        if match is not None:
            candidates.append((top1_from_summary(path), int(match.group(1))))
    if not candidates:
        raise ValueError(f"no validation summaries found in {results_dir}")
    return max(candidates, key=lambda item: (item[0], -item[1]))


def parse_predictions(path, beam_size):
    lines = path.read_text(encoding="utf-8").splitlines()
    records = []
    cursor = 0
    source_row = 0
    while cursor < len(lines):
        header = lines[cursor].split()
        cursor += 1
        if len(header) < 3:
            raise ValueError(f"invalid GLN prediction header at row {source_row}")
        reaction = header[-2]
        count = int(header[-1])
        predictions = []
        for _ in range(count):
            item = lines[cursor].split(maxsplit=1)
            cursor += 1
            predictions.append(item[1] if len(item) == 2 else "")
        target = canonicalize(reaction.split(">", 1)[0])
        canonical_predictions = [canonicalize(value) for value in predictions[:beam_size]]
        try:
            native_rank = canonical_predictions.index(target) + 1
        except ValueError:
            native_rank = None
        records.append(
            {
                "source_row": source_row,
                "predictions": predictions[:beam_size],
                "native_supported": True,
                "native_hit_rank": native_rank,
            }
        )
        source_row += 1
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--beam-size", type=int, default=10)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--selection-json", type=Path, required=True)
    args = parser.parse_args()

    validation_top1, checkpoint_index = select_checkpoint(args.results_dir)
    prediction_path = args.results_dir / f"test-{checkpoint_index}.pred"
    if not prediction_path.is_file():
        raise FileNotFoundError(prediction_path)
    records = parse_predictions(prediction_path, args.beam_size)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    selection = {
        "selection_rule": "highest validation native Top-1; earliest checkpoint breaks ties",
        "checkpoint_index": checkpoint_index,
        "validation_top1": validation_top1,
        "prediction_file": str(prediction_path),
        "rows": len(records),
    }
    args.selection_json.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
