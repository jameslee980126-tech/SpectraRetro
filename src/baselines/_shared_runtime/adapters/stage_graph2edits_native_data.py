#!/usr/bin/env python3
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import joblib
import pandas as pd


EXPECTED_HASHES = {
    "selected_train.csv": "957E27D354D6B82F6BC1EAB1FDA8F5EC365517BEDE2E4D31112C3677CA50F648",
    "selected_eval.csv": "9731BD8CFA1E6865DEEA5B5498184A823E37908C570769D8D75A4B6152860503",
    "selected_test.csv": "9FD0E110A733462561BE5E1C3BE01B45E7B4F2A9C2488D7A7646DC50F9A00DE4",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_split(path):
    actual_hash = sha256(path)
    if actual_hash != EXPECTED_HASHES[path.name]:
        raise ValueError(f"Unexpected SHA-256 for {path}: {actual_hash}")
    frame = pd.read_csv(path)
    reaction_parts = frame["rxn_smiles"].str.split(">", n=2, expand=True)
    return pd.DataFrame(
        {
            "id": frame.index.astype(int),
            "class": frame["class"].astype(int),
            "mapped_rxn": reaction_parts[0] + ">>" + reaction_parts[2],
        }
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=9)
    parser.add_argument("--replace-existing", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    source_dir = args.source_dir.resolve()
    sys.path.insert(0, str(repo_root))
    import prepare_data
    import preprocess

    frames = {
        "train": load_split(source_dir / "selected_train.csv"),
        "valid": load_split(source_dir / "selected_eval.csv"),
        "test": load_split(source_dir / "selected_test.csv"),
    }
    data_root = repo_root / "data" / "uspto_50k"
    if data_root.exists():
        if not args.replace_existing:
            raise FileExistsError(f"refusing to replace existing data directory: {data_root}")
        shutil.rmtree(data_root)
    data_root.mkdir(parents=True)

    previous_cwd = Path.cwd()
    try:
        __import__("os").chdir(repo_root)
        for mode in ("train", "valid", "test"):
            native_args = SimpleNamespace(
                dataset="uspto_50k",
                mode=mode,
                kekulize=True,
                print_every=1000,
            )
            preprocess.preprocessing(
                rxns=frames[mode]["mapped_rxn"],
                args=native_args,
                rxn_classes=frames[mode]["class"],
                rxns_id=frames[mode]["id"],
            )

        prepare_data.prepare_data(
            SimpleNamespace(
                dataset="uspto_50k",
                mode="train",
                use_rxn_class=True,
                batch_size=args.batch_size,
                max_steps=args.max_steps,
                print_every=1000,
            )
        )
    finally:
        __import__("os").chdir(previous_cwd)

    manifest = {
        "model": "Graph2Edits",
        "source_commit": "b12e97391c378e46e79e617d80017e95b21bba65",
        "dataset_alias": "uspto_50k",
        "known_class": True,
        "max_steps": args.max_steps,
        "source_hashes": EXPECTED_HASHES,
        "source_rows": {split: len(frame) for split, frame in frames.items()},
        "native_rows": {
            split: len(joblib.load(data_root / split / f"{split}.file.kekulized"))
            for split in frames
        },
        "vocabulary_source": "training_split_only",
        "external_source_changed": False,
    }
    manifest_path = data_root / "ch6_69pct_adapter_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
