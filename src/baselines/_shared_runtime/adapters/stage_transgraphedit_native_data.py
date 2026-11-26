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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_split(path: Path) -> pd.DataFrame:
    actual_hash = sha256(path)
    expected_hash = EXPECTED_HASHES[path.name]
    if actual_hash != expected_hash:
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


def rename_mode_files(directory: Path, source_mode: str, target_mode: str) -> None:
    for path in list(directory.iterdir()):
        if path.name.startswith(f"{source_mode}.") or path.name.startswith(f"{source_mode}_"):
            path.rename(directory / path.name.replace(source_mode, target_mode, 1))


def call_native_preprocessing(preprocess_module, frame: pd.DataFrame, mode: str, dataset: str) -> None:
    native_args = SimpleNamespace(dataset=dataset, mode=mode, kekulize=True, print_every=1000)
    preprocess_module.preprocessing(
        rxns=frame["mapped_rxn"],
        args=native_args,
        rxn_classes=frame["class"],
        rxns_id=frame["id"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="C-H Arylation")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=9)
    parser.add_argument("--skip-tensors", action="store_true")
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

    data_root = repo_root / "data" / args.dataset
    for split in ("train", "valid", "test"):
        shutil.rmtree(data_root / split, ignore_errors=True)
    data_root.mkdir(parents=True, exist_ok=True)

    previous_cwd = Path.cwd()
    try:
        os_chdir = __import__("os").chdir
        os_chdir(repo_root)

        call_native_preprocessing(preprocess, frames["train"], "test", args.dataset)
        (data_root / "test").rename(data_root / "train")
        rename_mode_files(data_root / "train", "test", "train")

        call_native_preprocessing(preprocess, frames["valid"], "valid", args.dataset)
        (data_root / "valid").rename(data_root / "valid_saved")
        call_native_preprocessing(preprocess, frames["test"], "valid", args.dataset)
        (data_root / "valid").rename(data_root / "test")
        rename_mode_files(data_root / "test", "valid", "test")
        (data_root / "valid_saved").rename(data_root / "valid")

        if not args.skip_tensors:
            native_args = SimpleNamespace(
                dataset=args.dataset,
                mode="train",
                use_rxn_class=True,
                batch_size=args.batch_size,
                max_steps=args.max_steps,
                print_every=1000,
            )
            prepare_data.prepare_data(native_args)
    finally:
        __import__("os").chdir(previous_cwd)

    manifest = {
        "adapter": "native TransGraphEdit preprocessing functions; external source unchanged",
        "dataset": args.dataset,
        "known_class": True,
        "max_steps": args.max_steps,
        "source_hashes": EXPECTED_HASHES,
        "source_rows": {split: len(frame) for split, frame in frames.items()},
        "native_rows": {
            split: len(joblib.load(data_root / split / f"{split}.file.kekulized"))
            for split in frames
        },
        "vocabulary_source": "training split only",
    }
    (data_root / "ch6_69pct_adapter_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
