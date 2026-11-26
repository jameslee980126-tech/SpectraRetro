#!/usr/bin/env python3
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
import yaml
from rdkit import Chem, RDLogger
from tqdm import tqdm


TOP_K = (1, 3, 5, 10)


def load_checkpoint(exp_dir, experiment, step, device, fallback_model_name=None):
    if "run" in experiment:
        files_dir = exp_dir / "wandb" / experiment / "files"
        checkpoint = torch.load(files_dir / f"{step}.pt", map_location=device)
        config_file = files_dir / "config.yaml"
        if config_file.is_file():
            with config_file.open() as handle:
                config = yaml.load(handle, Loader=yaml.FullLoader)
            model_name = config["model"]["value"]
        elif fallback_model_name:
            model_name = fallback_model_name
        else:
            raise FileNotFoundError(
                f"{config_file} is missing; provide the checkpoint model name explicitly"
            )
    else:
        checkpoint = torch.load(
            exp_dir / experiment / "checkpoints" / f"{step}.pt", map_location=device
        )
        model_name = experiment.split("_")[0]
    return checkpoint, model_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphretro-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--exp-dir", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--edits-exp", required=True)
    parser.add_argument("--lg-exp", required=True)
    parser.add_argument("--edits-model-name")
    parser.add_argument("--lg-model-name")
    parser.add_argument("--edits-step", default="best_model")
    parser.add_argument("--lg-step", default="best_model")
    parser.add_argument("--beam-width", type=int, default=10)
    parser.add_argument("--use-rxn-class", action="store_true")
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--native-metrics", type=Path, required=True)
    args = parser.parse_args()

    os.environ["SEQ_GRAPH_RETRO"] = str(args.graphretro_root.resolve())
    sys.path.insert(0, str(args.graphretro_root.resolve()))
    from seq_graph_retro.models import EditLGSeparate
    from seq_graph_retro.search import BeamSearch
    from seq_graph_retro.utils.edit_mol import canonicalize, generate_reac_set

    RDLogger.DisableLog("rdApp.*")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_frame = pd.read_csv(args.test_file)
    edits_loaded, edit_net_name = load_checkpoint(
        args.exp_dir,
        args.edits_exp,
        args.edits_step,
        device,
        fallback_model_name=args.edits_model_name,
    )
    lg_loaded, lg_net_name = load_checkpoint(
        args.exp_dir,
        args.lg_exp,
        args.lg_step,
        device,
        fallback_model_name=args.lg_model_name,
    )
    edits_config = edits_loaded["saveables"]
    lg_config = lg_loaded["saveables"]
    lg_toggles = lg_config["toggles"]
    if "tensor_file" in lg_config and not Path(lg_config["tensor_file"]).is_file():
        class_dir = "with_rxn" if lg_toggles.get("use_rxn_class", False) else "without_rxn"
        lg_config["tensor_file"] = str(
            args.data_dir / "train" / "h_labels" / class_dir / "lg_inputs.pt"
        )

    model = EditLGSeparate(
        edits_config=edits_config,
        lg_config=lg_config,
        edit_net_name=edit_net_name,
        lg_net_name=lg_net_name,
        device=device,
    )
    model.load_state_dict(edits_loaded["state"], lg_loaded["state"])
    model.to(device)
    model.eval()
    beam_model = BeamSearch(model=model, beam_width=args.beam_width, max_edits=1)

    records = []
    native_hits = Counter()
    for _, row in tqdm(test_frame.iterrows(), total=len(test_frame)):
        source_row = int(row["id"])
        reactants, product = row["reactants>reagents>production"].split(">>")
        native_target = set(canonicalize(reactants).split("."))
        reaction_class = int(row["class"])
        predictions = []
        native_hit_rank = None
        try:
            product_molecule = Chem.MolFromSmiles(canonicalize(product))
            for atom in product_molecule.GetAtoms():
                atom.SetAtomMapNum(atom.GetIdx() + 1)
            canonical_product = Chem.MolToSmiles(product_molecule)
            if args.use_rxn_class:
                nodes = beam_model.run_search(
                    canonical_product, max_steps=6, rxn_class=reaction_class
                )
            else:
                nodes = beam_model.run_search(canonical_product, max_steps=6)
            for node in nodes[: args.beam_width]:
                predicted_edit = node.edit[0] if isinstance(node.edit, list) else node.edit
                predicted_set = generate_reac_set(
                    canonical_product, predicted_edit, node.lg_groups, verbose=False
                )
                if predicted_set is None:
                    predictions.append("")
                    continue
                predictions.append(".".join(sorted(predicted_set)))
                if native_hit_rank is None and predicted_set == native_target:
                    native_hit_rank = len(predictions)
        except Exception:
            predictions = []
            native_hit_rank = None

        for k in TOP_K:
            if native_hit_rank is not None and native_hit_rank <= k:
                native_hits[k] += 1
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

    metrics = {
        "model": "GraphRetro",
        "beam_width": args.beam_width,
        "reaction_class": "known" if args.use_rxn_class else "unknown",
        "total_native_test_rows": len(records),
        **{f"Top-{k}": native_hits[k] / len(records) for k in TOP_K},
    }
    args.native_metrics.parent.mkdir(parents=True, exist_ok=True)
    args.native_metrics.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
