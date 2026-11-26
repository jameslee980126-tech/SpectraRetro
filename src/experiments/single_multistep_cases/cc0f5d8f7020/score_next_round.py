"""Product-only Stage1 probe of the three targets and their recovered precursors.

This does not generate completed reactants and is not a multistep success claim.
Literature bond annotations are attached only AFTER inference, for comparison.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
spec = importlib.util.spec_from_file_location("frozen_stage1", PARENT / "extract_stage1_bond_probabilities.py")
frozen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frozen)
torch = frozen.torch
from rdkit import Chem
from torchdrug import data, datasets, models, tasks


def unmapped(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(smiles)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol)


def make_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    for i, atom in enumerate(mol.GetAtoms()):
        atom.SetAtomMapNum(i + 1)
    graph = data.Molecule.from_molecule(mol, atom_feature="center_identification", bond_feature="default", kekulize=True)
    return mol, graph


def main():
    torch.set_num_threads(2)
    assert frozen.sha256(frozen.CHECKPOINT) == frozen.EXPECTED_SHA256
    evidence_path = PARENT / "large_literature_cases" / "evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    # Existing data used only to instantiate exactly the original network heads.
    ds = datasets.USPTO50k(str(PARENT / "stage1_large_cases_probabilities_inputs.csv"),
                          atom_feature="center_identification", as_synthon=False,
                          with_hydrogen=False, kekulize=True, verbose=0)
    model = models.FSpecFaithfulReactionCenter(input_dim=ds.node_feature_dim, hidden_dims=[512] * 3,
        edge_input_dim=ds.edge_feature_dim, max_distance=5, dropout=0.0)
    task = tasks.CenterIdentification(model, None, feature=("graph", "atom", "bond"),
                                     mlp_hidden_dim=512, fusion_mode="raw")
    task.preprocess(ds, [], [])
    payload = torch.load(frozen.CHECKPOINT, map_location="cpu")
    assert payload["metadata"]["run_signature_payload"]["reaction_class_mode"] == "unknown"
    task.load_state_dict(payload["model"], strict=True)
    task.eval().cpu()
    inputs = frozen.pd.read_csv(PARENT / "stage1_large_cases_probabilities_inputs.csv")
    previous_scores = json.loads((PARENT / "stage1_large_cases_probabilities.json").read_text(encoding="utf-8"))
    outputs = []
    for case in evidence["cases"]:
        cid = case["source_row"]
        source = case["source"]
        archived_hit = next(row for row in case["raw_and_restored_top10"] if row["rank"] == case["rank"])
        assert unmapped(archived_hit["h_filled"]) == unmapped(source["precursor2"] + "." + source["precursor3"])
        root = unmapped(inputs.loc[inputs._source_row == cid, "product"].iloc[0])
        for name, smiles in [("root", root), ("precursor2", source["precursor2"]), ("precursor3", source["precursor3"])]:
            mol, graph = make_graph(smiles)
            # No true reactant, reaction class, reference center, or labels passed.
            batch = {"graph": (None, data.Molecule.pack([graph]))}
            with torch.no_grad():
                logits = task.predict(batch).cpu()
            prob = logits.softmax(0)
            directed = graph.directed()
            candidates = []
            for idx, (a, b, _) in enumerate(directed.edge_list.tolist()):
                bond = mol.GetBondBetweenAtoms(a, b)
                candidates.append(dict(kind="bond", atom_indices=[a, b],
                    elements=[mol.GetAtomWithIdx(a).GetSymbol(), mol.GetAtomWithIdx(b).GetSymbol()],
                    bond_type=str(bond.GetBondType()), in_ring=bond.IsInRing(),
                    probability=float(prob[idx])))
            for a in range(mol.GetNumAtoms()):
                candidates.append(dict(kind="atom", atom_indices=[a], elements=[mol.GetAtomWithIdx(a).GetSymbol()],
                    probability=float(prob[len(directed.edge_list) + a])))
            assert len(candidates) == len(prob)
            ranked = sorted(candidates, key=lambda r: -r["probability"])
            for rank, row in enumerate(ranked, 1):
                row["rank"] = rank
            if name == "root":
                # Control: independent product-only call reproduces archived score.
                expected = previous_scores["cases"][str(cid)]["reference_probability"]
                assert abs(ranked[0]["probability"] - expected) < 1e-6
            # Scoring finished. Literature annotation is never an inference input.
            literature = None
            if name == "precursor3":
                if cid == 8719:
                    match = mol.GetSubstructMatch(Chem.MolFromSmarts("[S]-[CH2]-[CH2]-[O]-[c]"))
                    wanted = sorted([match[2], match[3]])
                    reaction = "Mitsunobu ether formation (not a C-H functionalization step)"
                else:
                    match = mol.GetSubstructMatch(Chem.MolFromSmarts("[C](=[O])-[N]"))
                    wanted = sorted([match[0], match[2]])
                    reaction = "Amide coupling (not a C-H functionalization step)"
                chosen = next(r for r in ranked if r["kind"] == "bond" and sorted(r["atom_indices"]) == wanted)
                literature = dict(chosen, reaction=reaction)
            record = dict(case_id=cid, node=name, smiles=smiles, heavy_atoms=mol.GetNumHeavyAtoms(),
                parent_conditional_precursor_rank=case["rank"] if name != "root" else None,
                literature_upstream_center= literature, candidates=ranked)
            outputs.append(record)
            print(json.dumps({k:v for k,v in record.items() if k != "candidates"}, ensure_ascii=False), flush=True)
            print("TOP1", json.dumps(ranked[0]), flush=True)
    result = dict(status="stage1_only_next_round_probe", checkpoint_sha256=frozen.EXPECTED_SHA256,
        inputs="Product graph only; no reaction labels or true reactants passed to predict.",
        caveat="Child nodes are selected from previously hydrogen-restored conditional Stage2 hits, not a new product-only Stage2 rollout.",
        stage2_rerun=False, training=False, evidence_sha256=hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
        nodes=outputs)
    (HERE / "stage1_next_round.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
