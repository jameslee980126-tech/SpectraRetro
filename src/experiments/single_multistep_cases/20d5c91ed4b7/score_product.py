"""Frozen Stage1 product-only ranking, isolated in the Stage1 environment."""
import json
import sys
from pathlib import Path
from score_next_round import frozen, torch, make_graph, data, datasets, models, tasks

def main():
    inp, out = map(Path, sys.argv[1:3])
    torch.set_num_threads(2)
    assert frozen.sha256(frozen.CHECKPOINT) == frozen.EXPECTED_SHA256
    ds = datasets.USPTO50k(str(frozen.HERE / "stage1_large_cases_probabilities_inputs.csv"),
        atom_feature="center_identification", as_synthon=False, with_hydrogen=False, kekulize=True, verbose=0)
    model = models.FSpecFaithfulReactionCenter(input_dim=ds.node_feature_dim, hidden_dims=[512]*3,
        edge_input_dim=ds.edge_feature_dim, max_distance=5, dropout=0.0)
    task = tasks.CenterIdentification(model, None, feature=("graph","atom","bond"), mlp_hidden_dim=512, fusion_mode="raw")
    task.preprocess(ds, [], [])
    payload = torch.load(frozen.CHECKPOINT, map_location="cpu")
    assert payload["metadata"]["run_signature_payload"]["reaction_class_mode"] == "unknown"
    task.load_state_dict(payload["model"], strict=True)
    task.eval().cpu()
    results=[]
    for node in json.loads(inp.read_text(encoding="utf-8")):
        mol, graph = make_graph(node["smiles"])
        with torch.no_grad():
            logits=task.predict({"graph":(None,data.Molecule.pack([graph]))}).cpu()
        probabilities=logits.softmax(0).tolist()
        candidates=[]
        for idx,(a,b,_) in enumerate(graph.directed().edge_list.tolist()):
            bond=mol.GetBondBetweenAtoms(a,b)
            candidates.append(dict(kind="bond",atom_indices=[a,b],in_ring=bond.IsInRing(),
                bond_type=str(bond.GetBondType()),probability=probabilities[idx]))
        offset=len(candidates)
        for a in range(mol.GetNumAtoms()):
            candidates.append(dict(kind="atom",atom_indices=[a],probability=probabilities[offset+a]))
        candidates.sort(key=lambda x:-x["probability"])
        for rank,c in enumerate(candidates,1):c["rank"]=rank
        results.append(dict(node, candidates=candidates))
    out.write_text(json.dumps(dict(checkpoint_sha256=frozen.EXPECTED_SHA256,nodes=results),indent=2),encoding="utf-8")

if __name__ == "__main__":main()
