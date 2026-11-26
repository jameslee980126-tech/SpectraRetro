"""Read-only FSpec diagnostics. No optimizer step, dataset training or test evaluation.

Uses a saved checkpoint, synthetic molecules and existing tiny fixtures.
Official class definitions are loaded from the pinned local upstream source;
only dependency imports are omitted. The model/training source is not modified.
"""

import ast
import copy
import json
from pathlib import Path
import random
from types import SimpleNamespace
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from rdkit import Chem

from torchdrug import data, datasets, models, tasks
from torchdrug.models.fspec_faithful_rc import _OfficialBivariateSpectralLayer


STAGE = Path(__file__).resolve().parent
CHECKPOINT = STAGE / "model/ch6_fspec_faithful_product_only_unknown_512x3_seed0_20260907.pth"
SMILES = (
    "CCO", "c1ccccc1", "c1ccc2ccccc2c1", "C1CC2CCC1C2",
    "Cc1ccccc1C(=O)N", "COc1ccc(C(=O)N)cc1", "c1ccncc1",
    "c1ccc2[nH]ccc2c1", "C1CCC2(CC1)CCCCC2", "C12C3C4C1C5C2C3C45",
)


def emit(name, value):
    print(json.dumps({"check": name, "result": value}, ensure_ascii=False), flush=True)


def make_graph(mol):
    return data.Molecule.from_molecule(
        mol, atom_feature="center_identification", bond_feature="default",
        with_hydrogen=False, kekulize=True,
    ).directed()


class SymmetrizeFrequency(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x):
        y = self.encoder(x)
        return (y + y.transpose(-3, -2)) / 2


def official_parity():
    path = STAGE / "fspec_reference_encoder.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    namespace = {"torch": torch, "nn": nn, "F": F, "Optional": Optional}
    exec(compile(tree, str(path), "exec"), namespace)
    args = SimpleNamespace(task=("audit", "e"))
    local = _OfficialBivariateSpectralLayer(16, 4).double().eval()
    official = namespace["BSGL"](args, 16, 16).double().eval()
    # Align the declared task adaptations; execute upstream forward unchanged.
    official.enc_E = SymmetrizeFrequency(copy.deepcopy(local.frequency_encoder))
    official.linear = copy.deepcopy(local.linear)
    official.enc = copy.deepcopy(local.pair_edge_embedding)
    official.mlp = copy.deepcopy(local.mlp)
    official.eps.data.copy_(local.eps.data)
    u, _ = torch.linalg.qr(torch.randn(7, 7, dtype=torch.double))
    e = torch.linspace(0, 2, 7, dtype=torch.double)
    s = torch.randn(7, 7, 16, dtype=torch.double)
    a = torch.randn(7, 7, 4, dtype=torch.double)
    with torch.no_grad():
        y1 = local(e, u, s, a)
        y2 = official(args, e[None], u[None], s[None], torch.ones(1, 7, 7), a[None])[0]
    emit("upstream_forward_parity_after_declared_adaptations", float((y1-y2).abs().max()))


def main():
    torch.manual_seed(0)
    torch.set_num_threads(2)
    official_parity()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fixture = datasets.USPTO50k(
        str(STAGE / "tests/fixtures/fspec_tiny.csv"),
        atom_feature="center_identification", as_synthon=False,
        with_hydrogen=False, kekulize=True, verbose=0,
    )
    model = models.FSpecFaithfulReactionCenter(
        fixture.node_feature_dim, [512]*3, fixture.edge_feature_dim, max_distance=5,
    )
    task = tasks.CenterIdentification(model, None, feature=("graph", "atom", "bond"),
                                      mlp_hidden_dim=512, fusion_mode="raw")
    task.preprocess(fixture, fixture, fixture)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    task.load_state_dict(checkpoint["model"], strict=True)
    task.to(device).eval()
    emit("checkpoint", {"strict_load": True, "completed_epochs": checkpoint["metadata"]["completed_epochs"],
                        "parameters": sum(p.numel() for p in task.parameters()), "device": str(device)})

    def bond_scores(graph):
        graph = graph.to(device)
        with torch.no_grad():
            out = model(graph, graph.node_feature.float())
            score = task.edge_mlp(out["pair_edge_feature"]).flatten().cpu()
        keys = [tuple(sorted(e)) for e in graph.edge_list[:, :2].tolist()]
        return dict(zip(keys, score.tolist()))

    def edge_attrs(graph, order=None):
        return {tuple(sorted((order[a], order[b]) if order else (a, b))): v
                for (a, b), v in zip(graph.edge_list[:, :2].tolist(), graph.edge_feature)}

    for smi in SMILES:
        # make_graph mutates its RDKit input by Kekulize: keep a fresh aromatic
        # copy for the end-to-end probe, and a kekulized copy for the fixed-input probe.
        source = Chem.MolFromSmiles(smi)
        fixed_mol = Chem.Mol(source)
        base = make_graph(fixed_mol)
        base_score = bond_scores(base)
        base_attrs = edge_attrs(base)
        result = {"smiles": smi}
        for mode in ("fixed_kekule", "fresh_aromatic"):
            errors, feature_changes, ranking_changes = [], [], 0
            for seed in range(5):
                order = list(range(source.GetNumAtoms()))
                random.Random(seed+101).shuffle(order)
                mol = fixed_mol if mode == "fixed_kekule" else source
                new = make_graph(Chem.RenumberAtoms(mol, order))
                score = {tuple(sorted((order[a], order[b]))): v for (a,b),v in bond_scores(new).items()}
                attrs = edge_attrs(new, order)
                errors.append(max(abs(base_score[k]-score[k]) for k in base_score))
                feature_changes.append(sum(not torch.equal(base_attrs[k],attrs[k]) for k in attrs))
                # Tied symmetry-equivalent candidates can flip due to roundoff.
                ranking_changes += max(base_score, key=base_score.get) != max(score, key=score.get)
            result[mode] = {"max_logit_error": max(errors), "max_changed_edge_features": max(feature_changes),
                            "argmax_changes_including_ties": ranking_changes, "trials": 5}
        emit("trained_atom_permutation", result)

    batch = next(iter(data.DataLoader(fixture, batch_size=2, shuffle=False)))
    batch["graph"] = tuple(g.to(device) for g in batch["graph"])
    batch["reaction"] = batch["reaction"].to(device)
    with torch.no_grad():
        logits = task.predict(batch)
        no_class = dict(batch)
        del no_class["reaction"]
        no_reactant = dict(no_class)
        no_reactant["graph"] = (None, batch["graph"][1])
        emit("input_isolation", {"delete_class_error": float((logits-task.predict(no_class)).abs().max()),
                                 "delete_reactant_error": float((logits-task.predict(no_reactant)).abs().max())})
        altered = []
        for mode in ("zero_filter", "constant_frequency"):
            def hook(module, args, output):
                if mode == "zero_filter":
                    return torch.zeros_like(output)
                return output.mean(dim=(0,1), keepdim=True).expand_as(output)
            handles = [l.frequency_encoder.register_forward_hook(hook) for l in model.layers]
            try:
                pred = task.predict(batch)
                altered.append({"mode": mode, "max_logit_change": float((logits-pred).abs().max())})
            finally:
                for h in handles:
                    h.remove()
        emit("trained_spectral_intervention_tiny_fixture_not_accuracy_ablation", altered)
        product = batch["graph"][1]
        packed = model(product, product.node_feature.float())["pair_edge_feature"]
        single = torch.cat([model(g,g.node_feature.float())["pair_edge_feature"] for g in product.unpack()])
        emit("packed_single_alignment", float((packed-single).abs().max()))

    # No optimizer step: check all three trained layers, not just layer 0.
    task.zero_grad(set_to_none=True)
    loss, _ = task(batch)
    loss.backward()
    emit("trained_frequency_gradients", [float(l.frequency_encoder.network[0].weight.grad.norm()) for l in model.layers])
    task.zero_grad(set_to_none=True)
    # Current dropout=0 must be invariant to rotations inside a repeated eigenspace.
    layer = copy.deepcopy(model.layers[0]).cpu().double().eval()
    torch.manual_seed(7)
    u, _ = torch.linalg.qr(torch.randn(6, 6, dtype=torch.double))
    e = torch.tensor([0, .5, .5, 1.5, 1.5, 2.],dtype=torch.double)
    rotation = torch.eye(6,dtype=torch.double)
    block,_ = torch.linalg.qr(torch.randn(2,2,dtype=torch.double))
    rotation[1:3,1:3] = block
    s = torch.randn(6,6,512,dtype=torch.double)
    a = torch.randn(6,6,fixture.edge_feature_dim,dtype=torch.double)
    with torch.no_grad():
        ref=layer(e,u,s,a)
        rot=layer(e,u@rotation,s,a)
    emit("repeated_eigenspace_eval_invariance", float((ref-rot).abs().max()))
    # Forward-only probe of the planned dropout option; isolate frequency dropout.
    layer.frequency_encoder.network[2].p=.05
    layer.frequency_encoder.train()
    with torch.no_grad():
        torch.manual_seed(19)
        ref=layer(e,u,s,a)
        torch.manual_seed(19)
        rot=layer(e,u@rotation,s,a)
    emit("planned_frequency_dropout_repeated_eigenspace_same_rng_error", float((ref-rot).abs().max()))
    emit("finished", "No optimizer steps, no checkpoint changes, no test split accessed")


if __name__ == "__main__":
    main()
