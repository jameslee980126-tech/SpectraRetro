"""Local structural checks for the clean task-faithful FSpec encoder.

This intentionally does not train on the full data set.  It checks the three
properties required before a formal run: Stage-1 tensor contract, first-step
gradient flow through the spectral parameters, and atom-renumbering
equivariance of node / candidate-bond outputs.
"""

from pathlib import Path
import random
import sys

import torch
from rdkit import Chem


STAGE1_ROOT = Path(__file__).resolve().parent
if str(STAGE1_ROOT) not in sys.path:
    sys.path.insert(0, str(STAGE1_ROOT))

from torchdrug import data, datasets, models, tasks  # noqa: E402


SMILES = (
    "CCO",
    "c1ccccc1",
    "c1ccc2ccccc2c1",
    "C1CC2CCC1C2",
)


def make_graph(molecule):
    return data.Molecule.from_molecule(
        molecule,
        atom_feature="center_identification",
        bond_feature="default",
        with_hydrogen=False,
        kekulize=True,
    ).directed()


def build_model(graph):
    return models.FSpecFaithfulReactionCenter(
        input_dim=int(graph.atom_feature.shape[-1]),
        hidden_dims=[32, 32],
        edge_input_dim=int(graph.edge_feature.shape[-1]),
        max_distance=5,
        dropout=0.0,
    )


def output_for(model, graph):
    model.eval()
    with torch.no_grad():
        return model(graph, graph.atom_feature)


def edge_dictionary(graph, edge_feature):
    result = {}
    for edge, feature in zip(graph.directed().edge_list[:, :2].tolist(), edge_feature):
        key = tuple(sorted((int(edge[0]), int(edge[1]))))
        result.setdefault(key, []).append(feature)
    return {key: torch.stack(values).mean(dim=0) for key, values in result.items()}


def renumber_error(model, molecule, seed):
    base_graph = make_graph(molecule)
    base_output = output_for(model, base_graph)

    order = list(range(molecule.GetNumAtoms()))
    random.Random(seed).shuffle(order)
    renumbered = Chem.RenumberAtoms(molecule, order)
    new_graph = make_graph(renumbered)
    new_output = output_for(model, new_graph)

    old_to_new = {old: new for new, old in enumerate(order)}
    node_error = max(
        float((base_output["node_feature"][old] - new_output["node_feature"][old_to_new[old]]).abs().max())
        for old in range(molecule.GetNumAtoms())
    )

    base_edges = edge_dictionary(base_graph, base_output["pair_edge_feature"])
    new_edges = edge_dictionary(new_graph, new_output["pair_edge_feature"])
    mapped_new_edges = {
        tuple(sorted((old_a, old_b))): value
        for (new_a, new_b), value in new_edges.items()
        for old_a, old_b in [(order[new_a], order[new_b])]
    }
    if set(base_edges) != set(mapped_new_edges):
        raise AssertionError(
            f"candidate-edge key mismatch after atom renumbering: {set(base_edges)} vs {set(mapped_new_edges)}"
        )
    edge_error = max(
        float((base_edges[key] - mapped_new_edges[key]).abs().max())
        for key in base_edges
    )
    return node_error, edge_error


def main():
    torch.manual_seed(0)
    molecule = Chem.MolFromSmiles(SMILES[1])
    graph = make_graph(molecule)
    model = build_model(graph)

    output = model(graph, graph.atom_feature)
    expected = {
        "node_feature": (int(graph.num_node), 32),
        "pair_edge_feature": (int(graph.directed().num_edge), 32),
        "graph_feature": (1, 32),
    }
    for name, shape in expected.items():
        if tuple(output[name].shape) != shape:
            raise AssertionError(f"{name} shape {tuple(output[name].shape)} != {shape}")

    loss = output["pair_edge_feature"].square().mean()
    loss.backward()
    gradient_names = (
        "pair_seed.network.0.weight",
        "layers.0.frequency_encoder.network.0.weight",
        "layers.0.linear.weight",
        "layers.0.mlp.network.0.weight",
    )
    gradient_norms = {}
    for name, parameter in model.named_parameters():
        if name in gradient_names:
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise AssertionError(f"missing or non-finite gradient: {name}")
            gradient_norms[name] = float(parameter.grad.norm())
            if gradient_norms[name] <= 0:
                raise AssertionError(f"zero first-step gradient: {name}")

    permutation_errors = []
    for index, smiles in enumerate(SMILES):
        molecule = Chem.MolFromSmiles(smiles)
        graph = make_graph(molecule)
        model = build_model(graph)
        error = renumber_error(model, molecule, seed=index + 17)
        permutation_errors.append(error)
        if max(error) > 5e-4:
            raise AssertionError(f"permutation error too large for {smiles}: {error}")

    # Exercise the real USPTO50k parser, TorchDrug graph collator and the
    # complete CenterIdentification ranking loss on mapped reaction examples.
    fixture = STAGE1_ROOT / "tests" / "fixtures" / "fspec_tiny.csv"
    dataset = datasets.USPTO50k(
        str(fixture),
        atom_feature="center_identification",
        as_synthon=False,
        with_hydrogen=False,
        kekulize=True,
        verbose=0,
    )
    task_model = models.FSpecFaithfulReactionCenter(
        input_dim=dataset.node_feature_dim,
        hidden_dims=[32, 32],
        edge_input_dim=dataset.edge_feature_dim,
        max_distance=5,
        dropout=0.0,
    )
    task = tasks.CenterIdentification(
        task_model,
        None,
        feature=("graph", "atom", "bond"),
        mlp_hidden_dim=32,
        fusion_mode="raw",
    )
    task.preprocess(dataset, dataset, dataset)
    batch = next(iter(data.DataLoader(dataset, batch_size=2, shuffle=False)))
    task_loss, task_metric = task(batch)
    if not torch.isfinite(task_loss):
        raise AssertionError(f"non-finite CenterIdentification loss: {task_loss}")
    task_loss.backward()
    task_frequency_grad = task_model.layers[0].frequency_encoder.network[0].weight.grad
    if (
        task_frequency_grad is None
        or not torch.isfinite(task_frequency_grad).all()
        or float(task_frequency_grad.norm()) <= 0
    ):
        raise AssertionError("task-level spectral frequency gradient is missing or zero")

    print("FSpecFaithful structural verification: PASS")
    print(f"gradient_norms={gradient_norms}")
    print(f"permutation_errors={permutation_errors}")
    print(
        f"task_loss={float(task_loss.detach()):.6f}, "
        f"task_frequency_grad_norm={float(task_frequency_grad.norm()):.6g}, "
        f"task_metrics={{{', '.join(task_metric)}}}"
    )


if __name__ == "__main__":
    main()
