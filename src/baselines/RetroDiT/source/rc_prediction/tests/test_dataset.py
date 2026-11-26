import pytest


torchdrug = pytest.importorskip("torchdrug")

from torchdrug import data

from torchdrug_pred.dataset import USPTOReactionCenter


def test_torchdrug_edge_and_node_labels_are_attached() -> None:
    reactant = data.Molecule.from_smiles(
        "[CH3:1].[CH3:2]",
        atom_feature="center_identification",
        kekulize=True,
    )
    product = data.Molecule.from_smiles(
        "[CH3:1][CH3:2]",
        atom_feature="center_identification",
        kekulize=True,
    )
    dataset = object.__new__(USPTOReactionCenter)
    labeled = dataset._label_reaction(
        reactant,
        product,
        "[CH3:1].[CH3:2]>>[CH3:1][CH3:2]",
    )
    assert labeled is not None
    _, labeled_product = labeled
    assert int(labeled_product.edge_label.sum()) > 0
    assert int(labeled_product.node_label.sum()) == 0
    assert sorted(labeled_product.reaction_center.tolist()) == [1, 2]


def test_multi_positive_labels_edge_and_node_candidates_together() -> None:
    reaction = "[CH3:1].[NH3:2]>>[CH3:1][NH3+:2]"
    reactant = data.Molecule.from_smiles(
        reaction.split(">>")[0],
        atom_feature="center_identification",
        kekulize=True,
    )
    product = data.Molecule.from_smiles(
        reaction.split(">>")[1],
        atom_feature="center_identification",
        kekulize=True,
    )
    dataset = object.__new__(USPTOReactionCenter)
    dataset.objective = "multi_positive"
    labeled = dataset._label_reaction(reactant, product, reaction)
    assert labeled is not None
    _, labeled_product = labeled
    assert int(labeled_product.edge_label.sum()) == 2
    assert int(labeled_product.node_label.sum()) >= 1
