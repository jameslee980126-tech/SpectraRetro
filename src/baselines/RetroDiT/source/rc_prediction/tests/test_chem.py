from rdkit import Chem

from torchdrug_pred.chem import (
    clear_map_canonical_smiles,
    find_permutation,
    get_canonical_map_order,
    get_positive_center_atom_maps,
    get_reaction_center,
    sanity_check,
)


def test_reaction_center_detects_formed_bond() -> None:
    reaction = "[CH3:1].[CH3:2]>>[CH3:1][CH3:2]"
    center = get_reaction_center(reaction)
    assert center is not None
    assert center["has_changes"]
    assert center["formed_bonds"] == [[0, 1]]


def test_positive_centers_keep_multiple_and_symmetric_candidates() -> None:
    reaction = (
        "[CH3:1].[CH2:2].[CH3:3]>>"
        "[CH3:1][CH2:2][CH3:3]"
    )
    centers = get_positive_center_atom_maps(reaction)
    assert {frozenset(center) for center in centers} >= {
        frozenset((1, 2)),
        frozenset((2, 3)),
    }


def test_canonical_root_order_forms_bijection() -> None:
    product = "[CH3:1][CH2:2][OH:3]"
    canonical = get_canonical_map_order(product)
    rooted = get_canonical_map_order(product, root=1)
    assert canonical is not None
    assert rooted is not None
    permutation = find_permutation(canonical, rooted)
    assert sorted(permutation) == list(range(3))
    canonical_smiles = clear_map_canonical_smiles(product)
    assert canonical_smiles is not None
    assert Chem.MolFromSmiles(canonical_smiles) is not None


def test_sanity_check_rejects_product_only_map() -> None:
    reactant = (
        "[CH3:1][CH2:2][CH2:3][CH2:4][OH:5]"
    )
    product = "[CH3:1][CH2:2][CH2:3][CH2:4][O:6]"
    assert sanity_check(product, reactant, 100, 10) is None
