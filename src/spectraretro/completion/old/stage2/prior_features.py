"""Chemical context computed exclusively from the observed, ordered synthon graph.

There is deliberately no argument for a target linker, reaction class, target
SMILES, coordinates, or a complete precursor. Values with uncertain chemical
semantics are accompanied by observation masks, never silently imputed as facts.
"""
from collections import deque
import numpy as np
from rdkit import Chem, rdBase


FEATURE_NAMES = (
    "formal_charge_div4", "known_degree_div4", "known_bond_valence_div4",
    "neutral_valence_budget_upper_bound_div4", "single_bonds_div4",
    "double_bonds_div4", "triple_bonds_div4", "aromatic_bonds_div4",
    "aromatic_atom", "ring_atom", "conjugated_neighbor_bond",
    "neighbor_carbon_fraction", "neighbor_nitrogen_fraction",
    "neighbor_oxygen_fraction", "neighbor_sulfur_fraction",
    "neighbor_halogen_fraction", "neighbor_other_fraction",
    "carbonyl_center", "adjacent_carbonyl_center", "amide_like_nitrogen",
    "anchor", "anchor_graph_distance_clipped_div4", "within_two_of_anchor",
    "disconnected_from_anchor",
)
FEATURE_DIM = 2 * len(FEATURE_NAMES)
FEATURE_SCHEMA = "synthon_graph_chemical_context_v1"
# A conservative single-valence subset. Hypervalent, charged, aromatic and
# unsupported elements have an UNKNOWN budget. This is a budget before implicit
# hydrogens, not the true number of atoms/bonds missing from the precursor.
NEUTRAL_VALENCE = {"B": 3, "C": 4, "N": 3, "O": 2, "F": 1,
                   "Si": 4, "Cl": 1, "Br": 1, "I": 1}


def _known_graph_copy(molecule):
    """Discard inherited H/radical/stereo/property/coordinate annotations."""
    graph = Chem.RWMol()
    for source in molecule.GetAtoms():
        atom = Chem.Atom(source.GetAtomicNum())
        atom.SetFormalCharge(source.GetFormalCharge())
        graph.AddAtom(atom)
    for source in molecule.GetBonds():
        graph.AddBond(source.GetBeginAtomIdx(), source.GetEndAtomIdx(), source.GetBondType())
    return graph.GetMol()


def extract_synthon_features(molecule, anchor_indices):
    if molecule is None or molecule.GetNumAtoms() == 0:
        raise ValueError("A nonempty observed synthon graph is required")
    graph = _known_graph_copy(molecule)
    n_atoms = graph.GetNumAtoms()
    anchors = sorted(set(int(x) for x in anchor_indices))
    if not anchors or any(x < 0 or x >= n_atoms for x in anchors):
        raise ValueError("Anchor indices must identify observed synthon atoms")
    # Never sanitize the data molecule in place. On failure, reconstruct again:
    # RDKit may have partially changed the attempted copy before reporting error.
    sanitized = Chem.Mol(graph)
    # Incomplete synthons can legitimately fail kekulization; the explicit
    # observation masks carry this fact instead of thousands of stderr lines.
    with rdBase.BlockLogs():
        success = Chem.SanitizeMol(sanitized, catchErrors=True) == Chem.SanitizeFlags.SANITIZE_NONE
    if success:
        graph = sanitized
    else:
        graph = _known_graph_copy(molecule)
        graph.UpdatePropertyCache(strict=False)
    ring_known = True
    try:
        Chem.GetSymmSSSR(graph)
    except (RuntimeError, ValueError):
        ring_known = False
    adjacency = [[a.GetIdx() for a in atom.GetNeighbors()] for atom in graph.GetAtoms()]
    distance = np.full(n_atoms, -1, dtype=np.int64)
    queue = deque(anchors)
    distance[anchors] = 0
    while queue:
        source = queue.popleft()
        for target in adjacency[source]:
            if distance[target] < 0:
                distance[target] = distance[source] + 1
                queue.append(target)
    carbonyl = set()
    for atom in graph.GetAtoms():
        if atom.GetSymbol() == "C" and any(
            bond.GetBondTypeAsDouble() == 2 and
            bond.GetOtherAtom(atom).GetSymbol() in ("O", "S")
            for bond in atom.GetBonds()
        ):
            carbonyl.add(atom.GetIdx())
    values = np.zeros((n_atoms, len(FEATURE_NAMES)), dtype=np.float32)
    observed = np.ones_like(values)
    for atom in graph.GetAtoms():
        i = atom.GetIdx()
        symbol = atom.GetSymbol()
        bonds = list(atom.GetBonds())
        orders = [b.GetBondTypeAsDouble() for b in bonds]
        bond_known = all(x in (1.0, 1.5, 2.0, 3.0) for x in orders)
        valence = sum(orders) if bond_known else 0.0
        aromatic = bool(atom.GetIsAromatic()) if success else False
        budget_known = (success and bond_known and not aromatic and
                        atom.GetFormalCharge() == 0 and symbol in NEUTRAL_VALENCE and
                        valence <= NEUTRAL_VALENCE[symbol])
        budget = NEUTRAL_VALENCE[symbol] - valence if budget_known else 0.0
        neighbors = [a.GetSymbol() for a in atom.GetNeighbors()]
        denominator = max(len(neighbors), 1)
        groups = [sum(x == elem for x in neighbors) / denominator for elem in ("C", "N", "O", "S")]
        halogens = sum(x in ("F", "Cl", "Br", "I") for x in neighbors) / denominator
        other = sum(x not in ("C", "N", "O", "S", "F", "Cl", "Br", "I") for x in neighbors) / denominator
        adjacent_carbonyl = any(j in carbonyl for j in adjacency[i])
        values[i] = [
            atom.GetFormalCharge() / 4, len(neighbors) / 4, valence / 4, budget / 4,
            orders.count(1.0) / 4, orders.count(2.0) / 4, orders.count(3.0) / 4,
            orders.count(1.5) / 4, aromatic,
            atom.IsInRing() if ring_known else 0,
            any(b.GetIsConjugated() for b in bonds) if success else 0,
            *groups, halogens, other, i in carbonyl, adjacent_carbonyl,
            symbol == "N" and adjacent_carbonyl, i in anchors,
            min(distance[i], 4) / 4 if distance[i] >= 0 else 1,
            0 <= distance[i] <= 2, distance[i] < 0,
        ]
        observed[i, 2] = bond_known
        observed[i, 3] = budget_known
        observed[i, 8] = success
        observed[i, 9] = ring_known
        observed[i, 10] = success
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite observed graph features")
    return np.concatenate([values, observed], axis=-1)


def feature_contract():
    return {
        "schema": FEATURE_SCHEMA,
        "names": list(FEATURE_NAMES) + [x + "_observed" for x in FEATURE_NAMES],
        "source": "table fragments synthon graph aligned by unique atom maps to fragment SDF; anchor indices only",
        "formal_charge_is_not_nuclear_charge": True,
        "uses_reaction_class": False,
        "uses_target_linker_or_full_precursor": False,
        "uses_coordinates": False,
        "valence_budget": "conservative neutral single-valence upper bound before implicit hydrogens; undefined cases masked",
    }
