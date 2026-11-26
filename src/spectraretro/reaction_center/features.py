"""Shared product descriptors and candidate prior for reaction-center prediction."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[2]
VENDOR = ROOT / 'vendor' / 'stage1'
sys.path.insert(0, str(VENDOR))
sys.dont_write_bytecode = True
os.environ.setdefault('PHYSDUAL_ATOM_VOCAB', 'official')
if (ROOT / 'orchestration' / 'local_runtime.py').exists():
    sys.path.insert(0, str(ROOT / 'orchestration'))
    from local_runtime import ensure_runtime
    ensure_runtime()
try:
    import torchvision  # before TorchDrug's compatibility patches
except ModuleNotFoundError as exc:
    if exc.name != 'torchvision':
        raise
import numpy as np
import torch
from torch import nn
from rdkit import Chem
import torchdrug

if VENDOR not in Path(torchdrug.__file__).resolve().parents:
    raise RuntimeError('A different TorchDrug was imported; start a fresh process')

NODE_COLUMNS = (
    'formal_charge_div4', 'degree_div6', 'explicit_valence_div8',
    'implicit_valence_div8', 'total_hydrogens_div4', 'aromatic', 'ring',
    'conjugated_fraction', 'neighbour_C_fraction', 'neighbour_N_fraction',
    'neighbour_O_fraction', 'neighbour_S_fraction', 'neighbour_P_fraction',
    'neighbour_halogen_fraction', 'neighbour_other_fraction',
    'carbonyl_atom', 'carbonyl_neighbour', 'aromatic_neighbour_fraction',
    'small_ring_3_or_4', 'ring_5_or_6',
)
BOND_COLUMNS = ('order_div3', 'aromatic', 'ring', 'conjugated',
                'single', 'double', 'triple')
NODE_DIM = len(NODE_COLUMNS)
EDGE_DIM = 2 * NODE_DIM + len(BOND_COLUMNS)
SCHEMA = 'product_rdkit_chemistry_v1'


def describe_product(product):
    """Return atom/edge descriptors in graph order; only product fields are read.

    Sanitization failure is explicit (zero availability), never a fake chemical
    zero. A valid product can legitimately have charge/implicit valence zero.
    No element-specific maximum-valence claim or unknown bond-energy lookup.
    """
    n = int(product.num_node)
    e = len(product.edge_list)
    node = np.zeros((n, NODE_DIM), dtype=np.float32)
    node_mask = np.zeros_like(node)
    edge = np.zeros((e, EDGE_DIM), dtype=np.float32)
    edge_mask = np.zeros_like(edge)
    try:
        mol = product.to_molecule()
        Chem.SanitizeMol(mol)
        if mol.GetNumAtoms() != n:
            raise ValueError('RDKit atom count changed')
        if [a.GetAtomicNum() for a in mol.GetAtoms()] != product.atom_type.tolist():
            raise ValueError('RDKit atom order changed')
        for atom in mol.GetAtoms():
            neighbours = list(atom.GetNeighbors())
            bonds = list(atom.GetBonds())
            denom = max(len(neighbours), 1)
            zs = [a.GetAtomicNum() for a in neighbours]
            fractions = [sum(z == target for z in zs) / denom for target in (6, 7, 8, 16, 15)]
            fractions += [sum(z in (9, 17, 35, 53) for z in zs) / denom,
                          sum(z not in (6, 7, 8, 16, 15, 9, 17, 35, 53) for z in zs) / denom]
            carbonyl = lambda a: a.GetAtomicNum() == 6 and any(
                b.GetBondType() == Chem.BondType.DOUBLE and b.GetOtherAtom(a).GetAtomicNum() == 8
                for b in a.GetBonds())
            if hasattr(atom, 'GetValence'):
                explicit = atom.GetValence(Chem.ValenceType.EXPLICIT)
                implicit = atom.GetValence(Chem.ValenceType.IMPLICIT)
            else:
                explicit, implicit = atom.GetExplicitValence(), atom.GetImplicitValence()
            values = [atom.GetFormalCharge() / 4., atom.GetDegree() / 6.,
                      explicit / 8., implicit / 8.,
                      atom.GetTotalNumHs() / 4., float(atom.GetIsAromatic()),
                      float(atom.IsInRing()), sum(b.GetIsConjugated() for b in bonds) / max(len(bonds), 1)]
            values += fractions
            values += [float(carbonyl(atom)), float(any(carbonyl(a) for a in neighbours)),
                       sum(a.GetIsAromatic() for a in neighbours) / denom,
                       float(atom.IsInRingSize(3) or atom.IsInRingSize(4)),
                       float(atom.IsInRingSize(5) or atom.IsInRingSize(6))]
            node[atom.GetIdx()] = values
        node_mask[:] = 1.
        for k, (u, v, _) in enumerate(product.edge_list.tolist()):
            bond = mol.GetBondBetweenAtoms(u, v)
            if bond is None:
                raise ValueError('Product edge has no RDKit bond')
            # Symmetric endpoint algebra is independent of parser orientation.
            order = bond.GetBondType()
            bond_values = [bond.GetBondTypeAsDouble() / 3., float(bond.GetIsAromatic()),
                           float(bond.IsInRing()), float(bond.GetIsConjugated()),
                           float(order == Chem.BondType.SINGLE), float(order == Chem.BondType.DOUBLE),
                           float(order == Chem.BondType.TRIPLE)]
            edge[k] = np.concatenate((node[u] + node[v], np.abs(node[u] - node[v]), bond_values))
        edge_mask[:] = 1.
        reason = None
    except (ValueError, RuntimeError) as exc:
        node.fill(0.)
        edge.fill(0.)
        node_mask.fill(0.)
        edge_mask.fill(0.)
        reason = str(exc)
    if not np.isfinite(node).all() or not np.isfinite(edge).all():
        raise FloatingPointError('Nonfinite product descriptor')
    return tuple(torch.from_numpy(a) for a in (node, node_mask, edge, edge_mask)), reason


def attach_product_prior(product):
    values, reason = describe_product(product)
    with product.node():
        product.chemical_prior_node = values[0]
        product.chemical_prior_node_mask = values[1]
    with product.edge():
        product.chemical_prior_edge = values[2]
        product.chemical_prior_edge_mask = values[3]
    return reason


class CandidatePriorAdapter(nn.Module):
    """Class + FSpec candidate context controls a bounded chemical residual."""

    def __init__(self, feature_dim, descriptor_dim, class_dim, hidden=64, residual_limit=1.):
        super().__init__()
        self.residual_limit = float(residual_limit)
        self.encode = nn.Sequential(nn.Linear(2 * descriptor_dim, hidden), nn.GELU(),
                                    nn.Linear(hidden, hidden), nn.LayerNorm(hidden))
        self.context = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden), nn.GELU())
        self.gate = nn.Sequential(nn.Linear(2 * hidden + class_dim, hidden), nn.GELU(),
                                  nn.Linear(hidden, feature_dim), nn.Sigmoid())
        self.output = nn.Linear(hidden, feature_dim)
        # First forward exactly matches the paired baseline. Output gradients
        # are live at step 1; gate/encoder gradients become live after step 1.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features, descriptors, availability, classes):
        prior = self.encode(torch.cat((descriptors * availability, availability), dim=-1))
        gate = self.gate(torch.cat((self.context(features), prior, classes), dim=-1))
        available = availability.any(dim=-1, keepdim=True).to(features.dtype)
        residual = available * self.residual_limit * gate * torch.tanh(self.output(prior))
        return features + residual, gate, residual
