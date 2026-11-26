"""Exact product-only C-H counterfactual views for Stage-1 bond ranking.

This module is original project code; it does not approximate or reimplement
HAlator, QM-augmented GNN, or another external model.  Chemical interventions
are executed by RDKit itself.  Invalid interventions are marked invalid and
must receive a zero residual from :class:`ExactCHCounterfactualEncoder`.
"""

from collections import OrderedDict, deque
import hashlib
import os
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from rdkit import Chem, rdBase


VIEW_VERSION = "exact_ch_counterfactual_v2"
MAX_DISTANCE = 14
UNREACHABLE_DISTANCE = 15
CUT_FEATURE_DIM = 9
ORIENTATION_FEATURE_DIM = 20


def _blocked_distances(adjacency, start, source, target):
    distance = [-1] * len(adjacency)
    distance[start] = 0
    queue = deque([start])
    while queue:
        current = queue.popleft()
        for neighbor in adjacency[current]:
            if ((current == source and neighbor == target) or
                    (current == target and neighbor == source)):
                continue
            if distance[neighbor] < 0:
                distance[neighbor] = distance[current] + 1
                queue.append(neighbor)
    return distance


def _distance_tensor(values):
    return torch.tensor([
        UNREACHABLE_DISTANCE if value < 0 else min(value, MAX_DISTANCE)
        for value in values
    ], dtype=torch.long)


def _topology_digest(graph, candidate_graph):
    digest = hashlib.sha256()
    for name in (
            "atom_type", "formal_charge", "explicit_hs", "chiral_tag",
            "radical_electrons", "atom_map"):
        value = getattr(graph, name).long().cpu().contiguous()
        digest.update(name.encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    edge_list = candidate_graph.edge_list.long().cpu().contiguous()
    digest.update(b"canonical_edge_list")
    digest.update(str(tuple(edge_list.shape)).encode("ascii"))
    digest.update(edge_list.numpy().tobytes())
    return digest.hexdigest()


def _h_cap(molecule, source, target, endpoint):
    """Return an RDKit-sanitized explicit-H counterfactual or ``None``."""
    editable = Chem.RWMol(molecule)
    editable.RemoveBond(int(source), int(target))
    hydrogen_index = editable.AddAtom(Chem.Atom(1))
    editable.AddBond(int(endpoint), hydrogen_index, Chem.BondType.SINGLE)
    capped = editable.GetMol()
    # Invalid candidate interventions are expected for many aromatic/internal
    # product bonds.  RDKit's result / exception remains authoritative, while
    # its per-candidate diagnostic stream is suppressed to avoid log flooding.
    log_blocker = rdBase.BlockLogs()
    try:
        try:
            Chem.SanitizeMol(capped)
            canonical = Chem.MolToSmiles(capped, canonical=True)
            roundtrip = Chem.MolFromSmiles(canonical)
        except (ValueError, RuntimeError):
            return None
    finally:
        del log_blocker
    return capped if roundtrip is not None else None


def _orientation_features(atom, capped_molecule, site_mass, other_mass, num_node):
    capped_atom = (
        capped_molecule.GetAtomWithIdx(atom.GetIdx())
        if capped_molecule is not None else None
    )
    capped_h_neighbors = (
        sum(neighbor.GetAtomicNum() == 1 for neighbor in capped_atom.GetNeighbors())
        if capped_atom is not None else 0
    )
    return torch.tensor([
        float(atom.GetAtomicNum() == 6),
        float(capped_atom is not None),
        float(atom.GetFormalCharge()) / 4.0,
        float(capped_atom.GetFormalCharge()) / 4.0 if capped_atom is not None else 0.0,
        min(float(atom.GetTotalNumHs()), 4.0) / 4.0,
        min(float(capped_h_neighbors), 4.0) / 4.0,
        min(float(atom.GetDegree()), 6.0) / 6.0,
        min(float(capped_atom.GetDegree()), 6.0) / 6.0 if capped_atom is not None else 0.0,
        min(float(atom.GetTotalValence()), 8.0) / 8.0,
        min(float(capped_atom.GetTotalValence()), 8.0) / 8.0 if capped_atom is not None else 0.0,
        float(atom.GetIsAromatic()),
        float(capped_atom.GetIsAromatic()) if capped_atom is not None else 0.0,
        float(atom.IsInRing()),
        float(capped_atom.IsInRing()) if capped_atom is not None else 0.0,
        min(float(atom.GetNumRadicalElectrons()), 2.0) / 2.0,
        (min(float(capped_atom.GetNumRadicalElectrons()), 2.0) / 2.0
         if capped_atom is not None else 0.0),
        min(float(int(atom.GetHybridization())), 6.0) / 6.0,
        (min(float(int(capped_atom.GetHybridization())), 6.0) / 6.0
         if capped_atom is not None else 0.0),
        float(site_mass) / max(int(num_node), 1),
        float(other_mass) / max(int(num_node), 1),
    ], dtype=torch.float32)


def build_exact_ch_counterfactual_view(graph):
    """Build exact deleted-bond and explicit-H views for every product bond.

    Ring-bond deletion is represented by exact endpoint distances in the
    deleted graph.  It is deliberately *not* split with a soft or arbitrary
    fragment assignment.
    """
    candidate_graph = graph.directed()
    edge_list = candidate_graph.edge_list.long().cpu()
    num_node = int(graph.num_node)
    adjacency = [set() for _ in range(num_node)]
    for source, target, _ in edge_list.tolist():
        adjacency[source].add(target)
        adjacency[target].add(source)

    molecule = graph.to_molecule()
    distances_u = []
    distances_v = []
    cut_features = []
    orientation_features = []
    orientation_valid = []
    for source, target, relation in edge_list.tolist():
        distance_u_raw = _blocked_distances(adjacency, source, source, target)
        distance_v_raw = _blocked_distances(adjacency, target, source, target)
        bridge = distance_u_raw[target] < 0
        mass_u = sum(value >= 0 for value in distance_u_raw)
        mass_v = sum(value >= 0 for value in distance_v_raw)
        if not bridge:
            # Both endpoint searches cover the same connected ring-opened graph.
            mass_u = mass_v = num_node
        bond = molecule.GetBondBetweenAtoms(source, target)
        if bond is None:
            raise RuntimeError("TorchDrug candidate edge is absent from its RDKit molecule")

        capped_u = _h_cap(molecule, source, target, source)
        capped_v = _h_cap(molecule, source, target, target)
        atom_u = molecule.GetAtomWithIdx(source)
        atom_v = molecule.GetAtomWithIdx(target)
        valid_u = capped_u is not None and atom_u.GetAtomicNum() == 6
        valid_v = capped_v is not None and atom_v.GetAtomicNum() == 6
        orientation_valid.append(torch.tensor([valid_u, valid_v], dtype=torch.bool))
        orientation_features.append(torch.stack([
            _orientation_features(atom_u, capped_u, mass_u, mass_v, num_node),
            _orientation_features(atom_v, capped_v, mass_v, mass_u, num_node),
        ]))

        alternative_path = distance_u_raw[target]
        cut_features.append(torch.tensor([
            float(bridge),
            float(bond.IsInRing()),
            float(relation == 0),
            float(relation == 1),
            float(relation == 2),
            float(mass_u) / max(num_node, 1),
            float(mass_v) / max(num_node, 1),
            min(float(num_node), 100.0) / 100.0,
            1.0 if alternative_path < 0 else min(float(alternative_path), 14.0) / 14.0,
        ], dtype=torch.float32))
        distances_u.append(_distance_tensor(distance_u_raw))
        distances_v.append(_distance_tensor(distance_v_raw))

    if edge_list.numel():
        distance_u = torch.stack(distances_u)
        distance_v = torch.stack(distances_v)
        cut_feature = torch.stack(cut_features)
        orientation_feature = torch.stack(orientation_features)
        valid = torch.stack(orientation_valid)
    else:
        distance_u = torch.empty((0, num_node), dtype=torch.long)
        distance_v = torch.empty((0, num_node), dtype=torch.long)
        cut_feature = torch.empty((0, CUT_FEATURE_DIM), dtype=torch.float32)
        orientation_feature = torch.empty(
            (0, 2, ORIENTATION_FEATURE_DIM), dtype=torch.float32
        )
        valid = torch.empty((0, 2), dtype=torch.bool)
    return {
        "version": VIEW_VERSION,
        "topology_sha256": _topology_digest(graph, candidate_graph),
        "edge_list": edge_list,
        "distance_u": distance_u,
        "distance_v": distance_v,
        "cut_feature": cut_feature,
        "orientation_feature": orientation_feature,
        "orientation_valid": valid,
    }


def validate_exact_ch_view(graph, view):
    candidate_graph = graph.directed()
    if view.get("version") != VIEW_VERSION:
        raise RuntimeError("C-H counterfactual cache version mismatch")
    if view.get("topology_sha256") != _topology_digest(graph, candidate_graph):
        raise RuntimeError("C-H counterfactual cache topology mismatch")
    if not torch.equal(view["edge_list"].long().cpu(), candidate_graph.edge_list.long().cpu()):
        raise RuntimeError("C-H counterfactual candidate order mismatch")
    expected = (int(candidate_graph.num_edge), int(graph.num_node))
    if tuple(view["distance_u"].shape) != expected or tuple(view["distance_v"].shape) != expected:
        raise RuntimeError("C-H counterfactual distance shape mismatch")
    if tuple(view["cut_feature"].shape) != (expected[0], CUT_FEATURE_DIM):
        raise RuntimeError("C-H counterfactual cut-feature shape mismatch")
    if tuple(view["orientation_feature"].shape) != (
            expected[0], 2, ORIENTATION_FEATURE_DIM):
        raise RuntimeError("C-H counterfactual orientation-feature shape mismatch")
    if tuple(view["orientation_valid"].shape) != (expected[0], 2):
        raise RuntimeError("C-H counterfactual validity shape mismatch")


def _cache_key(graph):
    # Content addressing is mandatory: DGT's integer ids depend on dataset and
    # subset order, whereas this cache must be reusable without collisions.
    return _topology_digest(graph, graph.directed())


class ExactCHViewStore:
    """Persistent, topology-validated counterfactual view store."""

    def __init__(self, root, memory_items=2048):
        self.root = Path(root).resolve()
        self.memory_items = int(memory_items)
        self.memory = OrderedDict()

    def path(self, graph):
        key = _cache_key(graph)
        return self.root / key[-2:] / f"{key}.pt"

    def prepare(self, graph):
        path = self.path(graph)
        if path.exists():
            payload = torch.load(path, map_location="cpu", weights_only=False)
            validate_exact_ch_view(graph, payload)
            return False
        payload = build_exact_ch_counterfactual_view(graph)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
        torch.save(payload, temporary)
        os.replace(temporary, path)
        return True

    def load(self, graph):
        key = _cache_key(graph)
        if key in self.memory:
            payload = self.memory.pop(key)
            self.memory[key] = payload
            validate_exact_ch_view(graph, payload)
            return payload
        path = self.path(graph)
        if not path.exists():
            raise RuntimeError(
                f"Missing exact C-H counterfactual cache entry {path}. "
                "Build the cache explicitly before training."
            )
        payload = torch.load(path, map_location="cpu", weights_only=False)
        validate_exact_ch_view(graph, payload)
        self.memory[key] = payload
        while len(self.memory) > self.memory_items:
            self.memory.popitem(last=False)
        return payload


def prepare_exact_ch_datasets(store, *splits):
    completed = 0
    created = 0
    total = sum(len(split) for split in splits)
    for split in splits:
        for sample in split:
            product = sample["graph"][1]
            created += int(store.prepare(product))
            completed += 1
            if completed == 1 or completed % 1000 == 0 or completed == total:
                print(
                    f"Exact C-H counterfactual cache: {completed}/{total} "
                    f"({created} new)", flush=True
                )


def reaction_orientation_targets(reactants, products, views):
    """Training-only label for which endpoint lost H; never an input feature."""
    result = []
    for reactant, product, view in zip(reactants, products, views):
        reactant_molecule = reactant.to_molecule()
        product_molecule = product.to_molecule()
        reactant_h = {
            atom.GetAtomMapNum(): atom.GetTotalNumHs()
            for atom in reactant_molecule.GetAtoms() if atom.GetAtomMapNum() > 0
        }
        product_h = {
            atom.GetAtomMapNum(): atom.GetTotalNumHs()
            for atom in product_molecule.GetAtoms() if atom.GetAtomMapNum() > 0
        }
        local = torch.full((len(view["edge_list"]),), -1, dtype=torch.long)
        candidate = product.directed()
        positive = candidate.edge_label.nonzero().flatten()
        if len(positive) == 1:
            edge_id = int(positive[0])
            source, target = view["edge_list"][edge_id, :2].long().tolist()
            endpoint_maps = [
                int(product.atom_map[source]), int(product.atom_map[target])
            ]
            delta = [
                reactant_h.get(atom_map, product_h.get(atom_map, 0))
                - product_h.get(atom_map, 0)
                for atom_map in endpoint_maps
            ]
            if max(delta) > 0 and delta[0] != delta[1]:
                orientation = int(delta[1] > delta[0])
                if bool(view["orientation_valid"][edge_id, orientation]):
                    local[edge_id] = orientation
        result.append(local)
    return torch.cat(result)


class ExactCHCounterfactualEncoder(nn.Module):
    """Candidate residual over exact deleted-bond / H-capped product views."""

    def __init__(self, node_dim, candidate_dim, hidden_dim=128, dropout=0.1,
                 residual_limit=2.0):
        super().__init__()
        self.residual_limit = float(residual_limit)
        self.node_projection = nn.Sequential(
            nn.LayerNorm(node_dim), nn.Linear(node_dim, hidden_dim), nn.GELU()
        )
        self.candidate_projection = nn.Sequential(
            nn.LayerNorm(candidate_dim + 1),
            nn.Linear(candidate_dim + 1, hidden_dim), nn.GELU(),
        )
        self.distance_embedding = nn.Embedding(UNREACHABLE_DISTANCE + 1, hidden_dim)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )
        oriented_dim = hidden_dim * 5 + CUT_FEATURE_DIM + ORIENTATION_FEATURE_DIM
        self.orientation_encoder = nn.Sequential(
            nn.LayerNorm(oriented_dim),
            nn.Linear(oriented_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        )
        self.orientation_score = nn.Linear(hidden_dim, 1)
        self.residual_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def _rooted_pool(self, node_hidden, distance_site, distance_other):
        token = (
            node_hidden.unsqueeze(0)
            + self.distance_embedding(distance_site)
            + self.distance_embedding(distance_other)
        )
        score = self.attention(token).squeeze(-1)
        reachable = distance_site != UNREACHABLE_DISTANCE
        score = score.masked_fill(~reachable, -torch.inf)
        weight = F.softmax(score, dim=-1)
        if not torch.isfinite(weight).all():
            raise RuntimeError("Every counterfactual orientation must reach its endpoint")
        return weight @ node_hidden

    def forward(self, packed_node_feature, candidate_feature, base_edge_pred,
                products, views):
        node_hidden = self.node_projection(packed_node_feature)
        candidate_hidden = self.candidate_projection(torch.cat([
            candidate_feature, base_edge_pred.unsqueeze(-1)
        ], dim=-1))
        all_delta = []
        all_orientation_logits = []
        node_offset = 0
        edge_offset = 0
        for product, view in zip(products, views):
            validate_exact_ch_view(product, view)
            num_node = int(product.num_node)
            num_edge = int(product.directed().num_edge)
            local_node = node_hidden[node_offset:node_offset + num_node]
            local_candidate = candidate_hidden[edge_offset:edge_offset + num_edge]
            edge_list = view["edge_list"].to(node_hidden.device)
            source, target = edge_list[:, 0].long(), edge_list[:, 1].long()
            distance_u = view["distance_u"].to(node_hidden.device)
            distance_v = view["distance_v"].to(node_hidden.device)
            pool_u = self._rooted_pool(local_node, distance_u, distance_v)
            pool_v = self._rooted_pool(local_node, distance_v, distance_u)
            cut = view["cut_feature"].to(node_hidden.device)
            meta = view["orientation_feature"].to(node_hidden.device)

            first = self.orientation_encoder(torch.cat([
                local_node[source], local_node[target], pool_u, pool_v,
                local_candidate, cut, meta[:, 0],
            ], dim=-1))
            second = self.orientation_encoder(torch.cat([
                local_node[target], local_node[source], pool_v, pool_u,
                local_candidate, cut, meta[:, 1],
            ], dim=-1))
            token = torch.stack([first, second], dim=1)
            logits = self.orientation_score(token).squeeze(-1)
            valid = view["orientation_valid"].to(node_hidden.device)
            masked_logits = logits.masked_fill(~valid, -torch.inf)
            any_valid = valid.any(dim=-1)
            safe_logits = torch.where(any_valid.unsqueeze(-1), masked_logits,
                                      torch.zeros_like(masked_logits))
            probability = F.softmax(safe_logits, dim=-1)
            mixed = (probability.unsqueeze(-1) * token).sum(dim=1)
            delta = self.residual_limit * torch.tanh(
                self.residual_head(mixed).squeeze(-1)
            )
            delta = delta * any_valid.to(delta.dtype)
            all_delta.append(delta)
            all_orientation_logits.append(logits)
            node_offset += num_node
            edge_offset += num_edge

        if node_offset != len(packed_node_feature) or edge_offset != len(candidate_feature):
            raise RuntimeError("Packed C-H counterfactual views are misaligned")
        return torch.cat(all_delta), torch.cat(all_orientation_logits)
