"""Product-only oriented mechanism constraints for Stage-1 centre ranking.

The module represents a candidate bond as two directed hypotheses.  A
hypothesis is admissible only when deleting the product bond and restoring H
on the proposed carbon endpoint survives full RDKit sanitization, and its
coarse edit signature occurs in the training split.  Reaction classes and
reactants are never consumed by the inference path.
"""

from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from rdkit import Chem

from .ch_counterfactual import (
    CUT_FEATURE_DIM,
    ExactCHViewStore,
    ORIENTATION_FEATURE_DIM,
    _h_cap,
    _orientation_features,
    build_exact_ch_counterfactual_view,
    validate_exact_ch_view,
)


VIEW_VERSION = "oriented_mechanism_projection_v4"
DONOR_KINDS = ("C_ar", "C_carbonyl", "C_sp", "C_sp2", "C_sp3", "C_other")
PARTNER_KINDS = DONOR_KINDS + ("N", "O", "X", "other")
BOND_KINDS = ("SINGLE", "DOUBLE", "TRIPLE", "AROMATIC")
NUM_SIGNATURES = len(DONOR_KINDS) * len(PARTNER_KINDS) * len(BOND_KINDS)


def _atom_kind(atom):
    atomic_number = atom.GetAtomicNum()
    if atomic_number == 6:
        if atom.GetIsAromatic():
            return "C_ar"
        if any(
                bond.GetBondType() == Chem.BondType.DOUBLE
                and bond.GetOtherAtom(atom).GetAtomicNum() == 8
                for bond in atom.GetBonds()):
            return "C_carbonyl"
        hybridization = atom.GetHybridization()
        if hybridization == Chem.HybridizationType.SP:
            return "C_sp"
        if hybridization == Chem.HybridizationType.SP2:
            return "C_sp2"
        if hybridization == Chem.HybridizationType.SP3:
            return "C_sp3"
        return "C_other"
    if atomic_number == 7:
        return "N"
    if atomic_number == 8:
        return "O"
    if atomic_number in (9, 17, 35, 53):
        return "X"
    return "other"


def _bond_kind(bond):
    value = str(bond.GetBondType())
    return value if value in BOND_KINDS else "AROMATIC" if bond.GetIsAromatic() else value


def encode_signature(donor_kind, partner_kind, bond_kind):
    """Encode a coarse oriented edit signature with a stable integer id."""
    if donor_kind not in DONOR_KINDS:
        raise ValueError("The donor of an oriented C-H hypothesis must be carbon")
    partner_kind = partner_kind if partner_kind in PARTNER_KINDS else "other"
    if bond_kind not in BOND_KINDS:
        raise ValueError(f"Unsupported product bond kind `{bond_kind}`")
    donor_id = DONOR_KINDS.index(donor_kind)
    partner_id = PARTNER_KINDS.index(partner_kind)
    bond_id = BOND_KINDS.index(bond_kind)
    return (donor_id * len(PARTNER_KINDS) + partner_id) * len(BOND_KINDS) + bond_id


def decode_signature(signature_id):
    signature_id = int(signature_id)
    if signature_id < 0 or signature_id >= NUM_SIGNATURES:
        raise ValueError(f"Invalid oriented signature id {signature_id}")
    bond_id = signature_id % len(BOND_KINDS)
    atom_pair_id = signature_id // len(BOND_KINDS)
    partner_id = atom_pair_id % len(PARTNER_KINDS)
    donor_id = atom_pair_id // len(PARTNER_KINDS)
    return DONOR_KINDS[donor_id], PARTNER_KINDS[partner_id], BOND_KINDS[bond_id]


def format_signature(signature):
    donor, partner, order = signature
    return f"{donor}->{partner}|{order}"


def parse_signature(text):
    atoms, order = str(text).split("|", 1)
    donor, partner = atoms.split("->", 1)
    return donor, partner, order


def load_union_template_support(path):
    """Load train-only global support from a candidate-space audit."""
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    library = payload.get("train_template_library", payload)
    records = library.get("global_templates")
    if records is None:
        # Backward-compatible reading of the first audit artifact.  New formal
        # experiments always use ``global_templates`` and never need a class
        # field to construct their admissible set.
        classes = library.get("classes")
        if not isinstance(classes, dict) or not classes:
            raise ValueError(f"No train template support found in {path}")
        records = [
            record
            for class_payload in classes.values()
            for record in class_payload.get("templates", [])
        ]
    allowed = torch.zeros(NUM_SIGNATURES, dtype=torch.bool)
    names = set()
    for record in records:
        text = record["signature"] if isinstance(record, dict) else str(record)
        signature = parse_signature(text)
        signature_id = encode_signature(*signature)
        allowed[signature_id] = True
        names.add(format_signature(decode_signature(signature_id)))
    if not allowed.any():
        raise ValueError(f"Train template support is empty in {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return allowed, tuple(sorted(names)), digest


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


def build_oriented_mechanism_view(graph, exact_view=None):
    """Build the exact product-only validity and edit-signature tensors."""
    if exact_view is None:
        exact = build_exact_ch_counterfactual_view(graph)
    else:
        validate_exact_ch_view(graph, exact_view)
        exact = exact_view
    # TorchDrug reconstructs a chemically connected RDKit object but does not
    # guarantee that hybridization / aromaticity property caches are current.
    # Template roles must be assigned only after full parent sanitization.
    molecule = Chem.Mol(graph.to_molecule())
    Chem.SanitizeMol(molecule)
    signature_id = torch.full(
        (len(exact["edge_list"]), 2), -1, dtype=torch.long
    )
    # An old positive is already backed by a fully sanitized H-capped child.
    # Only old negatives need rechecking after sanitizing the parent first;
    # this repairs Kekule/aromatic false negatives without repeating millions
    # of successful RDKit interventions.
    orientation_valid = exact["orientation_valid"].bool().clone()
    orientation_feature = exact["orientation_feature"].float().clone()
    num_node = int(graph.num_node)
    for edge_id, (source, target, _) in enumerate(exact["edge_list"].tolist()):
        bond = molecule.GetBondBetweenAtoms(int(source), int(target))
        if bond is None:
            raise RuntimeError("Candidate bond is absent from its RDKit molecule")
        order = _bond_kind(bond)
        mass_u = float(exact["cut_feature"][edge_id, 5]) * max(num_node, 1)
        mass_v = float(exact["cut_feature"][edge_id, 6]) * max(num_node, 1)
        for orientation, (donor, partner) in enumerate(
                ((source, target), (target, source))):
            donor_atom = molecule.GetAtomWithIdx(int(donor))
            valid = bool(orientation_valid[edge_id, orientation])
            if not valid and donor_atom.GetAtomicNum() == 6:
                capped = _h_cap(molecule, source, target, donor)
                valid = capped is not None
                if valid:
                    orientation_valid[edge_id, orientation] = True
                    site_mass, other_mass = (
                        (mass_u, mass_v) if orientation == 0 else (mass_v, mass_u)
                    )
                    orientation_feature[edge_id, orientation] = _orientation_features(
                        donor_atom,
                        capped,
                        site_mass,
                        other_mass,
                        num_node,
                    )
            if not valid:
                continue
            signature_id[edge_id, orientation] = encode_signature(
                _atom_kind(donor_atom),
                _atom_kind(molecule.GetAtomWithIdx(int(partner))),
                order,
            )
    candidate_graph = graph.directed()
    return {
        "version": VIEW_VERSION,
        "topology_sha256": _topology_digest(graph, candidate_graph),
        "edge_list": exact["edge_list"],
        "cut_feature": exact["cut_feature"],
        "orientation_feature": orientation_feature,
        "orientation_valid": orientation_valid,
        "orientation_signature_id": signature_id,
    }


def validate_oriented_mechanism_view(graph, view):
    candidate_graph = graph.directed()
    if view.get("version") != VIEW_VERSION:
        raise RuntimeError("Oriented mechanism cache version mismatch")
    if view.get("topology_sha256") != _topology_digest(graph, candidate_graph):
        raise RuntimeError("Oriented mechanism cache topology mismatch")
    edge_list = candidate_graph.edge_list.long().cpu()
    if not torch.equal(view["edge_list"].long().cpu(), edge_list):
        raise RuntimeError("Oriented mechanism candidate order mismatch")
    num_edge = int(candidate_graph.num_edge)
    expected = {
        "cut_feature": (num_edge, CUT_FEATURE_DIM),
        "orientation_feature": (num_edge, 2, ORIENTATION_FEATURE_DIM),
        "orientation_valid": (num_edge, 2),
        "orientation_signature_id": (num_edge, 2),
    }
    for key, shape in expected.items():
        if tuple(view[key].shape) != shape:
            raise RuntimeError(
                f"Oriented mechanism tensor `{key}` has shape {tuple(view[key].shape)}; "
                f"expected {shape}"
            )
    valid = view["orientation_valid"].bool()
    signature_id = view["orientation_signature_id"].long()
    if ((signature_id >= 0) != valid).any():
        raise RuntimeError("Every and only valid orientation must have a signature id")
    if valid.any() and int(signature_id[valid].max()) >= NUM_SIGNATURES:
        raise RuntimeError("Oriented mechanism signature id is out of range")


def _cache_key(graph):
    topology = _topology_digest(graph, graph.directed())
    return hashlib.sha256(f"{VIEW_VERSION}:{topology}".encode("ascii")).hexdigest()


class OrientedMechanismViewStore:
    """Persistent content-addressed store for exact oriented hypotheses."""

    def __init__(self, root, memory_items=2048, source_exact_cache=None):
        self.root = Path(root).resolve()
        self.memory_items = int(memory_items)
        self.memory = OrderedDict()
        self.source_exact_store = (
            ExactCHViewStore(source_exact_cache, memory_items=1)
            if source_exact_cache else None
        )

    def path(self, graph):
        key = _cache_key(graph)
        return self.root / key[-2:] / f"{key}.pt"

    def prepare(self, graph):
        path = self.path(graph)
        if path.exists():
            payload = torch.load(path, map_location="cpu", weights_only=False)
            validate_oriented_mechanism_view(graph, payload)
            return False
        exact_view = None
        if self.source_exact_store is not None:
            exact_path = self.source_exact_store.path(graph)
            if exact_path.exists():
                exact_view = self.source_exact_store.load(graph)
        payload = build_oriented_mechanism_view(graph, exact_view=exact_view)
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
            validate_oriented_mechanism_view(graph, payload)
            return payload
        path = self.path(graph)
        if not path.exists():
            raise RuntimeError(
                f"Missing oriented mechanism cache entry {path}. "
                "Build the cache explicitly before training."
            )
        payload = torch.load(path, map_location="cpu", weights_only=False)
        validate_oriented_mechanism_view(graph, payload)
        self.memory[key] = payload
        while len(self.memory) > self.memory_items:
            self.memory.popitem(last=False)
        return payload


def prepare_oriented_mechanism_datasets(store, *splits, allowed_signatures=None):
    """Build views and optionally prove that every labeled centre is admissible."""
    completed = 0
    created = 0
    gold_checked = 0
    total = sum(len(split) for split in splits)
    for split_id, split in enumerate(splits):
        for sample_id, sample in enumerate(split):
            product = sample["graph"][1]
            created += int(store.prepare(product))
            if allowed_signatures is not None:
                view = store.load(product)
                candidate = product.directed()
                positive_edge = candidate.edge_label.bool()
                positive_node = candidate.node_label.bool()
                if int(positive_edge.sum()) + int(positive_node.sum()) != 1:
                    raise RuntimeError("Expected exactly one labeled Stage-1 centre")
                if positive_node.any():
                    raise RuntimeError(
                        "Oriented mechanism projection cannot admit a node-labelled centre"
                    )
                signature = view["orientation_signature_id"].long()
                valid = view["orientation_valid"].bool()
                supported = allowed_signatures.bool()[signature.clamp_min(0)]
                admissible_edge = (valid & supported).any(dim=-1)
                if not bool(admissible_edge[positive_edge].all()):
                    edge_id = int(positive_edge.nonzero().flatten()[0])
                    raise RuntimeError(
                        "Train-derived hard mechanism support rejected a labeled edge: "
                        f"split_id={split_id}, sample_id={sample_id}, packed_position={completed}, "
                        f"edge_id={edge_id}, signatures={signature[edge_id].tolist()}, "
                        f"valid={valid[edge_id].tolist()}, "
                        f"atom_maps={product.atom_map[view['edge_list'][edge_id, :2]].tolist()}"
                    )
                gold_checked += 1
            completed += 1
            if completed == 1 or completed % 1000 == 0 or completed == total:
                print(
                    f"Oriented mechanism cache: {completed}/{total} ({created} new, "
                    f"{gold_checked} gold retained)",
                    flush=True,
                )


class OrientedMechanismProjector(nn.Module):
    """Shared oriented score head plus a non-learnable admissible-set mask."""

    def __init__(self, node_dim, candidate_dim, allowed_signatures,
                 hidden_dim=128, dropout=0.1, residual_limit=2.0):
        super().__init__()
        if tuple(allowed_signatures.shape) != (NUM_SIGNATURES,):
            raise ValueError("allowed_signatures has an invalid shape")
        self.residual_limit = float(residual_limit)
        # This repository vendors a legacy TorchDrug ``register_buffer``
        # wrapper that still references the removed ``torch._six`` API.  The
        # support mask is reconstructed from a hashed JSON contract at every
        # launch, so keeping the canonical CPU tensor as a plain non-learned
        # attribute is both deterministic and checkpoint-independent.
        self.allowed_signatures = allowed_signatures.bool().cpu().clone()
        self.node_projection = nn.Sequential(
            nn.LayerNorm(node_dim), nn.Linear(node_dim, hidden_dim), nn.GELU()
        )
        self.candidate_projection = nn.Sequential(
            nn.LayerNorm(candidate_dim + 1),
            nn.Linear(candidate_dim + 1, hidden_dim), nn.GELU(),
        )
        self.signature_embedding = nn.Embedding(NUM_SIGNATURES, hidden_dim)
        oriented_dim = hidden_dim * 4 + CUT_FEATURE_DIM + ORIENTATION_FEATURE_DIM
        self.orientation_encoder = nn.Sequential(
            nn.LayerNorm(oriented_dim),
            nn.Linear(oriented_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        )
        self.orientation_score = nn.Linear(hidden_dim, 1)
        self.residual_head = nn.Linear(hidden_dim, 1)
        # At initialization the admissible candidate scores exactly match the
        # base model; only the deterministic support projection is active.
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def forward(self, packed_node_feature, candidate_feature, base_edge_pred,
                products, views):
        node_hidden = self.node_projection(packed_node_feature)
        candidate_hidden = self.candidate_projection(torch.cat([
            candidate_feature, base_edge_pred.unsqueeze(-1)
        ], dim=-1))
        all_delta = []
        all_logits = []
        all_allowed = []
        all_signature = []
        node_offset = 0
        edge_offset = 0
        for product, view in zip(products, views):
            validate_oriented_mechanism_view(product, view)
            num_node = int(product.num_node)
            num_edge = int(product.directed().num_edge)
            local_node = node_hidden[node_offset:node_offset + num_node]
            local_candidate = candidate_hidden[edge_offset:edge_offset + num_edge]
            edge_list = view["edge_list"].to(node_hidden.device)
            source, target = edge_list[:, 0].long(), edge_list[:, 1].long()
            signature = view["orientation_signature_id"].to(node_hidden.device)
            valid = view["orientation_valid"].to(node_hidden.device)
            safe_signature = signature.clamp_min(0)
            supported = self.allowed_signatures.to(node_hidden.device)[safe_signature]
            allowed = valid & supported
            cut = view["cut_feature"].to(node_hidden.device)
            meta = view["orientation_feature"].to(node_hidden.device)

            first = self.orientation_encoder(torch.cat([
                local_node[source], local_node[target], local_candidate,
                self.signature_embedding(safe_signature[:, 0]), cut, meta[:, 0],
            ], dim=-1))
            second = self.orientation_encoder(torch.cat([
                local_node[target], local_node[source], local_candidate,
                self.signature_embedding(safe_signature[:, 1]), cut, meta[:, 1],
            ], dim=-1))
            token = torch.stack([first, second], dim=1)
            logits = self.orientation_score(token).squeeze(-1)
            masked_logits = logits.masked_fill(~allowed, -torch.inf)
            any_allowed = allowed.any(dim=-1)
            safe_logits = torch.where(
                any_allowed.unsqueeze(-1), masked_logits, torch.zeros_like(masked_logits)
            )
            probability = F.softmax(safe_logits, dim=-1)
            mixed = (probability.unsqueeze(-1) * token).sum(dim=1)
            delta = self.residual_limit * torch.tanh(
                self.residual_head(mixed).squeeze(-1)
            )
            delta = delta * any_allowed.to(delta.dtype)
            all_delta.append(delta)
            all_logits.append(logits)
            all_allowed.append(allowed)
            all_signature.append(signature)
            node_offset += num_node
            edge_offset += num_edge

        if node_offset != len(packed_node_feature) or edge_offset != len(candidate_feature):
            raise RuntimeError("Packed oriented mechanism views are misaligned")
        return {
            "edge_delta": torch.cat(all_delta),
            "orientation_logits": torch.cat(all_logits),
            "orientation_allowed": torch.cat(all_allowed),
            "orientation_signature_id": torch.cat(all_signature),
        }


__all__ = [
    "BOND_KINDS", "DONOR_KINDS", "NUM_SIGNATURES", "PARTNER_KINDS",
    "OrientedMechanismProjector", "OrientedMechanismViewStore",
    "build_oriented_mechanism_view", "decode_signature", "encode_signature",
    "format_signature", "load_union_template_support",
    "prepare_oriented_mechanism_datasets", "validate_oriented_mechanism_view",
]
