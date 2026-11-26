"""TorchDrug adapter of the downloaded Dual Graph Transformer (DGT).

This is a dependency-free adaptation of the computational path in the downloaded
``third_party/DGT`` implementation: NodeEdgeEncoder, DenseEdgeEncoder,
SharedEdgeEncoder, transforms.line_graph and DGTLayer. The original code targets
PyG/GraphGym property prediction; this adapter only changes the graph container
and returns the TorchDrug Stage-1 encoder contract.
"""

from collections.abc import Sequence
import math

import torch
from torch import nn

from torchdrug import core
from torchdrug.core import Registry as R


class _DGTMultiHeadAttention(nn.Module):
    """Batch-first implementation of DGT's original MultiHeadAttentionLayer."""

    def __init__(self, dim, num_heads, attn_dropout=0.0):
        super().__init__()
        if dim % num_heads:
            raise ValueError("hidden_dim must be divisible by dgt_num_heads")
        self.head_dim = dim // num_heads
        self.num_heads = num_heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(self, token, edge_attention, edge_values, attention_mask):
        batch_size, num_token, _ = token.shape
        query = self.q(token).view(batch_size, num_token, self.num_heads, self.head_dim)
        key = self.k(token).view(batch_size, num_token, self.num_heads, self.head_dim)
        # DGT keeps V flattened at [B, N, hidden_dim]; only Q/K are split
        # into heads before the scores are reshaped back to hidden_dim.
        value = self.v(token)
        key = key * (float(self.head_dim) ** -0.5)
        score = torch.einsum("bihd,bjhd->bijh", query, key).unsqueeze(-1)
        mask = attention_mask.view(batch_size, num_token, num_token, 1, 1)
        score = score - 1e24 * (~mask)
        score = score + edge_attention.view(
            batch_size, num_token, num_token, self.num_heads, self.head_dim
        )
        score = score.reshape(batch_size, num_token, num_token, -1).softmax(dim=2)
        score = score * self.attn_dropout(mask.float()).squeeze(-1)
        message = torch.einsum("bijk,bjk->bik", score, value)
        message = message + (score * edge_values).sum(dim=2)
        return message.contiguous()


class _DGTLayer(nn.Module):
    """DGTLayer applied to atom tokens and bond-line-graph tokens."""

    def __init__(self, dim, num_heads, dropout=0.0, attn_dropout=0.0, batch_norm=True):
        super().__init__()
        self.atom_attention = _DGTMultiHeadAttention(dim, num_heads, attn_dropout)
        self.bond_attention = _DGTMultiHeadAttention(dim, num_heads, attn_dropout)
        self.atom_linear = nn.Linear(dim, dim)
        self.bond_linear = nn.Linear(dim, dim)
        # The downloaded DGT configs use BatchNorm1d (layer_norm=False,
        # batch_norm=True). Keep LayerNorm available for tiny/debug runs.
        norm = nn.BatchNorm1d if batch_norm else nn.LayerNorm
        self.atom_norm1, self.atom_norm2 = norm(dim), norm(dim)
        self.bond_norm1, self.bond_norm2 = norm(dim), norm(dim)
        self.atom_ffn = nn.Sequential(nn.Linear(dim, 2 * dim), nn.ReLU(), nn.Linear(2 * dim, dim))
        self.bond_ffn = nn.Sequential(nn.Linear(dim, 2 * dim), nn.ReLU(), nn.Linear(2 * dim, dim))
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _apply_norm(norm, token, token_mask):
        """Apply a norm only to real tokens, never to padding tokens."""
        flat = token[token_mask]
        if not flat.numel():
            return token
        normalized = norm(flat)
        output = token.new_zeros(token.shape)
        output[token_mask] = normalized
        return output

    def forward(self, atom, bond, atom_att, atom_val, bond_att, bond_val, atom_mask, bond_mask):
        # This is the downloaded DGTLayer computation on a padded mini-batch.
        # In particular, BatchNorm sees all valid tokens in the mini-batch,
        # rather than the atoms of one molecule at a time.
        atom_pair_mask = atom_mask[:, :, None] & atom_mask[:, None, :]
        atom_update = self.atom_attention(atom, atom_att, atom_val, atom_pair_mask)
        atom = self._apply_norm(self.atom_norm1, atom + self.dropout(self.atom_linear(atom_update)), atom_mask)
        atom = self._apply_norm(self.atom_norm2, atom + self.dropout(self.atom_ffn(atom)), atom_mask)

        if bond.size(1) == 0:
            return atom, bond
        bond_pair_mask = bond_mask[:, :, None] & bond_mask[:, None, :]
        bond_update = self.bond_attention(bond, bond_att, bond_val, bond_pair_mask)
        bond = self._apply_norm(self.bond_norm1, bond + self.dropout(self.bond_linear(bond_update)), bond_mask)
        bond = self._apply_norm(self.bond_norm2, bond + self.dropout(self.bond_ffn(bond)), bond_mask)
        return atom, bond


@R.register("models.DGTNodeBondTransformer")
class DGTNodeBondTransformer(nn.Module, core.Configurable):
    """Downloaded-DGT-compatible atom--bond transformer for Stage-1.

    It constructs DGT's undirected bond-line graph directly from a TorchDrug
    product graph. The pre-existing CenterIdentification head and planar-dual
    branch remain untouched, preserving the original reaction-centre objective.
    """

    def __init__(self, input_dim, hidden_dims, edge_input_dim=None, num_heads=8,
                 dropout=0.0, attn_dropout=0.5, batch_norm=True, readout="sum",
                 emit_bond_feature=True, **unused):
        super().__init__()
        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        if len(set(hidden_dims)) != 1:
            raise ValueError("DGTNodeBondTransformer requires equal hidden dimensions")
        self.input_dim = int(input_dim)
        self.output_dim = int(hidden_dims[-1])
        # Stage-1 predicts a reaction *edge*. Expose the final line-graph token
        # so that the reaction-centre loss supervises the bond stream directly.
        self.emit_bond_feature = bool(emit_bond_feature)
        self.edge_feature_dim = self.output_dim if self.emit_bond_feature else 0
        self.edge_input_dim = int(edge_input_dim or 0)
        self.readout = readout
        self.atom_encoder = nn.Linear(self.input_dim, self.output_dim)
        self.raw_bond_encoder = nn.Linear(max(self.edge_input_dim, 1), self.output_dim)
        # DGT NodeEdgeEncoder.
        self.edge_mlp = nn.Sequential(nn.Linear(2 * self.output_dim, self.output_dim), nn.Mish(), nn.Linear(self.output_dim, self.output_dim))
        self.node_mlp = nn.Sequential(nn.Linear(2 * self.output_dim, self.output_dim), nn.Mish(), nn.Linear(self.output_dim, self.output_dim))
        # DGT DenseEdgeEncoder + SharedEdgeEncoder.
        self.atom_pair_type = nn.Embedding(3, self.output_dim)
        self.bond_pair_type = nn.Embedding(3, self.output_dim)
        self.atom_attention_encoder = nn.Linear(self.output_dim, self.output_dim)
        self.atom_value_encoder = nn.Linear(self.output_dim, self.output_dim)
        self.bond_attention_encoder = nn.Linear(self.output_dim, self.output_dim)
        self.bond_value_encoder = nn.Linear(self.output_dim, self.output_dim)
        self.layers = nn.ModuleList([
            _DGTLayer(self.output_dim, num_heads, dropout, attn_dropout, batch_norm=batch_norm)
            for _ in hidden_dims
        ])

    @staticmethod
    def _unique_undirected_bonds(graph, node_feature):
        edge_list = graph.edge_list
        source, target = edge_list[:, 0].long(), edge_list[:, 1].long()
        left, right = torch.minimum(source, target), torch.maximum(source, target)
        key = left * graph.num_node + right
        unique_key, inverse = torch.unique(key, sorted=True, return_inverse=True)
        bond_index = torch.stack((unique_key.div(graph.num_node, rounding_mode="floor"), unique_key.remainder(graph.num_node)), dim=-1).long()
        raw_edge = graph.edge_feature.float() if hasattr(graph, "edge_feature") and graph.edge_feature.numel() else node_feature.new_ones((edge_list.size(0), 1))
        raw_bond = raw_edge.new_zeros((unique_key.numel(), raw_edge.size(-1)))
        count = raw_edge.new_zeros((unique_key.numel(), 1))
        raw_bond.index_add_(0, inverse, raw_edge)
        count.index_add_(0, inverse, raw_edge.new_ones((raw_edge.size(0), 1)))
        return bond_index, raw_bond / count.clamp_min(1.0)

    @staticmethod
    def _incident_sum(num_atom, bond_index, feature):
        output = feature.new_zeros((num_atom, feature.size(-1)))
        if bond_index.numel():
            output.index_add_(0, bond_index[:, 0], feature)
            output.index_add_(0, bond_index[:, 1], feature)
        return output

    def _adapt_raw_bond(self, raw_bond):
        target_dim = max(self.edge_input_dim, 1)
        if raw_bond.size(-1) == target_dim:
            return raw_bond
        adapted = raw_bond.new_zeros((raw_bond.size(0), target_dim))
        adapted[:, :min(target_dim, raw_bond.size(-1))] = raw_bond[:, :target_dim]
        return adapted

    def _dense_dgt_pairs(self, atom, bond_index, raw_bond_embedding):
        """DenseEdgeEncoder/line_graph pair features, including DGT topology types."""
        num_atom, dim = atom.shape
        num_bond = bond_index.size(0)
        atom_type = torch.full((num_atom, num_atom), 2, dtype=torch.long, device=atom.device)
        atom_type.fill_diagonal_(1)
        if num_bond:
            atom_type[bond_index[:, 0], bond_index[:, 1]] = 0
            atom_type[bond_index[:, 1], bond_index[:, 0]] = 0
        atom_dense = self.atom_pair_type(atom_type)
        if num_bond:
            bond_context = raw_bond_embedding + atom[bond_index[:, 0]] + atom[bond_index[:, 1]]
            atom_dense[bond_index[:, 0], bond_index[:, 1]] += bond_context
            atom_dense[bond_index[:, 1], bond_index[:, 0]] += bond_context
        if num_bond == 0:
            return atom_dense, atom.new_zeros((0, 0, dim))

        first, second = bond_index[:, 0], bond_index[:, 1]
        ff = first[:, None] == first[None, :]
        ss = second[:, None] == second[None, :]
        fs = first[:, None] == second[None, :]
        sf = second[:, None] == first[None, :]
        connected = ff | ss | fs | sf
        bond_type = torch.full((num_bond, num_bond), 2, dtype=torch.long, device=atom.device)
        bond_type.fill_diagonal_(1)
        eye = torch.eye(num_bond, dtype=torch.bool, device=atom.device)
        bond_type[connected & ~eye] = 0
        bond_dense = self.bond_pair_type(bond_type)
        node_context = atom + self._incident_sum(num_atom, bond_index, raw_bond_embedding)
        shared_node = torch.full((num_bond, num_bond), -1, dtype=torch.long, device=atom.device)
        first_expand, second_expand = first[:, None].expand(num_bond, num_bond), second[:, None].expand(num_bond, num_bond)
        shared_node[ff] = first_expand[ff]
        shared_node[ss] = second_expand[ss]
        shared_node[fs] = first_expand[fs]
        shared_node[sf] = second_expand[sf]
        valid = shared_node >= 0
        bond_dense[valid] += node_context[shared_node[valid]]
        return atom_dense, bond_dense

    def _prepare_product(self, atom, bond_index, raw_bond):
        # DGT FeatureEncoder ordering: NodeEdgeEncoder -> dense/shared edge encoders -> DGT layers.
        raw_bond_embedding = self.raw_bond_encoder(raw_bond)
        atom = self.node_mlp(torch.cat((atom, self._incident_sum(atom.size(0), bond_index, raw_bond_embedding)), dim=-1))
        bond = self.edge_mlp(torch.cat((raw_bond_embedding, atom[bond_index[:, 0]] + atom[bond_index[:, 1]]), dim=-1)) if bond_index.numel() else atom.new_zeros((0, self.output_dim))
        atom_dense, bond_dense = self._dense_dgt_pairs(atom, bond_index, raw_bond_embedding)
        atom_attention, atom_values = self.atom_attention_encoder(atom_dense), self.atom_value_encoder(atom_dense)
        bond_attention, bond_values = self.bond_attention_encoder(bond_dense), self.bond_value_encoder(bond_dense)
        return atom, bond, atom_attention, atom_values, bond_attention, bond_values

    def forward(self, graph, input, all_loss=None, metric=None):
        del all_loss, metric
        atom_feature = self.atom_encoder(input.float())
        bond_index, raw_bond = self._unique_undirected_bonds(graph, atom_feature)
        raw_bond = self._adapt_raw_bond(raw_bond)
        # Molecule objects expose ``num_node`` while packed TorchDrug batches
        # additionally expose ``num_nodes``. node2graph is the common contract.
        if hasattr(graph, "num_nodes"):
            graph_num_nodes = [int(value) for value in graph.num_nodes.tolist()]
        else:
            graph_num_nodes = torch.bincount(graph.node2graph).tolist()
        prepared, bond_indices, graph_num_bonds = [], [], []
        node_offset = 0
        for graph_id, num_node in enumerate(graph_num_nodes):
            node_end = node_offset + num_node
            bond_mask = graph.node2graph[bond_index[:, 0]] == graph_id
            prepared.append(self._prepare_product(
                atom_feature[node_offset:node_end], bond_index[bond_mask] - node_offset, raw_bond[bond_mask]
            ))
            graph_num_bonds.append(int(bond_mask.sum()))
            if bond_mask.any():
                bond_indices.append(bond_index[bond_mask])
            node_offset = node_end
        if self.readout not in {"sum", "mean"}:
            raise ValueError("readout must be `sum` or `mean`")

        batch_size = len(prepared)
        max_node = max(item[0].size(0) for item in prepared)
        max_bond = max(graph_num_bonds, default=0)
        dim = self.output_dim
        atom_batch = atom_feature.new_zeros((batch_size, max_node, dim))
        atom_attention_batch = atom_feature.new_zeros((batch_size, max_node, max_node, dim))
        atom_value_batch = atom_feature.new_zeros((batch_size, max_node, max_node, dim))
        atom_mask = torch.zeros((batch_size, max_node), dtype=torch.bool, device=atom_feature.device)
        bond_batch = atom_feature.new_zeros((batch_size, max_bond, dim))
        bond_attention_batch = atom_feature.new_zeros((batch_size, max_bond, max_bond, dim))
        bond_value_batch = atom_feature.new_zeros((batch_size, max_bond, max_bond, dim))
        bond_mask = torch.zeros((batch_size, max_bond), dtype=torch.bool, device=atom_feature.device)
        for graph_id, item in enumerate(prepared):
            atom, bond, atom_attention, atom_values, bond_attention, bond_values = item
            num_atom, num_bond = atom.size(0), bond.size(0)
            atom_batch[graph_id, :num_atom] = atom
            atom_attention_batch[graph_id, :num_atom, :num_atom] = atom_attention
            atom_value_batch[graph_id, :num_atom, :num_atom] = atom_values
            atom_mask[graph_id, :num_atom] = True
            if num_bond:
                bond_batch[graph_id, :num_bond] = bond
                bond_attention_batch[graph_id, :num_bond, :num_bond] = bond_attention
                bond_value_batch[graph_id, :num_bond, :num_bond] = bond_values
                bond_mask[graph_id, :num_bond] = True
        for layer in self.layers:
            atom_batch, bond_batch = layer(
                atom_batch, bond_batch,
                atom_attention_batch, atom_value_batch,
                bond_attention_batch, bond_value_batch,
                atom_mask, bond_mask,
            )

        outputs = [atom_batch[graph_id, :num_node] for graph_id, num_node in enumerate(graph_num_nodes)]
        graph_outputs = [
            encoded.mean(dim=0) if self.readout == "mean" else encoded.sum(dim=0)
            for encoded in outputs
        ]
        if bond_indices:
            final_bond_feature = torch.cat([
                bond_batch[graph_id, :num_bond]
                for graph_id, num_bond in enumerate(graph_num_bonds) if num_bond
            ], dim=0)
            final_bond_index = torch.cat(bond_indices, dim=0)
        else:
            final_bond_feature = atom_feature.new_zeros((0, self.edge_feature_dim))
            final_bond_index = torch.empty((0, 2), dtype=torch.long, device=atom_feature.device)
        output = {
            "node_feature": torch.cat(outputs, dim=0),
            "graph_feature": torch.stack(graph_outputs, dim=0),
        }
        if self.emit_bond_feature:
            output["bond_feature"] = final_bond_feature
            output["bond_index"] = final_bond_index
        return output
