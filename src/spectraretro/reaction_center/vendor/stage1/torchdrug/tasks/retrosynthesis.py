import inspect
from collections import deque

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data as torch_data
from torch_scatter import scatter_max, scatter_add

from torchdrug import core, tasks, data, metrics, transforms
from torchdrug.data import constant
from torchdrug.layers import functional
from torchdrug.core import Registry as R
from torchdrug import layers

import rdkit
from rdkit import rdBase
rdBase.rdkitVersion
#print("retor", rdBase.rdkitVersion)

import logging
logger = logging.getLogger(__name__)

_RETRO_MTGR_ELECTRONEGATIVITY = {
    "C": 2.55, "N": 3.04, "O": 3.44, "S": 2.58, "F": 3.98,
    "Si": 1.90, "P": 2.19, "Cl": 3.16, "Br": 2.96, "Mg": 1.31,
    "Na": 0.93, "Ca": 1.00, "Fe": 1.83, "Al": 1.61, "I": 2.66,
    "B": 2.04, "K": 0.82, "Se": 2.55, "Zn": 1.65, "H": 2.20,
    "Cu": 1.90, "Mn": 1.55,
}

_RETRO_MTGR_BOND_ENERGY = {
    "B-F": 644, "B-O": 515, "Br-Br": 193, "C-B": 393, "C-Br": 276,
    "C-C": 332, "C=C": 611, "C#C": 837, "C~C": 533, "C-Cl": 328,
    "C-F": 485, "C-H": 414, "C-I": 240, "C-N": 305, "C~N": 483,
    "C=N": 615, "C#N": 891, "C-O": 326, "C=O": 728, "C~O": 534,
    "C-P": 305, "C=P": 410, "C-S": 272, "C~S": 425, "C=S": 536,
    "C-Si": 347, "Cl-Cl": 243, "N-H": 389, "N-N": 159, "N~N": 637,
    "N=N": 456, "N#N": 946, "N-O": 230, "N~O": 250, "N=O": 607,
    "N=S": 325, "Na-Br": 367, "Na-Cl": 412, "Na-F": 519, "Na-H": 186,
    "Na-I": 304, "O-H": 464, "O-O": 146, "O=O": 498, "P-Br": 272,
    "P-Cl": 331, "P-H": 322, "P-O": 410, "P=O": 632, "P=S": 430,
    "P-P": 213, "N-P": 189, "Pb-O": 382, "Pb-S": 346, "Rb-Br": 381,
    "Rb-Cl": 428, "Cs-I": 337, "F-F": 153, "F-S": 253, "H-H": 436,
    "H-Br": 366, "H-Cl": 431, "H-F": 565, "H-I": 298, "I-I": 151,
    "K-Br": 380, "K-Cl": 433, "K-F": 498, "K-I": 325, "Li-Cl": 469,
    "Li-H": 238, "Li-I": 345, "Rb-F": 494, "Rb-I": 319, "S-H": 339,
    "S-O": 364, "S=O": 343, "S-S": 368, "S=S": 290, "N-S": 312,
    "N~S": 286, "Se-H": 314, "Se~C": 130, "Se-C": 150, "Se-N": 180,
    "Se-Se": 232, "Si-Cl": 360, "Si-F": 552, "Si-H": 377, "Si-O": 460,
    "Si-Si": 176, "Si-N": 382, "Sn-C": 102, "Cl-S": 350, "P-S": 380,
    "C~P": 489,
}
_RETRO_MTGR_BOND_SYMBOL = {0: "-", 1: "=", 2: "#", 3: "~"}
_RETRO_MTGR_EDGE_FEATURE_DIM = 5
_AEE_EDGE_FEATURE_DIM = 20
_DEFAULT_RETRO_MTGR_PRIOR_GATE_CLASSES = (3, 4, 6, 7)


class _PairDualGatedFusion(nn.Module):
    """Normalize and fuse FSpec pair representations with dual-face context.

    The old FSpec path concatenated the two branches immediately before the
    prediction MLPs.  This module makes the complementarity explicit: both
    views are projected to a common scale, a sample-wise gate decides how much
    dual context to use, and candidate bonds receive a symmetric endpoint
    interaction in addition to the fused edge token.
    """

    def __init__(self, product_node_dim, product_edge_dim, dual_node_dim, fusion_dim):
        super().__init__()
        product_node_dim = int(product_node_dim)
        product_edge_dim = int(product_edge_dim)
        dual_node_dim = int(dual_node_dim)
        fusion_dim = int(fusion_dim)
        if min(product_node_dim, product_edge_dim, dual_node_dim, fusion_dim) <= 0:
            raise ValueError("Pair dual fusion dimensions must be positive")
        self.fusion_dim = fusion_dim
        self.product_node = nn.Sequential(
            nn.Linear(product_node_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
        )
        self.dual_node = nn.Sequential(
            nn.Linear(dual_node_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
        )
        self.node_gate = nn.Sequential(
            nn.Linear(3 * fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Linear(fusion_dim, 1),
        )
        self.product_edge = nn.Sequential(
            nn.Linear(product_edge_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
        )
        self.dual_edge = nn.Sequential(
            nn.Linear(2 * fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
        )
        self.edge_gate = nn.Sequential(
            nn.Linear(3 * fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Linear(fusion_dim, 1),
        )
        # A zero logit starts from an even, non-destructive blend while still
        # allowing the model to suppress a noisy dual view.
        nn.init.zeros_(self.node_gate[-1].bias)
        nn.init.zeros_(self.edge_gate[-1].bias)

    def forward(self, product_node, product_edge, dual_node, edge_index):
        if product_node.dim() != 2 or dual_node.dim() != 2:
            raise RuntimeError("Pair dual fusion expects rank-2 node features")
        if product_node.size(0) != dual_node.size(0):
            raise RuntimeError(
                "Product and dual atom views have different node counts: "
                f"{product_node.size(0)} vs {dual_node.size(0)}"
            )
        if edge_index.dim() != 2 or edge_index.size(-1) != 2:
            raise RuntimeError("Pair dual fusion expects an [E, 2] candidate index")

        product_node = self.product_node(product_node)
        dual_node = self.dual_node(dual_node)
        node_context = torch.cat(
            (product_node, dual_node, torch.abs(product_node - dual_node)), dim=-1
        )
        node_gate = torch.sigmoid(self.node_gate(node_context))
        fused_node = product_node + node_gate * dual_node

        product_edge = self.product_edge(product_edge)
        endpoint_left = dual_node[edge_index[:, 0].long()]
        endpoint_right = dual_node[edge_index[:, 1].long()]
        dual_edge = self.dual_edge(torch.cat(
            (endpoint_left + endpoint_right,
             torch.abs(endpoint_left - endpoint_right)), dim=-1
        ))
        edge_context = torch.cat(
            (product_edge, dual_edge, torch.abs(product_edge - dual_edge)), dim=-1
        )
        edge_gate = torch.sigmoid(self.edge_gate(edge_context))
        fused_edge = product_edge + edge_gate * dual_edge
        # Multiplicative interaction is deliberately low-dimensional: it
        # couples the two views without constructing an O(N^2 D) second pair
        # tensor or changing the FSpec spectral encoder itself.
        fused_edge = torch.cat((fused_edge, product_edge * dual_edge), dim=-1)
        return fused_node, fused_edge, node_gate, edge_gate


from rdkit import Chem

from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import AllChem


@R.register("tasks.CenterIdentification1")
class CenterIdentification(tasks.Task, core.Configurable):
    """
    Reaction center identification task.
    This class is a part of retrosynthesis prediction.
    Parameters:
        model (nn.Module): graph representation model
        feature (str or list of str, optional): additional features for prediction. Available features are
            reaction: type of the reaction
            graph: graph representation of the product
            atom: original atom feature
            bond: original bond feature
        num_mlp_layer (int, optional): number of MLP layers
    """

    _option_members = set(["feature"])

    def __init__(self, model, model2, feature=("reaction", "graph", "atom", "bond"), num_mlp_layer=2, mlp_act='relu',
                 metric=("top-1", "top-2", "top-3", "top-5", "top-10"), retro_mtgr_edge_features=False,
                 retro_mtgr_prior_gate="none", retro_mtgr_manual_gate_classes=None, aee_edge_features=False,
                 aee_prior_gate="none", aee_manual_gate_classes=None, mlp_hidden_dim=None,
                 fusion_mode="raw", fusion_dim=256, ch_counterfactual=False,
                 ch_counterfactual_cache=None, ch_counterfactual_hidden_dim=128,
                 ch_counterfactual_dropout=0.1, ch_counterfactual_residual_limit=2.0,
                 ch_orientation_weight=0.2, ch_safety_weight=1.0,
                 ch_residual_weight=1e-3, oriented_mechanism=False,
                 oriented_mechanism_cache=None, oriented_template_library=None,
                 oriented_source_exact_cache=None,
                 oriented_mechanism_hidden_dim=128, oriented_mechanism_dropout=0.1,
                 oriented_mechanism_residual_limit=2.0,
                 oriented_orientation_weight=0.2,
                 oriented_competition_weight=0.5,
                 oriented_competition_margin=1.0,
                 oriented_residual_weight=1e-3):
        super(CenterIdentification, self).__init__()
        self.model = model
        self.model2=model2
        self.num_mlp_layer = num_mlp_layer
        self.feature = feature
        self.mlp_act = mlp_act
        self.mlp_hidden_dim = int(mlp_hidden_dim) if mlp_hidden_dim is not None else self.model.output_dim
        if fusion_mode not in {"raw", "balanced", "gated"}:
            raise ValueError("fusion_mode must be `raw`, `balanced`, or `gated`")
        if int(fusion_dim) <= 0:
            raise ValueError("fusion_dim must be positive")
        self.fusion_mode = fusion_mode
        self.fusion_dim = int(fusion_dim)
        self.ch_counterfactual = bool(ch_counterfactual)
        self.ch_counterfactual_cache = ch_counterfactual_cache
        self.ch_counterfactual_hidden_dim = int(ch_counterfactual_hidden_dim)
        self.ch_counterfactual_dropout = float(ch_counterfactual_dropout)
        self.ch_counterfactual_residual_limit = float(ch_counterfactual_residual_limit)
        self.ch_orientation_weight = float(ch_orientation_weight)
        self.ch_safety_weight = float(ch_safety_weight)
        self.ch_residual_weight = float(ch_residual_weight)
        self.oriented_mechanism = bool(oriented_mechanism)
        self.oriented_mechanism_cache = oriented_mechanism_cache
        self.oriented_template_library = oriented_template_library
        self.oriented_source_exact_cache = oriented_source_exact_cache
        self.oriented_mechanism_hidden_dim = int(oriented_mechanism_hidden_dim)
        self.oriented_mechanism_dropout = float(oriented_mechanism_dropout)
        self.oriented_mechanism_residual_limit = float(oriented_mechanism_residual_limit)
        self.oriented_orientation_weight = float(oriented_orientation_weight)
        self.oriented_competition_weight = float(oriented_competition_weight)
        self.oriented_competition_margin = float(oriented_competition_margin)
        self.oriented_residual_weight = float(oriented_residual_weight)
        if self.ch_counterfactual and self.oriented_mechanism:
            raise ValueError("The old soft C-H branch and hard oriented mechanism mode are mutually exclusive")
        if self.ch_counterfactual and not self.ch_counterfactual_cache:
            raise ValueError("Exact C-H counterfactual mode requires a persistent cache path")
        if self.oriented_mechanism and (
                not self.oriented_mechanism_cache or not self.oriented_template_library):
            raise ValueError(
                "Oriented mechanism mode requires a cache path and train template library"
            )
        if self.ch_counterfactual_hidden_dim <= 0:
            raise ValueError("C-H counterfactual hidden dimension must be positive")
        if min(self.ch_orientation_weight, self.ch_safety_weight,
               self.ch_residual_weight) < 0:
            raise ValueError("C-H auxiliary loss weights must be non-negative")
        if self.oriented_mechanism_hidden_dim <= 0:
            raise ValueError("Oriented mechanism hidden dimension must be positive")
        if min(
                self.oriented_orientation_weight,
                self.oriented_competition_weight,
                self.oriented_competition_margin,
                self.oriented_residual_weight) < 0:
            raise ValueError("Oriented mechanism loss weights and margin must be non-negative")
        self.ch_counterfactual_encoder = None
        self.ch_view_store = None
        self._last_ch_auxiliary = None
        self.oriented_mechanism_projector = None
        self.oriented_mechanism_view_store = None
        self._last_oriented_auxiliary = None
        self.oriented_template_names = tuple()
        self.oriented_template_library_sha256 = None
        self.freeze_base_for_counterfactual = False
        # Balanced-fusion modules are created in preprocess(), once every
        # dataset-dependent input dimension is known. Raw mode leaves these as
        # None and retains the historical state dict and computation path.
        self.atom_view_projection = None
        self.graph_view_projection = None
        self.bond_view_projection = None
        self.face_view_projection = None
        # FSpec's pair-domain path uses a separate fusion module.  It is
        # created in preprocess(), after the encoder dimensions are known.
        self.pair_dual_fusion = None
        self._last_pair_fusion_gates = None
        self.metric = metric
        self.retro_mtgr_edge_features = retro_mtgr_edge_features
        if retro_mtgr_prior_gate not in {"none", "manual", "learned"}:
            raise ValueError("retro_mtgr_prior_gate must be one of `none`, `manual`, or `learned`")
        self.retro_mtgr_prior_gate = retro_mtgr_prior_gate
        self.retro_mtgr_manual_gate_classes = self._parse_manual_gate_classes(retro_mtgr_manual_gate_classes)
        self.aee_edge_features = aee_edge_features
        if aee_prior_gate not in {"none", "manual", "learned"}:
            raise ValueError("aee_prior_gate must be one of `none`, `manual`, or `learned`")
        self.aee_prior_gate = aee_prior_gate
        self.aee_manual_gate_classes = self._parse_manual_gate_classes(aee_manual_gate_classes)
        if "reaction" not in self.feature and (
            self.retro_mtgr_prior_gate != "none" or self.aee_prior_gate != "none"
        ):
            raise ValueError(
                "Unknown-class CenterIdentification forbids gates driven by reaction labels"
            )
        if self.retro_mtgr_prior_gate == "learned":
            self.retro_mtgr_class_gate_logit = nn.Parameter(torch.zeros(32, _RETRO_MTGR_EDGE_FEATURE_DIM))
        else:
            self.register_parameter("retro_mtgr_class_gate_logit", None)
        if self.aee_prior_gate == "learned":
            self.aee_class_gate_logit = nn.Parameter(torch.zeros(32, _AEE_EDGE_FEATURE_DIM))
        else:
            self.register_parameter("aee_class_gate_logit", None)

    @staticmethod
    def _parse_manual_gate_classes(classes):
        if classes is None:
            return _DEFAULT_RETRO_MTGR_PRIOR_GATE_CLASSES
        if isinstance(classes, str):
            if not classes.strip():
                return tuple()
            return tuple(int(value.strip()) for value in classes.split(",") if value.strip())
        return tuple(int(value) for value in classes)

    def train(self, mode=True):
        super().train(mode)
        if mode and self.freeze_base_for_counterfactual:
            # Parameter freezing alone does not disable Dropout or BatchNorm
            # running-stat updates.  Keep the complete inherited Stage-1 base
            # deterministic while the new residual branch learns.
            for module in (
                    self.model, self.model2, self.atom_view_projection,
                    self.graph_view_projection, self.bond_view_projection,
                    self.face_view_projection, getattr(self, "edge_mlp", None),
                    getattr(self, "node_mlp", None)):
                if module is not None:
                    module.eval()
            auxiliary = (
                self.ch_counterfactual_encoder
                if self.ch_counterfactual_encoder is not None
                else self.oriented_mechanism_projector
            )
            if auxiliary is None:
                raise RuntimeError("Frozen-base mode requires an auxiliary mechanism branch")
            auxiliary.train(True)
        return self

    def preprocess(self, train_set, valid_set, test_set):
        reaction_types = set()
        bond_types = set()
        expected_num_relation = int(train_set[0]["graph"][1].num_relation)
        for split_name, split in (("train", train_set), ("valid", valid_set), ("test", test_set)):
            for sample_index, sample in enumerate(split):
                if "reaction" in self.feature:
                    reaction_types.add(sample["reaction"])
                for graph in sample["graph"]:
                    if int(graph.num_relation) != expected_num_relation:
                        raise RuntimeError(
                            f"Inconsistent relation schema in {split_name}[{sample_index}]: "
                            f"expected {expected_num_relation}, got {int(graph.num_relation)}"
                        )
                    bond_types.update(graph.edge_list[:, 2].tolist())
                product = sample["graph"][1]
                expected_node_dim = getattr(self.model, 'input_dim', None)
                expected_edge_dim = getattr(self.model, 'edge_input_dim', None)
                if getattr(self.model, 'is_pair_center_model', False):
                    for field, width, rows in (
                        ('node_feature', expected_node_dim, int(product.num_node)),
                        ('edge_feature', expected_edge_dim, len(product.edge_list)),
                    ):
                        if width is None or (field == 'edge_feature' and width == 0):
                            continue
                        value = getattr(product, field, None)
                        if value is None or tuple(value.shape) != (rows, width):
                            actual = None if value is None else tuple(value.shape)
                            raise ValueError(
                                f'Stage1 {split_name}[{sample_index}] {field} shape mismatch: '
                                f'expected {(rows, width)}, got {actual}; rebuild feature cache')
                # TorchDrug stores molecular bonds bidirectionally; Stage-1
                # ranks the deterministic u < v candidate view.
                self._assert_canonical_product_edges(product.directed())
                num_positive = int(product.edge_label.sum()) + int(product.node_label.sum())
                if num_positive != 1:
                    raise RuntimeError(
                        f"Stage-1 {split_name}[{sample_index}] has {num_positive} positive centres; expected 1"
                    )
        # In unknown-class mode the class labels are not even inspected here.
        # They may remain in the CSV solely for stratified post-hoc reporting.
        if "reaction" in self.feature:
            if not reaction_types or min(reaction_types) < 0:
                raise RuntimeError("Known-class reaction ids must be non-negative integers")
            # Subsets may omit an intermediate class; len(set(...)) would make
            # the one-hot width too small for the original dense class id.
            self.num_reaction = int(max(reaction_types)) + 1
        else:
            self.num_reaction = 0
        if bond_types and (min(bond_types) < 0 or max(bond_types) >= expected_num_relation):
            raise RuntimeError("Observed bond relation id lies outside the graph schema")
        self.num_relation = expected_num_relation
        node_feature_dim = train_set[0]["graph"][0].node_feature.shape[-1]
        edge_feature_dim = train_set[0]["graph"][0].edge_feature.shape[-1]
        self.bond_feature_dim = edge_feature_dim

        # FSpecReactionCenter already performs the complete pair-domain lift and
        # returns candidate-edge and candidate-node representations.  The raw
        # mode below is retained as the exact historical direct-concat control;
        # gated mode is the new, explicitly normalized interaction path.
        if getattr(self.model, "is_pair_center_model", False):
            if self.retro_mtgr_edge_features or self.aee_edge_features:
                raise ValueError("Pair-domain reaction-center encoders cannot use engineered edge features")
            if self.ch_counterfactual or self.oriented_mechanism:
                raise ValueError("Pair-domain reaction-center encoders cannot use auxiliary mechanism branches")
            dual_dim = int(self.model2.output_dim) if self.model2 is not None else 0
            if self.fusion_mode == "gated" and dual_dim <= 0:
                raise ValueError("FSpec gated fusion requires --planar-dual-branch rgcn")
            if self.fusion_mode == "gated":
                self.pair_dual_fusion = _PairDualGatedFusion(
                    product_node_dim=int(self.model.output_dim),
                    product_edge_dim=int(self.model.edge_feature_dim),
                    dual_node_dim=dual_dim,
                    fusion_dim=self.fusion_dim,
                )
                node_input_dim = self.fusion_dim
                edge_input_dim = 2 * self.fusion_dim
            else:
                node_input_dim = int(self.model.output_dim) + dual_dim
                edge_input_dim = int(self.model.edge_feature_dim) + 2 * dual_dim
            hidden_dims = [self.mlp_hidden_dim] * (self.num_mlp_layer - 1)
            self.edge_mlp = layers.MLP(
                edge_input_dim,
                hidden_dims + [1], activation=self.mlp_act
            )
            self.node_mlp = layers.MLP(
                node_input_dim,
                hidden_dims + [1], activation=self.mlp_act
            )
            return

        model_graph_dim = int(getattr(
            self.model, "graph_feature_dim", self.model.output_dim
        ))
        model_bond_dim = int(getattr(self.model, "edge_feature_dim", 0))
        if self.fusion_mode == "balanced":
            self.atom_view_projection = nn.Sequential(
                nn.Linear(int(self.model.output_dim), self.fusion_dim),
                nn.LayerNorm(self.fusion_dim),
            )
            if "graph" in self.feature:
                self.graph_view_projection = nn.Sequential(
                    nn.Linear(model_graph_dim, self.fusion_dim),
                    nn.LayerNorm(self.fusion_dim),
                )
            if model_bond_dim:
                self.bond_view_projection = nn.Sequential(
                    nn.Linear(model_bond_dim, self.fusion_dim),
                    nn.LayerNorm(self.fusion_dim),
                )
            if self.model2 is not None:
                self.face_view_projection = nn.Sequential(
                    nn.Linear(int(self.model2.output_dim), self.fusion_dim),
                    nn.LayerNorm(self.fusion_dim),
                )

        node_dim = self.fusion_dim if self.fusion_mode == "balanced" else self.model.output_dim
        edge_dim = 0
        graph_dim = 0
        for _feature in sorted(self.feature):
            if _feature == "reaction":
                graph_dim += self.num_reaction
            elif _feature == "graph":
                graph_dim += self.fusion_dim if self.fusion_mode == "balanced" else model_graph_dim
            elif _feature == "atom":
                node_dim += node_feature_dim
            elif _feature == "bond":
                edge_dim += self.bond_feature_dim
            else:
                raise ValueError("Unknown feature `%s`" % _feature)
        if self.retro_mtgr_edge_features:
            edge_dim += _RETRO_MTGR_EDGE_FEATURE_DIM
            self._init_retro_mtgr_tables()
        if self.aee_edge_features:
            edge_dim += _AEE_EDGE_FEATURE_DIM
        # DGT emits one learned token per undirected product bond. It is mapped
        # to the two directed candidate edges in ``predict`` below.
        edge_dim += (
            self.fusion_dim
            if self.fusion_mode == "balanced" and model_bond_dim
            else model_bond_dim
        )
        node_dim += graph_dim  # inherit graph features
        

        if self.model2 is not None:
            node_dim += (
                self.fusion_dim
                if self.fusion_mode == "balanced"
                else self.model2.output_dim
            )
        edge_dim += node_dim * 2 # inherit node features
        if self.ch_counterfactual:
            self.ch_counterfactual_encoder = layers.ExactCHCounterfactualEncoder(
                node_dim=node_dim,
                candidate_dim=edge_dim,
                hidden_dim=self.ch_counterfactual_hidden_dim,
                dropout=self.ch_counterfactual_dropout,
                residual_limit=self.ch_counterfactual_residual_limit,
            )
            self.ch_view_store = layers.ExactCHViewStore(
                self.ch_counterfactual_cache
            )
        if self.oriented_mechanism:
            allowed, names, digest = layers.load_union_template_support(
                self.oriented_template_library
            )
            self.oriented_mechanism_projector = layers.OrientedMechanismProjector(
                node_dim=node_dim,
                candidate_dim=edge_dim,
                allowed_signatures=allowed,
                hidden_dim=self.oriented_mechanism_hidden_dim,
                dropout=self.oriented_mechanism_dropout,
                residual_limit=self.oriented_mechanism_residual_limit,
            )
            self.oriented_mechanism_view_store = layers.OrientedMechanismViewStore(
                self.oriented_mechanism_cache,
                source_exact_cache=self.oriented_source_exact_cache,
            )
            self.oriented_template_names = names
            self.oriented_template_library_sha256 = digest
        hidden_dims = [self.mlp_hidden_dim] * (self.num_mlp_layer - 1)
        self.edge_mlp = layers.MLP(edge_dim, hidden_dims + [1], activation=self.mlp_act)
        self.node_mlp = layers.MLP(node_dim, hidden_dims + [1], activation=self.mlp_act)

    def _init_retro_mtgr_tables(self):
        num_atom = len(constant.ATOM_SYMBOL)
        electronegativity = torch.zeros(num_atom, dtype=torch.float32)
        for atom_id, symbol in enumerate(constant.ATOM_SYMBOL):
            electronegativity[atom_id] = _RETRO_MTGR_ELECTRONEGATIVITY.get(symbol, 0.0)

        bond_energy = torch.zeros(num_atom, num_atom, len(_RETRO_MTGR_BOND_SYMBOL), dtype=torch.float32)
        symbol_to_atom_id = {symbol: atom_id for atom_id, symbol in enumerate(constant.ATOM_SYMBOL)}
        for bond_key, energy in _RETRO_MTGR_BOND_ENERGY.items():
            for bond_id, symbol in _RETRO_MTGR_BOND_SYMBOL.items():
                if symbol not in bond_key:
                    continue
                left, right = bond_key.split(symbol)
                if left not in symbol_to_atom_id or right not in symbol_to_atom_id:
                    continue
                left_id = symbol_to_atom_id[left]
                right_id = symbol_to_atom_id[right]
                normalized_energy = float(energy) / 1000.0
                bond_energy[left_id, right_id, bond_id] = normalized_energy
                bond_energy[right_id, left_id, bond_id] = normalized_energy

        self._retro_mtgr_electronegativity = electronegativity
        self._retro_mtgr_bond_energy = bond_energy

    def _class_gate_for_edges(self, graph, source, reaction, feature_dim, gate_mode, manual_classes, gate_logit):
        if reaction is None or gate_mode == "none":
            return None

        edge2graph = graph.node2graph[source]
        class_id = reaction[edge2graph].long()
        if gate_mode == "manual":
            gate = torch.zeros(len(class_id), dtype=torch.float32, device=class_id.device)
            for one_based_class in manual_classes:
                gate = torch.maximum(gate, (class_id == int(one_based_class) - 1).float())
            return gate.unsqueeze(-1).expand(-1, feature_dim)

        gate_logit = gate_logit.to(class_id.device)
        class_id = class_id.clamp(min=0, max=gate_logit.size(0) - 1)
        return torch.sigmoid(gate_logit[class_id])

    def _retro_mtgr_edge_feature(self, graph, reaction=None):
        edge_list = graph.edge_list
        if not edge_list.numel():
            return edge_list.new_zeros((0, _RETRO_MTGR_EDGE_FEATURE_DIM), dtype=torch.float32)

        source = edge_list[:, 0].long()
        target = edge_list[:, 1].long()
        bond_type = edge_list[:, 2].long().clamp(min=0, max=len(_RETRO_MTGR_BOND_SYMBOL) - 1)
        electronegativity = self._retro_mtgr_electronegativity.to(edge_list.device)
        bond_energy_table = self._retro_mtgr_bond_energy.to(edge_list.device)
        atom_type = graph.atom_type.long().clamp(min=0, max=electronegativity.numel() - 1)
        source_atom = atom_type[source]
        target_atom = atom_type[target]

        bond_energy = bond_energy_table[source_atom, target_atom, bond_type]
        high_energy = (bond_energy >= 0.45).to(bond_energy.dtype)
        en_diff = (
            electronegativity[source_atom] -
            electronegativity[target_atom]
        ).abs() / 4.0

        max_neighbor_energy = scatter_max(bond_energy, source, dim_size=graph.num_node)[0]
        max_neighbor_energy = torch.where(
            torch.isfinite(max_neighbor_energy),
            max_neighbor_energy,
            torch.zeros_like(max_neighbor_energy),
        )
        energy_margin = (
            (max_neighbor_energy[source] - bond_energy).clamp(min=0) +
            (max_neighbor_energy[target] - bond_energy).clamp(min=0)
        ) / 2.0

        degree = scatter_add(torch.ones_like(bond_energy), source, dim_size=graph.num_node)
        degree_delta = (degree[source] - degree[target]).abs() / degree.max().clamp(min=1.0)

        feature = torch.stack([bond_energy, high_energy, en_diff, energy_margin, degree_delta], dim=-1)
        gate = self._class_gate_for_edges(
            graph,
            source,
            reaction,
            _RETRO_MTGR_EDGE_FEATURE_DIM,
            self.retro_mtgr_prior_gate,
            self.retro_mtgr_manual_gate_classes,
            self.retro_mtgr_class_gate_logit,
        )
        if gate is not None:
            feature = feature * gate
        return feature

    def _aee_edge_feature(self, graph, reaction=None):
        edge_list = graph.edge_list
        if not edge_list.numel():
            return edge_list.new_zeros((0, _AEE_EDGE_FEATURE_DIM), dtype=torch.float32)

        source = edge_list[:, 0].long()
        target = edge_list[:, 1].long()
        bond_type = edge_list[:, 2].long().clamp(min=0, max=3)
        atom_type = graph.atom_type.long()

        source_atom = atom_type[source]
        target_atom = atom_type[target]
        source_is_c = (source_atom == 6).float()
        target_is_c = (target_atom == 6).float()
        source_is_n = (source_atom == 7).float()
        target_is_n = (target_atom == 7).float()
        source_is_o = (source_atom == 8).float()
        target_is_o = (target_atom == 8).float()
        source_halogen = ((source_atom == 9) | (source_atom == 17) | (source_atom == 35) | (source_atom == 53)).float()
        target_halogen = ((target_atom == 9) | (target_atom == 17) | (target_atom == 35) | (target_atom == 53)).float()
        source_hetero = ((source_atom == 7) | (source_atom == 8) | (source_atom == 15) | (source_atom == 16) |
                         (source_atom == 34)).float()
        target_hetero = ((target_atom == 7) | (target_atom == 8) | (target_atom == 15) | (target_atom == 16) |
                         (target_atom == 34)).float()

        if getattr(graph, "node_feature", None) is not None and graph.node_feature.size(-1) >= 2:
            aromatic = graph.node_feature[:, -2].float()
            ring = graph.node_feature[:, -1].float()
        else:
            aromatic = torch.zeros(graph.num_node, dtype=torch.float32, device=edge_list.device)
            ring = torch.zeros_like(aromatic)

        ones = torch.ones(len(edge_list), dtype=torch.float32, device=edge_list.device)
        degree = scatter_add(ones, source, dim_size=graph.num_node).clamp(min=1.0)
        atom_is_hetero = ((atom_type == 7) | (atom_type == 8) | (atom_type == 15) | (atom_type == 16) |
                          (atom_type == 34)).float()
        atom_is_c = (atom_type == 6).float()
        neighbor_hetero = scatter_add(atom_is_hetero[target], source, dim_size=graph.num_node) / degree
        neighbor_carbon = scatter_add(atom_is_c[target], source, dim_size=graph.num_node) / degree
        double_to_oxygen = ((bond_type == 1) & (target_atom == 8)).float()
        double_neighbor = scatter_add((bond_type == 1).float(), source, dim_size=graph.num_node)
        carbonyl_like = ((atom_type == 6) & (scatter_add(double_to_oxygen, source, dim_size=graph.num_node) > 0)).float()

        bond_onehot = F.one_hot(bond_type, num_classes=4).float()
        endpoint_c = (source_is_c + target_is_c) / 2.0
        endpoint_n = (source_is_n + target_is_n) / 2.0
        endpoint_o = (source_is_o + target_is_o) / 2.0
        endpoint_halogen = (source_halogen + target_halogen) / 2.0
        endpoint_hetero = (source_hetero + target_hetero) / 2.0
        endpoint_aromatic = (aromatic[source] + aromatic[target]) / 2.0
        endpoint_ring = (ring[source] + ring[target]) / 2.0
        endpoint_carbonyl = (carbonyl_like[source] + carbonyl_like[target]) / 2.0
        source_degree = (degree[source] / 6.0).clamp(max=1.0)
        target_degree = (degree[target] / 6.0).clamp(max=1.0)
        degree_delta = ((degree[source] - degree[target]).abs() / 6.0).clamp(max=1.0)
        endpoint_unsaturation = ((double_neighbor[source] + double_neighbor[target]) / 2.0).clamp(max=1.0)

        feature = torch.cat([
            bond_onehot,
            endpoint_c.unsqueeze(-1),
            endpoint_n.unsqueeze(-1),
            endpoint_o.unsqueeze(-1),
            endpoint_halogen.unsqueeze(-1),
            endpoint_hetero.unsqueeze(-1),
            endpoint_aromatic.unsqueeze(-1),
            endpoint_ring.unsqueeze(-1),
            endpoint_carbonyl.unsqueeze(-1),
            source_degree.unsqueeze(-1),
            target_degree.unsqueeze(-1),
            degree_delta.unsqueeze(-1),
            neighbor_hetero[source].unsqueeze(-1),
            neighbor_hetero[target].unsqueeze(-1),
            neighbor_carbon[source].unsqueeze(-1),
            neighbor_carbon[target].unsqueeze(-1),
            endpoint_unsaturation.unsqueeze(-1),
        ], dim=-1)
        gate = self._class_gate_for_edges(
            graph,
            source,
            reaction,
            _AEE_EDGE_FEATURE_DIM,
            self.aee_prior_gate,
            self.aee_manual_gate_classes,
            self.aee_class_gate_logit,
        )
        if gate is not None:
            feature = feature * gate
        return feature


    def forward(self, batch):
        """"""
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        pred = self.predict(batch, all_loss, metric)
        target = self.target(batch)
        if self.oriented_mechanism:
            allowed_candidate = self._last_oriented_auxiliary["allowed_candidate"]
            positive = target[0].bool()
            if not bool(allowed_candidate[positive].all()):
                rejected = (~allowed_candidate & positive).nonzero().flatten().tolist()
                raise RuntimeError(
                    "Hard oriented mechanism projection rejected a gold centre at packed "
                    f"candidate indices {rejected[:8]}"
                )
        metric.update(self.evaluate(pred, target))

        target, size = target
        target = functional.variadic_max(target, size)[1]
        loss = functional.variadic_cross_entropy(pred, target, size)

        name = tasks._get_criterion_name("ce")
        metric[name] = loss

        all_loss += loss

        if self.ch_counterfactual and self.ch_orientation_weight > 0:
            auxiliary = self._last_ch_auxiliary
            if auxiliary is None:
                raise RuntimeError("Missing C-H counterfactual auxiliary output")
            reactant, product = batch["graph"]
            orientation_target = layers.reaction_orientation_targets(
                reactant.unpack(), product.unpack(), auxiliary["views"]
            ).to(auxiliary["orientation_logits"].device)
            labeled = orientation_target >= 0
            if labeled.any():
                orientation_loss = F.cross_entropy(
                    auxiliary["orientation_logits"][labeled],
                    orientation_target[labeled],
                )
                all_loss += self.ch_orientation_weight * orientation_loss
                metric["C-H orientation loss"] = orientation_loss
                metric["C-H orientation accuracy"] = (
                    auxiliary["orientation_logits"][labeled].argmax(dim=-1)
                    == orientation_target[labeled]
                ).float().mean()
            metric["C-H orientation coverage"] = labeled.float().sum() / len(product)

        if self.ch_counterfactual:
            auxiliary = self._last_ch_auxiliary
            if self.ch_safety_weight > 0:
                base_margin = self._variadic_margin(
                    auxiliary["base_pred"], target, size
                ).detach()
                fused_margin = self._variadic_margin(pred, target, size)
                base_correct = base_margin > 0
                if base_correct.any():
                    safety_loss = F.relu(
                        base_margin[base_correct] - fused_margin[base_correct]
                    ).mean()
                    all_loss += self.ch_safety_weight * safety_loss
                    metric["C-H safety loss"] = safety_loss
                    metric["C-H protected baseline fraction"] = base_correct.float().mean()
            if self.ch_residual_weight > 0:
                residual_loss = auxiliary["edge_delta"].square().mean()
                all_loss += self.ch_residual_weight * residual_loss
                metric["C-H residual L2"] = residual_loss

        if self.oriented_mechanism:
            auxiliary = self._last_oriented_auxiliary
            reactant, product = batch["graph"]
            orientation_target = layers.reaction_orientation_targets(
                reactant.unpack(), product.unpack(), auxiliary["views"]
            ).to(auxiliary["orientation_logits"].device)
            labeled = orientation_target >= 0
            if labeled.any() and self.oriented_orientation_weight > 0:
                orientation_loss = F.cross_entropy(
                    auxiliary["orientation_logits"][labeled],
                    orientation_target[labeled],
                )
                all_loss += self.oriented_orientation_weight * orientation_loss
                metric["oriented mechanism orientation loss"] = orientation_loss
                metric["oriented mechanism orientation accuracy"] = (
                    auxiliary["orientation_logits"][labeled].argmax(dim=-1)
                    == orientation_target[labeled]
                ).float().mean()
            metric["oriented mechanism orientation coverage"] = (
                labeled.float().sum() / len(product)
            )
            competition_loss, competition_coverage = self._same_template_competition_loss(
                auxiliary["fused_edge_pred"],
                auxiliary["orientation_allowed"],
                auxiliary["orientation_signature_id"],
                orientation_target,
                target,
                product,
            )
            if competition_loss is not None and self.oriented_competition_weight > 0:
                all_loss += self.oriented_competition_weight * competition_loss
                metric["same-template competition loss"] = competition_loss
            metric["same-template competition coverage"] = competition_coverage
            metric["oriented admissible edge fraction"] = (
                auxiliary["orientation_allowed"].any(dim=-1).float().mean()
            )
            if self.oriented_residual_weight > 0:
                residual_loss = auxiliary["edge_delta"].square().mean()
                all_loss += self.oriented_residual_weight * residual_loss
                metric["oriented mechanism residual L2"] = residual_loss

        return all_loss, metric

    def _same_template_competition_loss(
            self, edge_score, allowed, signature_id, orientation_target,
            target_index, product):
        """Rank the gold site above alternatives with the same edit template."""
        losses = []
        eligible = 0
        edge_offset = 0
        for graph_id, graph in enumerate(product.unpack()):
            num_edge = int(graph.directed().num_edge)
            gold_edge = int(target_index[graph_id])
            if gold_edge >= num_edge:
                edge_offset += num_edge
                continue
            global_gold = edge_offset + gold_edge
            gold_orientation = int(orientation_target[global_gold])
            if gold_orientation < 0:
                edge_offset += num_edge
                continue
            gold_signature = signature_id[global_gold, gold_orientation]
            local_signature = signature_id[edge_offset:edge_offset + num_edge]
            local_allowed = allowed[edge_offset:edge_offset + num_edge]
            same_template = (
                (local_signature == gold_signature) & local_allowed
            ).any(dim=-1)
            same_template[gold_edge] = False
            if same_template.any():
                competitor = edge_score[edge_offset:edge_offset + num_edge][same_template].max()
                losses.append(F.relu(
                    self.oriented_competition_margin
                    - edge_score[global_gold] + competitor
                ))
                eligible += 1
            edge_offset += num_edge
        if edge_offset != len(edge_score):
            raise RuntimeError("Packed same-template competition tensors are misaligned")
        coverage = edge_score.new_tensor(float(eligible) / max(len(product), 1))
        return (torch.stack(losses).mean() if losses else None), coverage

    @staticmethod
    def _variadic_margin(value, target, size):
        offset = size.cumsum(0) - size
        absolute_target = offset + target
        target_value = value[absolute_target]
        competitor = value.clone()
        competitor[absolute_target] = -torch.inf
        competitor_value = functional.variadic_max(competitor, size)[0]
        return target_value - competitor_value

  
    def _collate(self, edge_data, node_data, graph):
        new_data = torch.zeros(len(edge_data) + len(node_data), *edge_data.shape[1:],
                               dtype=edge_data.dtype, device=edge_data.device)
        num_cum_xs = graph.num_cum_edges + graph.num_cum_nodes
        num_xs = graph.num_edges + graph.num_nodes
        starts = num_cum_xs - num_xs
        ends = starts + graph.num_edges
        index = functional.multi_slice_mask(starts, ends, num_cum_xs[-1])
        new_data[index] = edge_data
        new_data[~index] = node_data

        return new_data

    def target(self, batch):
        _, product = batch["graph"]
        graph = product.directed()
        target = self._collate(graph.edge_label, graph.node_label, graph)
        size = graph.num_edges + graph.num_nodes
        return target, size

    @staticmethod
    def _packed_planar_faces_like_dual_graph(product):
        faces = []
        node_offset = 0
        for graph in product.unpack():
            for face in graph.planar_face1():
                faces.append([int(node) + node_offset for node in face])
            node_offset += int(graph.num_node)
        return faces

    @staticmethod
    def _aggregate_face_features(faces, face_feature, num_node, normalize=False):
        """Map face tokens to atoms, optionally normalizing face incidence."""
        atom_feature = face_feature.new_zeros((int(num_node), face_feature.size(-1)))
        face_count = face_feature.new_zeros(int(num_node)) if normalize else None
        for face_index, face in enumerate(faces):
            # planar_face1 closes a face by repeating its first two atoms.
            for atom_index in face[:-2]:
                atom_index = int(atom_index)
                if atom_index < 0 or atom_index >= int(num_node):
                    raise RuntimeError(f"Planar face references invalid atom {atom_index}")
                atom_feature[atom_index] += face_feature[face_index]
                if face_count is not None:
                    face_count[atom_index] += 1
        if face_count is not None:
            atom_feature = atom_feature / face_count.clamp_min(1).sqrt().unsqueeze(-1)
        return atom_feature

    @staticmethod
    def _assert_canonical_product_edges(product):
        """Check the one-candidate-per-bond Stage-1 graph contract."""
        edge_index = product.edge_list[:, :2].long()
        if edge_index.numel() and not bool((edge_index[:, 0] < edge_index[:, 1]).all()):
            raise RuntimeError(
                "Stage-1 products must contain one canonical edge u < v per bond."
            )
        if edge_index.numel():
            keys = edge_index[:, 0] * int(product.num_node) + edge_index[:, 1]
            if int(torch.unique(keys).numel()) != int(keys.numel()):
                raise RuntimeError("Stage-1 product contains duplicate bond candidates")

    @staticmethod
    def _bond_tokens_for_directed_edges(output, graph, num_product_node):
        """Map canonical undirected DGT bond tokens to directed candidates."""
        if "bond_feature" not in output:
            return None
        bond_feature = output["bond_feature"]
        bond_index = output["bond_index"]
        if bond_feature.numel() == 0:
            return bond_feature.new_zeros((graph.edge_list.size(0), bond_feature.size(-1)))

        canonical = torch.sort(bond_index, dim=-1).values
        canonical_key = canonical[:, 0] * num_product_node + canonical[:, 1]
        canonical_key, canonical_order = canonical_key.sort()
        bond_feature = bond_feature[canonical_order]
        candidate = torch.sort(graph.edge_list[:, :2].long(), dim=-1).values
        candidate_key = candidate[:, 0] * num_product_node + candidate[:, 1]
        position = torch.searchsorted(canonical_key, candidate_key)
        if (position >= canonical_key.numel()).any() or not torch.equal(canonical_key[position], candidate_key):
            raise RuntimeError("A directed candidate edge could not be matched to a DGT bond token.")
        return bond_feature[position]

    def predict(self, batch, all_loss=None, metric=None):
        _, product = batch["graph"]    
        output = self.model(product, product.node_feature.float(), all_loss, metric)
        # In unknown-class mode, do not pass the label tensor to any feature
        # builder at all. This makes leakage impossible even if a future prior
        # implementation accidentally starts consulting its reaction argument.
        reaction = batch["reaction"] if "reaction" in self.feature else None
        atom_view = output["node_feature"]
        if self.atom_view_projection is not None:
            atom_view = self.atom_view_projection(atom_view)
        node_feature = [atom_view]
        graph = product.directed()
        self._assert_canonical_product_edges(graph)

        if getattr(self.model, "is_pair_center_model", False):
            pair_edge_feature = output.get("pair_edge_feature")
            if pair_edge_feature is None:
                raise RuntimeError("Pair-domain encoder did not return pair_edge_feature")
            if pair_edge_feature.size(0) != graph.edge_list.size(0):
                raise RuntimeError(
                    "Pair-domain edge output is not aligned with the canonical candidate graph: "
                    f"{pair_edge_feature.size(0)} vs {graph.edge_list.size(0)}"
                )
            node_feature = output["node_feature"]
            if self.fusion_mode == "gated":
                if self.model2 is None or self.pair_dual_fusion is None:
                    raise RuntimeError("FSpec gated fusion was not initialized with a dual branch")
                dual_product = product.dual_graph()
                dual_output = self.model2(
                    dual_product, dual_product.node_feature.float()
                )
                faces = self._packed_planar_faces_like_dual_graph(product)
                if len(faces) != len(dual_output["node_feature"]):
                    raise RuntimeError(
                        "Packed planar face count does not match dual-graph node output: "
                        f"{len(faces)} vs {len(dual_output['node_feature'])}"
                    )
                dualnode_feature = self._aggregate_face_features(
                    faces,
                    dual_output["node_feature"],
                    product.num_node,
                    # Degree-normalized incidence is required before the
                    # learned gate; otherwise atoms in many faces receive a
                    # systematically larger dual magnitude.
                    normalize=True,
                )
                node_feature, pair_edge_feature, node_gate, edge_gate = self.pair_dual_fusion(
                    node_feature,
                    pair_edge_feature,
                    dualnode_feature,
                    graph.edge_list[:, :2].long(),
                )
                self._last_pair_fusion_gates = {
                    "node": node_gate.detach(),
                    "edge": edge_gate.detach(),
                }
            elif self.model2 is not None:
                dual_product = product.dual_graph()
                dual_output = self.model2(
                    dual_product, dual_product.node_feature.float()
                )
                faces = self._packed_planar_faces_like_dual_graph(product)
                if len(faces) != len(dual_output["node_feature"]):
                    raise RuntimeError(
                        "Packed planar face count does not match dual-graph node output: "
                        f"{len(faces)} vs {len(dual_output['node_feature'])}"
                    )
                dualnode_feature = self._aggregate_face_features(
                    faces,
                    dual_output["node_feature"],
                    product.num_node,
                    normalize=self.fusion_mode == "balanced",
                )
                node_feature = torch.cat((node_feature, dualnode_feature), dim=-1)
                dual_edge_feature = dualnode_feature[graph.edge_list[:, :2]].flatten(1)
                pair_edge_feature = torch.cat(
                    (pair_edge_feature, dual_edge_feature), dim=-1
                )
            edge_pred = self.edge_mlp(pair_edge_feature).squeeze(-1)
            node_pred = self.node_mlp(node_feature).squeeze(-1)
            return self._collate(edge_pred, node_pred, graph)

        dualnode_feature = None
        if self.model2 is not None:
            # TorchDrug products already contain both bond directions.  This
            # is the representation consumed by the original dual branch.
            dual_product = product.dual_graph()
            dual_output = self.model2(dual_product, dual_product.node_feature.float())
            dual_feature = dual_output["node_feature"]
            faces = self._packed_planar_faces_like_dual_graph(product)
            if len(faces) != len(dual_feature):
                raise RuntimeError(
                    f"Packed planar face count ({len(faces)}) does not match dual graph nodes ({len(dual_feature)})."
                )
            dualnode_feature = self._aggregate_face_features(
                faces,
                dual_feature,
                product.num_node,
                normalize=self.fusion_mode == "balanced",
            )
            if self.fusion_mode == "balanced":
                dualnode_feature = self.face_view_projection(dualnode_feature)

        node_feature = [atom_view]
        edge_feature = []
        graph_feature = []
        for _feature in sorted(self.feature):
            if _feature == "reaction":
                reaction_feature = torch.zeros(len(graph), self.num_reaction, dtype=torch.float32, device=self.device)
                reaction_feature.scatter_(1, batch["reaction"].unsqueeze(-1), 1)
                graph_feature.append(reaction_feature)
            elif _feature == "graph":
                learned_graph_feature = output["graph_feature"]
                if self.graph_view_projection is not None:
                    learned_graph_feature = self.graph_view_projection(learned_graph_feature)
                graph_feature.append(learned_graph_feature)
            elif _feature == "atom":
                node_feature.append(graph.node_feature.float())
            elif _feature == "bond":
                edge_feature.append(graph.edge_feature.float()[..., :self.bond_feature_dim])
            else:
                raise ValueError("Unknown feature `%s`" % _feature)
        if self.retro_mtgr_edge_features:
            edge_feature.append(self._retro_mtgr_edge_feature(graph, reaction))
        if self.aee_edge_features:
            edge_feature.append(self._aee_edge_feature(graph, reaction))
        dgt_bond_feature = self._bond_tokens_for_directed_edges(output, graph, product.num_node)
        if dgt_bond_feature is not None:
            if self.bond_view_projection is not None:
                dgt_bond_feature = self.bond_view_projection(dgt_bond_feature)
            edge_feature.append(dgt_bond_feature)
        graph_feature = torch.cat(graph_feature, dim=-1)
        node_feature.append(graph_feature[graph.node2graph])

        if dualnode_feature is not None:
            node_feature.append(dualnode_feature)
        
        node_feature = torch.cat(node_feature, dim=-1)
        edge_feature.append(node_feature[graph.edge_list[:, :2]].flatten(1))
        edge_feature = torch.cat(edge_feature, dim=-1)
        edge_pred = self.edge_mlp(edge_feature).squeeze(-1)
        node_pred = self.node_mlp(node_feature).squeeze(-1)
        if self.ch_counterfactual:
            base_pred = self._collate(edge_pred, node_pred, graph)
            products = product.unpack()
            views = [self.ch_view_store.load(item) for item in products]
            edge_delta, orientation_logits = self.ch_counterfactual_encoder(
                node_feature, edge_feature, edge_pred, products, views
            )
            edge_pred = edge_pred + edge_delta
            self._last_ch_auxiliary = {
                "views": views,
                "base_pred": base_pred,
                "edge_delta": edge_delta,
                "orientation_logits": orientation_logits,
            }
        if self.oriented_mechanism:
            products = product.unpack()
            views = [self.oriented_mechanism_view_store.load(item) for item in products]
            oriented = self.oriented_mechanism_projector(
                node_feature, edge_feature, edge_pred, products, views
            )
            edge_pred = edge_pred + oriented["edge_delta"]
            allowed_edge = oriented["orientation_allowed"].any(dim=-1)
            edge_pred = edge_pred.masked_fill(~allowed_edge, -torch.inf)
            # This dataset's mechanistic event is an added product bond.  Node
            # candidates are excluded from the admissible output space.
            node_pred = torch.full_like(node_pred, -torch.inf)
            allowed_candidate = self._collate(
                allowed_edge,
                torch.zeros_like(node_pred, dtype=torch.bool),
                graph,
            )
            self._last_oriented_auxiliary = {
                **oriented,
                "views": views,
                "fused_edge_pred": edge_pred,
                "allowed_candidate": allowed_candidate,
            }
        pred = self._collate(edge_pred, node_pred, graph)
        
        return pred
       
    
    def evaluate(self, pred, target):
        target, size = target

        metric = {}
        ranking = []
        target = functional.variadic_max(target, size)[1]
        topk=10
        _, indices= functional.variadic_topk(pred, size, topk, largest=True)
        infinity = torch.iinfo(torch.long).max - 1
        for i in range(len(target)):
            for j in range(topk):
               if target[i] == indices[i][j]:
                  break
            else:
               j = infinity
            ranking.append(j + 1)      

        infinity = torch.iinfo(torch.long).max - 1
        ranking = torch.tensor(ranking, device=self.device)
        for _metric in self.metric:
            if _metric.startswith("top-"):
                threshold = int(_metric[4:])
                
                score = (ranking <= threshold).float().mean()
                metric["top-%d accuracy" % threshold] = score
            else:
                raise ValueError("Unknown metric `%s`" % _metric)

        return metric
    

    @torch.no_grad()
    def predict_synthon(self, batch, k=1):
        """
        Predict top-k synthons from target molecules.
        Parameters:
            batch (dict): batch of target molecules
            k (int, optional): return top-k results
        Returns:
            list of dict: top k records.
                Each record is a batch dict of keys ``synthon``, ``num_synthon``, ``reaction_center``,
                ``log_likelihood`` and ``reaction``.
        """
        _, product = batch["graph"]
        graph = product.directed()

        pred = self.predict(batch)
        _, size = self.target(batch)


        logp = functional.variadic_log_softmax(pred, size)
        
        with graph.graph():
            graph.product_id = torch.arange(len(graph), device=self.device)

        graph = graph.repeat_interleave(k)
        reaction = None
        if "reaction" in self.feature:
            if "reaction" not in batch:
                raise RuntimeError("Known-class center inference requires reaction labels")
            reaction = batch["reaction"].repeat_interleave(k)
        with graph.graph():
            graph.split_id = torch.arange(k, device=self.device).repeat(len(graph) // k)

        logp, center_topk = functional.variadic_topk(logp, size, k)
        logp = logp.flatten()
        center_topk = center_topk.flatten()

        is_edge = center_topk < graph.num_edges
        node_index = center_topk + graph.num_cum_nodes - graph.num_nodes - graph.num_edges
        edge_index = center_topk + graph.num_cum_edges - graph.num_edges
        center_topk_shifted = torch.cat([-torch.ones(1, dtype=torch.long, device=self.device),
                                         center_topk[:-1]])
        product_id_shifted = torch.cat([-torch.ones(1, dtype=torch.long, device=self.device),
                                        graph.product_id[:-1]])
        is_duplicate = (center_topk == center_topk_shifted) & (graph.product_id == product_id_shifted)
        node_index = node_index[~is_edge]
        edge_index = edge_index[is_edge]
        edge_mask = ~functional.as_mask(edge_index, graph.num_edge)

        reaction_center = torch.zeros(len(graph), 2, dtype=torch.long, device=self.device)
        reaction_center[is_edge] = graph.atom_map[graph.edge_list[edge_index, :2]]
        reaction_center[~is_edge, 0] = graph.atom_map[node_index]

        # remove the edges from products
        graph = graph.edge_mask(edge_mask)
        graph = graph[~is_duplicate]
        reaction_center = reaction_center[~is_duplicate]
        logp = logp[~is_duplicate]
        if reaction is not None:
            reaction = reaction[~is_duplicate]
        synthon, num_synthon = graph.connected_components()
        synthon = synthon.undirected() # (< num_graph * k)

        result = {
            "synthon": synthon,
            "num_synthon": num_synthon,
            "reaction_center": reaction_center,
            "log_likelihood": logp,
        }
        if reaction is not None:
            result["reaction"] = reaction

        return result

    @torch.no_grad()
    def sample_center_reaction(self, batch, k=1):
        _, product = batch["graph"]
        graph = product.directed()

        pred = self.predict(batch)
        _, size = self.target(batch)

        logp = functional.variadic_log_softmax(pred, size)

        with graph.graph():
            graph.product_id = torch.arange(len(graph), device=self.device)

        graph = graph.repeat_interleave(k)
        with graph.graph():
            graph.split_id = torch.arange(k, device=self.device).repeat(len(graph) // k)

        sample_k = functional.variadic_sample_distribution(logp.exp(), size, k)
        sample_k = sample_k.flatten()

        is_edge = sample_k < graph.num_edges
        node_index = sample_k + graph.num_cum_nodes - graph.num_nodes - graph.num_edges
        edge_index = sample_k + graph.num_cum_edges - graph.num_edges

        node_index = node_index[~is_edge]
        edge_index = edge_index[is_edge]

        reaction_center = torch.zeros(len(graph), 2, dtype=torch.long, device=self.device)
        reaction_center[is_edge] = graph.atom_map[graph.edge_list[edge_index, :2]]
        reaction_center[~is_edge, 0] = graph.atom_map[node_index]

        reaction_center = reaction_center.view(len(size), k, 2)

        return reaction_center


class RandomBFSOrder(object):

    def __call__(self, item):
        assert hasattr(item["graph"][0], "reaction_center")
        reactant, synthon = item["graph"]

        edge_list = reactant.edge_list[:, :2].tolist()
        neighbor = [[] for _ in range(reactant.num_node)]
        for h, t in edge_list:
            neighbor[h].append(t)
        depth = [-1] * reactant.num_node

        # select a mapped atom as BFS root
        reactant2id = reactant.atom_map
        id2synthon = -torch.ones(synthon.atom_map.max() + 1, dtype=torch.long, device=synthon.device)
        id2synthon[synthon.atom_map] = torch.arange(synthon.num_node, device=synthon.device)
        reactant2synthon = id2synthon[reactant2id]

        candidate = (reactant2synthon != -1).nonzero().squeeze(-1)
        i = candidate[torch.randint(len(candidate), (1,))].item()

        queue = deque([i])
        depth[i] = 0
        order = []
        while queue:
            h = queue.popleft()
            order.append(h)
            for t in neighbor[h]:
                if depth[t] == -1:
                    depth[t] = depth[h] + 1
                    queue.append(t)

        reactant = reactant.subgraph(order)

        if reactant.num_edge > 0:
            node_index = reactant.edge_list[:, :2]
            node_large = node_index.max(dim=-1)[0]
            node_small = node_index.min(dim=-1)[0]
            undirected_edge_id = node_large * (node_large + 1) + node_small
            undirected_edge_id = undirected_edge_id * 2 + (node_index[:, 0] > node_index[:, 1])

            # rearrange edges into autoregressive order
            edge_order = undirected_edge_id.argsort()
            reactant = reactant.edge_mask(edge_order)

        assert hasattr(reactant, "reaction_center")

        item = item.copy()
        item["graph"] = (reactant, synthon)

        return item


@R.register("tasks.SynthonCompletion")
class SynthonCompletion(tasks.Task, core.Configurable):
    """

    Synthon completion task.
    This class is a part of retrosynthesis prediction.
    Parameters:
        model (nn.Module): graph representation model
        feature (str or list of str, optional): additional features for prediction. Available features are
            reaction: type of the reaction
            graph: graph representation of the synthon
            atom: original atom feature
        num_mlp_layer (int, optional): number of MLP layers
    """

    _option_members = set(["feature"])

    def __init__(self, model, feature=("reaction", "graph", "atom"), num_mlp_layer=2, mlp_act='relu'):
        super(SynthonCompletion, self).__init__()
        self.model = model
        self.num_mlp_layer = num_mlp_layer
        self.feature = feature
        self.input_linear = nn.Linear(2, self.model.input_dim)
        self.mlp_act = mlp_act

    def preprocess(self, train_set, valid_set, test_set):
        reaction_types = set()
        atom_types = set()
        bond_types = set()
        for sample in train_set:
            reaction_types.add(sample["reaction"])
            for graph in sample["graph"]:
                atom_types.update(graph.atom_type.tolist())
                bond_types.update(graph.edge_list[:, 2].tolist())
        # TODO: only for fast debugging, to remove
        # atom_types = torch.tensor([5, 6, 7, 8, 9, 12, 14, 15, 16, 17, 29, 30, 34, 35, 50, 53])
        # bond_types = torch.tensor([0, 1, 2])
        atom_types = torch.tensor(sorted(atom_types))
        atom2id = -torch.ones(atom_types.max() + 1, dtype=torch.long)
        atom2id[atom_types] = torch.arange(len(atom_types))
        self.register_buffer("id2atom", atom_types)
        self.register_buffer("atom2id", atom2id)
        self.num_reaction = len(reaction_types)
        self.num_atom_type = len(atom_types)
        self.num_bond_type = len(bond_types)
        node_feature_dim = train_set[0]["graph"][0].node_feature.shape[-1]

        if isinstance(train_set, torch_data.Subset):
            dataset = train_set.dataset
        else:
            dataset = train_set
        dataset.transform = transforms.Compose([
            dataset.transform,
            RandomBFSOrder(),
        ])
        sig = inspect.signature(data.PackedMolecule.from_molecule)
        keys = set(sig.parameters.keys())
        kwargs = dataset.config_dict()
        feature_kwargs = {}
        for k, v in kwargs.items():
            if k in keys:
                feature_kwargs[k] = v
        self.feature_kwargs = feature_kwargs

        node_dim = self.model.output_dim
        edge_dim = 0
        graph_dim = 0
        for _feature in sorted(self.feature):
            if _feature == "reaction":
                graph_dim += self.num_reaction
            elif _feature == "graph":
                graph_dim += self.model.output_dim
            elif _feature == "atom":
                node_dim += node_feature_dim
            else:
                raise ValueError("Unknown feature `%s`" % _feature)

        self.new_atom_feature = nn.Embedding(self.num_atom_type, node_dim)

        node_dim += graph_dim  # inherit graph features
        edge_dim += node_dim * 2  # inherit node features

        hidden_dims = [self.model.output_dim] * (self.num_mlp_layer - 1)
        self.node_in_mlp = layers.MLP(node_dim, hidden_dims + [1], activation=self.mlp_act)
        self.node_out_mlp = layers.MLP(edge_dim, hidden_dims + [1], activation=self.mlp_act)
        self.edge_mlp = layers.MLP(edge_dim, hidden_dims + [1], activation=self.mlp_act)
        self.bond_mlp = layers.MLP(edge_dim, hidden_dims + [self.num_bond_type], activation=self.mlp_act)
        self.stop_mlp = layers.MLP(graph_dim, hidden_dims + [1], activation=self.mlp_act)

    def _update_molecule_feature(self, graphs):

        graphs = graphs.ion_to_molecule()
        mols = graphs.to_molecule(ignore_error=True)
        valid = [mol is not None for mol in mols]
        valid = torch.tensor(valid, device=graphs.device)
        new_graphs = type(graphs).from_molecule(mols, **self.feature_kwargs)

        node_feature = torch.zeros(graphs.num_node, *new_graphs.node_feature.shape[1:],
                                   dtype=new_graphs.node_feature.dtype, device=graphs.device)
        edge_feature = torch.zeros(graphs.num_edge, *new_graphs.edge_feature.shape[1:],
                                   dtype=new_graphs.edge_feature.dtype, device=graphs.device)
        bond_type = torch.zeros_like(graphs.bond_type)
        node_mask = valid[graphs.node2graph]
        edge_mask = valid[graphs.edge2graph]

        if node_mask.size()==torch.Size([0]):
            node_mask=torch.empty(0, dtype=torch.bool)
        if edge_mask.size()==torch.Size([0]):
            edge_mask=torch.empty(0, dtype=torch.bool)
        node_feature[node_mask] = new_graphs.node_feature.to(device=graphs.device)
        edge_feature[edge_mask] = new_graphs.edge_feature.to(device=graphs.device)
        bond_type[edge_mask] = new_graphs.bond_type.to(device=graphs.device)

        with graphs.node():
            graphs.node_feature = node_feature
        with graphs.edge():
            graphs.edge_feature = edge_feature
            graphs.bond_type = bond_type

        return graphs, valid

    @torch.no_grad()
    def _all_prefix_slice(self, num_xs, lengths=None):
        # extract a bunch of slices that correspond to the following num_repeat * n masks
        # ------ repeat 0 -----
        # graphs[0]: [0, 0, ..., 0]
        # ...
        # graphs[-1]: [0, 0, ..., 0]
        # ------ repeat 1 -----
        # graphs[0]: [1, 0, ..., 0]
        # ...
        # graphs[-1]: [1, 0, ..., 0]
        # ...
        # ------ repeat -1 -----
        # graphs[0]: [1, ..., 1, 0]
        # ...
        # graphs[-1]: [1, ..., 1, 0]
        num_cum_xs = num_xs.cumsum(0)
        starts = num_cum_xs - num_xs
        if lengths is None:
            num_max_x = num_xs.max().item()
            lengths = torch.arange(0, num_max_x, 2, device=num_xs.device)

        pack_offsets = torch.arange(len(lengths), device=num_xs.device) * num_cum_xs[-1]
        # starts, lengths, ends: (num_repeat, num_graph)
        starts = starts.unsqueeze(0) + pack_offsets.unsqueeze(-1)
        valid = lengths.unsqueeze(-1) <= num_xs.unsqueeze(0) - 2
        lengths = torch.min(lengths.unsqueeze(-1), num_xs.unsqueeze(0) - 2).clamp(0)
        ends = starts + lengths

        starts = starts.flatten()
        ends = ends.flatten()
        valid = valid.flatten()

        return starts, ends, valid

    @torch.no_grad()
    def _get_reaction_feature(self, reactant, synthon):

        def get_edge_map(graph, num_nodes):
            node_in, node_out = graph.edge_list.t()[:2]
            node_in2id = graph.atom_map[node_in]
            node_out2id = graph.atom_map[node_out]
            edge_map = node_in2id * num_nodes[graph.edge2graph] + node_out2id
            # edges containing any unmapped node is considered to be unmapped
            edge_map[(node_in2id == 0) | (node_out2id == 0)] = 0
            return edge_map

        def get_mapping(reactant_x, synthon_x, reactant_x2graph, synthon_x2graph):
            num_xs = scatter_max(reactant_x, reactant_x2graph)[0]
            num_xs = num_xs.clamp(0) + 1
            num_cum_xs = num_xs.cumsum(0)
            offset = num_cum_xs - num_xs
            reactant2id = reactant_x + offset[reactant_x2graph]
            synthon2id = synthon_x + offset[synthon_x2graph]
            assert synthon2id.min() > 0
            id2synthon = -torch.ones(num_cum_xs[-1], dtype=torch.long, device=self.device)
            id2synthon[synthon2id] = torch.arange(len(synthon2id), device=self.device)
            reactant2synthon = id2synthon[reactant2id]

            return reactant2synthon

        # reactant & synthon may have different number of nodes
        # reactant.num_nodes >= synthon.num_nodes
        assert (reactant.num_nodes >= synthon.num_nodes).all()
        reactant_edge_map = get_edge_map(reactant, reactant.num_nodes)
        synthon_edge_map = get_edge_map(synthon, reactant.num_nodes)

        node_r2s = get_mapping(reactant.atom_map, synthon.atom_map, reactant.node2graph, synthon.node2graph)
        edge_r2s = get_mapping(reactant_edge_map, synthon_edge_map, reactant.edge2graph, synthon.edge2graph)

        is_new_node = node_r2s == -1
        is_new_edge = edge_r2s == -1
        is_modified_edge = (edge_r2s != -1) & (reactant.bond_type != synthon.bond_type[edge_r2s])
        is_reaction_center = (reactant.atom_map > 0) & \
                             (reactant.atom_map.unsqueeze(-1) ==
                              reactant.reaction_center[reactant.node2graph]).any(dim=-1)

        return node_r2s, edge_r2s, is_new_node, is_new_edge, is_modified_edge, is_reaction_center

    @torch.no_grad()
    def all_edge(self, reactant, synthon):
        graph = reactant.clone()
        node_r2s, edge_r2s, is_new_node, is_new_edge, is_modified_edge, is_reaction_center = \
            self._get_reaction_feature(reactant, synthon)
        with graph.node():
            graph.node_r2s = node_r2s
            graph.is_new_node = is_new_node
            graph.is_reaction_center = is_reaction_center
        with graph.edge():
            graph.edge_r2s = edge_r2s
            graph.is_new_edge = is_new_edge
            graph.is_modified_edge = is_modified_edge

        starts, ends, valid = self._all_prefix_slice(reactant.num_edges)
        num_repeat = len(starts) // len(reactant)
        graph = graph.repeat(num_repeat)

        # autoregressive condition range for each sample
        condition_mask = functional.multi_slice_mask(starts, ends, graph.num_edge) # using cumsum
        # special case: end == graph.num_edge. In this case, valid is always false
        assert ends.max() <= graph.num_edge
        ends = ends.clamp(0, graph.num_edge - 1) # returns values from min  to max 
        node_in, node_out, bond_target = graph.edge_list[ends].t()
        # modified edges which don't appear in conditions should keep their old bond types
        # i.e. bond types in synthons
        unmodified = ~condition_mask & graph.is_modified_edge
        unmodified = unmodified.nonzero().squeeze(-1)
        assert not (graph.bond_type[unmodified] == synthon.bond_type[graph.edge_r2s[unmodified]]).any()
        graph.edge_list[unmodified, 2] = synthon.edge_list[graph.edge_r2s[unmodified], 2]

        reverse_target = graph.edge_list[ends][:, [1, 0, 2]]
        is_reverse_target = (graph.edge_list == reverse_target[graph.edge2graph]).all(dim=-1)
        # keep edges that exist in the synthon
        # remove the reverse of new target edges
        edge_mask = (condition_mask & ~is_reverse_target) | ~graph.is_new_edge

        atom_in = graph.atom_type[node_in]
        atom_out = graph.atom_type[node_out]
        # keep one supervision for undirected edges
        # remove samples that try to predict existing edges
        valid &= (node_in < node_out) & (graph.is_new_edge[ends] | graph.is_modified_edge[ends])
        graph = graph.edge_mask(edge_mask)

        # sanitize the molecules
        # this will change atom index, so we manually remap the target nodes
        compact_mapping = -torch.ones(graph.num_node, dtype=torch.long, device=self.device)
        node_mask = graph.degree_in + graph.degree_out > 0
        # special case: for graphs without any edge, the first node should be kept
        index = torch.arange(graph.num_node, device=self.device)
        single_node_mask = (graph.num_edges == 0)[graph.node2graph] & \
                           (index == (graph.num_cum_nodes - graph.num_nodes)[graph.node2graph])
        node_index = (node_mask | single_node_mask).nonzero().squeeze(-1)
        compact_mapping[node_index] = torch.arange(len(node_index), device=self.device)
        node_in = compact_mapping[node_in]
        node_out = compact_mapping[node_out]
        graph = graph.subgraph(node_index)

        node_in_target = node_in - graph.num_cum_nodes + graph.num_nodes
        assert (node_in_target[valid] < graph.num_nodes[valid]).all() and (node_in_target[valid] >= 0).all()
        # node2 might be a new node
        node_out_target = torch.where(node_out == -1, self.atom2id[atom_out] + graph.num_nodes,
                                  node_out - graph.num_cum_nodes + graph.num_nodes)
        stop_target = torch.zeros(len(node_in_target), device=self.device)

        graph = graph[valid]
        node_in_target = node_in_target[valid]
        node_out_target = node_out_target[valid]
        bond_target = bond_target[valid]
        stop_target = stop_target[valid]

        assert (graph.num_edges % 2 == 0).all()
        # node / edge features may change because we mask some nodes / edges
        graph, feature_valid = self._update_molecule_feature(graph)

        if feature_valid.size()==torch.Size([0]):
                feature_valid=torch.empty(0, dtype=torch.bool)
        return graph[feature_valid], node_in_target[feature_valid], node_out_target[feature_valid], \
               bond_target[feature_valid], stop_target[feature_valid]

    @torch.no_grad()
    def all_stop(self, reactant, synthon):
        graph = reactant.clone()
        node_r2s, edge_r2s, is_new_node, is_new_edge, is_modified_edge, is_reaction_center = \
            self._get_reaction_feature(reactant, synthon)
        with graph.node():
            graph.node_r2s = node_r2s
            graph.is_new_node = is_new_node
            graph.is_reaction_center = is_reaction_center
        with graph.edge():
            graph.edge_r2s = edge_r2s
            graph.is_new_edge = is_new_edge
            graph.is_modified_edge = is_modified_edge

        node_in_target = torch.zeros(len(graph), dtype=torch.long, device=self.device)
        node_out_target = torch.zeros_like(node_in_target)
        bond_target = torch.zeros_like(node_in_target)
        stop_target = torch.ones(len(graph), device=self.device)

        # keep consistent with other training data
        graph, feature_valid = self._update_molecule_feature(graph)

        return graph[feature_valid], node_in_target[feature_valid], node_out_target[feature_valid], \
               bond_target[feature_valid], stop_target[feature_valid]

    def forward(self, batch):
        """"""
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        pred, target = self.predict_and_target(batch, all_loss, metric)
        node_in_pred, node_out_pred, bond_pred, stop_pred = pred
        node_in_target, node_out_target, bond_target, stop_target, size = target

        loss = functional.variadic_cross_entropy(node_in_pred, node_in_target, size, reduction="none")
        loss = functional.masked_mean(loss, stop_target == 0)
        metric["node in ce loss"] = loss
        all_loss += loss

        loss = functional.variadic_cross_entropy(node_out_pred, node_out_target, size, reduction="none")
        loss = functional.masked_mean(loss, stop_target == 0)
        metric["node out ce loss"] = loss
        all_loss += loss

        loss = F.cross_entropy(bond_pred, bond_target, reduction="none")
        loss = functional.masked_mean(loss, stop_target == 0)
        metric["bond ce loss"] = loss
        all_loss += loss

        # Do we need to balance stop pred?
        loss = F.binary_cross_entropy_with_logits(stop_pred, stop_target)
        metric["stop bce loss"] = loss
        all_loss += loss

        metric["total loss"] = all_loss
        metric.update(self.evaluate(pred, target))

        return all_loss, metric

    def evaluate(self, pred, target):
        node_in_pred, node_out_pred, bond_pred, stop_pred = pred
        node_in_target, node_out_target, bond_target, stop_target, size = target

        metric = {}

        node_in_acc = metrics.variadic_accuracy(node_in_pred, node_in_target, size)
        accuracy = functional.masked_mean(node_in_acc, stop_target == 0)
        metric["node in accuracy"] = accuracy

        node_out_acc = metrics.variadic_accuracy(node_out_pred, node_out_target, size)
        accuracy = functional.masked_mean(node_out_acc, stop_target == 0)
        metric["node out accuracy"] = accuracy

        bond_acc = (bond_pred.argmax(-1) == bond_target).float()
        accuracy = functional.masked_mean(bond_acc, stop_target == 0)
        metric["bond accuracy"] = accuracy

        stop_acc = ((stop_pred > 0.5) == (stop_target > 0.5)).float()
        metric["stop accuracy"] = stop_acc.mean()

        total_acc = (node_in_acc > 0.5) & (node_out_acc > 0.5) & (bond_acc > 0.5) & (stop_acc > 0.5)
        total_acc = torch.where(stop_target == 0, total_acc, stop_acc > 0.5).float()
        metric["total accuracy"] = total_acc.mean()

        return metric

    def _cat(self, graphs):
        for i, graph in enumerate(graphs):
            if not isinstance(graph, data.PackedGraph):
                graphs[i] = graph.pack([graph])

        edge_list = torch.cat([graph.edge_list for graph in graphs])
        pack_num_nodes = torch.stack([graph.num_node for graph in graphs])
        pack_num_edges = torch.stack([graph.num_edge for graph in graphs])
        pack_num_cum_edges = pack_num_edges.cumsum(0)
        graph_index = pack_num_cum_edges < len(edge_list)
        pack_offsets = scatter_add(pack_num_nodes[graph_index], pack_num_cum_edges[graph_index],
                                   dim_size=len(edge_list))
        pack_offsets = pack_offsets.cumsum(0)

        edge_list[:, :2] += pack_offsets.unsqueeze(-1)
        offsets = torch.cat([graph._offsets for graph in graphs]) + pack_offsets

        edge_weight = torch.cat([graph.edge_weight for graph in graphs])
        num_nodes = torch.cat([graph.num_nodes for graph in graphs])
        num_edges = torch.cat([graph.num_edges for graph in graphs])
        num_relation = graphs[0].num_relation
        assert all(graph.num_relation == num_relation for graph in graphs)

        # only keep attributes that exist in all graphs
        keys = set(graphs[0].meta_dict.keys())
        for graph in graphs:
            keys = keys.intersection(graph.meta_dict.keys())

        meta_dict = {k: graphs[0].meta_dict[k] for k in keys}
        data_dict = {}
        for k in keys:
            data_dict[k] = torch.cat([graph.data_dict[k] for graph in graphs])

        return type(graphs[0])(edge_list, edge_weight=edge_weight,
                               num_nodes=num_nodes, num_edges=num_edges, num_relation=num_relation, offsets=offsets,
                               meta_dict=meta_dict, **data_dict)

    def target(self, batch):
        reactant, synthon = batch["graph"]

        graph1, node_in_target1, node_out_target1, bond_target1, stop_target1 = self.all_edge(reactant, synthon)
        graph2, node_in_target2, node_out_target2, bond_target2, stop_target2 = self.all_stop(reactant, synthon)

        node_in_target = torch.cat([node_in_target1, node_in_target2])
        node_out_target = torch.cat([node_out_target1, node_out_target2])
        bond_target = torch.cat([bond_target1, bond_target2])
        stop_target = torch.cat([stop_target1, stop_target2])
        size = torch.cat([graph1.num_nodes, graph2.num_nodes])
        # add new atom candidates into the size of each graph
        size_ext = size + self.num_atom_type

        return node_in_target, node_out_target, bond_target, stop_target, size_ext

    def _topk_action(self, graph, k):
        synthon_feature = torch.stack([graph.is_new_node, graph.is_reaction_center], dim=-1).float()
        node_feature = graph.node_feature.float() + self.input_linear(synthon_feature)
        output = self.model(graph, node_feature)

        node_feature = [output["node_feature"]]
        graph_feature = []
        for _feature in sorted(self.feature):
            if _feature == "reaction":
                reaction_feature = torch.zeros(len(graph), self.num_reaction, dtype=torch.float32, device=self.device)
                reaction_feature.scatter_(1, graph.reaction.unsqueeze(-1), 1)
                graph_feature.append(reaction_feature)
            elif _feature == "graph":
                graph_feature.append(output["graph_feature"])
            elif _feature == "atom":
                node_feature.append(graph.node_feature.float())
            else:
                raise ValueError("Unknown feature `%s`" % _feature)

        graph_feature = torch.cat(graph_feature, dim=-1)
        # inherit graph features
        node_feature.append(graph_feature[graph.node2graph])
        node_feature = torch.cat(node_feature, dim=-1)

        new_node_feature = self.new_atom_feature.weight.repeat(len(graph), 1)
        new_graph_feature = graph_feature.unsqueeze(1).repeat(1, self.num_atom_type, 1).flatten(0, 1)
        new_node_feature = torch.cat([new_node_feature, new_graph_feature], dim=-1)
        node_feature, num_nodes_ext = self._extend(node_feature, graph.num_nodes, new_node_feature)

        node2graph_ext = functional._size_to_index(num_nodes_ext)
        num_cum_nodes_ext = num_nodes_ext.cumsum(0)
        starts = num_cum_nodes_ext - num_nodes_ext + graph.num_nodes
        ends = num_cum_nodes_ext
        is_new_node = functional.multi_slice_mask(starts, ends, num_cum_nodes_ext[-1])
        infinity = float("inf")

        node_in_pred = self.node_in_mlp(node_feature).squeeze(-1)
        stop_pred = self.stop_mlp(graph_feature).squeeze(-1)

        # mask out node-in prediction on new atoms
        node_in_pred[is_new_node] = -infinity
        node_in_logp = functional.variadic_log_softmax(node_in_pred, num_nodes_ext) # (num_node,)
        stop_logp = F.logsigmoid(stop_pred)
        act_logp = F.logsigmoid(-stop_pred)
        node_in_topk = functional.variadic_topk(node_in_logp, num_nodes_ext, k)[1]
        assert (node_in_topk >= 0).all() and (node_in_topk < num_nodes_ext.unsqueeze(-1)).all()
        node_in = node_in_topk + (num_cum_nodes_ext - num_nodes_ext).unsqueeze(-1) # (num_graph, k)

        # (num_node, node_in_k, feature_dim)
        node_out_feature = torch.cat([node_feature[node_in][node2graph_ext],
                                      node_feature.unsqueeze(1).expand(-1, k, -1)], dim=-1)
        node_out_pred = self.node_out_mlp(node_out_feature).squeeze(-1)
        # mask out node-out prediction on self-loops
        node_out_pred.scatter_(0, node_in, -infinity)
        # (num_node, node_in_k)
        node_out_logp = functional.variadic_log_softmax(node_out_pred, num_nodes_ext)
        # (num_graph, node_out_k, node_in_k)
        node_out_topk = functional.variadic_topk(node_out_logp, num_nodes_ext, k)[1]
        assert (node_out_topk >= 0).all() and (node_out_topk < num_nodes_ext.view(-1, 1, 1)).all()
        node_out = node_out_topk + (num_cum_nodes_ext - num_nodes_ext).view(-1, 1, 1)

        # (num_graph, node_out_k, node_in_k, feature_dim * 2)
        edge = torch.stack([node_in.unsqueeze(1).expand_as(node_out), node_out], dim=-1)
        bond_feature = node_feature[edge].flatten(-2)
        bond_pred = self.bond_mlp(bond_feature).squeeze(-1)
        bond_logp = F.log_softmax(bond_pred, dim=-1) # (num_graph, node_out_k, node_in_k, num_relation)
        bond_type = torch.arange(bond_pred.shape[-1], device=self.device)
        bond_type = bond_type.view(1, 1, 1, -1).expand_as(bond_logp)

        # (num_graph, node_out_k, node_in_k, num_relation)
        node_in_logp = node_in_logp.gather(0, node_in.flatten(0, 1)).view(-1, 1, k, 1)
        node_out_logp = node_out_logp.gather(0, node_out.flatten(0, 1)).view(-1, k, k, 1)
        act_logp = act_logp.view(-1, 1, 1, 1)
        logp = node_in_logp + node_out_logp + bond_logp + act_logp

        # (num_graph, node_out_k, node_in_k, num_relation, 4)
        node_in_topk = node_in_topk.view(-1, 1, k, 1).expand_as(logp)
        node_out_topk = node_out_topk.view(-1, k, k, 1).expand_as(logp)
        action = torch.stack([node_in_topk, node_out_topk, bond_type, torch.zeros_like(bond_type)], dim=-1)

        # add stop action
        logp = torch.cat([logp.flatten(1), stop_logp.unsqueeze(-1)], dim=1)
        stop = torch.tensor([0, 0, 0, 1], device=self.device)
        stop = stop.view(1, 1, -1).expand(len(graph), -1, -1)
        action = torch.cat([action.flatten(1, -2), stop], dim=1)
        topk = logp.topk(k, dim=-1)[1]

        return action.gather(1, topk.unsqueeze(-1).expand(-1, -1, 4)), logp.gather(1, topk)

    def _apply_action(self, graph, action, logp):
        # only support non-variadic k-actions
        assert len(graph) == len(action)
        num_action = action.shape[1]

        graph = graph.repeat_interleave(num_action)

        action = action.flatten(0, 1) # (num_graph * k, 4)
        logp = logp.flatten(0, 1) # (num_graph * k)
        new_node_in, new_node_out, new_bond_type, stop = action.t()

        # add new nodes
        has_new_node = (new_node_out >= graph.num_nodes) & (stop == 0)
        new_atom_id = (new_node_out - graph.num_nodes)[has_new_node]
        new_atom_type = self.id2atom[new_atom_id]
        is_new_node = torch.ones(len(new_atom_type), dtype=torch.bool, device=self.device)
        is_reaction_center = torch.zeros(len(new_atom_type), dtype=torch.bool, device=self.device)
        atom_type, num_nodes = functional._extend(graph.atom_type, graph.num_nodes, new_atom_type, has_new_node)
        is_new_node = functional._extend(graph.is_new_node, graph.num_nodes, is_new_node, has_new_node)[0]
        is_reaction_center = functional._extend(graph.is_reaction_center, graph.num_nodes, is_reaction_center, has_new_node)[0]

        # cast to regular node ids
        new_node_out = torch.where(has_new_node, graph.num_nodes, new_node_out)

        # modify edges
        new_edge = torch.stack([new_node_in, new_node_out], dim=-1)
        edge_list = graph.edge_list.clone()
        bond_type = graph.bond_type.clone()
        edge_list[:, :2] -= graph._offsets.unsqueeze(-1)
        is_modified_edge = (edge_list[:, :2] == new_edge[graph.edge2graph]).all(dim=-1) & \
                           (stop[graph.edge2graph] == 0)
        has_modified_edge = scatter_max(is_modified_edge.long(), graph.edge2graph, dim_size=len(graph))[0] > 0
        bond_type[is_modified_edge] = new_bond_type[has_modified_edge]
        edge_list[is_modified_edge, 2] = new_bond_type[has_modified_edge]
        # modify reverse edges
        new_edge = new_edge.flip(-1)
        is_modified_edge = (edge_list[:, :2] == new_edge[graph.edge2graph]).all(dim=-1) & \
                           (stop[graph.edge2graph] == 0)
        bond_type[is_modified_edge] = new_bond_type[has_modified_edge]
        edge_list[is_modified_edge, 2] = new_bond_type[has_modified_edge]

        # add new edges
        has_new_edge = (~has_modified_edge) & (stop == 0)
        new_edge_list = torch.stack([new_node_in, new_node_out, new_bond_type], dim=-1)[has_new_edge]
        bond_type = functional._extend(bond_type, graph.num_edges, new_bond_type[has_new_edge], has_new_edge)[0]
        edge_list, num_edges = functional._extend(edge_list, graph.num_edges, new_edge_list, has_new_edge)
        # add reverse edges
        new_edge_list = torch.stack([new_node_out, new_node_in, new_bond_type], dim=-1)[has_new_edge]
        bond_type = functional._extend(bond_type, num_edges, new_bond_type[has_new_edge], has_new_edge)[0]
        edge_list, num_edges = functional._extend(edge_list, num_edges, new_edge_list, has_new_edge)

        logp = logp + graph.logp

        # inherit attributes
        data_dict = graph.data_dict
        meta_dict = graph.meta_dict
        for key in ["atom_type", "bond_type", "is_new_node", "is_reaction_center", "logp"]:
            data_dict.pop(key)
        # pad 0 for node / edge attributes
        for k, v in data_dict.items():
            if "node" in meta_dict[k]:
                shape = (len(new_atom_type), *v.shape[1:])
                new_data = torch.zeros(shape, dtype=v.dtype, device=self.device)
                data_dict[k] = functional._extend(v, graph.num_nodes, new_data, has_new_node)[0]
            if "edge" in meta_dict[k]:
                shape = (len(new_edge_list) * 2, *v.shape[1:])
                new_data = torch.zeros(shape, dtype=v.dtype, device=self.device)
                data_dict[k] = functional._extend(v, graph.num_edges, new_data, has_new_edge * 2)[0]

        new_graph = type(graph)(edge_list, atom_type=atom_type, bond_type=bond_type, num_nodes=num_nodes,
                                num_edges=num_edges, num_relation=graph.num_relation,
                                is_new_node=is_new_node, is_reaction_center=is_reaction_center, logp=logp,
                                meta_dict=meta_dict, **data_dict)
        with new_graph.graph():
            new_graph.is_stopped = stop == 1
        valid = logp > float("-inf")
        new_graph = new_graph[valid]

        new_graph, feature_valid = self._update_molecule_feature(new_graph)
        return new_graph[feature_valid]

    @torch.no_grad()
    def predict_reactant(self, batch, num_beam=10, max_prediction=20, max_step=20):
        if "synthon" in batch:
            synthon = batch["synthon"]
            synthon2product = functional._size_to_index(batch["num_synthon"])
            assert (synthon2product < len(batch["reaction"])).all()
            reaction = batch["reaction"][synthon2product]
        else:
            reactant, synthon = batch["graph"]
            reaction = batch["reaction"]

        # In any case, ensure that the synthon is a molecule rather than an ion
        # This is consistent across train/test routines in synthon completion
        synthon, feature_valid = self._update_molecule_feature(synthon)
        synthon = synthon[feature_valid]
        reaction = reaction[feature_valid]

        graph = synthon
        with graph.graph():
            # for convenience, because we need to manipulate graph a lot
            graph.reaction = reaction
            graph.synthon_id = torch.arange(len(graph), device=graph.device)
            if not hasattr(graph, "logp"):
                graph.logp = torch.zeros(len(graph), device=graph.device)
        with graph.node():
            graph.is_new_node = torch.zeros(graph.num_node, dtype=torch.bool, device=graph.device)
            graph.is_reaction_center = (graph.atom_map > 0) & \
                                       (graph.atom_map.unsqueeze(-1) ==
                                        graph.reaction_center[graph.node2graph]).any(dim=-1)

        result = []
        num_prediction = torch.zeros(len(synthon), dtype=torch.long, device=self.device)
        for i in range(max_step):
            #logger.warning("action step: %d" % i)
            #logger.warning("batched beam size: %d" % len(graph))

            # each candidate has #beam actions
            action, logp = self._topk_action(graph, num_beam)
            #print("action", action)
            #print("action.shape[1]",action.shape[1]) # it's always 1???

            # each candidate is expanded to at most #beam (depending on validity) new candidates
            new_graph = self._apply_action(graph, action, logp)
            # assert (new_graph[is_stopped].logp > float("-inf")).all()
            offset = -2 * (new_graph.logp.max() - new_graph.logp.min())
            key = new_graph.synthon_id * offset + new_graph.logp
            order = key.argsort(descending=True)
            new_graph = new_graph[order]

            num_candidate = new_graph.synthon_id.bincount(minlength=len(synthon))
            topk = functional.variadic_topk(new_graph.logp, num_candidate, num_beam)[1]
            topk_index = topk + (num_candidate.cumsum(0) - num_candidate).unsqueeze(-1)
            topk_index = torch.unique(topk_index)
            new_graph = new_graph[topk_index]
            result.append(new_graph[new_graph.is_stopped])
            #till now we chose k best candidates and what is new_graph.stopped?????
            #print("result111", result)
            #what happened here???
            num_added = scatter_add(new_graph.is_stopped.long(), new_graph.synthon_id, dim_size=len(synthon))
            num_prediction += num_added

            # remove samples that already hit max prediction
            is_continue = (~new_graph.is_stopped) & (num_prediction[new_graph.synthon_id] < max_prediction)
            graph = new_graph[is_continue]
            if len(graph) == 0:
                break

        result = self._cat(result)
        # sort by synthon id
        order = result.synthon_id.argsort()
        result = result[order]

        # remove duplicate predictions
        is_duplicate = []
        synthon_id = -1
        #print("result",result)
        for graph in result:
            if graph.synthon_id != synthon_id:
                synthon_id = graph.synthon_id
                smiles_set = set()
            #print("graph", graph)
            smiles = graph.to_smiles(isomeric=False, atom_map=False, canonical=True)
            is_duplicate.append(smiles in smiles_set)
            smiles_set.add(smiles)
        is_duplicate = torch.tensor(is_duplicate, device=self.device)
        #print("is_duplicate", is_duplicate)
        result = result[~is_duplicate]
        num_prediction = result.synthon_id.bincount(minlength=len(synthon))

        # remove extra predictions
        topk = functional.variadic_topk(result.logp, num_prediction, max_prediction)[1]
        topk_index = topk + (num_prediction.cumsum(0) - num_prediction).unsqueeze(-1)
        topk_index = topk_index.flatten(0)
        topk_index_shifted = torch.cat([-torch.ones(1, dtype=torch.long, device=self.device), topk_index[:-1]])
        is_duplicate = topk_index == topk_index_shifted
        result = result[topk_index[~is_duplicate]]

        #print("result", result)
        return result # (< num_graph * max_prediction)

    def _extend(self, data, num_xs, input, input2graph=None):
        if input2graph is None:
            num_input_per_graph = len(input) // len(num_xs)
            input2graph = torch.arange(len(num_xs), device=data.device).unsqueeze(-1)
            input2graph = input2graph.repeat(1, num_input_per_graph).flatten()
        num_inputs = input2graph.bincount(minlength=len(num_xs))
        new_num_xs = num_xs + num_inputs
        new_num_cum_xs = new_num_xs.cumsum(0)
        new_num_x = new_num_cum_xs[-1].item()
        new_data = torch.zeros(new_num_x, *data.shape[1:], dtype=data.dtype, device=data.device)
        starts = new_num_cum_xs - new_num_xs
        ends = starts + num_xs
        index = functional.multi_slice_mask(starts, ends, new_num_x)
        new_data[index] = data
        new_data[~index] = input
        return new_data, new_num_xs

    def predict_and_target(self, batch, all_loss=None, metric=None):
        reactant, synthon = batch["graph"]
        reactant = reactant.clone()
        with reactant.graph():
            reactant.reaction = batch["reaction"]

        graph1, node_in_target1, node_out_target1, bond_target1, stop_target1 = self.all_edge(reactant, synthon)
        graph2, node_in_target2, node_out_target2, bond_target2, stop_target2 = self.all_stop(reactant, synthon)

        graph = self._cat([graph1, graph2])

        node_in_target = torch.cat([node_in_target1, node_in_target2])
        node_out_target = torch.cat([node_out_target1, node_out_target2])
        bond_target = torch.cat([bond_target1, bond_target2])
        stop_target = torch.cat([stop_target1, stop_target2])
        size = graph.num_nodes
        # add new atom candidates into the size of each graph
        size_ext = size + self.num_atom_type

        synthon_feature = torch.stack([graph.is_new_node, graph.is_reaction_center], dim=-1).float()
        node_feature = graph.node_feature.float() + self.input_linear(synthon_feature)
        output = self.model(graph, node_feature, all_loss, metric)

        node_feature = [output["node_feature"]]
        graph_feature = []
        for _feature in sorted(self.feature):
            if _feature == "reaction":
                reaction_feature = torch.zeros(len(graph), self.num_reaction, dtype=torch.float32, device=self.device)
                reaction_feature.scatter_(1, graph.reaction.unsqueeze(-1), 1)
                graph_feature.append(reaction_feature)
            elif _feature == "graph":
                graph_feature.append(output["graph_feature"])
            elif _feature == "atom":
                node_feature.append(graph.node_feature)
            else:
                raise ValueError("Unknown feature `%s`" % _feature)

        graph_feature = torch.cat(graph_feature, dim=-1)
        # inherit graph features
        node_feature.append(graph_feature[graph.node2graph])
        node_feature = torch.cat(node_feature, dim=-1)

        new_node_feature = self.new_atom_feature.weight.repeat(len(graph), 1)
        new_graph_feature = graph_feature.unsqueeze(1).repeat(1, self.num_atom_type, 1).flatten(0, 1)
        new_node_feature = torch.cat([new_node_feature, new_graph_feature], dim=-1)
        node_feature, num_nodes_ext = self._extend(node_feature, graph.num_nodes, new_node_feature)
        assert (num_nodes_ext == size_ext).all()

        node2graph_ext = functional._size_to_index(num_nodes_ext)
        num_cum_nodes_ext = num_nodes_ext.cumsum(0)
        starts = num_cum_nodes_ext - num_nodes_ext + graph.num_nodes
        ends = num_cum_nodes_ext
        is_new_node = functional.multi_slice_mask(starts, ends, num_cum_nodes_ext[-1])

        node_in = node_in_target + num_cum_nodes_ext - num_nodes_ext
        node_out = node_out_target + num_cum_nodes_ext - num_nodes_ext
        edge = torch.stack([node_in, node_out], dim=-1)

        node_out_feature = torch.cat([node_feature[node_in][node2graph_ext], node_feature], dim=-1)
        bond_feature = node_feature[edge].flatten(-2)
        node_in_pred = self.node_in_mlp(node_feature).squeeze(-1)
        node_out_pred = self.node_out_mlp(node_out_feature).squeeze(-1)
        bond_pred = self.bond_mlp(bond_feature).squeeze(-1)
        stop_pred = self.stop_mlp(graph_feature).squeeze(-1)

        infinity = torch.tensor(float("inf"), device=self.device)
        # mask out node-in prediction on new atoms
        node_in_pred[is_new_node] = -infinity
        # mask out node-out prediction on self-loops
        node_out_pred[node_in] = -infinity

        return (node_in_pred, node_out_pred, bond_pred, stop_pred), \
               (node_in_target, node_out_target, bond_target, stop_target, size_ext)


@R.register("tasks.Retrosynthesis")
class Retrosynthesis(tasks.Task, core.Configurable):
    """

    Retrosynthesis task.
    This class wraps pretrained center identification and synthon completion modeules into a pipeline.
    Parameters:
        center_identification (CenterIdentification): sub task of center identification
        synthon_completion (SynthonCompletion): sub task of synthon completion
        center_topk (int, optional): number of reaction centers to predict for each product
        num_synthon_beam (int, optional): size of beam search for each synthon
        max_prediction (int, optional): max number of final predictions for each product
        metric (str or list of str, optional): metric(s). Available metrics are ``top-K``.
    """

    _option_members = set(["metric"])

    def __init__(self, center_identification, synthon_completion, center_topk=2, num_synthon_beam=10, max_prediction=20, max_step=20,
                 metric=("top-1", "top-3", "top-5", "top-10")):
        super(Retrosynthesis, self).__init__()
        self.center_identification = center_identification
        self.synthon_completion = synthon_completion
        self.center_topk = center_topk
        self.num_synthon_beam = num_synthon_beam
        self.max_prediction = max_prediction
        self.metric = metric
        self.max_step = max_step

    def load_state_dict(self, state_dict, strict=True):
        if not strict:
            raise ValueError("Retrosynthesis only supports load_state_dict() with strict=True")
        keys = set(state_dict.keys())
        #print("1", state_dict)
        for model in [self.center_identification, self.synthon_completion]:
            #print("2", model)
            if set(model.state_dict().keys()) == keys:
                return model.load_state_dict(state_dict, strict)
        raise RuntimeError("Neither of sub modules matches with state_dict")

    def predict(self, batch, all_loss=None, metric=None):
        synthon_batch = self.center_identification.predict_synthon(batch, self.center_topk)

        synthon = synthon_batch["synthon"]
        #print(synthon.to_smiles())
        num_synthon = synthon_batch["num_synthon"]
        #print( num_synthon >= 3)
        assert (num_synthon >= 1).all() and (num_synthon <= 2).all()
        synthon2split = functional._size_to_index(num_synthon)
        with synthon.graph():
            synthon.reaction_center = synthon_batch["reaction_center"][synthon2split]
            synthon.split_logp = synthon_batch["log_likelihood"][synthon2split]

        reactant = self.synthon_completion.predict_reactant(synthon_batch, self.num_synthon_beam, self.max_prediction, self.max_step)

        logps = []
        reactant_ids = []
        product_ids = []

        # case 1: one synthon
        is_single = num_synthon[synthon2split[reactant.synthon_id]] == 1
        reactant_id = is_single.nonzero().squeeze(-1)
        logps.append(reactant.split_logp[reactant_id] + reactant.logp[reactant_id])
        product_ids.append(reactant.product_id[reactant_id])
        # pad -1
        reactant_ids.append(torch.stack([reactant_id, -torch.ones_like(reactant_id)], dim=-1))

        # case 2: two synthons
        # use proposal to avoid O(n^2) complexity
        reactant1 = torch.arange(len(reactant), device=self.device)
        reactant1 = reactant1.unsqueeze(-1).expand(-1, self.max_prediction * 2)
        reactant2 = reactant1 + torch.arange(self.max_prediction * 2, device=self.device)
        valid = reactant2 < len(reactant)
        reactant1 = reactant1[valid]
        reactant2 = reactant2[valid]
        synthon1 = reactant.synthon_id[reactant1]
        synthon2 = reactant.synthon_id[reactant2]
        valid = (synthon1 < synthon2) & (synthon2split[synthon1] == synthon2split[synthon2])
        reactant1 = reactant1[valid]
        reactant2 = reactant2[valid]
        logps.append(reactant.split_logp[reactant1] + reactant.logp[reactant1] + reactant.logp[reactant2])
        product_ids.append(reactant.product_id[reactant1])
        reactant_ids.append(torch.stack([reactant1, reactant2], dim=-1))

        # combine case 1 & 2
        logps = torch.cat(logps)
        reactant_ids = torch.cat(reactant_ids)
        product_ids = torch.cat(product_ids)

        order = product_ids.argsort()
        logps = logps[order]
        reactant_ids = reactant_ids[order]
        num_prediction = product_ids.bincount()
        logps, topk = functional.variadic_topk(logps, num_prediction, self.max_prediction)
        topk_index = topk + (num_prediction.cumsum(0) - num_prediction).unsqueeze(-1)
        topk_index_shifted = torch.cat([-torch.ones(len(topk_index), 1, dtype=torch.long, device=self.device),
                                        topk_index[:, :-1]], dim=-1)
        is_duplicate = topk_index == topk_index_shifted
        reactant_id = reactant_ids[topk_index] # (num_graph, k, 2)

        # why we need to repeat the graph?
        # because reactant_id may be duplicated, which is not directly supported by graph indexing
        is_padding = reactant_id == -1
        num_synthon = (~is_padding).sum(dim=-1)
        num_synthon = num_synthon[~is_duplicate]
        logps = logps[~is_duplicate]
        offset = torch.arange(self.max_prediction, device=self.device) * len(reactant)
        reactant_id = reactant_id + offset.view(1, -1, 1)
        reactant_id = reactant_id[~(is_padding | is_duplicate.unsqueeze(-1))]
        reactant = reactant.repeat(self.max_prediction)
        reactant = reactant[reactant_id]
        assert num_synthon.sum() == len(reactant)
        synthon2graph = functional._size_to_index(num_synthon)
        first_synthon = num_synthon.cumsum(0) - num_synthon
        # inherit graph attributes from the first synthon
        data_dict = reactant.data_mask(graph_index=first_synthon, include="graph")[0]
        # merge synthon pairs from the same split into a single graph
        reactant = reactant.merge(synthon2graph)
        with reactant.graph():
            for k, v in data_dict.items():
                setattr(reactant, k, v)
            reactant.logps = logps

        num_prediction = reactant.product_id.bincount()

        return reactant, num_prediction # (num_graph * k)

    def target(self, batch):
        reactant, product = batch["graph"]
        reactant = reactant.ion_to_molecule()
        return reactant

    def evaluate(self, pred, target):
        pred, num_prediction = pred
        infinity = torch.iinfo(torch.long).max - 1

        metric = {}
        ranking = []
        # any better solution for parallel graph isomorphism?
        num_cum_prediction = num_prediction.cumsum(0)
        for i in range(len(target)):
            target_smiles = target[i].to_smiles(isomeric=False, atom_map=False, canonical=True)
            offset = (num_cum_prediction[i] - num_prediction[i]).item()
            for j in range(num_prediction[i]):
                pred_smiles = pred[offset + j].to_smiles(isomeric=False, atom_map=False, canonical=True)
                if pred_smiles == target_smiles:
                    break
            else:
                j = infinity
            ranking.append(j + 1)

        ranking = torch.tensor(ranking, device=self.device)
        for _metric in self.metric:
            if _metric.startswith("top-"):
                threshold = int(_metric[4:])
                score = (ranking <= threshold).float().mean()
                metric["top-%d accuracy" % threshold] = score
            else:
                raise ValueError("Unknown metric `%s`" % _metric)

        return metric
