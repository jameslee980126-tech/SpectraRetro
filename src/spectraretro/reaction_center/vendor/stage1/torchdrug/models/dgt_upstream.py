"""Faithful Stage-1 adapter around the vendored upstream DGT implementation.

The DGT computation is *not* reimplemented in this file.  The adapter loads
the pinned source files under ``third_party/DGT`` and instantiates upstream
``FeatureEncoder`` / ``DGTLayer`` objects directly.  Project-specific code is
limited to:

1. converting a TorchDrug product batch to PyG ``Data`` objects;
2. invoking the upstream preprocessing functions in their original order;
3. exposing upstream atom and undirected-bond tokens through the TorchDrug
   Stage-1 encoder contract.

The strict source hashes deliberately make an edited or incomplete upstream
checkout fail before a model can be trained.
"""

from collections.abc import Sequence
from functools import lru_cache
import hashlib
import importlib
from importlib import metadata
from pathlib import Path
import json
import os
import sys
import zipfile

import torch
from torch import nn

from torchdrug import core
from torchdrug.core import Registry as R


UPSTREAM_COMMIT = "2f5a4c3ef2e376d60238f481c36a87eeb5a581e8"
UPSTREAM_ARCHIVE_SHA256 = "1cb35b1a9ea5e892f9b35a2b76e0ad2e24042e407abde6ed1f3e1f7b6bc83f0e"
UPSTREAM_PYTHON_VERSION = "3.10.16"
UPSTREAM_CUDA_VERSION = "12.1"
UPSTREAM_PACKAGE_VERSIONS = {
    "torch": "2.1.0",
    "torchvision": "0.16.0",
    "torchaudio": "2.1.0",
    "torch-geometric": "2.0.4",
    "torchmetrics": "1.2.0",
    "ogb": "1.3.6",
    "tensorboardX": "2.6.2.2",
    "wandb": "0.18.7",
    "rdkit": "2025.9.1",
    "libauc": "1.1.0",
}
DGT_RUNTIME_PROFILE_ENV = "DGT_RUNTIME_PROFILE"
RUNTIME_PROFILE_EXACT = "exact"
RUNTIME_PROFILE_LOCAL_COMPAT = "local_compat"


def dgt_runtime_profile():
    profile = os.environ.get(DGT_RUNTIME_PROFILE_ENV, RUNTIME_PROFILE_EXACT)
    if profile not in {RUNTIME_PROFILE_EXACT, RUNTIME_PROFILE_LOCAL_COMPAT}:
        raise RuntimeError(
            f"Unsupported {DGT_RUNTIME_PROFILE_ENV}={profile!r}; expected "
            f"{RUNTIME_PROFILE_EXACT!r} or {RUNTIME_PROFILE_LOCAL_COMPAT!r}"
        )
    return profile

# GraphGym keeps its configuration in a process-global singleton and upstream
# ``line_graph`` reads that singleton directly.  Mixing two DGT configurations
# in one Python process would therefore make the first model silently use the
# second model's preprocessing settings.  Fail closed instead.
_ACTIVE_DGT_CONFIG_SIGNATURE = None


def _live_dgt_config_signature(cfg):
    return {
        "input_dim": int(cfg.share.dim_in),
        "edge_input_dim": int(cfg.dataset.edge_encoder_num_types),
        "hidden_dim": int(cfg.gnn.dim_inner),
        "num_layers": int(cfg.gt.layers),
        "num_heads": int(cfg.gt.n_heads),
        "dropout": float(cfg.gt.dropout),
        "attn_dropout": float(cfg.gt.attn_dropout),
        "batch_norm": bool(cfg.gt.batch_norm),
        "layer_norm": bool(cfg.gt.layer_norm),
        "rwse_steps": tuple(int(step) for step in cfg.posenc_RWSE.kernel.times),
        "spd_max_length": int(cfg.dataset.spd_max_length),
        "rings_max_length": int(cfg.dataset.rings_max_length),
        "rings_coalesce_edges": bool(cfg.dataset.rings_coalesce_edges),
    }

# Every upstream file imported or used by this adapter is pinned.  Paths are
# POSIX-style so the manifest is stable on Windows and Linux.
UPSTREAM_FILE_SHA256 = {
    "configs/physical_chemistry/ESOL-RWSE-SPDE-Rings.yaml": "33fb3a0e1cb05266c71765e1727e2dd804b46fe88e79aa36930e1652b75d134d",
    "graphgps/config/custom_gnn_config.py": "436e005b1df40c23037eab68bf3f9256601d3c5080d1ab39cb6310dc8d1f55c3",
    "graphgps/config/dataset_config.py": "49a0ab6eeb1cadca6bce3900a6129869c7a4beca00448eff3be8b5463bb63c82",
    "graphgps/config/defaults_config.py": "630104921ccb8a72bc87a475b2f2cd37704e13b7b48db8ca869c0f23f9db184c",
    "graphgps/config/example.py": "85daaa6f6d255fd74b49af729d82b5abc49d3b01fba23129a7e2edb26e2a0400",
    "graphgps/config/gt_config.py": "321131bc6f3cc332fdd65fae670ad85be22a9ca5ed2039a2967deb515c4fb5c6",
    "graphgps/config/optimizers_config.py": "dbda9ad486976f11af66e3a6ac5e038e1faf9582f5516a6b1649a2066114c9e8",
    "graphgps/config/posenc_config.py": "1138490c402d8bd8bdda9b4bcc6fe7f15e16961cc2234daca617c9150a58dc02",
    "graphgps/config/pretrained_config.py": "4f74e775451367923d6bb25543916cbe1a61052312ca586f471d5091c676c87a",
    "graphgps/config/split_config.py": "5a80a1566a1af925e8c6b04becac264939ab08616afecf0a8d0a311d199b5426",
    "graphgps/config/wandb_config.py": "36315cc0b87eb17baafc13bb58e8c2d5fdcc4af8a3381aeb1a82acf0b9e2a73e",
    "graphgps/encoder/composed_edge_encoders.py": "b2f23ed1f67c0667ab2277294dc1dcfc7609e9f1fcbde56d3574a10d8283ccbe",
    "graphgps/encoder/dense_edge_encoder.py": "413c59c1746462edd0fb3410f9a290c4fd22ac472e68ba881c679707a545db40",
    "graphgps/encoder/linear_edge_encoder.py": "cc980536fd4a70d81c6bec4a79df9722533dee08169d5b6012651d75cbf28853",
    "graphgps/encoder/linear_node_encoder.py": "5a97ce6d16f965c57bf5483147384cc96e46417947baf6e45822a98aef7a64cb",
    "graphgps/encoder/relative_pe_encoder.py": "fb893c5e389d7f8f5e3902b00d106631306ef9089f38d924ac7e1d9dbf6f87b5",
    "graphgps/encoder/topology_edge_encoder.py": "83b8427329ddc3c9d46b14c4795416eca33055b76064d66f742ee0c397efb2f5",
    "graphgps/head/san_graph.py": "c7f1e0e9a0d2028adb416120483fb374ae761406764740b89d0cb9f0c06caec3",
    "graphgps/layer/dgt_layer.py": "4e397b489648dd7ffb01630b23423a425c1017af2b444786247968bdf794d60f",
    "graphgps/network/dgt_model.py": "7b6f239e2944cffd70c276ca087cb9e1efa36fbdaf38a364d0c45be32d51dfab",
    "graphgps/transform/posenc_stats.py": "3180a073d05fbd83502c2b62995be989c069fd6462c5b5b7c15ad171293bd59e",
    "graphgps/transform/transforms.py": "a1ae6128e2bc9fb8d1247c14469d1d8b3b1500153836567820f7efc2f6a4f974",
}


def default_upstream_root():
    return Path(__file__).resolve().parents[5] / "third_party" / "DGT"


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_upstream_environment():
    """Reject formal runs outside the environment declared by upstream DGT."""
    profile = dgt_runtime_profile()
    if profile == RUNTIME_PROFILE_LOCAL_COMPAT:
        from dgt_local_compat import RING_BACKEND
        import networkx as nx

        if not hasattr(nx, "chordless_cycles"):
            raise RuntimeError(
                "DGT local compatibility requires NetworkX with chordless_cycles"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("DGT local compatibility training requires a CUDA GPU")
        if torch.version.cuda is None:
            raise RuntimeError("The local DGT PyTorch build has no CUDA runtime")
        print(
            "PASS: DGT local_compat runtime; complete pinned upstream source is used "
            f"with ring_backend={RING_BACKEND}, torch={torch.__version__}, "
            f"cuda={torch.version.cuda}. This is not the exact Linux environment."
        )
        return True

    mismatches = []
    python_version = ".".join(map(str, sys.version_info[:3]))
    if python_version != UPSTREAM_PYTHON_VERSION:
        mismatches.append(
            f"python (expected {UPSTREAM_PYTHON_VERSION}, got {python_version})"
        )
    for package, expected in UPSTREAM_PACKAGE_VERSIONS.items():
        try:
            actual = metadata.version(package)
        except metadata.PackageNotFoundError:
            actual = "MISSING"
        if actual != expected:
            mismatches.append(f"{package} (expected {expected}, got {actual})")
    try:
        import graph_tool
        actual_graph_tool = str(getattr(graph_tool, "__version__", "UNKNOWN"))
        if not actual_graph_tool.startswith("2.45"):
            mismatches.append(
                f"graph-tool (expected 2.45, got {actual_graph_tool})"
            )
    except Exception as error:
        mismatches.append(f"graph-tool (unavailable: {type(error).__name__}: {error})")
    if torch.version.cuda != UPSTREAM_CUDA_VERSION:
        mismatches.append(
            f"torch CUDA (expected {UPSTREAM_CUDA_VERSION}, got {torch.version.cuda})"
        )
    if mismatches:
        raise RuntimeError(
            "Formal upstream DGT is locked because the runtime differs from the "
            "downloaded repository's declared environment:\n  " + "\n  ".join(mismatches)
        )
    return True


def verify_upstream_source(upstream_root):
    """Fail closed unless all DGT files match the pinned upstream commit."""
    upstream_root = Path(upstream_root).resolve()
    archive_path = upstream_root.parent / "DGT-main.zip"
    if not archive_path.is_file():
        raise RuntimeError(
            f"Missing pinned upstream archive used as source-of-truth: {archive_path}"
        )
    archive_hash = _sha256(archive_path)
    if archive_hash != UPSTREAM_ARCHIVE_SHA256:
        raise RuntimeError(
            "Upstream DGT archive hash mismatch: expected "
            f"{UPSTREAM_ARCHIVE_SHA256}, got {archive_hash}"
        )

    # Compare every file in the extracted checkout against the pinned archive,
    # not just the modules currently known to the adapter. This prevents a
    # partial or locally edited upstream tree from being treated as canonical.
    archive_missing, archive_changed = [], []
    with zipfile.ZipFile(archive_path, "r") as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            parts = Path(member.filename).parts
            if len(parts) < 2:
                continue
            relative_path = Path(*parts[1:])
            extracted_path = upstream_root / relative_path
            if not extracted_path.is_file():
                archive_missing.append(relative_path.as_posix())
                continue
            expected_hash = hashlib.sha256(archive.read(member)).hexdigest()
            actual_hash = _sha256(extracted_path)
            if actual_hash != expected_hash:
                archive_changed.append(relative_path.as_posix())
    if archive_missing or archive_changed:
        raise RuntimeError(
            "Extracted DGT checkout differs from the pinned upstream archive: "
            f"missing={archive_missing}; changed={archive_changed}"
        )

    missing, changed = [], []
    for relative_path, expected_hash in UPSTREAM_FILE_SHA256.items():
        path = upstream_root / Path(relative_path)
        if not path.is_file():
            missing.append(relative_path)
            continue
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            changed.append((relative_path, expected_hash, actual_hash))
    if missing or changed:
        details = []
        if missing:
            details.append("missing=" + ", ".join(missing))
        if changed:
            details.append(
                "changed=" + ", ".join(
                    f"{path} (expected {expected}, got {actual})"
                    for path, expected, actual in changed
                )
            )
        raise RuntimeError(
            "Vendored DGT source does not match pinned commit "
            f"{UPSTREAM_COMMIT}: " + "; ".join(details)
        )
    return upstream_root


def _import_complete_upstream_package(upstream_root):
    """Execute the upstream package's own complete registration path."""
    # In the Windows PyTorch 2.8 compatibility stack, importing Lightning
    # through GraphGym before torchvision can trigger a Python 3.9 ABC
    # metaclass conflict.  Eagerly importing the matching torchvision wheel
    # changes no DGT code or tensors and makes the complete registration path
    # deterministic.
    if dgt_runtime_profile() == RUNTIME_PROFILE_LOCAL_COMPAT:
        import torchvision  # noqa: F401
    graphgps_root = upstream_root / "graphgps"
    existing = sys.modules.get("graphgps")
    if existing is not None:
        existing_paths = {str(Path(path).resolve()) for path in getattr(existing, "__path__", [])}
        if str(graphgps_root.resolve()) not in existing_paths:
            raise RuntimeError(
                "A different `graphgps` package is already imported. Start a clean process "
                "before constructing the faithful DGT adapter."
            )
        return
    upstream_root_string = str(upstream_root)
    if upstream_root_string not in sys.path:
        sys.path.insert(0, upstream_root_string)
    package = importlib.import_module("graphgps")
    package_path = Path(package.__file__).resolve()
    if package_path != (graphgps_root / "__init__.py").resolve():
        raise RuntimeError(f"Imported a non-vendored graphgps package: {package_path}")


@lru_cache(maxsize=1)
def load_upstream_bindings(upstream_root_string):
    """Import exact upstream classes and preprocessing functions."""
    upstream_root = verify_upstream_source(Path(upstream_root_string))
    _import_complete_upstream_package(upstream_root)

    network_module = importlib.import_module("graphgps.network.dgt_model")
    layer_module = importlib.import_module("graphgps.layer.dgt_layer")
    posenc_module = importlib.import_module("graphgps.transform.posenc_stats")
    transform_module = importlib.import_module("graphgps.transform.transforms")
    if dgt_runtime_profile() == RUNTIME_PROFILE_LOCAL_COMPAT:
        from dgt_local_compat import (
            exact_chordless_rings,
            restore_graphgym_activation_semantics,
        )
        restore_graphgym_activation_semantics()
        transform_module.get_rings = exact_chordless_rings
    return {
        "DGTModel": network_module.DGTModel,
        "FeatureEncoder": network_module.FeatureEncoder,
        "DGTLayer": layer_module.DGTLayer,
        "compute_posenc_stats": posenc_module.compute_posenc_stats,
        "compute_shortest_paths": transform_module.compute_shortest_paths,
        "add_rings": transform_module.add_rings,
        "line_graph": transform_module.line_graph,
    }


def configure_upstream_dgt(upstream_root, input_dim, edge_input_dim, hidden_dim,
                           num_layers, num_heads, dropout, attn_dropout,
                           batch_norm, rwse_steps, spd_max_length,
                           rings_max_length):
    """Load the pinned ESOL config, changing only Stage-1 dimensions."""
    global _ACTIVE_DGT_CONFIG_SIGNATURE
    from torch_geometric.graphgym.config import cfg, set_cfg

    requested_signature = {
        "input_dim": int(input_dim),
        "edge_input_dim": int(edge_input_dim),
        "hidden_dim": int(hidden_dim),
        "num_layers": int(num_layers),
        "num_heads": int(num_heads),
        "dropout": float(dropout),
        "attn_dropout": float(attn_dropout),
        "batch_norm": bool(batch_norm),
        "layer_norm": not bool(batch_norm),
        "rwse_steps": tuple(int(step) for step in rwse_steps),
        "spd_max_length": int(spd_max_length),
        "rings_max_length": int(rings_max_length),
        "rings_coalesce_edges": False,
    }
    if _ACTIVE_DGT_CONFIG_SIGNATURE is not None:
        if requested_signature != _ACTIVE_DGT_CONFIG_SIGNATURE:
            raise RuntimeError(
                "GraphGym uses a process-global configuration. Constructing formal DGT "
                "models with different settings in one process is unsafe; start a clean "
                "process for the second configuration."
            )
        if _live_dgt_config_signature(cfg) != requested_signature:
            raise RuntimeError("The process-global GraphGym configuration was externally modified")
        # Upstream FeatureEncoder.__init__ writes cfg.gnn.dim_edge. Temporarily
        # defrost for that exact constructor path; the model freezes it again.
        cfg.defrost()
        return cfg

    cfg.defrost()
    cfg.clear()
    set_cfg(cfg)
    if dgt_runtime_profile() == RUNTIME_PROFILE_LOCAL_COMPAT:
        # PyG 2.6 removed several GraphGym schema keys that existed in the
        # upstream PyG 2.0.4 runtime (for example train.mode).  Load the
        # untouched upstream YAML as the source of those values, then close the
        # schema again. No architecture value is invented here.
        cfg.set_new_allowed(True)
    try:
        cfg.merge_from_file(str(
            Path(upstream_root) / "configs" / "physical_chemistry" /
            "ESOL-RWSE-SPDE-Rings.yaml"
        ))
    finally:
        if dgt_runtime_profile() == RUNTIME_PROFILE_LOCAL_COMPAT:
            cfg.set_new_allowed(False)

    # Dataset-specific dimensions are the only non-architectural overrides.
    cfg.share.dim_in = int(input_dim)
    cfg.dataset.name = "Stage1USPTO"
    cfg.dataset.format = "PyG-MoleculeNet"
    cfg.dataset.edge_encoder_num_types = int(edge_input_dim)
    cfg.dataset.node_encoder_name = "LinearNode"
    cfg.dataset.edge_encoder_name = "LinearEdge+RWSE-SPDE"
    cfg.dataset.edge_encoder_shared = True
    cfg.dataset.spd = True
    cfg.dataset.spd_max_length = int(spd_max_length)
    cfg.dataset.rings = True
    cfg.dataset.rings_max_length = int(rings_max_length)
    cfg.dataset.rings_coalesce_edges = False
    cfg.posenc_RWSE.enable = True
    cfg.posenc_RWSE.enable_edges = True
    cfg.posenc_RWSE.precompute = True
    cfg.posenc_RWSE.kernel.times_func = ""
    cfg.posenc_RWSE.kernel.times = list(int(step) for step in rwse_steps)
    cfg.gnn.dim_inner = int(hidden_dim)
    cfg.gnn.act = "relu"
    cfg.gt.layers = int(num_layers)
    cfg.gt.n_heads = int(num_heads)
    cfg.gt.dim_hidden = int(hidden_dim)
    cfg.gt.dropout = float(dropout)
    cfg.gt.attn_dropout = float(attn_dropout)
    cfg.gt.layer_norm = not bool(batch_norm)
    cfg.gt.batch_norm = bool(batch_norm)
    _ACTIVE_DGT_CONFIG_SIGNATURE = requested_signature
    return cfg


@R.register("models.UpstreamDGTNodeBondTransformer")
class UpstreamDGTNodeBondTransformer(nn.Module, core.Configurable):
    """Direct upstream DGT atom/bond backbone with a TorchDrug boundary."""

    def __init__(self, input_dim, hidden_dims, edge_input_dim, num_heads=32,
                 dropout=0.0, attn_dropout=0.5, batch_norm=True,
                 readout="sum", emit_bond_feature=True, upstream_root=None,
                 rwse_steps=tuple(range(1, 17)), spd_max_length=8,
                 rings_max_length=18, preprocess_cache=None):
        super().__init__()
        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        if not hidden_dims or len(set(int(dim) for dim in hidden_dims)) != 1:
            raise ValueError("Upstream DGT requires non-empty equal hidden dimensions")
        hidden_dim = int(hidden_dims[-1])
        if hidden_dim % int(num_heads):
            raise ValueError("hidden_dim must be divisible by num_heads")
        if readout not in {"sum", "mean"}:
            raise ValueError("readout must be `sum` or `mean`")

        self.upstream_root = verify_upstream_source(
            Path(upstream_root) if upstream_root else default_upstream_root()
        )
        verify_upstream_environment()
        bindings = load_upstream_bindings(str(self.upstream_root))
        self.upstream_cfg = configure_upstream_dgt(
            self.upstream_root,
            input_dim=input_dim,
            edge_input_dim=edge_input_dim,
            hidden_dim=hidden_dim,
            num_layers=len(hidden_dims),
            num_heads=num_heads,
            dropout=dropout,
            attn_dropout=attn_dropout,
            batch_norm=batch_norm,
            rwse_steps=rwse_steps,
            spd_max_length=spd_max_length,
            rings_max_length=rings_max_length,
        )
        # Instantiate the complete upstream model class. Stage-1 stops before
        # its property-prediction head and attaches the reaction-centre head at
        # the adapter boundary; encoder and DGT layers remain the exact objects
        # constructed by upstream DGTModel.__init__.
        try:
            self.backbone = bindings["DGTModel"](dim_in=int(input_dim), dim_out=1)
        finally:
            self.upstream_cfg.freeze()
        # Preserve the complete upstream object for provenance/equivalence, but
        # its property-regression head is outside the Stage-1 task boundary.
        # Freezing it prevents unused parameters from entering optimizer state.
        self.backbone.post_mp.requires_grad_(False)
        self._upstream_preprocess = bindings
        self.input_dim = int(input_dim)
        self.edge_input_dim = int(edge_input_dim)
        self.output_dim = hidden_dim
        self.edge_feature_dim = hidden_dim if emit_bond_feature else 0
        self.graph_feature_dim = 2 * hidden_dim
        self.emit_bond_feature = bool(emit_bond_feature)
        self.readout = readout
        self.preprocess_cache = (
            Path(preprocess_cache).resolve() if preprocess_cache is not None else None
        )
        self._preprocess_signature = {
            "upstream_commit": UPSTREAM_COMMIT,
            "rwse_steps": list(int(step) for step in rwse_steps),
            "spd_max_length": int(spd_max_length),
            "rings_max_length": int(rings_max_length),
            "rings_coalesce_edges": False,
            "input_dim": self.input_dim,
            "edge_input_dim": self.edge_input_dim,
            "torchdrug_edge_policy": "native_or_canonical_to_bidirected_v2",
            "runtime_profile": dgt_runtime_profile(),
        }
        if dgt_runtime_profile() == RUNTIME_PROFILE_LOCAL_COMPAT:
            from dgt_local_compat import RING_BACKEND
            self._preprocess_signature["ring_backend"] = RING_BACKEND
        self._config_signature = dict(_ACTIVE_DGT_CONFIG_SIGNATURE)
        self._cache_keys = []
        self._warned_cache_id_fallback = False

    @staticmethod
    def _tensor_digest(digest, tensor):
        tensor = tensor.detach().cpu().contiguous()
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes())

    def _raw_upstream_data(self, molecule):
        try:
            from torch_geometric.data import Data
        except ImportError as error:
            raise RuntimeError(
                "Faithful DGT requires the upstream PyG / GraphGym dependencies."
            ) from error

        # TorchDrug Molecule.from_smiles stores both directions of every bond,
        # while Stage-1 ranks the u < v view returned by ``directed()``.  DGT's
        # molecular preprocessing also expects both directions, but in the
        # deterministic order [all canonical edges, all reverse edges].  Make
        # that representation explicitly at this adapter boundary.  Accepting
        # the canonical view as well keeps standalone adapter tests useful.
        native_edge_index = molecule.edge_list[:, :2].long().detach().cpu()
        native_edge_attr = molecule.edge_feature.float().detach().cpu()
        if native_edge_index.size(0) != native_edge_attr.size(0):
            raise RuntimeError("Stage-1 bond indices and features have different lengths")
        if native_edge_index.numel() and bool(
            (native_edge_index[:, 0] == native_edge_index[:, 1]).any()
        ):
            raise RuntimeError("Self-loops are not valid chemical bonds at the DGT boundary")

        canonical_mask = native_edge_index[:, 0] < native_edge_index[:, 1]
        reverse_mask = native_edge_index[:, 0] > native_edge_index[:, 1]
        canonical_edges = native_edge_index[canonical_mask]
        canonical_edge_attr = native_edge_attr[canonical_mask]
        reverse_edges = native_edge_index[reverse_mask]
        reverse_edge_attr = native_edge_attr[reverse_mask]
        num_nodes = int(molecule.num_node)

        if canonical_edges.numel():
            canonical_keys = canonical_edges[:, 0] * num_nodes + canonical_edges[:, 1]
            if int(torch.unique(canonical_keys).numel()) != int(canonical_keys.numel()):
                raise RuntimeError("Duplicate canonical product bonds at the DGT boundary")
        else:
            canonical_keys = torch.empty(0, dtype=torch.long)

        if reverse_edges.numel():
            if reverse_edges.size(0) != canonical_edges.size(0):
                raise RuntimeError(
                    "TorchDrug product graph does not contain exactly two directions per bond"
                )
            reverse_keys = reverse_edges[:, 1] * num_nodes + reverse_edges[:, 0]
            canonical_order = torch.argsort(canonical_keys)
            reverse_order = torch.argsort(reverse_keys)
            if not torch.equal(
                canonical_keys[canonical_order], reverse_keys[reverse_order]
            ):
                raise RuntimeError("TorchDrug reverse bonds do not match canonical bonds")
            if molecule.edge_list.size(1) > 2:
                canonical_relation = molecule.edge_list[canonical_mask, 2].detach().cpu()
                reverse_relation = molecule.edge_list[reverse_mask, 2].detach().cpu()
                if not torch.equal(
                    canonical_relation[canonical_order], reverse_relation[reverse_order]
                ):
                    raise RuntimeError("Reverse bond relations differ at the DGT boundary")
            if not torch.equal(
                canonical_edge_attr[canonical_order], reverse_edge_attr[reverse_order]
            ):
                raise RuntimeError("Reverse bond features differ at the DGT boundary")

        canonical_edge_index = canonical_edges.t().contiguous()

        edge_index = torch.cat(
            (canonical_edge_index, canonical_edge_index.flip(0)), dim=1
        )
        edge_attr = torch.cat((canonical_edge_attr, canonical_edge_attr), dim=0)
        return Data(
            x=molecule.node_feature.float().detach().cpu(),
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=int(molecule.num_node),
            stage1_num_bonds=int(canonical_edge_index.size(1)),
            stage1_bidirected_boundary=True,
        )

    def _assert_global_config(self):
        if (
            _ACTIVE_DGT_CONFIG_SIGNATURE != self._config_signature
            or _live_dgt_config_signature(self.upstream_cfg) != self._config_signature
        ):
            raise RuntimeError(
                "The process-global GraphGym DGT configuration changed after model construction."
            )

    def _cache_key(self, data):
        digest = hashlib.sha256()
        digest.update(json.dumps(
            self._preprocess_signature, sort_keys=True, separators=(",", ":")
        ).encode("utf-8"))
        self._tensor_digest(digest, data.x)
        self._tensor_digest(digest, data.edge_index)
        self._tensor_digest(digest, data.edge_attr)
        return digest.hexdigest()

    def _cache_path(self, key):
        if self.preprocess_cache is None:
            raise RuntimeError(
                "Formal upstream DGT runs require a persistent preprocessing cache. "
                "Set --dgt-preprocess-cache and build it before training."
            )
        return self.preprocess_cache / key[:2] / f"{key}.pt"

    def _run_exact_preprocessing(self, data):
        self._assert_global_config()
        try:
            data = self._upstream_preprocess["compute_posenc_stats"](
                data,
                pe_types=["RWSE"],
                is_undirected=True,
                cfg=self.upstream_cfg,
            )
            data = self._upstream_preprocess["compute_shortest_paths"](
                data, config=self.upstream_cfg.dataset
            )
            data = self._upstream_preprocess["add_rings"](
                data, config=self.upstream_cfg.dataset
            )
            data = self._upstream_preprocess["line_graph"](data)
        except (ImportError, ModuleNotFoundError) as error:
            raise RuntimeError(
                "Faithful DGT preprocessing requires every upstream dependency, "
                "including graph-tool==2.45 in the exact profile. The explicit "
                "local_compat profile may use only the exact chordless-cycle "
                "compatibility backend; no approximate DGT encoder is permitted."
            ) from error
        self._validate_preprocessed(data)
        return data

    @staticmethod
    def _validate_preprocessed(data):
        required = (
            "pestat_RWSE", "edge_RWSE", "spd_index", "spd_lengths",
            "ring_index", "e_batch", "e2e_node_index", "e2e_edge_index",
            "e2e_edge_RWSE", "e2e_spd_index", "e2e_spd_lengths",
            "e2e_ring_index",
        )
        missing = [name for name in required if not hasattr(data, name)]
        if missing:
            raise RuntimeError(
                "Incomplete upstream DGT preprocessing; missing attributes: "
                + ", ".join(missing)
            )
        if data.edge_index.dim() != 2 or data.edge_index.size(0) != 2:
            raise RuntimeError("DGT edge_index must have shape [2, num_directed_edges]")
        if not bool(getattr(data, "stage1_bidirected_boundary", False)):
            raise RuntimeError("DGT cache predates the required bidirected Stage-1 boundary")
        num_bonds = int(data.stage1_num_bonds)
        if int(data.edge_index.size(1)) != 2 * num_bonds:
            raise RuntimeError("DGT boundary must contain exactly two directions per bond")
        forward = data.edge_index[:, :num_bonds]
        reverse = data.edge_index[:, num_bonds:]
        if not torch.equal(reverse, forward.flip(0)):
            raise RuntimeError("DGT reverse bond order does not match canonical bond order")
        if data.edge_attr.size(0) != 2 * num_bonds or not torch.equal(
            data.edge_attr[:num_bonds], data.edge_attr[num_bonds:]
        ):
            raise RuntimeError("DGT reverse bonds do not carry identical input features")
        undirected_mask = data.edge_index[0] < data.edge_index[1]
        if int(undirected_mask.sum()) != int(data.e_batch.numel()):
            raise RuntimeError(
                "DGT line-graph node count differs from the number of canonical product bonds"
            )
        if data.edge_RWSE.dim() != 2 or data.e2e_edge_RWSE.dim() != 2:
            raise RuntimeError("DGT RWSE tensors must be flattened dense pair features")
        for name in ("spd_index", "ring_index", "e2e_edge_index", "e2e_spd_index", "e2e_ring_index"):
            index = getattr(data, name)
            if index.dim() != 2 or index.size(0) != 2:
                raise RuntimeError(f"DGT {name} must have shape [2, num_connections]")

    @staticmethod
    def _load_cached_data(path):
        try:
            return torch.load(str(path), map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(str(path), map_location="cpu")

    def _load_preprocessed(self, molecule, allow_create=False):
        self._assert_global_config()
        raw_data = self._raw_upstream_data(molecule)
        key = self._cache_key(raw_data)
        path = self._cache_path(key)
        if path.is_file():
            data = self._load_cached_data(path)
            if getattr(data, "dgt_preprocess_key", None) != key:
                raise RuntimeError(f"DGT preprocessing cache key mismatch: {path}")
            self._validate_preprocessed(data)
            return data
        if not allow_create:
            raise RuntimeError(
                f"Missing exact DGT preprocessing cache entry {key}. "
                "Run train.py --prepare-dgt-cache-only with the same data and DGT settings first."
            )

        data = self._run_exact_preprocessing(raw_data)
        data.dgt_preprocess_key = key
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
        torch.save(data, str(temporary))
        os.replace(temporary, path)
        return data

    def _load_preprocessed_key(self, key):
        self._assert_global_config()
        path = self._cache_path(key)
        if not path.is_file():
            raise RuntimeError(f"Missing exact DGT preprocessing cache entry {key}")
        data = self._load_cached_data(path)
        if getattr(data, "dgt_preprocess_key", None) != key:
            raise RuntimeError(f"DGT preprocessing cache key mismatch: {path}")
        self._validate_preprocessed(data)
        return data

    def precompute_datasets(self, *datasets):
        """Build exact upstream positional/topological features once per product."""
        self._cache_keys = []
        total = sum(len(dataset) for dataset in datasets)
        completed = 0
        for split_index, dataset in enumerate(datasets):
            for sample in dataset:
                graphs = sample["graph"]
                if len(graphs) != 2:
                    raise RuntimeError("Stage-1 reaction sample must contain reactant and product")
                product = graphs[1]
                raw_data = self._raw_upstream_data(product)
                key = self._cache_key(raw_data)
                self._load_preprocessed(product, allow_create=True)
                cache_id = len(self._cache_keys)
                self._cache_keys.append(key)
                with product.graph():
                    product.dgt_cache_id = torch.tensor(cache_id, dtype=torch.long)
                completed += 1
                if completed == 1 or completed % 1000 == 0 or completed == total:
                    print(
                        f"Exact upstream DGT cache: {completed}/{total} "
                        f"(split {split_index + 1}/{len(datasets)})"
                    )

        manifest_path = self.preprocess_cache / "manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = dict(self._preprocess_signature)
        manifest["num_dataset_rows"] = total
        temporary = manifest_path.with_suffix(f".tmp.{os.getpid()}.json")
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, manifest_path)

    def _to_upstream_data_list(self, graph):
        if hasattr(graph, "dgt_cache_id"):
            cache_ids = graph.dgt_cache_id.detach().cpu().long().flatten().tolist()
            if len(cache_ids) != len(graph):
                raise RuntimeError("Packed DGT cache-id count does not match batch size")
            if any(cache_id < 0 or cache_id >= len(self._cache_keys) for cache_id in cache_ids):
                raise RuntimeError("Packed graph contains an invalid DGT cache id")
            return [
                self._load_preprocessed_key(self._cache_keys[cache_id])
                for cache_id in cache_ids
            ]
        if not self._warned_cache_id_fallback:
            print(
                "WARNING: DGT batch has no preassigned cache ids; falling back to "
                "per-molecule unpacking and hashing. Formal training should call "
                "precompute_datasets() before Engine construction."
            )
            self._warned_cache_id_fallback = True
        return [self._load_preprocessed(molecule) for molecule in graph.unpack()]

    @staticmethod
    def _pool(feature, assignment, batch_size, reduction):
        pooled = feature.new_zeros((batch_size, feature.size(-1)))
        if feature.numel():
            pooled.index_add_(0, assignment, feature)
        if reduction == "mean":
            count = torch.bincount(assignment, minlength=batch_size).clamp_min(1)
            pooled = pooled / count.to(feature.dtype).unsqueeze(-1)
        return pooled

    def forward(self, graph, input, all_loss=None, metric=None):
        del input, all_loss, metric
        from torch_geometric.data import Batch

        self._assert_global_config()

        device = graph.device
        upstream_batch = Batch.from_data_list(self._to_upstream_data_list(graph)).to(device)
        upstream_batch = self.backbone.encoder(upstream_batch)
        if hasattr(self.backbone, "pre_mp"):
            upstream_batch = self.backbone.pre_mp(upstream_batch)
        upstream_batch = self.backbone.layers(upstream_batch)

        batch_size = int(upstream_batch.batch.max()) + 1 if upstream_batch.batch.numel() else 1
        atom_graph_feature = self._pool(
            upstream_batch.x, upstream_batch.batch, batch_size, self.readout
        )
        bond_graph_feature = self._pool(
            upstream_batch.e, upstream_batch.e_batch, batch_size, self.readout
        )
        graph_feature = torch.cat((atom_graph_feature, bond_graph_feature), dim=-1)

        output = {
            "node_feature": upstream_batch.x,
            "graph_feature": graph_feature,
            "upstream_commit": UPSTREAM_COMMIT,
        }
        if self.emit_bond_feature:
            undirected_mask = upstream_batch.edge_index[0] < upstream_batch.edge_index[1]
            output["bond_feature"] = upstream_batch.e
            output["bond_index"] = upstream_batch.edge_index[:, undirected_mask].t().contiguous()
            if output["bond_index"].size(0) != output["bond_feature"].size(0):
                raise RuntimeError(
                    "Upstream DGT bond-token order does not match undirected product bonds"
                )
        return output
