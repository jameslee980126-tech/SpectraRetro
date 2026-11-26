"""Direct deletion ablation using the archived NoSpectralFSpec implementation."""
from pathlib import Path
import json
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from training_core import seed_all, digest_state, evaluate
from runtime import ROOT, verify, sha256, save_json
from historical_models_ablation import NoSpectralFSpec
from torchdrug import core, tasks
import torch

def build_engine(train, valid, cpu=False):
    model = NoSpectralFSpec(train.node_feature_dim, [512] * 3, train.edge_feature_dim)
    task = tasks.CenterIdentification(model, None, feature=('graph', 'atom', 'bond'),
                                      mlp_hidden_dim=512, fusion_mode='raw')
    optimizer = torch.optim.Adam(task.parameters(), lr=1e-4, weight_decay=0.)
    return core.Engine(task, train, valid, [], optimizer, gpus=None if cpu else [0],
                       batch_size=64, num_worker=0, log_interval=100)

def verify_extension():
    verify()
    manifest = json.loads((HERE / 'MANIFEST.json').read_text())
    for name, digest in manifest['files'].items():
        assert sha256(HERE / name) == digest, name
    return sha256(HERE / 'MANIFEST.json')
