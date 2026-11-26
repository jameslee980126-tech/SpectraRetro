"""Ordinary product RGCN: same factory as the completed small-data baseline."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training_core import seed_all, digest_state, evaluate
from runtime import ROOT, verify, sha256, save_json
from torchdrug import core, models, tasks
import torch

def build_engine(train, valid, cpu=False):
    model = models.RGCN(input_dim=train.node_feature_dim, hidden_dims=[512]*3,
        batch_norm=False, short_cut=True,
        num_relation=int(train[0]['graph'][1].num_relation), concat_hidden=False)
    task = tasks.CenterIdentification(model, None, feature=('graph','atom','bond'),
        mlp_hidden_dim=512, fusion_mode='raw')
    optimizer = torch.optim.Adam(task.parameters(), lr=1e-4, weight_decay=0.)
    return core.Engine(task, train, valid, [], optimizer, gpus=None if cpu else [0],
        batch_size=64, num_worker=0, log_interval=100)

def verify_extension():
    verify()
    folder = Path(__file__).resolve().parent
    import json
    manifest = json.loads((folder/'MANIFEST.json').read_text())
    for name, digest in manifest['files'].items():
        assert sha256(folder/name) == digest, name
    return sha256(folder/'MANIFEST.json')
