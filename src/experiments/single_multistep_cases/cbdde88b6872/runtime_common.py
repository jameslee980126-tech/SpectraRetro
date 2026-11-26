"""Runtime adapters around unchanged original EDM; no scientific model edits."""
import hashlib
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch

BASE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(BASE / 'baseline_snapshot'))
from src.lightning import DDPM
from src.edm import EDM
from src.datasets import collate, collate_with_fragment_edges
from src.linker_size_lightning import SizeClassifier
from src import const


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temp.replace(path)


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def setup(threads=4):
    torch.set_num_threads(threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def records(dims, split):
    return torch.load(BASE / f'data_{dims}d' / f'geometry20k_{split}.pt', map_location='cpu', weights_only=False)


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def create(dims, device, size=False):
    if size:
        return SizeClassifier(data_path=str(BASE / 'data_3d'), train_data_prefix='geometry20k_train', val_data_prefix='geometry20k_eval',
                              in_node_nf=26, hidden_nf=128, out_node_nf=len(const.ZINC_TRAIN_LINKER_ID2SIZE), n_layers=3,
                              batch_size=32, lr=1e-4, torch_device=str(device), normalization='batch_norm').to(device)
    cfg = json.loads((BASE / 'configs' / f'edm_{dims}d.json').read_text())
    cfg.update(data_path=str(BASE / f'data_{dims}d'), torch_device=str(device))
    model = DDPM(**cfg).to(device)
    assert type(model.edm) is EDM
    return model


def provenance():
    protocol = json.loads((BASE / 'method_protocol.json').read_text())
    for name, expected in protocol['snapshot_sha256'].items():
        assert sha(BASE / 'baseline_snapshot' / name) == expected, name
    pair = json.loads((BASE / 'paired_data_manifest.json').read_text())
    for name, expected in pair['sha256'].items():
        assert sha(BASE / name) == expected, name
    return {'method_protocol_sha256': sha(BASE / 'method_protocol.json'), 'paired_data_sha256': sha(BASE / 'paired_data_manifest.json')}
