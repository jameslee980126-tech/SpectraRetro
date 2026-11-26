"""Instantiate the archived native DiffER modules without altering their equations."""
import os
import hashlib
import json
import random
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

RELEASE = Path(__file__).resolve().parents[4]
SOURCE_ROOT = Path(__file__).resolve().parents[1] / 'source_full'
if 'SPECTRARETRO_DIFFER_WORK' not in os.environ:
    raise RuntimeError('Use scripts/differ.py with an explicit new research workspace')
ROOT = Path(os.environ['SPECTRARETRO_DIFFER_WORK']).resolve()
if not (RELEASE / 'outputs') in ROOT.parents:
    raise ValueError('DiffER workspaces must be below the release outputs/ directory')
CONFIG = Path(os.environ.get('SPECTRARETRO_DIFFER_CONFIG', ROOT / 'protocol.json')).resolve()
sys.path.insert(0, str(SOURCE_ROOT))
from source.data import RSmilesUspto50
from source.tokeniser import load_tokeniser_from_rsmiles
from source.conditional_model import ConditionalModel
from source.discrete_diffusion import UnifiedDiscreteDiffusion
from source.trainer import UnifiedTrainer
from source.sampler import UnifiedSampler

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def state_hash(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode()); h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def build(mode, directory):
    torch.set_num_threads(2)
    seed_all(0)
    assert torch.cuda.is_available()
    protocol = json.loads(CONFIG.read_text(encoding='utf8'))
    assert protocol['native_candidate_budget'] == 50
    assert protocol['test_augmentation'] * protocol['samples_per_test_augmentation'] == 50
    assert protocol['test'] == 9250 and protocol['valid'] == 9395 and protocol['train'] == 74146
    data = ROOT / 'shared' / mode
    tokeniser = load_tokeniser_from_rsmiles(data)
    if mode == 'known':
        assert all(f'[RXN_{k}]' in tokeniser.vocab for k in range(1, 7))
        ids = tokeniser.convert_tokens_to_ids(tokeniser.tokenise(['[RXN_1]C'])['original_tokens'])[0]
        assert ids[1] == tokeniser.vocab['[RXN_1]'] != tokeniser.unk_id
    else:
        assert not any(t.startswith('[RXN_') for t in tokeniser.vocab)
    (directory / 'vocabulary.json').write_text(json.dumps(tokeniser.vocab, indent=2) + '\n')
    conf = dict(protocol['model_config'])
    kwargs = {k: conf[k] for k in ['max_seq_len', 'd_model', 'num_layers', 'num_heads', 'd_feedforward', 'activation', 'dropout']}
    model = ConditionalModel(tokeniser=tokeniser, **kwargs).cuda()
    initial = state_hash(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=protocol['learning_rate'], weight_decay=protocol['weight_decay'])
    diffuser = UnifiedDiscreteDiffusion(num_steps=conf['num_timesteps'], num_classes=len(tokeniser),
               noise_schedule_type=conf['noise_schedule'], noise_schedule_args=conf['noise_schedule_args'])
    sampler = UnifiedSampler(diffuser, tokeniser, conf['num_timesteps'], conf['max_seq_len'], min_time=0.01, pad_limit=protocol['pad_limit'])
    original_sample = sampler.sample
    def quiet_sample(*a, **kw):
        kw['verbose'] = False
        return original_sample(*a, **kw)
    sampler.sample = quiet_sample
    trainer = UnifiedTrainer(model, optimizer, diffuser, sampler, name=mode,
               length_loss=conf['length_loss'], coeff_ce=conf['coeff_ce'], coeff_vlb=conf['coeff_vlb'], use_gpu=True)
    return protocol, tokeniser, model, optimizer, sampler, trainer, initial

def dataset(tokeniser, mode, split, canonical_only=False):
    ds = RSmilesUspto50(tokeniser, ROOT / 'shared' / mode, split, forward=False,
                       pad_limit=20, max_seq_len=256, selfies=False)
    if canonical_only:
        # Preprocessing emits exactly twenty adjacent augmentations per raw row.
        ds.products = ds.products[::20]; ds.reactants = ds.reactants[::20]
    return ds

def loader(ds, batch=64, shuffle=False):
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=0, collate_fn=ds.collate_fn)

def rng_state():
    return {'python':random.getstate(), 'numpy':np.random.get_state(), 'torch':torch.get_rng_state(), 'cuda':torch.cuda.get_rng_state_all()}

def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch']); torch.cuda.set_rng_state_all(state['cuda'])
