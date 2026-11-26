"""Scoped Linux adapters; frozen algorithms are copied with recorded provenance."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
BUNDLE = Path(os.environ.get('RETRO_STAGE2_BUNDLE_ROOT', PACKAGE.parent)).resolve()
ROOT = BUNDLE / 'runner'
OLD = Path(os.environ.get('RETRO_STAGE2_OLD_ROOT', BUNDLE / 'old')).resolve()
CYCLE = Path(os.environ.get('RETRO_STAGE2_CYCLE_ROOT', BUNDLE / 'cycle')).resolve()
WORK = Path(os.environ.get('RETRO_STAGE2_WORK_ROOT', BUNDLE / 'work')).resolve()
DATA = WORK / 'experiments/mechanism_prior_fspec_edm_20260919'
ARMS = ['control', 'compatibility', 'graph', 'angle', 'steric', 'energy']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf8')
    os.replace(tmp, path)


def configure(bundle_root, run_root):
    global BUNDLE, ROOT, OLD, CYCLE, WORK, DATA
    BUNDLE = Path(bundle_root).resolve()
    ROOT = Path(run_root).resolve()
    OLD = Path(os.environ.get('RETRO_STAGE2_OLD_ROOT', BUNDLE / 'old')).resolve()
    CYCLE = Path(os.environ.get('RETRO_STAGE2_CYCLE_ROOT', BUNDLE / 'cycle')).resolve()
    WORK = Path(os.environ.get('RETRO_STAGE2_WORK_ROOT', BUNDLE / 'work')).resolve()
    DATA = WORK / 'experiments/mechanism_prior_fspec_edm_20260919'
    os.environ['RETRO_STAGE2_BUNDLE_ROOT'] = str(BUNDLE)
    os.environ['RETRO_STAGE2_OLD_ROOT'] = str(OLD)
    os.environ['RETRO_STAGE2_CYCLE_ROOT'] = str(CYCLE)
    for path in [BUNDLE, WORK, OLD, OLD / 'stage2', OLD / 'vendor/stage2']:
        sys.path.insert(0, str(path))


def seed(value):
    import torch
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def setup():
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
