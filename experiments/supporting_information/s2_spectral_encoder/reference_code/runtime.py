"""Read-only identity guards shared by both independent experiment scopes."""
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def verify():
    manifest = json.loads((ROOT / 'PAIR_MANIFEST.json').read_text())
    for name, digest in manifest['files'].items():
        if sha256(ROOT / name) != digest:
            raise RuntimeError('Frozen input changed: ' + name)
    code = json.loads((ROOT / 'EXECUTION_MANIFEST.json').read_text())
    for name, digest in code.items():
        if sha256(ROOT / name) != digest:
            raise RuntimeError('Execution changed: ' + name)
    return manifest


def activate():
    snapshot = json.loads((ROOT / 'snapshot_manifest.json').read_text())
    for name, digest in snapshot['files'].items():
        if sha256(ROOT / 'source_snapshot' / name) != digest:
            raise RuntimeError('Snapshot changed: ' + name)
    os.environ.setdefault('PHYSDUAL_ATOM_VOCAB', 'official')
    sys.path.insert(0, str(ROOT / 'source_snapshot'))
    sys.dont_write_bytecode = True
    # TorchDrug patches nn.Module; torchvision must be imported first on the
    # server environment. Its absence is allowed for older CPU-only checks.
    try:
        import torchvision
    except ModuleNotFoundError as error:
        if error.name != 'torchvision':
            raise


def load_split(scope, split):
    from train import load_stage1_dataset
    manifest = json.loads((ROOT / 'PAIR_MANIFEST.json').read_text())
    folder = ROOT / ('subset/data' if scope == 'subset' else 'data')
    path = folder / (split + '.csv')
    if sha256(path) != manifest['files'][path.relative_to(ROOT).as_posix()]:
        raise RuntimeError('Split hash changed')
    dataset = load_stage1_dataset(path)
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    count = manifest['counts'][scope][split]
    if len(rows) != count or len(dataset) != count:
        raise RuntimeError('Parser changed the evaluation denominator')
    if dataset.targets['sample id'] != list(range(count)):
        raise RuntimeError('Parser reordered or expanded rows')
    if dataset.smiles_list != [row['rxn_smiles'] for row in rows]:
        raise RuntimeError('Source reaction identity differs')
    mapping = json.loads((folder / (split + '_mapping.json')).read_text()) if scope == 'subset' else None
    return dataset, mapping
