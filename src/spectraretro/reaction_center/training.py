"""Shared data, training and evaluation routines for the reaction-center models."""
import csv
import hashlib
import json
import os
from pathlib import Path
import random

from .features import REPOSITORY, ROOT, VENDOR, SCHEMA, attach_product_prior
import numpy as np
import torch
from rdkit import rdBase
from torchdrug import data, datasets, utils
from torchdrug.layers import functional


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf8')
    os.replace(temporary, path)


def source_manifest(mode_dir):
    files = sorted(VENDOR.rglob('*.py')) + sorted(mode_dir.glob('*.py'))
    files.extend([ROOT / 'features.py', ROOT / 'training.py'])
    runtime = ROOT / 'orchestration' / 'local_runtime.py'
    if runtime.exists():
        files.append(runtime)
    return {p.relative_to(REPOSITORY).as_posix(): sha(p) for p in files}


def torch_load(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def load_split(path, sources):
    path = Path(path)
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    index_path = path.with_name(path.stem + '_index.csv')
    if index_path.exists():
        with index_path.open(encoding='utf-8-sig', newline='') as stream:
            index_rows = list(csv.DictReader(stream))
        if len(index_rows) != len(rows):
            raise ValueError('Index rows do not align with reaction CSV')
    else:
        index_rows = [{'sample_id': f'{path.stem}:{i:06d}', 'similarity_group': 'unassigned'}
                      for i in range(len(rows))]
    identity = dict(csv_sha256=sha(path), sources=sources, schema=SCHEMA,
                    rdkit=rdBase.rdkitVersion, torch=torch.__version__, numpy=np.__version__,
                    atom_feature='center_identification', bond_feature='default',
                    with_hydrogen=False, kekulize=True, official_atom_vocab=True)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache = Path(os.environ.get('SPECTRARETRO_STAGE1_CACHE',
                               str(REPOSITORY / 'outputs' / 'center_cache')))
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / (key + '.pth')
    if target.exists():
        saved = torch_load(target)
        if saved['identity'] != identity:
            raise ValueError('Cache identity mismatch')
        dataset, report = saved['dataset'], saved['report']
    else:
        dataset = datasets.USPTO50k(str(path), atom_feature='center_identification', bond_feature='default',
                                    as_synthon=False, with_hydrogen=False, kekulize=True, verbose=0)
        if len(dataset) != len(rows) or dataset.targets['sample id'] != list(range(len(rows))):
            raise RuntimeError('Stage1 parser dropped/expanded/reordered frozen rows')
        failures = []
        for i, (_, product) in enumerate(dataset.data):
            reason = attach_product_prior(product)
            if reason:
                failures.append({'row': i, 'reason': reason})
        report = dict(rows=len(rows), descriptor_available=len(rows) - len(failures), failures=failures)
        temporary = target.with_suffix('.tmp')
        torch.save(dict(identity=identity, dataset=dataset, report=report), temporary)
        os.replace(temporary, target)
    if len(dataset) != len(rows) or dataset.targets['sample id'] != list(range(len(rows))):
        raise RuntimeError('Cached dataset row identity mismatch')
    if dataset.smiles_list != [r['rxn_smiles'] for r in rows]:
        raise RuntimeError('Cached reaction strings mismatch')
    if dataset.targets['reaction'] != [int(r['class']) - 1 for r in rows]:
        raise RuntimeError('Cached class mapping mismatch')
    for i, row in enumerate(rows):
        product = dataset.data[i][1]
        if int(product.edge_label.sum() + product.node_label.sum()) != 1:
            raise RuntimeError('Stage1 requires exactly one center per frozen row')
        meta = index_rows[i]
        if 'class' in meta and int(meta['class']) != int(row['class']):
            raise RuntimeError('Index class mismatch')
    return dataset, rows, index_rows, report


def common_state_hash(task):
    result = hashlib.sha256()
    for name, value in sorted(task.state_dict().items()):
        if name.startswith(('edge_prior.', 'node_prior.')):
            continue
        result.update(name.encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def move(batch, device):
    return utils.cuda(batch, device=device) if device.type == 'cuda' else batch


def metric_records(records):
    def summarize(values):
        if not values:
            return {'n': 0}
        return {'n': len(values), 'cross_entropy': float(np.mean([v['cross_entropy'] for v in values])),
                **{f'top-{k} accuracy': float(np.mean([v['rank'] <= k for v in values])) for k in (1, 2, 3, 5, 10)}}
    return dict(overall=summarize(records), by_similarity={
        group: summarize([r for r in records if r['similarity_group'] == group])
        for group in sorted({r['similarity_group'] for r in records})})


@torch.no_grad()
def evaluate(task, dataset, rows, index_rows, batch_size, device, save_predictions=None):
    was_training = task.training
    torch_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    generator = torch.Generator().manual_seed(0)
    records = []
    try:
        task.eval()
        for batch in data.DataLoader(dataset, batch_size, shuffle=False, num_workers=0):
            batch = move(batch, device)
            pred = task.predict(batch)
            labels, sizes = task.target(batch)
            if not torch.isfinite(pred).all():
                raise FloatingPointError('Nonfinite evaluation logits')
            # Corrected local sorting. Count strict tie-broken order, not an
            # offset-based float32 sort across unrelated graphs.
            starts = sizes.cumsum(0) - sizes
            for j, (start, count) in enumerate(zip(starts.tolist(), sizes.tolist())):
                row_id = int(batch['sample id'][j])
                logits = pred[start:start + count].cpu()
                target = int(labels[start:start + count].argmax())
                order = functional.variadic_topk(logits, torch.tensor([count]), count)[1][0]
                rank = int((order == target).nonzero()[0]) + 1
                logp = logits.log_softmax(0)
                sampled = torch.multinomial(logp.exp(), 300, replacement=True, generator=generator)
                metadata = index_rows[row_id]
                record = dict(row_index=row_id, sample_id=metadata['sample_id'],
                              product_group=metadata.get('product_group', ''),
                              product=rows[row_id]['rxn_smiles'].split('>')[-1],
                              reaction_class=int(rows[row_id]['class']),
                              similarity_group=metadata.get('similarity_group', 'unassigned'),
                              candidate_count=count, gold_candidate=target, rank=rank,
                              cross_entropy=-float(logp[target]), correct_center_draws=int((sampled == target).sum()))
                if save_predictions is not None:
                    record.update(logits=logits.tolist(), probabilities=logp.exp().tolist(),
                                  sampled_candidate_indices=sampled.tolist())
                records.append(record)
        if [r['row_index'] for r in records] != list(range(len(dataset))):
            raise RuntimeError('Evaluation row order mismatch')
        if save_predictions is not None:
            with Path(save_predictions).open('w', encoding='utf8') as stream:
                for record in records:
                    stream.write(json.dumps(record) + '\n')
        return metric_records(records)
    finally:
        task.train(was_training)
        torch.set_rng_state(torch_rng)
        if cuda_rng:
            torch.cuda.set_rng_state_all(cuda_rng)


def checkpoint(task, optimizer, epoch, best, best_epoch, stale, contract):
    return dict(model=task.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch,
                best=best, best_epoch=best_epoch, stale=stale, contract=contract,
                torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                python_rng=random.getstate(), numpy_rng=np.random.get_state())


def save_checkpoint(path, state):
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    os.replace(temporary, path)
