"""Frozen class-free identity features for complete-reaction candidate transfer.

No CSV is opened by inference. The caller supplies whitelisted observable rows
and original same-draw candidate decompositions. No fitting, class routing,
alpha selection, true missing sizes, target molecules, or candidate generation.
"""
from collections import defaultdict
import contextlib
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
from functools import lru_cache
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / 'assets/identity.pt'
CONTRACT = ROOT / 'assets/feature_contract.json'
FEATURE_IMPLEMENTATION = ROOT / 'fusion_features.py'
FROZEN_HASHES = {ROOT / name: digest for name, digest in json.loads((ROOT / 'manifest.json').read_text(encoding='utf-8'))['sha256'].items()}
REQUIRED_FIELDS = frozenset(('uuid', 'reaction_id', 'product', 'fragments', 'anchors'))
ALLOWED_FIELDS = REQUIRED_FIELDS | {'center1', 'center2'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def check_stop():
    if (ROOT / 'STOP').exists():
        raise RuntimeError('This transfer experiment is stopped')


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _frozen_module():
    from . import identity_inference
    return identity_inference


@lru_cache(maxsize=1)
def _probability_module():
    if sha(FEATURE_IMPLEMENTATION) != FROZEN_HASHES[FEATURE_IMPLEMENTATION]:
        raise ValueError('Frozen probability-feature implementation differs')
    previous_bytecode = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        return _load('_transfer_probability_features', FEATURE_IMPLEMENTATION)
    finally:
        sys.dont_write_bytecode = previous_bytecode


class FrozenIdentityAdapter:
    """CPU frozen context_descriptors head; it never accepts class/target fields.

    predict_fragments(rows) returns a dict with:
      uuids: tuple[str], probabilities: float32 [N,28], labels: tuple[str],
      probabilities_by_uuid: dict[str, ndarray], index: dict[label,int],
      observed_by_uuid: whitelisted rows, row_info: metadata, provenance: dict.

    Complete companion fragment groups MUST be supplied together. reaction_id
    joins observed partner synthons; its characters and the UUID are not model
    features. anchors are zero-based indices into the given fragment SMILES,
    with optional observed center1/center2 map labels for the archived fallback.
    """
    def __init__(self):
        check_stop()
        self.hashes = {str(path): sha(path) for path in FROZEN_HASHES}
        for path, expected in FROZEN_HASHES.items():
            if self.hashes[str(path)] != expected:
                raise ValueError('Frozen identity input hash differs: ' + str(path))
        self.contract = json.loads(CONTRACT.read_text(encoding='utf-8'))
        self.module = _frozen_module()
        self.feature_module = _probability_module()
        state = torch.load(CHECKPOINT, weights_only=False, map_location='cpu')
        self.labels = tuple(state['labels'])
        if (state.get('arm') != 'context_descriptors' or state['d'] != 3616
                or not state.get('training_source_onlyclass5')
                or list(self.labels) != self.contract['labels']
                or list(self.labels) != list(self.module.LABELS)):
            raise ValueError('Frozen checkpoint/feature contract differs')
        self.index = {name: i for i, name in enumerate(self.labels)}
        with torch.random.fork_rng(devices=[]):
            self.model = self.module.Head(state['d']).cpu()
        self.model.load_state_dict(state['state_dict'], strict=True)
        self.model.requires_grad_(False)
        self.model.eval()
        self.weights = state['weights'].cpu()
        if self.weights.shape != (len(self.labels),) or not torch.isfinite(self.weights).all() or torch.any(self.weights <= 0):
            raise ValueError('Invalid archived training weights')
        self.epoch = int(state['epoch'])

    def predict_fragments(self, rows, batch_size=1024):
        check_stop()
        if batch_size < 1:
            raise ValueError('batch_size must be positive')
        observed = []
        for row in rows:
            keys = set(row)
            if not REQUIRED_FIELDS.issubset(keys) or keys - ALLOWED_FIELDS:
                raise ValueError('Only observable identity fields are allowed; missing=' + str(sorted(REQUIRED_FIELDS - keys)) + '; extra=' + str(sorted(keys - ALLOWED_FIELDS)))
            current = {key: str(row.get(key, '')) for key in sorted(ALLOWED_FIELDS)}
            if any(not current[key].strip() for key in ('uuid', 'reaction_id', 'product', 'fragments')):
                raise ValueError('Empty identifier, product, or observed synthon')
            observed.append(current)
        if not observed:
            raise ValueError('At least one complete reaction group is required')
        uuids = tuple(row['uuid'] for row in observed)
        if len(set(uuids)) != len(uuids):
            raise ValueError('Duplicate fragment UUID')
        group_products = {}
        for row in observed:
            key = self.module.canonical_product(row['product'])
            if group_products.setdefault(row['reaction_id'], key) != key:
                raise ValueError('One reaction_id contains different products')
        legacy_rows = [{
            'uuid': row['uuid'], '_source_row': row['reaction_id'],
            'product': row['product'], 'fragments': row['fragments'],
            'anchors': row['anchors'], 'center1': row['center1'], 'center2': row['center2'],
        } for row in observed]
        with contextlib.redirect_stdout(io.StringIO()):
            features, info = self.module.feature_rows(legacy_rows)
        if features.shape != (len(observed), 3616) or not np.isfinite(features).all():
            raise ValueError('Invalid frozen observable feature array')
        if [str(row['uuid']) for row in info] != list(uuids):
            raise ValueError('Feature UUID order changed')
        if [str(row['source_row']) for row in info] != [row['reaction_id'] for row in observed]:
            raise ValueError('Feature reaction grouping changed')
        predicted = []
        with torch.inference_mode():
            for start in range(0, len(observed), batch_size):
                check_stop()
                logits = self.model(torch.from_numpy(features[start:start + batch_size]))
                predicted.append(torch.softmax(logits - self.weights.log(), dim=-1).numpy())
        probabilities = np.concatenate(predicted)
        if (probabilities.shape != (len(observed), len(self.labels))
                or not np.isfinite(probabilities).all()
                or np.any(probabilities < 0) or np.any(probabilities > 1)
                or not np.allclose(probabilities.sum(1), 1, rtol=0, atol=1e-5)):
            raise ValueError('Invalid identity probability matrix')
        probabilities.setflags(write=False)
        for path, expected in FROZEN_HASHES.items():
            if sha(path) != expected:
                raise ValueError('Frozen identity input changed during inference')
        provenance = {
            'schema': 'class_free_frozen_context_identity_transfer_v1',
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'rows': len(observed), 'reactions': len(group_products),
            'feature_dimensions': 3616, 'labels': list(self.labels),
            'device': 'cpu', 'training_performed': False,
            'class_used_as_model_input': False, 'target_used_as_model_input': False,
            'probability_formula': 'softmax(logits-log(frozen_training_weights))',
            'frozen_epoch': self.epoch, 'frozen_inputs_sha256': self.hashes,
            'feature_array_sha256': hashlib.sha256(features.tobytes(order='C')).hexdigest(),
            'feature_whitelist': sorted(ALLOWED_FIELDS),
            'partner_policy': 'All other provided observed synthons with the same reaction_id; bitwise OR fingerprints, exactly as frozen implementation.',
            'head_training_scope': 'arylation only; remaining reaction families are out-of-domain transfer',
            'candidate_scores_or_labels_read_by_predict_fragments': False,
            'caveats': ['MULTI does not model detailed multi-atom composition or connectivity.',
                        'The caller must provide complete companion groups and audit teacher training/development product overlap.',
                        'No reaction class is inferred or used for routing in this adapter.'],
        }
        reaction_uuids = defaultdict(list)
        for row in observed:
            reaction_uuids[row['reaction_id']].append(row['uuid'])
        return {
            'uuids': uuids, 'probabilities': probabilities, 'labels': self.labels,
            'probabilities_by_uuid': {uid: probabilities[i] for i, uid in enumerate(uuids)},
            'index': dict(self.index), 'observed_by_uuid': {r['uuid']: r for r in observed},
            'reaction_uuids': {rid: tuple(ids) for rid, ids in reaction_uuids.items()},
            'row_info': info, 'provenance': provenance,
        }

    def save_bank(self, bank, output_path):
        """Accept either a .npz path (sibling .manifest.json) or new directory."""
        return save_bank(bank, output_path)


def candidate_features(bank, fragment_uuids, candidate_parts):
    """Exact frozen identity_prior_* features for ONE complete original candidate.

    candidate_parts[i] is the ORIGINAL generated precursor assigned to
    fragment_uuids[i], not obtained by splitting the joined candidate on dots.
    MISSING/INVALID candidates retain the archived floor-probability behavior.
    """
    check_stop()
    ids = [str(uid) for uid in fragment_uuids]
    if not ids or len(ids) != len(candidate_parts) or len(set(ids)) != len(ids):
        raise ValueError('Candidate decomposition and unique fragment IDs must align')
    if any(uid not in bank['observed_by_uuid'] for uid in ids):
        raise ValueError('Unknown candidate fragment UUID')
    reactions = {bank['observed_by_uuid'][uid]['reaction_id'] for uid in ids}
    if len(reactions) != 1:
        raise ValueError('Candidate mixes separate reaction groups')
    reaction_id = next(iter(reactions))
    expected = set(bank['reaction_uuids'][reaction_id])
    if set(ids) != expected:
        raise ValueError('Candidate omitted a companion fragment')
    if any(not isinstance(part, str) for part in candidate_parts):
        raise TypeError('Each original candidate part must be a string')
    # Stable surrogate integers permit UUID strings while reusing the exact
    # original routine, which casts lookup IDs to int. Identifiers never score.
    local_ids = list(range(len(ids)))
    probabilities = {i: bank['probabilities_by_uuid'][uid] for i, uid in enumerate(ids)}
    fragments = [bank['observed_by_uuid'][uid]['fragments'] for uid in ids]
    module = _probability_module()
    return module.probability_features(local_ids, fragments, list(candidate_parts), probabilities, bank['index'])


def vectorize_identity(feature_dicts, frozen_identity_names):
    """Use the existing fusion features.json identity column order, fill zero."""
    names = tuple(frozen_identity_names)
    if len(set(names)) != len(names) or any(not name.startswith('identity_prior_') for name in names):
        raise ValueError('Expected unique frozen identity feature names')
    result = np.asarray([[row.get(name, 0.0) for name in names] for row in feature_dicts], dtype=np.float32).reshape(-1, len(names))
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite candidate identity features')
    return result


def save_bank(bank, output_path):
    """Exclusive NPZ/JSON export; contains observations/probabilities, no truth."""
    check_stop()
    for path, expected in FROZEN_HASHES.items():
        if sha(path) != expected:
            raise ValueError('Frozen identity input changed before export')
    output_path = Path(output_path)
    if output_path.suffix.lower() == '.npz':
        destination = output_path
        manifest_path = output_path.with_suffix('.manifest.json')
        if destination.exists() or manifest_path.exists():
            raise FileExistsError('Refusing to overwrite identity export')
        destination.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = output_path
        if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
            raise FileExistsError('Refusing to overwrite identity export')
        output_dir.mkdir(parents=True, exist_ok=True)
        destination = output_dir / 'identity_probabilities.npz'
        manifest_path = output_dir / 'manifest.json'
    np.savez_compressed(destination, uuids=np.asarray(bank['uuids']),
                        reaction_ids=np.asarray([bank['observed_by_uuid'][u]['reaction_id'] for u in bank['uuids']]),
                        product_keys=np.asarray([r['product_key'] for r in bank['row_info']]),
                        probabilities=bank['probabilities'], labels=np.asarray(bank['labels']))
    manifest = dict(bank['provenance'])
    manifest.update(probability_output=str(destination.resolve()), probability_output_sha256=sha(destination),
                    adapter_path=str(Path(__file__).resolve()), adapter_sha256=sha(__file__))
    with manifest_path.open('x', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, allow_nan=False)
    return manifest
