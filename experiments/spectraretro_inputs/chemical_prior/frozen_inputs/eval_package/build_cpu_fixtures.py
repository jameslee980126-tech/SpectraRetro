"""Local read-only fixtures: frozen ranking replay and complete XYZ reference bank."""
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

BUNDLE = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get('RETRO_STAGE2_WORK_ROOT', BUNDLE / 'work')).resolve()
CYCLE = Path(os.environ.get('RETRO_STAGE2_CYCLE_ROOT', BUNDLE / 'cycle')).resolve()
DATA = WORK / 'experiments/mechanism_prior_fspec_edm_20260919'
OUT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def main():
    for name in ['ranking_replay_fixture.json', 'reference_bank.npz', 'fixture_manifest.json']:
        if (OUT / name).exists():
            raise FileExistsError(name)
    sys.path.insert(0, str(Path(__file__).resolve().parents[5] / 'src/spectraretro'))
    sys.path.insert(0, str(WORK))
    import torch
    from rdkit import RDLogger
    from fusion_reranker.runtime import FrozenFusionRanker, candidate_records
    from fusion_reranker.identity_adapter import FrozenIdentityAdapter
    torch.set_num_threads(2)
    RDLogger.DisableLog('rdApp.*')
    meta_path = DATA / 'data/metadata/eval_record_map.csv'
    table_path = DATA / 'data/stage2/similarity_eval_table.csv'
    decode_path = CYCLE / 'evaluation/control/decoded/result.csv'
    decoded = {r['uuid']: r for r in read_csv(decode_path)}
    table = {r['uuid']: r for r in read_csv(table_path)}
    grouped = defaultdict(list)
    for row in read_csv(meta_path):
        grouped[row['reaction_id']].append(row)
    chosen = []
    for cls in range(1, 7):
        keys = [key for key, group in grouped.items() if int(group[0]['class_metadata_only']) == cls]
        keys.sort(key=lambda key: hashlib.sha256(('server_cpu_replay_v1|' + key).encode()).hexdigest())
        chosen.extend(keys[:4])
    requests = []
    observations = []
    for key in chosen:
        group = sorted(grouped[key], key=lambda row: int(row['uuid']))
        products = {table[row['uuid']]['product'] for row in group}
        assert len(products) == 1
        request = dict(reaction_id=key, product=products.pop(), fragments=[])
        for row in group:
            uid = row['uuid']
            t = table[uid]
            fragment = {k: t[k] for k in ['fragments', 'anchors', 'center1', 'center2'] if k in t}
            fragment.update(uuid=uid, draws=[decoded[uid][str(i + 1)] for i in range(50)])
            request['fragments'].append(fragment)
            observations.append({k: fragment[k] for k in ['uuid', 'fragments', 'anchors', 'center1', 'center2'] if k in fragment} |
                                dict(reaction_id=key, product=request['product']))
        requests.append(request)
    ranker = FrozenFusionRanker('unknown', 'fusion')
    results = ranker.rank_batch(requests)
    identity = FrozenIdentityAdapter().predict_fragments(observations)
    fixture = dict(requests=requests,
                   expected=[dict(reaction_id=r['reaction_id'], candidates=[dict(candidate=c['candidate'], score=c['fusion_score']) for c in r['candidates']]) for r in results],
                   candidate_features=[[c['features'] for c in candidate_records(request)] for request in requests],
                   identity_uuids=list(identity['uuids']), identity_probabilities=identity['probabilities'].tolist(),
                   selected_reactions=len(chosen), selection='four hash-fixed existing validation reactions per metadata class; no class input to ranker',
                   not_accuracy_evaluation=True)
    (OUT / 'ranking_replay_fixture.json').write_text(json.dumps(fixture, ensure_ascii=False, allow_nan=False), encoding='utf8')
    uuids, positions, types, offsets, sources = [], [], [], [0], {}
    for path in sorted((CYCLE / 'evaluation/control/raw').glob('batch_*.npz')):
        sources[str(path)] = sha(path)
        with np.load(path, allow_pickle=False) as saved:
            ref = saved['true_offsets']
            for i, uid in enumerate(saved['uuids'].astype(str)):
                a, b = ref[i:i + 2]
                xyz = saved['true_positions'][a:b].copy()
                elements = saved['true_atom_types'][a:b].copy()
                assert len(xyz) == len(elements) and np.isfinite(xyz).all()
                uuids.append(uid)
                positions.append(xyz)
                types.append(elements)
                offsets.append(offsets[-1] + len(xyz))
    assert len(uuids) == len(set(uuids)) == 4161 and set(uuids) == set(decoded)
    np.savez_compressed(OUT / 'reference_bank.npz', uuids=np.asarray(uuids), positions=np.concatenate(positions),
                        atom_types=np.concatenate(types), offsets=np.asarray(offsets, dtype=np.int64),
                        windows_decoded=np.asarray([decoded[uid]['51'] for uid in uuids]))
    manifest = dict(status='passed', no_training=True, no_new_sampling=True, all_reference_rows=4161,
                    ranking_replay_reactions=24, ranking_replay_candidates=sum(len(r['candidates']) for r in results),
                    fusion_manifest_sha256=sha(WORK / 'fusion_reranker/manifest.json'),
                    sources={str(meta_path): sha(meta_path), str(table_path): sha(table_path), str(decode_path): sha(decode_path)},
                    reference_archive_hashes=sources,
                    artifacts={name: sha(OUT / name) for name in ['ranking_replay_fixture.json', 'reference_bank.npz']})
    (OUT / 'fixture_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf8')
    print(json.dumps({k: manifest[k] for k in ['status', 'all_reference_rows', 'ranking_replay_reactions', 'ranking_replay_candidates']}), flush=True)


if __name__ == '__main__':
    main()
