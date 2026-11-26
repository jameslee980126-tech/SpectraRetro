"""Read-only diagnosis of cross-platform frozen-fusion replay discrepancies.

This diagnostic neither changes frozen assets/fixtures nor relaxes their checks.
Scores are aligned by original candidate string before differences are measured.
"""
import argparse
from collections import Counter
import json
import platform
import sys
from pathlib import Path
import numpy as np
import portable_common as common


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle-root', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    common.configure(args.bundle_root, args.run_root)
    package = Path(__file__).resolve().parent
    fixture_path = package / 'ranking_replay_fixture.json'
    manifest = json.loads((package / 'fixture_manifest.json').read_text(encoding='utf8'))
    if common.digest(fixture_path) != manifest['artifacts'][fixture_path.name]:
        raise ValueError('Original ranking fixture changed; no diagnostic substitution')
    fixture = json.loads(fixture_path.read_text(encoding='utf8'))
    import torch
    import sklearn
    import rdkit
    from rdkit import RDLogger
    from fusion_reranker.runtime import FrozenFusionRanker, candidate_records
    from fusion_reranker.identity_adapter import FrozenIdentityAdapter, candidate_features
    torch.set_num_threads(2)
    RDLogger.DisableLog('rdApp.*')
    ranker = FrozenFusionRanker('unknown', 'fusion')
    results = ranker.rank_batch(fixture['requests'])
    observations = []
    for request in fixture['requests']:
        for fragment in request['fragments']:
            observations.append({key: fragment[key] for key in ['uuid', 'fragments', 'anchors', 'center1', 'center2'] if key in fragment} |
                                dict(reaction_id=request['reaction_id'], product=request['product']))
    bank = FrozenIdentityAdapter().predict_fragments(observations)
    identity_actual = bank['probabilities']
    identity_expected = np.asarray(fixture['identity_probabilities'], dtype=identity_actual.dtype)
    if list(bank['uuids']) != fixture['identity_uuids'] or identity_actual.shape != identity_expected.shape:
        raise ValueError('Identity UUID/schema changed; cannot align probabilities')
    identity_delta = identity_actual.astype(np.float64) - identity_expected.astype(np.float64)
    identity_largest = np.unravel_index(np.argmax(np.abs(identity_delta)), identity_delta.shape)
    expected_bank = dict(bank)
    expected_bank['probabilities_by_uuid'] = {uid: identity_expected[i] for i, uid in enumerate(bank['uuids'])}
    features_max, identity_features_max, score_max = 0., 0., 0.
    chemical_count, identity_feature_count, score_count = 0, 0, 0
    differences, candidate_rows, changed, near_ties, pools = [], [], [], [], []
    for request, actual, expected, frozen_features in zip(fixture['requests'], results, fixture['expected'], fixture['candidate_features']):
        reaction_id = request['reaction_id']
        actual_rows = actual['candidates']
        expected_rows = expected['candidates']
        a_order = [r['candidate'] for r in actual_rows]
        e_order = [r['candidate'] for r in expected_rows]
        amap = {r['candidate']: (i + 1, r['fusion_score']) for i, r in enumerate(actual_rows)}
        emap = {r['candidate']: (i + 1, r['score']) for i, r in enumerate(expected_rows)}
        missing, extra = sorted(set(emap) - set(amap)), sorted(set(amap) - set(emap))
        if missing or extra:
            pools.append(dict(reaction_id=reaction_id, missing_candidates=missing, extra_candidates=extra))
        if a_order != e_order:
            changes = []
            for candidate in sorted(set(amap) & set(emap), key=lambda s: emap[s][0]):
                arank, ascore = amap[candidate]
                erank, escore = emap[candidate]
                if arank != erank:
                    changes.append(dict(candidate=candidate, expected_rank=erank, actual_rank=arank,
                                        expected_score=escore, actual_score=ascore, score_delta=ascore-escore))
            changed.append(dict(reaction_id=reaction_id, changed_candidates=len(changes),
                                expected_order=e_order, actual_order=a_order, changes=changes))
        for candidate in set(amap) & set(emap):
            arank, ascore = amap[candidate]
            erank, escore = emap[candidate]
            delta = float(ascore-escore)
            score_max = max(score_max, abs(delta))
            score_count += int(delta != 0)
            candidate_rows.append(dict(reaction_id=reaction_id, candidate=candidate, actual_score=ascore,
                                       expected_score=escore, score_delta=delta, actual_rank=arank, expected_rank=erank))
        # Both expected feature list and candidate_records use stable frequency order.
        raw = candidate_records(request)
        if len(raw) != len(frozen_features):
            pools.append(dict(reaction_id=reaction_id, feature_candidate_count_mismatch=True))
            continue
        ids = [fragment['uuid'] for fragment in request['fragments']]
        for c, expected_feature in zip(raw, frozen_features):
            feature = c['features']
            keys = set(feature) | set(expected_feature)
            for key in keys:
                av, ev = feature.get(key, 0.), expected_feature.get(key, 0.)
                delta = float(av-ev)
                features_max = max(features_max, abs(delta))
                chemical_count += int(delta != 0)
                if delta != 0:
                    differences.append(dict(kind='chemical', reaction_id=reaction_id, candidate=c['candidate'],
                                            feature=key, expected=ev, actual=av, delta=delta,
                                            key_present_expected=key in expected_feature, key_present_actual=key in feature))
            af = candidate_features(bank, ids, c['parts'])
            ef = candidate_features(expected_bank, ids, c['parts'])
            for key in set(af) | set(ef):
                av, ev = af.get(key, 0.), ef.get(key, 0.)
                delta = float(av-ev)
                identity_features_max = max(identity_features_max, abs(delta))
                identity_feature_count += int(delta != 0)
                if delta != 0:
                    differences.append(dict(kind='identity', reaction_id=reaction_id, candidate=c['candidate'],
                                            feature=key, expected=ev, actual=av, delta=delta))
        # Record reversed adjacent pairs and how close the expected scores were.
        for left, right in zip(expected_rows, expected_rows[1:]):
            a, b = left['candidate'], right['candidate']
            if a in amap and b in amap and amap[a][0] > amap[b][0]:
                near_ties.append(dict(reaction_id=reaction_id, first_candidate=a, second_candidate=b,
                                      expected_score_gap=left['score']-right['score'],
                                      actual_score_gap=amap[a][1]-amap[b][1],
                                      expected_exact_tie=left['score']==right['score'],
                                      actual_exact_tie=amap[a][1]==amap[b][1]))
    differences.sort(key=lambda row: (-abs(row['delta']), row['reaction_id'], row['feature']))
    score_largest = sorted(candidate_rows, key=lambda row: -abs(row['score_delta']))[:12]
    probability_examples = []
    indices = np.argsort(-np.abs(identity_delta).reshape(-1), kind='stable')[:12]
    for idx in indices:
        i, j = np.unravel_index(int(idx), identity_delta.shape)
        if identity_delta[i, j] != 0:
            probability_examples.append(dict(uuid=bank['uuids'][i], label=bank['labels'][j],
                                             expected=float(identity_expected[i, j]), actual=float(identity_actual[i, j]),
                                             delta=float(identity_delta[i, j])))
    report = dict(
        status='diagnostic_only_frozen_equivalence_not_approved', reactions=len(results), candidates=len(candidate_rows),
        changed_order_reactions=len(changed), changed_score_candidates=score_count,
        max_score_abs_error=score_max, max_chemical_feature_abs_error=features_max,
        chemical_feature_nonzero_differences=chemical_count, max_identity_candidate_feature_abs_error=identity_features_max,
        identity_candidate_feature_nonzero_differences=identity_feature_count,
        max_identity_probability_abs_error=float(np.max(np.abs(identity_delta))),
        identity_probability_nonzero_differences=int(np.count_nonzero(identity_delta)),
        identity_max_error_uuid=bank['uuids'][identity_largest[0]], identity_max_error_label=bank['labels'][identity_largest[1]],
        pool_discrepancies=pools, order_changes=changed, reversed_adjacent_pairs=near_ties,
        largest_score_differences=score_largest, largest_feature_differences=differences[:30],
        differing_feature_names=dict(Counter(row['kind']+':'+row['feature'] for row in differences)),
        identity_probability_examples=probability_examples,
        fixture_sha256=common.digest(fixture_path),
        fusion_manifest_sha256=common.digest(common.WORK/'fusion_reranker/manifest.json'),
        diagnostic_source_sha256=common.digest(__file__),
        versions=dict(python=sys.version, numpy=np.__version__, sklearn=sklearn.__version__, rdkit=rdkit.__version__, torch=torch.__version__),
        platform=platform.platform(), frozen_files_modified=False, checks_relaxed=False,
        note='Scores aligned by original candidate string. No ties are rounded and no frozen rank order is changed. The existing preflight still governs launch.')
    destination = args.output or common.ROOT/'fusion_replay_debug.json'
    if destination.exists():
        raise FileExistsError('Refusing to overwrite diagnostic evidence')
    common.write_json(destination, report)
    print(json.dumps({key: report[key] for key in ['status', 'reactions', 'candidates', 'changed_order_reactions',
                'max_score_abs_error', 'max_chemical_feature_abs_error', 'max_identity_probability_abs_error',
                'max_identity_candidate_feature_abs_error']}), flush=True)


if __name__ == '__main__':
    main()
