"""CPU per-candidate replay of frozen fusion models and observable identity features."""
import argparse
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
    args = parser.parse_args()
    common.configure(args.bundle_root, args.run_root)
    root = common.ROOT
    if (root / 'environment_preflight.json').exists():
        raise FileExistsError('Environment evidence already exists')
    package = Path(__file__).resolve().parent
    manifest = json.loads((package / 'fixture_manifest.json').read_text(encoding='utf8'))
    fixture_path = package / 'ranking_replay_fixture.json'
    if common.digest(fixture_path) != manifest['artifacts'][fixture_path.name]:
        raise ValueError('Ranking fixture changed')
    fixture = json.loads(fixture_path.read_text(encoding='utf8'))
    import torch
    import sklearn
    import rdkit
    from rdkit import RDLogger
    from fusion_reranker.runtime import FrozenFusionRanker, candidate_records
    from fusion_reranker.identity_adapter import FrozenIdentityAdapter
    torch.set_num_threads(2)
    RDLogger.DisableLog('rdApp.*')
    ranker = FrozenFusionRanker('unknown', 'fusion')
    results = ranker.rank_batch(fixture['requests'])
    max_score_error, max_feature_error = 0., 0.
    candidates = 0
    for request, actual, expected, features in zip(fixture['requests'], results, fixture['expected'], fixture['candidate_features']):
        assert actual['reaction_id'] == expected['reaction_id']
        assert [c['candidate'] for c in actual['candidates']] == [c['candidate'] for c in expected['candidates']], 'Frozen order differs'
        for a, e in zip(actual['candidates'], expected['candidates']):
            max_score_error = max(max_score_error, abs(a['fusion_score'] - e['score']))
            candidates += 1
        rebuilt = candidate_records(request)
        assert len(rebuilt) == len(features)
        for a, e in zip(rebuilt, features):
            assert set(a['features']) == set(e)
            for key in e:
                max_feature_error = max(max_feature_error, abs(a['features'][key] - e[key]))
    observations = []
    for request in fixture['requests']:
        for fragment in request['fragments']:
            observations.append({key: fragment[key] for key in ['uuid', 'fragments', 'anchors', 'center1', 'center2'] if key in fragment} |
                                dict(reaction_id=request['reaction_id'], product=request['product']))
    identity = FrozenIdentityAdapter().predict_fragments(observations)
    assert list(identity['uuids']) == fixture['identity_uuids']
    max_identity_error = float(np.max(np.abs(identity['probabilities'] - np.asarray(fixture['identity_probabilities']))))
    assert max_score_error < 1e-6 and max_identity_error < 1e-6 and max_feature_error < 1e-6
    first = fixture['requests'][0]
    for key, value in [('class_id', 5), ('target', 'C')]:
        try:
            ranker.rank(dict(first, **{key: value}))
        except ValueError:
            pass
        else:
            raise AssertionError('Forbidden inference input accepted: ' + key)
    full = ranker.rank(first)['candidates']
    top5 = ranker.rank(first, top_k=5)['candidates']
    assert [c['candidate'] for c in full[:5]] == [c['candidate'] for c in top5]
    common.write_json(root / 'environment_preflight.json', dict(
        status='passed', device='cpu', new_training=False, new_diffusion_sampling=False,
        reactions=len(results), candidates=candidates, all_candidate_orders_equal=True,
        max_score_abs_error=max_score_error, max_chemical_feature_abs_error=max_feature_error,
        max_identity_probability_abs_error=max_identity_error, forbidden_inputs_rejected=True,
        full_pool_before_topk=True, fusion_manifest_sha256=common.digest(common.WORK / 'fusion_reranker/manifest.json'),
        fixture_sha256=common.digest(fixture_path), versions=dict(python=sys.version, numpy=np.__version__,
            sklearn=sklearn.__version__, rdkit=rdkit.__version__, torch=torch.__version__),
        platform=platform.platform(), scope='hash-fixed 24 existing validation reactions spanning six metadata classes; packaging replay, not accuracy evidence'))
    print(json.dumps(dict(status='passed', reactions=len(results), candidates=candidates,
                         max_score_error=max_score_error, max_identity_error=max_identity_error)), flush=True)


if __name__ == '__main__':
    main()
