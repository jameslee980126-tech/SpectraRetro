"""Use archived paired bootstrap math and the same exploratory regression guards."""
import argparse
import json
from pathlib import Path
import portable_common as common


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle-root', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--arms', nargs='+', required=True)
    args = parser.parse_args()
    common.configure(args.bundle_root, args.run_root)
    import paired_math
    if args.arms[0] != 'control' or len(set(args.arms)) != len(args.arms):
        raise ValueError('Unique arms must start with control')
    protocol = json.loads((common.ROOT / 'protocol.json').read_text(encoding='utf8'))
    if set(args.arms) - set(protocol['arms']):
        raise ValueError('Arm not frozen in evaluation protocol')
    paired_math.ROOT, paired_math.ARMS = common.ROOT, args.arms
    paired_math.main()
    result = json.loads((common.ROOT / 'analysis.json').read_text(encoding='utf8'))
    result['next'] = 'evaluate remaining frozen arms' if len(result['available']) < len(args.arms) else 'review all single-variable results before combination'
    result['linux_same_environment_comparison'] = True
    result['decisions'] = {}
    if 'control' in result['available']:
        baseline = json.loads((common.ROOT / 'evaluation/control/metrics.json').read_text(encoding='utf8'))
        a = baseline['metrics']['stage1_gated']
        for arm in result['available'][1:]:
            current = json.loads((common.ROOT / 'evaluation' / arm / 'metrics.json').read_text(encoding='utf8'))
            if current['protocol_sha256'] != baseline['protocol_sha256'] or current['ranking_manifest_sha256'] != baseline['ranking_manifest_sha256']:
                raise ValueError('Comparison methods differ')
            b = current['metrics']['stage1_gated']
            delta = {key: b['all'][key] - a['all'][key] for key in ['fusion_top1', 'fusion_top5', 'fusion_top10', 'pool_coverage']}
            violations = []
            for key in ['fusion_top1', 'fusion_top5', 'fusion_top10']:
                if delta[key] < -.005 - 1e-12:
                    violations.append('overall ' + key + ' worse than -0.5pp')
            for cls in range(1, 7):
                key = 'class' + str(cls)
                if a[key]['n'] >= 100:
                    for metric in ['fusion_top1', 'fusion_top10']:
                        if b[key][metric] - a[key][metric] < -.02 - 1e-12:
                            violations.append(key + ' ' + metric + ' worse than -2pp')
            if b['all']['missing_draw_fraction'] - a['all']['missing_draw_fraction'] > .01 + 1e-12:
                violations.append('missing draw fraction increase >1pp')
            positive = any(delta[key] > 0 for key in ['fusion_top1', 'fusion_top5', 'fusion_top10'])
            result['decisions'][arm] = dict(
                status='exploratory_candidate_not_deployed' if positive and not violations else 'rejected_keep_control',
                regression_violations=violations, delta_percentage_points={key: 100 * value for key, value in delta.items()},
                confidence='pointwise paired bootstrap, reused validation, not independent generalization evidence',
                formal_model_changed=False)
    common.write_json(common.ROOT / 'analysis.json', result)
    common.write_json(common.ROOT / 'evaluation_completion.json', dict(
        state='complete' if len(result['available']) == len(args.arms) else 'pending',
        evaluated_arms=result['available'], requested_arms=args.arms,
        analysis_sha256=common.digest(common.ROOT / 'analysis.json'), no_deployment=True))


if __name__ == '__main__':
    main()
