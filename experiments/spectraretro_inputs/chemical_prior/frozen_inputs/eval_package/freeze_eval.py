"""Freeze all checkpoint paths and the common 50-draw validation evaluation."""
import argparse
import json
from pathlib import Path
import portable_common as common


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle-root', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--checkpoints-json', type=Path, required=True,
                        help='JSON mapping arm names to absolute or run-root-relative best.pt paths')
    args = parser.parse_args()
    common.configure(args.bundle_root, args.run_root)
    import torch
    from portable_common import CYCLE, OLD, DATA, WORK, ROOT, digest, write_json
    mapping = json.loads(args.checkpoints_json.read_text(encoding='utf8'))
    if 'control' not in mapping or set(mapping) - set(common.ARMS):
        raise ValueError('Need control plus recognized single-variable arms')
    artifacts = [OLD / 'newdata/eval_geometry.pt', CYCLE / 'shared_sizes.npz', CYCLE / 'batch_order.json',
                 CYCLE / 'unknown_compatibility/eval_prior.npz',
                 DATA / 'data/stage2/similarity_eval_table.csv', DATA / 'data/metadata/eval_record_map.csv',
                 DATA / 'runs/stage1_unknown_baseline/valid_predictions.jsonl', WORK / 'fusion_reranker/manifest.json']
    artifacts.append(ROOT / 'reference_replay.csv')
    for name in ['environment_preflight.json', 'reference_replay_manifest.json']:
        evidence = ROOT / name
        value = json.loads(evidence.read_text(encoding='utf8'))
        if value.get('status') != 'passed':
            raise ValueError('Unverified Linux runtime/reference: ' + name)
        artifacts.append(evidence)
    checkpoints = {}
    for arm, text in mapping.items():
        path = Path(text)
        if not path.is_absolute():
            path = ROOT / path
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        if checkpoint.get('arm') != arm or checkpoint.get('epoch', 0) < 1:
            raise ValueError('Checkpoint is not a complete validated arm: ' + arm)
        if checkpoint.get('compatibility_enabled') != (arm == 'compatibility'):
            raise ValueError('Compatibility flag mismatch')
        if checkpoint.get('observed_graph_enabled') != (arm == 'graph'):
            raise ValueError('Graph flag mismatch')
        checkpoints[arm] = {'path': str(path.resolve()), 'sha256': digest(path),
                            'epoch': checkpoint['epoch'], 'steps': checkpoint['steps'],
                            'validation_l2': checkpoint['validation_l2'],
                            'training_protocol_sha256': checkpoint['protocol_sha256']}
        artifacts.append(path)
    artifacts += list(Path(__file__).parent.glob('*.py'))
    artifacts += list((common.BUNDLE / 'graph_package').glob('*.py'))
    if (ROOT / 'protocol.json').exists():
        raise FileExistsError('Evaluation protocol already frozen')
    ROOT.mkdir(parents=True, exist_ok=True)
    for name in ['shared_sizes.npz', 'batch_order.json']:
        target = ROOT / name
        if target.exists():
            raise FileExistsError(target)
        target.write_bytes((CYCLE / name).read_bytes())
        assert digest(target) == digest(CYCLE / name)
    write_json(ROOT / 'protocol.json', dict(
        arms=list(mapping), mode='unknown', rows=4161, reactions=2400, draws=50, steps=100,
        batch_size=64, seed=0, checkpoints=checkpoints,
        sources={str(path.resolve()): digest(path) for path in artifacts},
        shared_sizes_sha256=digest(ROOT / 'shared_sizes.npz'),
        batch_order_sha256=digest(ROOT / 'batch_order.json'),
        reference='Linux OpenBabel XYZ reference and candidates; same environment for all arms',
        gate='unknown Stage1 gold-index hit count among stored first50 draws; retain first k Stage2 draws',
        ranking='all same-draw unique candidates, frozen unknown fusion; frequency diagnostic',
        evaluation='complete reused validation, 50 draws, reference-center synthons; not independent test',
        no_deployment=True, class_input=False,
        absolute_scores_not_comparable_to_windows=True))


if __name__ == '__main__':
    main()
