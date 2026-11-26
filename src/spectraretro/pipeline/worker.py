"""Internal model workers for the continuous research launcher."""
import argparse
import csv
import importlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
BENCHMARK = ROOT / 'experiments/section_4_results/4_1_benchmark'
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'src'))


def center_module(mode):
    import numpy as np
    if int(np.__version__.split('.')[0]) < 2:
        # NumPy 2 uses this module name in saved RNG arrays. The array
        # implementation is the same public NumPy type in NumPy 1 environments.
        for name in ['multiarray', 'numeric']:
            sys.modules.setdefault('numpy._core.' + name, getattr(np.core, name))
        sys.modules.setdefault('numpy._core', np.core)
    return importlib.import_module(f"spectraretro.{mode}-class.reaction_center.run_stage1")


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def predict_centers(plan):
    import torch
    module = center_module(plan['mode'])
    common = getattr(module, 'common', module)
    common.seed_all(0)
    output = Path(plan['output']) / 'centers'
    output.mkdir(parents=True, exist_ok=False)
    os.environ['SPECTRARETRO_STAGE1_CACHE'] = str(output / 'cache')
    saved = common.torch_load(BENCHMARK / f"{plan['mode']}-class/checkpoints/reaction_center.pth")
    contract = saved.get('contract') or saved['metadata']['contract']
    known_class = contract.get('known_class', not contract.get('unknown_class', True))
    if bool(known_class) != (plan['mode'] == 'known'):
        raise ValueError('Center checkpoint conditioning differs from the requested mode')
    # Architecture/schema construction uses training reactions, never test labels.
    representatives = {}
    for row in read_csv(ROOT / 'datasets/full/train.csv'):
        representatives.setdefault(int(row['class']), row)
    schema = output / 'schema.csv'
    with schema.open('w', encoding='utf8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(representatives[min(representatives)]))
        writer.writeheader(); writer.writerows(representatives[k] for k in sorted(representatives))
    sources = module.source_manifest()
    schema_data, _, _, _ = common.load_split(schema, sources)
    task = module.build_task(schema_data, [], prior=contract['arm']=='prior',
                             width=contract['width'], num_layers=contract['layers'])
    task.load_state_dict(saved['model'], strict=True)
    test_csv = ROOT / 'datasets/full/test.csv'
    if plan['diagnostic']:
        selected = read_csv(test_csv)[:2]
        test_csv = output / 'test.csv'
        with test_csv.open('w', encoding='utf8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(selected[0]))
            writer.writeheader(); writer.writerows(selected)
    test, rows, index, _ = common.load_split(test_csv, sources)
    device = torch.device(plan['device'])
    metrics = common.evaluate(task.to(device), test, rows, index,
                              plan['center_settings'].get('batch_size', 8), device,
                              plan['center_predictions'])
    common.save_json(output / 'metrics.json', metrics)
    if plan['diagnostic']:
        common.save_json(Path(plan['selection']), dict(reaction_ids=[r['sample_id'] for r in index]))


def fit_centers(plan):
    module = center_module(plan['mode'])
    output = Path(plan['output']) / 'training/center'
    output.parent.mkdir(parents=True, exist_ok=True)
    os.environ['SPECTRARETRO_STAGE1_CACHE'] = str(output.parent / 'center_cache')
    settings = dict(epochs=50, patience=100000, batch_size=64, width=512, layers=3, seed=0, lr=1e-4)
    unknown = set(plan['center_settings']) - set(settings)
    if unknown:
        raise ValueError('Unknown center training settings: ' + ', '.join(sorted(unknown)))
    settings.update(plan['center_settings'])
    argv = [module.__file__, '--arm', 'baseline',
            '--output', str(output), '--device', 'cuda' if plan['device'].startswith('cuda') else 'cpu']
    for name, value in settings.items():
        argv += ['--' + name.replace('_', '-'), str(value)]
    for split in ['train', 'valid', 'test']:
        argv += ['--' + split + '-csv', str(ROOT / f'datasets/full/{split}.csv')]
    sys.argv = argv
    module.main()
    if not Path(plan['center_predictions']).is_file():
        raise RuntimeError('Selected center training did not export complete test predictions')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action')
    parser.add_argument('--run', type=Path, required=True)
    cli = parser.parse_args()
    plan = json.loads(cli.run.read_text(encoding='utf8'))
    output = Path(plan['output'])
    action = cli.action
    if action.startswith('check_'):
        role = action[6:]
        if role == 'center':
            module = center_module(plan['mode'])
            common = getattr(module, 'common', module)
            common.torch_load(BENCHMARK / f"{plan['mode']}-class/checkpoints/reaction_center.pth")
        elif role == 'model':
            from spectraretro.completion import api
            api.setup(); api.configure(plan['device'])
            if plan['workflow'] == 'train-and-evaluate':
                from composition_model import construct
                config = dict(api.default_config(), torch_device='cpu')
                model = construct(plan['mode'], **config)
                initial = api.torch_load(BENCHMARK / f"{plan['mode']}-class/checkpoints/initial_model.pt", map_location='cpu')
                model.load_state_dict(initial['state_dict'], strict=True)
            else:
                api.load_model(plan['mode'], plan['completion_checkpoint'], 'cpu')
        elif role == 'decode':
            import numpy  # noqa: F401
            from openbabel import openbabel as ob
            conversion = ob.OBConversion()
            if not conversion.SetInAndOutFormats('xyz', 'smi'):
                raise RuntimeError('Open Babel XYZ/SMILES plugins unavailable')
        elif role == 'score':
            from spectraretro.fusion_reranker.runtime import FrozenFusionRanker
            FrozenFusionRanker(plan['mode'], 'fusion')
        else:
            raise ValueError('Unknown environment: ' + role)
        if role in ['center', 'model']:
            import torch
            if plan['device'].startswith('cuda'):
                value = torch.ones(2, device=plan['device'])
                if float((value + value).sum()) != 4:
                    raise RuntimeError('CUDA execution check failed')
        print(json.dumps(dict(environment=role, status='passed', python=sys.executable)), flush=True)
        return
    if action == 'predict_centers':
        predict_centers(plan)
        return
    if action == 'fit_centers':
        fit_centers(plan)
        return
    from spectraretro.completion import api
    args = SimpleNamespace(mode=plan['mode'], device=plan['device'], diagnostic=plan['diagnostic'],
                           draws=plan['draws'], steps=plan['steps'], selection=plan['selection'],
                           center_predictions=Path(plan['center_predictions']),
                           checkpoint=Path(plan['completion_checkpoint']), workers=plan['workers'])
    if action == 'fit_completion':
        args.output = output / 'training/completion'
        api.train(args)
    elif action == 'generate':
        centers = [json.loads(line) for line in args.center_predictions.read_text(encoding='utf8').splitlines()]
        mapping = read_csv(api.DATA / 'final_test/prepared/test_record_map.csv')
        if plan['selection']:
            selected = set(json.loads(Path(plan['selection']).read_text(encoding='utf8'))['reaction_ids'])
            mapping = [r for r in mapping if r['reaction_id'] in selected]
        identities = {r['reaction_id']: r['product'] for r in mapping}
        if len(centers) != len(identities) or {r['sample_id']: r['product'] for r in centers} != identities:
            raise ValueError('Center predictions and conditional geometry refer to different reactions')
        api.write_json(output / 'conditioning.json', dict(
            center_predictions_sha256=api.sha(args.center_predictions),
            completion_checkpoint_sha256=api.sha(args.checkpoint), reactions=len(identities),
            fragments=len(mapping), geometry='prepared CHRetro test synthons (SI S3.6)',
            center_use='correct-draw count gate at complete-reaction evaluation',
            size_bank_sha256=api.sha(BENCHMARK / f"{plan['mode']}-class/data/evaluation/shared_sizes.npz")))
        args.output = output / 'samples'
        api.sample(args)
    elif action == 'decode':
        from spectraretro.completion import decode
        sys.argv = [str(Path(decode.__file__)), '--input', str(output / 'samples'),
                    '--output', str(output / 'decoded'), '--workers', str(args.workers)]
        decode.main()
    elif action == 'evaluate':
        binding = json.loads((output / 'conditioning.json').read_text(encoding='utf8'))
        if (api.sha(args.center_predictions) != binding['center_predictions_sha256'] or
                api.sha(args.checkpoint) != binding['completion_checkpoint_sha256']):
            raise ValueError('Selected weights or center predictions changed during the workflow')
        args.input = output / 'decoded/result.csv'
        args.output = output / 'evaluation'
        api.score(args)
    else:
        raise ValueError('Unknown workflow action: ' + action)


if __name__ == '__main__':
    main()
