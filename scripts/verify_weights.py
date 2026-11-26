"""Verify checkpoint identities, coverage and available validation selections."""
from pathlib import Path
import csv
import hashlib
import json
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def jsonlines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def main():
    registry = read(ROOT/'checkpoints/WEIGHTS.json')
    weights = {item['path']: item for item in registry['files']}
    issues = []
    for rel, item in weights.items():
        path = ROOT/rel
        if not path.is_file(): issues.append('Missing checkpoint: '+rel)
        elif path.stat().st_size != item['bytes'] or digest(path) != item['sha256']:
            issues.append('Checkpoint identity differs from the recorded experiment: '+rel)
        else:
            with path.open('rb') as stream:
                zip_header = stream.read(4) == b'PK\x03\x04'
            if zip_header and not zipfile.is_zipfile(path):
                issues.append('Incomplete ZIP checkpoint: '+rel)
    non_parameters = {
        'experiments/spectraretro_inputs/final_test/prepared/test_geometry.pt',
        'experiments/spectraretro_inputs/old/newdata/eval_geometry.pt',
        'experiments/spectraretro_inputs/old/newdata/train_geometry.pt',
        'src/baselines/GLN/prepared_full_known/preprocessing/tpl-default/cano_smarts.pkl',
    }
    for item in read(ROOT/'datasets/ADDITIONAL_BINARY_RESOURCES.json')['files']:
        rel=item['path'];path=ROOT/rel
        if not path.is_file() or path.stat().st_size!=item['bytes'] or digest(path)!=item['sha256']:
            issues.append('Registered data binary differs: '+rel)
        non_parameters.add(rel)
    manifest = registry['runtime_ranking_manifest']
    ranking_manifest = ROOT / manifest['path']
    if ranking_manifest.stat().st_size != manifest['bytes'] or digest(ranking_manifest) != manifest['sha256']:
        issues.append('Ranking feature or parameter manifest differs')
    experiment_weights = set()
    for experiment in read(ROOT/'experiments/manifest.json')['experiments']:
        materials = ROOT / experiment['local_artifact_manifest']
        for artifact in read(materials)['files']:
            registered = weights.get(artifact.get('source_path'))
            if registered is None:
                continue
            target = (materials.parent / artifact['path']).resolve()
            if (target != (ROOT / registered['path']).resolve() or
                    registered['sha256'] != artifact['sha256'] or
                    registered['bytes'] != artifact['bytes']):
                issues.append('Experiment parameter differs from its registered model: '+artifact['path'])
            else:
                experiment_weights.add(registered['path'])
    for path in ROOT.rglob('*'):
        if not path.is_file() or path.suffix.lower() not in {'.pt','.pth','.ckpt','.pkl','.dump','.joblib','.safetensors','.onnx'}:
            continue
        rel = path.relative_to(ROOT).as_posix()
        if rel.startswith('outputs/'): continue
        if rel not in weights and rel not in non_parameters and rel not in experiment_weights:
            issues.append('Unregistered parameter or binary resource: '+rel)
    replayed = []
    for mode in ['known','unknown']:
        mode_path = f'experiments/section_4_results/4_1_benchmark/{mode}-class'
        base = ROOT / mode_path / 'results/training'
        completion = read(base/'completion/validation.json')['records']
        size = jsonlines(base/'size/curves.jsonl')
        center = jsonlines(base/'center/metrics.jsonl')
        def center_value(r): return r['valid'].get('overall',r['valid'])['top-1 accuracy']
        selections = {
            'completion_model.pt': min(completion, key=lambda r:(r['validation_l2'],r['epoch']))['epoch'],
            'size_model.pt': min(size, key=lambda r:(r['validation_ce'],r['epoch']))['epoch'],
            'reaction_center.pth': max(center,key=lambda r:(center_value(r),-r['epoch']))['epoch'],
        }
        config = read(ROOT / mode_path / 'configs/full.json')
        for key, name in [('checkpoint','completion_model.pt'),('size_checkpoint','size_model.pt')]:
            if config[key] != f'{mode_path}/checkpoints/{name}':
                issues.append('Full-run configuration uses a different checkpoint: '+mode+'/'+key)
        for name, epoch in selections.items():
            rel=f'{mode_path}/checkpoints/{name}'
            if epoch != weights[rel]['selection']['epoch']:
                issues.append('Validation-selected epoch differs: '+rel)
            replayed.append(dict(path=rel, selected_epoch=epoch))
    for method, mode, epoch in [('Graph2Edits','full_unknown',58),('TransGraphEdit','full_unknown',20),
                               ('Graph2Edits','full_known',124),('TransGraphEdit','full_known',56)]:
        p=ROOT/f'results/baselines/checkpoint_selection/{method}/{mode}/training.csv'
        text=p.read_text(encoding='utf-8-sig').splitlines()
        start=next(i for i,line in enumerate(text) if line.startswith('epoch,train_acc,valid_acc'))
        rows=list(csv.DictReader(text[start:]))
        tie=1 if (method,mode)==('Graph2Edits','full_unknown') else -1
        best=max(rows,key=lambda r:(float(r['valid_acc']),tie*int(r['epoch'])))
        if int(best['epoch']) != epoch: issues.append('Validation-selected epoch differs: '+method)
        replayed.append(dict(method=method,mode=mode,selected_epoch=epoch))
    print(json.dumps(dict(status='passed' if not issues else 'failed', files=len(weights),
        experiment_parameter_paths=len(experiment_weights),
        parameter_files=sum(x['role']!='serialized_configuration' for x in weights.values()),
        unique_parameter_digests=len({x['sha256'] for x in weights.values() if x['role']!='serialized_configuration'}),
        selection_replays=replayed, coverage_notes=registry['coverage_notes'], issues=issues),indent=2))
    return 1 if issues else 0


if __name__ == '__main__':
    sys.exit(main())
