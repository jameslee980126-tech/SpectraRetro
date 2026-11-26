"""Check paper navigation, frozen reaction partitions and supplementary data identities."""
import csv
import hashlib
import json
import re
from pathlib import Path, PureWindowsPath

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def check_hash(path, digest):
    require(path.is_file(), 'Missing source record: ' + str(path))
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    require(actual == digest, 'Source record differs: ' + str(path))


def relative_path(name):
    path = Path(name)
    require(bool(name) and not PureWindowsPath(name).anchor and '\\' not in name,
            'Expected a relative repository path: ' + name)
    return path


def main():
    experiments = json.loads((ROOT/'experiments/manifest.json').read_text(encoding='utf8'))['experiments']
    require(len(experiments) == 18, 'Expected 18 main-text and SI topics')
    paths = set()
    local_files, shared_records, table_excerpts, model_bindings = 0, 0, 0, 0
    weight_registry = {r['path']: r for r in json.loads((ROOT/'checkpoints/WEIGHTS.json').read_text(encoding='utf8'))['files']}
    for e in experiments:
        for key in ['code','data','configuration','checkpoints','results','figures','file_indices','experiment_materials','method_reference']:
            for name in e.get(key, []):
                p = ROOT / name
                require(p.exists(), 'Missing experiment material: ' + name)
                if p.is_dir():
                    require(any(q.is_file() for q in p.rglob('*')), 'Empty material directory: ' + name)
                paths.add(name)
        local_manifest = (ROOT / relative_path(e['local_artifact_manifest'])).resolve()
        require(local_manifest.is_relative_to(ROOT/'experiments'), 'Manifest outside experiment directory')
        local = json.loads(local_manifest.read_text(encoding='utf8'))
        require(local['experiment'] == e['id'] and local['paper_items'] == e['items'], 'Experiment identity differs')
        require(local['files'], 'Experiment has no scientific materials')
        targets = set()
        topic_shared = 0
        for record in local['files']:
            target = (local_manifest.parent / relative_path(record['path'])).resolve()
            if 'storage' in record:
                require(record['storage'] == 'repository_reference' and record['kind'] == 'original_file',
                        'Unsupported experiment storage type')
                allowed = (ROOT/'results', ROOT/'datasets/supplementary_data', ROOT/'checkpoints', ROOT/'experiments/section_4_results/4_1_benchmark')
                require(any(target.is_relative_to(base) for base in allowed),
                        'Shared material must be within an approved repository data or model directory')
                if target.is_relative_to(ROOT/'checkpoints') or '/checkpoints/' in target.as_posix():
                    require(target.relative_to(ROOT).as_posix() in weight_registry, 'Unregistered shared model')
                topic_shared += 1
            else:
                require(target.is_relative_to(local_manifest.parent), 'Local artifact outside its experiment')
            require(target not in targets, 'Duplicate experiment material: ' + record['path'])
            targets.add(target)
            require(target.is_file() and target.stat().st_size == record['bytes'], 'Missing or wrong-sized artifact: ' + record['path'])
            check_hash(target, record['sha256'])
            if record['kind'] == 'original_file':
                source_rel = relative_path(record['source_path'])
                source = (ROOT / source_rel).resolve()
                require('..' not in source_rel.parts and source.is_relative_to(ROOT), 'Invalid source path')
                if 'storage' in record:
                    require(target == source, 'Shared material link differs from its registered source')
                if source != target:
                    check_hash(source, record['sha256'])
            elif record['kind'] == 'verbatim_latex_table':
                require(record['paper_item'] in e['items'], 'Table assigned to a different experiment')
                start = record['source_character_start']
                end = record['source_character_end']
                require(record['source_document'] and re.fullmatch('[0-9a-f]{64}', record['source_sha256']),
                        'Table lacks source-document provenance: ' + str(target))
                require(0 <= start < end and end - start == len(target.read_bytes().decode('utf-8')),
                        'Table length differs from the recorded source span: ' + str(target))
                table_excerpts += 1
            else:
                raise ValueError('Unknown local artifact type')
        topic_local = len(targets) - topic_shared
        require(e['local_material_count'] == topic_local, 'Local material count differs: ' + e['id'])
        if topic_shared or 'shared_result_count' in e or 'material_count' in e:
            require(e.get('shared_result_count') == topic_shared and e.get('material_count') == len(targets),
                    'Shared or total material count differs: ' + e['id'])
        local_files += topic_local
        shared_records += topic_shared
        if local['checkpoint_index']:
            index = json.loads((local_manifest.parent/local['checkpoint_index']).read_text(encoding='utf8'))
            for weight in index['files']:
                require(weight == weight_registry[weight['path']], 'Experiment checkpoint binding differs')
                require((ROOT/weight['path']).is_file(), 'Missing shared checkpoint')
                model_bindings += 1
    require(table_excerpts == 29, 'Expected 3 main-text and 26 SI table excerpts')
    paper_map = json.loads((ROOT/'docs/paper_map.json').read_text(encoding='utf8'))
    displays = paper_map['items']
    expected = {f'Figure {i}' for i in range(1,6)} | {f'Table {i}' for i in range(1,4)}
    expected |= {f'Figure S{i}' for i in range(1,9)} | {f'Table S{i}' for i in range(1,28) if i != 11}
    require(paper_map['display_count'] == len(displays) == len(expected)
            and {r['item'] for r in displays} == expected, 'Paper display mapping differs')
    for r in displays:
        require((ROOT/r['experiment']).is_file(), 'Missing experiment guide')
        for name in r['source_paths'] + r['display_paths']:
            require((ROOT/name).exists(), 'Missing display material: '+name)
        directory = (ROOT / r['experiment']).parent
        local = json.loads((directory / 'artifacts.json').read_text(encoding='utf8'))
        recorded = {(directory / a['path']).resolve().relative_to(ROOT).as_posix() for a in local['files']
                    if a.get('paper_item') == r['item'] or a['role'].startswith(r['item'] + ':')}
        expected_files = 3 if r['item'] == 'Figure S6' else 1
        require(len(recorded) == len(r['local_display_paths']) == len(r['display_paths']) == expected_files
                and recorded == set(r['local_display_paths']) == set(r['display_paths']),
                'Incorrect figure or table files: '+r['item'])
        if r['item'] == 'Figure S6':
            require(all(Path(name).suffix.lower() == '.png' for name in recorded),
                    'Expected the three PNG pages of Figure S6')
    counts = {}
    for split, n_full, n_small in [('train',74146,18370),('valid',9395,2400),('test',9250,2400)]:
        full = read_csv(ROOT/f'datasets/full/{split}.csv')
        small = read_csv(ROOT/f'datasets/small/{split}.csv')
        indices = read_csv(ROOT/f'datasets/small/{split}_index.csv')
        require(len(full)==n_full and len(small)==len(indices)==n_small, 'Partition size differs')
        require(set(full[0])=={'class','rxn_smiles','original_class'}, 'Reaction schema differs')
        source_rows = [int(r['source_row']) for r in indices]
        require(len(source_rows)==len(set(source_rows)), 'Repeated parent-row index')
        for row, idx in zip(small, source_rows):
            require(0<=idx<len(full) and row==full[idx], 'Subset differs from its frozen parent row')
        counts[split]=dict(full=len(full),small=len(small))
    supplements=ROOT/'datasets/supplementary_data'
    rows=read_csv(supplements/'source_index.csv')
    for row in rows:
        check_hash(supplements/row['relative_file'],row['sha256'])
    mechanism=ROOT/'results/source_data/mechanistic_interpretability/supplement'
    data=json.loads((mechanism/'data_index.json').read_text(encoding='utf8'))
    for row in data['files']:
        check_hash(mechanism/row['path'],row['sha256'])
    cases=json.loads((ROOT/'results/cases/source_index.json').read_text(encoding='utf8'))['files']
    for row in cases:
        check_hash(ROOT/row['path'],row['sha256'])
    result=dict(status='passed',experiment_guides=len(experiments),paper_displays=len(displays),linked_material_paths=len(paths),
                local_experiment_materials=local_files,shared_experiment_references=shared_records,
                experiment_materials=local_files+shared_records,verbatim_table_excerpts=table_excerpts,experiment_checkpoint_bindings=model_bindings,
                dataset_counts=counts,supplementary_source_records=len(rows),
                mechanistic_source_records=len(data['files']),case_source_records=len(cases),
                scope='Distributed file and table checksums, table provenance and source span lengths, and exact subset-to-parent binding.')
    print(json.dumps(result))


if __name__=='__main__':
    main()
