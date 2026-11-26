"""Prepare a new DiffER workspace or run full-data training, sampling and scoring."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / 'experiments/section_4_results/4_1_benchmark'
RUNNER=ROOT/'src/baselines/DiffER/full_experiment'
DEFAULT_CONFIG=ROOT/'configs/baselines/differ_full.json'
NATIVE_RUNS=ROOT/'results/baselines/native_runs/DiffER/full'


def rooted(value):
    path=Path(value).expanduser()
    if path.parts and path.parts[0] in {'known-class', 'unknown-class'}:
        return (BENCHMARK / path).resolve()
    return path.resolve() if path.is_absolute() else (ROOT/path).resolve()


def workspace(value,new=False):
    path=rooted(value)
    if ROOT/'outputs' not in path.parents:
        raise ValueError('DiffER work and output directories must be below this release outputs/')
    if new and path.exists():
        raise FileExistsError('Refusing to overwrite an existing directory: '+str(path))
    return path


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):
            value.update(block)
    return value.hexdigest()


def config(path):
    value=json.loads(path.read_text(encoding='utf8'))
    if any(k in value for k in ['global_deadline','created_at']):
        raise ValueError('Only scientific settings and relative stage durations belong in a new run configuration')
    if [value[k] for k in ['train','valid','test']]!=[74146,9395,9250]:
        raise ValueError('Full-data split must contain 74146/9395/9250 reactions')
    if value['native_candidate_budget']!=50 or value['test_augmentation']*value['samples_per_test_augmentation']!=50:
        raise ValueError('The full-data candidate protocol requires 5 augmentations x 10 samples = 50')
    if value['model_config']['num_timesteps']!=100 or value['training_augmentation']!=20:
        raise ValueError('The native full-data protocol uses 100 diffusion steps and 20 training augmentations')
    if any(int(value['budget_seconds'][k])<=0 for k in ['training_per_mode','inference_per_mode','scoring']):
        raise ValueError('Relative stage durations must be positive')
    return value


def expected_cache_members():
    return {f'shared/{mode}/{split}/{prefix}-{split}.{suffix}'
            for mode in ['known','unknown'] for split in ['train','val','test']
            for prefix,suffix in [('src','txt'),('tgt','txt'),('row-index','jsonl')]}


def inspect_cache(path):
    with zipfile.ZipFile(path) as archive:
        members=archive.namelist()
        if len(members)!=len(set(members)) or set(members)!=expected_cache_members():
            raise ValueError('The cache must contain exactly the 18 shared/<mode>/<split> native text/index members')
        return len(members)


def prepare(path,configuration,archive):
    path.mkdir(parents=True,exist_ok=False)
    inspect_cache(archive)
    member_hashes={}
    with zipfile.ZipFile(archive) as source:
        for info in source.infolist():
            destination=(path/info.filename).resolve()
            if path not in destination.parents:
                raise ValueError('Unsafe cache member path')
            destination.parent.mkdir(parents=True,exist_ok=True)
            member_hash=hashlib.sha256()
            with source.open(info) as src,destination.open('xb') as dst:
                for block in iter(lambda:src.read(2**20),b''):
                    dst.write(block)
                    member_hash.update(block)
            member_hashes[info.filename]=member_hash.hexdigest()
    (path/'raw').mkdir()
    for split in ['train','valid','test']:
        shutil.copyfile(ROOT/f'datasets/full/{split}.csv',path/f'raw/{split}.csv')
    shutil.copyfile(configuration,path/'protocol.json')
    record=dict(status='prepared',scientific_config_sha256=digest(path/'protocol.json'),
        native_cache_sha256=digest(archive),native_cache_members=18,
        cache_member_sha256=member_hashes,
        raw_sha256={s:digest(path/f'raw/{s}.csv') for s in ['train','valid','test']},
        calculation='Independent new calculation; original distributed results are read-only')
    (path/'workspace.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(status='prepared',work_dir=str(path),**{k:v for k,v in record.items() if k!='status'}),indent=2))


def verify_prepared(path):
    receipt=json.loads((path/'workspace.json').read_text(encoding='utf8'))
    if digest(path/'protocol.json')!=receipt['scientific_config_sha256']:
        raise ValueError('Workspace configuration changed after preparation')
    for split,expected in receipt['raw_sha256'].items():
        if digest(path/f'raw/{split}.csv')!=expected:
            raise ValueError('Workspace reaction CSV changed: '+split)
    for name in expected_cache_members():
        if not (path/name).is_file():
            raise FileNotFoundError('Missing prepared native cache member: '+name)
        if digest(path/name)!=receipt['cache_member_sha256'][name]:
            raise ValueError('Prepared native cache content changed: '+name)
    config(path/'protocol.json')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['check','prepare','train','sample','score'])
    parser.add_argument('--mode',choices=['known','unknown'],default='unknown')
    parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    parser.add_argument('--cache',type=Path,help='External native preprocessing ZIP containing both modes, required for prepare')
    parser.add_argument('--work-dir',type=Path,help='New prepared workspace under outputs/; required for train and sample')
    parser.add_argument('--checkpoint',type=Path,help='Checkpoint for sampling; defaults to the distributed mode-specific best.pkl')
    parser.add_argument('--samples',type=Path,help='Native sample stream for scoring; defaults to the distributed full-data sample stream')
    parser.add_argument('--output',type=Path,help='New score directory under outputs/')
    parser.add_argument('--python',default=sys.executable,help='Python executable in the selected DiffER environment')
    parser.add_argument('--check-only',action='store_true',help='Inspect configuration and paths using only the Python standard library; write no files')
    args=parser.parse_args()
    try:
        args.config=rooted(args.config)
        if args.cache is not None:
            args.cache=rooted(args.cache)
        if args.command=='prepare' and args.cache is None:
            raise ValueError('prepare requires --cache pointing to a native preprocessing ZIP; see experiments/section_4_results/4_1_benchmark/baseline_preprocessing_manifests/README.md')
        stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        if args.work_dir is not None:
            args.work_dir=workspace(args.work_dir,new=args.command=='prepare')
        elif args.command=='prepare':
            args.work_dir=workspace(Path('outputs')/f'differ_full_{stamp}',new=True)
        args.checkpoint=rooted(args.checkpoint or (NATIVE_RUNS / args.mode / 'best.pkl'))
        args.samples=rooted(args.samples or (NATIVE_RUNS / args.mode / 'native_samples_batches.jsonl'))
        if args.command=='score':
            args.output=workspace(args.output or Path('outputs')/f'differ_score_{args.mode}_{stamp}',new=True)
        if args.command in ['train','sample'] and args.work_dir is not None:
            destination=args.work_dir/('runs' if args.command=='train' else 'samples')/args.mode
            workspace(destination,new=True)
        settings=config(args.config)
        required=[args.config,RUNNER/'model_runtime.py',ROOT/'src/baselines/DiffER/source_full/source/conditional_model.py']
        required += [ROOT/f'datasets/full/{split}.csv' for split in ['train','valid','test']]
        if args.cache is not None and args.command in ['check','prepare']:
            required += [args.cache]
        if args.command=='sample':
            required += [args.checkpoint]
        if args.command=='score':
            required += [args.samples]
        missing=[str(p) for p in required if not p.is_file()]
        if missing:
            raise FileNotFoundError('Missing files: '+json.dumps(missing))
        members=inspect_cache(args.cache) if args.cache is not None and args.command in ['check','prepare'] else None
        if args.check_only or args.command=='check':
            print(json.dumps(dict(status='passed',command=args.command,mode=args.mode,
                checked_files=len(required),cache_members=members,candidate_budget=50,test_reactions=9250,
                work_dir=str(args.work_dir) if args.work_dir else None,
                prepared_workspace_required=args.command in ['train','sample'],
                model_libraries_imported=False,files_written=False),indent=2))
            return
        if args.command=='prepare':
            prepare(args.work_dir,args.config,args.cache)
            return
        if args.command in ['train','sample']:
            if args.work_dir is None:
                raise ValueError('--work-dir is required; run prepare first')
            verify_prepared(args.work_dir)
        environment=os.environ.copy()
        environment['PYTHONDONTWRITEBYTECODE']='1'
        command=[args.python]
        if args.command in ['train','sample']:
            environment['SPECTRARETRO_DIFFER_WORK']=str(args.work_dir)
            environment['SPECTRARETRO_DIFFER_CONFIG']=str(args.work_dir/'protocol.json')
            command += [str(RUNNER/('train_mode.py' if args.command=='train' else 'sample_mode.py')),'--mode',args.mode]
            if args.command=='sample':
                command += ['--checkpoint',str(args.checkpoint)]
        else:
            command += [str(RUNNER/'score_mode.py'),'--root',str(ROOT),'--mode',args.mode,
                '--samples',str(args.samples),'--output',str(args.output),
                '--test-csv',str(ROOT/'datasets/full/test.csv'),'--protocol',str(args.config)]
        raise SystemExit(subprocess.call(command,cwd=ROOT,env=environment))
    except (ValueError,FileNotFoundError,FileExistsError,zipfile.BadZipFile) as exc:
        parser.error(str(exc))


if __name__=='__main__':
    main()
