"""Run a complete research workflow with automatic artifact handoff."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
BENCHMARK = ROOT / 'experiments/section_4_results/4_1_benchmark'
WORKER = Path(__file__).with_name('worker.py')


def rooted(value):
    path = Path(value).expanduser()
    if path.parts and path.parts[0] in {'known-class', 'unknown-class'}:
        return (BENCHMARK / path).resolve()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    os.replace(temporary, path)


def python_path(value):
    path = Path(value).expanduser()
    if path.is_absolute() or path.parent != Path('.'):
        result = rooted(path)
    else:
        found = shutil.which(str(path))
        if not found:
            raise ValueError('Python executable not found: ' + str(path))
        result = Path(found).resolve()
    if not result.is_file():
        raise ValueError('Python executable not found: ' + str(result))
    return str(result)


def stream_process(argv, log, environment):
    """Keep one launcher active while model environments run sequentially."""
    with log.open('w', encoding='utf8') as handle:
        child = subprocess.Popen(argv, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, encoding='utf8', errors='replace')
        try:
            for line in child.stdout:
                handle.write(line); handle.flush()
                print(line, end='', flush=True)
            code = child.wait()
            if code:
                raise subprocess.CalledProcessError(code, argv)
        except BaseException:
            if child.poll() is None:
                child.terminate(); child.wait()
            raise


def build_plan(mode, args):
    mode_dir = BENCHMARK / f'{mode}-class'
    config_path = rooted(args.config) if args.config else mode_dir / 'configs/full.json'
    config = json.loads(config_path.read_text(encoding='utf8'))
    if config['mode'] != mode or bool(config['class_input']) != (mode == 'known'):
        raise ValueError('Configuration belongs to a different reaction-class regime')
    if (config['draws'], config['steps'], config['reactions'], config['rows']) != (300, 100, 9250, 15830):
        raise ValueError('The full benchmark requires 9250 reactions, 15830 fragments, 300 draws and 100 intervals')
    if args.smoke and args.command == 'train':
        raise ValueError('Smoke mode checks prediction through evaluation; it does not train reduced-data models')
    if args.command == 'train' and args.device == 'cpu':
        raise ValueError('The complete training workflow requires CUDA')
    settings = config.get('pipeline', {})
    environments = settings.get('python', {})
    executables = {name: python_path(getattr(args, name + '_python') or environments.get(name) or sys.executable)
                   for name in ['center', 'model', 'decode', 'score']}
    output = rooted(args.output or f'outputs/{mode}_{args.command}_{datetime.now():%Y%m%d_%H%M%S_%f}')
    if output == ROOT or (ROOT in output.parents and output.relative_to(ROOT).parts[0] != 'outputs'):
        raise ValueError('Outputs inside the repository must be placed under outputs/')
    if output.exists():
        raise FileExistsError('Choose a new output directory: ' + str(output))
    training = args.command == 'train'
    centers = output / ('training/center/test_predictions.jsonl' if training else 'centers/test_predictions.jsonl')
    checkpoint = output / 'training/completion/best.pt' if training else rooted(config['checkpoint'])
    data = ROOT / 'experiments/spectraretro_inputs'
    required = [config_path, WORKER, rooted(config['checkpoint']),
                mode_dir / 'checkpoints/reaction_center.pth',
                mode_dir / 'data/evaluation/shared_sizes.npz',
                mode_dir / 'data/evaluation/batch_order.json',
                mode_dir / 'data/evaluation/reference_replay.csv',
                ROOT / 'src/spectraretro/fusion_reranker/manifest.json']
    required += [ROOT / f'datasets/full/{split}.csv' for split in ['train', 'valid', 'test']]
    required += [data / ('final_test/prepared/' + name) for name in
                 ['test_geometry.pt', 'test_prior.npz', 'test_record_map.csv', 'similarity_test_table.csv']]
    if training:
        required += [mode_dir / 'checkpoints/initial_model.pt']
        required += [data / f'old/newdata/{s}_geometry.pt' for s in ['train', 'eval']]
        required += [data / f'cycle/unknown_compatibility/{s}_prior.npz' for s in ['train', 'eval']]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError('Missing workflow inputs:\n' + '\n'.join(missing))
    for name in ['shared_sizes', 'batch_order']:
        suffix = '.npz' if name == 'shared_sizes' else '.json'
        if digest(mode_dir / f'data/evaluation/{name}{suffix}') != config[name + '_sha256']:
            raise ValueError('Fixed evaluation input hash differs from configuration: ' + name)
    plan = dict(mode=mode, workflow='train-and-evaluate' if training else 'pretrained-evaluation',
                diagnostic=args.smoke, config=str(config_path), output=str(output), python=executables,
                device=args.device or ('cpu' if args.smoke else 'cuda:0'),
                draws=2 if args.smoke else 300, steps=2 if args.smoke else 100,
                center_predictions=str(centers), completion_checkpoint=str(checkpoint),
                selection=str(output / 'selection.json') if args.smoke else None,
                center_settings=settings.get('center', {}), workers=args.workers,
                required_files=len(required),
                protocol='CHRetro precursor recovery (SI S3.6). Model scores incorporate candidate sampling frequency.')
    if training:
        jobs = [('stage1_reaction_center', 'fit_centers', 'center'),
                ('stage2_completion', 'fit_completion', 'model')]
    else:
        jobs = [('stage1_reaction_center', 'predict_centers', 'center')]
    jobs.extend([
        ('stage2_completion', 'generate', 'model'),
        ('stage2_completion', 'decode', 'decode'),
        ('fusion_ranking_and_evaluation', 'evaluate', 'score'),
    ])
    plan['jobs'] = [dict(stage=stage, action=action, environment=env) for stage, action, env in jobs]
    return plan


def run(mode, args):
    plan = build_plan(mode, args)
    if args.check_only or args.command == 'check':
        print(json.dumps(dict(status='passed', files_written=False, models_executed=False, **plan),
                         ensure_ascii=False, indent=2))
        return
    output = Path(plan['output'])
    output.mkdir(parents=True, exist_ok=False)
    (output / 'logs').mkdir()
    spec = output / 'run.json'
    write_json(spec, plan)
    state = dict(status='running', mode=mode, workflow=plan['workflow'], diagnostic=plan['diagnostic'],
                 started_at=datetime.now(timezone.utc).isoformat(), completed=[])
    state_path = output / 'status.json'
    write_json(state_path, state)
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1')
    try:
        # Validate every environment before any long training or generation job.
        checks = [dict(action='check_' + env, environment=env) for env in ['center', 'model', 'decode', 'score']]
        for job in checks + plan['jobs']:
            state['current'] = job['action']
            write_json(state_path, state)
            print('Running: ' + job['action'], flush=True)
            argv = [plan['python'][job['environment']], '-B', str(WORKER), job['action'], '--run', str(spec)]
            stream_process(argv, output / 'logs' / (job['action'] + '.log'), environment)
            state['completed'].append(job['action'])
            write_json(state_path, state)
        result = json.loads((output / 'evaluation/metrics.json').read_text(encoding='utf8'))
        summary = dict(mode=mode, workflow=plan['workflow'], diagnostic=plan['diagnostic'],
                       protocol=plan['protocol'], metrics=result['metrics'],
                       decoded_candidates='decoded/result.csv',
                       ranked_candidates='evaluation/ranked_candidates.jsonl',
                       per_reaction='evaluation/reaction_metrics.json',
                       center_predictions_sha256=digest(plan['center_predictions']),
                       completion_checkpoint_sha256=digest(plan['completion_checkpoint']))
        write_json(output / 'summary.json', summary)
        lines = ['# ' + mode + '-class reproduction', '',
                 'Diagnostic smoke run.' if plan['diagnostic'] else 'Full CHRetro precursor-recovery benchmark (SI S3.6).', '',
                 '| Reactions | Top-1 | Top-3 | Top-5 | Top-10 |', '|---:|---:|---:|---:|---:|']
        metrics = result['metrics']['stage1_gated']['all']
        rates = [f"{100 * metrics[f'fusion_top{k}']:.4f}%" for k in (1, 3, 5, 10)]
        lines.append('| ' + ' | '.join([str(metrics['n']), *rates]) + ' |')
        lines.extend(['', '[Precursor candidates in model-score order](evaluation/ranked_candidates.jsonl)',
                      '· [Metrics including the frequency-only comparison](evaluation/metrics.json)'])
        (output / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
        state.update(status='complete', current=None, finished_at=datetime.now(timezone.utc).isoformat())
        write_json(state_path, state)
        print(json.dumps(dict(status='complete', diagnostic=plan['diagnostic'],
                              results=str(output / 'RESULTS.md'),
                              ranked_candidates=str(output / summary['ranked_candidates'])), ensure_ascii=False), flush=True)
    except BaseException as exc:
        state.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                     error=str(exc), finished_at=datetime.now(timezone.utc).isoformat())
        write_json(state_path, state)
        raise


def main(mode):
    """Compatibility entry for callers of the original mode-specific launcher."""
    from spectraretro.run import main as model_main
    model_main(mode)
