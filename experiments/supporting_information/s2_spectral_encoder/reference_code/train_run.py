"""One full/subset run; validation selects checkpoints, test is explicitly locked."""
import argparse
import json
import math
import os
import platform
import random
import time
from runtime import ROOT, verify, load_split, save_json, sha256
from training_core import seed_all, digest_state, build_engine, evaluate
import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scope', choices=('full', 'subset'), required=True)
    p.add_argument('--arm', choices=('full_fspec', 'local_pair'), required=True)
    p.add_argument('--seed', type=int, choices=(0, 1, 2), required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--test-only', action='store_true')
    args = p.parse_args()
    manifest = verify()
    if platform.system() != 'Linux' or not torch.cuda.is_available():
        raise RuntimeError('Formal training/evaluation runs on the GPU server only')
    gate = json.loads((ROOT / 'SERVER_PREFLIGHT.json').read_text())
    if (not gate['server_ready'] or gate['execution_sha256'] != sha256(ROOT / 'EXECUTION_MANIFEST.json')
            or gate['pair_manifest_sha256'] != sha256(ROOT / 'PAIR_MANIFEST.json')):
        raise RuntimeError('Missing or stale server preflight')
    if args.seed > 0 and not args.test_only:
        decision = json.loads((ROOT / f'CONTINUE_{args.scope}_seed{args.seed}.json').read_text())
        if not decision.get('approved') or decision.get('basis') != 'paired_validation_review':
            raise RuntimeError('Later seed requires documented paired validation review')
    seed_all(args.seed)
    train, _ = load_split(args.scope, 'train')
    valid, valid_mapping = load_split(args.scope, 'valid')
    engine = build_engine(args.arm, train, valid)
    contract = {'scope': args.scope, 'arm': args.arm, 'seed': args.seed, 'epochs': 50,
                'width': 512, 'layers': 3, 'batch_size': 64, 'optimizer': 'Adam',
                'lr': 1e-4, 'weight_decay': 0, 'dropout': 0, 'unknown_class': True,
                'train_rows': len(train), 'valid_rows': len(valid), 'test_loaded': False,
                'selection': 'max validation Top1, earliest tie',
                'candidate_domain': 'unchanged canonical bonds plus atoms',
                'initial_state_sha256': digest_state(engine.model),
                'parameters': sum(p.numel() for p in engine.model.parameters()),
                'execution_sha256': sha256(ROOT / 'EXECUTION_MANIFEST.json'),
                'pair_manifest_sha256': sha256(ROOT / 'PAIR_MANIFEST.json'),
                'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0)}
    if contract['parameters'] != 5349894:
        raise RuntimeError('Parameter count differs from matched FSpec')
    if args.scope == 'subset':
        old = json.loads((ROOT / f'subset/baseline_seed{args.seed}/manifest.json').read_text())
        keys = ('seed', 'epochs', 'width', 'layers', 'batch_size', 'optimizer', 'lr',
                'weight_decay', 'dropout', 'unknown_class', 'train_rows', 'valid_rows',
                'parameters', 'initial_state_sha256', 'torch')
        if any(old[k] != contract[k] for k in keys):
            raise RuntimeError('Subset baseline training contract/initial state differs')
    elif args.arm == 'local_pair':
        paired = json.loads((ROOT / 'runs/full' / f'full_fspec_seed{args.seed}' / 'manifest.json').read_text())
        keys = ('seed', 'epochs', 'width', 'layers', 'batch_size', 'optimizer', 'lr',
                'weight_decay', 'dropout', 'unknown_class', 'train_rows', 'valid_rows',
                'parameters', 'initial_state_sha256', 'torch', 'execution_sha256',
                'pair_manifest_sha256')
        if any(paired[k] != contract[k] for k in keys):
            raise RuntimeError('Full paired training contract differs')
    run = ROOT / 'runs' / args.scope / f'{args.arm}_seed{args.seed}'
    if args.test_only:
        lock = json.loads((ROOT / f'TEST_LOCK_{args.scope}.json').read_text())
        if lock.get('selection') != 'validation_only' or not lock.get('runs'):
            raise RuntimeError('Missing frozen test evaluation run list')
        for entry in lock['runs']:
            if sha256(ROOT / entry['checkpoint']) != entry['sha256']:
                raise RuntimeError('A locked checkpoint changed')
        key = f'{args.arm}_seed{args.seed}'
        selected = [r for r in lock['runs'] if r['key'] == key]
        if len(selected) != 1:
            raise RuntimeError('Run is not in the frozen test list')
        checkpoint = ROOT / selected[0]['checkpoint']
        state = torch.load(checkpoint, map_location=engine.device, weights_only=False)
        engine.model.load_state_dict(state['model'])
        test, mapping = load_split(args.scope, 'test')
        metrics, records = evaluate(engine, test, mapping)
        output = ROOT / 'test_results' / args.scope / (key + '.json')
        if output.exists():
            raise RuntimeError('Refusing to overwrite test inference')
        save_json(output, {'metrics': metrics, 'records': records,
                          'checkpoint_sha256': selected[0]['sha256'], 'scope': args.scope,
                          'evaluation': 'original Stage1 task ranking, all frozen scope test rows'})
        return
    initial, best, best_epoch = 0, -math.inf, 0
    run.mkdir(parents=True, exist_ok=args.resume)
    if args.resume:
        if json.loads((run / 'manifest.json').read_text()) != contract:
            raise RuntimeError('Resume contract differs')
        state = torch.load(run / 'last.pth', map_location=engine.device, weights_only=False)
        engine.model.load_state_dict(state['model'])
        engine.optimizer.load_state_dict(state['optimizer'])
        meta = state['metadata']
        initial, best, best_epoch = meta['completed_epochs'], meta['best_top1'], meta['best_epoch']
        engine.meter.epoch_id = initial
        engine.meter.epoch2batch = [0] * (initial + 1)
        engine.meter.time = [time.time()] * (initial + 1)
        torch.set_rng_state(meta['torch_rng'].cpu())
        torch.cuda.set_rng_state_all([x.cpu() for x in meta['cuda_rng']])
        random.setstate(meta['python_rng'])
        np.random.set_state(meta['numpy_rng'])
    else:
        save_json(run / 'manifest.json', contract)
    online = {'count': 0, 'loss': 0.}
    def capture(module, inputs, output):
        if not torch.isfinite(output[0]):
            raise FloatingPointError('Nonfinite loss')
        count = len(inputs[0]['graph'][1])
        online['count'] += count
        online['loss'] += float(output[0].detach()) * count
    engine.model.register_forward_hook(capture)
    start = time.monotonic()
    try:
        for epoch in range(initial + 1, 51):
            tick = time.monotonic()
            online.update(count=0, loss=0.)
            engine.train(num_epoch=1)
            if online['count'] != len(train):
                raise RuntimeError('Incomplete full epoch')
            metrics, records = evaluate(engine, valid, valid_mapping)
            improved = metrics['top-1 accuracy'] > best
            if improved:
                best, best_epoch = metrics['top-1 accuracy'], epoch
            record = {'epoch': epoch, 'valid': metrics, 'train_rows_seen': online['count'],
                      'train_loss': online['loss'] / online['count'], 'best_top1': best,
                      'best_epoch': best_epoch, 'epoch_seconds': time.monotonic() - tick,
                      'elapsed_seconds': time.monotonic() - start}
            meta = {'completed_epochs': epoch, 'best_top1': best, 'best_epoch': best_epoch,
                    'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                    'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
                    'record': record, 'contract': contract}
            engine.save(str(run / 'last.tmp.pth'), metadata=meta)
            os.replace(run / 'last.tmp.pth', run / 'last.pth')
            if improved:
                engine.save(str(run / 'best.tmp.pth'), metadata=meta)
                os.replace(run / 'best.tmp.pth', run / 'best.pth')
                save_json(run / 'valid_predictions.json', {'metrics': metrics, 'records': records, 'epoch': epoch})
            with (run / 'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(record) + '\n')
            save_json(run / 'status.json', {'status': 'running', 'pid': os.getpid(), **record})
            print('EPOCH_RESULT', json.dumps(record), flush=True)
        save_json(run / 'result.json', {'status': 'trained', 'best_top1': best, 'best_epoch': best_epoch,
                  'best_checkpoint_sha256': sha256(run / 'best.pth'), 'test_loaded': False})
        save_json(run / 'status.json', {'status': 'trained', 'completed_epochs': 50})
    except BaseException as exc:
        save_json(run / 'status.json', {'status': 'failed', 'error': repr(exc)})
        raise


if __name__ == '__main__':
    main()
