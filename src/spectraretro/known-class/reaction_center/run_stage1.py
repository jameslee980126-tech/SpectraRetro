"""Train and evaluate known-class FSpec reaction-center identification."""
import argparse
import json
import os
from pathlib import Path
import random
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from spectraretro.reaction_center import training as common
from spectraretro.reaction_center.features import BOND_COLUMNS, NODE_COLUMNS, SCHEMA
from spectraretro.reaction_center.known_class.prior_model import build_task
import numpy as np
import torch
from rdkit import rdBase
from torchdrug import data


def source_manifest():
    return common.source_manifest(Path(__file__).resolve().parent)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', required=True, choices=('baseline', 'prior'))
    for name in ('train', 'valid', 'test'):
        p.add_argument('--' + name + '-csv', type=Path, default=common.REPOSITORY / 'datasets/full' / (name + '.csv'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--width', type=int, default=512)
    p.add_argument('--layers', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--preflight-only', action='store_true')
    p.add_argument('--preflight-batch-policy', choices=('first', 'largest'), default='largest')
    p.add_argument('--expected-common-sha256')
    args = p.parse_args()
    if min(args.epochs, args.patience, args.width, args.layers, args.batch_size) <= 0 or args.epochs > 100:
        p.error('Require positive settings and at most 100 epochs')
    if len({q.resolve() for q in (args.train_csv, args.valid_csv, args.test_csv)}) != 3:
        p.error('Train, valid and test must be distinct files')
    if args.device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA requested but unavailable')
    args.output.mkdir(parents=True, exist_ok=args.resume)
    device = torch.device(args.device)
    sources = source_manifest()
    common.seed_all(args.seed)
    train, train_rows, train_index, train_report = common.load_split(args.train_csv, sources)
    valid, valid_rows, valid_index, valid_report = common.load_split(args.valid_csv, sources)
    common.seed_all(args.seed)  # Dataset construction/cache must not change initialization.
    task = build_task(train, valid, prior=args.arm == 'prior', width=args.width, num_layers=args.layers).to(device)
    common_sha = common.common_state_hash(task)
    if args.expected_common_sha256 and common_sha != args.expected_common_sha256:
        raise RuntimeError('Paired common parameter initialization mismatch')
    optimizer = torch.optim.Adam(task.parameters(), lr=args.lr, weight_decay=0.)
    contract = dict(arm=args.arm, known_class=True, backbone='FSpecFaithfulReactionCenter', dual_branch=False,
                    seed=args.seed, epochs=args.epochs, patience=args.patience, width=args.width, layers=args.layers,
                    batch_size=args.batch_size, lr=args.lr, dropout=0., candidate_domain='canonical bonds plus atoms',
                    selection='max valid top-1; earliest tie', common_initial_sha256=common_sha,
                    descriptor_schema=SCHEMA, node_columns=NODE_COLUMNS, bond_columns=BOND_COLUMNS,
                    test_loaded_for_selection=False, sources=sources,
                    data={name: common.sha(getattr(args, name + '_csv')) for name in ('train', 'valid', 'test')},
                    indices={name: common.sha(getattr(args, name + '_csv').with_name(name + '_index.csv'))
                             for name in ('train', 'valid', 'test')
                             if getattr(args, name + '_csv').with_name(name + '_index.csv').exists()},
                    train=train_report, valid=valid_report, parameters=sum(p.numel() for p in task.parameters()),
                    device=str(device), torch=torch.__version__, numpy=np.__version__, rdkit=rdBase.rdkitVersion)
    # JSON round-trip normalizes tuple schema columns for resume comparison.
    contract = json.loads(json.dumps(contract))
    if args.preflight_only:
        indices = list(range(len(train)))
        if args.preflight_batch_policy == 'largest':
            indices.sort(key=lambda i: int(train.data[i][1].num_node), reverse=True)
        selected = indices[:args.batch_size]
        molecules = [train[i] for i in selected]
        batch = common.move(next(iter(data.DataLoader(molecules, len(molecules), shuffle=False))), device)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        loss, _ = task(batch)
        loss.backward()
        if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in task.parameters()):
            raise FloatingPointError('Preflight nonfinite loss/gradient')
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        common.save_json(args.output / 'preflight.json', dict(status='passed', loss=float(loss.detach()), contract=contract,
            optimizer_steps=0, policy=args.preflight_batch_policy, selected_train_row_indices=selected,
            node_counts=[int(train.data[i][1].num_node) for i in selected],
            forward_backward_seconds=time.perf_counter() - started,
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0,
            cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else 0))
        return
    completed, best, best_epoch, stale = 0, -float('inf'), 0, 0
    if args.resume:
        saved_contract = json.loads((args.output / 'contract.json').read_text())
        if saved_contract != contract:
            raise RuntimeError('Resume contract differs')
        state = common.torch_load(args.output / ('last.pth' if (args.output / 'last.pth').exists() else 'best.pth'))
        if state['contract'] != contract:
            raise RuntimeError('Checkpoint contract differs')
        task.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        completed, best, best_epoch, stale = (state[k] for k in ('epoch', 'best', 'best_epoch', 'stale'))
        random.setstate(state['python_rng'])
        np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng'])
        if state['cuda_rng']:
            torch.cuda.set_rng_state_all(state['cuda_rng'])
    else:
        common.save_json(args.output / 'contract.json', contract)
    common.save_json(args.output / 'status.json', dict(status='training', pid=os.getpid(), epoch=completed))
    for epoch in range(completed + 1, args.epochs + 1):
        if stale >= args.patience:
            break
        task.train()
        start = time.perf_counter()
        seen, loss_sum = 0, 0.
        for batch_index, batch in enumerate(data.DataLoader(train, args.batch_size, shuffle=True, num_workers=0), 1):
            batch = common.move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = task(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            for parameter in task.parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError('Nonfinite training gradient')
            optimizer.step()
            count = len(batch['reaction'])
            seen += count
            loss_sum += float(loss.detach()) * count
            if batch_index == 1 or batch_index % 100 == 0:
                progress = dict(status='training', pid=os.getpid(), epoch=epoch, batch=batch_index,
                                train_rows_seen=seen, train_rows_total=len(train),
                                train_loss_running=loss_sum / seen, best=best if np.isfinite(best) else None,
                                best_epoch=best_epoch, epoch_elapsed_seconds=time.perf_counter() - start)
                common.save_json(args.output / 'status.json', progress)
                print(json.dumps(progress), flush=True)
        if seen != len(train):
            raise RuntimeError('Epoch did not consume every frozen training row')
        metrics = common.evaluate(task, valid, valid_rows, valid_index, args.batch_size, device)
        improved = metrics['overall']['top-1 accuracy'] > best
        if improved:
            best, best_epoch, stale = metrics['overall']['top-1 accuracy'], epoch, 0
        else:
            stale += 1
        state = common.checkpoint(task, optimizer, epoch, best, best_epoch, stale, contract)
        if improved:
            common.save_checkpoint(args.output / 'best.pth', state)
        common.save_checkpoint(args.output / 'last.pth', state)
        log = dict(epoch=epoch, train_loss=loss_sum / seen, train_rows_seen=seen,
                   valid=metrics, best=best, best_epoch=best_epoch, stale=stale,
                   epoch_seconds=time.perf_counter() - start)
        with (args.output / 'metrics.jsonl').open('a', encoding='utf8') as stream:
            stream.write(json.dumps(log) + '\n')
        common.save_json(args.output / 'status.json', dict(status='training', pid=os.getpid(), **log))
        print(json.dumps(log), flush=True)
    # Commit model selection before reading any test reaction or target.
    freeze = dict(status='frozen', best_epoch=best_epoch, valid_top1=best,
                  best_sha256=common.sha(args.output / 'best.pth'), test_loaded=False)
    common.save_json(args.output / 'selection_freeze.json', freeze)
    state = common.torch_load(args.output / 'best.pth')
    task.load_state_dict(state['model'], strict=True)
    valid_metrics = common.evaluate(task, valid, valid_rows, valid_index, args.batch_size, device,
                             args.output / 'valid_predictions.jsonl')
    if (common.ROOT / 'TRAIN_ONLY').exists():
        common.save_json(args.output / 'status.json', dict(status='completed_train_validation_only', best_epoch=best_epoch, completed_epochs=epoch, test_loaded=False))
        return
    test, test_rows, test_index, test_report = common.load_split(args.test_csv, sources)
    test_metrics = common.evaluate(task, test, test_rows, test_index, args.batch_size, device,
                            args.output / 'test_predictions.jsonl')
    result = dict(status='completed', arm=args.arm, best_epoch=best_epoch, valid=valid_metrics,
                  test=test_metrics, test_descriptor_coverage=test_report,
                  common_initial_sha256=common_sha, interpretation='single-seed exploratory high/medium/low similarity pilot')
    common.save_json(args.output / 'result.json', result)
    common.save_json(args.output / 'status.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
