"""Native train_step/val_step, full validation, validation-only checkpointing and finite early stop."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import Subset
from model_runtime import ROOT, build, dataset, loader, rng_state, restore_rng, seed_all

def atomic(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n'); temp.replace(path)

def save(path, value):
    temp = path.with_suffix(path.suffix + '.tmp'); torch.save(value, temp); temp.replace(path)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--mode', choices=['known','unknown'], required=True); ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    out = ROOT / ('smoke' if args.smoke else 'runs') / args.mode
    out.mkdir(parents=True, exist_ok=True)
    assert not (out/'training_events.jsonl').exists() and not (out/'best.pkl').exists(), 'Refusing to overwrite training evidence'
    protocol, tok, model, opt, sampler, trainer, initial = build(args.mode, out)
    started = time.time()
    deadline = min(started + 64800, protocol['global_deadline'])
    def event(kind, **kw):
        value = {'event':kind, 'time':time.time(), **kw}
        with (out/'training_events.jsonl').open('a') as f: f.write(json.dumps(value)+'\n')
        atomic(out/'progress.json', value)
        print(json.dumps(value), flush=True)
    event('random_initialization', mode=args.mode, seed=0, restored_checkpoint=False, initial_state_sha256=initial,
          parameters=sum(p.numel() for p in model.parameters()), test_data_loaded=False, smoke=args.smoke)
    train = dataset(tok, args.mode, 'train')
    valid = dataset(tok, args.mode, 'val', canonical_only=True)
    assert len(train) == 74146*20 and len(valid) == 9395
    if args.smoke:
        # Stress the longest training sequences, plus actual validation sampling.
        lengths = [len(s.split()) + len(t.split()) for s,t in zip(train.products,train.reactants)]
        indices = sorted(range(len(lengths)), key=lengths.__getitem__, reverse=True)[:64]
        train = Subset(train, indices); train.collate_fn = train.dataset.collate_fn
        valid = Subset(valid, list(range(8))); valid.collate_fn = valid.dataset.collate_fn
    tl, vl = loader(train, shuffle=True), loader(valid)
    best, reference, bad, best_epoch = -1.0, -1.0, 0, None
    max_epochs = 1 if args.smoke else 30
    for epoch in range(1, max_epochs+1):
        losses, examples = [], 0
        for i, batch in enumerate(tl, 1):
            if time.time() >= deadline: raise TimeoutError('Frozen training/overall time budget exhausted')
            n = len(batch['encoder_smiles'])
            loss = float(trainer.train_step(batch))
            assert math.isfinite(loss), loss
            losses.append(loss); examples += n
            if i == 1 or i % 100 == 0 or i == len(tl):
                event('training_batch', epoch=epoch, batch=i, batches=len(tl), examples=examples, loss=loss)
        event('training_epoch_complete', epoch=epoch, examples=examples, batches=len(tl), mean_batch_loss=math.fsum(losses)/len(losses))
        prior_rng = rng_state(); seed_all(1998)
        correct, total, val_losses = 0, 0, []
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(vl, 1):
                if time.time() >= deadline: raise TimeoutError('Frozen validation/training budget exhausted')
                metrics, samples = trainer.val_step(batch, pred_lengths=True)
                n = len(samples)
                correct += round(float(metrics['mol_accuracy']) * n); total += n
                val_losses.append(float(metrics['val_loss']))
                if i == 1 or i % 25 == 0 or i == len(vl):
                    event('validation_batch', epoch=epoch, batch=i, batches=len(vl), examples=total, correct=correct)
        restore_rng(prior_rng)
        assert total == (8 if args.smoke else 9395)
        accuracy = correct/total
        improved = accuracy >= reference+0.01
        if improved: reference, bad = accuracy, 0
        else: bad += 1
        if accuracy > best:
            best, best_epoch = accuracy, epoch
            save(out/'best.pkl', model.state_dict())
        early_stop = epoch >= 5 and bad >= 4
        save(out/'last_checkpoint.pt', {'model':model.state_dict(), 'optimizer':opt.state_dict(), 'rng':rng_state(),
                 'epoch':epoch, 'best_validation':best, 'best_epoch':best_epoch, 'plateau_reference':reference, 'bad_events':bad})
        event('validation_complete', epoch=epoch, examples=total, correct=correct, accuracy=accuracy,
              best_validation=best, selected_epoch=best_epoch, plateau_reference=reference, bad_events=bad,
              early_stop=early_stop, mean_batch_val_loss=math.fsum(val_losses)/len(val_losses))
        if early_stop: break
    receipt = {'passed':True, 'mode':args.mode, 'seed':0, 'seed_applied':True, 'smoke':args.smoke,
               'epochs':epoch, 'best_epoch':best_epoch, 'best_validation':best, 'stop':'early_stop' if early_stop else 'max_epochs',
               'best_model_sha256':hashlib.sha256((out/'best.pkl').read_bytes()).hexdigest(),
               'initial_state_sha256':initial, 'test_data_loaded':False, 'training_rows_per_epoch':examples,
               'validation_rows_per_epoch':total, 'started_at':started, 'finished_at':time.time(), 'deadline':deadline,
               'peak_cuda_bytes':torch.cuda.max_memory_allocated()}
    atomic(out/'training_receipt.json', receipt)
    event('training_complete', **receipt)

if __name__ == '__main__': main()
