"""One full-data direct no-spectral seed. No automatic next seed or test evaluation."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import time
import numpy as np
from model_adapter import *
from runtime import load_split
from training_core import build_engine as build_full

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    identity=verify_extension()
    gate=json.loads(Path(__file__).with_name('SERVER_PREFLIGHT.json').read_text())
    assert gate['passed'] and not gate['cpu'] and gate['extension_sha256']==identity
    seed_all(0)
    train,_=load_split('full','train')
    valid,mapping=load_split('full','valid')
    # Replay the original full-data initialization order, including dataset
    # preprocessing, then restore the exact RNG state for the deletion arm.
    init_rng=(torch.get_rng_state(),torch.cuda.get_rng_state_all(),random.getstate(),np.random.get_state())
    reference=build_full('full_fspec',train,valid)
    paired=json.loads((ROOT/'runs/full/full_fspec_seed0/manifest.json').read_text())
    reference_digest=digest_state(reference.model)
    assert reference_digest==paired['initial_state_sha256'],(reference_digest,paired['initial_state_sha256'])
    reference_state={k:v.detach().cpu().clone() for k,v in reference.model.state_dict().items()}
    del reference
    torch.cuda.empty_cache()
    torch.set_rng_state(init_rng[0]);torch.cuda.set_rng_state_all(init_rng[1])
    random.setstate(init_rng[2]);np.random.set_state(init_rng[3])
    engine=build_engine(train,valid)
    shared=engine.model.state_dict()
    assert all(k in reference_state and torch.equal(v.cpu(),reference_state[k]) for k,v in shared.items())
    removed=sorted(set(reference_state)-set(shared))
    assert removed and all('frequency_encoder' in k for k in removed)
    save_json(HERE/'FULL_INITIALIZATION_VERIFIED.json',{'paired_full_initial_sha256':reference_digest,
        'no_spectral_initial_sha256':digest_state(engine.model),'shared_tensors':len(shared),
        'removed_tensors':removed,'passed':True})
    del reference_state
    contract={'scope':'full','arm':'no_spectral','seed':0,'epochs':50,'width':512,
              'layers':3,'batch_size':64,'optimizer':'Adam','lr':1e-4,'weight_decay':0,
              'dropout':0,'unknown_class':True,'train_rows':len(train),'valid_rows':len(valid),
              'test_loaded':False,'selection':'max validation Top1, earliest tie',
              'candidate_domain':'unchanged canonical bonds plus atoms','parameters':4557318,
              'initial_state_sha256':digest_state(engine.model),'extension_sha256':identity,
              'execution_sha256':sha256(ROOT/'EXECUTION_MANIFEST.json'),
              'pair_manifest_sha256':sha256(ROOT/'PAIR_MANIFEST.json'),'torch':torch.__version__}
    assert sum(p.numel() for p in engine.model.parameters())==4557318
    paired=json.loads((ROOT/'runs/full/full_fspec_seed0/manifest.json').read_text())
    for key in ('seed','epochs','width','layers','batch_size','optimizer','lr','weight_decay',
                'dropout','unknown_class','train_rows','valid_rows','selection','candidate_domain',
                'execution_sha256','pair_manifest_sha256','torch'):
        assert contract[key]==paired[key], key
    run=ROOT/'runs/full/no_spectral_seed0'
    run.mkdir(parents=True,exist_ok=args.resume)
    initial,best,best_epoch=0,-math.inf,0
    if args.resume:
        assert json.loads((run/'manifest.json').read_text())==contract
        state=torch.load(run/'last.pth',map_location=engine.device,weights_only=False)
        engine.model.load_state_dict(state['model'])
        engine.optimizer.load_state_dict(state['optimizer'])
        meta=state['metadata']
        initial,best,best_epoch=meta['completed_epochs'],meta['best_top1'],meta['best_epoch']
        engine.meter.epoch_id=initial
        engine.meter.epoch2batch=[0]*(initial+1)
        engine.meter.time=[time.time()]*(initial+1)
        torch.set_rng_state(meta['torch_rng'].cpu())
        torch.cuda.set_rng_state_all([x.cpu() for x in meta['cuda_rng']])
        random.setstate(meta['python_rng'])
        np.random.set_state(meta['numpy_rng'])
    else:
        save_json(run/'manifest.json',contract)
    online={'count':0,'loss':0.}
    def capture(module,inputs,output):
        if not torch.isfinite(output[0]): raise FloatingPointError('Nonfinite loss')
        count=len(inputs[0]['graph'][1])
        online['count']+=count
        online['loss']+=float(output[0].detach())*count
    engine.model.register_forward_hook(capture)
    save_json(run/'status.json',{'status':'running','pid':os.getpid(),'epoch':initial})
    print('TRAIN_START',json.dumps(contract),flush=True)
    start=time.monotonic()
    try:
        for epoch in range(initial+1,51):
            tick=time.monotonic()
            online.update(count=0,loss=0.)
            engine.train(num_epoch=1)
            assert online['count']==len(train)
            metrics,records=evaluate(engine,valid,mapping)
            improved=metrics['top-1 accuracy']>best
            if improved: best,best_epoch=metrics['top-1 accuracy'],epoch
            record={'epoch':epoch,'valid':metrics,'train_rows_seen':online['count'],
                    'train_loss':online['loss']/online['count'],'best_top1':best,'best_epoch':best_epoch,
                    'epoch_seconds':time.monotonic()-tick,'elapsed_seconds':time.monotonic()-start}
            meta={'completed_epochs':epoch,'best_top1':best,'best_epoch':best_epoch,
                  'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
                  'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),
                  'record':record,'contract':contract}
            engine.save(str(run/'last.tmp.pth'),metadata=meta)
            os.replace(run/'last.tmp.pth',run/'last.pth')
            if improved:
                engine.save(str(run/'best.tmp.pth'),metadata=meta)
                os.replace(run/'best.tmp.pth',run/'best.pth')
                save_json(run/'valid_predictions.json',{'metrics':metrics,'records':records,'epoch':epoch})
            with (run/'metrics.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
            save_json(run/'status.json',{'status':'running','pid':os.getpid(),**record})
            print('EPOCH_RESULT',json.dumps(record),flush=True)
        save_json(run/'result.json',{'status':'trained','best_top1':best,'best_epoch':best_epoch,
            'best_checkpoint_sha256':sha256(run/'best.pth'),'test_loaded':False})
        save_json(run/'status.json',{'status':'trained','completed_epochs':50})
    except BaseException as exc:
        save_json(run/'status.json',{'status':'failed','error':repr(exc)})
        raise

if __name__=='__main__': main()
