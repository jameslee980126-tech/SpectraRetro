"""Stream native DiffER samples in original batch/source/draw order; no reference-based ranking."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from model_runtime import ROOT, build, dataset, loader, seed_all
from source.utils import move_batch_to_gpu, repeat_batch

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['known','unknown'],required=True);ap.add_argument('--smoke',action='store_true');ap.add_argument('--checkpoint',type=Path,required=True)
    args=ap.parse_args();out=ROOT/('smoke_samples' if args.smoke else 'samples')/args.mode
    out=out.resolve()
    if ROOT not in out.parents:
        raise ValueError('Sampling output escaped the prepared workspace')
    out.mkdir(parents=True,exist_ok=False)
    protocol,tok,model,opt,sampler,trainer,initial=build(args.mode,out)
    checkpoint=args.checkpoint.resolve()
    checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    model.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=True));model.eval()
    ds=dataset(tok,args.mode,'test')
    assert len(ds)==9250*20
    augmentations=protocol['test_augmentation']
    samples_per_input=protocol['samples_per_test_augmentation']
    microbatch=protocol['inference_base_microbatch']
    selected=[i for i in range(len(ds)) if i%20<augmentations]
    ds.products=[ds.products[i] for i in selected]
    ds.reactants=[ds.reactants[i] for i in selected]
    assert len(ds)==9250*augmentations
    if args.smoke:
        ds.products=ds.products[:16];ds.reactants=ds.reactants[:16]
    # The native evaluator uses shuffled augmented inputs and sample-major expansion.
    seed_all(1998);dl=loader(ds,batch=microbatch,shuffle=True)
    started=time.time();deadline=started+protocol['budget_seconds']['inference_per_mode']
    draws=0
    with (out/'native_samples_batches.jsonl').open('w') as f, torch.no_grad():
        for i,batch in enumerate(dl):
            if time.time()>=deadline:raise TimeoutError('Independent sampling invocation reached its stage time limit')
            sources=list(batch['encoder_smiles']);targets=list(batch['target_smiles'])
            n=len(sources)
            move_batch_to_gpu(batch);repeat_batch(batch,samples_per_input)
            samples,_=sampler.sample(batch,model,verbose=False,pred_lengths=True,clean=False)
            assert len(samples)==n*samples_per_input
            f.write(json.dumps({'batch':i,'sources':sources,'reference_targets':targets,'samples':samples})+'\n')
            f.flush();draws+=len(samples)
            progress={'phase':'test_inference','batch':i+1,'batches':len(dl),'augmented_inputs':min((i+1)*microbatch,len(ds)),
                      'total_augmented_inputs':len(ds),'raw_draws':draws,'time':time.time()}
            temp=out/'inference_progress.json.tmp';temp.write_text(json.dumps(progress,indent=2)+'\n');temp.replace(out/'inference_progress.json')
            if i%25==0 or i+1==len(dl):print(json.dumps(progress),flush=True)
    receipt={'passed':True,'mode':args.mode,'raw_draws':draws,'augmented_inputs':len(ds),'samples_per_input':samples_per_input,
             'augmentations':augmentations,'candidates_per_condition':augmentations*samples_per_input,'base_microbatch':microbatch,'precision':'native float32',
             'predicted_lengths':True,'reference_targets_used_for_generation':False,
             'raw_source_order':'native shuffled loader, sample-major repeated batches','sampling_seed':1998,
             'best_model_sha256':checkpoint_sha256,'started_at':started,'finished_at':time.time(),'deadline':deadline,
             'peak_cuda_bytes':torch.cuda.max_memory_allocated()}
    (out/'sampling_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
