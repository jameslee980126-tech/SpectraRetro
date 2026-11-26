"""Independent bounded exact-case training; all source fragment associations retained."""
import sys,os,time,random,json,traceback,importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import psutil
from run_last_epoch_diagnostic import REPO,ROOT,read,write,sha
HERE=REPO/'experiments/case_all_exact_memory_20261012'
TRAINER=REPO/'experiments/external_case_retrain_20261009/training/run.py'

def runtime_for(component,output,device='cuda:0'):
    p=read(HERE/'protocol.json');cfg=p['components'][component]
    for path,digest in p['source_sha256'].items():assert sha(path)==digest,path
    args=SimpleNamespace(component=component,mode='train',init_mode='warm_start',device=device,
        parent_checkpoint=cfg['parent']['path'],parent_sha256=cfg['parent']['sha256'],
        lr=cfg['lr'],batch_size=p['batch_size'],seed=p['seed'],threads=4)
    spec=importlib.util.spec_from_file_location('exact_case_frozen_trainer',TRAINER)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    contract=m.data_contract(HERE/'data_manifest.json',component)
    rt=m.stage2(args,contract,output);initial=m.initialize(torch,rt['model'],component,rt['architecture'],args)
    rt['model'].to(device)
    return p,cfg,m,rt,initial

def run(component):
    out=HERE/'training'/component;out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();state=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),
        component=component,state='loading',completed_epochs=0,optimizer_steps=0,test_read=False)
    def status(**v):state.update(v,updated_unix=time.time());write(out/'status.json',state)
    status();torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    try:
        seed=read(HERE/'protocol.json')['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        p,cfg,tr,rt,initial=runtime_for(component,out)
        model=rt['model'];optimizer=model.configure_optimizers();tr.validate_fresh_optimizer(optimizer,model)
        write(out/'manifest.json',dict(initialization=initial,architecture=rt['architecture'],train_rows=len(rt['train']),
            eval_rows=len(rt['valid']),data_check=rt['size_label_audit'],protocol_sha256=sha(HERE/'protocol.json')))
        train=rt['loader'](rt['train'],True);valid=rt['loader'](rt['valid'],False)
        exposure={r['uuid']:0 for r in rt['train']}
        def checkpoint(epoch,metric):
            obj={rt['state_key']:model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,
                'hyper_parameters':dict(model.hparams),'metadata':dict(run_signature_payload=rt['architecture'],completed_epochs=epoch,metrics=metric),
                'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),
                'cuda_rng':torch.cuda.get_rng_state_all(),'exposure':exposure}
            model.on_save_checkpoint(obj);temp=out/'last.ckpt.tmp';torch.save(obj,temp);os.replace(temp,out/'last.ckpt')
        def validate(epoch):
            status(state='validating',epoch_in_progress=epoch);model.eval();total=seen=correct=0
            with torch.no_grad(),torch.random.fork_rng(devices=[0]):
                torch.manual_seed(p['validation_seed'])
                for batch in valid:
                    batch=rt['move'](batch,torch.device('cuda:0'));loss,metrics=rt['loss'](batch,False)
                    assert torch.isfinite(loss);n=rt['count'](batch);total+=float(loss)*n;seen+=n
                    correct+=metrics.get('top-1 accuracy',0)*n
            assert seen==len(rt['valid'])
            return dict(epoch=epoch,eval_loss=total/seen,eval_top1=correct/seen if component=='size' else None,eval_rows=seen)
        baseline=validate(0);write(out/'parent_validation.json',baseline)
        for epoch in range(1,cfg['epochs']+1):
            model.train();status(state='training',epoch_in_progress=epoch);total=seen=0
            for repeat in range(p['passes_per_epoch']):
                for batch in train:
                    if time.monotonic()-start>cfg['max_seconds'] or (HERE/'STOP').exists():
                        checkpoint(epoch-1,dict(partial_epoch=epoch));status(state='stopped_budget_or_user');return
                    batch=rt['move'](batch,torch.device('cuda:0'));optimizer.zero_grad(set_to_none=True)
                    loss,_=rt['loss'](batch,True);assert torch.isfinite(loss);loss.backward()
                    assert all(q.grad is None or torch.isfinite(q.grad).all() for q in model.parameters())
                    optimizer.step();state['optimizer_steps']+=1
                    for u in batch['uuid']:exposure[u]+=1
                    n=rt['count'](batch);seen+=n;total+=float(loss.detach())*n
            assert seen==len(rt['train'])*p['passes_per_epoch']
            metric=dict(epoch=epoch,train_loss=total/seen,train_exposures=seen,seconds=time.monotonic()-start)
            if epoch==cfg['epochs']:metric.update(validate(epoch))
            checkpoint(epoch,metric);state['completed_epochs']=epoch
            with (out/'metrics.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(metric)+'\n')
            status(state='epoch_complete',**metric);print(json.dumps(metric),flush=True)
        write(out/'exposure.json',exposure);status(state='complete',last_sha256=sha(out/'last.ckpt'))
    except BaseException as e:
        status(state='failed',error=str(e),traceback=traceback.format_exc());raise

if __name__=='__main__':run(sys.argv[1])
