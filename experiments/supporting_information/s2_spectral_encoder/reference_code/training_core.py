"""Audited helper functions; no training runs on import."""
import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from runtime import ROOT, activate, save_json, sha256
activate()
import numpy as np
import torch
from torchdrug import core, data, tasks, utils
from torchdrug.layers import functional
from train import load_stage1_dataset
from local_pair import make_model

ARMS = ('full_fspec','local_pair')

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def digest_state(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def build_engine(arm, train, valid, width=512, batch_size=64, cpu=False):
    model = make_model(arm,train,width)
    task = tasks.CenterIdentification(model,None,feature=('graph','atom','bond'),
        mlp_hidden_dim=width,fusion_mode='raw')
    optimizer = torch.optim.Adam(task.parameters(),lr=1e-4,weight_decay=0.)
    return core.Engine(task,train,valid,[],optimizer,gpus=None if cpu else [0],batch_size=batch_size,num_worker=0,log_interval=100)

@torch.no_grad()
def evaluate(engine, dataset, mapping=None):
    state = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    was_training = engine.model.training
    try:
        engine.model.eval()
        sampler = torch.utils.data.DistributedSampler(dataset,num_replicas=1,rank=0,seed=0)
        order = list(iter(sampler))
        preds,targets=[],[]
        for batch in data.DataLoader(dataset,engine.batch_size,sampler=sampler,num_workers=0):
            batch=utils.cuda(batch,device=engine.device) if engine.device.type=='cuda' else batch
            pred,target=engine.model.predict_and_target(batch)
            if not torch.isfinite(pred).all():
                raise FloatingPointError('Nonfinite evaluation scores')
            preds.append(pred)
            targets.append(target)
        pred,target=utils.cat(preds),utils.cat(targets)
        metrics={k:float(v) for k,v in engine.model.evaluate(pred,target).items()}
        labels,sizes=target
        gold=functional.variadic_max(labels,sizes)[1]
        metrics['cross_entropy']=float(functional.variadic_cross_entropy(pred,gold,sizes))
        top=functional.variadic_topk(pred,sizes,10,largest=True)[1].cpu().tolist()
        records=[]
        offset=0
        for local,candidates,g,size in zip(order,top,gold.cpu().tolist(),sizes.cpu().tolist()):
            meta = mapping[local] if mapping is not None else {'dataset_index':local}
            if mapping is not None and (g!=meta['gold_candidate'] or size!=meta['candidate_count']):
                raise RuntimeError('Prediction / source candidate alignment failed')
            rank=candidates.index(g)+1 if g in candidates else None
            scores=pred[offset:offset+size]
            records.append({**{k:v for k,v in meta.items() if k not in ('product_smiles','bond_atom_maps')}, **{
                'gold_candidate':g,'candidate_count':size,'top10_candidates':candidates,
                'rank_within_top10':rank,'gold_score':float(scores[g]),
                'top1_score':float(scores[candidates[0]]),'candidate_scores':scores.detach().cpu().tolist()}})
            offset+=size
        for k in (1,2,3,5,10):
            measured=sum(r['rank_within_top10'] is not None and r['rank_within_top10']<=k for r in records)/len(records)
            if abs(measured-metrics[f'top-{k} accuracy'])>1e-6:
                raise RuntimeError('Per-record ranking differs from unchanged task metric')
        return metrics,sorted(records,key=lambda r:r['dataset_index'])
    finally:
        engine.model.train(was_training)
        torch.set_rng_state(state[0])
        torch.cuda.set_rng_state_all(state[1])
