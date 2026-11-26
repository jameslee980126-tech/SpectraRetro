"""Minimal sampler whitelist; no reaction class or supervised geometry targets."""
import torch
from unknown_model import OLD
from src import const
from src.datasets import collate
def collate_unknown(items,probabilities=None):
    allowed=set(const.DATA_LIST_ATTRS)|set(const.DATA_ATTRS_TO_PAD)
    batch=collate([{k:v for k,v in x.items() if k in allowed} for x in items])
    if probabilities is not None:
        if len(probabilities)!=len(items):raise ValueError('Prior/row count mismatch')
        batch['composition_probabilities']=torch.as_tensor(probabilities,dtype=torch.float32)
    return batch
def inference_inputs(batch):
    allowed={'uuid','positions','one_hot','charges','anchors','fragment_mask','linker_mask','atom_mask','edge_mask','num_atoms','composition_probabilities'}
    safe={k:v for k,v in batch.items() if k in allowed}
    for key in ['positions','one_hot','charges']:
        if key in safe:
            mask=safe['fragment_mask']
            while mask.ndim>safe[key].ndim:mask=mask.squeeze(-1)
            safe[key]=torch.where(mask.bool(),safe[key],torch.zeros_like(safe[key]))
    return safe
