"""Same SizeGNN with selectable legacy/masked readout and explicit label support."""
import torch
from torch import nn
from runtime import ROOT
from src.linker_size import SizeGNN
from src.egnn import coord2diff

LEGACY_SUPPORT=[0,1,2,3,4,5,6,7,8,9,10,13,19,20]

class SizeModel(nn.Module):
    def __init__(self, support, pooling='masked', normalization='batch_norm'):
        super().__init__()
        self.support=list(support);self.pooling=pooling
        self.gnn=SizeGNN(26,128,len(support),3,normalization,device='cpu')

    def forward(self,batch):
        h,x,m=batch['one_hot'],batch['positions'],batch['fragment_mask']
        bs,nodes=x.shape[:2]
        x=(x*m).reshape(-1,3);h=(h*m).reshape(-1,26)
        flat=m.reshape(-1,1)
        distances,_=coord2diff(x,batch['edges'])
        edge_mask=(batch['edge_mask'].bool() & (distances<6)).long()
        out=self.gnn(h,batch['edges'],distances,flat,edge_mask).reshape(bs,nodes,-1)
        if self.pooling=='legacy':return out.mean(1)
        return (out*m).sum(1)/m.sum(1).clamp_min(1)

    def labels(self, sizes):
        lookup={s:i for i,s in enumerate(self.support)}
        # Only legacy control retains the historical unsupported -> 20 mapping.
        if self.pooling=='legacy':
            return torch.tensor([lookup.get(int(s),lookup[20]) for s in sizes],device=sizes.device)
        if any(int(s) not in lookup for s in sizes):
            raise ValueError('Unsupported size; do not silently merge labels')
        return torch.tensor([lookup[int(s)] for s in sizes],device=sizes.device)

def collate(records):
    """Vectorized equivalent of vendor collate_with_fragment_edges."""
    out={k:torch.nn.utils.rnn.pad_sequence([r[k] for r in records],batch_first=True)
         for k in ('one_hot','positions','fragment_mask','linker_mask')}
    out['uuid']=[str(r['uuid']) for r in records]
    out['sizes']=out['linker_mask'].sum(1).long()
    m=out['fragment_mask'];bs,n=m.shape
    edges=m[:,:,None]*m[:,None,:]*(1-torch.eye(n))[None]
    out['edge_mask']=edges.reshape(-1,1)
    index=torch.arange(bs*n).reshape(bs,n)
    out['edges']=[index[:,:,None].expand(bs,n,n).reshape(-1),index[:,None,:].expand(bs,n,n).reshape(-1)]
    out['fragment_mask']=m[:,:,None]
    return out

def move(batch,device):
    return {k:([x.to(device) for x in v] if k=='edges' else v.to(device) if torch.is_tensor(v) else v)
            for k,v in batch.items()}
