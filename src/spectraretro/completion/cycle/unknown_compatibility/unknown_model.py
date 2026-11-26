"""Class-free chemical prior applied inside every denoising step, no reranking."""
from contextlib import contextmanager
from pathlib import Path
import sys
import torch
from torch import nn
HERE=Path(__file__).resolve().parent
import os
OLD=Path(os.environ['RETRO_STAGE2_OLD_ROOT']).resolve()
sys.path.insert(0,str(OLD/'stage2'))
from prior_model import PriorDDPM

class CompatibilityAdapter(nn.Module):
    def __init__(self,width=128,scale=.1):
        super().__init__();self.scale=scale
        self.encode=nn.Sequential(nn.Linear(26,64),nn.SiLU(),nn.Linear(64,width))
        self.gate=nn.Sequential(nn.Linear(width+1,64),nn.SiLU(),nn.Linear(64,1),nn.Sigmoid())
        nn.init.zeros_(self.encode[-1].weight);nn.init.zeros_(self.encode[-1].bias)
    def forward(self,hidden,probabilities,time,unknown_mask):
        lp=probabilities.clamp_min(1e-8).log();lp=lp-lp.mean(-1,keepdim=True)
        residual=self.scale*torch.tanh(self.encode(lp))
        return hidden+self.gate(torch.cat([hidden,time],-1))*residual[:,None,:]*unknown_mask

class UnknownCompatibilityDDPM(PriorDDPM):
    def __init__(self,compatibility_enabled=False,**kwargs):
        kwargs['prior_enabled']=False
        super().__init__(**kwargs)
        self.compatibility_enabled=bool(compatibility_enabled)
        self._composition=None;self._step=None;self._capture_enabled=False;self._captured=None
        with torch.random.fork_rng(devices=[]):
            self.edm.compatibility_adapter=CompatibilityAdapter(kwargs['hidden_nf'])
        dynamics=self.edm.dynamics;original=dynamics.forward
        def forward(t,xh,node_mask,linker_mask,edge_mask,context):
            previous=self._step;self._step=(t,xh,linker_mask)
            try:epsilon=original(t,xh,node_mask,linker_mask,edge_mask,context)
            finally:self._step=previous
            if self._capture_enabled:self._captured=(t,xh,epsilon)
            return epsilon
        dynamics.forward=forward
        dynamics.base.dynamics.embedding.register_forward_hook(self._embedding_hook)
    def _embedding_hook(self,module,inputs,hidden):
        if not self.compatibility_enabled:return hidden
        if self._composition is None or self._step is None:raise RuntimeError('Unscoped compatibility prior')
        t,xh,unknown=self._step;b,n=xh.shape[:2]
        tt=t.reshape(-1,1,1)
        if tt.shape[0]==1:tt=tt.expand(b,1,1)
        tt=tt.expand(b,n,1)
        h=self.edm.compatibility_adapter(hidden.reshape(b,n,-1),self._composition.to(hidden),tt,unknown)
        return h.reshape_as(hidden)
    @contextmanager
    def condition(self,data):
        if {'reaction_class','class_index','class_id','class_metadata_only'} & set(data):
            raise ValueError('Reaction class is forbidden in unknown-class model input')
        old=self._composition
        if self.compatibility_enabled:
            p=data.get('composition_probabilities')
            if p is None or p.ndim!=2 or p.shape[-1]!=26 or not torch.isfinite(p).all() or torch.any(p<0):
                raise ValueError('Need observable-context probabilities [B,26]')
            if not torch.allclose(p.sum(-1),torch.ones_like(p[:,0]),atol=1e-5,rtol=0):raise ValueError('Probabilities must sum to1')
            self._composition=p
        try:yield
        finally:self._composition=old
    def forward(self,data,training):
        with self.condition(data):return super().forward(data,training)
    def sample_chain(self,data,sample_fn=None,keep_frames=None):
        if sample_fn is None:raise ValueError('Predicted-size sampler is required')
        with self.condition(data):return super().sample_chain(data,sample_fn,keep_frames)
    def forward_with_clean_prediction(self,data,training=True,max_time=.5):
        """Low-noise x0 for later train-only losses; avoids inverse-alpha high-noise overflow."""
        self._capture_enabled=True;self._captured=None
        try:base=self.forward(data,training)
        finally:self._capture_enabled=False
        t,z,eps=self._captured;self._captured=None
        time=t.reshape(-1)
        if time.numel()==1:time=time.expand(z.shape[0])
        active=time<=max_time;zero=z.reshape(-1)[:0].sum()+eps.reshape(-1)[:0].sum()
        clean=torch.zeros_like(z)+zero;alpha=z.new_zeros(z.shape[0])
        if active.any():
            gamma=self.edm.inflate_batch_array(self.edm.gamma(t[active]),z[active])
            aa=self.edm.alpha(gamma,z[active]);ss=self.edm.sigma(gamma,z[active])
            pred=(z[active]-ss*eps[active]*data['linker_mask'][active])/aa
            clean[active]=pred*data['linker_mask'][active]+z[active]*data['fragment_mask'][active]
            alpha[active]=aa.reshape(-1)
        xyz,h=self.edm.unnormalize(clean[:,:,:3],clean[:,:,3:])
        return base,dict(coordinates=xyz,features=h,time=time,alpha=alpha,active=active)
