"""A new shared composition prior, trained across all six classes without labels."""
import torch
from torch import nn
class CompositionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.network=nn.Sequential(nn.Linear(3616,256),nn.SiLU(),nn.Linear(256,128),nn.SiLU(),nn.Linear(128,26))
    def forward(self,features):
        if features.ndim!=2 or features.shape[-1]!=3616:raise ValueError('Expected observable context [B,3616]')
        return self.network(features)
