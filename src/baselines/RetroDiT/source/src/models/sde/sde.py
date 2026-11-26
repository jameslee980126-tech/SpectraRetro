import warnings
import torch.nn as nn
from torch import Tensor

from typing import Literal, Optional
from abc import ABC, abstractmethod
from functools import wraps

from src.data.batch_class import *

__all__ = ['SDE']

class SDE(ABC):
    def __init__(
            self,
            val_sampling_steps: int, test_sampling_steps: int,
            **kwargs
        ):
        self.cond_on = ['cond', 'p_X', 'p_E', 'p_mask']

        self.val_sampling_steps = val_sampling_steps
        self.test_sampling_steps = test_sampling_steps

        

    def extend_like(self, src: Tensor, out: Tensor) -> Tensor:
        return src.view(src.size(0), *[1] * (out.ndim - 1))
    
    @abstractmethod
    def __call__(
            self,
            batch: dict[str, Tensor],
            t: Tensor | None = None
        ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        pass
    
    @abstractmethod
    def loss(
            self,
            t: Tensor,
            logits: Tensor,
            target: Tensor,
            loss_mask: Tensor,
            **kwargs
        ):
        pass

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        
        original_sampling = cls.__dict__['sampling']
        @wraps(original_sampling)
        def wrapped_sampling(self, net, batch, **kwargs):
            safe_batch = self.safe_sampling_batch(batch)
            return original_sampling(self, net, safe_batch, **kwargs)
        setattr(cls, 'sampling', wrapped_sampling)


    def safe_sampling_batch(
            self, batch: GraphBatch | GraphSamplingBatch
        ) -> GraphSamplingBatch: 
        return batch.to_sampling()

    @abstractmethod
    def sampling(
            self,
            net: nn.Module,
            batch: GraphBatch,
            **kwargs
        ) -> GraphBatch:
        pass


