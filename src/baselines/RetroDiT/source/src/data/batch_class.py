import warnings

from typing import Callable, Optional
from dataclasses import dataclass, fields, field

from copy import deepcopy

import torch

from torch import Tensor


__all__ = [
    'DenseGraph',
    'BaseBatchClass',
    'GraphSamplingBatch', 'GraphBatch',
]


class DenseGraph:
    def __init__(self, X, E, y=None):
        self.X = X
        self.E = E
        self.y = y if y is not None else torch.zeros(size=(self.X.shape[0], 0), dtype=torch.float, device=X.device)

    def type_as(self, x: Tensor):
        self.X = self.X.type_as(x)
        self.E = self.E.type_as(x)
        self.y = self.y.type_as(x)
        return self

    def mask(self, node_mask: Tensor, collapse=False):
        x_mask = node_mask.unsqueeze(-1)          # bs, n, 1
        e_mask1 = x_mask.unsqueeze(-2)            # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(-3)            # bs, 1, n, 1

        if collapse:
            self.X = torch.argmax(self.X, dim=-1)
            self.E = torch.argmax(self.E, dim=-1)

            self.X[node_mask == 0] = - 1
            self.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = - 1
        else:
            self.X = self.X * x_mask
            self.E = self.E * e_mask1 * e_mask2
            if not torch.allclose(self.E, torch.transpose(self.E, 1, 2)):
                warnings.warn('non-symmetric graph E is generated!!!')

        return self

    def clone(self):
        if self.y is not None:
            return DenseGraph(self.X.clone(), self.E.clone(), self.y.clone())
        else:
            return DenseGraph(self.X.clone(), self.E.clone())

    def detach(self):
        if self.y is not None:
            return DenseGraph(self.X.clone().detach(), self.E.clone().detach(), self.y.clone().detach())
        else:
            return DenseGraph(self.X.clone().detach(), self.E.clone().detach())


class BaseBatchClass:
    aug_batch = False
    def __getitem__(self, key):
        return getattr(self, key)
    
    @property
    def keys(self) -> list[str]:
        return [field.name for field in fields(self)]
    
    def shapes(self) -> dict[str, tuple[int, ...]]:
        shapes = {}
        for key in self.keys:
            v = getattr(self, key)
            shapes[key] = v.shape if isinstance(v, Tensor) else None
        return shapes
    
    def get_shape(self, key: str) -> tuple[int, ...]:
        if hasattr(self, key):
            return getattr(self, key).shape
        else:
            raise KeyError(f"Key '{key}' not found in BatchDataClass")
    
    def summary(self) -> None:
        print("BatchDataClass Summary:")
        shapes = self.shapes()
        for key, shape in shapes.items():
            print(f"{key}: {shape}")
    
    def _apply_(self, func: Callable[[Tensor], Tensor]) -> None:
        for field in fields(self):
            field_name = field.name
            original_value = getattr(self, field_name)
            if isinstance(original_value, Tensor):
                setattr(self, field_name, func(original_value))
            elif isinstance(original_value, None):
                pass
            else:
                raise ValueError(f"Field '{field_name}' is not a Tensor or None: {type(original_value)}")
        return
    
    def apply(self, func: Callable[[Tensor], Tensor]) -> 'BaseBatchClass':
        applied_data = {}
        for field in fields(self):
            field_name = field.name
            original_value = getattr(self, field_name)

            if original_value is None:
                applied_data[field_name] = None
            else:
                applied_data[field_name] = func(original_value)
        
        return self.__class__(**applied_data)

    def mask_(self) -> None:
        for field in fields(self):
            field_name = field.name
            value = getattr(self, field_name)
            if field_name in ['r_X', 'p_X']:               # GraphBatch
                setattr(self, field_name, value * self.x_mask)
            elif field_name in ['r_E', 'p_E']:             # GraphBatch
                setattr(self, field_name, value * self.e_mask)
            elif field_name in ['r_onehot', 'p_onehot']:   # SmilesBatch
                setattr(self, field_name, value * self.smi_mask.unsqueeze(-1))
        return

    def clone(self, deep: bool = True) -> 'BaseBatchClass':
        if deep:
            return self.apply(deepcopy)
        else:
            return self.apply(lambda x: x)
    
    def clone_detached(self) -> 'BaseBatchClass':
        return self.apply(lambda x: x.detach().clone())
    
    def to_device(self, device: torch.device, clone: bool = False) -> 'BaseBatchClass':
        if clone:
            return self.apply(lambda x: x.clone().to(device))
        else:
            return self.apply(lambda x: x.to(device))
    
    def flatten_beam(self) -> 'BaseBatchClass':
        """Flatten beam dimension: [B, K, ...] -> [B*K, ...]"""
        flattened_data = {}
        for field in fields(self):
            field_name = field.name
            value = getattr(self, field_name)
            if isinstance(value, Tensor) and value.ndim >= 2:
                # Flatten first two dimensions
                batch_size, beam_size = value.shape[:2]
                remaining_shape = value.shape[2:]
                flattened_data[field_name] = value.view(batch_size * beam_size, *remaining_shape)
            else:
                flattened_data[field_name] = value
        return self.__class__(**flattened_data)
    
    def unflatten_beam(self, batch_size: int, beam_size: int) -> 'BaseBatchClass':
        """Unflatten beam dimension: [B*K, ...] -> [B, K, ...]"""
        unflattened_data = {}
        for field in fields(self):
            field_name = field.name
            value = getattr(self, field_name)
            if isinstance(value, Tensor) and value.ndim >= 1:
                # Unflatten first dimension back to batch_size and beam_size
                flat_batch_size = value.shape[0]
                if flat_batch_size == batch_size * beam_size:
                    remaining_shape = value.shape[1:]
                    unflattened_data[field_name] = value.view(batch_size, beam_size, *remaining_shape)
                else:
                    unflattened_data[field_name] = value
            else:
                unflattened_data[field_name] = value
        return self.__class__(**unflattened_data)
    

@dataclass
class GraphSamplingBatch(BaseBatchClass):
    p_X: Tensor
    p_E: Tensor
    cond: Tensor

    node_mask: Tensor
    p_mask: Tensor
    x_mask: Optional[Tensor] = field(init=True, default=None)
    e_mask: Optional[Tensor] = field(init=True, default=None)

    target: Optional[Tensor] = field(init=True, default=None)

    def __post_init__(self):
        if self.x_mask is None:
            self.x_mask = self.node_mask.unsqueeze(-1)
        if self.e_mask is None:
            self.e_mask = self.x_mask.unsqueeze(-2) * self.x_mask.unsqueeze(-3)
        self.mask_()

    def to_data(self, r_X: Tensor, r_E: Tensor) -> 'GraphBatch':
        shared_fields = {f.name for f in fields(self)} - {'r_X', 'r_E'}
        kwargs = {
            field: getattr(self, field)
            for field in shared_fields
        }
        return GraphBatch(r_X=r_X, r_E=r_E, **kwargs)

    def to_sampling(self) -> 'GraphSamplingBatch':
        return self
    
@dataclass
class GraphBatch(BaseBatchClass):
    r_X: Tensor
    p_X: Tensor
    r_E: Tensor
    p_E: Tensor
    cond: Tensor

    node_mask: Tensor
    p_mask: Tensor
    x_mask: Optional[Tensor] = field(init=True, default=None)
    e_mask: Optional[Tensor] = field(init=True, default=None)

    target: Optional[Tensor] = field(init=True, default=None)

    def __post_init__(self):
        if self.x_mask is None:
            self.x_mask = self.node_mask.unsqueeze(-1)
        if self.e_mask is None:
            self.e_mask = self.x_mask.unsqueeze(-2) * self.x_mask.unsqueeze(-3)
        self.mask_()
    
    def to_sampling(self) -> GraphSamplingBatch:
        sampling_fields = {f.name for f in fields(GraphSamplingBatch)} - {'r_X', 'r_E'}
        kwargs = {
            field: getattr(self, field)
            for field in sampling_fields
        }
        return GraphSamplingBatch(**kwargs)

    def update_r_(self, r_X: Tensor, r_E: Tensor) -> None:
        self.r_X = r_X * self.x_mask
        self.r_E = r_E * self.e_mask






