import random
import numpy as np
from typing import Any, Literal

__all__ = ['ProbSampler']

class ProbSampler():
    def __init__(
            self,
            data_dict: dict[Any, dict[Any, float]],
            sample_single: Literal['prob', 'greedy'],
            sample_all: Literal['prob', 'all'],
            top_k: int,
            temperature: float = 1.0,
        ):
        self.data_dict = data_dict

        self.top_k = top_k
        self.sample_single = sample_single
        self.sample_all = sample_all
        self.temperature = temperature

    def __new__(
        cls,
        data_dict: dict[Any, dict[Any, float]] | None,
        **kwargs
    ):
        if data_dict is None:
            return None
        return super().__new__(cls)
    
    def sample_1(
            self, key: str
        ) -> tuple[Any, float] | tuple[None, None]:
        sample2weight = self.data_dict.get(key, None)
        if sample2weight is None:
            return None, None
        
        match self.sample_single:
            case 'prob':
                sample = self._sample_1_p(sample2weight)
            case 'greedy':
                sample = self._sample_1_g(sample2weight)
            case _:
                raise NotImplementedError()
        return sample, sample2weight[sample]
            
    def sample_k(
            self, key: str
        ) -> tuple[list[Any], list[float], int] | tuple[None, None, None]:
        sample2weight = self.data_dict.get(key, None)
        if sample2weight is None:
            return None, None, None
        
        match self.sample_all:
            case 'prob':
                return self._sample_k_p(sample2weight)
            case 'all':
                return self._sample_k_all(sample2weight)
            case _:
                raise NotImplementedError()

    def _sample_1_p(self, sample2weight: dict[Any, float]) -> Any:
        return random.choices(
            population=list(sample2weight.keys()),
            weights=list(sample2weight.values()),
            k=1
        )[0]

    def _sample_1_g(self, sample2weight: dict[Any, float]) -> Any:
        return max(sample2weight.items(), key=lambda x: x[1])[0]
    
    def _sample_k_p(
            self, sample2weight: dict[Any, float]
        ) -> tuple[list[Any], list[float], int]:
        keys = list(sample2weight.keys())
        probs = np.array(list(sample2weight.values()))
        probs = probs / np.sum(probs)
        log_probs = np.log(probs + 1e-20)
        
        uniform_samples = np.random.uniform(0, 1, (self.top_k, len(probs)))
        gumbel_noise = -np.log(-np.log(uniform_samples + 1e-20) + 1e-20)
        
        perturbed = log_probs[np.newaxis, :] + gumbel_noise / self.temperature
        sampled_indices = np.argmax(perturbed, axis=1)
        
        return [keys[idx] for idx in sampled_indices], \
            probs[sampled_indices].tolist(), len(keys)
        
    def _sample_k_all(
            self, sample2weight: dict[Any, float]
        ) -> tuple[list[Any], list[float], int]:
        keys = list(sample2weight.keys())
        n_keys = len(keys)

        probs = np.array(list(sample2weight.values()))
        probs = probs / np.sum(probs)
        sampled_indices = np.argsort(-probs)

        samples = [keys[idx] for idx in sampled_indices]
        sample_ps: list[float] = probs[sampled_indices].tolist()

        pad_n = self.top_k - n_keys
        samples.extend([samples[-1] for _ in range(pad_n)])
        sample_ps.extend([-float('inf')] * pad_n)
        return samples[:self.top_k], sample_ps[:self.top_k], n_keys
        



