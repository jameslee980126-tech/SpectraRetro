import json
import numpy as np

from omegaconf import DictConfig
from typing import List, Optional, Dict, Iterable
from collections import defaultdict, Counter

from . import RankedLogger
from src.models.forward_syn import ForwardSyn

__all__ = ['ResultStore']

log = RankedLogger(__name__, rank_zero_only=True)

class ResultStore:
    """
    A clean storage class for sampling results.
    """
    
    ref = 'ref'
    gen = 'gen'
    cond = 'cond'
    gen_freq = 'gen_freq'
    feat_freq = 'feat_freq'
    
    def __init__(
            self,
            top_ks: List[int],
            eval_skip_gen_null: bool = True,
            test_round_trip: bool = True,
            forward_syn_cfg: Optional[DictConfig] = None
        ):
        """
        Initialize the store with specified top-k values to track.
        
        Args:
            top_ks: List of k values to track (e.g., [1, 3, 5, 10])
            eval_skip_gen_null: if True, generated None will be skipped and equivalently considered as replaced by the next sample
        """
        
        self.top_ks = sorted(top_ks)
        self.max_k = max(top_ks)

        self.eval_skip_gen_null = eval_skip_gen_null

        if forward_syn_cfg and test_round_trip:
            self.forward_syn = ForwardSyn(**forward_syn_cfg)
        else:
            self.forward_syn = None
        self.test_round_trip = self.forward_syn is not None
            
        # Storage for all samples
        self.samples: list[dict[str, int]] = []  # List[Dict[str, Any]]
        self.log = []
        
    def setup_device(self, device):
        if self.forward_syn is not None:
            self.forward_syn.to(device)

    def store(
            self, 
            cond_list: List[Optional[str]],
            gen_list: List[List[Optional[str]]],
            ref_list: Optional[List[Optional[str]]] = None, 
            feat_list: Optional[List[Optional[str]]] = None
        ):
        """
        Store sampling results for a batch.
        
        Args:
            gen_list: Generated samples, shape [max_k, batch_size]
        """
        batch_size = len(cond_list)
        if ref_list is None:
            ref_list = [None] * batch_size
        
        # Validation
        assert len(cond_list) == batch_size, \
            f"Mismatch: ref_list={len(ref_list)}, cond_list={len(cond_list)}"
        assert len(gen_list) == self.max_k, \
            f"Expected {self.max_k} generation rounds, got {len(gen_list)}"
        
        for gen_round in gen_list:
            assert len(gen_round) == batch_size
        if feat_list is not None:
            for feat in feat_list:
                assert len(feat) == batch_size
            
        # Store each sample
        for i in range(batch_size):
            gen_samples = [gen_list[k][i] for k in range(self.max_k)]
            data = {
                self.ref: ref_list[i], self.cond: cond_list[i], 
                self.gen: gen_samples, self.gen_freq: dict(Counter(gen_samples))
            }
            if feat_list is not None:
                feats = ["_".join(map(str, feat_list[k][i])) for k in range(self.max_k)]
                data[self.feat_freq] = dict(Counter(feats))
            self.samples.append(data)
    

    def eval(
            self,
            samples: Optional[List[Dict]] = None,
            rank_temp_file: Optional[str] = None,
            is_str_form: bool = True
        ) -> Dict[str, float]:
        if samples is None:
            samples = self.samples
        zero_fill = '0.0/0=0.0' if is_str_form else 0.0
        if not samples:
            results = {}
            for k in self.top_ks:
                results[f'top_{k}_accuracy'] = zero_fill
                results[f'deduplicate_top_{k}_accuracy'] = zero_fill
                results[f'round_trip'] = zero_fill
            return results
        
        topk_success = defaultdict(int)
        dedup_topk_success = defaultdict(int)
        dedup_rt_cov = defaultdict(int)
        dedup_rt_acc = defaultdict(int)
        total_valid = 0
        
        forward_syn_input, gt_products = [], []
        for sample in samples:
            forward_syn_input.extend(sample[self.gen])
            gt_products.extend([sample[self.cond]] * self.max_k)

            gt_react = sample[self.ref]
            if gt_react is None:
                self.log.append("Ground truth sample is None!!")
                continue
            total_valid += 1
            
            valid_generated = sample[self.gen]
            if self.eval_skip_gen_null:
                valid_generated = [x for x in valid_generated if x is not None]

            dedup_valid_gens = unique_topk(valid_generated, self.max_k)
            for k in self.top_ks:
                if gt_react in valid_generated[:k]:
                    topk_success[k] += 1
                if gt_react in dedup_valid_gens[:k]:
                    dedup_topk_success[k] += 1

        # =====================================================
        # =====================================================
        # Round trip calculation
        if self.test_round_trip:
            assert len(forward_syn_input) == self.max_k * len(samples)

            pred_products = self.forward_syn.predict(forward_syn_input)
            round_trip = np.array([
                gt_prod == pred_prod
                for gt_prod, pred_prod in zip(gt_products, pred_products)
            ], dtype=np.bool_)

            react2prod = {}
            for reactant, product in zip(forward_syn_input, pred_products):
                if reactant not in react2prod:
                    react2prod[reactant] = product


        for sample in samples:
            gt_react = sample[self.ref]
            gt_product = sample[self.cond]
            valid_generated = sample[self.gen]

            if gt_product is None:
                continue
            if self.eval_skip_gen_null:
                valid_generated = [x for x in valid_generated if x is not None]
            
            dedup_valid_gens = unique_topk(valid_generated, self.max_k)
            
            if self.test_round_trip:
                dedup_rt_matches = [
                    gt_product == react2prod.get(reactant, None) \
                        or gt_react == reactant
                    for reactant in dedup_valid_gens
                ]
                
                for k in self.top_ks:
                    k_matches = dedup_rt_matches[:k]
                    if any(k_matches):
                        dedup_rt_cov[k] += 1
                    if len(k_matches) > 0:
                        dedup_rt_acc[k] += sum(k_matches) / len(k_matches)
        # =====================================================
        # =====================================================
        # =====================================================

        results = {}

        if is_str_form:
            for k in self.top_ks:
                if total_valid > 0:
                    accuracy = f"{topk_success[k]}/{total_valid}={topk_success[k]/total_valid:.4f}"
                    dedup_accuracy = f"{dedup_topk_success[k]}/{total_valid}={dedup_topk_success[k]/total_valid:.4f}"
                    if self.test_round_trip:
                        rt_cov = f"{dedup_rt_cov[k]}/{total_valid}={dedup_rt_cov[k]/total_valid:.4f}"
                        rt_acc = f"{dedup_rt_acc[k]}/{total_valid}={dedup_rt_acc[k]/total_valid:.4f}"
                else:
                    accuracy = zero_fill
                    dedup_accuracy = zero_fill
                    if self.test_round_trip:
                        rt_cov = zero_fill
                        rt_acc = zero_fill

                results[f'top_{k}_accuracy'] = accuracy
                results[f'deduplicate_top_{k}_accuracy'] = dedup_accuracy
                if self.test_round_trip:
                    results[f'rt_cov_top_{k}'] = rt_cov
                    results[f'rt_acc_top_{k}'] = rt_acc

            if self.test_round_trip:
                results[f'round_trip'] = f"{round_trip.sum()}/{len(forward_syn_input)}={round_trip.mean():.4f}"

        else:
            for k in self.top_ks:
                if total_valid > 0:
                    accuracy = topk_success[k] / total_valid
                    dedup_accuracy = dedup_topk_success[k] / total_valid
                    if self.test_round_trip:
                        rt_cov = dedup_rt_cov[k] / total_valid
                        rt_acc = dedup_rt_acc[k] / total_valid
                else:
                    accuracy = zero_fill
                    dedup_accuracy = zero_fill
                    if self.test_round_trip:
                        rt_cov = zero_fill
                        rt_acc = zero_fill

                results[f'top_{k}_accuracy'] = accuracy
                results[f'deduplicate_top_{k}_accuracy'] = dedup_accuracy
                if self.test_round_trip:
                    results[f'rt_cov_top_{k}'] = rt_cov
                    results[f'rt_acc_top_{k}'] = rt_acc

            if self.test_round_trip:
                results[f'round_trip'] = round_trip.mean()

        if rank_temp_file is not None:
            self.save_results(results, rank_temp_file)
        return results


    def save_results(self, to_log: dict, write_path: str):
        with open(write_path, 'w') as f:
            f.write(json.dumps(to_log) + '\n')
            for sample in self.samples:
                # Keep raw generations in rank-local files as well, so the
                # merged test artifact is protocol-equivalent in single- and
                # multi-process evaluation.
                f.write(json.dumps(sample) + '\n')
        log.info(f"Saved {len(self.samples)} samples to {write_path}")

    def load_results(self, rank_temp_file: str):
        flattened_samples = []
        total_metrics = defaultdict(lambda: [0, 0])
        with open(rank_temp_file, 'r') as f:
            lines = f.readlines()
            to_log: dict[str, str] = json.loads(lines[0])
            for k in to_log.keys():
                n_acc_div_n_tol = to_log[k].split('=')[0]
                n_acc, n_tol = n_acc_div_n_tol.split('/')
                total_metrics[k][0] += int(float(n_acc))
                total_metrics[k][1] += int(float(n_tol))
            
            rank_samples = [
                json.loads(line) for line in lines[1:]
            ]
            flattened_samples.extend(rank_samples)
        log.info(f"Loaded {len(flattened_samples)} samples from {rank_temp_file}")
        return total_metrics, flattened_samples

    def get_samples(self) -> list[dict[str, int]]:
        # Preserve the draw order in final sampling artifacts.  ``gen_freq`` is
        # sufficient for RetroDiT's native frequency-ranked metric, but it
        # cannot reproduce a raw-slot Top-k protocol because Counter
        # aggregation discards the original sequence (including invalid and
        # duplicate slots).  Returning a shallow copy keeps the upstream
        # metric calculation unchanged while allowing the common benchmark
        # evaluator to consume the exact generated order.
        return [sample.copy() for sample in self.samples]

    def clear(self):
        """Clear all stored samples."""
        self.samples.clear()

def unique_topk(gens: Iterable, top_k: int) -> list:
    counter = Counter(gens)
    return [elem for elem, count in counter.most_common(top_k)]
