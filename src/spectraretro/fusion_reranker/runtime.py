"""Fusion v1: all same-draw candidates, immutable features/weights, stable ties."""
from collections import Counter, defaultdict
from functools import lru_cache
from importlib import import_module
import hashlib
import json
import math
from pathlib import Path
import pickle

import numpy as np
from threadpoolctl import threadpool_limits
from . import chemical_features as chem

ROOT = Path(__file__).resolve().parent
VERSION = 'fusion_v1_20260929'


def verify_bundle():
    manifest = json.loads((ROOT / 'manifest.json').read_text(encoding='utf-8'))
    for name, digest in manifest['sha256'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError('Frozen fusion bundle changed: ' + name)
    return manifest


def candidate_records(reaction):
    """Inputs contain observable synthons and aligned draws, never targets."""
    allowed = {'reaction_id', 'product', 'fragments', 'class_id'}
    if set(reaction) - allowed or not {'reaction_id', 'product', 'fragments'} <= set(reaction):
        raise ValueError('Unexpected/missing reaction fields')
    fragments = reaction['fragments']
    if not fragments:
        raise ValueError('Complete companion fragments required')
    ids = [str(f['uuid']) for f in fragments]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate fragment UUIDs')
    for f in fragments:
        if set(f) - {'uuid', 'fragments', 'anchors', 'center1', 'center2', 'draws'}:
            raise ValueError('Only observable fragment fields and generated draws accepted')
    draws = [f['draws'] for f in fragments]
    n = len(draws[0])
    if not n or any(len(d) != n or any(not isinstance(s, str) for s in d) for d in draws):
        raise ValueError('Nonempty aligned string draw slots required; preserve failures as MISSING')
    merged = ['.'.join(d[i] for d in draws) for i in range(n)]
    counts, marginal = Counter(merged), [Counter(d) for d in draws]
    first, stats = {}, defaultdict(list)
    for i, raw in enumerate(merged):
        first.setdefault(raw, i)
        fractions = [marginal[j][draws[j][i]] / n for j in range(len(draws))]
        stats[raw].append((sum(math.log(max(f, 1e-12)) for f in fractions), min(fractions), max(fractions), max(fractions)-min(fractions)))
    ranked = sorted(counts, key=lambda s: (-counts[s], first[s]))
    top = counts[ranked[0]]
    product = chem.canonical_molecule(reaction['product'])
    if product is None:
        raise ValueError('Invalid product')
    candidates = []
    for rank, raw in enumerate(ranked, 1):
        count, values = counts[raw], stats[raw]
        feature = dict(frequency_count=float(count), frequency_fraction=count/n,
            frequency_log1p=math.log1p(count), frequency_rank=float(rank),
            frequency_top_ratio=chem.safe_ratio(count,top), frequency_top_gap=chem.safe_ratio(top-count,top),
            frequency_first_index=first[raw]/n, frequency_unique_candidates=float(len(counts)),
            marginal_log_product_max=max(v[0] for v in values),
            marginal_log_product_mean=sum(v[0] for v in values)/len(values),
            marginal_min_fraction_max=max(v[1] for v in values),
            marginal_max_fraction_max=max(v[2] for v in values),
            marginal_imbalance_min=min(v[3] for v in values), reaction_component_rows=float(len(draws)))
        normalized = chem.canonical_precursor_set(raw)
        if normalized is not None:
            feature.update(chem.summarize_candidate(normalized, product))
        candidates.append(dict(candidate=raw, parts=[d[first[raw]] for d in draws],
            first_draw_index=first[raw], frequency=count, baseline_rank=rank, features=feature))
    return candidates


class FrozenFusionRanker:
    def __init__(self, mode='unknown', method='fusion'):
        if mode not in ('unknown', 'known') or method not in ('fusion', 'original_chemical', 'frequency'):
            raise ValueError('Unsupported ranking mode/method')
        self.manifest = verify_bundle()
        self.mode, self.method = mode, method
        self.ranking = import_module(f'spectraretro.{mode}-class.ranking')
        self.schema = json.loads((ROOT/'assets/features.json').read_text(encoding='utf-8'))
        if any(n.startswith('reaction_class_') for n in self.schema['chemical'] + self.schema['identity']):
            raise RuntimeError('Class feature in shared fusion schema')
        self.models = []
        if mode == 'unknown' and method == 'fusion':
            self.models = self.ranking.load_ensemble(ROOT / 'assets')
        self.chemical = None
        if method == 'original_chemical' or (mode == 'known' and method == 'fusion'):
            self.chemical = pickle.loads((ROOT/f'assets/{mode}_chemical_inference.pkl').read_bytes())
        self.identity = None
        if method == 'fusion':
            from .identity_adapter import FrozenIdentityAdapter
            self.identity = FrozenIdentityAdapter()

    def rank_batch(self, reactions, top_k=None):
        from .identity_adapter import candidate_features
        if not reactions:
            return []
        if top_k is not None and (type(top_k) is not int or top_k < 1):
            raise ValueError('top_k must be positive')
        all_candidates, rows = [], []
        for i, reaction in enumerate(reactions):
            self.ranking.validate_reaction(reaction)
            all_candidates.append(candidate_records(reaction))
            for j, f in enumerate(reaction['fragments']):
                row = {k: f[k] for k in ('fragments', 'anchors', 'center1', 'center2') if k in f}
                row.update(uuid=f'{i}:{j}', reaction_id=str(i), product=reaction['product'])
                row.setdefault('anchors', '')
                rows.append(row)
        bank = self.identity.predict_fragments(rows) if self.identity else None
        features, old_features, logps, ranks, classes = [], [], [], [], []
        for i, (reaction, candidates) in enumerate(zip(reactions, all_candidates)):
            ids = [f'{i}:{j}' for j in range(len(reaction['fragments']))]
            for c in candidates:
                f = c['features']
                identity = candidate_features(bank, ids, c['parts']) if bank else {}
                features.append([f.get(n,0.) for n in self.schema['chemical']] + [identity.get(n,0.) for n in self.schema['identity']])
                if self.chemical:
                    old_features.append(self.ranking.chemical_features(f, reaction, self.chemical['feature_names']))
                logps.append(identity.get('identity_prior_logp_sum',0.))
                ranks.append(c['baseline_rank'])
                classes.append(reaction.get('class_id',0))
        ranks = np.array(ranks)
        with threadpool_limits(limits=2):
            if self.method == 'frequency':
                scores = -ranks.astype(float)
            elif self.models:
                scores = self.ranking.fusion_scores(self.models, features)
            else:
                b = self.chemical
                columns = [b['feature_names'].index(n) for n in b['feature_sets'][b['selected_model']]]
                x = np.asarray(old_features,dtype=np.float32)
                prob = b['models'][b['selected_model']].predict_proba(x[:,columns])[:,1]
                if self.method == 'fusion':
                    scores = self.ranking.fusion_scores(prob, ranks, classes, logps)
                else:
                    scores = prob - ranks*1e-12
        if not np.isfinite(scores).all():
            raise RuntimeError('Nonfinite frozen reranker scores')
        results, offset = [], 0
        for reaction, candidates in zip(reactions, all_candidates):
            values = scores[offset:offset+len(candidates)]
            offset += len(candidates)
            order = np.argsort(-values,kind='stable')
            ranked = []
            for rank, idx in enumerate(order,1):
                c = {k:v for k,v in candidates[idx].items() if k != 'features'}
                score = float(values[idx])
                # Bounded API utility preserves order; NOT a calibrated feasibility probability.
                if self.method == 'frequency':
                    utility = 1/c['baseline_rank']
                elif self.method == 'fusion' and self.mode == 'known':
                    utility = self.ranking.ranking_utility(score, reaction['class_id'])
                else:
                    utility = max(0., min(1., score))
                c.update(rank=rank, fusion_score=score, score=utility)
                ranked.append(c)
            results.append(dict(reaction_id=str(reaction['reaction_id']), product=reaction['product'],
                ranking_version=VERSION, mode=self.mode, method=self.method,
                candidate_pool='all_existing_unique_same_draw_candidates',
                class_input=self.mode=='known', score_definition='uncalibrated_ranking_utility',
                candidates=ranked if top_k is None else ranked[:top_k]))
        return results

    def rank(self, reaction, top_k=None):
        return self.rank_batch([reaction], top_k=top_k)[0]


@lru_cache(maxsize=6)
def get_ranker(mode='unknown', method='fusion'):
    return FrozenFusionRanker(mode, method)
