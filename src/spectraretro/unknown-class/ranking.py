"""Class-free ensemble ranking of assembled precursor candidates."""
import pickle

import numpy as np


def validate_reaction(reaction):
    if 'class_id' in reaction:
        raise ValueError('Unknown-class inference forbids class_id')


def chemical_features(features, reaction, names):
    return [features.get(name, 0.) for name in names]


def load_ensemble(assets):
    return [
        pickle.loads((assets / f'unified_fusion_fold{fold}_inference.pkl').read_bytes())
        for fold in range(5)
    ]


def fusion_scores(models, features):
    """Apply each fold to its selected columns and average candidate probabilities."""
    x = np.asarray(features, dtype=np.float32)
    return np.mean([
        fold['model'].predict_proba(x[:, fold['indices']])[:, 1]
        for fold in models
    ], axis=0)
