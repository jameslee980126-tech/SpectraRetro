"""Class-conditioned precursor features and fusion scores for SpectraRetro."""
import math

import numpy as np


def validate_reaction(reaction):
    class_id = reaction.get('class_id')
    if type(class_id) is not int or class_id not in range(1, 7):
        raise ValueError('Known-class scorer requires explicit six-class id 1..6')


def chemical_features(features, reaction, names):
    return [
        float(reaction['class_id'] == int(name.rsplit('_', 1)[1]))
        if name.startswith('reaction_class_') else features.get(name, 0.)
        for name in names
    ]


def fusion_scores(probabilities, ranks, classes, identity_logps):
    """The class-5 scorer combines the chemical score and fragment identity prior."""
    scores = probabilities - ranks * 1e-12
    return np.where(
        np.array(classes) == 5,
        np.log(np.clip(probabilities, 1e-12, 1)) + 2 * np.array(identity_logps),
        scores,
    )


def ranking_utility(score, class_id):
    if class_id == 5:
        return 1 / (1 + math.exp(-max(-700, min(700, score))))
    return max(0., min(1., score))
