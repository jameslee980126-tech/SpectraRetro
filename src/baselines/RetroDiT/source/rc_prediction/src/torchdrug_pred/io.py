from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

from rdkit import Chem

from torchdrug_pred.chem import (
    clear_map_canonical_smiles,
    find_permutation,
    get_canonical_map_order,
)

PermutationInfo = dict[str, dict[str, float]]


def calibrate_permutation_scores(
    permutation_info: PermutationInfo,
    temperature: float,
) -> PermutationInfo:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    calibrated: PermutationInfo = {}
    inverse_temperature = 1.0 / temperature
    for product, scores in permutation_info.items():
        transformed = {
            permutation: max(float(score), 0.0) ** inverse_temperature
            for permutation, score in scores.items()
        }
        total = sum(transformed.values())
        if total <= 0:
            raise ValueError(f"product has no positive score mass: {product}")
        calibrated[product] = {
            permutation: score / total
            for permutation, score in transformed.items()
        }
    return calibrated


@lru_cache(maxsize=None)
def _rooted_permutation_entry(
    product_smiles: str,
    root_index: int,
) -> tuple[str, str] | None:
    """Cache RDKit canonicalization shared by temperature / aggregation grids."""
    canonical_smiles = clear_map_canonical_smiles(product_smiles)
    canonical_order = get_canonical_map_order(product_smiles)
    rooted_order = get_canonical_map_order(product_smiles, root=root_index)
    if canonical_smiles is None or canonical_order is None or rooted_order is None:
        return None
    permutation = find_permutation(canonical_order, rooted_order)
    return canonical_smiles, "_".join(map(str, permutation))


@lru_cache(maxsize=None)
def _identity_permutation_entry(product_smiles: str) -> tuple[str, str] | None:
    canonical_smiles = clear_map_canonical_smiles(product_smiles)
    canonical_order = get_canonical_map_order(product_smiles)
    if canonical_smiles is None or canonical_order is None:
        return None
    return canonical_smiles, "_".join(map(str, range(len(canonical_order))))


def add_permutation_score(
    permutation_info: PermutationInfo,
    product_smiles: str,
    root_index: int,
    score: float,
) -> bool:
    entry = _rooted_permutation_entry(product_smiles, root_index)
    if entry is None:
        return False
    canonical_smiles, key = entry
    scores = permutation_info.setdefault(canonical_smiles, {})
    scores[key] = scores.get(key, 0.0) + float(score)
    return True


def complete_identity_permutations(
    permutation_info: PermutationInfo,
    reaction_smiles: list[str],
) -> None:
    for reaction in reaction_smiles:
        parts = reaction.split(">")
        if len(parts) != 3:
            continue
        product_smiles = parts[2]
        entry = _identity_permutation_entry(product_smiles)
        if entry is None:
            continue
        canonical_smiles, identity = entry
        if canonical_smiles not in permutation_info or not permutation_info[canonical_smiles]:
            permutation_info[canonical_smiles] = {identity: 1.0}


def normalize_scores(permutation_info: PermutationInfo) -> None:
    for scores in permutation_info.values():
        denominator = sum(score for score in scores.values() if score > 0)
        if denominator <= 0:
            continue
        for key in scores:
            scores[key] = float(scores[key]) / float(denominator)


def validate_permutation_info(permutation_info: PermutationInfo) -> None:
    if not isinstance(permutation_info, dict):
        raise TypeError("permutation output must be a JSON object")
    for product_smiles, scores in permutation_info.items():
        if not isinstance(product_smiles, str) or not product_smiles:
            raise ValueError("product SMILES keys must be non-empty strings")
        product_molecule = Chem.MolFromSmiles(product_smiles)
        if product_molecule is None:
            raise ValueError(f"invalid product SMILES key: {product_smiles}")
        if clear_map_canonical_smiles(product_smiles) != product_smiles:
            raise ValueError(f"product SMILES key is not canonical: {product_smiles}")
        if not isinstance(scores, dict) or not scores:
            raise ValueError(f"permutation mapping is empty: {product_smiles}")
        atom_count = product_molecule.GetNumAtoms()
        has_positive_score = False
        for permutation_key, score in scores.items():
            if not isinstance(permutation_key, str) or not permutation_key:
                raise ValueError(f"invalid permutation key: {permutation_key}")
            try:
                permutation = [int(value) for value in permutation_key.split("_")]
            except ValueError as error:
                raise ValueError(
                    f"invalid permutation key: {permutation_key}"
                ) from error
            if len(permutation) != atom_count:
                raise ValueError(
                    f"permutation length does not match product atoms: {permutation_key}"
                )
            if sorted(permutation) != list(range(atom_count)):
                raise ValueError(f"permutation is not a bijection: {permutation_key}")
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise TypeError(f"permutation score must be numeric: {permutation_key}")
            if not math.isfinite(score) or score < 0:
                raise ValueError(
                    f"permutation score must be finite and non-negative: {permutation_key}"
                )
            has_positive_score = has_positive_score or score > 0
        if not has_positive_score:
            raise ValueError(
                f"permutation mapping has no positive score: {product_smiles}"
            )


def write_json(path: str | Path, value: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as file:
        json.dump(value, file, indent=4, allow_nan=False)


def write_permutation_json(
    path: str | Path,
    permutation_info: PermutationInfo,
) -> None:
    validate_permutation_info(permutation_info)
    write_json(path, permutation_info)
