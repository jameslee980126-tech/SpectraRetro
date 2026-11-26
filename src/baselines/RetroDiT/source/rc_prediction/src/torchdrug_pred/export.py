from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

from torchdrug_pred.io import (
    PermutationInfo,
    add_permutation_score,
    complete_identity_permutations,
    write_json,
    write_permutation_json,
)


@dataclass(frozen=True)
class CenterCandidate:
    atom_indices: tuple[int, ...]
    atom_maps: tuple[int, ...]
    log_likelihood: float


@dataclass(frozen=True)
class ProductPrediction:
    product_smiles: str
    true_atom_maps: tuple[int, ...]
    candidates: tuple[CenterCandidate, ...]
    product_atom_maps: tuple[int, ...] = ()
    true_center_atom_maps: tuple[tuple[int, ...], ...] = ()


def _true_center_sets(prediction: ProductPrediction) -> list[set[int]]:
    centers = prediction.true_center_atom_maps or (prediction.true_atom_maps,)
    return [
        {atom_map for atom_map in center if atom_map > 0}
        for center in centers
        if any(atom_map > 0 for atom_map in center)
    ]


def aggregate_root_scores(
    candidates: Iterable[CenterCandidate],
    center_top_k: int,
    temperature: float,
    root_limit: int,
    aggregation_method: str = "current",
) -> list[tuple[int, float]]:
    if center_top_k <= 0:
        raise ValueError("center_top_k must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if root_limit <= 0:
        raise ValueError("root_limit must be positive")
    if aggregation_method not in {
        "current",
        "split_edge",
        "reciprocal_rank",
        "probability_rank_hybrid",
    }:
        raise ValueError(f"Unsupported root aggregation: {aggregation_method}")
    selected = list(candidates)[:center_top_k]
    if not selected:
        return []
    scaled = [candidate.log_likelihood / temperature for candidate in selected]
    maximum = max(scaled)
    probability_weights = [math.exp(value - maximum) for value in scaled]
    reciprocal_weights = [1.0 / rank for rank in range(1, len(selected) + 1)]
    if aggregation_method in {"current", "split_edge"}:
        weights = probability_weights
    elif aggregation_method == "reciprocal_rank":
        weights = reciprocal_weights
    else:
        probability_total = sum(probability_weights)
        reciprocal_total = sum(reciprocal_weights)
        weights = [
            0.5 * probability / probability_total
            + 0.5 * reciprocal / reciprocal_total
            for probability, reciprocal in zip(
                probability_weights,
                reciprocal_weights,
            )
        ]
    root_scores: dict[int, float] = {}
    for candidate, weight in zip(selected, weights):
        endpoint_weight = weight
        if aggregation_method == "split_edge":
            endpoint_weight /= len(candidate.atom_indices)
        for atom_index in candidate.atom_indices:
            root_scores[atom_index] = (
                root_scores.get(atom_index, 0.0) + endpoint_weight
            )
    denominator = sum(root_scores.values())
    if denominator <= 0:
        return []
    ranked = sorted(
        (
            (atom_index, score / denominator)
            for atom_index, score in root_scores.items()
        ),
        key=lambda item: (-item[1], item[0]),
    )
    return ranked[:root_limit]


def build_permutation_info(
    predictions: Iterable[ProductPrediction],
    reaction_smiles: list[str],
    center_top_k: int,
    temperature: float,
    root_limit: int,
    aggregation_method: str = "current",
) -> tuple[PermutationInfo, dict[str, float | int]]:
    permutation_info: PermutationInfo = {}
    center_hit = {1: 0, 3: 0, 5: 0}
    center_reciprocal_rank = 0.0
    root_hit = {1: 0, 3: 0, 5: 0}
    root_reciprocal_rank = 0.0
    candidate_count = 0
    covered = 0
    sample_count = 0
    for prediction in predictions:
        sample_count += 1
        true_centers = _true_center_sets(prediction)
        true_maps = set().union(*true_centers) if true_centers else set()
        center_rank: int | None = None
        for index, candidate in enumerate(prediction.candidates[:center_top_k]):
            candidate_maps = {atom_map for atom_map in candidate.atom_maps if atom_map > 0}
            if candidate_maps in true_centers:
                center_rank = index + 1
                break
        if center_rank is not None:
            center_reciprocal_rank += 1.0 / center_rank
            for k in center_hit:
                center_hit[k] += int(center_rank <= k)
        roots = aggregate_root_scores(
            prediction.candidates,
            center_top_k,
            temperature,
            root_limit,
            aggregation_method,
        )
        retained_root_score = sum(score for _, score in roots)
        if retained_root_score > 0:
            roots = [
                (root_index, score / retained_root_score)
                for root_index, score in roots
            ]
        candidate_count += len(roots)
        true_root_indices = {
            atom_index
            for atom_index, atom_map in enumerate(prediction.product_atom_maps)
            if atom_map in true_maps
        }
        if not true_root_indices:
            molecule = _molecule_atom_maps(prediction.product_smiles)
            true_root_indices = {
                index for index, atom_map in enumerate(molecule) if atom_map in true_maps
            }
        root_rank: int | None = None
        exported = False
        for index, (root_index, score) in enumerate(roots):
            export_root_index = _export_root_index(prediction, root_index)
            exported = (
                add_permutation_score(
                    permutation_info,
                    prediction.product_smiles,
                    export_root_index,
                    score,
                )
                or exported
            )
            if root_rank is None and root_index in true_root_indices:
                root_rank = index + 1
        covered += int(exported)
        if root_rank is not None:
            root_reciprocal_rank += 1.0 / root_rank
            for k in root_hit:
                root_hit[k] += int(root_rank <= k)
    complete_identity_permutations(permutation_info, reaction_smiles)
    denominator = max(sample_count, 1)
    metrics: dict[str, float | int] = {
        "center_top_k": center_top_k,
        "root_limit": root_limit,
        "temperature": temperature,
        "root_aggregation": aggregation_method,
        "center_hit_at_1": center_hit[1] / denominator,
        "center_hit_at_3": center_hit[3] / denominator,
        "center_hit_at_5": center_hit[5] / denominator,
        "center_mrr": center_reciprocal_rank / denominator,
        "root_hit_at_1": root_hit[1] / denominator,
        "root_hit_at_3": root_hit[3] / denominator,
        "root_hit_at_5": root_hit[5] / denominator,
        "root_mrr": root_reciprocal_rank / denominator,
        "mean_candidates": candidate_count / denominator,
        "prediction_coverage": covered / denominator,
        "retained_samples": sample_count,
    }
    return permutation_info, metrics


def calculate_ranking_metrics(
    predictions: Iterable[ProductPrediction],
    center_top_k: int,
    temperature: float,
    root_limit: int,
    aggregation_method: str = "current",
) -> dict[str, float | int]:
    """Calculate checkpoint-selection metrics without RDKit permutation export."""
    center_hit = {1: 0, 3: 0, 5: 0}
    root_hit = {1: 0, 3: 0, 5: 0}
    center_reciprocal_rank = 0.0
    root_reciprocal_rank = 0.0
    candidate_count = 0
    covered = 0
    sample_count = 0
    for prediction in predictions:
        sample_count += 1
        true_centers = _true_center_sets(prediction)
        true_maps = set().union(*true_centers) if true_centers else set()
        center_rank: int | None = None
        for index, candidate in enumerate(prediction.candidates[:center_top_k]):
            candidate_maps = {atom_map for atom_map in candidate.atom_maps if atom_map > 0}
            if candidate_maps in true_centers:
                center_rank = index + 1
                break
        if center_rank is not None:
            center_reciprocal_rank += 1.0 / center_rank
            for k in center_hit:
                center_hit[k] += int(center_rank <= k)

        roots = aggregate_root_scores(
            prediction.candidates,
            center_top_k,
            temperature,
            root_limit,
            aggregation_method,
        )
        candidate_count += len(roots)
        covered += int(bool(roots))
        true_root_indices = {
            atom_index
            for atom_index, atom_map in enumerate(prediction.product_atom_maps)
            if atom_map in true_maps
        }
        if not true_root_indices:
            molecule = _molecule_atom_maps(prediction.product_smiles)
            true_root_indices = {
                index for index, atom_map in enumerate(molecule) if atom_map in true_maps
            }
        root_rank = next(
            (
                index
                for index, (root_index, _) in enumerate(roots, start=1)
                if root_index in true_root_indices
            ),
            None,
        )
        if root_rank is not None:
            root_reciprocal_rank += 1.0 / root_rank
            for k in root_hit:
                root_hit[k] += int(root_rank <= k)

    denominator = max(sample_count, 1)
    return {
        "center_top_k": center_top_k,
        "root_limit": root_limit,
        "temperature": temperature,
        "root_aggregation": aggregation_method,
        "center_hit_at_1": center_hit[1] / denominator,
        "center_hit_at_3": center_hit[3] / denominator,
        "center_hit_at_5": center_hit[5] / denominator,
        "center_mrr": center_reciprocal_rank / denominator,
        "root_hit_at_1": root_hit[1] / denominator,
        "root_hit_at_3": root_hit[3] / denominator,
        "root_hit_at_5": root_hit[5] / denominator,
        "root_mrr": root_reciprocal_rank / denominator,
        "mean_candidates": candidate_count / denominator,
        "prediction_coverage": covered / denominator,
        "retained_samples": sample_count,
    }


def _molecule_atom_maps(product_smiles: str) -> list[int]:
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(product_smiles)
    if molecule is None:
        return []
    return [atom.GetAtomMapNum() for atom in molecule.GetAtoms()]


def _export_root_index(
    prediction: ProductPrediction,
    graph_root_index: int,
) -> int:
    """Map a TorchDrug graph atom index to the raw product atom order."""
    if not prediction.product_atom_maps:
        return graph_root_index
    if graph_root_index >= len(prediction.product_atom_maps):
        raise IndexError("graph root index exceeds product atom map count")
    root_map = prediction.product_atom_maps[graph_root_index]
    raw_atom_maps = _molecule_atom_maps(prediction.product_smiles)
    try:
        return raw_atom_maps.index(root_map)
    except ValueError as error:
        raise ValueError(
            f"root atom map {root_map} is absent from raw product"
        ) from error


def _candidate_product_ids(
    result: dict[str, Any],
    batch_size: int,
    center_top_k: int,
) -> list[int]:
    candidate_count = int(result["reaction_center"].shape[0])
    if candidate_count == batch_size * center_top_k:
        return torch.arange(batch_size).repeat_interleave(center_top_k).tolist()
    synthon = result["synthon"]
    num_synthon = result["num_synthon"]
    if not hasattr(synthon, "product_id") or len(num_synthon) != candidate_count:
        raise RuntimeError("Unable to align TorchDrug center candidates with products")
    starts = num_synthon.cumsum(0) - num_synthon
    return synthon.product_id[starts].detach().cpu().tolist()


@torch.no_grad()
def collect_predictions(
    task: Any,
    dataset: Any,
    center_top_k: int,
    batch_size: int,
    num_worker: int,
) -> list[ProductPrediction]:
    from torchdrug import data, utils

    from torchdrug_pred.chem import get_positive_center_atom_maps

    loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_worker,
    )
    task.eval()
    predictions: list[ProductPrediction] = []
    sample_offset = 0
    for batch in loader:
        if task.device.type == "cuda":
            batch = utils.cuda(batch, device=task.device)
        _, product = batch["graph"]
        result = task.predict_synthon(batch, k=center_top_k)
        centers = result["reaction_center"].detach().cpu()
        log_likelihoods = result["log_likelihood"].detach().cpu()
        product_ids = _candidate_product_ids(result, len(product), center_top_k)
        candidates_by_product: list[list[CenterCandidate]] = [
            [] for _ in range(len(product))
        ]
        product_graphs = list(product)
        for center, log_likelihood, product_id in zip(
            centers,
            log_likelihoods,
            product_ids,
        ):
            molecule = product_graphs[product_id]
            atom_maps = molecule.atom_map.detach().cpu().tolist()
            center_maps = tuple(
                int(atom_map) for atom_map in center.tolist() if int(atom_map) > 0
            )
            atom_indices = tuple(
                index for index, atom_map in enumerate(atom_maps) if atom_map in center_maps
            )
            if not atom_indices:
                continue
            candidates_by_product[product_id].append(
                CenterCandidate(
                    atom_indices=atom_indices,
                    atom_maps=center_maps,
                    log_likelihood=float(log_likelihood.item()),
                )
            )
        batch_reactions = dataset.smiles_list[
            sample_offset:sample_offset + len(product)
        ]
        product_smiles = []
        for reaction in batch_reactions:
            parts = reaction.split(">")
            if len(parts) != 3:
                raise ValueError(f"invalid reaction SMILES: {reaction}")
            product_smiles.append(parts[2])
        true_centers = product.reaction_center.detach().cpu().reshape(len(product), -1)
        for index, smiles in enumerate(product_smiles):
            true_maps = tuple(
                int(atom_map)
                for atom_map in true_centers[index].tolist()
                if int(atom_map) > 0
            )
            predictions.append(
                ProductPrediction(
                    product_smiles=smiles,
                    true_atom_maps=true_maps,
                    candidates=tuple(candidates_by_product[index]),
                    product_atom_maps=tuple(
                        int(atom_map)
                        for atom_map in product_graphs[index].atom_map.detach().cpu().tolist()
                    ),
                    true_center_atom_maps=tuple(
                        get_positive_center_atom_maps(
                            batch_reactions[index],
                            include_symmetric=True,
                        )
                    ),
                )
            )
        sample_offset += len(product)
    if sample_offset != len(dataset):
        raise RuntimeError(
            f"prediction / dataset length mismatch: {sample_offset} != {len(dataset)}"
        )
    return predictions


def export_prediction_sets(
    predictions: list[ProductPrediction],
    reaction_smiles: list[str],
    output_dir: str | Path,
    prediction_prefix: str,
    center_top_ks: list[int],
    strict_root_top_ks: list[int],
    reference_root_limit: int,
    temperature: float | list[float],
    split: str | None = None,
    metrics_dir: str | Path | None = None,
    aggregation_methods: list[str] | None = None,
    export_policies: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    if len(center_top_ks) != len(strict_root_top_ks):
        raise ValueError("center_top_ks and strict_root_top_ks must have equal lengths")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    metrics_path_root = (
        Path(metrics_dir) if metrics_dir is not None else output_path.parent / "metrics"
    )
    manifest: dict[str, dict[str, Any]] = {}
    temperatures = (
        [float(temperature)]
        if isinstance(temperature, (int, float))
        else [float(value) for value in temperature]
    )
    methods = aggregation_methods or ["current"]
    requested_policies = export_policies or ["reference", "strict_root"]
    invalid_policies = set(requested_policies) - {"reference", "strict_root"}
    if invalid_policies:
        raise ValueError(f"Unsupported export policies: {sorted(invalid_policies)}")
    is_grid = len(temperatures) > 1 or len(methods) > 1 or methods != ["current"]
    for center_top_k, strict_root_top_k in zip(
        center_top_ks,
        strict_root_top_ks,
    ):
        policy_limits = {
            "reference": reference_root_limit,
            "strict_root": strict_root_top_k,
        }
        policies = tuple(
            (policy, policy_limits[policy]) for policy in requested_policies
        )
        for policy, root_limit in policies:
            for value in temperatures:
                for aggregation_method in methods:
                    permutation_info, metrics = build_permutation_info(
                        predictions,
                        reaction_smiles,
                        center_top_k,
                        value,
                        root_limit,
                        aggregation_method,
                    )
                    stem = (
                        f"{prediction_prefix}_{policy}_center_top{center_top_k}"
                        f"_root_top{root_limit}"
                    )
                    if is_grid:
                        temperature_label = format(value, "g").replace(".", "p")
                        stem += (
                            f"_temp{temperature_label}"
                            f"_agg_{aggregation_method}"
                        )
                    prediction_path = output_path / f"{stem}.json"
                    metrics_path = metrics_path_root / f"{stem}.json"
                    write_permutation_json(prediction_path, permutation_info)
                    write_json(metrics_path, metrics)
                    manifest[stem] = {
                        "split": split,
                        "prediction_path": str(prediction_path),
                        "metrics_path": str(metrics_path),
                        "metrics": metrics,
                    }
    write_json(output_path / "index.json", manifest)
    return manifest
