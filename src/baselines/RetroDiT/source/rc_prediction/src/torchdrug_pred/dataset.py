from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torchdrug import datasets
from tqdm import tqdm

from torchdrug_pred.chem import (
    get_positive_center_atom_maps,
    get_reaction_center,
    sanity_check,
)


class USPTOReactionCenter(datasets.USPTO50k):
    def __init__(
        self,
        path: str | Path,
        dataset: str,
        verbose: int = 1,
        objective: str = "single_positive",
    ) -> None:
        if objective not in {
            "single_positive",
            "multi_positive",
            "multi_positive_margin",
        }:
            raise ValueError(f"Unsupported objective: {objective}")
        self.path = str(path)
        self.dataset = dataset
        self.objective = objective
        if dataset == "50k":
            self.max_atoms = 100
            self.max_atom_difference = 10
        elif dataset == "full":
            self.max_atoms = 100
            self.max_atom_difference = 20
        else:
            raise NotImplementedError(f"Dataset {dataset} is not implemented")
        if Path(path).suffix == ".pt":
            self._load_processed(path)
        else:
            self._load_csv(path, verbose)

    def _load_csv(self, path: str | Path, verbose: int) -> None:
        self.load_csv(
            str(path),
            smiles_field="reactants>reagents>production",
            target_fields=self.target_fields,
            atom_feature="center_identification",
            kekulize=True,
            verbose=verbose,
        )
        original_data = self.data
        original_smiles = self.smiles_list
        original_targets = self.targets
        self.data = []
        self.smiles_list = []
        self.targets = defaultdict(list)
        indices = range(len(original_data))
        iterator = tqdm(indices, desc="Computing reaction centers") if verbose else indices
        invalid = 0
        for index in iterator:
            reactant, product = original_data[index]
            reaction_smiles = original_smiles[index]
            reactant_smiles, _, product_smiles = reaction_smiles.split(">")
            if (
                sanity_check(
                    product_smiles,
                    reactant_smiles,
                    self.max_atoms,
                    self.max_atom_difference,
                )
                is None
            ):
                invalid += 1
                continue
            reactant.bond_stereo[:] = 0
            product.bond_stereo[:] = 0
            labeled = self._label_reaction(reactant, product, reaction_smiles)
            if labeled is None:
                invalid += 1
                continue
            labeled_reactant, labeled_product = labeled
            self.data.append((labeled_reactant, labeled_product))
            self.smiles_list.append(reaction_smiles)
            for key in original_targets:
                target_key = self.target_alias.get(key, key)
                self.targets[target_key].append(original_targets[key][index] - 1)
            self.targets["sample id"].append(index)
        self.valid_rate = 1.0 - invalid / max(len(original_data), 1)

    def _label_reaction(
        self,
        reactant: Any,
        product: Any,
        reaction_smiles: str,
    ) -> tuple[Any, Any] | None:
        if getattr(self, "objective", "single_positive").startswith("multi_positive"):
            return self._label_multi_positive(reactant, product, reaction_smiles)
        center = get_reaction_center(reaction_smiles)
        if center is None or not center["has_changes"]:
            return None
        edge_label = torch.zeros(product.num_edge, dtype=torch.long)
        node_label = torch.zeros(product.num_node, dtype=torch.long)
        reaction_center: torch.Tensor | None = None
        priority = [
            ("formed_bonds", "edge"),
            ("broken_bonds", "node"),
            ("bond_order_changes", "edge"),
            ("charge_changes", "node"),
            ("h_count_changes", "node"),
            ("hybridization_changes", "node"),
            ("aromatic_changes", "node"),
            ("chirality_changes", "node"),
        ]
        for key, level in priority:
            candidates = center[key]
            if not isinstance(candidates, list) or not candidates:
                continue
            candidate = candidates[0]
            if level == "edge" and len(candidate) == 2:
                source, target = candidate
                pattern = torch.tensor(
                    [[source, target, -1], [target, source, -1]],
                    dtype=torch.long,
                )
                matched, _ = product.match(pattern)
                if matched.numel() > 0:
                    edge_label[matched] = 1
                    reaction_center = torch.tensor(
                        [
                            int(product.atom_map[source].item()),
                            int(product.atom_map[target].item()),
                        ],
                        dtype=torch.long,
                    )
                    break
            elif candidate:
                node_index = candidate[0]
                if node_index < product.num_node:
                    node_label[node_index] = 1
                    reaction_center = torch.tensor(
                        [int(product.atom_map[node_index].item()), 0],
                        dtype=torch.long,
                    )
                    break
        if reaction_center is None or edge_label.sum() + node_label.sum() == 0:
            return None
        with product.edge():
            product.edge_label = edge_label
        with product.node():
            product.node_label = node_label
        with reactant.graph():
            reactant.reaction_center = reaction_center
        with product.graph():
            product.reaction_center = reaction_center
        return reactant, product

    def _label_multi_positive(
        self,
        reactant: Any,
        product: Any,
        reaction_smiles: str,
    ) -> tuple[Any, Any] | None:
        positive_centers = get_positive_center_atom_maps(
            reaction_smiles,
            include_symmetric=True,
        )
        if not positive_centers:
            return None
        edge_label = torch.zeros(product.num_edge, dtype=torch.long)
        node_label = torch.zeros(product.num_node, dtype=torch.long)
        graph_atom_maps = [int(value) for value in product.atom_map.tolist()]
        map_to_graph_index = {
            atom_map: index for index, atom_map in enumerate(graph_atom_maps)
        }
        for center_maps in positive_centers:
            if len(center_maps) == 2:
                try:
                    source, target = (
                        map_to_graph_index[center_maps[0]],
                        map_to_graph_index[center_maps[1]],
                    )
                except KeyError:
                    continue
                pattern = torch.tensor(
                    [[source, target, -1], [target, source, -1]],
                    dtype=torch.long,
                )
                matched, _ = product.match(pattern)
                if matched.numel() > 0:
                    edge_label[matched] = 1
            elif len(center_maps) == 1 and center_maps[0] in map_to_graph_index:
                node_label[map_to_graph_index[center_maps[0]]] = 1
        if edge_label.sum() + node_label.sum() == 0:
            return None
        first_center = positive_centers[0]
        reaction_center = torch.tensor(
            [first_center[0], first_center[1] if len(first_center) == 2 else 0],
            dtype=torch.long,
        )
        with product.edge():
            product.edge_label = edge_label
        with product.node():
            product.node_label = node_label
        with reactant.graph():
            reactant.reaction_center = reaction_center
        with product.graph():
            product.reaction_center = reaction_center
        return reactant, product

    def save_processed(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": "torchdrug_pred.uspto.v2",
                "dataset": self.dataset,
                "objective": self.objective,
                "data": self.data,
                "smiles_list": self.smiles_list,
                "targets": dict(self.targets),
                "valid_rate": self.valid_rate,
                "max_atoms": self.max_atoms,
                "max_atom_difference": self.max_atom_difference,
            },
            output_path,
        )

    def _load_processed(self, path: str | Path) -> None:
        payload: dict[str, Any] = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
        if payload["format"] not in {
            "torchdrug_pred.uspto.v1",
            "torchdrug_pred.uspto.v2",
        }:
            raise ValueError(f"Unsupported processed dataset: {path}")
        if payload["dataset"] != self.dataset:
            raise ValueError(
                f"Processed dataset {path} was built for "
                f"{payload['dataset']}, not {self.dataset}"
            )
        self.data = payload["data"]
        self.smiles_list = payload["smiles_list"]
        self.targets = defaultdict(list, payload["targets"])
        self.transform = None
        self.lazy = False
        self.kwargs = {
            "atom_feature": "center_identification",
            "kekulize": True,
        }
        self.valid_rate = float(payload["valid_rate"])
        self.max_atoms = int(payload["max_atoms"])
        self.max_atom_difference = int(payload["max_atom_difference"])
        stored_objective = payload.get("objective", "single_positive")
        stored_label_mode = (
            "multi_positive"
            if stored_objective.startswith("multi_positive")
            else "single_positive"
        )
        requested_label_mode = (
            "multi_positive"
            if self.objective.startswith("multi_positive")
            else "single_positive"
        )
        if stored_label_mode != requested_label_mode:
            relabeled = []
            for (reactant, product), reaction_smiles in zip(
                self.data,
                self.smiles_list,
            ):
                labeled = self._label_reaction(reactant, product, reaction_smiles)
                if labeled is None:
                    raise ValueError(
                        "Relabeling a retained reaction produced no positive center"
                    )
                relabeled.append(labeled)
            self.data = relabeled
