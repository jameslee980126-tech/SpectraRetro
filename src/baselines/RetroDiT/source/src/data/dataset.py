import os
import json

from rdkit import Chem
from functools import partial
from typing import Mapping, Literal, Optional, Sequence, Any, Union

import torch

from torch import Tensor
from torch_geometric.data import Data, InMemoryDataset


from .transforms import shuffle_graph
from .process import RawDataConverter, compute_perm_order
from .p_sampler import ProbSampler

__all__ = ['DatasetModule', 'load_graph_vocab', 'EXCLUDE_KEYS']

EXCLUDE_KEYS = ['perm_weight', 'n_perm']

class DatasetModule(InMemoryDataset):
    def __init__(
            self,
            stage: Literal['train', 'val', 'test'],
            root: str, graph_vocab: str, 

            # dataset settings
            is_leaky: bool = True,
            max_n_len: int = 100, n_dummy: int = 10, 
            shuffle_order: bool = False, canonicalize: bool = True,
            dn_last: bool = True, 

            perm_prob_sampler: partial[ProbSampler] = None,
            val_top_k: int = None, test_top_k: int = None,
            perm_types: Union[list[str], str] = None,
            external_perm_paths: Optional[Mapping[str, Optional[str]]] = None,
            **kwargs
        ):
        self.set_stage(stage)

        # Load types and bonds based on graph_vocab
        self.graph_vocab = graph_vocab

        # Dataset settings
        self.is_leaky = is_leaky
        self.max_n_len = max_n_len
        self.n_dummy = n_dummy
        self.shuffle_order = shuffle_order
        self.canonicalize = canonicalize
        self.dn_last = dn_last
        self.external_perm_paths = external_perm_paths

        super().__init__(root=root)
        
        # load graph data
        self.data, self.slices = torch.load(
            self.processed_paths[self.file_idx], weights_only=False
        )

        # load label mappings
        if os.path.exists(self.label_mapping_path):
            label_mappings: dict[str, dict[str, int]] = self.load_json(self.label_mapping_path)
            self.label_dims = {
                label: len(mapping.values())
                for label, mapping in label_mappings.items()
            }
        else:
            self.label_dims = {}

        # load permutations
        if perm_prob_sampler:
            match stage:
                case 'train':
                    top_k = 0
                case 'val':
                    top_k = val_top_k
                case 'test':
                    top_k = test_top_k
                case _:
                    raise ValueError
                
            permutations: dict[str, dict[str, float]] = self.setup_permutations(perm_types)
            
            self.perm_prob_sampler: ProbSampler = \
                perm_prob_sampler(
                    data_dict=permutations,
                    top_k=top_k
                )
        else:
            self.perm_prob_sampler = None
            

    def set_stage(self, stage: str):
        self.stage = stage
        if self.stage == 'train':
            self.file_idx = 0
        elif self.stage == 'val':
            self.file_idx = 1
        elif self.stage == 'test':
            self.file_idx = 2
        else:
            raise ValueError()
        return self.file_idx

    @property
    def filefolder(self) -> str:
        filefolder = f'ndummy_{self.n_dummy}_maxlen_{self.max_n_len}'
        if self.canonicalize:
            filefolder += '_cano'
        if self.shuffle_order:
            filefolder += '_shuffled'
        if not self.dn_last:
            filefolder += '_dnfirst'
        return filefolder
    
    @property
    def processed_dir(self) -> str:
        filefolder = os.path.join(
            'processed_leaky' if self.is_leaky else 'processed',
            self.filefolder
        )
        return os.path.join(self.root, filefolder)
    
    @property
    def split_paths(self):
        return [os.path.join(self.raw_dir, f) for f in self.raw_file_names]

    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']
    


    def download(self):
        raise NotImplementedError()

    def get_react_data(self) -> Sequence[str]:
        raise NotImplementedError()
    
    def get_label_data(self) -> Sequence[str]:
        raise NotImplementedError()

    def setup_graph_vocab(
            self, converter: RawDataConverter, react_data: list[str]
        ) -> tuple[dict, dict]:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        all_graph_vocab_path = os.path.join(
            current_dir, '..', '..', 'data', 'graph_vocab.json'
        )
        # Load existing graph vocabs or create new dict
        all_graph_vocabs = {}
        if os.path.exists(all_graph_vocab_path):
            all_graph_vocabs = self.load_json(all_graph_vocab_path)

        if self.graph_vocab not in all_graph_vocabs:
            assert self.file_idx == 0
            print(f"Creating new graph vocabulary: {self.graph_vocab}")
            x_enc, e_enc = converter.compute_graph_vocab(react_data)
            graph_vocab_dict = {
                'node_types': x_enc, 'edge_types': e_enc
            }
            all_graph_vocabs[self.graph_vocab] = graph_vocab_dict
            with open(all_graph_vocab_path, 'w') as f:
                json.dump(all_graph_vocabs, f)
            print(f"Graph vocabulary saved with {len(x_enc)} node types and {len(e_enc)} edge types")
        else:
            print(f"Using existing graph vocabulary: {self.graph_vocab}")

        x_enc, _, e_enc, _ = load_graph_vocab(self.graph_vocab)
        return x_enc, e_enc

    def load_json(self, path: str) -> Any:
        with open(path, 'r') as f:
            data = json.load(f)
        return data

    #############################################################
    #############################################################
    ###############         Label config      ###################
    #############################################################
    #############################################################
    @property
    def label_mapping_path(self) -> str:
        return os.path.join(self.root, 'label_mappings.json')
    
    def save_label_mappings(
            self, label_mappings: dict[str, dict[Any, int]]
        ):
        with open(self.label_mapping_path, 'w') as f:
            json.dump(label_mappings, f, indent=2)
        print(f"Label mappings saved to {self.label_mapping_path}")
        print(f"  - {len(label_mappings)} label types saved")


    #############################################################
    #############################################################
    ###############      Permutation utils    ###################
    #############################################################
    #############################################################
    @property
    def permutation_path(self) -> str:
        return os.path.join(self.root, f'permutations_{self.stage}.json')
    
    def setup_permutations(self, perm_types: Union[list[str], str] = None):
        
        RC_PERM_TYPES = [
            'formed_bonds',
            'broken_bonds', 
            'bond_order_changes',
            'charge_changes',
            'h_count_changes',
            'chirality_changes',
            'aromatic_changes',
            'hybridization_changes'
        ]
        external_path = self._external_permutation_path()
        if external_path is not None:
            permutations_types, perm_types = self._load_external_permutations(
                external_path
            )
        elif isinstance(perm_types, str):
            permutations_types, perm_types = self._load_external_permutations(
                perm_types
            )
        elif perm_types is None:
            return None
        elif len(perm_types) == 0:
            return None
        else:
            if os.path.exists(self.permutation_path):
                permutations_types: dict[str, dict[str, dict[str, float]]] = \
                    self.load_json(self.permutation_path)
            else:
                permutations_types = {}
            
            if perm_types == ['all']:
                perm_types = RC_PERM_TYPES
            else:
                for perm_type in perm_types:
                    assert perm_type in RC_PERM_TYPES

            need_process = [
                perm_type for perm_type in perm_types
                if perm_type not in permutations_types
            ]
            if need_process:
                react_data = self.get_react_data()
                rc_results = compute_perm_order(
                    react_data, need_process, self.max_n_len, self.n_dummy
                )
                permutations_types.update(rc_results)
                
                self.save_permutation(permutations_types)
    
        merged_dict = {}
        for perm_type in perm_types:
            current_dict = permutations_types[perm_type]
            for cond, feat_freq in current_dict.items():
                if cond not in merged_dict:
                    merged_dict[cond] = feat_freq.copy()
                else:
                    existing_feat_freq = merged_dict[cond]
                    for key, value in feat_freq.items():
                        if key in existing_feat_freq:
                            existing_feat_freq[key] += value
                        else:
                            existing_feat_freq[key] = value
        
        return merged_dict    

    def _external_permutation_path(self) -> str | None:
        paths = self.external_perm_paths
        if paths is None:
            return None
        if not isinstance(paths, Mapping):
            paths = dict(paths)
        keys = ('val', 'valid') if self.stage == 'val' else (self.stage,)
        for key in keys:
            path = paths.get(key)
            if path:
                return str(path)
        return None

    def _load_external_permutations(
            self, path: str
        ) -> tuple[dict[str, dict[str, dict[str, float]]], list[str]]:
        permutations_data = self.load_json(path)
        if not permutations_data:
            return {}, []

        first_value = next(iter(permutations_data.values()))
        if isinstance(first_value, dict) and first_value:
            nested_first_value = next(iter(first_value.values()))
            if isinstance(nested_first_value, dict):
                # Legacy schema: {perm_type: {smiles: {perm: score}}}
                return permutations_data, list(permutations_data.keys())

        # Direct TorchDrug export: {smiles: {perm: score}}
        return {'pred_perm': permutations_data}, ['pred_perm']



    def save_permutation(
            self, permutations: dict[str, dict[str, dict[str, float]]]
        ):
        """
        permutations = {
            method1: {
                smiles1: {
                    0_1_2_3: float(prob1),
                    1_0_2_3: float(prob2),
                }, ...
            },
            method2: {...}
        }
        """
        with open(self.permutation_path, 'w') as f:
            json.dump(permutations, f, indent=2)
        print(f"Permutations saved to {self.permutation_path}")

    def perm_str2list(self, perm_str: str) -> list:
        return list(map(int, perm_str.split('_')))

    def sample_perm_1(
            self, cano_smiles: str, perm_len: int
        ) -> Tensor | None:
        perm, p_weight = self.perm_prob_sampler.sample_1(cano_smiles)
        if perm is not None:
            assert p_weight is not None
            perm: Tensor = torch.tensor(self.perm_str2list(perm))
            if self.dn_last:
                perm = torch.cat((
                    perm, torch.arange(perm.size(0), perm_len)
                ), -1)
            else:
                n_dn = perm_len - perm.size(0)
                perm = torch.cat((
                    torch.arange(n_dn), perm + n_dn
                ), -1)
        else:
            assert p_weight is None
        return perm
    
    def sample_perm_k(
            self, cano_smiles: str, perm_len: int
        ) -> tuple[Tensor, list[float], list[int]] | tuple[list[None], list[None], list[None]]:
        top_k = self.perm_prob_sampler.top_k
        perms, p_weights, n_perm = self.perm_prob_sampler.sample_k(cano_smiles)
        if perms is None:
            assert p_weights is None and n_perm is None
            return [None] * top_k, [None] * top_k, [None] * top_k
        else:
            assert p_weights is not None and n_perm is not None

        perms: Tensor = torch.tensor(list(map(
            self.perm_str2list, perms
        )))
        if self.dn_last:
            perms = torch.cat((
                perms,
                torch.arange(
                    perms.size(1), perm_len
                ).unsqueeze(0).repeat(perms.size(0), 1)
            ), -1)
        else:
            n_dn = perm_len - perms.size(1)
            perms = torch.cat((
                torch.arange(
                    n_dn
                ).unsqueeze(0).repeat(perms.size(0), 1),
                perms + n_dn
            ), -1)
        return perms, p_weights, [n_perm]*(perms.size(0))


    def shuffle_batch(self, batch: Data, perm: Optional[Tensor] = None) -> Data:
        perm_batch = batch.clone()
        if perm is None:
            return perm_batch
        elif not is_valid_permutation(perm):
            return perm_batch
        perm_batch.p_x, perm_batch.p_edge_index, perm_batch.p_edge_attr = \
            shuffle_graph(
                batch.p_x, batch.p_edge_index, batch.p_edge_attr,
                perm
            )

        perm_batch.r_x, perm_batch.r_edge_index, perm_batch.r_edge_attr = \
            shuffle_graph(
                batch.r_x, batch.r_edge_index, batch.r_edge_attr,
                perm
            )
        return perm_batch


    #############################################################
    #############################################################
    ###############         __getitem__       ###################
    #############################################################
    #############################################################
    def __getitem__(self, idx):
        batch = super().__getitem__(idx)
        if self.perm_prob_sampler:
            perm_len = batch.p_x.size(0)
            if self.stage == 'train':
                perm = self.sample_perm_1(
                    batch.p_smiles, perm_len
                )
                return self.shuffle_batch(batch, perm=perm)
            
            else:
                perms_pweights_nperm = self.sample_perm_k(
                    batch.p_smiles, perm_len
                )
                
                aug_batches = []
                for perm, p_weight, n_perm in zip(*perms_pweights_nperm):
                    # if perm is None:
                    #     raise ValueError()
                    aug_batch = self.shuffle_batch(batch, perm=perm)

                    aug_batch.perm_weight = p_weight
                    aug_batch.n_perm = n_perm
                    aug_batches.append(aug_batch)
                return aug_batches

        else:
            return batch



def load_graph_vocab(graph_vocab: str) -> tuple[dict, dict, dict, dict]:
    """Load atom and bond mappings from JSON files"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    graph_vocab_path = os.path.join(
        current_dir, '..', '..', 'data', 'graph_vocab.json'
    )
    with open(graph_vocab_path, 'r') as f:
        graph_vocab_maps = json.load(f)[graph_vocab]
    x_enc = graph_vocab_maps['node_types']
    x_dec = {v: k for k, v in x_enc.items()}
    bond_config = graph_vocab_maps['edge_types']
    e_enc = {getattr(Chem.BondType, k): v for k, v in bond_config.items()}
    e_dec = {v: k for k, v in e_enc.items()}
    e_dec[0] = None
    e_enc = {k: v-1 for k, v in e_enc.items()}
    return x_enc, x_dec, e_enc, e_dec

def is_valid_permutation(perm: Tensor) -> bool:
    perm_list = perm.tolist()
    return set(perm_list) == set(range(len(perm)))
