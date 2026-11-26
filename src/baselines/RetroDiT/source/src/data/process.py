import torch
from torch_geometric.data import Data

from tqdm import tqdm
from rdkit import Chem
from typing import Sequence
from functools import partial
from collections import defaultdict
from multiprocessing import Pool, cpu_count

from .transforms import *

__all__ = ['RawDataConverter', 'compute_perm_order']

class RawDataConverter():
    def __init__(
            self,
            dn_last: bool,
            max_n_len: int = 100, n_dummy: int = 10, 
            shuffle_order: bool = False, canonicalize: bool = True,
        ):
        self.max_n_len = max_n_len
        self.n_dummy = n_dummy
        self.dn_last = dn_last

        self.shuffle_order = shuffle_order
        self.canonicalize = canonicalize
        self.is_enc_setup = False
        
    def setup_enc(self, x_enc: dict, e_enc: dict):
        self.x_enc = x_enc
        self.e_enc = e_enc
        self.is_enc_setup = True
    
    def _prepare_kwargs(self, kwargs: dict) -> dict:
        processed = {}
        for key, value in kwargs.items():
            if isinstance(value, np.ndarray):
                processed[key] = torch.from_numpy(value)
            elif isinstance(value, (list, tuple)):
                processed[key] = torch.tensor(value)
            elif isinstance(value, str):
                processed[key] = value
            elif torch.is_tensor(value):
                processed[key] = value
            else:
                try:
                    processed[key] = torch.tensor(value)
                except (TypeError, ValueError):
                    processed[key] = value
        return processed

    def rawdata2graph(self, reaction: str, **kwargs) -> list[dict]:
        data_list = []
        if self.canonicalize:
            if self.shuffle_order:
                raise NotImplementedError()
            graph_data_infos = self.compute_cano_graphinfo(reaction)
        else:
            graph_data_infos = self.compute_align_graphinfo(reaction)
        if graph_data_infos is not None:
            processed_kwargs = self._prepare_kwargs(kwargs)
            for graph_data_info in graph_data_infos:
                data_list.append({**graph_data_info, **processed_kwargs})

        return data_list

    
    def compute_graph_vocab(
        self, reaction_chains: Sequence[str]
    ) -> tuple[dict, dict]:
        # reaction_chains: [psmi>>rsmi, ..., psmi>>rsmi]
        pbar = tqdm(
            reaction_chains, 
            desc=f'Processing graph vocab'
        )
        atom_appearance = {}
        for reaction_smiles in pbar:
            rsmi, _, psmi = reaction_smiles.split('>')
            sanity_check_res = sanity_check(psmi, rsmi, self.max_n_len, self.n_dummy)
            if sanity_check_res is not None:
                _, pmol, rmol = sanity_check_res
                for atom in pmol.GetAtoms():
                    if atom.GetSymbol() not in atom_appearance:
                        atom_appearance[atom.GetSymbol()] = 1
                    else:
                        atom_appearance[atom.GetSymbol()] += 1
                for atom in rmol.GetAtoms():
                    if atom.GetSymbol() not in atom_appearance:
                        atom_appearance[atom.GetSymbol()] = 1
                    else:
                        atom_appearance[atom.GetSymbol()] += 1
        # sort the atom apperance 
        sorted_num_atom_stats = dict(sorted(
            atom_appearance.items(),
            key=lambda item: item[1],
            reverse=True
        ))
        x_enc = {
            key: idx
            for idx, key in enumerate(sorted_num_atom_stats.keys())
        }
        x_enc['*'] = max(x_enc.values()) + 1        
        e_enc = {
            "SINGLE": 1,
            "DOUBLE": 2,
            "TRIPLE": 3,
            "AROMATIC": 4
        }
        return x_enc, e_enc
    
    def compute_cano_graphinfo(self, reaction: str) -> list[dict] | None:
        rsmi, _, psmi = reaction.split('>')
        sanity_check_res = sanity_check(psmi, rsmi, self.max_n_len, self.n_dummy)
        
        if sanity_check_res is None:
            return 

        cano_psmi = clear_map_canonical_smiles(psmi)
        canop_atom_map = get_cano_map_number(psmi, root=-1)
        canop_smi_with_map = get_cano_smi_with_map(cano_psmi, canop_atom_map)
        rmol = Chem.MolFromSmiles(rsmi)
        pmol = Chem.MolFromSmiles(canop_smi_with_map)

        p_num_nodes = pmol.GetNumAtoms()
        r_num_nodes = rmol.GetNumAtoms()
        if self.n_dummy > 0:
            padding_num_nodes = p_num_nodes + self.n_dummy
        else:
            padding_num_nodes = r_num_nodes   
        
        p_map, r_map = compute_atom_map(pmol, rmol, True)
        if not p_map.items() <= r_map.items():
            print('Warning: product atoms not in reactants!')
            return 
                
        try:
            r_x, r_edge_index, r_edge_attr = compute_graph(
                rmol, r_map, padding_num_nodes,
                node_types=self.x_enc, edge_types=self.e_enc,
                dn_last=self.dn_last
            )
            p_x, p_edge_index, p_edge_attr = compute_graph(
                pmol, p_map, padding_num_nodes,
                node_types=self.x_enc, edge_types=self.e_enc,
                dn_last=self.dn_last
            )
        except:
            print("Error in computing graph due to unkown atoms ")
            return
    
        cano_rsmi = clear_map_canonical_smiles(rsmi)

        data = {
            'r_x': r_x, 'r_edge_index': r_edge_index, 'r_edge_attr': r_edge_attr,
            'p_x': p_x, 'p_edge_index': p_edge_index, 'p_edge_attr': p_edge_attr,
            'num_nodes': padding_num_nodes,
            'r_smiles': cano_rsmi, 'p_smiles': cano_psmi
        }
        return [data]

    
    def compute_cano_info(self, reaction: str) -> list[dict] | None:
        rsmi, _, psmi = reaction.split('>')
        sanity_check_res = sanity_check(psmi, rsmi, self.max_n_len, self.n_dummy)
        
        if sanity_check_res is None:
            return

        padding_num_nodes = None
            
        aug_cano_psmis, aug_cano_rsmis = canonicalize_reaction(reaction, 0)
        if aug_cano_psmis is None or aug_cano_rsmis is None:
            return
                
        graphinfos = []
        for product_smi, reactants_smi in zip(aug_cano_psmis, aug_cano_rsmis):
            if product_smi is None or reactants_smi is None:
                continue
            rmol = Chem.MolFromSmiles(reactants_smi)
            pmol = Chem.MolFromSmiles(product_smi)

            if rmol is None or pmol is None: 
                continue

            p_map, r_map = compute_atom_map(pmol, rmol, True)

            if not p_map.items() <= r_map.items():
                print('Warning: product atoms not in reactants!')
                continue
                
            if padding_num_nodes is None:
                p_num_nodes = pmol.GetNumAtoms()
                r_num_nodes = rmol.GetNumAtoms()
                if self.n_dummy > 0:
                    padding_num_nodes = p_num_nodes + self.n_dummy
                else:
                    padding_num_nodes = r_num_nodes
                    
            try:
                r_x, r_edge_index, r_edge_attr = compute_graph(
                    rmol, r_map, padding_num_nodes,
                    node_types=self.x_enc, edge_types=self.e_enc,
                    dn_last=self.dn_last
                )
                p_x, p_edge_index, p_edge_attr = compute_graph(
                    pmol, p_map, padding_num_nodes,
                    node_types=self.x_enc, edge_types=self.e_enc,
                    dn_last=self.dn_last
                )
            except:
                print("Error in computing graph due to unkown atoms ")
                continue

            cano_psmi = clear_map_canonical_smiles(product_smi)
            cano_rsmi = clear_map_canonical_smiles(reactants_smi)

            data = {
                'r_x': r_x, 'r_edge_index': r_edge_index, 'r_edge_attr': r_edge_attr,
                'p_x': p_x, 'p_edge_index': p_edge_index, 'p_edge_attr': p_edge_attr,
                'num_nodes': padding_num_nodes,
                'r_smiles': cano_rsmi, 'p_smiles': cano_psmi
            }

            graphinfos.append(data)
        
        if len(graphinfos) == 0:
            return
        return graphinfos
    
    
    def compute_align_graphinfo(self, reaction: str) -> list[dict] | None:
        rsmi, _, psmi = reaction.split('>')
        sanity_check_res = sanity_check(psmi, rsmi, self.max_n_len, self.n_dummy)
        if sanity_check_res is None:
            return
        _, pmol, rmol = sanity_check_res 

        p_num_nodes, r_num_nodes = pmol.GetNumAtoms(), rmol.GetNumAtoms()
        
        if self.n_dummy > 0:
            padding_num_nodes = p_num_nodes + self.n_dummy
        else:
            padding_num_nodes = r_num_nodes
        
        p_map, pmol = compute_nodes_order_mapping(pmol)
        r_map, rmol = compute_nodes_order_mapping(rmol)
        if not p_map.items() <= r_map.items():
            print('Warning: product atoms not in reactants!')
            return
        try:
            r_x, r_edge_index, r_edge_attr = compute_graph(
                rmol, r_map, padding_num_nodes,
                node_types=self.x_enc, edge_types=self.e_enc,
                dn_last=self.dn_last
            )
            p_x, p_edge_index, p_edge_attr = compute_graph(
                pmol, p_map, padding_num_nodes,
                node_types=self.x_enc, edge_types=self.e_enc,
                dn_last=self.dn_last
            )
        except:
            print('Error in computing graph due to atoms not in the graph vocab')
            return        
            
        cano_psmi = clear_map_canonical_smiles(psmi)
        cano_rsmi = clear_map_canonical_smiles(rsmi)
        for _ in range(1):
            if self.shuffle_order:
                r_x, p_x, r_edge_index, p_edge_index, \
                    r_edge_attr, p_edge_attr = shuffle_order(
                        r_x, r_edge_index, r_edge_attr,
                        p_x, p_edge_index, p_edge_attr,
                    )

            data = {
                'r_x': r_x, 'r_edge_index': r_edge_index, 'r_edge_attr': r_edge_attr,
                'p_x': p_x, 'p_edge_index': p_edge_index, 'p_edge_attr': p_edge_attr,
                'num_nodes': padding_num_nodes, 
                'r_smiles': cano_rsmi, 'p_smiles': cano_psmi
            }        
                
        return [data]
    

    def canop2sortp(self, reaction: str, **kwargs) -> list[Data] | None:
        assert self.is_enc_setup
        _, _, psmi = reaction.split('>')
        sanity_check_res = sanity_check_p(psmi)
        if sanity_check_res is None:
            return
        else:
            _, pmol = sanity_check_res
        
        p_num_nodes = pmol.GetNumAtoms()
        
        # construct sort atom map order p Data:
        sort_order_map, pmol = compute_nodes_order_mapping(pmol)

        # construct canonical order p Data:
        cano_map_list = get_cano_map_number(psmi)
        cano_p_map = {atom_map: idx for idx, atom_map in enumerate(cano_map_list)}
        try:
            canop_x, canop_edge_index, canop_edge_attr = compute_graph(
                pmol, cano_p_map, p_num_nodes,
                node_types=self.x_enc, edge_types=self.e_enc,
                dn_last=self.dn_last
            )
        except KeyError as e:
            print(f'Missing key: {e}')
            return
        
        # compute cano p coordinates
        cano_psmi = clear_map_canonical_smiles(psmi)
        cano_psmi_with_map = get_cano_smi_with_map(cano_psmi, cano_map_list)
        cano_pmol = Chem.MolFromSmiles(cano_psmi_with_map)

        perm = torch.tensor([
            cano_p_map[i]
            for i in sort_order_map.keys()
        ], dtype=torch.long) # size=(n_seq, )
        
        data = {
            'p_x': canop_x,
            'p_edge_index': canop_edge_index,
            'p_edge_attr': canop_edge_attr,
            'num_nodes': p_num_nodes,
            'smiles': cano_psmi,
            'perm': perm
        }

        processed_kwargs = self._prepare_kwargs(kwargs)
        return [Data(**data, **processed_kwargs)]
    

    def r_nnodes(self, reaction: str, **kwargs) -> Data | None:
        rsmi, _, psmi = reaction.split('>')
        sanity_check_res = sanity_check(psmi, rsmi, self.max_n_len, self.n_dummy)
        
        if sanity_check_res is None:
            return

        cano_psmis, ano_rsmis = map(
            lambda x: x[0], canonicalize_reaction(reaction, 0)
        )
        if cano_psmis is None or ano_rsmis is None:
            return
                
        rmol = Chem.MolFromSmiles(ano_rsmis)
        pmol = Chem.MolFromSmiles(cano_psmis)

        if rmol is None or pmol is None: 
            return

        p_map, _ = compute_atom_map(pmol, rmol, alignment=False)
        p_num_nodes = pmol.GetNumAtoms()
        r_num_nodes = rmol.GetNumAtoms()
   
        try:
            p_x, p_edge_index, p_edge_attr = compute_graph(
                pmol, p_map, p_num_nodes,
                node_types=self.x_enc, edge_types=self.e_enc,
                dn_last=self.dn_last
            )
        except:
            print("Error in computing graph due to unkown atoms ")
            return

        cano_psmi = clear_map_canonical_smiles(cano_psmis)

        data = {
            'p_x': p_x, 'p_edge_index': p_edge_index, 'p_edge_attr': p_edge_attr,
            'num_nodes': p_num_nodes,
            'p_smiles': cano_psmi,
            'r_num_nodes': r_num_nodes
        }

        processed_kwargs = self._prepare_kwargs(kwargs)
        return Data(**data, **processed_kwargs)
    
    
def process_single_reaction(reaction_smiles: str, perm_types: list[str], max_n_len: int, n_dummy: int) -> dict:
    results = {perm_type: {} for perm_type in perm_types}
    
    rsmi, _, psmi = reaction_smiles.split('>')
    
    check_res = sanity_check(psmi, rsmi, max_n_len, n_dummy)
    if check_res is None:
        return results
    
    pmol = check_res[1]
    
    cano_psmi = clear_map_canonical_smiles(psmi)
    cano_order = get_cano_map_number(psmi, root=-1)
    
    rc_atom_map_dict = None

    rc_atom_map_dict = get_reaction_center(reaction_smiles)
    if rc_atom_map_dict is None or not rc_atom_map_dict['has_changes']:
        for perm_type in perm_types:
            perm_counts = defaultdict(int)
            perm_order = get_cano_map_number(psmi, root=-1)
            perm = find_permutations(cano_order, perm_order)
            perm_key = '_'.join(map(str, perm))
            perm_counts[perm_key] = 1
            results[perm_type][cano_psmi] = dict(perm_counts)
    
    for perm_type in perm_types:
        if perm_type in results and cano_psmi in results[perm_type]:
            continue
            
        root_idxs = []
        if rc_atom_map_dict and rc_atom_map_dict['has_changes']:
            rc_atom_maps = []
            
            if 'formed' in perm_type or 'form' in perm_type:
                rc_atom_maps = rc_atom_map_dict.get('formed_bonds', [])
            elif 'broken' in perm_type or 'break' in perm_type:
                rc_atom_maps = rc_atom_map_dict.get('broken_bonds', [])
            elif 'order' in perm_type:
                rc_atom_maps = rc_atom_map_dict.get('bond_order_changes', [])
            elif 'charge' in perm_type:
                rc_atom_maps = rc_atom_map_dict.get('charge_changes', [])
            elif 'h_count' in perm_type or 'hydrogen' in perm_type:
                rc_atom_maps = rc_atom_map_dict.get('h_count_changes', [])
            elif 'chiral' in perm_type:
                rc_atom_maps = rc_atom_map_dict.get('chirality_changes', [])
            elif 'aromatic' in perm_type:
                rc_atom_maps = rc_atom_map_dict.get('aromatic_changes', [])
            elif 'hybrid' in perm_type:
                rc_atom_maps = rc_atom_map_dict.get('hybridization_changes', [])
            else:
                raise NotImplementedError(f"Unknown rc perm_type: {perm_type}")
            
            for atom in pmol.GetAtoms():
                atom_map = atom.GetAtomMapNum()
                if atom_map in rc_atom_maps:
                    root_idxs.extend([atom.GetIdx()] * rc_atom_maps.count(atom_map))

        
        if root_idxs:
            perm_counts = defaultdict(int)
            for root_idx in root_idxs:
                perm_order = get_cano_map_number(psmi, root=int(root_idx))
                perm = find_permutations(cano_order, perm_order)
                perm_key = '_'.join(map(str, perm))
                perm_counts[perm_key] += 1
            
            if perm_counts:
                results[perm_type][cano_psmi] = dict(perm_counts)
    
    return results


def compute_perm_order(
        reaction_chains: Sequence[str],
        perm_types: list[str], 
        max_n_len: int,
        n_dummy: int,
        n_jobs: int = None
    ) -> dict[str, dict]:
    if n_jobs is None:
        n_jobs = cpu_count()
    
    process_func = partial(
        process_single_reaction, 
        perm_types=perm_types, 
        max_n_len=max_n_len, 
        n_dummy=n_dummy
    )
    
    with Pool(processes=n_jobs) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_func, reaction_chains, chunksize=10),
            total=len(reaction_chains),
            desc=f'Processing Order Permutations for {len(perm_types)} types'
        ))
    
    return merge_results(results)


def merge_results(results: list[dict[str, dict]]) -> dict[str, dict]:
    merged = {}
    
    for result in results:
        for perm_type, perm_data in result.items():
            if perm_type not in merged:
                merged[perm_type] = {}
            
            for cano_psmi, perm_counts in perm_data.items():
                if cano_psmi not in merged[perm_type]:
                    merged[perm_type][cano_psmi] = {}
                
                for perm_key, count in perm_counts.items():
                    merged[perm_type][cano_psmi][perm_key] = \
                        merged[perm_type][cano_psmi].get(perm_key, 0) + count
    
    return merged