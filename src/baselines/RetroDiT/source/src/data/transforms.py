import numpy as np

from torch import Tensor
from typing import Optional

from rdkit.Chem import Mol

import torch
import torch.nn.functional as F

from rdkit import Chem

from torch_geometric.utils import to_dense_adj, to_dense_batch, remove_self_loops

from .batch_class import *
import re
import random

# __all__ = [
#     'sanity_check', 
#     'shuffle_order', 'shuffle_graph',
#     'compute_graph',
#     'compute_atom_map',
#     'sort_edges',
#     'to_dense',
#     'dense2smiles',
# ]

def find_permutations(ori_order, new_order):
    """
    ori_order: list of int
    new_order: list of int
    return: list of int, the permutation to convert ori_order to new_order
    """
    perm = []
    for atom_map in new_order:
        idx_in_new = ori_order.index(atom_map)
        perm.append(idx_in_new)
    return perm

def sanity_check_p(psmi:str):
    pmol = Chem.MolFromSmiles(psmi)
    # sanity check
    if "" == psmi:
        print(f"Empty product in reaction")
        return None
    if pmol is None:
        print(f'Failed to parse product')
        return None
    p_num_nodes = pmol.GetNumAtoms()
    if p_num_nodes == 1:  
        print(f'Product too small')
        return None
    if not all([a.HasProp('molAtomMapNumber') for a in pmol.GetAtoms()]):
        print(f'Product atom mapping missing')
        return None
    return True, pmol



def sanity_check(psmi: str, rsmi: str, max_n_len: int, n_dummy: int, verbose: bool = False):
    def log_message(message):
        if verbose:
            print(message)
    
    # sanity check
    if "" == psmi:
        log_message(f"Empty product in reaction")
        return None
    if "" == rsmi:
        log_message(f"Empty reactant in reaction")
        return None
    
    pmol = Chem.MolFromSmiles(psmi)
    rmol = Chem.MolFromSmiles(rsmi)
    
    if rmol is None:
        log_message(f"Failed to parse reactant")
        return None
    r_num_nodes = rmol.GetNumAtoms()
    if r_num_nodes < 5: 
        log_message(f'Reactant too small')
        return None
    
    if pmol is None:
        log_message(f'Failed to parse product')
        return None
    p_num_nodes = pmol.GetNumAtoms()
    if p_num_nodes == 1:  
        log_message(f'Product too small')
        return None  

    cano_psmi = clear_map_canonical_smiles(psmi)
    cano_rsmi = clear_map_canonical_smiles(rsmi)
    if cano_psmi is None or cano_rsmi is None:
        log_message("Failed to canonicalize SMILES")
        return None
    if cano_psmi == cano_rsmi:
        log_message(f"Product and reactant are the same")
        return None
    
    if not all([a.HasProp('molAtomMapNumber') for a in pmol.GetAtoms()]):
        log_message(f'Product atom mapping missing')
        return None
    
    p_atom_map_list = [atom.GetAtomMapNum() for atom in pmol.GetAtoms() if atom.GetAtomMapNum() > 0]
    if len(p_atom_map_list) != len(set(p_atom_map_list)):
        log_message('Warning: Repeat atom map in products')
        return None
    
    if len(p_atom_map_list) != p_num_nodes:
        log_message('Warning: Not all product atoms have non-zero mapping')
        return None
    
    r_atom_map_list = [atom.GetAtomMapNum() for atom in rmol.GetAtoms() if atom.GetAtomMapNum() > 0]
    if len(r_atom_map_list) != len(set(r_atom_map_list)):
        log_message('Warning: Repeat atom map in reactants')
        return None
    
    r_map_to_atom = {}
    p_map_to_atom = {}
    
    for atom in rmol.GetAtoms():
        map_num = atom.GetAtomMapNum()
        if map_num > 0:
            r_map_to_atom[map_num] = atom
    
    for atom in pmol.GetAtoms():
        map_num = atom.GetAtomMapNum()
        if map_num > 0:
            p_map_to_atom[map_num] = atom
    
    common_maps = set(r_map_to_atom.keys()) & set(p_map_to_atom.keys())
    for map_num in common_maps:
        r_atom = r_map_to_atom[map_num]
        p_atom = p_map_to_atom[map_num]
        r_val = r_atom.GetSymbol()
        p_val = p_atom.GetSymbol()
        if r_val != p_val:
            log_message('Warning: Mismatch atom map in product and reactants')
            return None
    
    p_only = set(p_map_to_atom.keys()) - set(r_map_to_atom.keys())
    if len(p_only) != 0:
        log_message('Warning: There are atom maps existing only in products')
        return None
    
    if r_num_nodes < p_num_nodes:
        log_message(f"Warning: reactants have fewer nodes than products")
        return None
    
    if r_num_nodes > max_n_len or p_num_nodes > max_n_len:
        log_message(f"Warning: reactants or products exceed max node limit")
        return None
    
    if n_dummy > 0:
        if r_num_nodes - p_num_nodes > n_dummy:
            log_message(f"Warning: too many dummy nodes")
            return None 

    if pmol.GetNumBonds() == 0:
        log_message(f'Warning: {rsmi}->{psmi} molecule has no bonds')
        return None
    
    if rmol.GetNumBonds() == 0:
        log_message(f'Warning: {rsmi}->{psmi} molecule has no bonds')
        return None
    
    return True, pmol, rmol


def compute_nodes_order_mapping(molecule: Mol) -> tuple[dict[int, int], Mol]:
    mol_copy = Chem.Mol(molecule)
    
    # In case if atomic map numbers do not start from 1
    order = []
    for atom in mol_copy.GetAtoms():
        if atom.GetAtomMapNum() != 0:
            order.append(atom.GetAtomMapNum())
    
    order = {
        atom_map_num: idx
        for idx, atom_map_num in enumerate(sorted(order))
    }
    
    # Handle atoms without atom map numbers and assign them to the molecule
    current_max_idx = len(order) - 1 if order else -1
    max_atom_map = max(order.keys()) if order else 0
    
    for atom in mol_copy.GetAtoms():
        if atom.GetAtomMapNum() == 0:
            current_max_idx += 1
            # Find the next available atom map number
            max_atom_map += 1
            new_atom_map = max_atom_map
            
            # Assign the new atom map number to the molecule copy
            atom.SetAtomMapNum(new_atom_map)
            order[new_atom_map] = current_max_idx
    return order, mol_copy
        
        
def clear_map_canonical_smiles(smi, canonical=True, root=-1):
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        for atom in mol.GetAtoms():
            if atom.HasProp('molAtomMapNumber'):
                atom.ClearProp('molAtomMapNumber')
        return Chem.MolToSmiles(mol, isomericSmiles=True, rootedAtAtom=root, canonical=canonical)
    else:
        return None


def get_cano_map_number(smi, root=-1):
    atommap_mol = Chem.MolFromSmiles(smi)
    canonical_mol = Chem.MolFromSmiles(clear_map_canonical_smiles(smi, root=root))
    cano2atommapIdx = atommap_mol.GetSubstructMatch(canonical_mol)
    correct_mapped = [
        canonical_mol.GetAtomWithIdx(i).GetSymbol() == atommap_mol.GetAtomWithIdx(index).GetSymbol()
        for i,index in enumerate(cano2atommapIdx)
    ]
    atom_number = len(canonical_mol.GetAtoms())
    if np.sum(correct_mapped) < atom_number or len(cano2atommapIdx) < atom_number:
        cano2atommapIdx = [0] * atom_number
        atommap2canoIdx = canonical_mol.GetSubstructMatch(atommap_mol)
        if len(atommap2canoIdx) != atom_number:
            return None
        for i, index in enumerate(atommap2canoIdx):
            cano2atommapIdx[index] = i
    id2atommap = [atom.GetAtomMapNum() for atom in atommap_mol.GetAtoms()]
    final_map = [id2atommap[cano2atommapIdx[i]] for i in range(atom_number)]
    max_atom_map = max(final_map)
    for i, atom_map in enumerate(final_map):
        if atom_map == 0:
            final_map[i] = max_atom_map + 1
            max_atom_map += 1
    return final_map


def get_root_id(mol, root_map_number):
    root = -1
    for i, atom in enumerate(mol.GetAtoms()):
        if atom.GetAtomMapNum() == root_map_number:
            root = i
            break
    return root


def get_cano_smi_with_map(cano_smi, cano_map):
    cano_mol = Chem.MolFromSmiles(cano_smi)
    if cano_mol is None:
        return None
    
    for i, atom in enumerate(cano_mol.GetAtoms()):
        atom.SetAtomMapNum(cano_map[i])
    cano_smi_with_atom_map = Chem.MolToSmiles(cano_mol, canonical=False)

    # sanity check: cano_smi_with_atom_map has correct structure
    ori_cano_mol = Chem.MolFromSmiles(cano_smi)
    if ori_cano_mol is None:
        return None
        
    ori_atom_list = ori_cano_mol.GetAtoms()
    current_atom_list = cano_mol.GetAtoms()
    
    if len(ori_atom_list) != len(current_atom_list):
        print('Error in computing graph due to atoms not in the graph vocab')
        return None
        
    for i, ori_atom in enumerate(ori_atom_list):
        current_atom = current_atom_list[i]
        if ori_atom.GetSymbol() != current_atom.GetSymbol():
            print(f"Warning: atom symbol mismatch at position {i}")
            return None
        if cano_map[i] != current_atom.GetAtomMapNum():
            print(f"Warning: atom map number mismatch at position {i}")
            return None

    return cano_smi_with_atom_map     

     
def canonicalize_reaction(
        reaction: str, aug_size: int = 0
    ) -> tuple[list, list] | None:
    reactant, _ , product = reaction.split('>')
    pro_mol = Chem.MolFromSmiles(product)

    reactant = reactant.split(".")
    
    # Initialize with -1 (no root specified) and handle augmentation
    if aug_size == 0:
        product_roots = [-1]
    else:
        product_roots = [-1]
        pro_atom_map_numbers = list(map(int, re.findall(r"(?<=:)\d+", product)))
        # Use set to avoid duplicates more efficiently
        available_roots = set(pro_atom_map_numbers)
        target_size = min(aug_size + 1, len(available_roots) + 1)  # +1 for the -1 case
        
        while len(product_roots) < target_size and available_roots:
            new_root = random.choice(list(available_roots))
            product_roots.append(new_root)
            available_roots.remove(new_root)
    
    times = len(product_roots)
    
    product_smiles = []
    reactant_smiles = []
    
    for k in range(times):
        valid_aug = True
        pro_root_atom_map = product_roots[k]
        pro_root = get_root_id(pro_mol, root_map_number=pro_root_atom_map)
        cano_atom_map = get_cano_map_number(product, root=pro_root)
        if cano_atom_map is None:
            print(f"Fail to match the canonical form of product")
            product_smiles.append(None)
            reactant_smiles.append(None)
            continue
        
        pro_smi = clear_map_canonical_smiles(product, canonical=True, root=pro_root)
        pro_smi_with_map = get_cano_smi_with_map(pro_smi, cano_atom_map)
        if pro_smi_with_map is None:
            print(f"Failed to create product SMILES with map")
            product_smiles.append(None)
            reactant_smiles.append(None)
            continue
        product_smiles.append(pro_smi_with_map)
        
        aligned_reactants = []
        aligned_reactants_order = []
        rea_atom_map_numbers = [list(map(int, re.findall(r"(?<=:)\d+", rea))) for rea in reactant]
        used_indices = []
        
        for i, rea_map_number in enumerate(rea_atom_map_numbers):
            for j, map_number in enumerate(cano_atom_map):
                # select mapping reactans
                if map_number in rea_map_number:
                    temp_r_mol = Chem.MolFromSmiles(reactant[i])
                    if temp_r_mol is None:
                        print(f"Failed to parse reactant: {reactant[i]}")
                        valid_aug = False
                        break
                    rea_root = get_root_id(temp_r_mol, root_map_number=map_number)
                    cano_atom_map_r = get_cano_map_number(reactant[i], root=rea_root)
                    if cano_atom_map_r is None:
                        print(f"Fail to match the canonical form of reactant")
                        valid_aug = False
                        break
                    
                    rea_smi = clear_map_canonical_smiles(reactant[i], canonical=True, root=rea_root)
                    rea_smi_with_map = get_cano_smi_with_map(rea_smi, cano_atom_map_r)
                    if rea_smi_with_map is None:
                        print(f"Failed to create reactant SMILES with map")
                        valid_aug = False
                        break
                    
                    aligned_reactants.append(rea_smi_with_map)
                    aligned_reactants_order.append(j)
                    used_indices.append(i)
                    break
            if not valid_aug:
                break
        if not valid_aug:
            product_smiles.append(None)
            reactant_smiles.append(None)
        else:
            sorted_reactants = sorted(list(zip(aligned_reactants, aligned_reactants_order)), key=lambda x: x[1])
            aligned_reactants = [item[0] for item in sorted_reactants]
            reactant_smi = ".".join(aligned_reactants)
            reactant_smiles.append(reactant_smi)
    return product_smiles, reactant_smiles


def get_reaction_center(rxn_smiles):
    try:
        reactants, _, products = rxn_smiles.split('>')
        reactant_mols = [Chem.MolFromSmiles(r) for r in reactants.split('.') if r]
        product_mols = [Chem.MolFromSmiles(p) for p in products.split('.') if p]
        
        if not reactant_mols or not product_mols:
            return None
        
        product_atom_maps = set()
        for mol in product_mols:
            if mol is None:
                continue
            for atom in mol.GetAtoms():
                map_num = atom.GetAtomMapNum()
                if map_num:
                    product_atom_maps.add(map_num)
        
        formed_bonds_atoms = []
        broken_bonds_atoms = []
        bond_order_changes_atoms = []
        charge_changes_atoms = []
        h_count_changes_atoms = []
        chirality_changes_atoms = []
        aromatic_changes_atoms = []
        hybridization_changes_atoms = []
        
        reactant_bonds = {}
        for mol in reactant_mols:
            if mol is None:
                continue
            for bond in mol.GetBonds():
                atom1 = bond.GetBeginAtom()
                atom2 = bond.GetEndAtom()
                map1 = atom1.GetAtomMapNum()
                map2 = atom2.GetAtomMapNum()
                if map1 and map2:
                    bond_key = tuple(sorted([map1, map2]))
                    bond_order = bond.GetBondTypeAsDouble()
                    reactant_bonds[bond_key] = bond_order
        
        product_bonds = {}
        for mol in product_mols:
            if mol is None:
                continue
            for bond in mol.GetBonds():
                atom1 = bond.GetBeginAtom()
                atom2 = bond.GetEndAtom()
                map1 = atom1.GetAtomMapNum()
                map2 = atom2.GetAtomMapNum()
                if map1 and map2:
                    bond_key = tuple(sorted([map1, map2]))
                    bond_order = bond.GetBondTypeAsDouble()
                    product_bonds[bond_key] = bond_order
        
        formed_bonds = set(product_bonds.keys()) - set(reactant_bonds.keys())
        for bond in formed_bonds:
            for atom_map in bond:
                if atom_map in product_atom_maps:
                    formed_bonds_atoms.append(atom_map)
        
        broken_bonds = set(reactant_bonds.keys()) - set(product_bonds.keys())
        for bond in broken_bonds:
            for atom_map in bond:
                if atom_map in product_atom_maps:
                    broken_bonds_atoms.append(atom_map)
        
        for bond_key in set(reactant_bonds.keys()) & set(product_bonds.keys()):
            if reactant_bonds[bond_key] != product_bonds[bond_key]:
                for atom_map in bond_key:
                    if atom_map in product_atom_maps:
                        bond_order_changes_atoms.append(atom_map)
        
        reactant_atom_props = {}
        for mol in reactant_mols:
            if mol is None:
                continue
            for atom in mol.GetAtoms():
                map_num = atom.GetAtomMapNum()
                if map_num:
                    reactant_atom_props[map_num] = {
                        'charge': atom.GetFormalCharge(),
                        'h_count': atom.GetTotalNumHs(),
                        'aromatic': atom.GetIsAromatic(),
                        'hybridization': str(atom.GetHybridization()),
                        'chirality': str(atom.GetChiralTag()) if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED else None
                    }
        
        product_atom_props = {}
        for mol in product_mols:
            if mol is None:
                continue
            for atom in mol.GetAtoms():
                map_num = atom.GetAtomMapNum()
                if map_num:
                    product_atom_props[map_num] = {
                        'charge': atom.GetFormalCharge(),
                        'h_count': atom.GetTotalNumHs(),
                        'aromatic': atom.GetIsAromatic(),
                        'hybridization': str(atom.GetHybridization()),
                        'chirality': str(atom.GetChiralTag()) if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED else None
                    }
        
        for map_num in product_atom_maps:
            if map_num in reactant_atom_props and map_num in product_atom_props:
                r_props = reactant_atom_props[map_num]
                p_props = product_atom_props[map_num]
                
                if r_props['charge'] != p_props['charge']:
                    charge_changes_atoms.append(map_num)
                
                if r_props['h_count'] != p_props['h_count']:
                    h_count_changes_atoms.append(map_num)
                
                if r_props['aromatic'] != p_props['aromatic']:
                    aromatic_changes_atoms.append(map_num)
                
                if r_props['hybridization'] != p_props['hybridization']:
                    hybridization_changes_atoms.append(map_num)
                
                if r_props['chirality'] != p_props['chirality']:
                    chirality_changes_atoms.append(map_num)
            
            elif map_num in product_atom_props and map_num not in reactant_atom_props:
                if product_atom_props[map_num]['chirality'] is not None:
                    chirality_changes_atoms.append(map_num)
        
        rc_center_dict = {
            'formed_bonds': formed_bonds_atoms,
            'broken_bonds': broken_bonds_atoms,
            'bond_order_changes': bond_order_changes_atoms,
            'charge_changes': charge_changes_atoms,
            'h_count_changes': h_count_changes_atoms,
            'chirality_changes': chirality_changes_atoms,
            'aromatic_changes': aromatic_changes_atoms,
            'hybridization_changes': hybridization_changes_atoms
        }
        
        rc_center_dict['has_changes'] = any([
            formed_bonds_atoms,
            broken_bonds_atoms,
            bond_order_changes_atoms,
            charge_changes_atoms,
            h_count_changes_atoms,
            chirality_changes_atoms,
            aromatic_changes_atoms,
            hybridization_changes_atoms
        ])
        
        return rc_center_dict
        
    except Exception as e:
        print(f"Error processing reaction: {e}")
        return None
        
    
def compute_graph(
        molecule: Mol, mapping: dict[int, int],
        max_num_nodes: int,
        node_types: dict, edge_types: dict,
        dn_last: bool
    ):
    max_num_nodes = max(molecule.GetNumAtoms(), max_num_nodes)  # in case |reactants|-|product| > max_n_dummy_nodes

    if dn_last:
        adjusted_mapping = mapping
    else:
        num_dummy = max_num_nodes - molecule.GetNumAtoms()
        assert num_dummy >= 0
        adjusted_mapping = {k: v + num_dummy for k, v in mapping.items()}


    type_idx = [len(node_types) - 1] * max_num_nodes
    for atom in molecule.GetAtoms():
        type_idx[adjusted_mapping[atom.GetAtomMapNum()]] = node_types[atom.GetSymbol()]

    num_classes = len(node_types)
    x = F.one_hot(torch.tensor(type_idx), num_classes=num_classes).float()

    row, col, edge_type = [], [], []
    for bond in molecule.GetBonds():
        start_atom_map_num = molecule.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetAtomMapNum()
        end_atom_map_num = molecule.GetAtomWithIdx(bond.GetEndAtomIdx()).GetAtomMapNum()
        start, end = adjusted_mapping[start_atom_map_num], adjusted_mapping[end_atom_map_num]
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [edge_types[bond.GetBondType()] + 1]

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    edge_attr = F.one_hot(edge_type, num_classes=len(edge_types) + 1).float()

    return x, edge_index, edge_attr


def compute_atom_map(pmol: Mol, rmol: Mol, alignment: bool) -> tuple[dict[int, int], dict[int, int]]:
    p_map, r_map = {}, {}
    for atom in pmol.GetAtoms():
        p_map[atom.GetAtomMapNum()] = atom.GetIdx()
    if alignment:
        start_idx = max(p_map.values()) + 1 if p_map else 0
        for atom in rmol.GetAtoms():
            atom_map_num = atom.GetAtomMapNum()
            if atom_map_num in p_map:
                r_map[atom_map_num] = p_map[atom_map_num]
            else:
                r_map[atom_map_num] = start_idx
                start_idx += 1
    else:
        for atom in rmol.GetAtoms():
            r_map[atom.GetAtomMapNum()] = atom.GetIdx()
    return p_map, r_map
        


def shuffle_order(
        r_x: Tensor, r_edge_index: Tensor, r_edge_attr: Tensor,
        p_x: Tensor, p_edge_index: Tensor, p_edge_attr: Tensor,
        coord: Tensor = None
    ) -> tuple:
    assert len(r_x) == len(p_x)
    perm = torch.randperm(len(p_x)).long()
    r_x, r_edge_index, r_edge_attr = shuffle_graph(
        r_x, r_edge_index, r_edge_attr, perm
    )
    p_x, p_edge_index, p_edge_attr = shuffle_graph(
        p_x, p_edge_index, p_edge_attr, perm
    )

    if coord is not None:
        coord = coord[perm]
        return r_x, p_x, r_edge_index, p_edge_index, \
        r_edge_attr, p_edge_attr, coord
    else:
        return r_x, p_x, r_edge_index, p_edge_index, \
        r_edge_attr, p_edge_attr


def shuffle_graph(
        x: Tensor, edge_index: Tensor, edge_attr: Tensor,
        perm: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor, Tensor]:
    n_nodes = len(x)
    if perm is None:
        perm = torch.randperm(n_nodes).long()
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(n_nodes)

    x = x[perm]
    edge_index = inv_perm[edge_index]
    edge_index, edge_attr = sort_edges(edge_index, edge_attr, n_nodes)
    return x, edge_index, edge_attr


def sort_edges(
        edge_index: Tensor, edge_attr: Tensor,
        max_num_nodes: int
    ):
    if len(edge_attr) != 0:
        perm = (edge_index[0] * max_num_nodes + edge_index[1]).argsort()
        edge_index = edge_index[:, perm]
        edge_attr = edge_attr[perm]

    return edge_index, edge_attr


def to_dense(
        x: Tensor, edge_index: Tensor, edge_attr: Tensor, batch: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
    X, node_mask = to_dense_batch(x=x, batch=batch)
    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    max_num_nodes = X.size(1)
    E = to_dense_adj(
        edge_index=edge_index, batch=batch,
        edge_attr=edge_attr, max_num_nodes=max_num_nodes
    )

    assert len(E.shape) == 4
    if E.shape[-1] != 0:
        no_edge = torch.sum(E, dim=3) == 0
        first_elt = E[:, :, :, 0]
        assert torch.all(first_elt == 0)
        first_elt[no_edge] = 1
        E[:, :, :, 0] = first_elt
        diag = torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
        E[diag] = 0

    x_mask = node_mask.unsqueeze(-1)
    e_mask1 = x_mask.unsqueeze(-2)
    e_mask2 = x_mask.unsqueeze(-3)
    E = E * e_mask1 * e_mask2
    return X, E, node_mask



def dense2smiles(
        X: Tensor, E: Tensor, node_mask: Tensor,
        x_dec: dict[int, str], e_dec: dict[int, Chem.rdchem.BondType],
        canonical: bool = True,
        perms: Optional[list[Tensor]] = None
    ) -> list[str | None]:
    if perms is not None:
        assert not canonical

    n_nodes = node_mask.sum(-1)

    mol_list = []
    for i in range(X.size(0)):
        n = n_nodes[i]
        if X.ndim == 3:    # one-hot
            atom_types = torch.argmax(X[i, :n], dim=-1)
        elif X.ndim == 2:  # ids
            atom_types = X[i, :n]
        else:
            raise ValueError()
        
        if E.ndim == 4:    # one-hot
            edge_types = torch.argmax(E[i, :n, :n], dim=-1)
        elif E.ndim == 3:  # ids
            edge_types = E[i, :n, :n]
        else:
            raise ValueError()
        
        if perms is not None:
            perm = perms[i]
            # perm = torch.randperm(n=len(perm), device=perm.device)
            if perm.max() >= len(atom_types):
                mol_list.append(None)
                continue
            atom_types = atom_types[perm]
            edge_types = edge_types[perm][:, perm]

        try:
            mol = _build_molecule(atom_types, edge_types, x_dec, e_dec)
            mol_list.append(Chem.MolToSmiles(mol, canonical=canonical))
        except:
            mol_list.append(None)
    
    return mol_list


def _build_molecule(
        atom_types: Tensor, edge_types: Tensor,
        x_dec: dict[int, str], e_dec: dict[int, Chem.rdchem.BondType]
    ):
    mol = Chem.RWMol()
    dummy_atoms = set()
    mapping = {}
    j = 0
    for i, atom in enumerate(atom_types):
        a = Chem.Atom(x_dec[atom.item()])
        if a.GetSymbol() == '*':
            dummy_atoms.add(i)
            continue

        mol.AddAtom(a)
        mapping[i] = j
        j += 1

    edge_types = torch.triu(edge_types)
    all_bonds = torch.nonzero(edge_types)
    for i, bond in enumerate(all_bonds):
        if bond[0].item() == bond[1].item():
            continue
        if bond[0].item() in dummy_atoms:
            continue
        if bond[1].item() in dummy_atoms:
            continue

        mol.AddBond(mapping[bond[0].item()], mapping[bond[1].item()], e_dec[edge_types[bond[0], bond[1]].item()])
    return mol



