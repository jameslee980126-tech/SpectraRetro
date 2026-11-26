import math
from collections import Counter
from functools import lru_cache
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
HALOGENS = {'F', 'Cl', 'Br', 'I'}
ELEMENTS = ('C', 'N', 'O', 'F', 'Cl', 'Br', 'I', 'B', 'Si', 'P', 'S')
MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
@lru_cache(maxsize=None)
def molecule_from_smiles(smiles):
    mol = Chem.MolFromSmiles(str(smiles), sanitize=False)
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    mol.UpdatePropertyCache(strict=False)
    try:
        Chem.GetSymmSSSR(mol)
    except Exception:
        pass
    return mol

@lru_cache(maxsize=None)
def canonical_molecule(smiles):
    smiles = str(smiles).strip()
    if not smiles or smiles == 'MISSING':
        return None
    mol = molecule_from_smiles(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None

@lru_cache(maxsize=None)
def canonical_precursor_set(value):
    parts = []
    for smiles in str(value).split('.'):
        smiles = smiles.strip()
        if not smiles:
            continue
        normalized = canonical_molecule(smiles)
        if normalized is None:
            return None
        parts.append(normalized)
    return tuple(sorted(set(parts))) if parts else None

def safe_ratio(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0

@lru_cache(maxsize=None)
def molecule_features(smiles):
    mol = molecule_from_smiles(smiles)
    if mol is None:
        return None
    counts = Counter((atom.GetSymbol() for atom in mol.GetAtoms()))
    features = {f'element_{element}': float(counts[element]) for element in ELEMENTS}
    features['atoms'] = float(mol.GetNumAtoms())
    features['heavy_atoms'] = float(sum((atom.GetAtomicNum() > 1 for atom in mol.GetAtoms())))
    features['aromatic_atoms'] = float(sum((atom.GetIsAromatic() for atom in mol.GetAtoms())))
    features['formal_charge_abs'] = float(sum((abs(atom.GetFormalCharge()) for atom in mol.GetAtoms())))
    features['radical_electrons'] = float(sum((atom.GetNumRadicalElectrons() for atom in mol.GetAtoms())))
    features['bonds'] = float(mol.GetNumBonds())
    features['single_bonds'] = 0.0
    features['double_bonds'] = 0.0
    features['triple_bonds'] = 0.0
    features['aromatic_bonds'] = 0.0
    features['carbon_halogen_bonds'] = 0.0
    features['carbon_boron_bonds'] = 0.0
    features['carbon_silicon_bonds'] = 0.0
    features['carbon_metal_bonds'] = 0.0
    features['nitrogen_oxygen_bonds'] = 0.0
    features['nitrogen_nitrogen_bonds'] = 0.0
    features['oxygen_oxygen_bonds'] = 0.0
    features['carbonyl_bonds'] = 0.0
    features['alkene_bonds'] = 0.0
    features['alkyne_bonds'] = 0.0
    features['acyl_halide_motifs'] = 0.0
    metals = {'Li', 'Na', 'K', 'Mg', 'Ca', 'Zn', 'Cu', 'Sn', 'Pd', 'Ni', 'Fe', 'Co'}
    for bond in mol.GetBonds():
        a = bond.GetBeginAtom()
        b = bond.GetEndAtom()
        symbols = {a.GetSymbol(), b.GetSymbol()}
        bond_type = bond.GetBondType()
        if bond.GetIsAromatic():
            features['aromatic_bonds'] += 1.0
        elif bond_type == Chem.BondType.SINGLE:
            features['single_bonds'] += 1.0
        elif bond_type == Chem.BondType.DOUBLE:
            features['double_bonds'] += 1.0
        elif bond_type == Chem.BondType.TRIPLE:
            features['triple_bonds'] += 1.0
        if 'C' in symbols and symbols & HALOGENS:
            features['carbon_halogen_bonds'] += 1.0
        if symbols == {'B', 'C'}:
            features['carbon_boron_bonds'] += 1.0
        if symbols == {'C', 'Si'}:
            features['carbon_silicon_bonds'] += 1.0
        if 'C' in symbols and symbols & metals:
            features['carbon_metal_bonds'] += 1.0
        if symbols == {'N', 'O'}:
            features['nitrogen_oxygen_bonds'] += 1.0
        if symbols == {'N'}:
            features['nitrogen_nitrogen_bonds'] += 1.0
        if symbols == {'O'}:
            features['oxygen_oxygen_bonds'] += 1.0
        if symbols == {'C', 'O'} and bond_type == Chem.BondType.DOUBLE:
            features['carbonyl_bonds'] += 1.0
        if symbols == {'C'} and bond_type == Chem.BondType.DOUBLE:
            features['alkene_bonds'] += 1.0
        if symbols == {'C'} and bond_type == Chem.BondType.TRIPLE:
            features['alkyne_bonds'] += 1.0
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != 'C':
            continue
        has_carbonyl = any((neighbor.GetSymbol() == 'O' and mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondType() == Chem.BondType.DOUBLE for neighbor in atom.GetNeighbors()))
        has_halogen = any((neighbor.GetSymbol() in HALOGENS for neighbor in atom.GetNeighbors()))
        if has_carbonyl and has_halogen:
            features['acyl_halide_motifs'] += 1.0
    try:
        features['rings'] = float(mol.GetRingInfo().NumRings())
    except Exception:
        features['rings'] = 0.0
    return features

@lru_cache(maxsize=None)
def fingerprint(smiles):
    mol = molecule_from_smiles(smiles)
    if mol is None:
        return None
    try:
        return MORGAN.GetFingerprint(mol)
    except Exception:
        return None

def tanimoto(smiles_a, smiles_b):
    fp_a = fingerprint(smiles_a)
    fp_b = fingerprint(smiles_b)
    if fp_a is None or fp_b is None:
        return 0.0
    return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))

def summarize_candidate(candidate, product_smiles):
    component_features = [molecule_features(smiles) for smiles in candidate]
    component_features = [features for features in component_features if features is not None]
    if not component_features:
        return {}
    keys = sorted(component_features[0])
    features = {}
    for key in keys:
        values = [item[key] for item in component_features]
        features[f'candidate_total_{key}'] = float(sum(values))
        features[f'candidate_max_{key}'] = float(max(values))
        features[f'candidate_min_{key}'] = float(min(values))
    component_atoms = [item['atoms'] for item in component_features]
    features['candidate_component_count'] = float(len(candidate))
    features['candidate_component_atom_imbalance'] = float(max(component_atoms) - min(component_atoms))
    product_features = molecule_features(product_smiles) if product_smiles else None
    if product_features is not None:
        for key in keys:
            candidate_total = features[f'candidate_total_{key}']
            product_value = product_features[key]
            features[f'product_{key}'] = float(product_value)
            features[f'candidate_minus_product_{key}'] = float(candidate_total - product_value)
            features[f'candidate_product_ratio_{key}'] = safe_ratio(candidate_total, product_value)
    similarities = [tanimoto(smiles, product_smiles) for smiles in candidate]
    features['product_similarity_max'] = float(max(similarities))
    features['product_similarity_min'] = float(min(similarities))
    features['product_similarity_mean'] = float(sum(similarities) / len(similarities))
    return features