"""Connect generated draws and observed synthons to the SpectraRetro ranking stage."""
from rdkit import Chem

from .runtime import get_ranker


def observed_synthon(mol, anchors):
    """Serialize an observed graph and remap zero-based anchor indices explicitly."""
    if mol is None:
        raise ValueError('Unreadable observed synthon')
    copy = Chem.Mol(mol)
    for atom in copy.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx()+1)
    smiles = Chem.MolToSmiles(copy, canonical=False, isomericSmiles=True)
    parsed = Chem.MolFromSmiles(smiles, sanitize=False)
    mapping = {a.GetAtomMapNum()-1:a.GetIdx() for a in parsed.GetAtoms()}
    if any(i not in mapping for i in anchors):
        raise ValueError('Observed anchor outside fragment')
    return dict(fragments=smiles, anchors=';'.join(str(mapping[i]) for i in anchors))


def product_fragments(product, bond_atoms, descriptors):
    """Only the current product and predicted cut define these synthons."""
    mol = Chem.MolFromSmiles(product)
    rw = Chem.RWMol(mol)
    rw.RemoveBond(*bond_atoms)
    groups = Chem.GetMolFrags(rw.GetMol(), asMols=False)
    mols = Chem.GetMolFrags(rw.GetMol(), asMols=True, sanitizeFrags=False)
    out = []
    for desc in descriptors:
        wanted = set(desc['atom_indices'])
        matches = [(group,m) for group,m in zip(groups,mols) if set(group)==wanted]
        if len(matches)!=1:
            raise ValueError('Predicted center fragment correspondence changed')
        group,m = matches[0]
        out.append(observed_synthon(m,[list(group).index(desc['anchor_atom'])]))
    return out


def rank_generated(product, observations, draws, class_id=None, reaction_id='live'):
    reaction = dict(reaction_id=reaction_id,product=product,fragments=[dict(uuid=str(j),**obs,draws=d)
        for j,(obs,d) in enumerate(zip(observations,draws))])
    if len(observations)!=len(draws):
        raise ValueError('Missing companion fragment draws')
    if class_id is not None:
        reaction['class_id']=class_id
    return get_ranker('known' if class_id is not None else 'unknown').rank(reaction)
