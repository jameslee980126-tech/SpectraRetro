import collections, hashlib, math
import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
SALT = 'aryl20260926|'
ELEMENTS = ['C','O','N','F','S','Cl','Br','I','P','B','Si','Se','Sn','Zn','Cu','Mg','Li','Na','K','Ca','Al','Fe','Pd','H','As','Hg']
LABELS = ['NONE'] + ['ONE_' + e for e in ELEMENTS if e != 'H'] + ['ONE_OTHER','MULTI']
def stopped(): pass
def canonical_product(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError('Unparseable product; must quarantine, not hash raw text')
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)

def split_product(smiles):
    key = canonical_product(smiles)
    bucket = int(hashlib.sha256((SALT + key).encode('utf-8')).hexdigest()[:8], 16) % 10
    return (key, 'fit' if bucket < 6 else 'dev' if bucket < 8 else 'audit', bucket)

def clean_synthon(smiles):
    """Rebuild known graph, discarding reference-inherited H/radicals/maps."""
    old = Chem.MolFromSmiles(str(smiles), sanitize=False)
    if old is None:
        return None
    graph = Chem.RWMol()
    for atom in old.GetAtoms():
        new = Chem.Atom(atom.GetAtomicNum())
        new.SetFormalCharge(atom.GetFormalCharge())
        new.SetIsAromatic(atom.GetIsAromatic())
        new.SetNoImplicit(False)
        new.SetNumExplicitHs(0)
        new.SetNumRadicalElectrons(0)
        graph.AddAtom(new)
    for bond in old.GetBonds():
        graph.AddBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondType())
    mol = graph.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        try:
            mol.UpdatePropertyCache(strict=False)
            Chem.GetSymmSSSR(mol)
        except Exception:
            return None
    return mol

def anchors(row, mol):
    try:
        raw = str(row['anchors']).strip()
        indices = [int(float(v)) for v in raw.replace(';', ',').replace('|', ',').split(',') if v.strip()]
        if indices and all((0 <= i < mol.GetNumAtoms() for i in indices)):
            return indices
    except Exception:
        pass
    original = Chem.MolFromSmiles(str(row['fragments']), sanitize=False)
    centers = {int(float(row[k])) for k in ['center1', 'center2'] if str(row.get(k, '')).strip()}
    return [a.GetIdx() for a in original.GetAtoms() if a.GetAtomMapNum() in centers] if original else []

def fp(mol, n=1024, roots=None):
    out = np.zeros(n, dtype=np.uint8)
    if mol is not None:
        try:
            bits = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n, fromAtoms=roots or [], useChirality=False)
            DataStructs.ConvertToNumpyArray(bits, out)
        except Exception:
            pass
    return out

def descriptors(mol, anchor_ids):
    out = np.zeros(32, dtype=np.float32)
    if mol is None:
        out[-1] = 1
        return out
    atoms = list(mol.GetAtoms())
    n = max(len(atoms), 1)
    out[:8] = [math.log1p(n), sum((a.GetIsAromatic() for a in atoms)) / n, sum((a.GetAtomicNum() not in (1, 6) for a in atoms)) / n, sum((a.GetFormalCharge() for a in atoms)) / 4, sum((a.IsInRing() for a in atoms)) / n, len(anchor_ids) / 2, sum((a.GetAtomicNum() in (9, 17, 35, 53) for a in atoms)) / n, sum((a.GetAtomicNum() == 7 for a in atoms)) / n]
    try:
        out[8:12] = [Descriptors.MolWt(mol) / 500, Descriptors.MolLogP(mol) / 5, rdMolDescriptors.CalcTPSA(mol) / 100, rdMolDescriptors.CalcNumRotatableBonds(mol) / 10]
    except Exception:
        pass
    charges = None
    try:
        c = Chem.Mol(mol)
        AllChem.ComputeGasteigerCharges(c)
        charges = np.array([float(a.GetProp('_GasteigerCharge')) for a in c.GetAtoms()])
        if not np.isfinite(charges).all():
            charges = None
    except Exception:
        pass
    if anchor_ids:
        a = atoms[anchor_ids[0]]
        neighbors = list(a.GetNeighbors())
        out[12:24] = [a.GetAtomicNum() / 53, a.GetDegree() / 4, a.GetIsAromatic(), a.IsInRing(), a.GetFormalCharge() / 2, int(a.GetHybridization()) / 8, sum((b.GetAtomicNum() not in (1, 6) for b in neighbors)) / 4, sum((b.GetIsAromatic() for b in neighbors)) / 4, sum((b.GetAtomicNum() == 7 for b in neighbors)) / 4, sum((b.GetAtomicNum() == 8 for b in neighbors)) / 4, sum((b.GetFormalCharge() for b in neighbors)) / 4, sum((b.IsInRing() for b in neighbors)) / 4]
        if charges is not None:
            q = charges[[b.GetIdx() for b in neighbors]]
            out[24:28] = [charges[a.GetIdx()], q.mean() if len(q) else 0, q.std() if len(q) else 0, np.max(np.abs(charges))]
        else:
            out[28] = 1
        nearby = {b.GetIdx() for b in neighbors}
        nearby |= {z.GetIdx() for b in neighbors for z in b.GetNeighbors()}
        out[29] = sum((atoms[i].GetAtomicNum() not in (1, 6) for i in nearby)) / 10
        out[30] = len(nearby) / 12
    return np.nan_to_num(out).clip(-10, 10)

def feature_rows(rows):
    """Only whitelisted observable columns enter feature computation."""
    view = [{k: r.get(k, '') for k in ['uuid', '_source_row', 'fragments', 'product', 'anchors', 'center1', 'center2']} for r in rows]
    partners = collections.defaultdict(list)
    for r in view:
        partners[str(r['_source_row'])].append(r)
    mats = []
    info = []
    for (idx, r) in enumerate(view):
        if idx % 200 == 0:
            stopped()
        mol = clean_synthon(r['fragments'])
        anc = anchors(r, mol) if mol else []
        local = np.concatenate([fp(mol, 1024), fp(mol, 512, anc) if anc else np.zeros(512, dtype=np.uint8)])
        product = Chem.MolFromSmiles(str(r['product']))
        if product:
            for a in product.GetAtoms():
                a.SetAtomMapNum(0)
        partner = np.zeros(1024, dtype=np.uint8)
        for other in partners[str(r['_source_row'])]:
            if other['uuid'] != r['uuid']:
                partner |= fp(clean_synthon(other['fragments']))
        context = np.concatenate([fp(product), partner])
        desc = descriptors(mol, anc)
        mats.append(np.concatenate([local, context, desc]).astype(np.float32))
        (key, split, bucket) = split_product(r['product'])
        info.append({'uuid': str(r['uuid']), 'source_row': str(r['_source_row']), 'product_key': key, 'split': split, 'bucket': bucket, 'parse_failed': mol is None})
        if idx % 2000 == 0:
            print('features', idx, len(view), flush=True)
    return (np.stack(mats), info)

class Head(torch.nn.Module):

    def __init__(self, d):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(d, 512), torch.nn.ReLU(), torch.nn.Dropout(0.1), torch.nn.Linear(512, 256), torch.nn.ReLU(), torch.nn.Dropout(0.1), torch.nn.Linear(256, len(LABELS)))

    def forward(self, x):
        return self.net(x)