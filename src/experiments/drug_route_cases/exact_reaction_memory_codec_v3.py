"""Isolated constitutional decoder for source-pair training-condition diagnostics.

Known atoms/bonds/coordinates are supplied explicitly. Unknown elements and
connectivity must come from generated XYZ. This is NOT product-only inference.
"""
import numpy as np
from rdkit import Chem

def canonical(smiles):
    mol=Chem.MolFromSmiles(smiles)
    if mol is None:raise ValueError('Invalid molecule')
    for atom in mol.GetAtoms():atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol,isomericSmiles=False)

def core_molecule(desc):
    rw=Chem.RWMol()
    for item in desc['atoms']:
        atom=Chem.Atom(item['atomic_number']);atom.SetFormalCharge(item['formal_charge'])
        atom.SetIsAromatic(item['aromatic']);atom.SetNumExplicitHs(item['explicit_h'])
        atom.SetNoImplicit(False);rw.AddAtom(atom)
    for a,b,kind in desc['bonds']:rw.AddBond(a,b,getattr(Chem.BondType,kind))
    return rw.GetMol()

def reconstruct(module,desc,xyz,elements):
    xyz=np.asarray(xyz);elements=np.asarray(elements);n=len(desc['atoms'])
    assert len(xyz)==len(elements)>=n
    assert [module.ELEMENTS[int(t)] for t in elements[:n]]==[a['symbol'] for a in desc['atoms']]
    known=np.asarray(desc['coordinates'])
    assert np.abs((xyz[:n]-xyz[:n].mean(0))-(known-known.mean(0))).max()<1e-4
    if len(elements)==n:
        mol=core_molecule(desc)
        try:
            Chem.SanitizeMol(mol)
            return canonical(Chem.MolToSmiles(mol)),None
        except (ValueError,RuntimeError):return None,'zero_completion_core_sanitize_failed'
    text=str(len(elements))+'\n\n'+''.join(f'{module.ELEMENTS[int(t)]} {p[0]:.9f} {p[1]:.9f} {p[2]:.9f}\n' for p,t in zip(xyz,elements))
    conversion=module.ob.OBConversion();assert conversion.SetInFormat('xyz')
    perceived=module.ob.OBMol();assert conversion.ReadString(perceived,text)
    rw=Chem.RWMol()
    for i,t in enumerate(elements):
        src=perceived.GetAtom(i+1)
        if i<n:
            spec=desc['atoms'][i];atom=Chem.Atom(spec['atomic_number'])
            atom.SetFormalCharge(spec['formal_charge']);atom.SetIsAromatic(spec['aromatic'])
            atom.SetNumExplicitHs(spec['explicit_h'])
        else:
            atom=Chem.Atom(src.GetAtomicNum());atom.SetFormalCharge(src.GetFormalCharge())
            atom.SetIsAromatic(src.IsAromatic())
        if atom.GetAtomicNum()==7 and atom.GetIsAromatic():
            atom.SetNumExplicitHs(max(atom.GetNumExplicitHs(),src.GetImplicitHCount(),3-src.GetExplicitValence()))
        atom.SetNoImplicit(False);rw.AddAtom(atom)
    for a,b,kind in desc['bonds']:rw.AddBond(a,b,getattr(Chem.BondType,kind))
    types={1:Chem.BondType.SINGLE,2:Chem.BondType.DOUBLE,3:Chem.BondType.TRIPLE}
    crossings=[];unknown_edges=set()
    for bond in module.ob.OBMolBondIter(perceived):
        a,b=bond.GetBeginAtomIdx()-1,bond.GetEndAtomIdx()-1
        if a<n and b<n:continue
        if (a<n)!=(b<n):crossings.append(a if a<n else b)
        kind=Chem.BondType.AROMATIC if bond.IsAromatic() else types.get(bond.GetBondOrder())
        if kind is None:return None,'unsupported_perceived_bond'
        rw.AddBond(a,b,kind);unknown_edges.add(tuple(sorted((a,b))))
    if sorted(crossings)!=sorted(desc['attachment_sites']):return None,'attachment_sites_mismatch'
    # XYZ without hydrogen can overassign terminal oxygen bond orders.
    # Resolve those local valences using fixed known bonds, elements, distances
    # and formal charge, never a reference precursor string or unknown graph.
    for i in range(n):
        center=rw.GetAtomWithIdx(i)
        terminal=[a.GetIdx() for a in center.GetNeighbors() if a.GetIdx()>=n
                  and a.GetAtomicNum()==8 and a.GetDegree()==1]
        if not terminal:continue
        other=sum(b.GetBondTypeAsDouble() for b in center.GetBonds()
                  if b.GetOtherAtomIdx(i) not in terminal)
        z=center.GetAtomicNum();charge=center.GetFormalCharge()
        if z==6 and center.GetIsAromatic():continue
        if z==6:limit=4
        elif z==7:limit=4 if charge==1 else 3
        else:continue
        available=int(limit-other)
        if available<len(terminal):continue
        # Prefer the shorter terminal O bond when one double bond is allowed.
        ordered=sorted(terminal,key=lambda j:float(np.linalg.norm(xyz[i]-xyz[j])))
        for k,j in enumerate(ordered):
            order=2 if k<available-len(ordered) else 1
            rw.GetBondBetweenAtoms(i,j).SetBondType(types[order])
            if z==7 and charge==1:
                rw.GetAtomWithIdx(j).SetFormalCharge(-1 if order==1 else 0)
    mol=rw.GetMol()
    try:
        Chem.SanitizeMol(mol)
        if len(Chem.GetMolFrags(mol))!=1:return None,'disconnected_precursor'
        if sum(a.GetFormalCharge() for a in mol.GetAtoms())!=0 or any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
            return None,'unbalanced_charge_or_radical'
        for a,b,kind in desc['bonds']:
            assert str(mol.GetBondBetweenAtoms(a,b).GetBondType())==kind
        actual={tuple(sorted((b.GetBeginAtomIdx(),b.GetEndAtomIdx()))) for b in mol.GetBonds()
            if b.GetBeginAtomIdx()>=n or b.GetEndAtomIdx()>=n}
        if actual!=unknown_edges:return None,'sanitize_changed_unknown_connectivity'
        return canonical(Chem.MolToSmiles(mol)),None
    except (ValueError,RuntimeError):return None,'generated_graph_sanitize_failed'
