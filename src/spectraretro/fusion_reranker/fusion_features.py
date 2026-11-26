"""Pure adaptation of Sept26 meta.prob_features; targets/classes never enter."""
from collections import Counter
from functools import lru_cache
import math
import numpy as np
from rdkit import Chem

@lru_cache(maxsize=65536)
def elements(smiles):
    if 'MISSING' in smiles or 'INVALID' in smiles:return None
    mol=Chem.MolFromSmiles(smiles,sanitize=False)
    return None if mol is None else Counter(a.GetSymbol() for a in mol.GetAtoms() if a.GetAtomicNum()>1)

def candidate_identity(candidate,known):
    a,b=elements(candidate),elements(known)
    if a is None or b is None:return 'INVALID'
    diff={e:a[e]-b[e] for e in set(a)|set(b)}
    if any(v<0 for v in diff.values()):return 'INVALID'
    total=sum(diff.values())
    if total==0:return 'NONE'
    if total>1:return 'MULTI'
    return 'ONE_'+next(e for e,n in diff.items() if n==1)

def probability_features(uuids,fragments,parts,probabilities,index):
    assert len(uuids)==len(fragments)==len(parts)>0
    logs=[];ps=[];entropies=[];out={};prefix='identity_prior_'
    for uid,part,known in zip(uuids,parts,fragments):
        typ=candidate_identity(part,known)
        vector=np.maximum(probabilities[int(uid)].astype(float),1e-12)
        p=1e-12 if typ=='INVALID' else float(vector[index.get(typ,index['ONE_OTHER'])])
        logp=math.log(p);logs.append(logp);ps.append(p)
        entropies.append(-float(np.sum(vector*np.log(vector))))
        out[prefix+'logp_'+typ]=out.get(prefix+'logp_'+typ,0.)+logp
        out[prefix+'p_'+typ]=out.get(prefix+'p_'+typ,0.)+p
        sortedp=np.sort(vector)
        out[prefix+'margin_sum']=out.get(prefix+'margin_sum',0.)+float(sortedp[-1]-sortedp[-2])
    out.update({prefix+'logp_sum':sum(logs),prefix+'logp_min':min(logs),prefix+'logp_max':max(logs),prefix+'logp_mean':float(np.mean(logs)),
                prefix+'p_min':min(ps),prefix+'p_max':max(ps),prefix+'p_mean':float(np.mean(ps)),prefix+'entropy_sum':sum(entropies),prefix+'entropy_max':max(entropies)})
    assert all(math.isfinite(v) for v in out.values())
    return out
