"""Independent chemical/source/grouping verification of the new frozen subset."""
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
from rdkit import Chem, rdBase, RDLogger

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'stage1_local_similarity20_20260911'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

def load(path): return json.loads(path.read_text())

def main():
    RDLogger.DisableLog('rdApp.warning')
    f=ROOT/'frozen'
    m=load(f/'manifest.json')
    assert m['selection_plan_sha256']==sha(ROOT/'selection_plan.json')
    assert m['build_script_sha256']==sha(ROOT/'build_dataset.py')
    for name,d in m['files'].items(): assert sha(f/name)==d,name
    spec=importlib.util.spec_from_file_location('reference_similarity',OLD/'similarity.py')
    sim=importlib.util.module_from_spec(spec); spec.loader.exec_module(sim)
    all_keys, summary={},{}
    chemistry_checked=0; rechecked=0
    rule=load(f/'thresholds.json')['rule']
    for s in ('train','valid','test'):
        source=m['source_provenance'][s]
        assert sha(source['csv'])==source['csv_sha256']
        assert sha(source['metadata'])==source['metadata_sha256']
        parent=list(csv.DictReader(Path(source['csv']).open(encoding='utf-8-sig')))
        metadata=load(Path(source['metadata']))
        rows=list(csv.DictReader((f/f'{s}.csv').open(encoding='utf-8-sig')))
        index=list(csv.DictReader((f/f'{s}_index.csv').open(encoding='utf-8-sig')))
        design=load(f/f'{s}_design.json')
        assert len(rows)==len(index)==len(design)==m['counts'][s]
        assert len({r['sample_id'] for r in index})==len(index)
        parentgroups=defaultdict(set)
        for r in metadata: parentgroups[r['product_group']].add(r['sample_id'])
        selectedgroups=defaultdict(set)
        recompute_ids={}
        for g in ('low','medium','high'):
            candidates=[r for r in design if r['similarity_group']==g]
            candidates.sort(key=lambda r:hashlib.sha256(('verify|'+r['sample_id']).encode()).hexdigest())
            recompute_ids.update({r['sample_id']:True for r in candidates[:8]})
        groupstats=defaultdict(list)
        for row,idx,d in zip(rows,index,design):
            k=int(idx['source_row']); assert row==parent[k]
            assert idx['sample_id']==d['sample_id']==f'{s}:{k:06d}'
            assert d['radii']==metadata[k]['radii']
            assert d['class']==row['class']==idx['class']
            react,_,prod=row['rxn_smiles'].split('>')
            rm,pm=Chem.MolFromSmiles(react),Chem.MolFromSmiles(prod)
            assert rm is not None and pm is not None
            maps=[a.GetAtomMapNum() for a in pm.GetAtoms()]
            assert len(set(maps))==len(maps) and 0 not in maps
            bonds={tuple(sorted((b.GetBeginAtom().GetAtomMapNum(),b.GetEndAtom().GetAtomMapNum()))) for b in rm.GetBonds()}
            added=[tuple(sorted((b.GetBeginAtom().GetAtomMapNum(),b.GetEndAtom().GetAtomMapNum())))
                   for b in pm.GetBonds() if tuple(sorted((b.GetBeginAtom().GetAtomMapNum(),b.GetEndAtom().GetAtomMapNum()))) not in bonds]
            assert len(added)==1 and list(added[0])==d['gold_atom_maps'],idx['sample_id']
            assert pm.GetNumAtoms()+pm.GetNumBonds()==d['candidate_count']
            assert pm.GetNumHeavyAtoms()==d['heavy_atoms']
            clone=Chem.Mol(pm)
            for a in clone.GetAtoms():a.SetAtomMapNum(0)
            Chem.RemoveStereochemistry(clone)
            key=Chem.MolToSmiles(clone,canonical=True,isomericSmiles=False)
            assert key==idx['product_group']==d['product_group'], (idx['sample_id'],key,idx['product_group'])
            selectedgroups[key].add(idx['sample_id'])
            value=d['radii']['2']['difficulty']
            g='low' if value<rule['low'] else 'medium' if value<rule['high'] else 'high'
            assert idx['similarity_group']==d['similarity_group']==g
            groupstats[g].append(d)
            chemistry_checked+=1
            if d['sample_id'] in recompute_ids:
                bnds=[(b.GetBeginAtomIdx(),b.GetEndAtomIdx()) for b in pm.GetBonds()]
                gold=next(i for i,(u,v) in enumerate(bnds) if sorted((maps[u],maps[v]))==list(added[0]))
                equiv=sim.rooted_product_key(pm,bnds[gold])
                graph=sim.molecular_graph(pm)
                _,reference=sim.local_environment(graph,bnds[gold],2)
                competitors=[i for i,b in enumerate(bnds) if i!=gold and sim.rooted_product_key(pm,b)!=equiv]
                measured=max(sim.similarity(reference,sim.local_environment(graph,bnds[i],2)[1]) for i in competitors)
                assert abs(measured-value)<1e-12,(d['sample_id'],measured,value)
                rechecked+=1
        for key,ids in selectedgroups.items():assert ids==parentgroups[key],key
        all_keys[s]=set(selectedgroups)
        result={}
        cells=[]
        for g,rs in groupstats.items():
            joint=Counter((r['class'],0 if r['heavy_atoms']<=15 else 1 if r['heavy_atoms']<=20 else 2 if r['heavy_atoms']<=30 else 3,
                           0 if r['candidate_count']<=30 else 1 if r['candidate_count']<=45 else 2 if r['candidate_count']<=65 else 3) for r in rs)
            result[g]={'n':len(rs),'products':len({r['product_group'] for r in rs}),
                'class_counts':dict(Counter(r['class'] for r in rs)),
                'mean_heavy_atoms':float(np.mean([r['heavy_atoms'] for r in rs])),
                'mean_candidates':float(np.mean([r['candidate_count'] for r in rs]))}
            cells.append(joint)
        if s!='train':
            assert cells[0]==cells[1]==cells[2]
            assert all(z['n']==800 and min(z['class_counts'].values())>=60 for z in result.values())
            for g in ('medium','high'):
                assert abs(result[g]['mean_heavy_atoms']-result['low']['mean_heavy_atoms'])<=.25+1e-9
                assert abs(result[g]['mean_candidates']-result['low']['mean_candidates'])<=.5+1e-9
        summary[s]={'rows':len(rows),'products':len(selectedgroups),'groups':result,
                    'complete_source_product_groups':True}
        print('VERIFIED',s,len(rows),len(selectedgroups),flush=True)
    overlaps={a+'_'+b:len(all_keys[a]&all_keys[b]) for a,b in [('train','valid'),('train','test'),('valid','test')]}
    assert not any(overlaps.values())
    assert .2<chemistry_checked/m['source_total_rows']<.3
    report={'passed':True,'chemical_rows_checked':chemistry_checked,'source_rows_checked':chemistry_checked,
            'r2_similarity_recomputed':rechecked,'original_product_overlap':overlaps,
            'summary':summary,'rdkit':rdBase.rdkitVersion,'manifest_sha256':sha(f/'manifest.json'),
            'verifier_sha256':sha(Path(__file__)),
            'limitations':['balanced diagnostic distribution','historically inspected source test',
                          'official task candidate mapping must be checked when connecting training loader']}
    (ROOT/'VERIFICATION.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('ALL_CHECKS_PASSED',chemistry_checked,'similarity_recomputed',rechecked,flush=True)

if __name__=='__main__': main()
