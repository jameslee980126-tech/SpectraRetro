"""Bounded product-only recursive preview using frozen models, no references.

Top ten original center candidates, 30 paired size/EDM draws per supported
center, 100 diffusion intervals. Save top ten sets; expand one ranked path.
This is a NEW exploratory inference protocol, never an official benchmark.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
import torch

HERE=Path(__file__).resolve().parent
EXP=HERE.parents[2]
sys.path.insert(0,str(EXP))
from runtime_common import create, setup, seed, collate, collate_with_fragment_edges, const, sha, write

S1=Path(os.environ.get("SPECTRARETRO_STAGE1_PYTHON", sys.executable))
DECODER=Path(os.environ.get("SPECTRARETRO_DECODER_PYTHON", sys.executable))
CHECKPOINT=EXP/'server_backups/stage2_backup_20260915_190005/snapshot/server_run/edm_3d/best.ckpt'
SIZE_CHECKPOINT=EXP/'server_results/size_best_epoch37.ckpt'
ROOTS=[
    dict(id='8719',smiles='CNc1ncc(-c2nc3cc(OCCSC)ccc3o2)c2cc(NC(=O)C3CC3)ncc12'),
    dict(id='8987',smiles='CCOC(=O)c1ccc(Cl)cc1-c1cc(C(=O)Nc2ccc(OCc3ccccc3)cc2)c2n1CCCC2'),
    dict(id='8988',smiles='COC(=O)c1cc(F)c(Cl)cc1-c1cc(C(=O)Nc2ccc(OCc3ccccc3)cc2)c(C)n1C'),
]
PROTOCOL=dict(version=1,scope='product-only two-level greedy rollout with retained top-10 local candidates; exploratory, not official evaluation',
    reaction_class_input=False,reference_inputs=False,oracle_size=False,center_topk=10,paired_draws_per_center=30,
    diffusion_intervals=100,max_depth=2,seed=0,embedding='ETKDGv3 of current product; MMFF94 or UFF; fragment coordinates sliced only from this product',
    unsupported='atom centers, ring bonds, non-single bonds, fragments smaller than two heavy atoms; no backfill beyond original top ten',
    ranking='sum across centers of center_softmax_probability * exact hydrogen-restored precursor-set frequency / 30; ties lexical',
    expansion='highest-ranked among retained top ten with every child smaller than parent and no ancestor cycle; expand both children once',
    leaves='depth/node-size stop only, not a claim of commercial availability',
    decoder='Windows Open Babel 3.1.0 XYZ, fixed generic hydrogen restoration, RDKit sanitization; reject radicals, charges and disconnected per-fragment outputs',
    validation='structural checks only; not a feasibility predictor or wet-lab validation',
    failures='retain failed slots; no target-guided retries or reseeding; diagnostic runtime failures abort',
    size_batching='each fragment independently to avoid padding-dependent size probabilities',
    stage1_correct_center_gate=False)

def canonical(s):
    m=Chem.MolFromSmiles(s)
    if m is None: raise ValueError(s)
    return Chem.MolToSmiles(m,isomericSmiles=True)

def stable_seed(s):return int(hashlib.sha256(s.encode()).hexdigest()[:8],16)%2147483647

def score(nodes,outdir):
    inp=outdir/'stage1_inputs.json';dest=outdir/'stage1_scores.json'
    write(inp,nodes)
    with (outdir/'stage1.log').open('w',encoding='utf-8') as log:
        subprocess.run([str(S1),str(HERE/'score_product.py'),str(inp),str(dest)],check=True,stdout=log,stderr=log)
    return json.loads(dest.read_text())['nodes']

def embed_product(smiles):
    mol=Chem.AddHs(Chem.MolFromSmiles(smiles))
    params=AllChem.ETKDGv3();params.randomSeed=stable_seed(smiles);params.numThreads=1
    if AllChem.EmbedMolecule(mol,params)!=0:raise ValueError('Product embedding failed')
    if AllChem.MMFFHasAllMoleculeParams(mol):status=AllChem.MMFFOptimizeMolecule(mol,maxIters=500)
    else:status=AllChem.UFFOptimizeMolecule(mol,maxIters=500)
    mol=Chem.RemoveHs(mol)
    return mol,status

def fragments(mol,candidate):
    if candidate['kind']!='bond':return None,'atom_center_not_supported'
    if candidate['in_ring']:return None,'ring_opening_not_supported'
    if candidate['bond_type']!='SINGLE':return None,'non_single_bond_not_supported'
    a,b=candidate['atom_indices']
    rw=Chem.RWMol(mol);rw.RemoveBond(a,b)
    groups=Chem.GetMolFrags(rw.GetMol(),asMols=False)
    if len(groups)!=2:return None,'not_two_components'
    if min(map(len,groups))<2:return None,'single_atom_fragment_not_supported'
    xyz=mol.GetConformer().GetPositions()
    records=[];information=[]
    for j,group in enumerate(groups):
        group=list(group);n=len(group)
        elements=[mol.GetAtomWithIdx(i).GetSymbol() for i in group]
        if any(e not in const.ATOM2IDX for e in elements):return None,'element_not_in_vocabulary'
        anchor=group.index(a if a in group else b)
        onehot=np.eye(26,dtype=np.float32)[[const.ATOM2IDX[e] for e in elements]]
        anchors=np.zeros(n,dtype=np.float32);anchors[anchor]=1
        records.append(dict(uuid=str(j),name='product_only',num_atoms=n,
            positions=torch.tensor(xyz[group],dtype=torch.float32,device='cuda'),
            one_hot=torch.tensor(onehot,device='cuda'),
            charges=torch.tensor([const.CHARGES[e] for e in elements],dtype=torch.float32,device='cuda'),
            anchors=torch.tensor(anchors,device='cuda'),fragment_mask=torch.ones(n,device='cuda'),
            linker_mask=torch.zeros(n,device='cuda')))
        information.append(dict(product_atom_indices=group,elements=elements,anchor_local_index=anchor,coordinates=xyz[group].tolist()))
    return (records,information),None

def generate_node(node,model,size_model,outdir,draws,steps,smoke=False):
    outdir.mkdir(exist_ok=True)
    scored=score([dict(id=node['id'],smiles=node['smiles'])],outdir)[0]
    mol,optimization=embed_product(node['smiles'])
    Chem.MolToMolFile(mol,str(outdir/'input_product_3d.mol'))
    branch_records=[];rawxyz=[];rawtypes=[];lengths=[];slots=[]
    centers=scored['candidates'][:1 if smoke else 10]
    started=time.time()
    for center in centers:
        branch=dict(center=center,draws=[],status='pending')
        built,why=fragments(mol,center)
        if why:
            branch.update(status='unsupported',reason=why);branch_records.append(branch);continue
        records,info=built
        branch['fragments']=info
        branchseed=stable_seed(node['smiles']+'|'+str(center['rank']))
        seed(branchseed)
        sizes=[]
        for rec,desc in zip(records,info):
            with torch.no_grad():logits,_=size_model.forward(collate_with_fragment_edges([rec]),return_loss=False)
            probability=logits.softmax(-1)[0]
            labels=torch.multinomial(probability,draws,replacement=True)
            picked=[int(size_model.linker_id2size[int(x)]) for x in labels.cpu()]
            sizes.append(picked)
            desc['size_probabilities']={str(size_model.linker_id2size[i]):float(p) for i,p in enumerate(probability.cpu())}
            desc['sampled_sizes']=picked
        # All paired fragment slots receive their own sampled size; no target counts.
        jobs=[(d,j,sizes[j][d]) for d in range(draws) for j in range(2)]
        for start in range(0,len(jobs),12):
            chunk=jobs[start:start+12]
            batch=collate([records[j] for d,j,n in chunk])
            sizes_tensor=torch.tensor([n for d,j,n in chunk],device='cuda',dtype=torch.long)
            try:
                with torch.no_grad():
                    chain,mask=model.sample_chain(batch,sample_fn=lambda _:sizes_tensor,keep_frames=5 if steps>=5 and start==0 else 1)
                if not torch.isfinite(chain).all():raise RuntimeError('nonfinite diffusion output')
            except Exception as exc:
                # Never change the draws or hide a numerical failure.
                from src.utils import FoundNaNException
                if not isinstance(exc,FoundNaNException) and 'nonfinite' not in str(exc):raise
                for d,j,n in chunk:slots.append(dict(center_rank=center['rank'],draw=d,fragment=j,raw_index=None,error=str(exc)))
                continue
            array=chain.detach().cpu().numpy();ns=mask.sum((1,2)).int().cpu().tolist()
            if start==0:
                np.savez_compressed(outdir/f'trajectory_center{center["rank"]}.npz',chain=array[:,0],length=ns[0],
                    fragment_size=records[chunk[0][1]]['num_atoms'],draw=chunk[0][0],fragment=chunk[0][1])
            for k,((d,j,n),count) in enumerate(zip(chunk,ns)):
                x=array[0,k,:count,:3];t=array[0,k,:count,3:].argmax(-1).astype(np.uint8)
                slots.append(dict(center_rank=center['rank'],draw=d,fragment=j,raw_index=len(lengths),size=n))
                rawxyz.append(x);rawtypes.append(t);lengths.append(count)
        branch.update(status='sampled',seed=branchseed)
        branch_records.append(branch)
        write(outdir/'progress.json',dict(node=node['id'],completed_center=center['rank'],seconds=time.time()-started))
        print(f'NODE {node["id"]} CENTER {center["rank"]} done {time.time()-started:.1f}s',flush=True)
    decoded=[]
    if lengths:
        rawpath=outdir/'raw_candidates.npz'
        np.savez_compressed(rawpath,xyz=np.concatenate(rawxyz),element=np.concatenate(rawtypes),length=np.array(lengths))
        with (outdir/'decoder.log').open('w',encoding='utf-8') as log:
            subprocess.run([str(DECODER),str(HERE/'decode_candidates.py'),str(rawpath),str(outdir/'decoded.json')],check=True,stdout=log,stderr=log)
        decoded=json.loads((outdir/'decoded.json').read_text())['rows']
    byslot={(s['center_rank'],s['draw'],s['fragment']):s for s in slots}
    weighted={};supports={}
    for branch in branch_records:
        if branch['status']!='sampled':continue
        counts=Counter()
        for d in range(draws):
            pair=[];reasons=[];raw=[]
            for j in range(2):
                s=byslot[(branch['center']['rank'],d,j)]
                value=decoded[s['raw_index']] if s['raw_index'] is not None else dict(smiles=None,raw=None,reason=s['error'])
                pair.append(value['smiles']);raw.append(value['raw']);reasons.append(value['reason'])
            key='.'.join(sorted(pair)) if all(pair) else None
            branch['draws'].append(dict(draw=d,raw=raw,precursors=pair,valid_set=key,reasons=reasons))
            if key:counts[key]+=1
        branch['valid_pairs']=sum(counts.values())
        for key,count in counts.items():
            weight=branch['center']['probability']*count/draws
            weighted[key]=weighted.get(key,0)+weight
            supports.setdefault(key,[]).append(dict(center_rank=branch['center']['rank'],count=count,draws=draws,weighted_score=weight))
    ranking=[dict(precursor_set=s,score=w,supports=supports[s]) for s,w in sorted(weighted.items(),key=lambda pair:(-pair[1],pair[0]))]
    for i,r in enumerate(ranking,1):r['rank']=i
    chosen=None
    for r in ranking[:10]:
        children=r['precursor_set'].split('.')
        if all(Chem.MolFromSmiles(c).GetNumHeavyAtoms()<mol.GetNumHeavyAtoms() and c not in node['ancestors']+[node['smiles']] for c in children):
            chosen=r;break
    result=dict(node,stage1=scored,product_optimization_status=optimization,branches=branch_records,
        ranking=ranking,top10=ranking[:10],selected=chosen,elapsed_seconds=time.time()-started,
        raw_candidate_count=len(lengths),status='generated' if ranking else 'no_valid_candidates')
    write(outdir/'result.json',result)
    print('RESULT',node['id'],'sets',len(ranking),'selected',chosen,flush=True)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');args=p.parse_args()
    out=HERE/('runtime_smoke' if args.smoke else 'product_only_v1');out.mkdir(exist_ok=True)
    freeze=json.loads((EXP/'server_results/evaluation_freeze_1500.json').read_text())
    assert sha(CHECKPOINT)==freeze['arms']['edm_3d']['best_sha256']
    assert sha(SIZE_CHECKPOINT)==freeze['arms']['size']['best_sha256']
    protocol=dict(PROTOCOL,checkpoints=dict(diffusion=sha(CHECKPOINT),size=sha(SIZE_CHECKPOINT)),
        stage1_sha256='a71715eb470c529fc76bf11e4f0d579e12e07e36647fd2300c3e0fa090b28900',
        runner_sha256=sha(Path(__file__)),torch=torch.__version__,gpu=torch.cuda.get_device_name(),roots=ROOTS)
    if (out/'protocol.json').exists():
        assert json.loads((out/'protocol.json').read_text())==protocol,'Protocol changed: do not resume under different rules'
    else:write(out/'protocol.json',protocol)
    setup(2);seed(0);RDLogger.DisableLog('rdApp.*')
    model=create(3,torch.device('cuda'));model.load_state_dict(torch.load(CHECKPOINT,map_location='cpu',weights_only=False)['state_dict'],strict=True);model.eval()
    size_model=create(3,torch.device('cuda'),size=True);size_model.load_state_dict(torch.load(SIZE_CHECKPOINT,map_location='cpu',weights_only=False)['state_dict'],strict=True);size_model.eval()
    steps=2 if args.smoke else 100;draws=2 if args.smoke else 30;model.edm.T=steps
    nodes=[dict(r,id=r['id']+'_root',case_id=r['id'],depth=0,ancestors=[],smiles=canonical(r['smiles'])) for r in ROOTS[:1 if args.smoke else 3]]
    results=[]
    while nodes:
        node=nodes.pop(0);folder=out/node['id']
        result_path=folder/'result.json'
        if result_path.exists():result=json.loads(result_path.read_text())
        else:result=generate_node(node,model,size_model,folder,draws,steps,args.smoke)
        results.append(result)
        if not args.smoke and node['depth']==0 and result['selected']:
            for j,smiles in enumerate(result['selected']['precursor_set'].split('.')):
                if Chem.MolFromSmiles(smiles).GetNumHeavyAtoms()<=3:continue
                nodes.append(dict(id=node['case_id']+f'_child{j}',case_id=node['case_id'],smiles=smiles,depth=1,
                    ancestors=[node['smiles']],parent=node['id'],parent_selected_rank=result['selected']['rank']))
        write(out/'status.json',dict(state='running' if nodes else 'completed',completed_nodes=[r['id'] for r in results],pending=[n['id'] for n in nodes]))
    write(out/'results.json',dict(protocol=protocol,nodes=results))

if __name__=='__main__':main()
