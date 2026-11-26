"""Fixed-core completion evaluation; no inventory structure assistance.

Known atoms, bonds, coordinates and attachment sites are explicit allowed inputs.
Unknown size, atoms, coordinates and bonds must come from the trained models.
"""
import os,sys,time,random,traceback
from pathlib import Path
from collections import Counter
import numpy as np
import torch,psutil
from run_last_epoch_diagnostic import REPO,ROOT,read,write,sha
from exact_case_memory_worker import runtime_for
from exact_reaction_memory_codec_v5 import reconstruct,canonical
from probe_known_graph_decoding_strict import load_decoder

BASE=REPO/'experiments/case_partition_completion_20261013'
OUT=BASE/'fixed_core_completion_20261013'
DATA=REPO/'experiments/case_all_exact_data_v3_20261012'

def main():
    assert not OUT.exists();OUT.mkdir()
    manifest=read(BASE/'selected_model_manifest.json')
    weights={k:manifest[k] for k in ['size','edm']}
    for m in weights.values():assert sha(m['path'])==m['sha256']
    files=[Path(__file__),ROOT/'exact_case_memory_worker.py',ROOT/'exact_reaction_memory_codec_v3.py',
        ROOT/'exact_reaction_memory_codec_v4.py',ROOT/'exact_reaction_memory_codec_v5.py',DATA/'reaction_pairs.json',BASE/'selected_model_manifest.json']
    binding={str(p):sha(p) for p in files}
    protocol=dict(evaluation_target='Known-core conditional precursor completion',
        allowed_inputs=['known core atoms, bonds, coordinates, properties and attachment sites'],
        reference_known_core_condition=True,unknown_reference_size_input=False,unknown_reference_elements_input=False,
        unknown_reference_coordinates_input=False,unknown_reference_bonds_input=False,inventory_used=False,LLM_used=False,
        stage1_used=False,product_only=False,case_training_contact=True,draws=30,sampling_steps=100,batch_size=8,
        seed=20261012,max_seconds=2400,weights=weights,source_sha256=binding,new_training=False,
        primary_criterion='All reference precursors occur in one paired draw',
        secondary_criterion='Each reference precursor occurs somewhere in its own generated fragment bank')
    write(OUT/'protocol.json',protocol)
    state=dict(state='loading',pid=os.getpid(),create_time=psutil.Process().create_time(),draws_done=0,started_unix=time.time())
    def status(**values):state.update(values,updated_unix=time.time());write(OUT/'status.json',state)
    status();start=time.monotonic()
    try:
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.manual_seed(protocol['seed']);np.random.seed(protocol['seed']);random.seed(protocol['seed'])
        _,_,_,rt,_=runtime_for('edm3d',OUT)
        model=rt['model'];payload=torch.load(weights['edm']['path'],map_location='cpu',weights_only=False)
        assert payload['epoch']==80;model.load_state_dict(payload['state_dict'],strict=True);model.eval();model.edm.T=100
        from src.linker_size_lightning import SizeClassifier
        from src.datasets import collate,collate_with_fragment_edges
        from src import const
        sp=torch.load(weights['size']['path'],map_location='cpu',weights_only=False);assert sp['epoch']==100
        hp=dict(sp['hyper_parameters']);hp['torch_device']='cuda:0';size=SizeClassifier(**hp);size.on_load_checkpoint(sp)
        size.load_state_dict(sp['state_dict'],strict=True);size.to('cuda:0').eval()
        decoder=load_decoder();source_pairs=read(DATA/'reaction_pairs.json')
        # The generation input whitelist omits every unknown target field.
        conditions=[dict(uuid=f['uuid'],step_id=p['step_id'],condition=f['condition'],primary_anchor=f['primary_anchor'])
            for p in source_pairs for f in p['fragments']]
        write(OUT/'generation_inputs.json',conditions)
        records=[]
        for f in conditions:
            d=f['condition'];n=len(d['atoms']);ids=[const.ATOM2IDX[a['symbol']] for a in d['atoms']]
            anchor=torch.zeros(n);anchor[f['primary_anchor']]=1
            records.append(dict(uuid=f['uuid'],name='given_known_core',num_atoms=n,
                positions=torch.tensor(d['coordinates'],dtype=torch.float32),one_hot=torch.eye(26)[ids],
                charges=torch.tensor([const.CHARGES[a['symbol']] for a in d['atoms']],dtype=torch.float32),
                anchors=anchor,fragment_mask=torch.ones(n),linker_mask=torch.zeros(n)))
        del rt['train'],rt['valid'],payload,sp,source_pairs
        bank={f['uuid']:[] for f in conditions};raw=[];max_known_coordinate_error=0.0
        from src.utils import FoundNaNException
        with torch.no_grad():
            for draw in range(30):
                assert time.monotonic()-start<2400 and not (OUT/'STOP').exists()
                for offset in range(0,len(records),8):
                    part=records[offset:offset+8]
                    sb=rt['move'](collate_with_fragment_edges(part),torch.device('cuda:0'))
                    logits,_=size(sb,return_loss=False);ids=torch.multinomial(logits.softmax(-1),1).flatten()
                    sizes=torch.tensor([size.linker_id2size[int(i)] for i in ids],device='cuda:0')
                    batch=rt['move'](collate(part),torch.device('cuda:0'))
                    try:
                        chain,mask=model.sample_chain(batch,sample_fn=lambda _:sizes,keep_frames=1)
                        if not torch.isfinite(chain).all():raise FloatingPointError('Nonfinite sample')
                    except (FoundNaNException,FloatingPointError):
                        for j,r in enumerate(part):bank[r['uuid']].append(dict(draw=draw,predicted_size=int(sizes[j]),smiles=None,decode_error='numerical_failure',raw_index=None))
                        continue
                    for j,r in enumerate(part):
                        f=conditions[offset+j];n=int(mask[j].sum());arr=chain[0,j,:n].cpu().numpy()
                        xyz=arr[:,:3];elem=arr[:,3:].argmax(-1);known=np.asarray(f['condition']['coordinates']);nk=len(known)
                        error=float(np.abs((xyz[:nk]-xyz[:nk].mean(0))-(known-known.mean(0))).max());assert error<1e-4
                        max_known_coordinate_error=max(max_known_coordinate_error,error)
                        got,reason=reconstruct(decoder,f['condition'],xyz,elem)
                        bank[r['uuid']].append(dict(draw=draw,predicted_size=int(sizes[j]),smiles=got,decode_error=reason,raw_index=len(raw)))
                        raw.append(dict(uuid=r['uuid'],draw=draw,xyz=xyz,elements=elem))
                status(state='sampling',draws_done=draw+1,completed_fragment_draws=sum(len(v) for v in bank.values()))
        torch.save(raw,OUT/'raw_samples.pt');write(OUT/'decoded_bank.json',bank)
        # Only scoring uses full precursor strings and unknown reference sizes.
        pairs=read(DATA/'reaction_pairs.json');results=[]
        for p in pairs:
            targets=sorted(canonical(s) for s in p['reference_precursors']);exact=0;individual=[]
            for draw in range(30):
                values=[bank[f['uuid']][draw]['smiles'] for f in p['fragments']]
                if all(v is not None for v in values) and sorted(values)==targets:exact+=1
            for f in p['fragments']:
                entries=bank[f['uuid']];wanted=canonical(f['expected_precursor'])
                individual.append(dict(uuid=f['uuid'],exact_draws=sum(v['smiles']==wanted for v in entries),
                    reference_unknown_atoms=f['unknown_atoms'],sampled_size_counts=dict(Counter(v['predicted_size'] for v in entries)),
                    decode_errors=dict(Counter(v['decode_error'] for v in entries if v['decode_error']))))
            results.append(dict(step_id=p['step_id'],exact_joint_draws=exact,all_precursors_found_separately=all(v['exact_draws']>0 for v in individual),individual=individual))
        cov=[r for r in results if r['step_id']!='LAP-06'];groups={}
        for name,prefix in [('Lapatinib','LAP-'),('Nemiralisib','NEMI-'),('3r','OSI-'),('Tafamidis','TAF-')]:
            rows=[r for r in cov if r['step_id'].startswith(prefix)]
            groups[name]=dict(step_records=len(rows),paired_exact_steps=sum(r['exact_joint_draws']>0 for r in rows),
                separate_bank_exact_steps=sum(r['all_precursors_found_separately'] for r in rows),
                failed_steps=[r['step_id'] for r in rows if r['exact_joint_draws']==0])
        report=dict(state='complete',protocol=protocol,steps=results,cases=groups,covalent_step_records=len(cov),
            paired_exact_steps=sum(r['exact_joint_draws']>0 for r in cov),separate_bank_exact_steps=sum(r['all_precursors_found_separately'] for r in cov),
            known_atoms_and_coordinates_preserved=True,max_centered_known_coordinate_error_angstrom=max_known_coordinate_error,
            fragment_draw_slots=len(conditions)*30,raw_sample_count=len(raw),raw_sha256=sha(OUT/'raw_samples.pt'),
            not_an_autonomous_multistep_test=True)
        write(OUT/'report.json',report)
        for p,digest in binding.items():assert sha(p)==digest,p
        for m in weights.values():assert sha(m['path'])==m['sha256']
        status(state='complete',paired_exact_steps=report['paired_exact_steps'],separate_bank_exact_steps=report['separate_bank_exact_steps'])
        print('COMPLETE',report['paired_exact_steps'],'/',len(cov),'separate',report['separate_bank_exact_steps'],'cases',groups,flush=True)
    except BaseException:
        status(state='failed',traceback=traceback.format_exc());raise
if __name__=='__main__':main()
