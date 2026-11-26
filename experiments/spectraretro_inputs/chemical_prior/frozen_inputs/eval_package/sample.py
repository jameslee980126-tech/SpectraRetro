"""Resumable generation with independently seeded draw slots and atomic archives."""
import argparse,json,os,time,traceback
import numpy as np
import torch
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import portable_common as common
from portable_common import *
ROOT=common.ROOT
CHECKPOINT=None
def load_model(arm):
    from graph_package.graph_model import UnknownObservedGraphDDPM
    c=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    if c['arm']!=arm:raise ValueError('Checkpoint arm mismatch')
    if c['compatibility_enabled']!=(arm=='compatibility') or c['observed_graph_enabled']!=(arm=='graph'):
        raise ValueError('Checkpoint condition flags mismatch')
    model=UnknownObservedGraphDDPM(compatibility_enabled=c['compatibility_enabled'],observed_graph_enabled=c['observed_graph_enabled'],**c['configuration'])
    model.load_state_dict(c['state_dict'],strict=True);model.edm.T=100
    return model.eval().cuda()

def save_npz(path,**arrays):
    tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:np.savez(f,**arrays)
    os.replace(tmp,path)
def main():
    global ROOT,OLD,WORK,DATA,CHECKPOINT
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--run-root',type=Path,required=True);p.add_argument('--bundle-root',type=Path,required=True);a=p.parse_args()
    common.configure(a.bundle_root,a.run_root);ROOT,OLD,WORK,DATA=common.ROOT,common.OLD,common.WORK,common.DATA;CHECKPOINT=a.checkpoint.resolve()
    setup()
    from graph_package.graph_data import collate_graph as collate_unknown,inference_inputs
    with np.load(common.CYCLE/'unknown_compatibility/eval_prior.npz',allow_pickle=False) as cached:
        probabilities={str(uid):p.copy() for uid,p in zip(cached['uuids'],cached['probabilities'])}
    def collate_internal(items):
        return collate_unknown(items,np.stack([probabilities[str(x['uuid'])] for x in items]))
    def move_batch(batch,device):
        return {k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}

    from src import utils
    protocol=json.loads((ROOT/'protocol.json').read_text(encoding='utf8'))
    if protocol['checkpoints'][a.arm]['sha256']!=digest(CHECKPOINT):raise ValueError('Wrong bound checkpoint')
    for path,h in protocol['sources'].items():
        if digest(path)!=h:raise ValueError('Frozen input changed: '+path)
    assert digest(ROOT/'shared_sizes.npz')==protocol['shared_sizes_sha256']
    assert digest(ROOT/'batch_order.json')==protocol['batch_order_sha256']
    rows=torch.load(OLD/'newdata/eval_geometry.pt',map_location='cpu',weights_only=False)
    shared=np.load(ROOT/'shared_sizes.npz',allow_pickle=False)
    assert shared['uuids'].tolist()==[str(r['uuid']) for r in rows]
    order=json.loads((ROOT/'batch_order.json').read_text())
    output=ROOT/'evaluation'/a.arm/'raw';output.mkdir(parents=True,exist_ok=True)
    status=output.parent/'sampling_status.json';model=load_model(a.arm)
    begun=time.monotonic();prior=0
    if status.exists():prior=json.loads(status.read_text()).get('accumulated_seconds',0)
    def update(**kwargs):
        value=dict(arm=a.arm,pid=os.getpid(),time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                   accumulated_seconds=prior+time.monotonic()-begun,**kwargs)
        write_json(status,value)
    completed_rows=0;bi=-1;draw=-1
    try:
        with torch.no_grad():
            for bi,start in enumerate(range(0,len(order),protocol['batch_size'])):
                ids=order[start:start+protocol['batch_size']];final=output/f'batch_{bi:05d}.npz'
                if final.exists():
                    with np.load(final,allow_pickle=False) as old:
                        assert old['uuids'].tolist()==[str(rows[i]['uuid']) for i in ids]
                        assert old['sizes'].shape==(50,len(ids))
                    completed_rows+=len(ids);continue
                batch=move_batch(collate_internal([rows[i] for i in ids]),'cuda:0')
                safe=inference_inputs(batch);sizes=shared['sizes'][ids]
                xyz=utils.remove_partial_mean_with_mask(batch['positions'],batch['atom_mask'],batch['fragment_mask'])
                refpos=[];refelements=[];refoff=[0]
                for i in range(len(ids)):
                    n=int(batch['atom_mask'][i].sum())
                    refpos.append(xyz[i,:n].cpu().numpy());refelements.append(batch['one_hot'][i,:n].argmax(-1).cpu().numpy());refoff.append(refoff[-1]+n)
                arrays=dict(uuids=np.asarray([str(rows[i]['uuid']) for i in ids]),sizes=sizes.T,
                    true_positions=np.concatenate(refpos).astype(np.float32),true_atom_types=np.concatenate(refelements).astype(np.int16),
                    true_offsets=np.asarray(refoff,dtype=np.int64),fragment_counts=batch['fragment_mask'].sum(1).flatten().cpu().numpy().astype(np.int16))
                progress=output/f'partial_{bi:05d}.npz';pos=[];elem=[];offset=[0];first=0
                if progress.exists():
                    with np.load(progress,allow_pickle=False) as old:
                        assert old['uuids'].tolist()==arrays['uuids'].tolist()
                        first=int(old['draws_done']);pos=[old['positions'].copy()];elem=[old['atom_types'].copy()];offset=old['offsets'].tolist()
                        assert len(offset)==first*len(ids)+1
                for draw in range(first,50):
                    if (ROOT/'STOP').exists() or (common.BUNDLE/'STOP').exists():
                        update(status='stopped',batch=bi,draw=draw,completed_rows=completed_rows);return
                    if prior+time.monotonic()-begun>18*3600:raise TimeoutError('Frozen18h arm budget reached; no automatic extension')
                    seed(100000+bi*50+draw)
                    draw_sizes=torch.from_numpy(sizes[:,draw].astype(np.int64)).cuda()
                    update(status='running',batch=bi,draw=draw,completed_rows=completed_rows)
                    chain,mask=model.sample_chain(safe,sample_fn=lambda _:draw_sizes,keep_frames=1)
                    if not torch.isfinite(chain).all():raise FloatingPointError('Nonfinite candidate; preserving completed slots and failure location')
                    for i in range(len(ids)):
                        n=int(mask[i].sum());pos.append(chain[0,i,:n,:3].cpu().numpy())
                        elem.append(chain[0,i,:n,3:].argmax(-1).cpu().numpy());offset.append(offset[-1]+n)
                    save_npz(progress,**arrays,positions=np.concatenate(pos).astype(np.float32),atom_types=np.concatenate(elem).astype(np.int16),
                        offsets=np.asarray(offset,dtype=np.int64),failed=np.zeros((50,len(ids)),dtype=bool),draws_done=np.asarray(draw+1))
                os.replace(progress,final);completed_rows+=len(ids)
                update(status='running',batch=bi,draw=50,completed_rows=completed_rows)
                print(json.dumps({'arm':a.arm,'completed_rows':completed_rows,'seconds':prior+time.monotonic()-begun}),flush=True)
        assert completed_rows==4161
        write_json(output/'completed.json',{'rows':4161,'candidates_per_row':50,'protocol_sha256':digest(ROOT/'protocol.json')})
        update(status='complete',completed_rows=completed_rows)
    except Exception as exc:
        update(status='failed',batch=bi,draw=draw,completed_rows=completed_rows,error=repr(exc),traceback=traceback.format_exc())
        raise
if __name__=='__main__':main()
