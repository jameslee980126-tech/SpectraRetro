"""Training, conditional generation and evaluation for the complete reaction set."""
from contextlib import nullcontext
from pathlib import Path
import csv
import hashlib
import json
import os
import sys
import time

HERE = Path(__file__).resolve().parent
RELEASE = HERE.parents[2]
BENCHMARK = RELEASE / 'experiments/section_4_results/4_1_benchmark'
DATA = RELEASE / "experiments/spectraretro_inputs"


def release_path(value):
    """Resolve command-line paths against the distributed repository."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (RELEASE/path).resolve()


def validate_output(value):
    """Require a new output directory and preserve the distributed artifacts."""
    path = release_path(value)
    if path == RELEASE:
        raise ValueError('The repository root cannot be an output directory')
    if RELEASE in path.parents and path.relative_to(RELEASE).parts[0] != 'outputs':
        raise ValueError('Outputs inside the repository must be placed under outputs/')
    if path.exists():
        raise FileExistsError('Output directory already exists; choose a new directory: '+str(path))
    return path


def setup():
    os.environ['RETRO_STAGE2_BUNDLE_ROOT'] = str(HERE)
    os.environ['RETRO_STAGE2_OLD_ROOT'] = str(HERE / 'old')
    os.environ['RETRO_STAGE2_CYCLE_ROOT'] = str(HERE / 'cycle')
    for path in [HERE, HERE/'old', HERE/'old/stage2', HERE/'old/vendor/stage2',
                 HERE/'cycle/unknown_compatibility', RELEASE/'src']:
        sys.path.insert(0, str(path))
    import bootstrap  # noqa: F401


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf8')
    os.replace(temporary, path)


def save_torch(path, value):
    import torch
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def seed(value):
    import torch
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def torch_load(path, **kwargs):
    import torch
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


def configure(device='cuda:0'):
    import torch
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('A CUDA device is required for diffusion training and sampling')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def default_config(batch_size=64):
    return dict(in_node_nf=26, n_dims=3, context_node_nf=2, hidden_nf=128,
                activation='silu', tanh=True, n_layers=8, attention=False,
                norm_constant=1e-6, inv_sublayers=2, sin_embedding=False,
                normalization_factor=100, aggregation_method='sum',
                diffusion_steps=500, diffusion_noise_schedule='polynomial_2',
                diffusion_noise_precision=1e-5, diffusion_loss_type='l2',
                normalize_factors=[1,4,10], include_charges=False,
                model='egnn_dynamics', data_path=str(DATA),
                train_data_prefix='similarity_train', val_data_prefix='similarity_eval',
                batch_size=batch_size, lr=2e-4, torch_device='cuda:0',
                test_epochs=10000, n_stability_samples=1, normalization='batch_norm',
                data_augmentation=False, center_of_mass='fragments', inpainting=False,
                anchors_context=True, diffusion_process='edm')


def load_model(mode, checkpoint, device='cuda:0'):
    setup()
    import torch
    from spectraretro.completion.composition_model import construct
    saved = torch_load(checkpoint, map_location='cpu')
    if saved.get('arm') != mode+'_seed0' or not saved.get('compatibility_enabled'):
        raise ValueError('Checkpoint mode or composition conditioning does not match')
    if saved.get('observed_graph_enabled') or bool(saved['class_input']) != (mode=='known'):
        raise ValueError('Checkpoint conditioning differs from the selected protocol')
    config = dict(saved['configuration'], torch_device=str(device))
    model = construct(mode, **config)
    model.load_state_dict(saved['state_dict'], strict=True)
    return model


def batch(rows, probabilities, mode, indices, device='cuda:0'):
    import torch
    from spectraretro.completion.composition_model import implementation
    value = implementation(mode).prepare_batch([rows[i] for i in indices], probabilities[indices])
    return {k:v.to(device) if torch.is_tensor(v) else v for k,v in value.items()}


def train(args):
    """Original L2 objective, complete train/eval sets and validation selection."""
    args.output = validate_output(args.output)
    setup()
    configure()
    import numpy as np
    import torch
    from spectraretro.completion.composition_model import construct
    data = {s:torch_load(DATA/f'old/newdata/{s}_geometry.pt', map_location='cpu')
            for s in ['train','eval']}
    if [len(data[s]) for s in ['train','eval']] != [126561,16059]:
        raise ValueError('The full-data fragment split must contain 126561/16059 records')
    probabilities = {}
    for s, rows in data.items():
        with np.load(DATA/f'cycle/unknown_compatibility/{s}_prior.npz') as prior:
            if prior['uuids'].astype(str).tolist() != [str(r['uuid']) for r in rows]:
                raise ValueError('Composition prior UUID order differs from geometry rows')
            probabilities[s] = prior['probabilities'].copy()
    args.output.mkdir(parents=True, exist_ok=False)
    config = default_config()
    seed(0)
    model = construct(args.mode, **config)
    initial = torch_load(BENCHMARK/f'{args.mode}-class/checkpoints/initial_model.pt', map_location='cpu')
    model.load_state_dict(initial['state_dict'], strict=True)
    model = model.cuda()
    optimizer = torch.optim.AdamW(model.edm.parameters(), lr=2e-4,
                                  weight_decay=1e-12, amsgrad=True)
    schedule = set(json.loads((HERE/'validation_schedule.json').read_text())[args.mode]['validation_epochs'])
    curves, best, step, consecutive = [], None, 0, 0
    started = time.monotonic()
    for epoch in range(1,2001):
        order = torch.randperm(len(data['train']), generator=torch.Generator().manual_seed(epoch)).tolist()
        total = 0.
        model.train()
        for lo in range(0,len(order),64):
            if (args.output/'STOP').exists() or time.monotonic()-started > 28*86400:
                raise InterruptedError('Training stopped before another complete epoch')
            ids = order[lo:lo+64]
            value = batch(data['train'],probabilities['train'],args.mode,ids)
            model.zero_grad(set_to_none=True)
            seed(100000+step)
            loss = model.forward(value,True)[4]
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite original EDM L2')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.edm.parameters(),1,error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach())*len(ids)
            step += 1
        report = dict(epoch=epoch,training_l2=total/126561,steps=step)
        if epoch in schedule:
            model.eval()
            loss_sum = 0.
            with torch.no_grad():
                for cls in range(6):
                    indices = [i for i,r in enumerate(data['eval']) if int(r['class_index'])==cls]
                    for lo in range(0,len(indices),64):
                        ids=indices[lo:lo+64]
                        seed(900000+cls*10000+lo)
                        loss=model.forward(batch(data['eval'],probabilities['eval'],args.mode,ids),False)[4]
                        if not torch.isfinite(loss):
                            raise FloatingPointError('Nonfinite validation L2')
                        loss_sum += float(loss)*len(ids)
            report['validation_l2']=loss_sum/16059
            curves.append(dict(epoch=epoch,validation_l2=report['validation_l2']))
            if best is None or report['validation_l2']<best['validation_l2']:
                best=dict(epoch=epoch,validation_l2=report['validation_l2'])
                save_torch(args.output/'best.pt',dict(**best,arm=args.mode+'_seed0',
                    state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                    configuration=config,class_input=args.mode=='known',
                    compatibility_enabled=True,observed_graph_enabled=False))
        save_torch(args.output/'last.pt',dict(state_dict=model.state_dict(),
            optimizer=optimizer.state_dict(),completed_epochs=epoch,steps=step,best=best,
            configuration=config,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()))
        write_json(args.output/'training_curve.json',curves)
        write_json(args.output/'status.json',dict(**report,best=best))
        print(json.dumps(report),flush=True)
        if epoch>=1100 and epoch%100==0:
            previous=min(r['validation_l2'] for r in curves if r['epoch']<epoch)
            gain=max(0.,(previous-report['validation_l2'])/max(abs(previous),1e-12))
            consecutive=consecutive+1 if gain<.001 else 0
            if consecutive>=2:
                break


def sample(args):
    """Replay the fixed 300-draw size bank and 100-step conditional sampler."""
    args.output = validate_output(args.output)
    setup()
    device=getattr(args,'device','cuda:0')
    configure(device)
    import numpy as np
    import torch
    from spectraretro.completion.composition_model import implementation
    from src import utils
    completion = implementation(args.mode)
    checkpoint=args.checkpoint
    model=load_model(args.mode,checkpoint,device).to(device).eval()
    draws=getattr(args,'draws',300)
    model.edm.T=getattr(args,'steps',100)
    rows=torch_load(DATA/'final_test/prepared/test_geometry.pt',map_location='cpu')
    with np.load(DATA/'final_test/prepared/test_prior.npz') as prior:
        if prior['uuids'].astype(str).tolist()!=[str(r['uuid']) for r in rows]:
            raise ValueError('Test prior UUIDs do not align with geometry')
        probabilities=prior['probabilities'].copy()
    sizes=np.load(BENCHMARK/f'{args.mode}-class/data/evaluation/shared_sizes.npz',allow_pickle=False)
    if sizes['uuids'].astype(str).tolist()!=[str(r['uuid']) for r in rows]:
        raise ValueError('Sampled size UUIDs do not align with geometry')
    order=json.loads((BENCHMARK/f'{args.mode}-class/data/evaluation/batch_order.json').read_text())
    if len(rows)!=15830 or sorted(order)!=list(range(15830)) or sizes['sizes'].shape!=(15830,300):
        raise ValueError('Incomplete 15830-fragment, 300-draw test bank')
    selection=getattr(args,'selection',None)
    if selection:
        selected=set(json.loads(Path(selection).read_text(encoding='utf8'))['reaction_ids'])
        with (DATA/'final_test/prepared/test_record_map.csv').open(encoding='utf-8-sig',newline='') as f:
            uuids={r['uuid'] for r in csv.DictReader(f) if r['reaction_id'] in selected}
        order=[i for i in order if str(rows[i]['uuid']) in uuids]
        if not order or len(order)!=len(uuids):
            raise ValueError('Incomplete diagnostic fragment selection')
    if not getattr(args,'diagnostic',False) and (draws!=300 or model.edm.T!=100 or selection):
        raise ValueError('The full protocol requires all reactions, 300 draws and 100 intervals')
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    with torch.no_grad():
        for bi,lo in enumerate(range(0,len(order),128)):
            ids=order[lo:lo+128]
            value=batch(rows,probabilities,args.mode,ids,device)
            safe = completion.sampling_inputs(value)
            xyz=utils.remove_partial_mean_with_mask(value['positions'],value['atom_mask'],value['fragment_mask'])
            refpos,refelem,refoff=[],[],[0]
            for i in range(len(ids)):
                n=int(value['atom_mask'][i].sum())
                refpos.append(xyz[i,:n].cpu().numpy())
                refelem.append(value['one_hot'][i,:n].argmax(-1).cpu().numpy())
                refoff.append(refoff[-1]+n)
            positions,elements,offsets=[],[],[0]
            for draw in range(draws):
                if (args.output/'STOP').exists() or time.monotonic()-started>96*3600:
                    raise InterruptedError('Sampling stopped at the configured boundary')
                seed(100000+bi*300+draw)
                sampled_sizes=torch.from_numpy(sizes['sizes'][ids,draw].astype(np.int64)).to(device)
                chain,mask=model.sample_chain(safe,sample_fn=lambda _:sampled_sizes,keep_frames=1)
                if not torch.isfinite(chain).all():
                    raise FloatingPointError('Nonfinite generated coordinates')
                for i in range(len(ids)):
                    n=int(mask[i].sum())
                    positions.append(chain[0,i,:n,:3].cpu().numpy())
                    elements.append(chain[0,i,:n,3:].argmax(-1).cpu().numpy())
                    offsets.append(offsets[-1]+n)
            target=args.output/f'batch_{bi:05d}.npz'
            temporary=target.with_suffix('.tmp')
            with temporary.open('wb') as stream:
                np.savez(stream,uuids=np.asarray([str(rows[i]['uuid']) for i in ids]),
                    sizes=sizes['sizes'][ids,:draws].T,positions=np.concatenate(positions).astype(np.float32),
                    atom_types=np.concatenate(elements).astype(np.int16),offsets=np.asarray(offsets,dtype=np.int64),
                    failed=np.zeros((draws,len(ids)),dtype=bool),draws_done=np.asarray(draws),
                    true_positions=np.concatenate(refpos).astype(np.float32),
                    true_atom_types=np.concatenate(refelem).astype(np.int16),true_offsets=np.asarray(refoff,dtype=np.int64),
                    fragment_counts=value['fragment_mask'].sum(1).flatten().cpu().numpy().astype(np.int16))
            os.replace(temporary,target)
            print(json.dumps(dict(batch=bi,completed_rows=min(lo+128,len(order)))),flush=True)
    write_json(args.output/'completed.json',dict(rows=len(order),candidates_per_row=draws,
        checkpoint_sha256=sha(checkpoint),diagnostic=getattr(args,'diagnostic',False)))


def write_ranked_reaction(stream, result):
    """Write the model-ranked candidates used for the main benchmark score."""
    record = dict(reaction_id=result['reaction_id'], product=result['product'],
                  mode=result['mode'], view='stage1_gated',
                  ranking_version=result['ranking_version'], candidates=[])
    for c in result['candidates']:
        record['candidates'].append(dict(
            rank=c['rank'], candidate=c['candidate'], fusion_score=c['fusion_score'],
            frequency=c['frequency'], frequency_rank=c['baseline_rank'],
            contains_missing='MISSING' in c['candidate']))
    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')


def score(args):
    """Score exact precursor matches for the main benchmark and its diagnostic comparisons."""
    args.output = validate_output(args.output)
    import numpy as np
    from collections import defaultdict
    from spectraretro.fusion_reranker.runtime import FrozenFusionRanker
    def read(path):
        with Path(path).open(encoding='utf-8-sig',newline='') as stream:
            return list(csv.DictReader(stream))
    decoded_rows=read(args.input)
    decoded={r['uuid']:r for r in decoded_rows}
    references={r['uuid']:r['reference'] for r in read(BENCHMARK/f'{args.mode}-class/data/evaluation/reference_replay.csv')}
    table={r['uuid']:r for r in read(DATA/'final_test/prepared/similarity_test_table.csv')}
    meta=read(DATA/'final_test/prepared/test_record_map.csv')
    draws=getattr(args,'draws',300)
    selection=getattr(args,'selection',None)
    if selection:
        selected=set(json.loads(Path(selection).read_text(encoding='utf8'))['reaction_ids'])
        meta=[r for r in meta if r['reaction_id'] in selected]
        uuids={r['uuid'] for r in meta}
        references={k:v for k,v in references.items() if k in uuids}
        table={k:v for k,v in table.items() if k in uuids}
    if len(decoded_rows)!=len(references) or len(decoded)!=len(references) or set(decoded)!=set(references) or set(decoded)!=set(table):
        raise ValueError('Full test fragment coverage is required')
    if any(decoded[uid][str(draws+1)]!=ref for uid,ref in references.items()):
        raise ValueError('Decoded references differ from the fixed XYZ replay')
    center_path=args.center_predictions
    center_rows=[json.loads(line) for line in Path(center_path).read_text(encoding='utf8').splitlines()]
    centers={r['sample_id']:r for r in center_rows}
    if len(centers)!=len(center_rows):
        raise ValueError('Duplicate reaction center prediction identity')
    groups=defaultdict(list)
    for row in meta:
        groups[row['reaction_id']].append(row)
    if not getattr(args,'diagnostic',False) and (len(groups)!=9250 or draws!=300 or selection):
        raise ValueError('The reaction denominator must be 9250')
    if set(groups)!=set(centers):
        raise ValueError('Center predictions must cover exactly the evaluated reactions')
    ranker=FrozenFusionRanker(args.mode,'fusion')
    args.output.mkdir(parents=True,exist_ok=False)
    ranked_path = args.output / 'ranked_candidates.jsonl'
    pending = ranked_path.with_suffix('.jsonl.tmp')
    metrics,individual={},{}
    for gated in [True,False]:
        requests,targets,classes,keys=[],[],[],sorted(groups)
        for key in keys:
            rows=sorted(groups[key],key=lambda r:int(r['uuid']))
            cls={int(r['class_metadata_only']) for r in rows}
            if len(cls)!=1:
                raise ValueError('Inconsistent reaction class metadata')
            cls=cls.pop()
            center=centers[key]
            sampled=center['sampled_candidate_indices'][:draws]
            if len(sampled)!=draws:
                raise ValueError('Stage1 must retain 300 sampled center indices')
            if center['product']!=rows[0]['product']:
                raise ValueError('Reaction center product differs from the fragment identity map')
            hits=sum(int(i)==int(center['gold_candidate']) for i in sampled)
            request=dict(reaction_id=key,product=table[rows[0]['uuid']]['product'],fragments=[])
            if args.mode=='known':
                request['class_id']=cls
            for row in rows:
                uid=row['uuid']; t=table[uid]; d=decoded[uid]
                request['fragments'].append(dict(uuid=uid,fragments=t['fragments'],anchors=t['anchors'],
                    draws=[d[str(j+1)] if not gated or j<hits else 'MISSING' for j in range(draws)]))
            requests.append(request)
            targets.append('.'.join(decoded[r['uuid']][str(draws+1)] for r in rows))
            classes.append(cls)
        scores=[]
        sink = pending.open('x', encoding='utf8', newline='\n') if gated else nullcontext()
        with sink as ranked_file:
            for lo in range(0,len(keys),32):
                for i,result in enumerate(ranker.rank_batch(requests[lo:lo+32]),lo):
                    if ranked_file is not None:
                        write_ranked_reaction(ranked_file, result)
                    candidates=result['candidates']; target=targets[i]
                    row=dict(reaction_id=keys[i],class_id=classes[i],
                        pool_coverage=int(target in [c['candidate'] for c in candidates]),
                        unique_candidates=len(candidates),
                        missing_draw_fraction=sum(c['frequency'] for c in candidates if 'MISSING' in c['candidate'])/draws)
                    for name,ranking in [('fusion',candidates),('frequency',sorted(candidates,key=lambda c:c['baseline_rank']))]:
                        for k in [1,3,5,10]:
                            row[f'{name}_top{k}']=int(target in [c['candidate'] for c in ranking[:k]])
                    scores.append(row)
        name='stage1_gated' if gated else 'stage2_ungated_diagnostic'
        individual[name]=scores
        metrics[name]={}
        for cls in range(7):
            chosen=[r for r in scores if cls==0 or r['class_id']==cls]
            if not chosen:
                metrics[name]['all' if cls==0 else f'class{cls}']=dict(n=0)
                continue
            metrics[name]['all' if cls==0 else f'class{cls}']=dict(n=len(chosen),**{
                k:float(np.mean([r[k] for r in chosen])) for k in scores[0] if k not in ['reaction_id','class_id']})
    write_json(args.output/'reaction_metrics.json',individual)
    write_json(args.output/'metrics.json',dict(mode=args.mode,metrics=metrics,
        evaluation='Complete-precursor exact match after XYZ decoding (SI S3.6)',
        diagnostic=getattr(args,'diagnostic',False),draws_per_fragment=draws,
        center_predictions_sha256=sha(center_path),
        decoded_sha256=sha(args.input),reference_sha256=sha(BENCHMARK/f'{args.mode}-class/data/evaluation/reference_replay.csv')))
    pending.replace(ranked_path)
