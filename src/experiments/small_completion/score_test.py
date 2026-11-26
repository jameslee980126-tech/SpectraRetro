"""Labels are kept outside frozen ranking inputs; complete reaction evaluation."""
import argparse,csv,json,sys
from collections import defaultdict
import numpy as np
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
sys.path.insert(0,str(HERE.parent/'eval_package'))
import eval_runtime as common
from eval_runtime import *
ROOT=common.ROOT

def read_csv(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def main():
    global ROOT,WORK,DATA,OLD
    p=argparse.ArgumentParser();p.add_argument('--arm',required=True,choices=ARMS)
    p.add_argument('--run-root',type=Path,required=True);p.add_argument('--bundle-root',type=Path,required=True);a=p.parse_args()
    common.configure(a.bundle_root,a.run_root);ROOT,WORK,DATA,OLD=common.ROOT,common.WORK,common.DATA,common.OLD
    if (ROOT/'evaluation'/a.arm/'metrics.json').exists():raise FileExistsError('Already scored arm')
    protocol=json.loads((ROOT/'protocol.json').read_text(encoding='utf8'))
    if a.arm not in protocol['arms']:raise ValueError('Arm outside frozen evaluation protocol')
    for path,h in protocol['sources'].items():
        if digest(path)!=h:raise ValueError('Frozen evaluation input changed: '+path)
    sys.path.insert(0,str(WORK))
    from fusion_reranker.runtime import FrozenFusionRanker
    out=ROOT/'evaluation'/a.arm
    decoded_rows=read_csv(out/'decoded/result.csv');decoded={r['uuid']:r for r in decoded_rows}
    reference_rows=read_csv(ROOT/'reference_replay.csv');references={r['uuid']:r['reference'] for r in reference_rows}
    if not len(reference_rows)==len(references)==protocol['rows']:raise ValueError('Incomplete frozen test XYZ reference bank')
    if set(references)!=set(decoded) or any(decoded[uid][str(protocol['draws']+1)]!=references[uid] for uid in decoded):
        raise ValueError('Arm references differ from shared Linux XYZ replay')
    table_rows=read_csv(common.BUNDLE/'final_test/prepared/similarity_test_table.csv');table={r['uuid']:r for r in table_rows}
    meta=read_csv(common.BUNDLE/'final_test/prepared/test_record_map.csv')
    assert len(decoded_rows)==len(decoded)==len(table)==len(meta)==protocol['rows']
    assert set(decoded)==set(table)=={r['uuid'] for r in meta}
    stage1={r['sample_id']:r for r in [json.loads(s) for s in (common.BUNDLE/('final_test/prepared/stage1_known_test_predictions.jsonl' if a.arm.startswith('known_') else 'final_test/prepared/stage1_test_predictions.jsonl')).read_text().splitlines()]}
    grouped=defaultdict(list)
    for r in meta:grouped[r['reaction_id']].append(r)
    assert len(grouped)==protocol['reactions']
    mode=a.arm.split('_seed')[0];ranker=FrozenFusionRanker(mode,'fusion');metrics={};individual={}
    for gate in [True,False]:
        requests=[];targets=[];classes=[];keys=sorted(grouped)
        for key in keys:
            group=sorted(grouped[key],key=lambda r:int(r['uuid']))
            cls={int(r['class_metadata_only']) for r in group};assert len(cls)==1
            products={table[r['uuid']]['product'] for r in group};assert len(products)==1
            center=stage1[key];hits=sum(int(v)==int(center['gold_candidate']) for v in center['sampled_candidate_indices'][:protocol['draws']])
            assert len(center['sampled_candidate_indices'])>=protocol['draws']
            report_class=cls.pop()
            request={'reaction_id':key,'product':products.pop(),'fragments':[]}
            if mode=='known':request['class_id']=report_class
            for m in group:
                uid=m['uuid'];t=table[uid];d=decoded[uid]
                request['fragments'].append({'uuid':uid,'fragments':t['fragments'],'anchors':t['anchors'],
                    'draws':[d[str(j+1)] if not gate or j<hits else 'MISSING' for j in range(protocol['draws'])]})
            requests.append(request);targets.append('.'.join(decoded[r['uuid']][str(protocol['draws']+1)] for r in group));classes.append(report_class)
        rows=[]
        for start in range(0,len(keys),32):
            ranked=ranker.rank_batch(requests[start:start+32])
            for i,result in enumerate(ranked,start):
                fusion=result['candidates'];freq=sorted(fusion,key=lambda c:c['baseline_rank']);target=targets[i]
                row={'reaction_id':keys[i],'class_id':classes[i],
                     'pool_coverage':int(target in [c['candidate'] for c in fusion]),
                     'unique_candidates':len(fusion),'missing_draw_fraction':sum(c['frequency'] for c in fusion if 'MISSING' in c['candidate'])/protocol['draws']}
                for method,candidates in [('fusion',fusion),('frequency',freq)]:
                    for k in [1,3,5,10]:row[f'{method}_top{k}']=int(target in [c['candidate'] for c in candidates[:k]])
                rows.append(row)
            print(json.dumps({'arm':a.arm,'gate':gate,'scored':min(start+32,len(keys))}),flush=True)
        name='stage1_gated' if gate else 'stage2_ungated_diagnostic';individual[name]=rows
        metrics[name]={}
        for cls in [0,1,2,3,4,5,6]:
            selected=[r for r in rows if not cls or r['class_id']==cls]
            metrics[name]['all' if cls==0 else f'class{cls}']={'n':len(selected),**{
                k:float(np.mean([r[k] for r in selected])) for k in rows[0] if k not in ['reaction_id','class_id']}}
    write_json(out/'reaction_metrics.json',individual)
    write_json(out/'metrics.json',{'arm':a.arm,'evaluation':'complete original test split, 300 draws; frozen custom fusion and frequency reported separately; exact XYZ-decoded string match',
        'not_historical_server_absolute_score':True,'no_deployment':True,
        'platform_note':'same Linux GPU generation plus frozen Windows CPU decode/rank for both arms; reference-conditioned Stage2, not product-only end-to-end','metrics':metrics,
        'protocol_sha256':digest(ROOT/'protocol.json'),'decoded_sha256':digest(out/'decoded/result.csv'),
        'reference_replay_csv_sha256':digest(ROOT/'reference_replay.csv'),
        'ranking_manifest_sha256':digest(WORK/'fusion_reranker/manifest.json')})
if __name__=='__main__':main()
