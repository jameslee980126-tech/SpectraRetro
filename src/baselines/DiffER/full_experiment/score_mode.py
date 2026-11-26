"""Replay raw sample streams; reproduce the prior raw-slot and native plurality views."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import time
from collections import Counter
from rdkit import RDLogger
from evaluate_full_retro_topk import canonicalize_precursor_set
from native_converter import product_identity, native_plurality_ranking

KS = [1,3,5,10]
def condition(raw, mode):
    match = re.match(r'^\[RXN_([1-6])\]', raw)
    if mode == 'known':
        assert match, raw
        label, raw = int(match.group(1)), raw[match.end():]
    else:
        assert match is None
        label = None
    product = product_identity(raw)
    assert product is not None
    return label, product

def score(root, mode, samples_path, output_dir, raw_test, protocol_path):
    RDLogger.DisableLog('rdApp.*')
    d=Path(output_dir).resolve()
    release=Path(__file__).resolve().parents[4]
    if release not in d.parents or d.relative_to(release).parts[0]!='outputs':
        raise ValueError('Score output must be under the release outputs/ directory')
    d.mkdir(parents=True,exist_ok=False)
    protocol=json.loads(Path(protocol_path).read_text(encoding='utf-8'))
    augmentations=protocol['test_augmentation'];samples_per_input=protocol['samples_per_test_augmentation']
    budget=augmentations*samples_per_input
    assert budget==50,'User-confirmed total candidate budget must remain50'
    raw=list(csv.DictReader(Path(raw_test).open(encoding='utf-8-sig')))
    assert len(raw)==9250
    # Native eval first appends samples per literal input, then its converter
    # merges literals in first-encounter dictionary order. Preserve both orders.
    literals={};raw_draws=0;inputs=0
    with Path(samples_path).open(encoding='utf-8') as f:
        for batch,line in enumerate(f):
            r=json.loads(line);assert r['batch']==batch
            src=r['sources'];samples=r['samples'];n=len(src)
            assert len(samples)==n*samples_per_input and len(r['reference_targets'])==n
            for s in src:literals.setdefault(s,[])
            for i,sample in enumerate(samples):
                bank=literals[src[i%n]]
                if len(bank)<budget:bank.append(sample)
            raw_draws+=len(samples);inputs+=n
    assert inputs==9250*augmentations and raw_draws==9250*budget
    banks={}
    for literal,samples in literals.items():
        key=condition(literal,mode);bank=banks.setdefault(key,[])
        bank.extend(samples[:max(0,budget-len(bank))])
    metrics={view:{'total':9250,'hits':{str(k):0 for k in KS},'by_class':{}} for view in ['unified_raw_slots','native_plurality']}
    rankings={}
    for key,bank in banks.items():
        assert len(bank)==budget,(key,len(bank))
        raw_predictions=[s.replace('?','') for s in bank[:10]]
        native=native_plurality_ranking(bank,key[1],samples_per_input)
        rankings[key]=(raw_predictions,native)
    per=[]
    with (d/'predictions.jsonl').open('w',encoding='utf-8') as out:
        for i,r in enumerate(raw):
            left,_,right=r['rxn_smiles'].split('>');label=int(r['class'])
            key=(label if mode=='known' else None,product_identity(right))
            assert key in rankings,(i,key)
            target=canonicalize_precursor_set(left);assert target is not None
            candidates,native=rankings[key]
            ranks=[]
            for view,predictions in [('unified_raw_slots',candidates),('native_plurality',native)]:
                rank=next((j for j,s in enumerate(predictions,1) if canonicalize_precursor_set(s)==target),None)
                ranks.append(rank)
                by=metrics[view]['by_class'].setdefault(str(label),{'total':0,'hits':{str(k):0 for k in KS}});by['total']+=1
                for k in KS:
                    hit=rank is not None and rank<=k
                    metrics[view]['hits'][str(k)]+=hit;by['hits'][str(k)]+=hit
            out.write(json.dumps({'source_row':i,'class':label,'canonical_product':key[1],'predictions':candidates,
                     'native_predictions':native[:10],'native_hit_rank':ranks[1]})+'\n')
            per.append({'source_row':i,'class':label,'raw_match_rank':ranks[0],'native_match_rank':ranks[1],
                        'raw_candidates':len(candidates),'native_candidates':len(native)})
    for view,record in metrics.items():
        record['accuracy']={k:v/9250 for k,v in record['hits'].items()}
        assert sum(c['total'] for c in record['by_class'].values())==9250
        for c in record['by_class'].values():c['accuracy']={k:v/c['total'] for k,v in c['hits'].items()}
    with (d/'per_reaction.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(per[0]));w.writeheader();w.writerows(per)
    result={'passed':True,'model':'DiffER single categorical diffusion','mode':mode,'seed':0,'metrics':metrics,
            'raw_draws':raw_draws,'augmented_inputs':inputs,'conditions':len(banks),'native_budget_per_condition':budget,
            'primary':'unified_raw_slots','test_denominator':9250,'raw_order_slots_compacted':False}
    (d/'metrics.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result),flush=True)
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--mode',choices=['known','unknown'],required=True)
    ap.add_argument('--samples',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--test-csv',type=Path,required=True);ap.add_argument('--protocol',type=Path,required=True)
    a=ap.parse_args();score(a.root,a.mode,a.samples,a.output,a.test_csv,a.protocol)

if __name__=='__main__':main()
