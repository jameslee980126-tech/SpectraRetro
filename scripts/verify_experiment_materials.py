"""Verify experiment-specific artifacts and recount saved observations offline."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]

def read(rel):return json.loads((ROOT/rel).read_text(encoding='utf-8-sig'))

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def checked(item):
    path=ROOT/item['path']
    if not path.is_file() or sha(path)!=item['sha256']:raise ValueError('Artifact identity differs: '+item['path'])
    if 'bytes' in item and path.stat().st_size!=item['bytes']:raise ValueError('Artifact length differs: '+item['path'])

def main():
    materials=read('results/reproduction/material_index.json')['files']
    for item in materials:checked(item)
    weights={x['path']:x for x in read('checkpoints/WEIGHTS.json')['files']}
    small=read('configs/experiments/small_completion.json')['runs']
    for run in small:
        p=read(run['protocol']);arm=run['mode']+'_seed'+str(run['training_seed'])
        assert p['reactions']==2400 and p['draws']==300 and p['steps']==100
        assert p['checkpoints'][arm]['sha256']==weights[run['completion_checkpoint']]['sha256']
        for item in run['inputs'].values():checked(item)
        for field in ['completion_checkpoint','reaction_center_checkpoint','size_checkpoint','composition_prior']:
            assert run[field] in weights
    prior=read('results/reproduction/chemical_prior/protocol.json')
    inputs=read('results/reproduction/chemical_prior/input_map.json')
    assert set(inputs)==set(prior['sources'])
    for original,item in inputs.items():
        assert item.get('source_sha256', item['sha256']) == prior['sources'][original]
        checked(item)
    recounted={}
    for arm,expected in [('control',[1319,1731,1802,1823]),('compatibility',[1458,1771,1817,1828])]:
        rel='checkpoints/ablations/chemical_prior/'+arm+'.pt'
        assert weights[rel]['sha256']==prior['checkpoints'][arm]['sha256']
        observations=read('results/reproduction/chemical_prior/'+arm+'/reaction_metrics.json')['stage1_gated']
        assert len(observations)==len({x['reaction_id'] for x in observations})==2400
        hits=[sum(int(x['fusion_top'+str(k)]) for x in observations) for k in [1,3,5,10]]
        assert hits==expected,(arm,hits);recounted[arm]=hits
    cases=read('results/reproduction/single_multistep/source_map.json')
    assert not cases['unresolved']
    for item in cases['files']:checked(item)
    for item in read('results/cases/agent_generation/connection_source_map.json').values():checked(item)
    assessments=read('results/cases/agent_assessment/manifest.json')['cases'];routes=attempts=0
    for case in assessments:
        checked(case);doc=read(case['path']);a=doc['route_assessment']
        assert sum(len(x['attempts']) for x in a['observations'])==a['calls']
        attempts+=a['calls'];routes+=len(doc['routes'])
        for route in doc['routes']:
            result=route['llm_assessment']
            total=round(sum(float(result['scores'][k])*v for k,v in a['criteria_weights'].items()),4)
            assert abs(total-result['total_score'])<1e-8
    quantum=read('results/source_data/mechanistic_interpretability/supplement/quantum/electronic_states.json')
    assert len(quantum['states'])==34
    for state in quantum['states']:
        result=ROOT/'results/source_data/mechanistic_interpretability/supplement/quantum/dft'/state['state']/'result.json'
        assert result.relative_to(ROOT/'results/source_data/mechanistic_interpretability/supplement/quantum').as_posix()==state['result']
        assert sha(result)==state['sha256']
    print(json.dumps(dict(status='passed',material_files=len(materials),small_protocols=len(small),
        prior_protocol_inputs=len(inputs),prior_top1_3_5_10_hits=recounted,case_source_bindings=len(cases['files']),
        assessed_routes=routes,saved_llm_attempts=attempts,electronic_states=len(quantum['states']),
        scope='Artifact integrity, protocol bindings and saved-result replay; no model inference or service calls.'),indent=2))

if __name__=='__main__':main()
