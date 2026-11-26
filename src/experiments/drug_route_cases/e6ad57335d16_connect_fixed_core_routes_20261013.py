"""Connect actual fixed-core model candidates; stock only defines terminal leaves.

Every intermediate's known core was explicitly supplied in the completion test.
This evaluates conditional route connectivity, not autonomous core prediction.
No stock lookup, stock graft, unknown reference structure or LLM generates edges.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,copy
from pathlib import Path
from collections import Counter
from run_last_epoch_diagnostic import ROOT,REPO,CASE,read,write,sha
sys.path[:0]=[str(REPO),str(CASE)]
from agentic_retrosynthesis.planner import Planner,canonical_smiles
from collect_nem_paired_results import references,canonical,pair,assert_route_connected

BASE=REPO/'experiments/case_partition_completion_20261013'
PARENT=BASE/'fixed_core_completion_20261013'
OUT=PARENT/'connected_routes'

def key(parts):return tuple(sorted(canonical(s) for s in parts))
def main():
    assert read(PARENT/'audit.json')['state']=='complete';assert not OUT.exists();OUT.mkdir()
    bank=read(PARENT/'decoded_bank.json');inputs=read(PARENT/'generation_inputs.json')
    raw_sha=sha(PARENT/'raw_samples.pt');assert raw_sha==read(PARENT/'report.json')['raw_sha256']
    requests=read(BASE/'whole_routes/requests.json')
    source_path=REPO/'experiments/case_all_exact_data_v3_20261012/reaction_pairs.json'
    # Only product mapping and input fragment UUIDs are used to construct pools.
    # Full precursor reference strings are not used below to make candidates.
    sources=read(source_path)
    templates=[dict(step_id=r['step_id'],product=canonical_smiles(r['product']),
        fragment_uuids=[f['uuid'] for f in r['fragments']]) for r in sources if r['step_id']!='LAP-06']
    write(OUT/'input_product_mapping.json',templates)
    files=[Path(__file__),PARENT/'audit.json',PARENT/'protocol.json',PARENT/'decoded_bank.json',PARENT/'raw_samples.pt',
        PARENT/'generation_inputs.json',OUT/'input_product_mapping.json',BASE/'whole_routes/requests.json']
    write(OUT/'protocol.json',dict(task='Fixed-core model candidates connected into routes',
        provided_intermediate_cores=True,core_prediction_or_LLM_not_tested=True,
        stock_used_for_candidate_generation=False,stock_used_only_for_terminal_membership=True,
        supplied_terminal_boundary_from_literature=True,new_training=False,new_sampling=False,
        candidate_order='Generated paired-draw frequency, canonical tie break',max_depth=12,max_product_queries=100,
        max_states=10000,max_routes=3,reference_precursors_for_evaluation_only=True,
        source_sha256={str(p):sha(p) for p in files}))
    pooled={}
    for t in templates:
        product=t['product'];record=pooled.setdefault(product,dict(product=product,draw_slots=0,counts=Counter(),support={}))
        record['draw_slots']+=30
        for draw in range(30):
            values=[bank[u][draw]['smiles'] for u in t['fragment_uuids']]
            if any(s is None for s in values):continue
            candidate=key(values)
            if product in candidate:continue
            record['counts'][candidate]+=1
            record['support'].setdefault(candidate,[]).append(dict(input_template=t['step_id'],draw=draw,
                fragment_uuids=t['fragment_uuids'],raw_indices=[bank[u][draw]['raw_index'] for u in t['fragment_uuids']]))
    pools={}
    for product,r in pooled.items():
        candidates=[]
        for precursors,count in sorted(r['counts'].items(),key=lambda x:(-x[1],x[0])):
            candidates.append(dict(precursors=list(precursors),score=count/r['draw_slots'],
                source='fixed_known_core_model_completion',metadata=dict(given_known_core_condition=True,
                    inventory_completion_used=False,reference_inputs=True,reference_unknown_structure_input=False,
                    source_bank=str(PARENT/'decoded_bank.json'),source_bank_sha256=sha(PARENT/'decoded_bank.json'),
                    raw_sha256=raw_sha,support=r['support'][precursors],paired_draw_frequency=count)))
        pools[product]=dict(product=product,candidates=candidates,paired_draw_slots=r['draw_slots'])
    write(OUT/'candidate_pools.json',pools)
    class FixedCoreReplayModel:
        def predict(self,product,top_k):return copy.deepcopy(pools.get(canonical_smiles(product),{}).get('candidates',[]))
    results={};evaluations={}
    for job,request in requests.items():
        folder=OUT/job;folder.mkdir();write(folder/'request.json',request)
        result=Planner(FixedCoreReplayModel(),set(request['stock']),None,max_depth=12,max_model_calls=100,
            expansion_top_k=100,max_routes=3,max_states=10000,cost_policy='rank',recovery_rounds=0).plan(request['target']).to_dict()
        assert not result['errors'];write(folder/'result.json',result);results[job]=result
    # Reference reactions are consulted for evaluation after every route search.
    for job,result in results.items():
        request=requests[job];_,refs=references(job);documented=[]
        for route in result['routes']:
            assert_route_connected(route,request);ids=[]
            for step in route['steps']:
                actual=pools[canonical_smiles(step['product'])]['candidates']
                assert any(key(c['precursors'])==key(step['precursors']) and c['score']==step['model_score'] for c in actual)
                assert step['metadata']['inventory_completion_used'] is False
                ids.append(next((r['step'] for r in refs if r['product']==canonical(step['product']) and r['precursors']==pair(step['precursors'])),None))
            documented.append(dict(reference_steps=ids,all_edges_documented=all(ids)))
        # Evaluate a specific documented branch without feeding answers to search.
        stock={canonical(s) for s in request['stock']};frontier=[canonical(request['target'])];visited=set();reached=[];blocked=[]
        while frontier:
            product=frontier.pop(0)
            if product in stock or product in visited:continue
            visited.add(product);available=[]
            for ref in refs:
                if ref['product']!=product:continue
                if any(pair(c['precursors'])==ref['precursors'] for c in pools.get(product,{}).get('candidates',[])):available.append(ref)
            if not available:
                blocked.append(dict(product=product,steps=[r['step'] for r in refs if r['product']==product]));continue
            for ref in available:reached.append(ref['step']);frontier.extend(ref['precursors'])
        evaluations[job]=dict(case_name=request['case_name'],solved=result['solved'],generated_route_count=len(result['routes']),
            generated_routes=documented,documented_edges_reachable=reached,first_unresolved_documented_frontier=blocked,
            edge_union_is_diagnostic_not_an_autonomous_route=True)
        print(job,'solved',result['solved'],'routes',documented,'reached',reached,'blocked',[b['steps'] for b in blocked],flush=True)
    for p,digest in read(OUT/'protocol.json')['source_sha256'].items():assert sha(p)==digest,p
    write(OUT/'report.json',dict(state='complete',cases=evaluations,new_training=False,new_sampling=False,
        supplied_fixed_cores=True,stock_not_used_to_construct_candidates=True,
        connected_route_sources_audited=True,conditional_connectivity_test_not_product_only=True))
    lines=['# 固定骨架补全候选的路线连接验证','',
        '仅使用新权重 fixed_core_completion_20261013 的实际生成候选，复用已有30次采样，没有新增采样或训练。每个单步的已知骨架作为给定输入；指定原料清单仅用于判断何时停止，不匹配、拼接或修复候选。','',
        '| 案例 | 完整文献匹配路线已连接 | 连上的步数 | 首个缺失的文献步骤 |','|---|---|---:|---|']
    for e in evaluations.values():
        routes=[r for r in e['generated_routes'] if r['all_edges_documented']]
        blocks=[s for b in e['first_unresolved_documented_frontier'] for s in b['steps']]
        lines.append(f"| {e['case_name']} | {'是' if routes else '否'} | {len(routes[0]['reference_steps']) if routes else '—'} | {', '.join(blocks) or '无'} |")
        if routes:lines+=['',e['case_name']+' 的搜索展开顺序：'+', '.join(routes[0]['reference_steps'])+'。分支展开顺序不等于线性反应顺序。','']
    lines+=['','此结果确认候选可以在给定各步骨架的条件下连接。它不证明前级模型／LLM能够独立给出所有下一步骨架。Nemiralisib和Tafamidis的正确前体仍缺一条，搜索不能凭排序创造缺失候选。']
    (OUT/'结果.md').write_text('\n'.join(lines),encoding='utf-8')
    receipt=[dict(path=str(p.resolve()),sha256=sha(p)) for p in OUT.rglob('*') if p.is_file()]
    write(OUT/'artifact_sha256.json',receipt)
    for r in receipt:assert sha(r['path'])==r['sha256']
if __name__=='__main__':main()
