"""Assemble only already generated fragments across draws, without answer filtering."""
from pathlib import Path
from collections import Counter
import itertools
from run_last_epoch_diagnostic import ROOT,REPO
UPDATE=REPO/'experiments/case_partition_completion_20261013/fixed_core_completion_20261013/blockers_300'
source=(ROOT/'connect_fixed_core_routes_20261013.py').read_text(encoding='utf-8')
source=source.replace("OUT=PARENT/'connected_routes'", "OUT=PARENT/'connected_routes_cross_draw'")
source=source.replace("bank=read(PARENT/'decoded_bank.json');inputs=read(PARENT/'generation_inputs.json')",
    "assert read(UPDATE/'audit.json')['state']=='complete';bank=read(PARENT/'decoded_bank.json');bank.update(read(UPDATE/'combined_decoded_bank.json'));inputs=read(PARENT/'generation_inputs.json')")
source=source.replace("write(OUT/'protocol.json',dict", "files.extend([UPDATE/'audit.json',UPDATE/'protocol.json',UPDATE/'combined_decoded_bank.json',UPDATE/'new_raw_samples.pt',ROOT/'connect_fixed_core_routes_20261013.py']);write(OUT/'protocol.json',dict")
begin=source.index('    pooled={}\n');end=source.index('    pools={}\n',begin)
replacement="""    pooled={}
    for t in templates:
        product=t['product'];record=pooled.setdefault(product,dict(product=product,draw_slots=0,counts=Counter(),support={}))
        uuids=t['fragment_uuids'];counts=[Counter(r['smiles'] for r in bank[u] if r['smiles'] is not None) for u in uuids]
        slots=1
        for u in uuids:slots*=len(bank[u])
        record['draw_slots']+=slots
        combinations=1
        for c in counts:combinations*=len(c)
        assert combinations<=10000,'Independent finite cross-draw assembly bound'
        for parts in itertools.product(*[sorted(c) for c in counts]):
            candidate=key(parts)
            if product in candidate:continue
            weight=1;evidence=[]
            for u,s,c in zip(uuids,parts,counts):
                weight*=c[s]
                evidence.append(dict(uuid=u,smiles=s,draws=[r['draw'] for r in bank[u] if r['smiles']==s],
                    samples=[dict(draw=r['draw'],raw_index=r['raw_index'],raw_bank_path=r.get('raw_bank_path',str(PARENT/'raw_samples.pt')))
                        for r in bank[u] if r['smiles']==s]))
            record['counts'][candidate]+=weight
            record['support'].setdefault(candidate,[]).append(dict(input_template=t['step_id'],
                composition_method='Cartesian combinations of existing generated fragment banks',fragments=evidence))
"""
source=source[:begin]+replacement+source[end:]
source=source.replace("candidate_order='Generated paired-draw frequency, canonical tie break'", "candidate_order='Product of empirical fragment frequencies, canonical tie break',cross_draw_assembly=True,correct_precursors_not_used_to_filter_combinations=True")
source=source.replace('max_product_queries=100','max_product_queries=300').replace('max_model_calls=100','max_model_calls=300').replace('expansion_top_k=100','expansion_top_k=10000')
source=source.replace('paired_draw_frequency=count','empirical_fragment_frequency_product=count')
source=source.replace("仅使用新权重 fixed_core_completion_20261013 的实际生成候选，复用已有30次采样，没有新增采样或训练", "仅组合新权重已经生成的片段：两个卡点为300次，其他步骤30次。组合不生成新分子，不读取正确前体筛选组合，无新增采样或训练")
source=source.replace('按配对生成频次排序', '按各片段经验频次乘积排序')
source=source.replace('Nemiralisib和Tafamidis的正确前体仍缺一条，搜索不能凭排序创造缺失候选。', '每条已返回路线再次核对完整文献前体集合。此臂改变的是跨采样的前体集合组合方式；单次配对采样的原始命中计数保持不变。')
exec(compile(source,__file__,'exec'),globals())
