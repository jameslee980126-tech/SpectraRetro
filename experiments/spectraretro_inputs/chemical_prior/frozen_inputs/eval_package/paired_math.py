import json
import numpy as np
from portable_common import *
def main():
    available=[a for a in ARMS if (ROOT/'evaluation'/a/'metrics.json').exists()]
    result={'available':available,'next':'finish missing historical evaluations' if len(available)<4 else 'freeze unknown compatibility-prior paired training protocol',
            'deployment':'none; exploratory validation evidence only','paired':{}}
    if 'control' in available:
        base=json.loads((ROOT/'evaluation/control/reaction_metrics.json').read_text())
        for arm in available[1:]:
            current=json.loads((ROOT/'evaluation'/arm/'reaction_metrics.json').read_text());result['paired'][arm]={}
            for gate in base:
                assert [r['reaction_id'] for r in base[gate]]==[r['reaction_id'] for r in current[gate]]
                result['paired'][arm][gate]={}
                for cls in [0,1,2,3,4,5,6]:
                    ids=[i for i,r in enumerate(base[gate]) if not cls or r['class_id']==cls]
                    comparison={}
                    for metric in ['fusion_top1','fusion_top3','fusion_top5','fusion_top10','pool_coverage']:
                        delta=np.array([current[gate][i][metric]-base[gate][i][metric] for i in ids])
                        rng=np.random.default_rng(20260929)
                        boot=np.array([rng.choice(delta,len(delta),replace=True).mean() for _ in range(2000)])
                        comparison[metric]={'n':len(ids),'delta_percentage_points':100*float(delta.mean()),
                            'paired_bootstrap95_pointwise_pp':(100*np.quantile(boot,[.025,.975])).tolist()}
                    result['paired'][arm][gate]['all' if not cls else f'class{cls}']=comparison
    write_json(ROOT/'analysis.json',result)
    lines=['# Stage2 内部模块真实生成开发对照','', '所有结果均为完整验证集、50 次候选；不是独立 test 或历史服务器成绩。',
           '新融合排序固定，另报频次排序。区间为探索性逐项区间，未校正多重比较。不自动发布模型。','',
           '|组|Top1|Top3|Top5|Top10|候选覆盖上限|','|---|---:|---:|---:|---:|---:|']
    for arm in available:
        m=json.loads((ROOT/'evaluation'/arm/'metrics.json').read_text())['metrics']['stage1_gated']['all']
        lines.append('|'+arm+'|'+'|'.join(f'{100*m[k]:.3f}%' for k in ['fusion_top1','fusion_top3','fusion_top5','fusion_top10','pool_coverage'])+'|')
    lines+=['','下一项：'+result['next'],'','完整分层和配对区间见 analysis.json。']
    (ROOT/'分析进度.md').write_text('\n'.join(lines),encoding='utf8')
if __name__=='__main__':main()
