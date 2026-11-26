"""Outcome-blind distribution report for the frozen sample."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
ROOT=Path(__file__).resolve().parent
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    manifest=load(ROOT/'frozen/manifest.json')
    rule=load(ROOT/'frozen/thresholds.json')['rule']
    result={}
    for s in ('train','valid','test'):
        source=load(Path(manifest['source_provenance'][s]['metadata']))
        chosen=load(ROOT/f'frozen/{s}_design.json')
        result[s]={}
        for name,rows in [('parent_original',source),('selected',chosen)]:
            profiles={}
            for g in ('low','medium','high'):
                rs=[]
                for r in rows:
                    d=r.get('radii',{}).get('2',{}).get('difficulty')
                    if d is None:continue
                    group='low' if d<rule['low'] else 'medium' if d<rule['high'] else 'high'
                    if group==g:rs.append(r)
                profiles[g]={'n':len(rs),'similarity_quantiles_0_25_50_75_100':np.quantile(
                    [r['radii']['2']['difficulty'] for r in rs],[0,.25,.5,.75,1]).tolist(),
                    'symmetry_flag_rows':sum(r.get('symmetry_count',0)>0 for r in rs),
                    'near_competitor_counts':dict(Counter(min(2,sum(v>=rule['high'] for v in r['radii']['2']['competitor_similarities'])) for r in rs))}
            result[s][name]=profiles
    (ROOT/'audit/similarity_distribution.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('DISTRIBUTION_REPORT_SAVED')
if __name__=='__main__':main()
