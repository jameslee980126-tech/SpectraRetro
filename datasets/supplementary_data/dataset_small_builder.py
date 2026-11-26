"""Outcome-blind grouped sampling; no model predictions or checkpoints are read."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'stage1_local_similarity20_20260911'
SEED = 20260912
SPLITS = ('train', 'valid', 'test')
GROUPS = ('low', 'medium', 'high')

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()

def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def stable(key):
    return int(hashlib.sha256(f'{SEED}|{key}'.encode()).hexdigest()[:13], 16) / 16**13

def cell(r):
    a, n = r['heavy_atoms'], r['candidate_count']
    return (r['class'], 0 if a <= 15 else 1 if a <= 20 else 2 if a <= 30 else 3,
            0 if n <= 30 else 1 if n <= 45 else 2 if n <= 65 else 3)

def load_pool():
    manifest = json.loads((OLD / 'data_manifest.json').read_text())
    rule = json.loads((OLD / 'purpose_audit/primary_threshold.json').read_text())
    sources, pools = {}, {}
    provenance = {}
    for s in SPLITS:
        p = OLD / f'parent_data/{s}.csv'
        assert digest(p) == manifest['source_sha256'][s], s
        sources[s] = list(csv.DictReader(p.open(encoding='utf-8-sig')))
        q = OLD / f'purpose_audit/{s}_population.json'
        pools[s] = json.loads(q.read_text())
        assert len(sources[s]) == len(pools[s])
        provenance[s] = {'csv': str(p), 'csv_sha256': digest(p), 'metadata': str(q),
                         'metadata_sha256': digest(q), 'rows': len(pools[s])}
        for i, r in enumerate(pools[s]):
            assert r['source_row'] == i and r['class'] == sources[s][i]['class']
            d = r.get('radii', {}).get('2', {}).get('difficulty')
            r['similarity_group'] = ('not_eligible' if d is None else
                'low' if d < rule['low'] else 'medium' if d < rule['high'] else 'high')
    keys = {s: {r['product_group'] for r in pools[s]} for s in SPLITS}
    forbidden = {'train': keys['valid'] | keys['test'], 'valid': keys['test'], 'test': set()}
    excluded = []
    for s in SPLITS:
        kept = []
        for r in pools[s]:
            if r['product_group'] in forbidden[s]:
                excluded.append({'split': s, 'sample_id': r['sample_id'],
                                 'product_group': r['product_group'], 'reason': 'product_in_later_original_split'})
            else:
                kept.append(r)
        pools[s] = kept
    return sources, pools, rule, provenance, excluded

def audit(pools):
    out = {}
    for s, rows in pools.items():
        counts = {g: Counter(cell(r) for r in rows if r['similarity_group'] == g) for g in GROUPS}
        cells = sorted(set().union(*(set(c) for c in counts.values())))
        common = {c: min(counts[g][c] for g in GROUPS) for c in cells}
        out[s] = {'rows': len(rows), 'products': len({r['product_group'] for r in rows}),
                  'groups': dict(Counter(r['similarity_group'] for r in rows)),
                  'class_by_group': {g: dict(Counter(r['class'] for r in rows if r['similarity_group'] == g)) for g in GROUPS},
                  'common_cell_capacity_per_group': sum(common.values()),
                  'common_capacity_by_class': {c: sum(n for k, n in common.items() if k[0] == c) for c in map(str, range(1,7))},
                  'cells': [{'cell': list(c), 'counts': {g: counts[g][c] for g in GROUPS}} for c in cells]}
    return out

def apportion(weights, total, capacities=None):
    keys = sorted(weights)
    result = dict.fromkeys(keys, 0)
    capacities = capacities or {k: total for k in keys}
    for _ in range(total):
        available = [k for k in keys if result[k] < capacities[k] and weights[k] > 0]
        if not available:
            raise RuntimeError('Insufficient structural capacity')
        k = max(available, key=lambda k: (weights[k] / (result[k]+1), str(k)))
        result[k] += 1
    return result

def target_quotas(pools, per_group=800):
    counts = {s: {g: Counter(cell(r) for r in pools[s] if r['similarity_group'] == g)
                   for g in GROUPS} for s in ('valid','test')}
    cells = sorted(set().union(*(set(z) for m in counts.values() for z in m.values())))
    # Target weights from validation only; test metadata merely checks feasibility.
    weights = {c: min(counts['valid'][g][c] for g in GROUPS) for c in cells}
    caps = {c: min(counts[s][g][c] for s in ('valid','test') for g in GROUPS) for c in cells}
    class_weights = {c: sum(w for k,w in weights.items() if k[0] == c) for c in map(str,range(1,7))}
    extras = apportion(class_weights, per_group-6*60)
    class_quota = {c: 60+extras[c] for c in class_weights}
    quotas = {}
    for c,n in class_quota.items():
        quotas.update(apportion({k:w for k,w in weights.items() if k[0] == c}, n, caps))
    quotas = {k:v for k,v in quotas.items() if v > 0}
    return quotas, class_quota

def matched_select(rows, quotas, split):
    from scipy.optimize import milp, Bounds, LinearConstraint
    from scipy.sparse import lil_matrix, csr_matrix
    groups = defaultdict(list)
    for r in rows:
        groups[r['product_group']].append(r)
    candidates = [(k,rs) for k,rs in sorted(groups.items())
                  if all(r['similarity_group'] in GROUPS and cell(r) in quotas for r in rs)]
    labels = [(g,c) for g in GROUPS for c in sorted(quotas)]
    lookup = {v:i for i,v in enumerate(labels)}
    # Exact joint class x atom-size-bin x candidate-count-bin quotas in each similarity group.
    a = lil_matrix((len(labels)+4, len(candidates)), dtype=float)
    for j,(k,rs) in enumerate(candidates):
        for r in rs:
            g = r['similarity_group']
            a[lookup[g,cell(r)],j] += 1
            for t,other in enumerate(('medium','high')):
                sign = (1 if g == other else -1 if g == 'low' else 0)
                a[len(labels)+t*2,j] += sign*r['heavy_atoms']
                a[len(labels)+t*2+1,j] += sign*r['candidate_count']
    n = sum(quotas.values())
    expected = np.array([quotas[c] for g,c in labels], float)
    tolerance = np.array([.25*n,.5*n,.25*n,.5*n])
    lo = np.concatenate([expected,-tolerance])
    hi = np.concatenate([expected,tolerance])
    objective = np.array([len(rs)*(.1+stable(split+'|'+k)) for k,rs in candidates])
    print('MATCH_SOLVE', split, len(candidates), len(labels), flush=True)
    fit = milp(objective, integrality=np.ones(len(candidates)), bounds=Bounds(0,1),
               constraints=LinearConstraint(csr_matrix(a),lo,hi),
               options={'time_limit':180, 'mip_rel_gap':.01})
    if fit.x is None:
        raise RuntimeError(f'{split} matching infeasible: {fit.message}')
    decision = np.rint(fit.x)
    observed = a.tocsr()@decision
    assert np.all(observed >= lo-1e-6) and np.all(observed <= hi+1e-6), split
    chosen = [r for j,(k,rs) in enumerate(candidates) if decision[j] > .5 for r in rs]
    save(ROOT/f'audit/{split}_solver.json', {'status':int(fit.status),'message':str(fit.message),
         'gap':float(fit.mip_gap), 'feasible':True, 'rows':len(chosen),'candidate_products':len(candidates),
         'excluded_outside_joint_support_products':len(groups)-len(candidates)})
    return sorted(chosen,key=lambda r:r['source_row'])

def train_select(rows, target=18400):
    groups = defaultdict(list)
    for r in rows:
        groups[r['product_group']].append(r)
    strata = defaultdict(list)
    for k,rs in groups.items():
        signature = tuple(sorted({(r['class'],r['similarity_group'],cell(r)[1],cell(r)[2]) for r in rs}))
        strata[signature].append((k,rs))
    selected=[]
    for signature, items in sorted(strata.items()):
        items.sort(key=lambda item:stable('train|'+item[0]))
        goal=sum(len(rs) for _,rs in items)*target/len(rows)
        n=0
        for k,rs in items:
            if abs(n+len(rs)-goal) < abs(n-goal):
                selected.extend(rs)
                n+=len(rs)
    return sorted(selected,key=lambda r:r['source_row'])

def csv_write(path, rows, fields):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        w.writeheader(); w.writerows(rows)

def summarize(rows):
    result={'rows':len(rows),'products':len({r['product_group'] for r in rows}),
            'class_counts':dict(Counter(r['class'] for r in rows)),'groups':{}}
    for g in GROUPS:
        rs=[r for r in rows if r['similarity_group']==g]
        result['groups'][g]={'n':len(rs),'products':len({r['product_group'] for r in rs}),
            'classes':dict(Counter(r['class'] for r in rs)),
            'mean_atoms':float(np.mean([r['heavy_atoms'] for r in rs])),
            'median_atoms':float(np.median([r['heavy_atoms'] for r in rs])),
            'mean_candidates':float(np.mean([r['candidate_count'] for r in rs])),
            'median_candidates':float(np.median([r['candidate_count'] for r in rs])),
            'joint_cells':{'|'.join(map(str,k)):v for k,v in sorted(Counter(cell(r) for r in rs).items())}}
    return result

def build(sources,pools,rule,provenance):
    if (ROOT/'frozen').exists():
        raise RuntimeError('Frozen output already exists; no overwrite')
    quotas,class_quota=target_quotas(pools)
    save(ROOT/'selection_plan.json', {'seed':SEED,'uses_model_outputs':False,
         'target_train':18400,'target_valid':2400,'target_test':2400,
         'class_quota_per_similarity_group':class_quota,
         'joint_quotas_per_group':{'|'.join(map(str,k)):v for k,v in quotas.items()},
         'threshold':rule,'strict_constraints':['complete product groups','original split boundaries',
         'same joint class/atom-size/candidate-bin counts across similarity groups',
         'mean atoms vs low within 0.25','mean candidates vs low within 0.5'],
         'selection_objective':'fixed hash preference under structural constraints; no prediction access',
         'budget_note':'one new approximately 25% version; old version remains archived',
         'interpretation':'balanced common-support diagnostic population, not original distribution accuracy'})
    selected={'train':train_select(pools['train'])}
    for s in ('valid','test'):
        selected[s]=matched_select(pools[s],quotas,s)
    assert .20 < sum(map(len,selected.values()))/92791 < .30
    for s,rs in selected.items():
        csv_write(ROOT/f'frozen/{s}.csv',[sources[s][r['source_row']] for r in rs],
                  ['class','rxn_smiles','original_class'])
        csv_write(ROOT/f'frozen/{s}_index.csv',rs,
            ['sample_id','source_row','product_group','class','original_class','similarity_group','heavy_atoms','candidate_count'])
        save(ROOT/f'frozen/{s}_design.json',rs)
    save(ROOT/'frozen/thresholds.json',{'radius':2,'rule':rule,
         'definition':'unchanged rooted 2-hop WL-count similarity; original validation calibration',
         'calibration_sha256':digest(OLD/'purpose_audit/primary_threshold.json')})
    summary={s:summarize(rs) for s,rs in selected.items()}
    save(ROOT/'audit/selected_summary.json',summary)
    overlap={}
    for s,rs in selected.items():
        old=list(csv.DictReader((OLD/f'data/{s}_index.csv').open(encoding='utf-8-sig')))
        old_ids={r['sample_id'] for r in old}; old_groups={r['product_group'] for r in old}
        overlap[s]={'old_rows':len(old),'new_rows':len(rs),
            'overlap_rows':sum(r['sample_id'] in old_ids for r in rs),
            'overlap_products':len({r['product_group'] for r in rs}&old_groups),
            'novel_to_previous_subset_rows':sum(r['sample_id'] not in old_ids for r in rs),
            'not_an_unseen_external_test':True}
    save(ROOT/'audit/old_overlap.json',overlap)
    save(ROOT/'frozen/manifest.json',{'protocol':'stage1_similarity_balanced_common_support_v4',
        'seed':SEED,'unknown_class':True,'dual_branch':False,
        'comparators':['current FSpec product-only','original ordinary product RGCN without dual'],
        'counts':{s:len(rs) for s,rs in selected.items()},
        'total_rows':sum(map(len,selected.values())),
        'source_total_rows':92791,'fraction':sum(map(len,selected.values()))/92791,
        'source_provenance':provenance,'selection_plan_sha256':digest(ROOT/'selection_plan.json'),
        'build_script_sha256':digest(Path(__file__)),'uses_model_predictions':False,
        'historical_test_seen':True,'old_overlap':overlap,
        'purpose':'balanced diagnostic; not an IID estimate of parent distribution',
        'files':{p.relative_to(ROOT/'frozen').as_posix():digest(p) for p in sorted((ROOT/'frozen').glob('*')) if p.is_file()}})
    for s,v in summary.items():
        print('FROZEN',s,v['rows'],{g:z['classes'] for g,z in v['groups'].items()},flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit', action='store_true')
    args = parser.parse_args()
    sources, pools, rule, provenance, excluded = load_pool()
    report = audit(pools)
    save(ROOT / 'population_audit.json', report)
    save(ROOT / 'source_provenance.json', provenance)
    save(ROOT / 'excluded_original_overlap.json', excluded)
    for s in SPLITS:
        print(s, json.dumps({k: v for k,v in report[s].items() if k != 'cells'}), flush=True)
    if not args.audit:
        build(sources,pools,rule,provenance)

if __name__ == '__main__':
    main()
