"""Audit complete train/eval graph masks and independently replay source features."""
import csv
import hashlib
import json
import os
from pathlib import Path
import time

import torch
from rdkit import Chem, RDLogger
from graph_model import OLD, HERE

SOURCE = Path(os.environ.get('RETRO_STAGE2_DATA_ROOT',
    HERE.parent / 'work/experiments/mechanism_prior_fspec_edm_20260919')).resolve()


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    output = HERE / 'observed_cache_audit.json'
    if output.exists():
        raise FileExistsError('Observed cache audit exists')
    torch.set_num_threads(2)
    RDLogger.DisableLog('rdApp.*')
    start = time.monotonic()
    result = {'scope':'All train/eval graph input masks, independent observable-feature reconstruction on 128 rows per split',
        'no_test_read':True, 'splits':{}, 'source_hashes':{}}
    for split, total in [('train',31325),('eval',4161)]:
        geometry_root = OLD / 'newdata' if 'RETRO_STAGE2_OLD_ROOT' in os.environ else HERE.parent / 'old/newdata'
        cache = geometry_root / (split + '_geometry.pt')
        table = SOURCE / 'data/stage2' / ('similarity_' + split + '_table.csv')
        alignment = SOURCE / 'data/stage2' / ('similarity_' + split + '_condition_alignment.json')
        before = {str(p):sha(p) for p in [cache,table,alignment]}
        rows = torch.load(cache,map_location='cpu',weights_only=False)
        assert len(rows) == total
        with table.open(encoding='utf-8-sig',newline='') as f:
            sources = {int(r['uuid']):r for r in csv.DictReader(f)}
        audits = {int(r['uuid']):r for r in json.loads(alignment.read_text(encoding='utf8'))}
        available = 0
        for r in rows:
            n = len(r['positions']); nf = int(r['fragment_mask'].sum())
            assert torch.equal(r['fragment_mask'],torch.cat([torch.ones(nf),torch.zeros(n-nf)]).to(r['fragment_mask']))
            assert r['observed_features'].shape == (n,30) and r['observed_adjacency'].shape == (n,n)
            assert torch.isfinite(r['observed_features']).all()
            assert not r['observed_features'][nf:].any()
            assert not r['observed_adjacency'][nf:].any() and not r['observed_adjacency'][:,nf:].any()
            assert torch.equal(r['observed_adjacency'],r['observed_adjacency'].T)
            assert not r['observed_adjacency'].diagonal().any()
            if r['observed_graph_available']:
                available += 1
                assert torch.equal(r['observed_features'][:nf,:26],r['one_hot'][:nf])
            else:
                assert not r['observed_features'].any() and not r['observed_adjacency'].any()
        candidates = [r for r in rows if r['observed_graph_available']]
        selected = sorted(candidates,key=lambda r: hashlib.sha256(('graph_observable_replay:' + str(r['uuid'])).encode()).hexdigest())[:128]
        checked = []
        for r in selected:
            uid = int(r['uuid']); source = sources[uid]; audit = audits[uid]
            mol = Chem.MolFromSmiles(source['fragments'],sanitize=False)
            assert mol is not None and audit['available']
            order = audit['sdf_to_condition_atom_index']
            nf = int(r['fragment_mask'].sum()); n = len(r['positions'])
            assert len(order) == nf and set(order) == set(range(nf))
            inverse = {j:i for i,j in enumerate(order)}
            features = torch.zeros(n,30)
            features[:nf,:26] = r['one_hot'][:nf]
            for i,j in enumerate(order):
                atom = mol.GetAtomWithIdx(j)
                features[i,26] = atom.GetFormalCharge()/4
                features[i,27] = float(atom.GetIsAromatic())
                features[i,28] = float(r['anchors'][i])
                features[i,29] = atom.GetDegree()/4
            adjacency = torch.zeros(n,n,dtype=torch.bool)
            for bond in mol.GetBonds():
                i,j = inverse[bond.GetBeginAtomIdx()],inverse[bond.GetEndAtomIdx()]
                adjacency[i,j] = adjacency[j,i] = True
            assert torch.equal(features,r['observed_features'])
            assert torch.equal(adjacency,r['observed_adjacency'])
            checked.append(uid)
        assert before == {str(p):sha(p) for p in [cache,table,alignment]}
        result['source_hashes'].update(before)
        result['splits'][split] = {'rows':total,'observed_graph_available':available,
            'unavailable_rows_preserved':total-available,'all_unknown_feature_and_edge_blocks_zero':True,
            'all_known_prefixes_and_element_channels_valid':True,'source_observable_replay_rows':len(checked),
            'source_observable_replay_uuids':checked,'feature_replay_exact':True,'adjacency_replay_exact':True}
        del rows,candidates,selected
    result.update(passed=True,source_hashes_unchanged=True,elapsed_seconds=time.monotonic()-start,
        audit_source_sha256=sha(Path(__file__)))
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({'passed':True,'splits':{s:{k:v for k,v in r.items() if not k.endswith('uuids')}
        for s,r in result['splits'].items()},'elapsed_seconds':result['elapsed_seconds']},indent=2),flush=True)


if __name__ == '__main__':
    main()
