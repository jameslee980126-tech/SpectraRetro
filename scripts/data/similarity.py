"""Fixed rooted 2-hop WL count similarity; never uses learned model features."""
from collections import Counter
import hashlib
import json
import networkx as nx
from rdkit import Chem

def token(value):
    return hashlib.sha256(repr(value).encode()).hexdigest()

def molecular_graph(mol):
    g = nx.Graph()
    for atom in mol.GetAtoms():
        g.add_node(atom.GetIdx(), chemical=(atom.GetAtomicNum(), atom.GetFormalCharge(),
            atom.GetIsotope(), int(atom.GetIsAromatic()), atom.GetTotalNumHs()))
    for bond in mol.GetBonds():
        g.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(),
            chemical=(str(bond.GetBondType()), int(bond.GetIsAromatic())))
    return g

def local_environment(graph, bond, radius=2):
    u, v = bond
    keep = set(nx.single_source_shortest_path_length(graph, u, cutoff=radius))
    keep.update(nx.single_source_shortest_path_length(graph, v, cutoff=radius))
    g = graph.subgraph(keep).copy()
    for n in g:
        g.nodes[n]['label'] = (g.nodes[n]['chemical'], int(n in bond))
    for a, b in g.edges:
        g.edges[a,b]['label'] = (g.edges[a,b]['chemical'], int({a,b} == {u,v}))
    labels = {n:token(g.nodes[n]['label']) for n in g}
    fingerprint = Counter((0, label) for label in labels.values())
    for iteration in (1, 2):
        labels = {n:token((labels[n], tuple(sorted((g.edges[n,m]['label'],labels[m]) for m in g[n])))) for n in g}
        fingerprint.update((iteration,label) for label in labels.values())
    return g, fingerprint

def similarity(a, b):
    keys = a.keys() | b.keys()
    return sum(min(a[k], b[k]) for k in keys) / sum(max(a[k], b[k]) for k in keys)

def rooted_product_key(mol, bond):
    # Artificial atom-map tags mark both endpoints equally; real map IDs erased.
    # Canonical isomeric SMILES retains chemical stereo in whole-product symmetry.
    copy = Chem.Mol(mol)
    for atom in copy.GetAtoms():
        atom.SetAtomMapNum(1 if atom.GetIdx() in bond else 0)
    return Chem.MolToSmiles(copy, canonical=True, isomericSmiles=True)

def describe(payload):
    mol = Chem.MolFromSmiles(payload['product_smiles'])
    if mol is None:
        raise ValueError('Cannot parse product: ' + payload['sample_id'])
    map_to_atom = {a.GetAtomMapNum():a.GetIdx() for a in mol.GetAtoms()}
    candidates = [tuple(map_to_atom[x] for x in endpoints) for endpoints in payload['bond_atom_maps']]
    graph = molecular_graph(mol)
    output = {k:v for k,v in payload.items() if k not in ('product_smiles', 'bond_atom_maps')}
    output['heavy_atoms'] = mol.GetNumHeavyAtoms()
    gold = payload['gold_candidate']
    if gold >= len(candidates):
        output.update(analysis_status='atom_center', radii={})
        return output
    refkey = rooted_product_key(mol, candidates[gold])
    equivalents = [i for i,b in enumerate(candidates) if i != gold and rooted_product_key(mol,b)==refkey]
    competitors = [i for i in range(len(candidates)) if i != gold and i not in equivalents]
    output['symmetry_equivalent_candidates'] = equivalents
    output['analysis_status'] = 'eligible' if competitors else 'no_non_equivalent_competitor'
    output['radii'] = {}
    for radius in (1,2,3):
        local = [local_environment(graph, b, radius) for b in candidates]
        reference, f = local[gold]
        scores = [similarity(f, fp) for _,fp in local]
        exact = [i for i in competitors if scores[i] >= 1-1e-12 and nx.is_isomorphic(reference, local[i][0],
            node_match=lambda a,b:a['label']==b['label'], edge_match=lambda a,b:a['label']==b['label'])]
        output['radii'][str(radius)] = {'difficulty':max((scores[i] for i in competitors),default=None),
            'similarity_to_gold':scores, 'competitors':competitors, 'exact_local_competitors':exact}
    return output

def calibrate(records):
    import numpy as np
    rules = {}
    for radius in ('1','2','3'):
        eligible = [r['radii'][radius]['difficulty'] for r in records if r['analysis_status']=='eligible']
        if not eligible:
            raise RuntimeError('No eligible validation samples')
        low, high = np.quantile(eligible, [1/3,2/3]).tolist()
        if low == high:
            # Ties are never randomly split. Prespecified fallback preserves all values.
            low, high = low, high
            mode = 'below_equal_above'
        else:
            mode = 'tertiles'
        rules[radius] = {'low':low,'high':high,'mode':mode,'n_valid':len(eligible),
            'equality_fallback_note':'if cutoffs coincide, groups are <, ==, > cutoff; empty groups reported'}
    return rules

def group_for(record, rule, radius='2'):
    if record['analysis_status'] != 'eligible':
        return record['analysis_status']
    score = record['radii'][radius]['difficulty']
    if rule['mode']=='below_equal_above':
        return 'low' if score < rule['low'] else 'high' if score > rule['high'] else 'medium'
    return 'low' if score < rule['low'] else 'medium' if score < rule['high'] else 'high'
