"""Recount saved prior, geometry and data-distribution analyses without model execution."""
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / 'experiments/section_4_results/4_1_benchmark'
BASE = ROOT / 'results/supporting_information'


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def truth(value):
    if str(value).lower() not in ['true', 'false', '0', '1']:
        raise ValueError('Expected a saved binary observation')
    return str(value).lower() in ['true', '1']


def quantile(values, q):
    location = (len(values) - 1) * q
    lower = int(location)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (location - lower)


def main():
    expected = read_csv(BASE / 'source_data_agent/prior_ablation_recomputed.csv')
    totals, identity = {}, {}
    for arm, name in [('control', 'without'), ('compatibility', 'with')]:
        data = json.loads((BENCHMARK / f'unknown-class/results/ablations/prior_ablation_{name}_per_reaction.json').read_text(encoding='utf8'))
        # The stored comparison table names the guided arm "prior" or "compatibility";
        # resolve that label from the table rather than changing its scientific identity.
        label = 'control' if name == 'without' else next(r['arm'] for r in expected if r['arm'] != 'control')
        for scope, records in data.items():
            ids = [r['reaction_id'] for r in records]
            require(len(records) == len(set(ids)) == 2400, 'Prior comparison requires 2400 unique validation reactions')
            identity[label, scope] = set(ids)
            for ranking in ['fusion', 'frequency']:
                for row in records:
                    values = [row[f'{ranking}_top{k}'] for k in [1, 3, 5, 10]]
                    require(all(v in [0, 1] for v in values) and values == sorted(values), 'Invalid saved Top-k observations')
                for k in [1, 3, 5, 10]:
                    totals[label, scope, ranking, k] = sum(r[f'{ranking}_top{k}'] for r in records)
    arms = {r['arm'] for r in expected}
    for scope in ['stage1_gated', 'stage2_ungated_diagnostic']:
        require(len({frozenset(identity[arm, scope]) for arm in arms}) == 1, 'Prior arms use different reactions')
    for row in expected:
        hits = totals[row['arm'], row['scope'], row['ranking'], int(row['k'])]
        require(int(row['reactions']) == 2400 and int(row['hits']) == hits, 'Prior integer hit recount differs')
        require(abs(float(row['accuracy_percent']) - 100 * hits / 2400) < 1e-10, 'Prior percentage differs')

    geometry = read_csv(BASE / 'source_data_agent/geometry_comparison_per_reaction.csv')
    summary = json.loads((BASE / 'source_data_agent/geometry_comparison_summary.json').read_text(encoding='utf8'))
    require(len(geometry) == len({r['source_row'] for r in geometry}) == 826, 'Geometry pairing is incomplete')
    require(all(r['stage1_correct_2d'] == r['stage1_correct_3d'] for r in geometry), 'Geometry arms have different center gates')
    for k, saved in summary['topk'].items():
        require(saved['n'] == len(geometry), 'Geometry denominator differs')
        rates = {name: sum(truth(r[f'top{k}_{name}']) for r in geometry) / len(geometry) for name in ['2d', '3d']}
        for name, value in rates.items():
            require(abs(value - saved[name]) < 1e-12, 'Geometry Top-k recount differs')
        require(abs(rates['3d'] - rates['2d'] - saved['difference']) < 1e-12, 'Paired geometry difference differs')

    nodes = read_csv(BASE / 'source_data_figures/figure2_nodes.csv')
    edges = read_csv(BASE / 'source_data_figures/figure2_edges.csv')
    ids = {r['node_id'] for r in nodes}
    require(len(ids) == len(nodes) == 11565 and len(edges) == 11564, 'Reaction landscape counts differ')
    require(sum(int(r['sample_record_count']) for r in nodes) == 12000, 'Reaction multiplicities differ')
    parent = {i: i for i in ids}
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for edge in edges:
        u, v = edge['source_node_id'], edge['target_node_id']
        require(u in ids and v in ids, 'Unknown landscape node')
        left, right = find(u), find(v)
        require(left != right, 'Landscape tree has a repeated edge or cycle')
        parent[left] = right
    require(len({find(i) for i in ids}) == 1, 'Landscape tree is disconnected')

    descriptors = read_csv(ROOT / 'datasets/supplementary_data/figure3_reaction_weighted_descriptors.csv')
    require(len(descriptors) == 92791, 'Reaction-weighted descriptor population differs')
    for saved in read_csv(BASE / 'source_data_figures/figure3_property_summary.csv'):
        values = sorted(float(r[saved['descriptor']]) for r in descriptors)
        require(all(math.isfinite(v) for v in values) and int(saved['records']) == len(values), 'Descriptor population is incomplete')
        measured = dict(minimum=values[0], maximum=values[-1], q05=quantile(values, .05),
                        q25=quantile(values, .25), median=quantile(values, .5),
                        q75=quantile(values, .75), q95=quantile(values, .95))
        for key, value in measured.items():
            require(abs(value - float(saved[key])) < 1e-9, 'Descriptor quantile differs: ' + key)
    unique = read_csv(ROOT / 'datasets/supplementary_data/figure3_unique_product_descriptors.csv')
    require(len(unique) == len({r['product_key'] for r in unique}) == 73522, 'Unique-product population differs')
    require(sum(truth(r['fully_specified']) for r in unique) == 73521, 'Fully specified product population differs')
    result = dict(status='passed', prior_validation_reactions_per_arm=2400,
                  prior_metric_cells=len(expected), geometry_paired_reactions=826,
                  landscape_sample_records=12000, landscape_nodes=11565, landscape_tree_edges=11564,
                  reaction_weighted_descriptor_rows=92791, unique_products=73522,
                  fully_specified_unique_products=73521,
                  scope='Recounts saved chemical-prior outcomes and geometry comparisons, checks the reaction landscape and summarizes molecular descriptors.')
    destination = ROOT / 'outputs/reproduced_supporting_analyses'
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'summary.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
