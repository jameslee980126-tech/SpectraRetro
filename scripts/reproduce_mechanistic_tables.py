"""Replay electronic contrasts and spatial summaries from saved scientific observations."""
import csv
import json
import math
import statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'results/source_data/mechanistic_interpretability'


def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def require(condition,message):
    if not condition:
        raise ValueError(message)


def close(a,b,label,tolerance=1e-10):
    require(math.isfinite(float(a)) and math.isfinite(float(b)) and abs(float(a)-float(b))<=tolerance,label)


def main():
    supplement=DATA/'supplement'
    structures=read_csv(supplement/'electronic/dft_per_structure.csv')
    require(len(structures)==17,'Expected 17 molecular structures')
    by_id={r['structure_id']:r for r in structures}
    require(len(by_id)==17,'Duplicate structure identity')
    for r in structures:
        for population in ['hirshfeld','mulliken','meta_lowdin']:
            for endpoint in ['C2','C5']:
                close(float(r[f'{population}_{endpoint}_qcation'])-float(r[f'{population}_{endpoint}_q0']),
                      r[f'{population}_{endpoint}_fminus'],'Electron-removal population difference')
            for quantity,field in [('q0','q_contrast'),('fminus','fminus_contrast')]:
                close(float(r[f'{population}_C2_{quantity}'])-float(r[f'{population}_C5_{quantity}']),
                      r[f'{population}_{field}'],'Within-structure electronic contrast')
    pairs=read_csv(DATA/'main_figure_six_pairs.csv')
    table=read_csv(supplement/'electronic/Table_S26_primary_method_comparison.csv')
    require(len(pairs)==len(table)==6,'Expected six primary electronic pairs')
    table_by_id={r['parent_id']:r for r in table}
    for pair in pairs:
        changed=by_id[pair['changed_structure']]
        parent=by_id[pair['neutral_structure']]
        def delta(field):
            return float(changed[field])-float(parent[field])
        q=delta('hirshfeld_q_contrast')
        f=delta('hirshfeld_fminus_contrast')
        model=delta('model_symmetric_full_fspec_margin')
        close(q,pair['delta_q_contrast'],'Primary DFT charge contrast')
        close(f,pair['delta_fminus_contrast'],'Primary DFT electron-removal contrast')
        close(model,pair['model_symmetric_delta'],'Direction-averaged model contrast')
        t=table_by_id[pair['parent_id']]
        for value,key in [(q,'DFT_Hirshfeld_delta_Q_e'),(f,'DFT_Hirshfeld_delta_F'),(model,'symmetric_model_delta_S_logit'),
                          (delta('xtb_q_contrast'),'GFN2_xTB_delta_Q_e'),(delta('xtb_fminus_contrast'),'GFN2_xTB_delta_F')]:
            close(value,t[key],'Table S26 differs')
    states=read_csv(supplement/'quantum/dft_34_state_summary.csv')
    require(len(states)==34 and len({(r['structure_id'],r['state']) for r in states})==34,'Electronic states differ')
    for row in states:
        saved=json.loads((supplement/f"quantum/dft/{row['structure_id']}/{row['state']}/result.json").read_text(encoding='utf8'))
        close(row['energy_hartree'],saved['energy_hartree'],'SCF state energy differs')
        close(row['expected_electrons'],saved['nelectron'],'Electron count differs')
        require(row['geometry_sha256']==saved['geometry_sha256'],'Geometry identity differs')
        require(saved['converged'] and saved['passed'],'Incomplete stored quantum state')
    spatial=read_csv(supplement/'spatial/reference/steric_test_fragment_replay.csv')
    summaries=read_csv(supplement/'spatial/spatial_noise_statistics.csv')
    require(len(spatial)==114 and len(summaries)==6,'Spatial table population differs')
    for summary in summaries:
        records=[r for r in spatial if float(r['t'])==float(summary['t'])]
        require(len(records)==38,'Spatial time has incomplete fragments')
        require(len({r['known_graph_key'] for r in records})==int(summary['scaffold_clusters'])==22,'Cluster count differs')
        field='original_R_deg' if summary['centering']=='Native' else 'common_center_R_deg'
        values=[float(r[field]) for r in records]
        close(statistics.mean(values),summary['mean_R_deg'],'Spatial mean differs')
        close(statistics.median(values),summary['median_R_deg'],'Spatial median differs')
        require(sum(v>0 for v in values)==int(summary['positive_fragments']),'Positive count differs')
        require(sum(v<0 for v in values)==int(summary['negative_fragments']),'Negative count differs')
    preferences=read_csv(supplement/'chemical/chemical_preferences.csv')
    correlations=read_csv(supplement/'chemical/descriptor_correlations.csv')
    require(len(preferences)==114 and len(correlations)==120,'Chemical source-table population differs')
    result=dict(status='passed',structures=17,electronic_states=34,primary_pairs=6,spatial_fragments=38,
                spatial_clusters=22,spatial_time_centering_summaries=6,chemical_slots=114,descriptor_pairs=120,
                scope='Saved atomic-population arithmetic, structure/pair identities, SCF state-table binding and spatial means/medians/sign counts. No DFT, model inference or bootstrap recomputation.')
    out=ROOT/'outputs/reproduced_mechanistic_tables'
    out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result))


if __name__=='__main__':
    main()
