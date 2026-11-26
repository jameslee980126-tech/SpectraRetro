#!/usr/bin/env python3
"""Summarize archived DFT results without changing raw calculations.

Standard library only. Outputs are derived tables, with all planned structures
retained even when a queue is incomplete. Sign counts never gate acceptance.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

NATIVE_PARTITIONS = ('meta_lowdin', 'mulliken')
PARTITIONS = NATIVE_PARTITIONS + ('hirshfeld',)
TOL = 1e-6
HARTREE_EV = 27.211386245988


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def atomic_json(path, obj):
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def atomic_csv(path, rows, columns):
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='raise')
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def verify_state(state, label, item):
    charge, spin, multiplicity = (0, 0, 1) if label == 'neutral' else (1, 1, 2)
    n = item['total_atoms']
    require(state['state'] == label, f'{label}: state label')
    require(state['charge'] == charge and state['pyscf_spin_nalpha_minus_nbeta'] == spin and state['multiplicity_input'] == multiplicity,
            f'{label}: electronic state mismatch')
    require(state['natoms'] == n and state['geometry_sha256'] == item['xyz_sha256'], f'{label}: coordinate binding')
    require(state['converged'] is True and state['passed'] is True, f'{label}: worker failed checks')
    required_checks = ('scf_converged', 'finite_energy_and_density', 'electron_count', 'mulliken_charge_sum',
                       'meta_lowdin_charge_sum', 'spin_electron_count', 'internal_stability')
    require(all(state['checks'].get(x) is True for x in required_checks), f'{label}: quality check flags')
    require(state['stability']['tested'] is True and state['stability']['internal_stable'] is True,
            f'{label}: internal stability absent or failed')
    expected_electrons = item['neutral_electron_count'] - charge
    require(state['nelectron'] == expected_electrons and finite(state['integrated_electrons_ao']) and
            abs(state['integrated_electrons_ao'] - expected_electrons) < TOL, f'{label}: electron count')
    require(sum(state['nalpha_nbeta']) == expected_electrons and state['nalpha_nbeta'][0] - state['nalpha_nbeta'][1] == spin,
            f'{label}: spin electron counts')
    require(finite(state['energy_hartree']) and finite(state['spin_square']), f'{label}: nonfinite energy/spin')
    for partition in NATIVE_PARTITIONS:
        charges = state[partition + '_charges']
        require(len(charges) == n and all(finite(x) for x in charges), f'{label}: {partition} array length/finite')
        require(abs(sum(charges) - charge) < TOL, f'{label}: {partition} charge sum')


def verify_result(result, item, run_dir, expected_protocol):
    require(result['status'] == 'complete' and result['passed'] is True, 'combined result incomplete/failed')
    require(result['geometry_sha256'] == item['xyz_sha256'], 'combined XYZ binding')
    require(result['atom_count'] == item['total_atoms'], 'combined atom count')
    require(result['fixed_geometry'] is True and result['electronic_singlepoints'] == 2, 'fixed-geometry state count')
    require(set(result['states']) == {'neutral', 'cation'}, 'state labels')
    for label in ('neutral', 'cation'):
        verify_state(result['states'][label], label, item)
    for partition in NATIVE_PARTITIONS:
        q0 = result['states']['neutral'][partition + '_charges']
        q1 = result['states']['cation'][partition + '_charges']
        fminus = result['f_minus'][partition]['values']
        require(len(fminus) == item['total_atoms'] and all(finite(x) for x in fminus), f'{partition}: fminus array')
        require(max(abs(c - a - f) for c, a, f in zip(q1, q0, fminus)) < 1e-10, f'{partition}: fminus replay')
        require(abs(sum(fminus) - 1.0) < TOL, f'{partition}: fminus sum')
    ip = (result['states']['cation']['energy_hartree'] - result['states']['neutral']['energy_hartree']) * HARTREE_EV
    require(finite(result['vertical_ionization_energy_ev']) and abs(ip - result['vertical_ionization_energy_ev']) < 1e-8,
            'vertical ionization energy replay')
    run_protocol = read(run_dir / 'protocol.json')
    require(run_protocol['input_sha256'] == item['xyz_sha256'], 'run protocol XYZ binding')
    if expected_protocol is not None:
        fields = {'functional': 'functional', 'orbital_basis': 'basis', 'auxiliary_basis': 'auxiliary_basis',
                  'grid_level': 'grid_level', 'scf_conv_tol_hartree': 'conv_tol_hartree',
                  'scf_conv_tol_grad': 'conv_tol_grad'}
        for output_field, input_field in fields.items():
            require(run_protocol[output_field] == expected_protocol[input_field], f'protocol mismatch: {output_field}')
    require(run_protocol['internal_stability_requested'] is True, 'run protocol internal stability not requested')
    require(run_protocol['solvent'] == 'none (gas phase)', 'run protocol solvent changed')
    require(run_protocol['dispersion'] == 'none', 'run protocol dispersion changed')
    require(run_protocol['populations'] == ['Mulliken', 'meta-Lowdin (ANO pre-orthogonalization)'], 'population definitions changed')
    return run_protocol


def verify_hirshfeld(result, item, run_dir, out_dir, root):
    """Replay native Hirshfeld acceptance and bind it to immutable molecular files."""
    require(result['status'] == 'complete' and result['passed'] is True, 'Hirshfeld result incomplete/failed')
    require(result['geometry_sha256'] == item['xyz_sha256'], 'Hirshfeld geometry binding')
    require(result['atom_elements'] == [a['element'] for a in item['atom_mapping']], 'Hirshfeld atom order')
    required_checks = ('source_molecular_results_verified', 'all_grids_conserve_electrons', 'grid_refinement_passed')
    require(all(result['checks'].get(x) is True for x in required_checks), 'Hirshfeld check flags')
    artifact_manifest = read(out_dir / 'artifact_manifest.json')
    protocol = read(out_dir / 'protocol.json')
    source_binding = read(out_dir / 'source_binding.json')
    required_sources = ['result.json', 'protocol.json', 'atoms.json', 'artifact_manifest.json']
    for label in ('neutral', 'cation'):
        required_sources.extend([f'{label}/molecule.json', f'{label}/wavefunction.npz', f'{label}/result.json'])
    require(set(required_sources) == set(source_binding), 'Hirshfeld required source-binding members')
    hashes = {}
    source_manifest = read(run_dir / 'artifact_manifest.json')
    for relative, expected in source_binding.items():
        require(relative in required_sources, 'Unsafe/unrecognized source binding path')
        path = run_dir / relative
        actual = sha(path)
        require(actual == expected, f'Hirshfeld source-binding SHA mismatch: {relative}')
        if relative != 'artifact_manifest.json':
            require(source_manifest[relative]['sha256'] == actual, f'Hirshfeld molecular artifact SHA mismatch: {relative}')
        hashes[str(path.relative_to(root)).replace('\\', '/')] = actual
    levels = protocol['grid_levels']
    require(len(levels) >= 2 and levels == sorted(set(levels)) and all(isinstance(x, int) and 0 <= x <= 9 for x in levels),
            'Hirshfeld grid levels')
    require(result['selected_grid_level'] == levels[-1] == protocol['selected_grid'], 'Hirshfeld selected grid')
    require(protocol['same_grid_for_neutral_and_cation'] is True and protocol['molecular_scf_recomputed'] is False,
            'Hirshfeld fixed-density protocol')
    require(protocol['native_api'] == 'pyscf.scf.hirshfeld.hirshfeld_charges(mol, dm, grids, verbose)', 'Hirshfeld native API')
    require(protocol['reference_is_dft'] is False, 'Hirshfeld atomic-reference declaration')
    tolerances = {key: protocol[key] for key in ('charge_sum_tolerance', 'grid_charge_tolerance', 'grid_fukui_tolerance')}
    require(all(finite(v) and 0 < v <= 1e-3 for v in tolerances.values()), 'Hirshfeld numerical tolerances')
    charge_tol = tolerances['charge_sum_tolerance']
    required_artifacts = ['result.json', 'protocol.json', 'source_binding.json'] + [f'grid_{level}.json' for level in levels]
    for relative in required_artifacts:
        path = out_dir / relative
        actual = sha(path)
        require(actual == artifact_manifest[relative]['sha256'], f'Hirshfeld artifact SHA mismatch: {relative}')
        require(path.stat().st_size == artifact_manifest[relative]['size_bytes'], f'Hirshfeld artifact size mismatch: {relative}')
        hashes[str(path.relative_to(root)).replace('\\', '/')] = actual
    hashes[str((out_dir / 'artifact_manifest.json').relative_to(root)).replace('\\', '/')] = sha(out_dir / 'artifact_manifest.json')
    grids = []
    for level in levels:
        grid = read(out_dir / f'grid_{level}.json')
        require(grid['grid_level'] == level, 'Hirshfeld grid identity')
        for label, charge in (('neutral', 0), ('cation', 1)):
            state = grid['states'][label]
            values = state['charges']
            require(len(values) == item['total_atoms'] and all(finite(v) for v in values), 'Hirshfeld charge arrays')
            require(state['passed'] is True and all(state['checks'].get(k) is True for k in
                    ('finite_charges_and_density', 'charge_conservation', 'numerical_electron_conservation', 'partition_matches_total_density')),
                    'Hirshfeld per-grid charge acceptance')
            require(abs(sum(values) - charge) <= charge_tol, f'Hirshfeld {label} charge conservation')
            require(abs(sum(values) - state['sum_charges']) < 1e-10, 'Hirshfeld saved charge sum')
            expected_electrons = item['neutral_electron_count'] - charge
            require(state['expected_electrons'] == expected_electrons and finite(state['integrated_electrons_grid']) and
                    abs(state['integrated_electrons_grid'] - expected_electrons) <= charge_tol, 'Hirshfeld grid electron conservation')
            require(abs(item['neutral_electron_count'] - sum(values) - state['integrated_electrons_grid']) <= charge_tol,
                    'Hirshfeld population/total-density conservation')
        values = grid['f_minus']
        require(len(values) == item['total_atoms'] and all(finite(v) for v in values), 'Hirshfeld fminus array')
        require(max(abs(c - n - f) for c, n, f in zip(grid['states']['cation']['charges'], grid['states']['neutral']['charges'], values)) < 1e-10,
                'Hirshfeld fminus charge-difference replay')
        require(abs(sum(values) - 1.0) <= 2 * charge_tol and grid['f_minus_sum_passed'] is True and
                abs(sum(values) - grid['f_minus_sum']) < 1e-10, 'Hirshfeld fminus conservation')
        grids.append(grid)
    require(len(result['grid_comparisons']) == len(grids) - 1, 'Hirshfeld grid comparison count')
    for previous, current, comparison in zip(grids[:-1], grids[1:], result['grid_comparisons']):
        require(comparison['coarser_grid'] == previous['grid_level'] and comparison['finer_grid'] == current['grid_level'],
                'Hirshfeld grid comparison identity')
        for label in ('neutral', 'cation'):
            change = max(abs(a - b) for a, b in zip(current['states'][label]['charges'], previous['states'][label]['charges']))
            require(change <= tolerances['grid_charge_tolerance'] and abs(change - comparison['max_charge_change_e'][label]) < 1e-10,
                    'Hirshfeld charge refinement replay')
        change = max(abs(a - b) for a, b in zip(current['f_minus'], previous['f_minus']))
        require(change <= tolerances['grid_fukui_tolerance'] and abs(change - comparison['max_f_minus_change']) < 1e-10 and
                comparison['passed'] is True, 'Hirshfeld fminus refinement replay')
    selected = grids[-1]
    for label in ('neutral', 'cation'):
        require(result[label + '_hirshfeld_charges'] == selected['states'][label]['charges'], 'Hirshfeld selected charge array')
    require(result['f_minus'] == selected['f_minus'] and abs(result['f_minus_sum'] - selected['f_minus_sum']) < 1e-10,
            'Hirshfeld selected fminus array')
    return hashes, {key: protocol[key] for key in ('pyscf_version', 'native_module_sha256', 'grid_levels', 'selected_grid',
                                                  'reference', 'charge_sum_tolerance', 'grid_charge_tolerance', 'grid_fukui_tolerance')}


def numeric_delta(a, b):
    return a - b if a is not None and b is not None else None


def model_value(item, kind):
    model = item['frozen_model']
    if kind == 'original':
        return model['primary_original_full_fspec_margin']
    return model['symmetric_readouts']['full_fspec']['margin']


def compare_signs(rows, descriptor, model):
    usable = [row for row in rows if finite(row.get(descriptor)) and finite(row.get(model))]
    products = [row[descriptor] * row[model] for row in usable]
    return {'available_pairs': len(usable), 'same_sign': sum(x > 0 for x in products),
            'opposite_sign': sum(x < 0 for x in products), 'zero_product': sum(x == 0 for x in products)}


def descriptive(values):
    values = [x for x in values if finite(x)]
    return {'n_available': len(values), 'positive': sum(x > 0 for x in values),
            'negative': sum(x < 0 for x in values), 'zero': sum(x == 0 for x in values),
            'mean': statistics.mean(values) if values else None,
            'median': statistics.median(values) if values else None,
            'min': min(values) if values else None, 'max': max(values) if values else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument('--output-dir', type=Path, help='Default: ROOT/analysis')
    ap.add_argument('--require-complete', action='store_true', help='Write current tables, then exit 2 unless 17/17 and 34/34 pass')
    ap.add_argument('--require-hirshfeld', action='store_true', help='Also require 17/17 independently verified Hirshfeld postprocessings')
    args = ap.parse_args()
    root = args.root.resolve()
    out = (args.output_dir or root / 'analysis').resolve()
    require(out != root / 'runs' and root / 'runs' not in out.parents, 'Derived output must be outside raw run directories')
    require(out != root / 'hirshfeld' and root / 'hirshfeld' not in out.parents, 'Derived output must be outside raw Hirshfeld directories')
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = root / 'input_audit/structure_manifest.json'
    manifest = read(manifest_path)
    items = manifest['structures']
    by_id = {x['structure_id']: x for x in items}
    require(len(items) == len(by_id) == 17, 'Expected exactly 17 unique structures')
    require(len({x['parent_id'] for x in items}) == 6, 'Expected six independent parent structures')
    protocol_path = root / 'protocol.json'
    expected_protocol = read(protocol_path) if protocol_path.exists() else None
    input_hashes = {'input_audit/structure_manifest.json': sha(manifest_path)}
    if expected_protocol is not None:
        input_hashes['protocol.json'] = sha(protocol_path)
    completed = {}
    hirshfeld_completed = {}
    hirshfeld_statuses = {}
    hirshfeld_errors = []
    hirshfeld_protocols = {}
    statuses = {}
    errors = []
    warnings = []
    raw_hashes = {}
    observed_state_ids = []
    protocols = {}
    structure_rows = []
    base_columns = ['structure_id', 'parent_id', 'variant', 'role', 'status', 'status_detail', 'hirshfeld_status', 'hirshfeld_status_detail', 'total_atoms',
                    'C2_atom_map', 'C2_index_0based', 'C5_atom_map', 'C5_index_0based', 'xyz_sha256',
                    'neutral_energy_hartree', 'cation_energy_hartree', 'vertical_ionization_energy_ev',
                    'cation_spin_square', 'neutral_internal_stable', 'cation_internal_stable',
                    'elapsed_seconds', 'peak_process_rss_mb', 'warnings',
                    'model_original_full_fspec_margin', 'model_symmetric_full_fspec_margin',
                    'xtb_C2_q0', 'xtb_C5_q0', 'xtb_q_contrast', 'xtb_C2_fminus', 'xtb_C5_fminus', 'xtb_fminus_contrast']
    pop_columns = [f'{partition}_{name}' for partition in PARTITIONS for name in
                   ('C2_q0', 'C5_q0', 'q_contrast', 'C2_qcation', 'C5_qcation', 'C2_fminus', 'C5_fminus', 'fminus_contrast')]
    for item in sorted(items, key=lambda x: (x['parent_id'], x['variant'])):
        sid = item['structure_id']
        run_dir = root / 'runs' / sid
        result_path = run_dir / 'result.json'
        status, detail = 'missing', 'No completed combined result'
        raw_status_path = run_dir / 'status.json'
        if raw_status_path.exists():
            try:
                raw_status = read(raw_status_path)
                status = 'awaiting_result' if raw_status.get('state') == 'complete' else raw_status.get('state', 'unknown')
                detail = raw_status.get('error') or raw_status.get('stage', '')
            except (ValueError, OSError) as exc:
                detail = f'Cannot read current status: {exc}'
        if result_path.exists():
            raw_hashes[str(result_path.relative_to(root)).replace('\\', '/')] = sha(result_path)
            try:
                result = read(result_path)
                run_protocol = verify_result(result, item, run_dir, expected_protocol)
                completed[sid] = result
                protocols[sid] = {key: run_protocol.get(key) for key in ('functional', 'orbital_basis', 'auxiliary_basis', 'grid_level', 'pyscf_version', 'numpy_version', 'libxc_version')}
                raw_hashes[str((run_dir / 'protocol.json').relative_to(root)).replace('\\', '/')] = sha(run_dir / 'protocol.json')
                status, detail = 'complete', 'Two electronic states independently checked'
                for label in ('neutral', 'cation'):
                    observed_state_ids.append(f'{sid}/{label}')
                    for warning in result['states'][label].get('warnings', []):
                        warnings.append({'structure_id': sid, 'state': label, 'warning': warning})
            except (KeyError, TypeError, ValueError, OSError) as exc:
                status, detail = 'invalid', str(exc)
                errors.append({'structure_id': sid, 'error': str(exc), 'raw_result': str(result_path.relative_to(root)).replace('\\', '/')})
        else:
            # Count completed standalone state records without inventing a pair.
            for label in ('neutral', 'cation'):
                state_path = run_dir / label / 'result.json'
                if state_path.exists():
                    try:
                        state_record = read(state_path)
                        verify_state(state_record, label, item)
                        observed_state_ids.append(f'{sid}/{label}')
                    except (KeyError, TypeError, ValueError, OSError) as exc:
                        errors.append({'structure_id': sid, 'state': label, 'error': str(exc)})
        statuses[sid] = {'status': status, 'detail': detail}
        h_dir = root / 'hirshfeld' / sid
        h_result_path = h_dir / 'result.json'
        h_status, h_detail = 'missing', 'No completed Hirshfeld result'
        if (h_dir / 'status.json').exists():
            try:
                live_h = read(h_dir / 'status.json')
                h_status = 'awaiting_result' if live_h.get('state') == 'complete' else live_h.get('state', 'unknown')
                h_detail = live_h.get('error') or live_h.get('stage', '')
            except (ValueError, OSError) as exc:
                h_detail = f'Cannot read current Hirshfeld status: {exc}'
        if h_result_path.exists():
            try:
                require(sid in completed, 'Hirshfeld source molecular pair has not passed summary validation')
                h_result = read(h_result_path)
                h_hashes, h_method = verify_hirshfeld(h_result, item, run_dir, h_dir, root)
                raw_hashes.update(h_hashes)
                hirshfeld_completed[sid] = h_result
                hirshfeld_protocols[sid] = h_method
                h_status, h_detail = 'complete', 'Source SHA, geometry, populations, both grids and refinement replay passed'
            except (KeyError, TypeError, ValueError, OSError) as exc:
                h_status, h_detail = 'invalid', str(exc)
                hirshfeld_errors.append({'structure_id': sid, 'error': str(exc), 'result_path': str(h_result_path.relative_to(root)).replace('\\', '/')})
        hirshfeld_statuses[sid] = {'status': h_status, 'detail': h_detail}
        xtb = item['prior_xtb']
        row = dict(structure_id=sid, parent_id=item['parent_id'], variant=item['variant'], role=item['role'],
                   status=status, status_detail=detail, hirshfeld_status=h_status, hirshfeld_status_detail=h_detail, total_atoms=item['total_atoms'],
                   C2_atom_map=item['sites']['C2']['atom_map'], C2_index_0based=item['sites']['C2']['atom_index_0based'],
                   C5_atom_map=item['sites']['C5']['atom_map'], C5_index_0based=item['sites']['C5']['atom_index_0based'],
                   xyz_sha256=item['xyz_sha256'], model_original_full_fspec_margin=model_value(item, 'original'),
                   model_symmetric_full_fspec_margin=model_value(item, 'symmetric'),
                   xtb_C2_q0=xtb['C2']['q0'], xtb_C5_q0=xtb['C5']['q0'], xtb_q_contrast=xtb['q_contrast'],
                   xtb_C2_fminus=xtb['C2']['f_minus'], xtb_C5_fminus=xtb['C5']['f_minus'], xtb_fminus_contrast=xtb['fminus_contrast'])
        if sid in completed:
            result = completed[sid]
            a, b = result['states']['neutral'], result['states']['cation']
            row.update(neutral_energy_hartree=a['energy_hartree'], cation_energy_hartree=b['energy_hartree'],
                       vertical_ionization_energy_ev=result['vertical_ionization_energy_ev'], cation_spin_square=b['spin_square'],
                       neutral_internal_stable=a['stability']['internal_stable'], cation_internal_stable=b['stability']['internal_stable'],
                       elapsed_seconds=result['elapsed_seconds'], peak_process_rss_mb=result['peak_process_rss_mb'],
                       warnings=' | '.join(a.get('warnings', []) + b.get('warnings', [])))
            c2, c5 = item['sites']['C2']['atom_index_0based'], item['sites']['C5']['atom_index_0based']
            for partition in NATIVE_PARTITIONS:
                q0, q1 = a[partition + '_charges'], b[partition + '_charges']
                fminus = result['f_minus'][partition]['values']
                row.update({f'{partition}_C2_q0': q0[c2], f'{partition}_C5_q0': q0[c5],
                            f'{partition}_q_contrast': q0[c2] - q0[c5], f'{partition}_C2_qcation': q1[c2],
                            f'{partition}_C5_qcation': q1[c5], f'{partition}_C2_fminus': fminus[c2],
                            f'{partition}_C5_fminus': fminus[c5], f'{partition}_fminus_contrast': fminus[c2] - fminus[c5]})
            if sid in hirshfeld_completed:
                h = hirshfeld_completed[sid]
                q0, q1, fminus = h['neutral_hirshfeld_charges'], h['cation_hirshfeld_charges'], h['f_minus']
                row.update(hirshfeld_C2_q0=q0[c2], hirshfeld_C5_q0=q0[c5], hirshfeld_q_contrast=q0[c2] - q0[c5],
                           hirshfeld_C2_qcation=q1[c2], hirshfeld_C5_qcation=q1[c5], hirshfeld_C2_fminus=fminus[c2],
                           hirshfeld_C5_fminus=fminus[c5], hirshfeld_fminus_contrast=fminus[c2] - fminus[c5])
        structure_rows.append(row)
    rows_by_id = {row['structure_id']: row for row in structure_rows}
    paired = []
    pair_columns = ['parent_id', 'variant', 'role', 'population', 'neutral_structure', 'changed_structure', 'status',
                    'delta_C2_q0', 'delta_C5_q0', 'delta_q_contrast', 'delta_C2_fminus', 'delta_C5_fminus', 'delta_fminus_contrast',
                    'neutral_fminus_contrast', 'changed_fminus_contrast',
                    'model_original_neutral_margin', 'model_original_changed_margin', 'model_original_delta',
                    'model_symmetric_neutral_margin', 'model_symmetric_changed_margin', 'model_symmetric_delta',
                    'xtb_delta_q_contrast', 'xtb_delta_fminus_contrast']
    for item in sorted(items, key=lambda x: (x['parent_id'], x['variant'])):
        if item['role'] == 'parent':
            continue
        sid, parent_sid = item['structure_id'], f"{item['parent_id']}_neutral"
        parent_item = by_id[parent_sid]
        a, b = rows_by_id[parent_sid], rows_by_id[sid]
        for partition in PARTITIONS:
            population_results = hirshfeld_completed if partition == 'hirshfeld' else completed
            pair_ok = sid in population_results and parent_sid in population_results
            row = dict(parent_id=item['parent_id'], variant=item['variant'], role=item['role'], population=partition,
                       neutral_structure=parent_sid, changed_structure=sid, status='complete' if pair_ok else 'incomplete',
                       model_original_neutral_margin=model_value(parent_item, 'original'), model_original_changed_margin=model_value(item, 'original'),
                       model_original_delta=numeric_delta(model_value(item, 'original'), model_value(parent_item, 'original')),
                       model_symmetric_neutral_margin=model_value(parent_item, 'symmetric'), model_symmetric_changed_margin=model_value(item, 'symmetric'),
                       model_symmetric_delta=numeric_delta(model_value(item, 'symmetric'), model_value(parent_item, 'symmetric')),
                       xtb_delta_q_contrast=item['prior_xtb']['q_contrast'] - parent_item['prior_xtb']['q_contrast'],
                       xtb_delta_fminus_contrast=item['prior_xtb']['fminus_contrast'] - parent_item['prior_xtb']['fminus_contrast'])
            if pair_ok:
                for feature in ('C2_q0', 'C5_q0', 'q_contrast', 'C2_fminus', 'C5_fminus', 'fminus_contrast'):
                    row['delta_' + feature] = b[f'{partition}_{feature}'] - a[f'{partition}_{feature}']
                row.update(neutral_fminus_contrast=a[f'{partition}_fminus_contrast'], changed_fminus_contrast=b[f'{partition}_fminus_contrast'])
            paired.append(row)
    paired_index = {(x['parent_id'], x['variant'], x['population']): x for x in paired}
    controls = []
    control_columns = ['parent_id', 'alternative_variant', 'population', 'status', 'target_delta_q_contrast',
                       'alternative_delta_q_contrast', 'target_minus_alternative_delta_q_contrast',
                       'target_delta_fminus_contrast', 'alternative_delta_fminus_contrast', 'target_minus_alternative_delta_fminus_contrast',
                       'model_symmetric_target_delta', 'model_symmetric_alternative_delta', 'model_symmetric_target_minus_alternative_delta',
                       'xtb_target_delta_fminus_contrast', 'xtb_alternative_delta_fminus_contrast', 'xtb_target_minus_alternative_delta_fminus_contrast']
    for alt in paired:
        if alt['role'] != 'alternative_N_oxide':
            continue
        target = paired_index[(alt['parent_id'], 'N_oxide', alt['population'])]
        ok = target['status'] == alt['status'] == 'complete'
        row = dict(parent_id=alt['parent_id'], alternative_variant=alt['variant'], population=alt['population'],
                   status='complete' if ok else 'incomplete', model_symmetric_target_delta=target['model_symmetric_delta'],
                   model_symmetric_alternative_delta=alt['model_symmetric_delta'],
                   model_symmetric_target_minus_alternative_delta=target['model_symmetric_delta'] - alt['model_symmetric_delta'],
                   xtb_target_delta_fminus_contrast=target['xtb_delta_fminus_contrast'],
                   xtb_alternative_delta_fminus_contrast=alt['xtb_delta_fminus_contrast'],
                   xtb_target_minus_alternative_delta_fminus_contrast=target['xtb_delta_fminus_contrast'] - alt['xtb_delta_fminus_contrast'])
        if ok:
            for feature in ('delta_q_contrast', 'delta_fminus_contrast'):
                row['target_' + feature] = target[feature]
                row['alternative_' + feature] = alt[feature]
                row['target_minus_alternative_' + feature] = target[feature] - alt[feature]
        controls.append(row)
    full = len(completed) == 17 and len(observed_state_ids) == 34 and not errors and expected_protocol is not None
    full_hirshfeld = len(hirshfeld_completed) == 17 and not hirshfeld_errors
    partition_summaries = {}
    for partition in PARTITIONS:
        target_rows = [row for row in paired if row['population'] == partition and row['role'] == 'target_N_oxide' and row['status'] == 'complete']
        alt_rows = [row for row in controls if row['population'] == partition and row['status'] == 'complete']
        partition_summaries[partition] = {
            'target_pairs_available': len(target_rows), 'target_pairs_expected': 6,
            'target_parent_ids_available': [row['parent_id'] for row in target_rows],
            'target_delta_q_contrast': descriptive([row['delta_q_contrast'] for row in target_rows]),
            'target_delta_fminus_contrast': descriptive([row['delta_fminus_contrast'] for row in target_rows]),
            'target_dft_vs_xtb_fminus_signs': compare_signs(target_rows, 'delta_fminus_contrast', 'xtb_delta_fminus_contrast'),
            'target_dft_vs_original_model_signs': compare_signs(target_rows, 'delta_fminus_contrast', 'model_original_delta'),
            'target_dft_vs_symmetric_model_signs': compare_signs(target_rows, 'delta_fminus_contrast', 'model_symmetric_delta'),
            'alternative_control_pairs_available': len(alt_rows), 'alternative_control_pairs_expected': 5,
            'alternative_control_distinct_parents_available': len({row['parent_id'] for row in alt_rows}),
            'alternative_control_distinct_parents_expected': 4,
            'target_minus_alternative_delta_fminus_contrast': descriptive([row['target_minus_alternative_delta_fminus_contrast'] for row in alt_rows])}
    summary = dict(schema='spectraretro.dft_summary.v1', created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                   state='complete' if full else ('partial_with_errors' if errors else 'partial'),
                   complete=full, completion_scope='34 fixed-geometry DFT singlepoints; Hirshfeld completion is recorded separately',
                   full_analysis_complete=full and full_hirshfeld,
                   hirshfeld_state='complete' if full_hirshfeld else ('partial_with_errors' if hirshfeld_errors else 'partial'),
                   valid_hirshfeld_structures=len(hirshfeld_completed), expected_hirshfeld_structures=17,
                   hirshfeld_structures=hirshfeld_statuses, hirshfeld_errors=hirshfeld_errors, hirshfeld_method_and_software=hirshfeld_protocols,
                   expected_structures=17, valid_complete_structures=len(completed),
                   expected_singlepoint_states=34, valid_states_in_complete_structure_pairs=2 * len(completed),
                   observed_valid_state_records=len(observed_state_ids), observed_valid_state_ids=observed_state_ids,
                   independent_parent_structures=6, target_perturbations=6, alternative_site_controls=5,
                   alternative_control_parent_structures=4,
                   geometry='DFT single points at original GFN2-xTB optimized charge-zero geometry',
                   populations={'meta_lowdin': 'PySCF meta-Lowdin with ANO pre-orthogonalization', 'mulliken': 'PySCF Mulliken',
                                'hirshfeld': 'Native PySCF Hirshfeld on saved molecular density; neutral spherical atomic HF references, independently verified grid refinement'},
                   quantities={'q_contrast': 'q(C2)-q(C5)', 'fminus': 'q_cation-q_neutral at the same coordinates',
                               'fminus_contrast': 'fminus(C2)-fminus(C5)', 'delta': 'oxidation variant minus its neutral parent',
                               'model_margin': 'logit(C2 exocyclic candidate bond)-logit(C5 exocyclic candidate bond)'},
                   partition_summaries=partition_summaries, structures=statuses, errors=errors, warnings=warnings,
                   run_method_and_software=protocols, input_sha256=input_hashes, raw_result_sha256=raw_hashes,
                   summary_script_sha256=sha(__file__),
                   completion_gate={'all_17_structure_results_valid': len(completed) == 17,
                                    'all_34_states_valid': len(observed_state_ids) == 34,
                                    'no_validation_errors': not errors, 'batch_protocol_available': expected_protocol is not None},
                   interpretation_notes=[
                       'Partial summaries use only explicitly listed available pairs; missing DFT values are blank, never zero.',
                       'The population methods are sensitivity readouts on the same electronic states, not independent replicates.',
                       'Hirshfeld values come from separately verified native density postprocessing. Missing or invalid Hirshfeld values are blank; the 34-singlepoint completion gate is separate.',
                       'Target and alternative controls sharing a parent are dependent; 17 structures do not give 17 independent reactions.',
                       'Original primary and later symmetric model scores are separate columns. Alternative-control comparisons use the common symmetric readout.',
                       'Signs and magnitudes are reported regardless of whether they support prior expectations; no outcome-based exclusion or p-value is applied.',
                       'Fixed-geometry electronic descriptors characterize model-input perturbations; these outputs contain no DFT geometry optimization, frequency, transition state or reaction-rate calculation.'])
    atomic_csv(out / 'per_structure.csv', structure_rows, base_columns + pop_columns)
    atomic_csv(out / 'paired_electronic_comparison.csv', paired, pair_columns)
    atomic_csv(out / 'alternative_N_controls.csv', controls, control_columns)
    for name in ('per_structure.csv', 'paired_electronic_comparison.csv', 'alternative_N_controls.csv'):
        summary.setdefault('derived_files_sha256', {})[name] = sha(out / name)
    # Check that source records read for this snapshot did not change mid-summary.
    for rel, expected in {**input_hashes, **raw_hashes}.items():
        require(sha(root / rel) == expected, f'Source changed during summary: {rel}')
    atomic_json(out / 'summary.json', summary)
    print(json.dumps({'state': summary['state'], 'valid_structures': len(completed),
                      'expected_structures': 17, 'valid_states': len(observed_state_ids), 'expected_states': 34,
                      'hirshfeld_state': summary['hirshfeld_state'], 'valid_hirshfeld_structures': len(hirshfeld_completed),
                      'errors': len(errors), 'hirshfeld_errors': len(hirshfeld_errors), 'output_dir': str(out)}, ensure_ascii=False))
    if args.require_complete and not full:
        raise SystemExit(2)
    if args.require_hirshfeld and not (full and full_hirshfeld):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
