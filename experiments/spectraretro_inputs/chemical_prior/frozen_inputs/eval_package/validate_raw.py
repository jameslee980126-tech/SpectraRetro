"""Read-only full candidate-bank audit before decoding or reuse of a control arm."""
import argparse
import json
from pathlib import Path
import numpy as np
import portable_common as common


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle-root', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--arm', required=True, choices=common.ARMS)
    args = parser.parse_args()
    common.configure(args.bundle_root, args.run_root)
    root = common.ROOT
    protocol = json.loads((root / 'protocol.json').read_text(encoding='utf8'))
    raw = root / 'evaluation' / args.arm / 'raw'
    completed = json.loads((raw / 'completed.json').read_text(encoding='utf8'))
    assert completed['rows'] == 4161 and completed['candidates_per_row'] == 50
    with np.load(root / 'shared_sizes.npz', allow_pickle=False) as saved:
        bank = {uid: sizes.copy() for uid, sizes in zip(saved['uuids'].astype(str), saved['sizes'])}
        original_ids = saved['uuids'].astype(str).tolist()
    with np.load(Path(__file__).resolve().parent / 'reference_bank.npz', allow_pickle=False) as saved:
        reference = {str(uid): (saved['positions'][a:b].copy(), saved['atom_types'][a:b].copy())
                     for uid, a, b in zip(saved['uuids'], saved['offsets'][:-1], saved['offsets'][1:])}
    order = json.loads((root / 'batch_order.json').read_text(encoding='utf8'))
    seen, sources, known_error, reference_error, slots = set(), {}, 0., 0., 0
    for batch_index, path in enumerate(sorted(raw.glob('batch_*.npz'))):
        sources[path.name] = common.digest(path)
        with np.load(path, allow_pickle=False) as saved:
            uuids = saved['uuids'].astype(str).tolist()
            expected = [original_ids[i] for i in order[batch_index * 64:(batch_index + 1) * 64]]
            assert uuids == expected
            assert not set(uuids) & seen
            seen.update(uuids)
            sizes = saved['sizes']
            assert sizes.shape == (50, len(uuids)) and np.array_equal(sizes, np.stack([bank[uid] for uid in uuids], axis=1))
            failed = saved['failed']
            assert failed.shape == sizes.shape and not failed.any()
            offsets = saved['offsets']
            ref_offsets = saved['true_offsets']
            coords, elements = saved['positions'], saved['atom_types']
            ref_coords, ref_elements = saved['true_positions'], saved['true_atom_types']
            assert offsets.shape == (50 * len(uuids) + 1,) and offsets[0] == 0 and offsets[-1] == len(coords) == len(elements)
            assert ref_offsets.shape == (len(uuids) + 1,) and ref_offsets[0] == 0 and ref_offsets[-1] == len(ref_coords) == len(ref_elements)
            assert np.isfinite(coords).all() and np.isfinite(ref_coords).all()
            assert elements.min() >= 0 and elements.max() < 26 and ref_elements.min() >= 0 and ref_elements.max() < 26
            fragments = saved['fragment_counts']
            for i, uid in enumerate(uuids):
                start, end = ref_offsets[i:i + 2]
                xyz, atom_types = reference[uid]
                assert np.array_equal(ref_elements[start:end], atom_types)
                assert len(ref_coords[start:end]) == len(xyz)
                err = float(np.max(np.abs(ref_coords[start:end] - xyz)))
                reference_error = max(reference_error, err)
                if err > 1e-5:
                    raise ValueError('Reference centering differs beyond frozen numerical tolerance: ' + uid)
                count = int(fragments[i])
                for draw in range(50):
                    a, b = offsets[draw * len(uuids) + i:draw * len(uuids) + i + 2]
                    assert b - a == count + sizes[draw, i]
                    assert np.array_equal(elements[a:a + count], ref_elements[start:start + count])
                    err = float(np.max(np.abs(coords[a:a + count] - ref_coords[start:start + count]))) if count else 0.
                    known_error = max(known_error, err)
                    if err != 0:
                        raise ValueError('Observed known coordinates changed')
                    slots += 1
    assert len(seen) == 4161 and slots == 208050 and len(sources) == 66
    report = dict(status='passed', arm=args.arm, rows=4161, candidate_slots=slots, archives=len(sources),
                  failed_slots=0, finite_coordinates=True, sizes_equal_shared_bank=True,
                  known_coordinates_max_abs_error=known_error, reference_centering_max_abs_error=reference_error,
                  reference_centering_tolerance=1e-5, protocol_sha256=common.digest(root / 'protocol.json'),
                  source_archive_sha256=sources, no_new_sampling=True, no_scores_changed=True)
    destination = root / 'evaluation' / args.arm / 'raw_verification.json'
    if destination.exists():
        raise FileExistsError(destination)
    common.write_json(destination, report)
    print(json.dumps({key: report[key] for key in ['status', 'arm', 'rows', 'candidate_slots', 'archives', 'known_coordinates_max_abs_error']}), flush=True)


if __name__ == '__main__':
    main()
