"""Decode retained 3D draws and references through the same XYZ/Open Babel path."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import importlib.metadata
import json
import multiprocessing
from pathlib import Path
import sys
import numpy as np


ELEMENTS = 'C N O S B Br Cl P I F Mg Si Sn Zn Cu Se H Li Na Cr Mn Ge Ru Ag Te Pt'.split()


def decode(job):
    from openbabel import openbabel as ob
    ob.obErrorLog.SetOutputLevel(-1)
    xyz, elements = job
    if not np.isfinite(xyz).all():
        return 'MISSING'
    text = str(len(elements)) + '\n\n' + ''.join(
        f'{ELEMENTS[int(t)]} {x[0]:.9f} {x[1]:.9f} {x[2]:.9f}\n' for x, t in zip(xyz, elements))
    conversion = ob.OBConversion()
    if not conversion.SetInAndOutFormats('xyz', 'smi'):
        raise RuntimeError('Open Babel XYZ/SMILES plugins unavailable')
    mol = ob.OBMol()
    if not conversion.ReadString(mol, text):
        return 'MISSING'
    answer = conversion.WriteString(mol).split('\t')[0]
    return answer if answer.strip() else 'MISSING'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    release = Path(__file__).resolve().parents[3]
    args.input = (release / args.input).resolve() if not args.input.is_absolute() else args.input.resolve()
    args.output = (release / args.output).resolve() if not args.output.is_absolute() else args.output.resolve()
    if args.output == release or (release in args.output.parents
            and args.output.relative_to(release).parts[0] != 'outputs'):
        parser.error('Outputs inside the repository must be placed under outputs/')
    if args.output.exists():
        parser.error('Output directory already exists; choose a new directory')
    complete = json.loads((args.input / 'completed.json').read_text(encoding='utf8'))
    n_samples = complete['candidates_per_row']
    args.output.mkdir(parents=True, exist_ok=False)
    seen, hashes, failed = set(), {}, 0
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        with (args.output / 'result.csv').open('w', encoding='utf8', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['uuid'] + [str(i) for i in range(1, n_samples + 2)])
            for path in sorted(args.input.glob('batch_*.npz')):
                hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
                with np.load(path, allow_pickle=False) as batch:
                    ids = batch['uuids'].astype(str).tolist()
                    if len(set(ids)) != len(ids) or seen.intersection(ids):
                        raise ValueError('Duplicate fragment UUID in archives')
                    seen.update(ids)
                    offsets = batch['offsets']
                    if len(offsets) != len(ids) * n_samples + 1:
                        raise ValueError('Draw-major candidate count mismatch')
                    if (offsets[0] != 0 or np.any(np.diff(offsets) < 0)
                            or offsets[-1] != len(batch['positions'])
                            or len(batch['positions']) != len(batch['atom_types'])):
                        raise ValueError('Invalid candidate offsets or array lengths')
                    failed_mask = batch['failed']
                    if failed_mask.shape != (n_samples, len(ids)):
                        raise ValueError('Failure slots do not match draw-major layout')
                    positions, atom_types = batch['positions'], batch['atom_types']
                    jobs = [(positions[lo:hi], atom_types[lo:hi])
                            for lo, hi in zip(offsets[:-1], offsets[1:])]
                    values = list(pool.map(decode, jobs, chunksize=32))
                    values = ['MISSING' if flag else value for flag, value in zip(failed_mask.flatten(), values)]
                    failed += sum(v == 'MISSING' for v in values)
                    ref = batch['true_offsets']
                    if (len(ref) != len(ids) + 1 or ref[0] != 0 or np.any(np.diff(ref) < 0)
                            or ref[-1] != len(batch['true_positions'])
                            or len(batch['true_positions']) != len(batch['true_atom_types'])):
                        raise ValueError('Invalid reference offsets or array lengths')
                    true_positions, true_atom_types = batch['true_positions'], batch['true_atom_types']
                    references = list(pool.map(decode, [(true_positions[lo:hi], true_atom_types[lo:hi])
                                                       for lo, hi in zip(ref[:-1], ref[1:])]))
                    if len(references) != len(ids) or any(v == 'MISSING' for v in references):
                        raise ValueError('Reference decoding failed; cannot count failed answers as success')
                    for i, uuid in enumerate(ids):
                        writer.writerow([uuid] + values[i::len(ids)] + [references[i]])
                    handle.flush()
                    print(json.dumps({'decoded_rows': len(seen), 'archive': path.name}), flush=True)
    if len(seen) != complete['rows']:
        raise ValueError('Decoded archive coverage incomplete')
    from openbabel import openbabel as ob
    from openbabel import _openbabel
    binary = Path(_openbabel.__file__)
    try:
        wheel_version = importlib.metadata.version('openbabel-wheel')
    except importlib.metadata.PackageNotFoundError:
        wheel_version = None
    report = {'rows': len(seen), 'n_samples': n_samples, 'failed_decodes': failed,
              'openbabel': ob.OBReleaseVersion(), 'source_hashes': hashes,
              'openbabel_wheel_version': wheel_version, 'openbabel_module': ob.__file__,
              'openbabel_binary': str(binary), 'openbabel_binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
              'reference': 'same original 3D XYZ decoding as predictions; not original full chemical structure',
              'protocol': 'Use the same Open Babel build for references and predictions; reference column follows the 300 generated draw columns'}
    (args.output / 'decode_manifest.json').write_text(json.dumps(report, indent=2), encoding='utf8')


if __name__ == '__main__':
    main()
