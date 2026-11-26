"""CPU replay of all frozen references in the current Linux OpenBabel environment."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import importlib.metadata
import json
import multiprocessing
from pathlib import Path
import platform
import numpy as np
import decode as frozen_decoder
import portable_common as common


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle-root', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    common.configure(args.bundle_root, args.run_root)
    root = common.ROOT
    root.mkdir(parents=True, exist_ok=True)
    for name in ['reference_replay.csv', 'reference_replay_manifest.json']:
        if (root / name).exists():
            raise FileExistsError(name)
    package = Path(__file__).resolve().parent
    fixture_manifest = json.loads((package / 'fixture_manifest.json').read_text(encoding='utf8'))
    bank = package / 'reference_bank.npz'
    if common.digest(bank) != fixture_manifest['artifacts'][bank.name]:
        raise ValueError('Reference bank changed')
    with np.load(bank, allow_pickle=False) as saved:
        uuids = saved['uuids'].astype(str).tolist()
        offsets = saved['offsets']
        jobs = [(saved['positions'][a:b].copy(), saved['atom_types'][a:b].copy()) for a, b in zip(offsets[:-1], offsets[1:])]
        expected_windows = saved['windows_decoded'].astype(str).tolist()
    assert len(uuids) == len(set(uuids)) == len(jobs) == 4161
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        strings = list(pool.map(frozen_decoder.decode, jobs, chunksize=32))
    failed = [uid for uid, value in zip(uuids, strings) if value == 'MISSING']
    if failed:
        common.write_json(root / 'reference_replay_failure.json', dict(status='failed', failed_uuids=failed))
        raise ValueError('Reference XYZ decoding failed; retain failures, halt evaluation')
    with (root / 'reference_replay.csv').open('w', encoding='utf8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['uuid', 'reference'])
        writer.writerows(zip(uuids, strings))
    from openbabel import openbabel as ob, _openbabel
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    sanitization_failures = []
    for uid, value in zip(uuids, strings):
        if Chem.MolFromSmiles(value) is None:
            sanitization_failures.append(uid)
    differences = [uid for uid, a, b in zip(uuids, strings, expected_windows) if a != b]
    binary = Path(_openbabel.__file__)
    try:
        wheel_version = importlib.metadata.version('openbabel-wheel')
    except importlib.metadata.PackageNotFoundError:
        wheel_version = None
    common.write_json(root / 'reference_replay_manifest.json', dict(
        status='passed', rows=4161, failed_reference_decodes=0,
        reference_bank_sha256=common.digest(bank), csv_sha256=common.digest(root / 'reference_replay.csv'),
        decoder_source_sha256=common.digest(package / 'decode.py'),
        openbabel=ob.OBReleaseVersion(), openbabel_wheel_version=wheel_version,
        openbabel_binary_sha256=common.digest(binary), platform=platform.platform(),
        windows_exact_string_matches=4161 - len(differences), windows_difference_uuids=differences,
        rdkit_sanitization_failure_uuids=sanitization_failures,
        caveat='sanitization failures are diagnosed and retained; XYZ decoded references are not original complete chemical structures; Linux absolute scores cannot be mixed with Windows',
        new_generation=False, new_training=False))
    print(json.dumps(dict(status='passed', rows=4161, windows_differences=len(differences),
                          rdkit_sanitization_failures=len(sanitization_failures))), flush=True)


if __name__ == '__main__':
    main()
