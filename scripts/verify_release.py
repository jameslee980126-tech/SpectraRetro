"""Verify the files distributed with SpectraRetro."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    manifest = ROOT / 'MANIFEST_SHA256.csv'
    expected = (ROOT / 'MANIFEST_SHA256.csv.sha256').read_text().split()[0]
    if digest(manifest) != expected:
        raise ValueError('Manifest checksum differs')
    with manifest.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    seen, total = set(), 0
    for row in rows:
        path = (ROOT / row['path']).resolve()
        if not path.is_relative_to(ROOT) or path in seen:
            raise ValueError('Unsafe or duplicate manifest path')
        seen.add(path)
        if not path.is_file() or path.stat().st_size != int(row['bytes']) or digest(path) != row['sha256']:
            raise ValueError('File missing or changed: ' + row['path'])
        total += int(row['bytes'])
    print(json.dumps(dict(status='passed', files=len(rows), bytes=total,
                         scope='Distributed file identities; scientific computations are checked separately.')))


if __name__ == '__main__':
    main()
