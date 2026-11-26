"""Extract a native preprocessing cache into a new directory and verify its members."""
import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path, help='ZIP path relative to the repository root')
    parser.add_argument('--output', type=Path, required=True, help='New destination directory')
    args = parser.parse_args()
    archive = args.archive if args.archive.is_absolute() else ROOT/args.archive
    target = (args.output if args.output.is_absolute() else ROOT/args.output).resolve()
    if target.is_relative_to(ROOT.resolve()) and not target.is_relative_to((ROOT/'outputs').resolve()):
        parser.error('An in-repository destination must be under outputs/')
    if target.exists():
        parser.error('The destination must be new')
    manifest = Path(str(archive)+'.members.json')
    if not manifest.exists():
        parser.error('The archive member manifest is required')
    records = json.loads(manifest.read_text(encoding='utf-8'))['members']
    by_name = {r['path']:r for r in records}
    with zipfile.ZipFile(archive) as z:
        names = [i.filename for i in z.infolist() if not i.is_dir()]
        if len(names) != len(set(names)) or set(names) != set(by_name):
            raise ValueError('Archive contents differ from the member manifest')
        for name in names:
            if not (target/name).resolve().is_relative_to(target):
                raise ValueError('Archive path outside the destination')
        target.mkdir(parents=True)
        for name in names:
            out = target/name
            out.parent.mkdir(parents=True, exist_ok=True)
            h = hashlib.sha256()
            with z.open(name) as inp, out.open('xb') as dst:
                while data := inp.read(4*1024*1024):
                    h.update(data)
                    dst.write(data)
            if h.hexdigest() != by_name[name]['sha256'] or out.stat().st_size != by_name[name]['bytes']:
                raise ValueError('Cache member identity mismatch: '+name)
    print(json.dumps({'passed':True, 'members':len(records), 'output':str(target)}))


if __name__ == '__main__':
    main()
