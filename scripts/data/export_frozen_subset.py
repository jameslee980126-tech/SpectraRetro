"""Reconstruct the frozen subset from parent reaction rows and distributed indices."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    target = (args.output if args.output.is_absolute() else ROOT/args.output).resolve()
    if target.exists() or target.is_relative_to(ROOT) and not target.is_relative_to(ROOT/'outputs'):
        parser.error('Select a new directory; in-repository outputs must be under outputs/')
    prepared = {}
    for split in ['train', 'valid', 'test']:
        with (ROOT/f'datasets/full/{split}.csv').open(encoding='utf-8-sig',newline='') as h:
            reader=csv.DictReader(h); fields=reader.fieldnames; parent=list(reader)
        with (ROOT/f'datasets/small/{split}_index.csv').open(encoding='utf-8-sig',newline='') as h:
            indices=list(csv.DictReader(h))
        selected=[parent[int(r['source_row'])] for r in indices]
        with (ROOT/f'datasets/small/{split}.csv').open(encoding='utf-8-sig',newline='') as h:
            expected=list(csv.DictReader(h))
        if selected != expected:
            raise ValueError('Frozen parent-row binding differs: '+split)
        prepared[split]=(fields,selected)
    target.mkdir(parents=True)
    for split,(fields,rows) in prepared.items():
        with (target/f'{split}.csv').open('w',encoding='utf-8',newline='') as h:
            writer=csv.DictWriter(h,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    print(json.dumps({'passed':True,'reactions':{s:len(v[1]) for s,v in prepared.items()},'comparison':'all reaction fields and order; serialization may differ'}))


if __name__=='__main__':
    main()
