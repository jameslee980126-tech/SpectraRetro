from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from rdkit import Chem
from rdkit import RDLogger


RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "C-H Arylation" / "C-H Arylation"
OUTPUT_DIR = ROOT / "ch_arylation_dataset_audit"
INPUT_FILES = [
    "canonicalized_train.csv",
    "canonicalized_valid.csv",
    "canonicalized_test.csv",
]


HALOGENS = {"F", "Cl", "Br", "I"}


def split_rxn(rxn: str) -> Optional[Tuple[str, str, str]]:
    rxn = (rxn or "").strip().split()[0]
    if rxn.count(">") == 2:
        left, middle, right = rxn.split(">")
        return left, middle, right
    if ">>" in rxn:
        left, right = rxn.split(">>", 1)
        return left, "", right
    return None


def mol_from_side(side: str) -> Optional[Chem.Mol]:
    try:
        return Chem.MolFromSmiles(side)
    except Exception:
        return None


def atom_info(mol: Chem.Mol) -> Dict[int, Dict[str, object]]:
    out: Dict[int, Dict[str, object]] = {}
    for atom in mol.GetAtoms():
        amap = atom.GetAtomMapNum()
        if amap:
            out[amap] = {
                "symbol": atom.GetSymbol(),
                "is_aromatic": atom.GetIsAromatic(),
                "total_h": atom.GetTotalNumHs(),
            }
    return out


def bond_info(mol: Chem.Mol) -> Dict[Tuple[int, int], str]:
    out: Dict[Tuple[int, int], str] = {}
    for bond in mol.GetBonds():
        a = bond.GetBeginAtom().GetAtomMapNum()
        b = bond.GetEndAtom().GetAtomMapNum()
        if not a or not b:
            continue
        out[tuple(sorted((a, b)))] = str(bond.GetBondType())
    return out


def classify_bond(a: Dict[str, object], b: Dict[str, object]) -> str:
    s1 = str(a["symbol"])
    s2 = str(b["symbol"])
    symbols = sorted([s1, s2])
    pair = "-".join(symbols)

    if pair == "C-C":
        a_arom = bool(a["is_aromatic"])
        b_arom = bool(b["is_aromatic"])
        if a_arom and b_arom:
            return "C-C arylation/biaryl or heteroaryl arylation"
        if a_arom or b_arom:
            return "C-C aryl-alkyl/benzyl/alkenyl coupling"
        return "C-C aliphatic/other carbon coupling"
    if pair == "C-N":
        return "C-N amination/amidation/N-functionalization"
    if pair == "C-O":
        return "C-O oxygenation/etherification/acylation"
    if pair == "C-P":
        return "C-P phosphination"
    if pair == "C-B":
        return "C-B borylation"
    if pair == "C-Si":
        return "C-Si silylation"
    if pair == "C-S":
        return "C-S thiolation/sulfur coupling"
    if "C" in symbols and (set(symbols) & HALOGENS):
        return "C-halogen halogenation/halide formation"
    return f"{pair} other bond formation"


def analyze_row(row: Dict[str, str]) -> Dict[str, object]:
    rxn = row.get("mapped_rxn") or row.get("reactants>reagents>production") or ""
    parts = split_rxn(rxn)
    result: Dict[str, object] = {
        "parse_ok": False,
        "major_subclass": "unparsed",
        "formed_bond_classes": "",
        "formed_bond_count": 0,
        "likely_c_h_functionalization": False,
        "element_changes": "",
        "notes": "",
    }
    if not parts:
        result["notes"] = "cannot split reaction"
        return result

    left, _middle, right = parts
    r_mol = mol_from_side(left)
    p_mol = mol_from_side(right)
    if r_mol is None or p_mol is None:
        result["notes"] = "cannot parse mapped reactant or product"
        return result

    r_atoms = atom_info(r_mol)
    p_atoms = atom_info(p_mol)
    r_bonds = bond_info(r_mol)
    p_bonds = bond_info(p_mol)

    formed = [pair for pair in p_bonds if pair not in r_bonds]
    # Some mapped reactions omit reagents on the reactant side but include
    # the installed group in the product. Bonds internal to such newly
    # installed groups are not reaction-forming bonds, so only keep formed
    # bonds touching at least one atom that existed on the reactant side.
    reactive_formed = [pair for pair in formed if pair[0] in r_atoms or pair[1] in r_atoms]
    changed_elements = []
    for amap, p_info in p_atoms.items():
        r_info = r_atoms.get(amap)
        if r_info and r_info["symbol"] != p_info["symbol"]:
            changed_elements.append(f"{amap}:{r_info['symbol']}->{p_info['symbol']}")

    classes: List[str] = []
    ch_like = False
    for pair in reactive_formed:
        a_info = p_atoms.get(pair[0])
        b_info = p_atoms.get(pair[1])
        if not a_info or not b_info:
            continue
        classes.append(classify_bond(a_info, b_info))
        for amap, info in [(pair[0], a_info), (pair[1], b_info)]:
            r_info = r_atoms.get(amap)
            if info["symbol"] == "C" and r_info:
                try:
                    if int(r_info["total_h"]) > int(info["total_h"]):
                        ch_like = True
                except Exception:
                    pass

    class_counts = Counter(classes)
    result["parse_ok"] = True
    result["formed_bond_count"] = len(reactive_formed)
    result["formed_bond_classes"] = "; ".join(f"{k} x{v}" for k, v in class_counts.items())
    result["likely_c_h_functionalization"] = ch_like
    result["element_changes"] = "; ".join(changed_elements)

    if classes:
        main_class, _ = class_counts.most_common(1)[0]
        if ch_like and main_class.startswith("C-C arylation"):
            result["major_subclass"] = "likely C-H arylation"
        elif ch_like and main_class.startswith("C-C"):
            result["major_subclass"] = "likely other C-H carbon coupling"
        elif ch_like:
            result["major_subclass"] = "likely non-arylation C-H functionalization"
        else:
            result["major_subclass"] = main_class
    elif changed_elements:
        result["major_subclass"] = "functional group transformation/no new mapped bond"
    else:
        result["major_subclass"] = "no detected mapped bond formation"

    return result


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    audited: List[Dict[str, object]] = []
    file_counts = Counter()

    for filename in INPUT_FILES:
        path = DATA_DIR / filename
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                analysis = analyze_row(row)
                out = dict(row)
                out["split"] = filename.replace("canonicalized_", "").replace(".csv", "")
                out.update(analysis)
                audited.append(out)
                file_counts[filename] += 1

    subclass_counts = Counter(str(row["major_subclass"]) for row in audited)
    formed_class_counts = Counter()
    split_subclass_counts: Dict[str, Counter] = defaultdict(Counter)
    for row in audited:
        split_subclass_counts[str(row["split"])][str(row["major_subclass"])] += 1
        for part in str(row["formed_bond_classes"]).split("; "):
            if not part:
                continue
            label = part.rsplit(" x", 1)[0]
            count = int(part.rsplit(" x", 1)[1]) if " x" in part else 1
            formed_class_counts[label] += count

    fields = list(audited[0].keys()) if audited else []
    audit_path = OUTPUT_DIR / "original_ch_arylation_subclass_audit.csv"
    summary_path = OUTPUT_DIR / "original_ch_arylation_subclass_summary.json"
    examples_path = OUTPUT_DIR / "original_ch_arylation_subclass_examples.csv"
    md_path = OUTPUT_DIR / "ORIGINAL_CH_ARYLATION_SUBCLASS_AUDIT.md"

    write_csv(audit_path, audited, fields)

    examples = []
    seen = set()
    for row in audited:
        key = str(row["major_subclass"])
        if key in seen:
            continue
        seen.add(key)
        examples.append(row)
    write_csv(examples_path, examples, fields)

    summary = {
        "input_files": {k: file_counts[k] for k in INPUT_FILES},
        "total_rows": len(audited),
        "subclass_counts": dict(subclass_counts.most_common()),
        "formed_bond_class_counts": dict(formed_class_counts.most_common()),
        "split_subclass_counts": {
            split: dict(counter.most_common()) for split, counter in split_subclass_counts.items()
        },
        "audit_file": str(audit_path),
        "examples_file": str(examples_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Original C-H Arylation Subclass Audit",
        "",
        f"Total rows audited: {len(audited)}",
        "",
        "## Major Subclass Counts",
        "",
        "| Subclass | Rows |",
        "|---|---:|",
    ]
    for label, count in subclass_counts.most_common():
        lines.append(f"| {label} | {count} |")
    lines.extend(
        [
            "",
            "## Formed Bond Class Counts",
            "",
            "| Formed bond class | Count |",
            "|---|---:|",
        ]
    )
    for label, count in formed_class_counts.most_common():
        lines.append(f"| {label} | {count} |")
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            f"- Full audit: `{audit_path}`",
            f"- One example per subclass: `{examples_path}`",
            f"- JSON summary: `{summary_path}`",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
