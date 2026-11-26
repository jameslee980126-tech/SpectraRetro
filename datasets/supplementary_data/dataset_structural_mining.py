from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

from rdkit import Chem
from rdkit import RDLogger


RDLogger.DisableLog("rdApp.*")
try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2**31 - 1)

HALOGENS = {"F", "Cl", "Br", "I"}


def split_rxn(rxn: str) -> Optional[Tuple[str, str, str]]:
    rxn = (rxn or "").strip().split()[0]
    if rxn.count(">") == 2:
        return tuple(rxn.split(">", 2))  # type: ignore[return-value]
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
        if not amap:
            continue
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
        if a and b:
            out[tuple(sorted((a, b)))] = str(bond.GetBondType())
    return out


def atom_by_map(mol: Chem.Mol) -> Dict[int, Chem.Atom]:
    return {a.GetAtomMapNum(): a for a in mol.GetAtoms() if a.GetAtomMapNum()}


def has_double_to(atom: Chem.Atom, symbol: str) -> bool:
    for bond in atom.GetBonds():
        other = bond.GetOtherAtom(atom)
        if other.GetSymbol() == symbol and str(bond.GetBondType()) == "DOUBLE":
            return True
    return False


def has_triple_to(atom: Chem.Atom, symbol: str) -> bool:
    for bond in atom.GetBonds():
        other = bond.GetOtherAtom(atom)
        if other.GetSymbol() == symbol and str(bond.GetBondType()) == "TRIPLE":
            return True
    return False


def has_double_to_carbon(atom: Chem.Atom) -> bool:
    for bond in atom.GetBonds():
        other = bond.GetOtherAtom(atom)
        if other.GetSymbol() == "C" and str(bond.GetBondType()) == "DOUBLE":
            return True
    return False


def has_triple_to_carbon(atom: Chem.Atom) -> bool:
    for bond in atom.GetBonds():
        other = bond.GetOtherAtom(atom)
        if other.GetSymbol() == "C" and str(bond.GetBondType()) == "TRIPLE":
            return True
    return False


def carbon_subclass(ch_atom: Chem.Atom, partner: Chem.Atom) -> str:
    if partner.GetSymbol() != "C":
        return "C-H carbon functionalization"
    if has_double_to(partner, "O"):
        return "C-H acylation/carbonylation"
    if has_triple_to(partner, "N"):
        return "C-H cyanation"
    if has_triple_to_carbon(partner):
        return "C-H alkynylation"
    if partner.GetIsAromatic():
        return "C-H arylation"
    if has_double_to_carbon(partner):
        return "C-H alkenylation/olefination"
    if ch_atom.GetIsAromatic() and partner.GetHybridization().name in {"SP2", "SP"}:
        return "C-H alkenylation/olefination"
    return "C-H alkylation"


def subclass_from_pair(
    p_mol: Chem.Mol,
    p_atoms: Dict[int, Dict[str, object]],
    p_atom_objs: Dict[int, Chem.Atom],
    r_atoms: Dict[int, Dict[str, object]],
    pair: Tuple[int, int],
) -> Tuple[str, bool, str]:
    a, b = pair
    a_info = p_atoms.get(a)
    b_info = p_atoms.get(b)
    a_obj = p_atom_objs.get(a)
    b_obj = p_atom_objs.get(b)
    if not a_info or not b_info or not a_obj or not b_obj:
        return "unclassified", False, "missing mapped atom"

    h_loss_maps = []
    for amap, p_info in ((a, a_info), (b, b_info)):
        r_info = r_atoms.get(amap)
        if not r_info:
            continue
        if p_info["symbol"] != "C":
            continue
        try:
            if int(r_info["total_h"]) > int(p_info["total_h"]):
                h_loss_maps.append(amap)
        except Exception:
            pass

    symbols = {str(a_info["symbol"]), str(b_info["symbol"])}
    ch_like = bool(h_loss_maps)

    if ch_like:
        # Prefer the atom that clearly lost H as the activated C-H site.
        ch_map = h_loss_maps[0]
        ch_atom = p_atom_objs[ch_map]
        partner = b_obj if ch_map == a else a_obj
        partner_symbol = partner.GetSymbol()
        if partner_symbol == "C":
            return carbon_subclass(ch_atom, partner), True, ""
        if partner_symbol == "B":
            return "C-H borylation", True, ""
        if partner_symbol == "Si":
            return "C-H silylation", True, ""
        if partner_symbol == "N":
            return "C-H amination/amidation", True, ""
        if partner_symbol == "O":
            return "C-H oxygenation", True, ""
        if partner_symbol in HALOGENS:
            return "C-H halogenation", True, ""
        if partner_symbol == "S":
            return "C-H thiolation/sulfur functionalization", True, ""
        if partner_symbol == "P":
            return "C-H phosphination", True, ""
        return f"C-H functionalization to {partner_symbol}", True, ""

    if symbols == {"C"}:
        if bool(a_info["is_aromatic"]) or bool(b_info["is_aromatic"]):
            return "review: C-C aryl/aryl-like coupling but C-H loss not proven", False, ""
        return "review: C-C coupling but C-H loss not proven", False, ""
    if "C" in symbols and symbols & HALOGENS:
        return "review: C-halogen formation but C-H loss not proven", False, ""
    if "C" in symbols:
        other = sorted(symbols - {"C"})
        return f"review: C-{other[0] if other else 'X'} formation but C-H loss not proven", False, ""
    return "exclude: non-carbon bond formation", False, ""


def canonical_rxn_core(rxn: str) -> str:
    parts = split_rxn(rxn)
    if not parts:
        return rxn.strip()
    left, _mid, right = parts

    def canon_side(side: str) -> str:
        mols = []
        for smi in side.split("."):
            if not smi:
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                mols.append(smi)
                continue
            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(0)
            mols.append(Chem.MolToSmiles(mol, canonical=True))
        return ".".join(sorted(mols))

    try:
        return f"{canon_side(left)}>>{canon_side(right)}"
    except Exception:
        return rxn.strip()


def classify_mapped_rxn(rxn: str) -> Dict[str, object]:
    parts = split_rxn(rxn)
    result: Dict[str, object] = {
        "parse_ok": False,
        "reaction_subclass": "unparsed",
        "subclass_confidence": "exclude",
        "needs_review": True,
        "formed_bond_count": 0,
        "formed_bond_subclasses": "",
        "likely_c_h_functionalization": False,
        "ring_count_delta": "",
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
    p_atom_objs = atom_by_map(p_mol)
    r_bonds = bond_info(r_mol)
    p_bonds = bond_info(p_mol)
    formed = [pair for pair in p_bonds if pair not in r_bonds]
    reactive_formed = [pair for pair in formed if pair[0] in r_atoms or pair[1] in r_atoms]

    subclasses = []
    ch_like = False
    notes = []
    for pair in reactive_formed:
        subclass, is_ch, note = subclass_from_pair(p_mol, p_atoms, p_atom_objs, r_atoms, pair)
        subclasses.append(subclass)
        ch_like = ch_like or is_ch
        if note:
            notes.append(note)

    ring_delta = p_mol.GetRingInfo().NumRings() - r_mol.GetRingInfo().NumRings()
    counts = Counter(subclasses)
    result["parse_ok"] = True
    result["formed_bond_count"] = len(reactive_formed)
    result["formed_bond_subclasses"] = "; ".join(f"{k} x{v}" for k, v in counts.items())
    result["likely_c_h_functionalization"] = ch_like
    result["ring_count_delta"] = ring_delta
    result["notes"] = "; ".join(notes)

    if ch_like and len(reactive_formed) >= 2 and ring_delta > 0:
        result["reaction_subclass"] = "C-H annulation"
        result["subclass_confidence"] = "high"
        result["needs_review"] = False
        return result

    if counts:
        main, _ = counts.most_common(1)[0]
        result["reaction_subclass"] = main
        if main.startswith("C-H "):
            result["subclass_confidence"] = "high"
            result["needs_review"] = False
        elif main.startswith("review:"):
            result["subclass_confidence"] = "review"
            result["needs_review"] = True
        else:
            result["subclass_confidence"] = "exclude"
            result["needs_review"] = True
    else:
        result["reaction_subclass"] = "exclude: no detected mapped bond formation"
        result["subclass_confidence"] = "exclude"
        result["needs_review"] = True
    return result


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def iter_rows(path: Path) -> Iterable[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        yield from csv.DictReader(f)


def process_csv(
    input_path: Path,
    output_dir: Path,
    source_id: str,
    source_label: str,
    rxn_col: str,
    confident_col: str = "",
    confident_only: bool = True,
    max_rows: int = 0,
    progress_every: int = 10000,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = output_dir / f"{source_id}_ch_activation_candidates_with_subclass.csv"
    review_path = output_dir / f"{source_id}_needs_review.csv"
    excluded_path = output_dir / f"{source_id}_excluded_summary.csv"

    candidate_fields = [
        "source_id",
        "source_label",
        "source_row",
        "mapped_rxn",
        "reaction_core_key",
        "reaction_subclass",
        "subclass_confidence",
        "needs_review",
        "formed_bond_count",
        "formed_bond_subclasses",
        "likely_c_h_functionalization",
        "ring_count_delta",
        "confident_mapping",
    ]
    summary_fields = ["source_row", "reaction_subclass", "subclass_confidence", "notes"]

    counters = Counter()
    subclass_counts = Counter()
    confidence_counts = Counter()
    started = time.time()

    with candidates_path.open("w", encoding="utf-8", newline="") as cand_f, review_path.open(
        "w", encoding="utf-8", newline=""
    ) as rev_f, excluded_path.open("w", encoding="utf-8", newline="") as exc_f:
        cand_writer = csv.DictWriter(cand_f, fieldnames=candidate_fields)
        rev_writer = csv.DictWriter(rev_f, fieldnames=candidate_fields)
        exc_writer = csv.DictWriter(exc_f, fieldnames=summary_fields)
        cand_writer.writeheader()
        rev_writer.writeheader()
        exc_writer.writeheader()

        for idx, row in enumerate(iter_rows(input_path), start=1):
            if max_rows and idx > max_rows:
                break
            counters["read_rows"] += 1
            if confident_col:
                confident = truthy(row.get(confident_col, ""))
                if confident_only and not confident:
                    counters["skipped_unconfident_mapping"] += 1
                    continue
            else:
                confident = ""

            rxn = row.get(rxn_col, "")
            analysis = classify_mapped_rxn(rxn)
            subclass = str(analysis["reaction_subclass"])
            confidence = str(analysis["subclass_confidence"])
            subclass_counts[subclass] += 1
            confidence_counts[confidence] += 1

            if confidence in {"high", "review"}:
                out = {
                    "source_id": source_id,
                    "source_label": source_label,
                    "source_row": idx,
                    "mapped_rxn": rxn,
                    "reaction_core_key": canonical_rxn_core(rxn),
                    "reaction_subclass": subclass,
                    "subclass_confidence": confidence,
                    "needs_review": analysis["needs_review"],
                    "formed_bond_count": analysis["formed_bond_count"],
                    "formed_bond_subclasses": analysis["formed_bond_subclasses"],
                    "likely_c_h_functionalization": analysis["likely_c_h_functionalization"],
                    "ring_count_delta": analysis["ring_count_delta"],
                    "confident_mapping": confident,
                }
                if confidence == "high":
                    cand_writer.writerow(out)
                    counters["high_confidence_candidates"] += 1
                else:
                    rev_writer.writerow(out)
                    counters["review_candidates"] += 1
            else:
                counters["excluded"] += 1
                if counters["excluded"] <= 5000:
                    exc_writer.writerow(
                        {
                            "source_row": idx,
                            "reaction_subclass": subclass,
                            "subclass_confidence": confidence,
                            "notes": analysis.get("notes", ""),
                        }
                    )

            if progress_every and counters["read_rows"] % progress_every == 0:
                elapsed = time.time() - started
                rate = counters["read_rows"] / elapsed if elapsed else 0
                print(
                    f"{source_id}: read={counters['read_rows']} high={counters['high_confidence_candidates']} "
                    f"review={counters['review_candidates']} excluded={counters['excluded']} rate={rate:.1f}/s",
                    flush=True,
                )

    summary = {
        "source_id": source_id,
        "source_label": source_label,
        "input_path": str(input_path),
        "rxn_col": rxn_col,
        "confident_col": confident_col,
        "confident_only": confident_only,
        "max_rows": max_rows,
        "counts": dict(counters),
        "subclass_counts": dict(subclass_counts.most_common()),
        "confidence_counts": dict(confidence_counts.most_common()),
        "outputs": {
            "high_confidence_candidates": str(candidates_path),
            "needs_review": str(review_path),
            "excluded_summary": str(excluded_path),
        },
    }
    summary_path = output_dir / f"{source_id}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--source-label", required=True)
    parser.add_argument("--rxn-col", required=True)
    parser.add_argument("--confident-col", default="")
    parser.add_argument("--include-unconfident", action="store_true")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=10000)
    args = parser.parse_args()

    process_csv(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        source_id=args.source_id,
        source_label=args.source_label,
        rxn_col=args.rxn_col,
        confident_col=args.confident_col,
        confident_only=not args.include_unconfident,
        max_rows=args.max_rows,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
