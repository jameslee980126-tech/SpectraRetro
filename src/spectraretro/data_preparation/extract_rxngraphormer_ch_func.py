from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

from rdkit import Chem
from rdkit import RDLogger

from mine_mapped_ch_activation import canonical_rxn_core


RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "external_sources" / "RXNGraphormer" / "figshare" / "benmark_dataset"
OUT_DIR = ROOT / "ch_activation_mining_outputs" / "RXNGraphormer_C_H_func"

RCT_FILE = DATA_DIR / "C_H_func_rct_tot.csv"
PDT_FILE = DATA_DIR / "C_H_func_pdt_tot.csv"
MECH_FILE = DATA_DIR / "C_H_func_mech_tot.csv"


def mol_is_aromatic_carbon_radical_like(smiles: str) -> bool:
    if "[c]" in smiles:
        return True
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return bool(re.search(r"\bc1|c1", smiles))
    return any(atom.GetSymbol() == "C" and atom.GetIsAromatic() for atom in mol.GetAtoms())


def classify_reactant_fragments(rct: str) -> str:
    frags = [frag for frag in rct.split(".") if frag]
    # In this benchmark the non-substrate radical/partner fragment determines
    # whether the C-H functionalization installs aryl/heteroaryl or alkyl groups.
    radical_like = [frag for frag in frags if "[" in frag and "]" in frag]
    partner_pool = radical_like or frags[1:]
    if any(mol_is_aromatic_carbon_radical_like(frag) for frag in partner_pool):
        return "C-H arylation"
    return "C-H alkylation"


def read_two_col(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            rows.append((row[0], row[1] if len(row) > 1 else ""))
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rct_rows = read_two_col(RCT_FILE)
    pdt_rows = read_two_col(PDT_FILE)
    mech_rows = read_two_col(MECH_FILE)
    if not (len(rct_rows) == len(pdt_rows) == len(mech_rows)):
        raise ValueError((len(rct_rows), len(pdt_rows), len(mech_rows)))

    rows: list[dict[str, object]] = []
    for idx, ((rct, y1), (pdt, y2), (mech, y3)) in enumerate(zip(rct_rows, pdt_rows, mech_rows), start=1):
        yield_value = y1 or y2 or y3
        rxn = f"{rct}>>{pdt}"
        rows.append(
            {
                "source_id": "RXNGraphormer_C_H_func",
                "source_label": "RXNGraphormer benchmark C_H_func from figshare benmark_dataset.zip",
                "source_row": idx,
                "reaction_smiles": rxn,
                "reaction_core_key": canonical_rxn_core(rxn),
                "reaction_subclass": classify_reactant_fragments(rct),
                "subclass_confidence": "high_by_benchmark_metadata",
                "needs_review": False,
                "yield_or_label": yield_value,
                "rct_smiles": rct,
                "pdt_smiles": pdt,
                "mech_smiles": mech,
                "source_file": str(RCT_FILE),
            }
        )

    seen = set()
    unique_rows = []
    duplicate_rows = []
    for row in rows:
        key = row["reaction_core_key"]
        if key in seen:
            duplicate_rows.append(row)
            continue
        seen.add(key)
        unique_rows.append(row)

    fields = [
        "source_id",
        "source_label",
        "source_row",
        "reaction_smiles",
        "reaction_core_key",
        "reaction_subclass",
        "subclass_confidence",
        "needs_review",
        "yield_or_label",
        "rct_smiles",
        "pdt_smiles",
        "mech_smiles",
        "source_file",
    ]
    all_path = OUT_DIR / "rxngraphormer_c_h_func_all_rows.csv"
    unique_path = OUT_DIR / "rxngraphormer_c_h_func_unique_reaction_cores.csv"
    duplicate_path = OUT_DIR / "rxngraphormer_c_h_func_duplicate_rows.csv"
    write_csv(all_path, rows, fields)
    write_csv(unique_path, unique_rows, fields)
    write_csv(duplicate_path, duplicate_rows, fields)

    by_subclass_dir = OUT_DIR / "by_subclass"
    by_subclass_dir.mkdir(exist_ok=True)
    by_subclass_paths = {}
    for subclass in sorted({row["reaction_subclass"] for row in unique_rows}):
        subset = [row for row in unique_rows if row["reaction_subclass"] == subclass]
        name = subclass.lower().replace("c-h ", "ch_").replace("/", "_").replace(" ", "_")
        path = by_subclass_dir / f"{name}.csv"
        write_csv(path, subset, fields)
        by_subclass_paths[subclass] = str(path)

    summary = {
        "raw_rows": len(rows),
        "unique_reaction_cores": len(unique_rows),
        "duplicates_removed": len(duplicate_rows),
        "subclass_counts_raw": dict(Counter(row["reaction_subclass"] for row in rows).most_common()),
        "subclass_counts_unique": dict(Counter(row["reaction_subclass"] for row in unique_rows).most_common()),
        "inputs": {
            "rct": str(RCT_FILE),
            "pdt": str(PDT_FILE),
            "mech": str(MECH_FILE),
        },
        "outputs": {
            "all_rows": str(all_path),
            "unique_reaction_cores": str(unique_path),
            "duplicates_removed": str(duplicate_path),
            "by_subclass_dir": str(by_subclass_dir),
            "by_subclass_files": by_subclass_paths,
        },
    }
    summary_path = OUT_DIR / "rxngraphormer_c_h_func_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
