from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict

from rdkit import Chem
from rdkit import RDLogger

from dataset_structural_mining import canonical_rxn_core, classify_mapped_rxn, split_rxn


RDLogger.DisableLog("rdApp.*")
try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2**31 - 1)

ROOT = Path(__file__).resolve().parent
INPUT = (
    ROOT
    / "ch_activation_mining_outputs"
    / "COMBINED_LOCAL_USPTO_ORD_CRD_RXNGRAPHORMER_BY_SUBCLASS"
    / "local_uspto_ord_crd_rxngraphormer_ch_activation_high_confidence_unique.csv"
)
GENERATED_MAPS = (
    ROOT
    / "ch_activation_cleaning_outputs"
    / "step1_unmapped_rxnmapper"
    / "unmapped_rxnmapper_results.csv"
)
ORIGINAL_DIR = ROOT / "C-H Arylation" / "C-H Arylation"
ORIGINAL_FILES = [
    ORIGINAL_DIR / "canonicalized_train.csv",
    ORIGINAL_DIR / "canonicalized_valid.csv",
    ORIGINAL_DIR / "canonicalized_test.csv",
]
OUT_DIR = ROOT / "ch_activation_cleaning_outputs" / "final_cleaned_ch_activation"

GENERATED_MAPPING_MIN_CONFIDENCE = 0.50

SUBCLASS_RENAME = {
    "C-H functionalization to Sn": "C-H stannylation",
    "C-H functionalization to Se": "C-H selenylation",
}


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        yield from csv.DictReader(f)


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def canonical_unmapped_side(side: str) -> str:
    pieces = []
    for smi in side.split("."):
        smi = smi.strip()
        if not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            pieces.append(smi)
            continue
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        pieces.append(Chem.MolToSmiles(mol, canonical=True))
    return ".".join(sorted(pieces))


def canonical_unmapped_rxn_from_any(rxn: str) -> str:
    parts = split_rxn(rxn)
    if not parts:
        return rxn.strip()
    left, _middle, right = parts
    try:
        return f"{canonical_unmapped_side(left)}>>{canonical_unmapped_side(right)}"
    except Exception:
        return canonical_rxn_core(rxn)


def mapped_product_coverage(mapped_rxn: str) -> float:
    parts = split_rxn(mapped_rxn)
    if not parts:
        return 0.0
    _left, _middle, right = parts
    mol = Chem.MolFromSmiles(right)
    if mol is None or mol.GetNumAtoms() == 0:
        return 0.0
    mapped = sum(1 for atom in mol.GetAtoms() if atom.GetAtomMapNum())
    return mapped / mol.GetNumAtoms()


def load_generated_maps() -> Dict[str, dict[str, str]]:
    out: Dict[str, dict[str, str]] = {}
    if not GENERATED_MAPS.exists():
        return out
    for row in read_csv(GENERATED_MAPS):
        out[row["global_id"]] = row
    return out


def load_original_keys() -> set[str]:
    keys: set[str] = set()
    for path in ORIGINAL_FILES:
        if not path.exists():
            continue
        for row in read_csv(path):
            rxn = row.get("reactants>reagents>production") or row.get("mapped_rxn") or ""
            if rxn:
                keys.add(canonical_unmapped_rxn_from_any(rxn))
            mapped = row.get("mapped_rxn", "")
            if mapped:
                keys.add(canonical_unmapped_rxn_from_any(mapped))
    return keys


def normalise_subclass(label: str) -> str:
    return SUBCLASS_RENAME.get(label, label)


def build_row(
    row: dict[str, str],
    generated: dict[str, str] | None,
    original_keys: set[str],
    seen_keys: set[str],
) -> tuple[str, dict[str, object]]:
    source_subclass = normalise_subclass(row.get("reaction_subclass", ""))
    mapped_rxn = row.get("mapped_rxn", "").strip()
    mapping_source = "existing_mapped_rxn"
    rxnmapper_confidence = ""
    if not mapped_rxn and generated:
        mapped_rxn = generated.get("mapped_rxn_generated", "").strip()
        mapping_source = "rxnmapper_generated"
        rxnmapper_confidence = generated.get("rxnmapper_confidence", "")

    raw_rxn = row.get("reaction_smiles", "").strip() or row.get("reaction_core_key", "").strip()
    canonical_key = canonical_unmapped_rxn_from_any(mapped_rxn or raw_rxn)
    analysis = classify_mapped_rxn(mapped_rxn)
    computed_subclass = normalise_subclass(str(analysis.get("reaction_subclass", "")))
    product_coverage = mapped_product_coverage(mapped_rxn)

    exclusion_reasons = []
    review_reasons = []
    if not mapped_rxn:
        exclusion_reasons.append("missing_mapped_rxn")
    if not analysis.get("parse_ok"):
        exclusion_reasons.append("mapped_rxn_parse_failed")
    if product_coverage < 0.95:
        exclusion_reasons.append("low_product_atom_map_coverage")
    if not bool(analysis.get("likely_c_h_functionalization")):
        exclusion_reasons.append("no_c_h_loss_detected_after_mapping_check")
    if not computed_subclass.startswith("C-H "):
        exclusion_reasons.append("computed_subclass_not_c_h")
    if mapping_source == "rxnmapper_generated":
        try:
            if float(rxnmapper_confidence) < GENERATED_MAPPING_MIN_CONFIDENCE:
                review_reasons.append("generated_mapping_confidence_below_0_50")
        except Exception:
            review_reasons.append("generated_mapping_confidence_missing")
    if canonical_key in original_keys:
        exclusion_reasons.append("duplicate_with_original_c_h_arylation_dataset")
    if canonical_key in seen_keys:
        exclusion_reasons.append("duplicate_within_cleaning_run")

    if computed_subclass.startswith("C-H "):
        final_subclass = computed_subclass
    else:
        final_subclass = source_subclass

    if source_subclass and computed_subclass.startswith("C-H ") and source_subclass != computed_subclass:
        review_reasons.append(f"source_subclass_differs_from_mapping_check:{source_subclass}->{computed_subclass}")

    status = "accepted"
    if exclusion_reasons:
        status = "excluded"
    elif review_reasons:
        status = "review"

    out = {
        "global_id": row.get("global_id", ""),
        "source_id": row.get("source_id", ""),
        "source_label": row.get("source_label", ""),
        "source_row": row.get("source_row", ""),
        "source_subclass": source_subclass,
        "computed_subclass": computed_subclass,
        "reaction_subclass": final_subclass,
        "status": status,
        "review_reasons": ";".join(review_reasons),
        "exclusion_reasons": ";".join(exclusion_reasons),
        "mapped_rxn": mapped_rxn,
        "reactants>reagents>production": canonical_key,
        "canonical_reaction_core": canonical_key,
        "mapping_source": mapping_source,
        "rxnmapper_confidence": rxnmapper_confidence,
        "product_atom_map_coverage": f"{product_coverage:.4f}",
        "formed_bond_count": analysis.get("formed_bond_count", ""),
        "formed_bond_subclasses": analysis.get("formed_bond_subclasses", ""),
        "likely_c_h_functionalization": analysis.get("likely_c_h_functionalization", ""),
        "ring_count_delta": analysis.get("ring_count_delta", ""),
        "original_reaction_smiles": row.get("reaction_smiles", ""),
        "source_file": row.get("source_file", ""),
    }
    return status, out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    generated_maps = load_generated_maps()
    original_keys = load_original_keys()
    seen_keys: set[str] = set()

    accepted: list[dict[str, object]] = []
    review: list[dict[str, object]] = []
    excluded: list[dict[str, object]] = []
    duplicate_original: list[dict[str, object]] = []

    for row in read_csv(INPUT):
        generated = generated_maps.get(row.get("global_id", ""))
        status, out = build_row(row, generated, original_keys, seen_keys)
        if status == "accepted":
            seen_keys.add(str(out["canonical_reaction_core"]))
            accepted.append(out)
        elif status == "review":
            seen_keys.add(str(out["canonical_reaction_core"]))
            review.append(out)
        else:
            if "duplicate_with_original_c_h_arylation_dataset" in str(out["exclusion_reasons"]):
                duplicate_original.append(out)
            excluded.append(out)

    provenance_fields = [
        "final_id",
        "global_id",
        "source_id",
        "source_label",
        "source_row",
        "reaction_subclass",
        "source_subclass",
        "computed_subclass",
        "mapped_rxn",
        "reactants>reagents>production",
        "canonical_reaction_core",
        "mapping_source",
        "rxnmapper_confidence",
        "product_atom_map_coverage",
        "formed_bond_count",
        "formed_bond_subclasses",
        "likely_c_h_functionalization",
        "ring_count_delta",
        "review_reasons",
        "exclusion_reasons",
        "original_reaction_smiles",
        "source_file",
    ]
    minimal_fields = [
        "id",
        "class",
        "mapped_rxn",
        "reactants>reagents>production",
        "catalyst",
        "T",
        "reaction_subclass",
    ]

    for i, row in enumerate(accepted, start=1):
        row["final_id"] = f"clean_ch_act_{i}"
    for i, row in enumerate(review, start=1):
        row["final_id"] = f"review_ch_act_{i}"
    for i, row in enumerate(excluded, start=1):
        row["final_id"] = f"excluded_ch_act_{i}"

    minimal_rows = [
        {
            "id": row["final_id"],
            "class": 1,
            "mapped_rxn": row["mapped_rxn"],
            "reactants>reagents>production": row["reactants>reagents>production"],
            "catalyst": "",
            "T": "",
            "reaction_subclass": row["reaction_subclass"],
        }
        for row in accepted
    ]

    high_minimal = OUT_DIR / "final_ch_activation_cleaned_high_confidence_minimal.csv"
    high_prov = OUT_DIR / "final_ch_activation_cleaned_high_confidence_with_provenance.csv"
    review_path = OUT_DIR / "final_ch_activation_needs_review.csv"
    excluded_path = OUT_DIR / "final_ch_activation_excluded.csv"
    dup_original_path = OUT_DIR / "final_ch_activation_duplicates_with_original_c_h_arylation.csv"

    write_csv(high_minimal, minimal_rows, minimal_fields)
    write_csv(high_prov, accepted, provenance_fields)
    write_csv(review_path, review, provenance_fields)
    write_csv(excluded_path, excluded, provenance_fields)
    write_csv(dup_original_path, duplicate_original, provenance_fields)

    arylation_rows = [row for row in minimal_rows if row["reaction_subclass"] == "C-H arylation"]
    arylation_path = OUT_DIR / "final_ch_arylation_only_cleaned_high_confidence_minimal.csv"
    write_csv(arylation_path, arylation_rows, minimal_fields)

    by_subclass_dir = OUT_DIR / "by_subclass_cleaned_high_confidence"
    by_subclass_dir.mkdir(exist_ok=True)
    by_subclass_paths = {}
    for subclass in sorted({str(row["reaction_subclass"]) for row in accepted}):
        subset = [row for row in minimal_rows if row["reaction_subclass"] == subclass]
        fname = (
            subclass.replace("C-H ", "ch_")
            .replace("/", "_")
            .replace(" ", "_")
            .replace("-", "_")
            .lower()
        )
        path = by_subclass_dir / f"{fname}.csv"
        write_csv(path, subset, minimal_fields)
        by_subclass_paths[subclass] = str(path)

    summary = {
        "input_file": str(INPUT),
        "generated_mapping_file": str(GENERATED_MAPS),
        "original_c_h_arylation_files": [str(path) for path in ORIGINAL_FILES],
        "generated_mapping_min_confidence": GENERATED_MAPPING_MIN_CONFIDENCE,
        "input_rows": len(accepted) + len(review) + len(excluded),
        "accepted_rows": len(accepted),
        "review_rows": len(review),
        "excluded_rows": len(excluded),
        "duplicates_with_original_c_h_arylation": len(duplicate_original),
        "accepted_subclass_counts": dict(Counter(row["reaction_subclass"] for row in accepted).most_common()),
        "review_reason_counts": dict(Counter(r for row in review for r in str(row["review_reasons"]).split(";") if r).most_common()),
        "exclusion_reason_counts": dict(Counter(r for row in excluded for r in str(row["exclusion_reasons"]).split(";") if r).most_common()),
        "accepted_source_counts": dict(Counter(row["source_id"] for row in accepted).most_common()),
        "review_source_counts": dict(Counter(row["source_id"] for row in review).most_common()),
        "excluded_source_counts": dict(Counter(row["source_id"] for row in excluded).most_common()),
        "outputs": {
            "final_minimal": str(high_minimal),
            "final_with_provenance": str(high_prov),
            "final_arylation_only_minimal": str(arylation_path),
            "needs_review": str(review_path),
            "excluded": str(excluded_path),
            "duplicates_with_original": str(dup_original_path),
            "by_subclass_dir": str(by_subclass_dir),
            "by_subclass_files": by_subclass_paths,
        },
    }
    summary_path = OUT_DIR / "final_ch_activation_cleaning_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    readme = [
        "# Cleaned C-H Activation / Functionalization Dataset",
        "",
        f"Input rows: {summary['input_rows']}",
        f"Accepted rows: {summary['accepted_rows']}",
        f"Review rows: {summary['review_rows']}",
        f"Excluded rows: {summary['excluded_rows']}",
        f"Duplicates with original C-H Arylation dataset: {summary['duplicates_with_original_c_h_arylation']}",
        "",
        "## Final Files",
        "",
        f"- Final minimal dataset: `{high_minimal}`",
        f"- Final dataset with provenance: `{high_prov}`",
        f"- C-H arylation only minimal dataset: `{arylation_path}`",
        f"- Needs-review dataset: `{review_path}`",
        f"- Excluded dataset: `{excluded_path}`",
        f"- By-subclass folder: `{by_subclass_dir}`",
        "",
        "## Accepted Subclass Counts",
        "",
        "| subclass | rows |",
        "|---|---:|",
    ]
    for subclass, count in Counter(row["reaction_subclass"] for row in accepted).most_common():
        readme.append(f"| {subclass} | {count} |")
    (OUT_DIR / "FINAL_C_H_ACTIVATION_CLEANED_README.md").write_text(
        "\n".join(readme) + "\n", encoding="utf-8"
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
