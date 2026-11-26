from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "ch_activation_cleaning_outputs" / "final_cleaned_ch_activation"
INPUT_PROVENANCE = INPUT_DIR / "final_ch_activation_cleaned_high_confidence_with_provenance.csv"
OUT_DIR = ROOT / "ch_activation_cleaning_outputs" / "final_cleaned_ch_activation_single_center_filtered"

DROP_SUBCLASSES = {
    "C-H phosphination",
    "C-H selenylation",
}

SINGLE_CENTER_ONLY_SUBCLASSES = {
    "C-H thiolation/sulfur functionalization",
    "C-H cyanation",
    "C-H alkynylation",
    "C-H stannylation",
    "C-H borylation",
    "C-H silylation",
}


def safe_name(label: str) -> str:
    return (
        label.replace("C-H ", "ch_")
        .replace("/", "_")
        .replace(" ", "_")
        .replace("-", "_")
        .lower()
    )


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        yield from csv.DictReader(f)


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def removal_reason(row: dict[str, str]) -> str:
    subclass = row.get("reaction_subclass", "")
    if subclass in DROP_SUBCLASSES:
        return "drop_low_count_subclass"
    if subclass in SINGLE_CENTER_ONLY_SUBCLASSES and row.get("formed_bond_count", "") != "1":
        return "drop_multicenter_or_multi_bond_in_small_subclass"
    return ""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    kept: list[dict[str, object]] = []
    removed: list[dict[str, object]] = []
    input_rows = 0
    for row in read_csv(INPUT_PROVENANCE):
        input_rows += 1
        reason = removal_reason(row)
        if reason:
            out = dict(row)
            out["filter_removal_reason"] = reason
            removed.append(out)
        else:
            kept.append(dict(row))

    for i, row in enumerate(kept, start=1):
        row["final_id"] = f"clean_ch_act_sc_{i}"

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
    removed_fields = provenance_fields + ["filter_removal_reason"]

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
        for row in kept
    ]

    final_minimal = OUT_DIR / "final_ch_activation_single_center_filtered_minimal.csv"
    final_provenance = OUT_DIR / "final_ch_activation_single_center_filtered_with_provenance.csv"
    removed_path = OUT_DIR / "final_ch_activation_single_center_filter_removed.csv"
    write_csv(final_minimal, minimal_rows, minimal_fields)
    write_csv(final_provenance, kept, provenance_fields)
    write_csv(removed_path, removed, removed_fields)

    arylation_rows = [row for row in minimal_rows if row["reaction_subclass"] == "C-H arylation"]
    arylation_path = OUT_DIR / "final_ch_arylation_only_single_center_filtered_minimal.csv"
    write_csv(arylation_path, arylation_rows, minimal_fields)

    by_subclass_dir = OUT_DIR / "by_subclass_single_center_filtered"
    by_subclass_dir.mkdir(exist_ok=True)
    by_subclass_paths = {}
    for subclass in sorted({str(row["reaction_subclass"]) for row in kept}):
        subset = [row for row in minimal_rows if row["reaction_subclass"] == subclass]
        path = by_subclass_dir / f"{safe_name(subclass)}.csv"
        write_csv(path, subset, minimal_fields)
        by_subclass_paths[subclass] = str(path)

    summary = {
        "input_provenance": str(INPUT_PROVENANCE),
        "input_rows": input_rows,
        "kept_rows": len(kept),
        "removed_rows": len(removed),
        "drop_subclasses": sorted(DROP_SUBCLASSES),
        "single_center_only_subclasses": sorted(SINGLE_CENTER_ONLY_SUBCLASSES),
        "kept_subclass_counts": dict(Counter(row["reaction_subclass"] for row in kept).most_common()),
        "removed_subclass_counts": dict(Counter(row["reaction_subclass"] for row in removed).most_common()),
        "removed_reason_counts": dict(Counter(row["filter_removal_reason"] for row in removed).most_common()),
        "outputs": {
            "final_minimal": str(final_minimal),
            "final_with_provenance": str(final_provenance),
            "removed": str(removed_path),
            "arylation_only_minimal": str(arylation_path),
            "by_subclass_dir": str(by_subclass_dir),
            "by_subclass_files": by_subclass_paths,
        },
    }
    summary_path = OUT_DIR / "final_ch_activation_single_center_filter_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    readme = [
        "# C-H Activation Single-Center Filtered Dataset",
        "",
        f"Input accepted rows: {input_rows}",
        f"Kept rows: {len(kept)}",
        f"Removed rows: {len(removed)}",
        "",
        "## Rules",
        "",
        "- Dropped C-H phosphination and C-H selenylation entirely.",
        "- Kept only formed_bond_count = 1 for C-H thiolation, cyanation, alkynylation, stannylation, borylation, and silylation.",
        "",
        "## Final Files",
        "",
        f"- Final minimal dataset: `{final_minimal}`",
        f"- Final dataset with provenance: `{final_provenance}`",
        f"- Removed rows: `{removed_path}`",
        f"- By-subclass folder: `{by_subclass_dir}`",
        "",
        "## Kept Subclass Counts",
        "",
        "| subclass | rows |",
        "|---|---:|",
    ]
    for subclass, count in Counter(row["reaction_subclass"] for row in kept).most_common():
        readme.append(f"| {subclass} | {count} |")
    (OUT_DIR / "FINAL_C_H_ACTIVATION_SINGLE_CENTER_FILTERED_README.md").write_text(
        "\n".join(readme) + "\n", encoding="utf-8"
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
