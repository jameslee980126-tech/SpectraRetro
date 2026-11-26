from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


csv.field_size_limit(2**31 - 1)

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "ch_activation_mining_outputs" / "COMBINED_LOCAL_USPTO_ORD_CRD_RXNGRAPHORMER_BY_SUBCLASS"
USPTO_HIGH = (
    ROOT
    / "ch_activation_mining_outputs"
    / "USPTO_FULL_LOCALMAPPER_full"
    / "USPTO_FULL_LOCALMAPPER_full_ch_activation_candidates_with_subclass.csv"
)
USPTO_REVIEW = (
    ROOT
    / "ch_activation_mining_outputs"
    / "USPTO_FULL_LOCALMAPPER_full"
    / "USPTO_FULL_LOCALMAPPER_full_needs_review.csv"
)
ORD_UNIQUE = (
    ROOT
    / "ch_activation_mining_outputs"
    / "ORD_selected_ch_activation"
    / "ord_selected_ch_activation_unique_reaction_cores.csv"
)
CRD_HIGH = (
    ROOT
    / "ch_activation_mining_outputs"
    / "CRD_MappedCRD_conf05_full"
    / "CRD_MappedCRD_rxnmapper_conf05_ch_activation_candidates_with_subclass.csv"
)
CRD_REVIEW = (
    ROOT
    / "ch_activation_mining_outputs"
    / "CRD_MappedCRD_conf05_full"
    / "CRD_MappedCRD_rxnmapper_conf05_needs_review.csv"
)
RXNG_HIGH = (
    ROOT
    / "ch_activation_mining_outputs"
    / "RXNGraphormer_C_H_func"
    / "rxngraphormer_c_h_func_unique_reaction_cores.csv"
)

HIGH_CONFIDENCE_VALUES = {"high", "high_by_dataset_metadata"}
SUBCLASS_RENAME = {
    "C-H functionalization to Sn": "C-H stannylation",
    "C-H functionalization to Se": "C-H selenylation",
}
RECLASSIFY_TO_REVIEW = {
    "C-H functionalization to H",
    "C-H functionalization to Li",
    "C-H functionalization to Mg",
    "C-H functionalization to Na",
    "C-H functionalization to K",
    "C-H functionalization to Cr",
    "C-H functionalization to Cu",
    "C-H functionalization to Zn",
    "C-H functionalization to Ge",
}


def safe_name(label: str) -> str:
    label = label.replace("C-H ", "ch_")
    label = label.replace("/", "_")
    label = label.replace(" ", "_")
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)
    return label.strip("_").lower()


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        yield from csv.DictReader(f)


def normalise_uspto(row: dict[str, str]) -> dict[str, object]:
    subclass = SUBCLASS_RENAME.get(row.get("reaction_subclass", ""), row.get("reaction_subclass", ""))
    return {
        "source_id": row.get("source_id", "USPTO_FULL_LOCALMAPPER_full"),
        "source_label": row.get("source_label", "LocalMapper remapped USPTO-FULL confident mappings"),
        "source_row": row.get("source_row", ""),
        "reaction_subclass": subclass,
        "subclass_confidence": row.get("subclass_confidence", ""),
        "needs_review": row.get("needs_review", ""),
        "reaction_core_key": row.get("reaction_core_key", ""),
        "mapped_rxn": row.get("mapped_rxn", ""),
        "reaction_smiles": "",
        "source_file": str(USPTO_HIGH),
    }


def normalise_ord(row: dict[str, str]) -> dict[str, object]:
    subclass = SUBCLASS_RENAME.get(row.get("reaction_subclass", ""), row.get("reaction_subclass", ""))
    return {
        "source_id": row.get("source_id", ""),
        "source_label": row.get("source_label", ""),
        "source_row": row.get("source_reaction_index", ""),
        "reaction_subclass": subclass,
        "subclass_confidence": row.get("subclass_confidence", ""),
        "needs_review": row.get("needs_review", ""),
        "reaction_core_key": row.get("reaction_core_key", ""),
        "mapped_rxn": "",
        "reaction_smiles": row.get("reaction_smiles", ""),
        "source_file": row.get("source_file", str(ORD_UNIQUE)),
    }


def normalise_crd(row: dict[str, str]) -> dict[str, object]:
    subclass = SUBCLASS_RENAME.get(row.get("reaction_subclass", ""), row.get("reaction_subclass", ""))
    needs_review = row.get("needs_review", "")
    confidence = row.get("subclass_confidence", "")
    if subclass in RECLASSIFY_TO_REVIEW:
        needs_review = "True"
        confidence = "review_reclassified_small_organometallic_or_mapping_artifact"
    return {
        "source_id": row.get("source_id", "CRD_MappedCRD_rxnmapper_conf05"),
        "source_label": row.get("source_label", "MappedCRD rxnmapper split from HuggingFace"),
        "source_row": row.get("source_row_global", ""),
        "reaction_subclass": subclass,
        "subclass_confidence": confidence,
        "needs_review": needs_review,
        "reaction_core_key": row.get("reaction_core_key", ""),
        "mapped_rxn": row.get("mapped_rxn", ""),
        "reaction_smiles": row.get("reaction", ""),
        "source_file": row.get("source_file", str(CRD_HIGH)),
    }


def normalise_rxngraphormer(row: dict[str, str]) -> dict[str, object]:
    subclass = SUBCLASS_RENAME.get(row.get("reaction_subclass", ""), row.get("reaction_subclass", ""))
    return {
        "source_id": row.get("source_id", "RXNGraphormer_C_H_func"),
        "source_label": row.get("source_label", "RXNGraphormer benchmark C_H_func"),
        "source_row": row.get("source_row", ""),
        "reaction_subclass": subclass,
        "subclass_confidence": row.get("subclass_confidence", "high_by_benchmark_metadata"),
        "needs_review": row.get("needs_review", "False"),
        "reaction_core_key": row.get("reaction_core_key", ""),
        "mapped_rxn": "",
        "reaction_smiles": row.get("reaction_smiles", ""),
        "source_file": row.get("source_file", str(RXNG_HIGH)),
    }


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    high_rows: list[dict[str, object]] = []
    review_rows: list[dict[str, object]] = []
    duplicates: list[dict[str, object]] = []
    seen: dict[str, dict[str, object]] = {}
    raw_counts = Counter()

    for row in read_csv(USPTO_HIGH):
        raw_counts["uspto_high_rows"] += 1
        high_rows.append(normalise_uspto(row))

    for row in read_csv(ORD_UNIQUE):
        norm = normalise_ord(row)
        if str(norm["subclass_confidence"]) in HIGH_CONFIDENCE_VALUES:
            raw_counts["ord_high_unique_rows"] += 1
            high_rows.append(norm)
        else:
            raw_counts["ord_review_unique_rows"] += 1
            review_rows.append(norm)

    for row in read_csv(CRD_HIGH):
        raw_counts["crd_high_rows"] += 1
        norm = normalise_crd(row)
        if str(norm["subclass_confidence"]).startswith("review_reclassified"):
            raw_counts["crd_reclassified_to_review_rows"] += 1
            review_rows.append(norm)
        else:
            high_rows.append(norm)

    for row in read_csv(RXNG_HIGH):
        raw_counts["rxngraphormer_high_rows"] += 1
        high_rows.append(normalise_rxngraphormer(row))

    unique_high = []
    for row in high_rows:
        key = str(row.get("reaction_core_key") or row.get("mapped_rxn") or row.get("reaction_smiles"))
        if key in seen:
            dup = dict(row)
            dup["duplicate_of_source_id"] = seen[key].get("source_id", "")
            duplicates.append(dup)
            continue
        seen[key] = row
        unique_high.append(row)

    fields = [
        "global_id",
        "source_id",
        "source_label",
        "source_row",
        "reaction_subclass",
        "subclass_confidence",
        "needs_review",
        "reaction_core_key",
        "mapped_rxn",
        "reaction_smiles",
        "source_file",
    ]
    for i, row in enumerate(unique_high, start=1):
        row["global_id"] = f"local_ch_act_{i}"
    for i, row in enumerate(review_rows, start=1):
        row["global_id"] = f"local_ch_act_review_{i}"

    high_path = OUT_DIR / "local_uspto_ord_crd_rxngraphormer_ch_activation_high_confidence_unique.csv"
    review_path = OUT_DIR / "ord_roche_and_small_reclassified_review_unique.csv"
    dup_path = OUT_DIR / "local_uspto_ord_crd_rxngraphormer_high_confidence_duplicates_removed.csv"
    write_csv(high_path, unique_high, fields)
    write_csv(review_path, review_rows, fields)
    write_csv(dup_path, duplicates, fields + ["duplicate_of_source_id"])

    by_subclass_dir = OUT_DIR / "by_subclass_high_confidence"
    by_subclass_dir.mkdir(exist_ok=True)
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in unique_high:
        grouped[str(row["reaction_subclass"])].append(row)
    by_subclass_paths = {}
    for subclass, rows in grouped.items():
        path = by_subclass_dir / f"{safe_name(subclass)}.csv"
        write_csv(path, rows, fields)
        by_subclass_paths[subclass] = str(path)

    summary = {
        "inputs": {
            "uspto_high_confidence": str(USPTO_HIGH),
            "uspto_needs_review_existing": str(USPTO_REVIEW),
            "ord_unique_reaction_cores": str(ORD_UNIQUE),
            "crd_high_confidence": str(CRD_HIGH),
            "crd_needs_review_existing": str(CRD_REVIEW),
            "rxngraphormer_c_h_func": str(RXNG_HIGH),
        },
        "raw_counts": dict(raw_counts),
        "high_confidence_unique_rows": len(unique_high),
        "high_confidence_duplicates_removed": len(duplicates),
        "ord_review_unique_rows": len(review_rows),
        "high_confidence_subclass_counts": dict(Counter(row["reaction_subclass"] for row in unique_high).most_common()),
        "high_confidence_source_counts": dict(Counter(row["source_id"] for row in unique_high).most_common()),
        "ord_review_subclass_counts": dict(Counter(row["reaction_subclass"] for row in review_rows).most_common()),
        "outputs": {
            "high_confidence_unique": str(high_path),
            "by_subclass_high_confidence_dir": str(by_subclass_dir),
            "ord_and_small_reclassified_review_unique": str(review_path),
            "uspto_needs_review_existing": str(USPTO_REVIEW),
            "crd_needs_review_existing": str(CRD_REVIEW),
            "duplicates_removed": str(dup_path),
            "by_subclass_files": by_subclass_paths,
        },
    }
    summary_path = OUT_DIR / "local_uspto_ord_crd_rxngraphormer_ch_activation_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [
        "# Local USPTO + ORD + CRD + RXNGraphormer C-H Activation Mining",
        "",
        f"High-confidence unique rows: {len(unique_high)}",
        f"High-confidence duplicates removed: {len(duplicates)}",
        f"ORD Roche borylation review unique rows: {len(review_rows)}",
        "",
        "## High-Confidence Subclass Counts",
        "",
        "| subclass | rows |",
        "|---|---:|",
    ]
    for subclass, count in Counter(row["reaction_subclass"] for row in unique_high).most_common():
        md.append(f"| {subclass} | {count} |")
    md.extend(
        [
            "",
            "## Main Files",
            "",
            f"- High-confidence unique set: `{high_path}`",
            f"- High-confidence subclass folder: `{by_subclass_dir}`",
            f"- ORD Roche borylation and small reclassified review set: `{review_path}`",
            f"- USPTO needs-review set: `{USPTO_REVIEW}`",
            f"- CRD needs-review set: `{CRD_REVIEW}`",
        ]
    )
    md_path = OUT_DIR / "LOCAL_USPTO_ORD_CRD_RXNGRAPHORMER_CH_ACTIVATION_README.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
