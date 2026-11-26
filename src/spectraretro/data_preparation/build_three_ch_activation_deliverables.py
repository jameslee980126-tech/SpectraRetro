from __future__ import annotations

import csv
import json
import re
import shutil
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
INPUT = (
    ROOT
    / "ch_activation_cleaning_outputs"
    / "final_cleaned_ch_activation_single_center_filtered"
    / "final_ch_activation_single_center_filtered_with_provenance.csv"
)
OUT_ROOT = ROOT / "C-H activation（碳氢键活化）数据集整理_三份"

COMMON_SUBCLASSES = [
    "C-H halogenation",
    "C-H alkylation",
    "C-H arylation",
    "C-H annulation",
    "C-H alkenylation/olefination",
    "C-H acylation/carbonylation",
    "C-H amination/amidation",
    "C-H oxygenation",
]

RARE_SUBCLASSES = [
    "C-H thiolation/sulfur functionalization",
    "C-H cyanation",
    "C-H stannylation",
    "C-H alkynylation",
    "C-H borylation",
    "C-H silylation",
]

CN = {
    "C-H halogenation": "卤化",
    "C-H alkylation": "烷基化",
    "C-H arylation": "芳基化",
    "C-H annulation": "环化",
    "C-H alkenylation/olefination": "烯基化",
    "C-H acylation/carbonylation": "酰基化_羰基化",
    "C-H amination/amidation": "胺化_酰胺化",
    "C-H oxygenation": "氧化_氧官能团化",
    "C-H thiolation/sulfur functionalization": "硫醚化_硫官能团化",
    "C-H cyanation": "氰基化",
    "C-H stannylation": "锡基化",
    "C-H alkynylation": "炔基化",
    "C-H borylation": "硼化",
    "C-H silylation": "硅基化",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def safe_file_name(subclass: str) -> str:
    base = CN.get(subclass, subclass)
    return re.sub(r"[\\/:*?\"<>|]+", "_", base)


def family_for(subclass: str) -> str:
    if subclass in COMMON_SUBCLASSES:
        return "common C-H activation（常规碳氢键活化）"
    if subclass in RARE_SUBCLASSES:
        return "rare C-H functionalization（稀有碳氢键官能团化）"
    return "other（其他）"


def dataset_rows(rows: list[dict[str, str]], subclasses: list[str]) -> list[dict[str, str]]:
    allowed = set(subclasses)
    return [row for row in rows if row["reaction_subclass"] in allowed]


def with_labels_rows(rows: list[dict[str, str]], prefix: str) -> list[dict[str, object]]:
    out = []
    for idx, row in enumerate(rows, start=1):
        out.append(
            {
                "id": f"{prefix}_{idx}",
                "class": 1,
                "mapped_rxn": row["mapped_rxn"],
                "reactants>reagents>production": row["reactants>reagents>production"],
                "catalyst": "",
                "T": "",
                "reaction_family": family_for(row["reaction_subclass"]),
                "reaction_subclass": row["reaction_subclass"],
                "source_id": row["source_id"],
            }
        )
    return out


def original_style_rows(rows: list[dict[str, str]], prefix: str) -> list[dict[str, object]]:
    out = []
    for idx, row in enumerate(rows, start=1):
        out.append(
            {
                "id": f"{prefix}_{idx}",
                "class": 1,
                "mapped_rxn": row["mapped_rxn"],
                "reactants>reagents>production": row["reactants>reagents>production"],
                "catalyst": "",
                "T": "",
            }
        )
    return out


def provenance_rows(rows: list[dict[str, str]], prefix: str) -> list[dict[str, object]]:
    out = []
    for idx, row in enumerate(rows, start=1):
        new_row: dict[str, object] = dict(row)
        new_row["deliverable_id"] = f"{prefix}_{idx}"
        new_row["reaction_family"] = family_for(row["reaction_subclass"])
        out.append(new_row)
    return out


def write_dataset(
    root: Path,
    rows: list[dict[str, str]],
    subclasses: list[str],
    prefix: str,
    title: str,
) -> dict[str, object]:
    root.mkdir(parents=True, exist_ok=True)
    subclass_dir = root / "按子类划分"
    subclass_dir.mkdir(exist_ok=True)

    label_fields = [
        "id",
        "class",
        "mapped_rxn",
        "reactants>reagents>production",
        "catalyst",
        "T",
        "reaction_family",
        "reaction_subclass",
        "source_id",
    ]
    original_fields = [
        "id",
        "class",
        "mapped_rxn",
        "reactants>reagents>production",
        "catalyst",
        "T",
    ]
    prov_fields = [
        "deliverable_id",
        "reaction_family",
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

    label_path = root / "总数据_with_labels（带标签）.csv"
    original_path = root / "总数据_original_style_minimal（原始格式最简）.csv"
    prov_path = root / "总数据_with_provenance（带来源追溯）.csv"
    write_csv(label_path, with_labels_rows(rows, prefix), label_fields)
    write_csv(original_path, original_style_rows(rows, prefix), original_fields)
    write_csv(prov_path, provenance_rows(rows, prefix), prov_fields)

    by_subclass_paths = {}
    for subclass in subclasses:
        subset = [row for row in rows if row["reaction_subclass"] == subclass]
        name = safe_file_name(subclass)
        path = subclass_dir / f"{name}_共{len(subset)}条_with_labels（带标签）.csv"
        write_csv(path, with_labels_rows(subset, f"{prefix}_{name}"), label_fields)
        by_subclass_paths[subclass] = str(path)

    counts = Counter(row["reaction_subclass"] for row in rows)
    summary = {
        "title": title,
        "rows": len(rows),
        "subclass_count": len(subclasses),
        "subclass_counts": {subclass: counts[subclass] for subclass in subclasses},
        "outputs": {
            "with_labels": str(label_path),
            "original_style_minimal": str(original_path),
            "with_provenance": str(prov_path),
            "by_subclass_dir": str(subclass_dir),
            "by_subclass_files": by_subclass_paths,
        },
    }
    (root / "summary（汇总）.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    readme = [
        f"# {title}",
        "",
        f"包含子类：{len(subclasses)} 类",
        f"共：{len(rows)} 条数据",
        "",
        "## 文件",
        "",
        f"- 带标签总表：`{label_path}`",
        f"- 原始格式最简总表：`{original_path}`",
        f"- 来源追溯总表：`{prov_path}`",
        f"- 按子类划分：`{subclass_dir}`",
        "",
        "## 子类数量",
        "",
        "| 子类 | 数量 |",
        "|---|---:|",
    ]
    for subclass in subclasses:
        readme.append(f"| {subclass}（{CN.get(subclass, '未标注')}） | {counts[subclass]} |")
    (root / "README（说明）.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    rows = read_csv(INPUT)
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    OUT_ROOT.mkdir(parents=True)

    common_rows = dataset_rows(rows, COMMON_SUBCLASSES)
    rare_rows = dataset_rows(rows, RARE_SUBCLASSES)
    total_rows = dataset_rows(rows, COMMON_SUBCLASSES + RARE_SUBCLASSES)

    folders = {
        "common": OUT_ROOT / f"01_常规C-H activation（碳氢键活化）_包含8类_共{len(common_rows)}条数据",
        "rare": OUT_ROOT / f"02_小样本稀有C-H functionalization（碳氢键官能团化）_包含6类_共{len(rare_rows)}条数据",
        "total": OUT_ROOT / f"03_总C-H activation（碳氢键活化）_包含14类_共{len(total_rows)}条数据",
    }

    summaries = {
        "common": write_dataset(
            folders["common"],
            common_rows,
            COMMON_SUBCLASSES,
            "common_ch_activation",
            f"常规 C-H activation（碳氢键活化）数据集：包含 8 类，共 {len(common_rows)} 条数据",
        ),
        "rare": write_dataset(
            folders["rare"],
            rare_rows,
            RARE_SUBCLASSES,
            "rare_ch_functionalization",
            f"小样本稀有 C-H functionalization（碳氢键官能团化）数据集：包含 6 类，共 {len(rare_rows)} 条数据",
        ),
        "total": write_dataset(
            folders["total"],
            total_rows,
            COMMON_SUBCLASSES + RARE_SUBCLASSES,
            "total_ch_activation",
            f"总 C-H activation（碳氢键活化）数据集：包含 14 类，共 {len(total_rows)} 条数据",
        ),
    }

    root_readme = [
        "# C-H activation（碳氢键活化）数据集整理：三份",
        "",
        f"来源文件：`{INPUT}`",
        "",
        "## 三份数据集",
        "",
        f"1. 常规 8 类：`{folders['common']}`，共 {len(common_rows)} 条",
        f"2. 小样本稀有 6 类：`{folders['rare']}`，共 {len(rare_rows)} 条",
        f"3. 总合集 14 类：`{folders['total']}`，共 {len(total_rows)} 条",
        "",
        "每个文件夹均包含：带标签总表、原始格式最简总表、来源追溯总表、按子类划分文件夹。",
    ]
    (OUT_ROOT / "README（总说明）.md").write_text("\n".join(root_readme) + "\n", encoding="utf-8")
    (OUT_ROOT / "summary（三份汇总）.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"out_root": str(OUT_ROOT), **summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
