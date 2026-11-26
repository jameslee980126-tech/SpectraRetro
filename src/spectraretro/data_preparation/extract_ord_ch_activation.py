from __future__ import annotations

import csv
import gzip
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

from ord_schema.proto import dataset_pb2
from ord_schema.proto import reaction_pb2
from rdkit import Chem
from rdkit import RDLogger


RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "ch_activation_mining_outputs" / "ORD_selected_ch_activation"

ORD_SOURCES = [
    {
        "source_id": "ORD_D261_SHIELDS_DIRECT_ARYLATION",
        "path": ROOT
        / "ord-data-main"
        / "ord-data-main"
        / "data"
        / "d2"
        / "ord_dataset-d26118acda314269becc35db5c22dc59.pb.gz",
        "reaction_subclass": "C-H arylation",
        "subclass_confidence": "high_by_dataset_metadata",
        "source_label": "ORD Shields direct arylation optimization set",
    },
    {
        "source_id": "ORD_0C75_SHIELDS_DIRECT_ARYLATION_EXTENDED",
        "path": ROOT
        / "ord-data-main"
        / "ord-data-main"
        / "data"
        / "0c"
        / "ord_dataset-0c75d67751634f0594b24b9f498b77c2.pb.gz",
        "reaction_subclass": "C-H arylation",
        "subclass_confidence": "high_by_dataset_metadata",
        "source_label": "ORD Shields direct arylation extended set",
    },
    {
        "source_id": "ORD_675_Pd_CH_ARYLATION_IMIDAZOLE",
        "path": ROOT
        / "ord-data-main"
        / "ord-data-main"
        / "data"
        / "67"
        / "ord_dataset-675eddcaa6674ce3ae61e79bbc1e1c08.pb.gz",
        "reaction_subclass": "C-H arylation",
        "subclass_confidence": "high_by_dataset_metadata",
        "source_label": "ORD Pd-catalyzed C-H arylation of imidazole with aryl bromide",
    },
    {
        "source_id": "ORD_541_OXAZOLE_CH_ACTIVATION",
        "path": ROOT
        / "ord-data-main"
        / "ord-data-main"
        / "data"
        / "54"
        / "ord_dataset-5415f83da68e4797856ddd330dda5c61.pb.gz",
        "reaction_subclass": "C-H arylation",
        "subclass_confidence": "high_by_dataset_metadata",
        "source_label": "ORD C-H activation of oxazoles from PyParser demo case",
    },
    {
        "source_id": "ORD_99_ROCHE_BORYLATION",
        "path": ROOT
        / "ord-data-main"
        / "ord-data-main"
        / "data"
        / "99"
        / "ord_dataset-99c23cf435dc42f1af884053bc8b11c7.pb.gz",
        "reaction_subclass": "C-H borylation",
        "subclass_confidence": "review_by_dataset_metadata",
        "source_label": "ORD Roche borylation literature set, needs C-H borylation structure review",
    },
]


def compound_identifier(compound, preferred=("SMILES", "CXSMILES")) -> str:
    for name in preferred:
        enum = reaction_pb2.CompoundIdentifier.IdentifierType.Value(name)
        for ident in compound.identifiers:
            if ident.type == enum and ident.value:
                return ident.value
    return ""


def product_identifier(product, preferred=("SMILES", "CXSMILES")) -> str:
    for name in preferred:
        enum = reaction_pb2.CompoundIdentifier.IdentifierType.Value(name)
        for ident in product.identifiers:
            if ident.type == enum and ident.value:
                return ident.value
    return ""


def reaction_identifier(reaction, preferred=("REACTION_SMILES", "REACTION_CXSMILES")) -> str:
    for name in preferred:
        enum = reaction_pb2.ReactionIdentifier.IdentifierType.Value(name)
        for ident in reaction.identifiers:
            if ident.type == enum and ident.value:
                return ident.value
    return ""


def clean_component_smiles(smiles: Iterable[str]) -> list[str]:
    out = []
    for smi in smiles:
        smi = (smi or "").strip()
        if not smi:
            continue
        out.append(smi)
    return out


def build_reaction_smiles(reaction) -> str:
    rxn_id = reaction_identifier(reaction)
    if rxn_id:
        return rxn_id

    reactants: list[str] = []
    reagents: list[str] = []
    role_enum = reaction_pb2.ReactionRole.ReactionRoleType
    for reaction_input in reaction.inputs.values():
        for component in reaction_input.components:
            smi = compound_identifier(component)
            if not smi:
                continue
            role = component.reaction_role
            if role == role_enum.Value("REACTANT"):
                reactants.append(smi)
            elif role in {
                role_enum.Value("REAGENT"),
                role_enum.Value("SOLVENT"),
                role_enum.Value("CATALYST"),
                role_enum.Value("INTERNAL_STANDARD"),
                role_enum.Value("AUTHENTIC_STANDARD"),
            }:
                reagents.append(smi)
            else:
                reactants.append(smi)

    products: list[str] = []
    for outcome in reaction.outcomes:
        desired = [p for p in outcome.products if p.is_desired_product]
        chosen = desired or list(outcome.products)
        for product in chosen:
            smi = product_identifier(product)
            if smi:
                products.append(smi)
        if products:
            break

    reactants = clean_component_smiles(reactants)
    reagents = clean_component_smiles(reagents)
    products = clean_component_smiles(products)
    if not reactants or not products:
        return ""
    return ".".join(reactants) + ">" + ".".join(reagents) + ">" + ".".join(products)


def canonical_rxn_core(rxn: str) -> str:
    if rxn.count(">") != 2:
        return rxn
    left, _mid, right = rxn.split(">", 2)

    def canon_side(side: str) -> str:
        parts = []
        for smi in side.split("."):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                parts.append(smi)
                continue
            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(0)
            parts.append(Chem.MolToSmiles(mol, canonical=True))
        return ".".join(sorted(p for p in parts if p))

    try:
        return f"{canon_side(left)}>>{canon_side(right)}"
    except Exception:
        return rxn


def extract_source(source: dict[str, object]) -> list[dict[str, object]]:
    dataset = dataset_pb2.Dataset()
    path = Path(source["path"])
    with gzip.open(path, "rb") as f:
        dataset.ParseFromString(f.read())

    rows: list[dict[str, object]] = []
    for idx, reaction in enumerate(dataset.reactions, start=1):
        rxn = build_reaction_smiles(reaction)
        if not rxn:
            continue
        rows.append(
            {
                "source_id": source["source_id"],
                "source_label": source["source_label"],
                "dataset_id": dataset.dataset_id,
                "dataset_name": dataset.name,
                "source_file": str(path),
                "source_reaction_index": idx,
                "reaction_smiles": rxn,
                "reaction_core_key": canonical_rxn_core(rxn),
                "reaction_subclass": source["reaction_subclass"],
                "subclass_confidence": source["subclass_confidence"],
                "needs_review": str(source["subclass_confidence"]).startswith("review"),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []
    by_source = {}
    for source in ORD_SOURCES:
        rows = extract_source(source)
        all_rows.extend(rows)
        by_source[source["source_id"]] = {
            "rows_with_reaction_smiles": len(rows),
            "unique_reaction_cores": len({row["reaction_core_key"] for row in rows}),
            "reaction_subclass": source["reaction_subclass"],
            "subclass_confidence": source["subclass_confidence"],
            "source_file": str(source["path"]),
        }

    fields = [
        "source_id",
        "source_label",
        "dataset_id",
        "dataset_name",
        "source_file",
        "source_reaction_index",
        "reaction_smiles",
        "reaction_core_key",
        "reaction_subclass",
        "subclass_confidence",
        "needs_review",
    ]
    all_path = OUT_DIR / "ord_selected_ch_activation_condition_rows.csv"
    write_csv(all_path, all_rows, fields)

    seen = set()
    unique_rows = []
    duplicate_rows = []
    for row in all_rows:
        key = row["reaction_core_key"]
        if key in seen:
            duplicate_rows.append(row)
            continue
        seen.add(key)
        unique_rows.append(row)
    unique_path = OUT_DIR / "ord_selected_ch_activation_unique_reaction_cores.csv"
    dup_path = OUT_DIR / "ord_selected_ch_activation_duplicate_condition_rows.csv"
    write_csv(unique_path, unique_rows, fields)
    write_csv(dup_path, duplicate_rows, fields)

    summary = {
        "total_condition_rows_with_reaction_smiles": len(all_rows),
        "unique_reaction_cores": len(unique_rows),
        "duplicate_condition_rows": len(duplicate_rows),
        "subclass_counts_condition_rows": dict(Counter(row["reaction_subclass"] for row in all_rows)),
        "subclass_counts_unique_cores": dict(Counter(row["reaction_subclass"] for row in unique_rows)),
        "by_source": by_source,
        "outputs": {
            "condition_rows": str(all_path),
            "unique_reaction_cores": str(unique_path),
            "duplicate_condition_rows": str(dup_path),
        },
    }
    summary_path = OUT_DIR / "ord_selected_ch_activation_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
