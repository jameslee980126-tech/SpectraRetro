from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

from rxnmapper import RXNMapper

from mine_mapped_ch_activation import split_rxn


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
OUT_DIR = ROOT / "ch_activation_cleaning_outputs" / "step1_unmapped_rxnmapper"
OUT = OUT_DIR / "unmapped_rxnmapper_results.csv"
SUMMARY = OUT_DIR / "unmapped_rxnmapper_summary.json"


def mapping_core_rxn(rxn: str) -> str:
    parts = split_rxn(rxn)
    if not parts:
        return rxn.strip()
    left, _middle, right = parts
    return f"{left}>>{right}"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    with INPUT.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("mapped_rxn", "").strip():
                continue
            rxn = row.get("reaction_smiles", "").strip()
            if not rxn:
                continue
            row["rxn_for_mapping"] = mapping_core_rxn(rxn)
            rows.append(row)

    mapper = RXNMapper()
    fields = [
        "global_id",
        "source_id",
        "reaction_subclass",
        "reaction_smiles",
        "rxn_for_mapping",
        "mapped_rxn_generated",
        "rxnmapper_confidence",
        "mapping_status",
        "error",
    ]
    counts = {"total_unmapped": len(rows), "mapped": 0, "failed": 0}
    started = time.time()
    batch_size = 32

    with OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            reactions = [row["rxn_for_mapping"] for row in batch]
            try:
                results = mapper.get_attention_guided_atom_maps(reactions)
            except Exception as exc:
                for row in batch:
                    counts["failed"] += 1
                    writer.writerow(
                        {
                            **row,
                            "mapped_rxn_generated": "",
                            "rxnmapper_confidence": "",
                            "mapping_status": "failed_batch",
                            "error": str(exc),
                        }
                    )
                continue
            for row, result in zip(batch, results):
                mapped = result.get("mapped_rxn", "")
                conf = result.get("confidence", "")
                if mapped:
                    counts["mapped"] += 1
                    status = "mapped"
                    error = ""
                else:
                    counts["failed"] += 1
                    status = "failed_empty"
                    error = ""
                writer.writerow(
                    {
                        **row,
                        "mapped_rxn_generated": mapped,
                        "rxnmapper_confidence": conf,
                        "mapping_status": status,
                        "error": error,
                    }
                )
            done = min(start + batch_size, len(rows))
            if done % 512 == 0 or done == len(rows):
                elapsed = time.time() - started
                print(
                    f"mapped {done}/{len(rows)} "
                    f"rate={done / elapsed if elapsed else 0:.1f}/s "
                    f"ok={counts['mapped']} failed={counts['failed']}",
                    flush=True,
                )

    SUMMARY.write_text(json.dumps(counts, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUT), **counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
