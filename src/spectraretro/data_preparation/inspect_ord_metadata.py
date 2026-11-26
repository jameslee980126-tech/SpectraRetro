from __future__ import annotations

import csv
import gzip
from pathlib import Path

from ord_schema.proto import dataset_pb2


ROOT = Path(__file__).resolve().parent
ORD_DATA = ROOT / "ord-data-main" / "ord-data-main" / "data"
OUT = ROOT / "ord_metadata_keyword_hits.csv"

KEYWORDS = [
    "c-h",
    "ch activation",
    "c h activation",
    "functionalization",
    "arylation",
    "borylation",
    "olefination",
    "alkylation",
    "amination",
    "amidation",
    "oxygenation",
    "halogenation",
    "cyanation",
    "carbonylation",
    "annulation",
]


def dataset_text(dataset: dataset_pb2.Dataset) -> str:
    parts: list[str] = []
    for attr in ("name", "description", "dataset_id", "doi"):
        value = getattr(dataset, attr, "")
        if value:
            parts.append(str(value))
    return "\n".join(p for p in parts if p)


def main() -> None:
    rows = []
    files = sorted(ORD_DATA.rglob("*.pb.gz"))
    for path in files:
        dataset = dataset_pb2.Dataset()
        try:
            with gzip.open(path, "rb") as f:
                dataset.ParseFromString(f.read())
        except Exception as exc:
            rows.append(
                {
                    "file": str(path),
                    "dataset_id": "",
                    "name": "",
                    "num_reactions": "",
                    "matched_keywords": "",
                    "description": f"PARSE_ERROR: {exc}",
                }
            )
            continue
        text = dataset_text(dataset)
        lower = text.lower()
        hits = [kw for kw in KEYWORDS if kw in lower]
        if not hits:
            continue
        rows.append(
            {
                "file": str(path),
                "dataset_id": dataset.dataset_id,
                "name": getattr(dataset, "name", ""),
                "num_reactions": len(dataset.reactions),
                "matched_keywords": "; ".join(hits),
                "description": text.replace("\r", " ").replace("\n", " ")[:1200],
            }
        )

    with OUT.open("w", encoding="utf-8", newline="") as f:
        fields = ["file", "dataset_id", "name", "num_reactions", "matched_keywords", "description"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"scanned_files={len(files)}")
    print(f"keyword_hit_datasets={len(rows)}")
    print(f"output={OUT}")
    for row in sorted(rows, key=lambda r: int(r["num_reactions"] or 0), reverse=True)[:30]:
        print(f"{row['num_reactions']}\t{row['matched_keywords']}\t{row['name']}\t{row['file']}")


if __name__ == "__main__":
    main()
