from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from mine_mapped_ch_activation import canonical_rxn_core, classify_mapped_rxn


csv.field_size_limit(2**31 - 1)


def write_header(path: Path, fields: list[str]) -> csv.DictWriter:
    f = path.open("w", encoding="utf-8", newline="")
    writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer._output_file = f  # type: ignore[attr-defined]
    return writer


def close_writer(writer: csv.DictWriter) -> None:
    f = getattr(writer, "_output_file", None)
    if f is not None:
        f.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--progress-every", type=int, default=100000)
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_id = f"CRD_MappedCRD_rxnmapper_conf{str(args.confidence_threshold).replace('.', '')}"
    files = sorted(input_dir.glob("*.parquet"))

    candidate_fields = [
        "source_id",
        "source_label",
        "source_file",
        "source_row_global",
        "source_row_in_file",
        "mapping_confidence",
        "reaction",
        "mapped_rxn",
        "reaction_core_key",
        "reaction_subclass",
        "subclass_confidence",
        "needs_review",
        "formed_bond_count",
        "formed_bond_subclasses",
        "likely_c_h_functionalization",
        "ring_count_delta",
    ]
    summary_fields = ["source_file", "source_row_global", "mapping_confidence", "reaction_subclass", "notes"]
    high_path = output_dir / f"{source_id}_ch_activation_candidates_with_subclass.csv"
    review_path = output_dir / f"{source_id}_needs_review.csv"
    excluded_path = output_dir / f"{source_id}_excluded_summary.csv"
    high_writer = write_header(high_path, candidate_fields)
    review_writer = write_header(review_path, candidate_fields)
    excluded_writer = write_header(excluded_path, summary_fields)

    counters = Counter()
    subclass_counts = Counter()
    confidence_counts = Counter()
    started = time.time()

    try:
        global_idx = 0
        for file_path in files:
            pf = pq.ParquetFile(file_path)
            file_idx = 0
            for batch in pf.iter_batches(
                batch_size=args.batch_size,
                columns=["reaction", "mapped_reaction", "confidence"],
            ):
                data = batch.to_pydict()
                for reaction, mapped_rxn, confidence in zip(
                    data["reaction"], data["mapped_reaction"], data["confidence"]
                ):
                    global_idx += 1
                    file_idx += 1
                    counters["read_rows"] += 1
                    if args.max_rows and counters["read_rows"] > args.max_rows:
                        raise StopIteration
                    try:
                        conf = float(confidence)
                    except Exception:
                        conf = 0.0
                    if conf < args.confidence_threshold:
                        counters["skipped_low_mapping_confidence"] += 1
                        continue
                    analysis = classify_mapped_rxn(mapped_rxn or "")
                    subclass = str(analysis["reaction_subclass"])
                    subclass_counts[subclass] += 1
                    confidence_label = str(analysis["subclass_confidence"])
                    confidence_counts[confidence_label] += 1

                    if confidence_label in {"high", "review"}:
                        out = {
                            "source_id": source_id,
                            "source_label": "MappedCRD rxnmapper split from HuggingFace",
                            "source_file": str(file_path),
                            "source_row_global": global_idx,
                            "source_row_in_file": file_idx,
                            "mapping_confidence": conf,
                            "reaction": reaction,
                            "mapped_rxn": mapped_rxn,
                            "reaction_core_key": canonical_rxn_core(mapped_rxn or ""),
                            "reaction_subclass": subclass,
                            "subclass_confidence": confidence_label,
                            "needs_review": analysis["needs_review"],
                            "formed_bond_count": analysis["formed_bond_count"],
                            "formed_bond_subclasses": analysis["formed_bond_subclasses"],
                            "likely_c_h_functionalization": analysis["likely_c_h_functionalization"],
                            "ring_count_delta": analysis["ring_count_delta"],
                        }
                        if confidence_label == "high":
                            high_writer.writerow(out)
                            counters["high_confidence_candidates"] += 1
                        else:
                            review_writer.writerow(out)
                            counters["review_candidates"] += 1
                    else:
                        counters["excluded"] += 1
                        if counters["excluded"] <= 5000:
                            excluded_writer.writerow(
                                {
                                    "source_file": str(file_path),
                                    "source_row_global": global_idx,
                                    "mapping_confidence": conf,
                                    "reaction_subclass": subclass,
                                    "notes": analysis.get("notes", ""),
                                }
                            )
                    if args.progress_every and counters["read_rows"] % args.progress_every == 0:
                        elapsed = time.time() - started
                        rate = counters["read_rows"] / elapsed if elapsed else 0
                        print(
                            f"{source_id}: read={counters['read_rows']} high={counters['high_confidence_candidates']} "
                            f"review={counters['review_candidates']} skipped_low_conf={counters['skipped_low_mapping_confidence']} "
                            f"excluded={counters['excluded']} rate={rate:.1f}/s",
                            flush=True,
                        )
    except StopIteration:
        pass
    finally:
        close_writer(high_writer)
        close_writer(review_writer)
        close_writer(excluded_writer)

    summary = {
        "source_id": source_id,
        "input_dir": str(input_dir),
        "input_files": [str(p) for p in files],
        "confidence_threshold": args.confidence_threshold,
        "counts": dict(counters),
        "subclass_counts": dict(subclass_counts.most_common()),
        "confidence_counts": dict(confidence_counts.most_common()),
        "outputs": {
            "high_confidence_candidates": str(high_path),
            "needs_review": str(review_path),
            "excluded_summary": str(excluded_path),
        },
    }
    summary_path = output_dir / f"{source_id}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
