"""Recount released reaction-level results with the Python standard library."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path


TOP_K = (1, 3, 5, 10)
METHODS = ("fusion", "frequency")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def equal_number(actual, expected, label, tolerance=1e-9):
    require(expected is not None, label + ": missing reference value")
    require(math.isfinite(float(actual)) and math.isfinite(float(expected)), label + ": nonfinite value")
    require(math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance),
            "{}: recomputed {} differs from {}".format(label, actual, expected))


def recount_arm(directory, dataset, mode, seed, expected_n):
    reaction_metrics = read_json(directory / "reaction_metrics.json")
    summary = read_json(directory / "metrics.json")["metrics"]
    exported = []
    seen_ids = None
    for view, rows in reaction_metrics.items():
        require(view in summary, "Missing summary for " + view)
        require(len(rows) == expected_n, "Incorrect denominator for " + directory.name + "/" + view)
        ids = [row["reaction_id"] for row in rows]
        require(len(set(ids)) == expected_n, "Duplicate reaction IDs in " + directory.name)
        require(all(re.fullmatch(r"test:[0-9]{6}", value) for value in ids), "Malformed reaction ID")
        if dataset == "full":
            require(set(ids) == {"test:{:06d}".format(i) for i in range(expected_n)},
                    "Full reaction IDs must cover 0 through 9249")
        if seen_ids is None:
            seen_ids = set(ids)
        require(set(ids) == seen_ids, "Views contain different reactions")
        groups = defaultdict(list)
        groups["all"] = rows
        for row in rows:
            cls = row["class_id"]
            require(type(cls) is int and 1 <= cls <= 6, "Invalid reaction class")
            groups["class" + str(cls)].append(row)
            if row.get("similarity_group") is not None:
                groups["similarity_" + row["similarity_group"]].append(row)
        require(set(summary[view]) == set(groups), "Summary strata differ from reaction-level strata")
        for stratum, selected in groups.items():
            aggregate = summary[view][stratum]
            n = len(selected)
            require(aggregate["n"] == n, "Incorrect stratum denominator")
            for diagnostic in ("pool_coverage", "unique_candidates", "missing_draw_fraction"):
                if diagnostic in aggregate:
                    equal_number(statistics.fmean(r[diagnostic] for r in selected),
                                 aggregate[diagnostic], diagnostic)
            for method in METHODS:
                previous_hits = -1
                for k in TOP_K:
                    key = "{}_top{}".format(method, k)
                    flags = [r[key] for r in selected]
                    require(all(type(f) is int and f in (0, 1) for f in flags), "Invalid hit flag")
                    hits = sum(flags)
                    require(hits >= previous_hits, "Nonmonotonic Top-k hits")
                    previous_hits = hits
                    equal_number(hits / n, aggregate[key], key)
                    exported.append({"dataset": dataset, "mode": mode, "seed": seed,
                                     "view": view, "method": method, "stratum": stratum,
                                     "top_k": k, "hits": hits, "n": n,
                                     "percent": 100.0 * hits / n})
    return exported, seen_ids


def recount_baseline_rank_files(root, record, csv_rank_column=None):
    """Replay saved ranks without decoding or chemical canonicalization."""
    checked = []
    n = record.get("n_per_run")
    if not isinstance(n, int) or n <= 0 or not record.get("percent"):
        return checked
    for name in record.get("files", []):
        path = root / name
        rows = None
        rank_key = None
        if path.suffix == ".csv" and "per_reaction" in path.name:
            with path.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                columns = reader.fieldnames or []
                if csv_rank_column:
                    require(csv_rank_column in columns, "Missing " + csv_rank_column + " in " + name)
                    rank_key = csv_rank_column
                else:
                    rank_key = next((key for key in ("raw_match_rank", "canonical_match_rank")
                                     if key in columns), None)
                if rank_key is not None:
                    rows = list(reader)
        elif path.suffix == ".jsonl" and "native_saved_ranks" in record.get("evidence", ""):
            with path.open(encoding="utf-8-sig") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
            rank_key = "native_hit_rank"
            require(all(rank_key in row for row in rows), "Native rank missing from JSONL")
        if rows is None:
            continue
        require(len(rows) == n, "Saved baseline rank denominator differs: " + name)
        row_ids = [int(row["source_row"]) for row in rows]
        require(len(set(row_ids)) == n, "Duplicate saved baseline reaction IDs: " + name)
        if record.get("dataset") == "full" and n == 9250:
            require(set(row_ids) == set(range(n)), "Full baseline reaction IDs are incomplete: " + name)
        ranks = []
        for row in rows:
            value = row[rank_key]
            if value is None or value == "":
                ranks.append(None)
            else:
                rank = int(value)
                require(rank >= 1, "Saved match ranks must be one-based")
                ranks.append(rank)
        hit_counts = {}
        for k, reference in record["percent"].items():
            if reference is None:
                continue
            hits = sum(rank is not None and rank <= int(k) for rank in ranks)
            hit_counts[k] = hits
            equal_number(100.0 * hits / n, reference, "Baseline saved rank " + name)
            if (record.get("hits") or {}).get(k) is not None:
                require(hits == record["hits"][k], "Baseline saved rank hit count differs")
        checked.append({"file": name, "rank_column": rank_key, "rows": n, "hits": hit_counts})
    return checked


def paper_differ_results(root, modes):
    """Recount the frequency-ranked DiffER results used by Table 1."""
    with (root / "results/paper_tables/table1_reported.csv").open(encoding="utf-8-sig", newline="") as f:
        paper = {r["mode"]: r for r in csv.DictReader(f)
                 if r["model"] == "DiffER" and r["dataset"] == "full"}
    output = []
    for mode in modes:
        row = paper[mode]
        require(row["run_protocol"].startswith("native_plurality"), "Unexpected paper ranking for DiffER")
        files = [name.strip() for name in row["source_files"].split(";")]
        metrics_file, = [root / name for name in files if name.endswith("metrics.json")]
        metrics = read_json(metrics_file)
        require(metrics["mode"] == mode and metrics["passed"], "DiffER run identity differs")
        reported = metrics["metrics"]["native_plurality"]
        n = int(row["recorded_denominator"])
        require(reported["total"] == n == 9250, "Incorrect DiffER denominator")
        reference = {str(k): 100.0 * reported["accuracy"][str(k)] for k in TOP_K}
        result, = recount_baseline_rank_files(
            root, {"files": files, "n_per_run": n, "dataset": "full", "percent": reference,
                   "hits": reported["hits"]}, csv_rank_column="native_match_rank")
        for k in TOP_K:
            hits = result["hits"][str(k)]
            percent = 100.0 * hits / n
            if k in (1, 5, 10):
                equal_number(round(percent, 2), float(row[f"reported_top{k}_percent"]), "DiffER Table 1")
            output.append(dict(model="DiffER", dataset="full", mode=mode, protocol="native_plurality",
                               top_k=k, hits=hits, n=n, percent=percent))
    return output


def baseline_rows(path, modes=None):
    document = read_json(path)
    records = document if isinstance(document, list) else document["records"]
    if modes is not None:
        records = [record for record in records if record['mode'] in modes]
    exported = []
    seed_checks = 0
    rank_checks = []
    for record in records:
        percents = record.get("percent") or {}
        hits = record.get("hits") or {}
        n = record.get("n_per_run", record.get("denominator_per_run", record.get("n")))
        seed_hits = record.get("seed_hits")
        standard_deviation = record.get("sample_sd_percentage_points")
        ordered_k = record.get("top_k", [1, 5, 10])
        label = "/".join(str(record.get(key, "")) for key in ("model", "dataset", "mode"))
        saved_ranks = recount_baseline_rank_files(path.parents[2], record)
        rank_checks.extend(saved_ranks)
        if seed_hits is not None:
            require(type(n) is int and n > 0, label + ": seed counts need a denominator")
            require(isinstance(seed_hits, list) and len(seed_hits) >= 1, label + ": empty seed counts")
            require(all(len(row) == len(ordered_k) for row in seed_hits), label + ": malformed seed counts")
            for position, k in enumerate(ordered_k):
                counts = [row[position] for row in seed_hits]
                require(all(type(value) is int and 0 <= value <= n for value in counts),
                        label + ": invalid seed count")
                values = [100.0 * value / n for value in counts]
                equal_number(statistics.fmean(values), percents.get(str(k)), label + ": seed mean")
                if standard_deviation is not None and standard_deviation[position] is not None:
                    require(len(values) >= 2, label + ": sample SD requires two runs")
                    equal_number(statistics.stdev(values), standard_deviation[position], label + ": sample SD")
            seed_checks += 1
        keys = sorted(set(percents) | set(hits), key=int) or ["1", "5", "10"]
        for key in keys:
            value = percents.get(key)
            count = hits.get(key)
            if value is not None:
                require(0 <= float(value) <= 100, label + ": invalid percentage")
            if count is not None:
                require(type(n) is int and n > 0, label + ": count needs a denominator")
                require(type(count) is int and 0 <= count <= n, label + ": invalid hit count")
                equal_number(100.0 * count / n, value, label + ": count percentage")
            sd = None
            if standard_deviation is not None and int(key) in ordered_k:
                sd = standard_deviation[ordered_k.index(int(key))]
            exported.append({"model": record.get("model"), "dataset": record.get("dataset", record.get("cohort")),
                             "mode": record.get("mode"), "top_k": int(key), "n_per_run": n,
                             "hits": count, "percent": value, "sample_sd_percentage_points": sd,
                             "seed": record.get("seed_basis", record.get("seed")),
                             "source_evidence": record.get("evidence", record.get("source_evidence")),
                             "protocol": record.get("protocol"), "note": record.get("note"),
                             "verification_scope": "saved_rank_integer_replay" if saved_ranks else
                             ("saved_seed_integer_summary" if seed_hits is not None else
                              ("saved_aggregate_only" if percents else "no_bound_numeric_result")),
                             "saved_rank_rows_recounted": sum(item["rows"] for item in saved_ranks)})
    return exported, len(records), seed_checks, rank_checks


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def recount_small_similarity(root, benchmark, modes):
    """Pool each seed's precursor-recovery hits using the frozen test index."""
    with (root / "datasets/small/test_index.csv").open(encoding="utf-8-sig", newline="") as stream:
        index = list(csv.DictReader(stream))
    groups = {row["sample_id"]: row["similarity_group"] for row in index}
    strata = ("high", "medium", "low")
    require(len(index) == len(groups) == 2400, "Small test index must contain 2,400 unique IDs")
    for stratum in strata:
        require(sum(group == stratum for group in groups.values()) == 800, "Incorrect similarity group size")

    result = []
    for mode in modes:
        counts = defaultdict(list)
        for seed in (0, 1, 2):
            path = benchmark / f"{mode}-class/results/small/seed{seed}/reaction_metrics.json"
            rows = read_json(path)["stage1_gated"]
            require(len(rows) == len(groups) and {r["reaction_id"] for r in rows} == groups.keys(),
                    "Small-set reaction IDs differ from the similarity index")
            by_stratum = defaultdict(list)
            for row in rows:
                group = groups[row["reaction_id"]]
                require(row.get("similarity_group") in (None, group), "Archived similarity group differs from index")
                by_stratum[group].append(row)
            by_stratum["all"] = rows
            for stratum, members in by_stratum.items():
                for k in TOP_K:
                    counts[stratum, k].append(sum(row[f"fusion_top{k}"] for row in members))

        for stratum in ("all", *strata):
            n = 2400 if stratum == "all" else 800
            for k in TOP_K:
                hits = counts[stratum, k]
                rates = [100.0 * h / n for h in hits]
                result.append(dict(mode=mode, stratum=stratum, top_k=k, n_per_seed=n,
                                   seeds="[0, 1, 2]", hits=json.dumps(hits),
                                   mean_percent=statistics.fmean(rates),
                                   sample_sd_percentage_points=statistics.stdev(rates)))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate all available counts and summaries before exporting.")
    parser.add_argument("--mode", choices=["known", "unknown"], help="Recount one reaction-class regime; omit to recount both.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="Repository root.")
    parser.add_argument("--outdir", type=Path, help="Directory for newly reproduced CSV files.")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    modes = (args.mode,) if args.mode else ('known', 'unknown')
    benchmark = root / 'experiments/section_4_results/4_1_benchmark'
    if all(
        not (benchmark / f'{mode}-class').exists() and (root / f'{mode}-class').is_dir() for mode in modes
    ):
        benchmark = root
    destination = args.outdir or Path('outputs/reproduced_results') / (args.mode or '')
    outdir = (destination if destination.is_absolute() else root / destination).resolve()
    if outdir.is_relative_to(root) and not outdir.is_relative_to(root / 'outputs'):
        parser.error('In-repository outputs must be placed under outputs/')
    rows = []
    ids_by_dataset = {}
    arm_counts = {"full": 0, "small": 0}
    small_aggregate_count = 0
    for mode in modes:
        recounted, ids = recount_arm(benchmark / (mode + '-class/results/full'), "full", mode, 0, 9250)
        rows.extend(recounted)
        arm_counts["full"] += 1
        ids_by_dataset.setdefault("full", ids)
        require(ids_by_dataset["full"] == ids, "Full modes contain different reaction IDs")
    small_summary_path = root / "results/small_summary.json"
    if small_summary_path.exists():
        for mode in modes:
            for seed in (0, 1, 2):
                directory = benchmark / (mode + '-class/results/small') / ('seed' + str(seed))
                require(directory.is_dir(), "Missing small-set arm " + directory.name)
                recounted, ids = recount_arm(directory, "small", mode, seed, 2400)
                rows.extend(recounted)
                arm_counts["small"] += 1
                ids_by_dataset.setdefault("small", ids)
                require(ids_by_dataset["small"] == ids, "Small arms contain different reaction IDs")
        document = read_json(small_summary_path)
        small_summary = dict(document,
                             records=[r for r in document['records'] if r['mode'] in modes],
                             aggregates=[r for r in document['aggregates'] if r['mode'] in modes])
        expected_arms = {(mode, seed) for mode in modes for seed in (0, 1, 2)}
        require(len(small_summary["records"]) == 3 * len(modes) and
                {(r["mode"], r["seed"]) for r in small_summary["records"]} == expected_arms,
                "Small summary must cover exactly the six released arms")
        for record in small_summary["records"]:
            require(record["n"] == 2400 and set(record["percent"]) == {str(k) for k in TOP_K},
                    "Incomplete small arm summary")
            for k, value in record["percent"].items():
                selected = [r for r in rows if r["dataset"] == "small" and r["mode"] == record["mode"]
                            and r["seed"] == record["seed"] and r["view"] == "stage1_gated"
                            and r["stratum"] == "all" and r["method"] == "fusion" and r["top_k"] == int(k)]
                require(len(selected) == 1, "Small summary does not identify one arm")
                equal_number(selected[0]["percent"], value, "Small aggregate")
        aggregate_keys = set()
        for aggregate in small_summary["aggregates"]:
            key = tuple(aggregate[name] for name in ("mode", "view", "method", "stratum", "top_k"))
            require(key not in aggregate_keys, "Duplicate small three-seed aggregate")
            aggregate_keys.add(key)
            selected = sorted([r for r in rows if r["dataset"] == "small" and
                               tuple(r[name] for name in ("mode", "view", "method", "stratum", "top_k")) == key],
                              key=lambda r: r["seed"])
            require(len(selected) == 3 and all(r["n"] == aggregate["n_per_run"] for r in selected),
                    "Small aggregate must cover all three seeds")
            require([r["hits"] for r in selected] == aggregate["seed_hits"], "Small aggregate seed hits differ")
            equal_number(statistics.fmean(r["percent"] for r in selected), aggregate["mean_percent"],
                         "Small three-seed mean")
            equal_number(statistics.stdev(r["percent"] for r in selected),
                         aggregate["sample_sd_percentage_points"], "Small three-seed sample SD")
        expected_keys = {(r["mode"], r["view"], r["method"], r["stratum"], r["top_k"])
                         for r in rows if r["dataset"] == "small" and r["seed"] in (1, 2)}
        require(aggregate_keys == expected_keys, "Small summary is missing a three-seed stratum")
        small_aggregate_count = len(aggregate_keys)
    baselines, baseline_count, seed_checks, rank_checks = baseline_rows(root / "results/baselines/summary.json", modes)
    paper_differ = paper_differ_results(root, modes)
    similarity = recount_small_similarity(root, benchmark, modes) if arm_counts["small"] else []
    outdir.mkdir(parents=True, exist_ok=True)
    full_rows = [r for r in rows if r["dataset"] == "full"]
    fields = ["dataset", "mode", "seed", "view", "method", "stratum", "top_k", "hits", "n", "percent"]
    write_csv(outdir / "main_full.csv", full_rows, fields)
    if arm_counts["small"]:
        write_csv(outdir / "main_small.csv", [r for r in rows if r["dataset"] == "small"], fields)
        aggregate_rows = []
        small = [r for r in rows if r["dataset"] == "small"]
        grouped = defaultdict(list)
        for row in small:
            grouped[(row["mode"], row["view"], row["method"], row["stratum"], row["top_k"])].append(row)
        for key, values in sorted(grouped.items()):
            if len(values) != 3:
                continue
            aggregate_rows.append(dict(zip(("mode", "view", "method", "stratum", "top_k"), key),
                                       runs=3, n_per_run=values[0]["n"],
                                       mean_percent=statistics.fmean(r["percent"] for r in values),
                                       sample_sd_percentage_points=statistics.stdev(r["percent"] for r in values)))
        write_csv(outdir / "small_stratum_seed_hits.csv", similarity, list(similarity[0]))
        write_csv(outdir / "main_small_summary.csv", aggregate_rows,
                  ["mode", "view", "method", "stratum", "top_k", "runs", "n_per_run", "mean_percent", "sample_sd_percentage_points"])
    write_csv(outdir / "baselines.csv", baselines,
              ["model", "dataset", "mode", "top_k", "n_per_run", "hits", "percent", "sample_sd_percentage_points", "seed", "source_evidence", "protocol", "note", "verification_scope", "saved_rank_rows_recounted"])
    write_csv(outdir / "differ_full_native_frequency.csv", paper_differ,
              ["model", "dataset", "mode", "protocol", "top_k", "hits", "n", "percent"])
    result = {"status": "passed", "full_arms": arm_counts["full"], "full_reactions": arm_counts["full"] * 9250,
              "small_arms": arm_counts["small"], "small_reactions": arm_counts["small"] * 2400,
              "small_three_seed_stratum_summaries_recounted": small_aggregate_count,
              "baseline_records": baseline_count, "baseline_seed_summaries_recounted": seed_checks,
              "differ_paper_frequency_cells": len(paper_differ),
              "baseline_saved_rank_files_recounted": len(rank_checks),
              "baseline_saved_rank_rows_recounted": sum(item["rows"] for item in rank_checks),
              "baseline_saved_rank_details": rank_checks,
              "candidate_chemical_rescoring_performed": False,
              "null_values_preserved": True}
    (outdir / "verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "baseline_saved_rank_details"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}), file=sys.stderr)
        raise SystemExit(1)
