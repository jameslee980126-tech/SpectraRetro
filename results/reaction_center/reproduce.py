"""Recompute frozen reaction-center ranks, groups and seed summaries on CPU."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
KS = (1, 2, 3, 5, 10)


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def prediction_metrics(path):
    value = load(path)
    records = value["records"]
    n = len(records)
    counts = {k: 0 for k in KS}
    direct = {k: 0 for k in KS}
    for row in records:
        candidates = row["top10_candidates"]
        gold = row["gold_candidate"]
        rank = candidates.index(gold) + 1 if gold in candidates else None
        assert rank == row["rank_within_top10"]
        scores = row["candidate_scores"]
        assert len(scores) == row["candidate_count"] and all(math.isfinite(s) for s in scores)
        direct_rank = 1 + sum(score > scores[gold] or (score == scores[gold] and i < gold)
                              for i, score in enumerate(scores))
        for k in KS:
            counts[k] += rank is not None and rank <= k
            direct[k] += direct_rank <= k
    for k in KS:
        assert abs(counts[k] / n - value["metrics"][f"top-{k} accuracy"]) < 1e-6
    return value, counts, direct


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check stored metrics and source hashes")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/reproduced_stage1")
    args = parser.parse_args()
    if args.check:
        for item in load(HERE / "source_manifest.json")["files"]:
            file = ROOT / item["file"]
            assert hashlib.sha256(file.read_bytes()).hexdigest() == item["sha256"], str(file)
    rows, assertions = [], []
    for split in ("test", "validation"):
        for arm in ("full_fspec", "no_spectral"):
            file = HERE / "full_spectral_ablation" / split / f"{arm}_predictions.json"
            value, counts, direct = prediction_metrics(file)
            assert len(value["records"]) == (9250 if split == "test" else 9395)
            rows.append({"experiment": "full_spectral_ablation", "split": split, "arm": arm,
                         "seed": 0, "n": len(value["records"]),
                         **{f"top{k}_hits": counts[k] for k in KS},
                         **{f"direct_top{k}_hits_diagnostic": direct[k] for k in KS}})
            pair = load(HERE / "full_spectral_ablation/original_and_direct_metrics.json")
            source_split = "test" if split == "test" else "valid"
            source_arm = "fspec" if arm == "full_fspec" else arm
            source = pair["splits"][source_split]["models"][source_arm]
            assert all(abs(direct[k] / len(value["records"]) - source["direct_topk_diagnostic"][str(k)]) < 1e-12 for k in KS)
        value, counts, direct = prediction_metrics(HERE / "full_rgcn_comparison" / split / "rgcn_predictions.json")
        rows.append({"experiment": "full_rgcn_comparison", "split": split, "arm": "rgcn", "seed": 0,
                     "n": len(value["records"]), **{f"top{k}_hits": counts[k] for k in KS},
                     **{f"direct_top{k}_hits_diagnostic": direct[k] for k in KS}})
    small_rows = []
    for split in ("test", "validation"):
        similarity = load(HERE / "small_local_propagation" / split / "similarity.json")
        for seed in range(3):
            for arm in ("full_fspec", "local_pair"):
                path = HERE / "small_local_propagation" / split / f"{arm}_seed{seed}.json"
                value, counts, direct = prediction_metrics(path)
                records = value["records"]
                assert len(records) == 2400
                assert [r["sample_id"] for r in records] == [r["sample_id"] for r in similarity]
                for group in ("all", "low", "medium", "high"):
                    indices = [i for i, r in enumerate(records) if group == "all" or r["similarity_group"] == group]
                    assert len(indices) == (2400 if group == "all" else 800)
                    hits = {k: sum(records[i]["rank_within_top10"] is not None and records[i]["rank_within_top10"] <= k
                                   for i in indices) for k in KS}
                    wins = []
                    for i in indices:
                        definition = similarity[i]["radii"]["2"]
                        competitors = definition["competitors"]
                        if not competitors:
                            continue
                        maximum = max(definition["similarity_to_gold"][j] for j in competitors)
                        closest = [j for j in competitors if abs(definition["similarity_to_gold"][j] - maximum) <= 1e-12]
                        score = records[i]["candidate_scores"]
                        wins.append(score[records[i]["gold_candidate"]] > max(score[j] for j in closest))
                    small_rows.append({"split": split, "arm": arm, "seed": seed, "group": group,
                                       "n": len(indices), **{f"top{k}_hits": hits[k] for k in KS},
                                       "closest_decoy_n": len(wins), "closest_decoy_wins": sum(wins)})
                rows.append({"experiment": "small_local_propagation", "split": split, "arm": arm,
                             "seed": seed, "n": len(records), **{f"top{k}_hits": counts[k] for k in KS},
                             **{f"direct_top{k}_hits_diagnostic": direct[k] for k in KS}})
        stored = load(HERE / "small_local_propagation" / split / "summary.json")
        for group in stored["groups"]:
            for arm in ("full_fspec", "local_pair"):
                selected = [r for r in small_rows if r["split"] == split and r["arm"] == arm and r["group"] == group["group"]]
                assert len(selected) == 3
                metrics = ("top1", "top2", "top3", "top5", "top10", "closest_decoy_win_rate")
                for metric in metrics:
                    key = f"{arm}_{metric}_mean"
                    if key not in group:
                        continue
                    values = ([r["closest_decoy_wins"] / r["closest_decoy_n"] for r in selected]
                              if metric == "closest_decoy_win_rate" else [r[metric + "_hits"] / r["n"] for r in selected])
                    assert abs(statistics.mean(values) - group[key]) < 1e-12, (split, key)
                    if f"{arm}_{metric}_sd" in group:
                        assert abs(statistics.stdev(values) - group[f"{arm}_{metric}_sd"]) < 1e-12
        assertions.append({"split": split, "seed_summary_and_closest_decoy": "passed"})
    args.output.mkdir(parents=True, exist_ok=True)
    for name, data in [("reaction_center_results.csv", rows), ("small_stratified_results.csv", small_rows)]:
        with (args.output / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    result = {"state": "passed", "prediction_files": 18, "prediction_records": sum(r["n"] for r in rows),
              "original_rank_metrics": "passed", "full_spectral_direct_rank_diagnostic": "passed",
              "small_seed_summaries": assertions,
              "scope": "Saved reaction-center predictions and ranks; no training, model forward, or chemical recanonicalization."}
    (args.output / "verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
