# Baseline implementations and result protocols

The distribution contains implementations for ten comparison methods. The [40-record result registry](../../results/baselines/summary.json) separates model, dataset scale and class setting. Each record identifies its available metrics, candidate-rank observations, seed summaries and scoring protocol.

| Method | Implementation | Additional run artifacts |
|---|---|---|
| DiffER | [source](../../src/baselines/DiffER) | [Full known/unknown native draws and selected weights](../../results/baselines/native_runs/DiffER) |
| GLN | [source](../../src/baselines/GLN) | [Selected weights](../../experiments/section_4_results/4_1_benchmark/unknown-class/checkpoints/baselines/GLN), [preprocessing member inventory](../../experiments/section_4_results/4_1_benchmark/baseline_preprocessing_manifests/README.md) |
| RetroXpert | [source](../../src/baselines/RetroXpert) | [Selected two-stage weights](../../experiments/section_4_results/4_1_benchmark/unknown-class/checkpoints/baselines/RetroXpert), [preprocessing member inventory](../../experiments/section_4_results/4_1_benchmark/baseline_preprocessing_manifests/README.md) |
| Graph2Edits | [source](../../src/baselines/Graph2Edits) | [Six small-set runs](../../results/baselines/native_runs/Graph2Edits) |
| GraphRetro | [source](../../src/baselines/GraphRetro) | [Available edit/leaving-group checkpoints and beams](../../results/baselines/native_runs/GraphRetro) |
| RetroSim | [source](../../src/baselines/RetroSim) | [Small-set predictions and settings](../../results/baselines/native_runs/RetroSim) |
| TransGraphEdit | [source](../../src/baselines/TransGraphEdit) | [Available small-set weights and beams](../../results/baselines/native_runs/TransGraphEdit) |
| Retroformer | [source](../../src/baselines/Retroformer) | [Full-known selected weights](../../src/baselines/Retroformer/weights_full_known), [registry](../../results/baselines/summary.json) |
| RetroDiT | [source](../../src/baselines/RetroDiT) | [Full-known selected weights](../../src/baselines/RetroDiT/weights_full_known), [registry](../../results/baselines/summary.json) |
| GDiffRetro | [source](../../src/baselines/GDiffRetro) | [Registry and saved records](../../results/baselines/summary.json) |

[Known-class records](../../experiments/section_4_results/4_1_benchmark/known-class/results/baselines) · [Unknown-class records](../../experiments/section_4_results/4_1_benchmark/unknown-class/results/baselines) · [Checkpoint registry](checkpoints.md)

## Scoring definitions

| Method / protocol | Full denominator | Ranking or sampling definition |
|---|---:|---|
| DiffER unified | 9,250 | First ten original slots; invalid and repeated slots retained |
| DiffER native frequency | 9,250 | Frequency ranking across five input enumerations × ten draws |
| GLN unified | 9,250 | Original ranked slots compared with the original test reactants |
| RetroXpert unified | 9,250 | Original beam slots compared with the original test reactants |
| RetroDiT native | 9,248 | Native deduplicated ranking; small denominator 2,398 |
| GDiffRetro product-level records | 7,398 where declared | Unique-product unit and reused completion candidates |

For full-data DiffER, Table 1 uses occurrence frequency over 50 draws. Recount the `native_match_rank` column with `python scripts/reproduce_results.py --check`. [DiffER Table 1 counts](../../results/reference_tables/reproduced_results/differ_full_native_frequency.csv)

The [Table 1 transcription](../../results/paper_tables/table1_reported.csv) identifies the printed entries. The [results registry](../../results/baselines/summary.json) records the available predictions or summaries for each method and setting.
