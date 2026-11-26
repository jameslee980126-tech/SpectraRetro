# Prediction records and source tables

[Browse results by paper section](../experiments/README.md).

| Research result | Location |
|---|---|
| Full-data complete-precursor predictions and decoded candidates | [Known](../experiments/section_4_results/4_1_benchmark/known-class/results/full), [unknown](../experiments/section_4_results/4_1_benchmark/unknown-class/results/full) |
| Small-data seeds 0, 1 and 2 | [Known](../experiments/section_4_results/4_1_benchmark/known-class/results/small), [unknown](../experiments/section_4_results/4_1_benchmark/unknown-class/results/small), [summary](small_summary.json) |
| Model training and checkpoint selection | [Known](../experiments/section_4_results/4_1_benchmark/known-class/results/training), [unknown](../experiments/section_4_results/4_1_benchmark/unknown-class/results/training) |
| Center ablations and local similarity | [reaction_center](reaction_center) |
| Numerical encoder checks | [diagnostics](diagnostics) |
| Composition-prior comparison | [unknown-class/results/ablations](../experiments/section_4_results/4_1_benchmark/unknown-class/results/ablations) |
| Baseline summaries and available native runs | [baselines](baselines) |
| Dataset landscape, descriptors and paired observations | [supporting_information](supporting_information) |
| Manuscript table transcription | [paper_tables](paper_tables) |
| Chemical, spatial and quantum analyses | [mechanistic source data](source_data/mechanistic_interpretability) |
| Representative and pharmaceutical cases | [cases](cases/README.md) |
| Error populations | [error_analysis](error_analysis) |

`metrics.json` contains aggregate values, `reaction_metrics.json` contains per-reaction hits and `result.csv` contains decoded fragment draws. [SpectraRetro evaluation](../docs/guides/evaluation.md) · [Baseline protocols](../docs/guides/baselines.md)
