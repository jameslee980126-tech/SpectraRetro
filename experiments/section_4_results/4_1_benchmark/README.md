# SpectraRetro single-step prediction on CHRetro

[Table 1](tables/table_1.tex) compares precursor recovery on CHRetro with the reaction class supplied or withheld.

| SpectraRetro model source | Reaction-class input at prediction time | Experiment files | Full-set results |
|---|---|---|---|
| [known-class/](../../../src/spectraretro/known-class/) | Supplied (for example, C–H halogenation) | [Known-class experiment](known-class/) | [Known metrics](known-class/results/full/metrics.json) |
| [unknown-class/](../../../src/spectraretro/unknown-class/) | Withheld | [Unknown-class experiment](unknown-class/) | [Unknown metrics](unknown-class/results/full/metrics.json) |

The full benchmark uses 9,250 test reactions and one run per class mode. Stage 2 conditions on the test synthon and generates 300 draws per fragment with 100 reverse diffusion intervals. The small test partition has 2,400 reactions. [Test inputs and evaluation](../../../docs/guides/evaluation.md)

| Paper item or input | What it contains | Files |
|---|---|---|
| Manuscript comparison | The manuscript table transcribes the reported SpectraRetro and baseline results. The CSV retains each printed model, scale and mode. | [Table 1](tables/table_1.tex), [reported values](../../../results/paper_tables/table1_reported.csv) |
| Full benchmark | Use the per-reaction metrics to recount exact-match hits, or open the decoded candidate table to inspect the precursor strings for one fragment. | [known-class per-reaction metrics](known-class/results/full/reaction_metrics.json), [unknown-class per-reaction metrics](unknown-class/results/full/reaction_metrics.json), [known-class candidates](known-class/results/full/result.csv), [unknown-class candidates](unknown-class/results/full/result.csv) |
| Small benchmark | The unknown-class seed-0 run gives 1,392/2,400 Top-1 hits (58.00%). SI Table S10 reports small-set accuracy over three seeds. | [per-seed results](../../../results/reference_tables/reproduced_results/main_small.csv), [three-seed summaries](../../../results/reference_tables/reproduced_results/main_small_summary.csv), [Table S10](../../supporting_information/s4_additional_results/tables/table_s10.tex) |
| Baselines | DiffER full-data entries use candidate frequency over 50 draws. | [DiffER results for Table 1](../../../results/reference_tables/reproduced_results/differ_full_native_frequency.csv), [other baseline results and DiffER original-slot diagnostic](../../../results/reference_tables/reproduced_results/baselines.csv), [all ten comparison methods](../../../docs/guides/baselines.md), [preprocessing file lists for three methods](baseline_preprocessing_manifests/README.md) |
| Inputs and weights | Use the fixed partitions and completion inputs with the selected parameters for the corresponding class mode. | [full reactions](../../../datasets/full), [small reactions](../../../datasets/small), [SpectraRetro experiment inputs](../../spectraretro_inputs/README.md), [known-class completion weights](known-class/checkpoints/completion_model.pt), [unknown-class completion weights](unknown-class/checkpoints/completion_model.pt), [parameter roles and identities](models/README.md) |

The ranking manifest identifies the features and fixed parameters SpectraRetro uses to order assembled candidates for the full-data Top-k results. [Ranking manifest](../../../src/spectraretro/fusion_reranker/manifest.json).

[Complete file list, sources and SHA-256 digests](artifacts.json).

## Reproduction

Run from the repository root. The recount reads saved results. Each model command runs the full pretrained benchmark in the specified class setting.

```bash
python scripts/reproduce_results.py --check
python src/spectraretro/unknown-class/run.py
python src/spectraretro/known-class/run.py
```
