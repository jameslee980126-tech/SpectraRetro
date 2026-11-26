# Reaction-center and chemical-prior ablations

[Table 3](tables/table_3.tex) · [Figure S1](../../supporting_information/s4_additional_results/figures/figure_s1_full_spectrum_family_gains.png) · [Figure S2](../../supporting_information/s4_additional_results/figures/figure_s2_chemical_prior_structural_match.png)

Table 3 compares model variants in the reaction-class-unknown setting. Reaction-center ranking, complete-precursor recovery and heavy-atom graph matching measure different outcomes.

| Paper item or input | What it contains | Files |
|---|---|---|
| Table 3, reaction centers | Compare full-spectrum learning to the recorded control models using saved candidate ranks. | [center scores](../../../results/reference_tables/reproduced_stage1/reaction_center_results.csv), [full-data paired predictions](../../../results/reaction_center/full_spectral_ablation/test/per_reaction.csv), [small-data seed summaries](../../../results/reaction_center/small_local_propagation/test/per_seed.csv) |
| Table 3, complete precursors | On the small test set, Table 3 reports Top-1 precursor recovery of 43.17% without full-spectrum learning and 60.13% with full-spectrum learning. | [Top-1, Top-3 and Top-5 results](tables/table_3.tex) |
| Table 3, chemical prior | Compare complete-precursor recovery across the paired chemical-prior arms evaluated on 2,400 reused validation reactions. | [control](../../../results/reproduction/chemical_prior/control/reaction_metrics.json), [compatibility prior](../../../results/reproduction/chemical_prior/compatibility/reaction_metrics.json), [recounted results](../../../results/supporting_information/source_data_agent/prior_ablation_recomputed.csv), [control parameters](../../../checkpoints/ablations/chemical_prior/control.pt), [compatibility parameters](../../../checkpoints/ablations/chemical_prior/compatibility.pt) |
| Figure S1 | Family and similarity strata summarize the paired change in center Top-1 accuracy. | [per-reaction comparisons](../../../results/supporting_information/source_data_figures/figure6_paired_reaction_results.csv), [stratum summary](../../../results/supporting_information/source_data_figures/figure6_family_similarity_summary.csv), [similarity thresholds](../../../results/supporting_information/source_data_figures/figure6_similarity_thresholds.json) |
| Figure S2 | Heavy-atom bond-graph recovery across 1,614 validation reactions requiring fragment generation, with 50 draws per reaction. Adding the chemical prior increased the match rate for 777 reactions, decreased it for 251 and left it unchanged for 586. | [Figure S2](../../supporting_information/s4_additional_results/figures/figure_s2_chemical_prior_structural_match.png), [reported summary statistics](../../../results/supporting_information/source_data_figures/reported_figure7_summary.csv) |

<details>
<summary>Additional source records</summary>

- [`results/reaction_center/full_spectral_ablation/configuration.json`](../../../results/reaction_center/full_spectral_ablation/configuration.json)
- [`results/reaction_center/small_local_propagation/configuration.json`](../../../results/reaction_center/small_local_propagation/configuration.json)
- [`results/reference_tables/reproduced_stage1/small_stratified_results.csv`](../../../results/reference_tables/reproduced_stage1/small_stratified_results.csv)
- [`results/reproduction/chemical_prior/protocol.json`](../../../results/reproduction/chemical_prior/protocol.json)
- [`results/reproduction/chemical_prior/compatibility/metrics.json`](../../../results/reproduction/chemical_prior/compatibility/metrics.json)
- [`results/reproduction/chemical_prior/control/metrics.json`](../../../results/reproduction/chemical_prior/control/metrics.json)
- [`results/supporting_information/source_data_figures/figure6_original_summary.json`](../../../results/supporting_information/source_data_figures/figure6_original_summary.json)

</details>

[Complete file list, sources and SHA-256 digests](artifacts.json). [Model files and selection details](models/README.md).

[Center rank evaluation](../../../results/reaction_center/reproduce.py) and [chemical-prior analysis](../../../scripts/reproduce_supporting_analyses.py).

## Reproduce

Recount the reaction-center ranks and the small-data seed summaries.

```bash
python results/reaction_center/reproduce.py --check
```

Recount the saved chemical-prior validation outcomes and check the accompanying data summaries.

```bash
python scripts/reproduce_supporting_analyses.py
```

Redraw Figure S1 from the paired center ranks for 9,250 test reactions. Each cell shows the Top-1 gain in percentage points and its reaction count, checked against the published summary. With NumPy, pandas, SciPy and Matplotlib installed, run:

```bash
python scripts/plot_paper_figure.py --figure S1
```

The plot is written to `outputs/figures/figure_S1.png`.
