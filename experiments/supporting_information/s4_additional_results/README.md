# S4. Diagnostics, ablations and additional comparisons

[Figure S1](figures/figure_s1_full_spectrum_family_gains.png) · [Figure S2](figures/figure_s2_chemical_prior_structural_match.png) · [Table S5](tables/table_s5.tex) · [Table S6](tables/table_s6.tex) · [Table S7](tables/table_s7.tex) · [Table S8](tables/table_s8.tex) · [Table S9](tables/table_s9.tex) · [Table S10](tables/table_s10.tex) · [Table S12](tables/table_s12.tex) · [Table S13](tables/table_s13.tex)

Figures S1–S2 and Tables S5–S10 and S12–S13 cover encoder diagnostics, reaction-center recovery, precursor recovery and mechanistic measurements. Use the data source for the outcome named in each table.

| Paper item or input | What it contains | Files |
|---|---|---|
| Table S5 | Functional and numerical encoder checks. | [diagnostic values](../../../results/diagnostics/spectral_encoder/recorded_diagnostics.json), [model input definitions](../../../results/diagnostics/spectral_encoder/trained_input_contract.json) |
| Tables S6–S7 | S6 includes center and precursor ablations. S7 stratifies center Top-1 results by local similarity. | [center scores](../../../results/reference_tables/reproduced_stage1/reaction_center_results.csv), [chemical-prior validation recount](../../../results/supporting_information/source_data_agent/prior_ablation_recomputed.csv), [full-test center pairs](../../../datasets/supplementary_data/figure6_paired_reaction_results.csv) |
| Table S8 | Mean heavy-atom bond-graph match rates with and without the chemical prior, and counts of reactions with increased or decreased match rates in Figure S2. | [reported summary statistics](../../../results/supporting_information/source_data_figures/reported_figure7_summary.csv) |
| Tables S9–S10 | S9 gives full-test precursor recovery. S10 reports small-set Top-k accuracy over three seeds. | [full results](../../../results/reference_tables/reproduced_results/main_full.csv), [small-set aggregates](../../../results/reference_tables/reproduced_results/main_small_summary.csv) |
| Tables S12–S13 | Matched-bond chemical concordances and electronic contrasts for the primary pairs. | [chemical associations](results/chemical_preferences.csv), [DFT contrasts](data/dft_paired_electronic_comparison.csv) |

<details>
<summary>Additional source records</summary>

- [`results/reference_tables/reproduced_results/main_small.csv`](../../../results/reference_tables/reproduced_results/main_small.csv)
- [`results/reference_tables/reproduced_stage1/small_stratified_results.csv`](../../../results/reference_tables/reproduced_stage1/small_stratified_results.csv)
- [`results/supporting_information/source_data_agent`](../../../results/supporting_information/source_data_agent)

</details>

[Complete file list, sources and SHA-256 digests](artifacts.json).

[Center ranks](../../../results/reaction_center/reproduce.py), [prior and geometry summaries](../../../scripts/reproduce_supporting_analyses.py), [precursor and baseline recount](../../../scripts/reproduce_results.py) and [electronic contrasts](../../../scripts/reproduce_mechanistic_tables.py).
