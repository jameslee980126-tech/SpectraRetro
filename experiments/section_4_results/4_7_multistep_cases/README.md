# Two-step conditional-completion examples

[Figure 3e,f](../4_5_error_analysis/figures/figure_3_case_studies_combined.png) · [Figure S3f](../../supporting_information/s5_agent_cases/figures/figure_s3_case_studies_additional.png)

Figure 3e,f traces predictions for an arylpyrrole carboxamide (8988) and a morpholinophenyl acylfuran (9773). Figure S3f traces a benzyloxy-substituted bromofluorobenzaldehyde (10435). Each intermediate serves as the product for a second prediction. Center probabilities and ranks refer to the product at that node. The precursor-pair positions are local to the conditional completion at the same node.

| Paper item or input | What it contains | Files |
|---|---|---|
| Case 10435 | Target and benzyl ether intermediate. | [target](../../../results/cases/multistep/10435_root.json), [intermediate](../../../results/cases/multistep/10435_benzylether.json) |
| Case 8988 | Target and first intermediate. | [target](../../../results/cases/multistep/8988_root.json), [intermediate](../../../results/cases/multistep/8988_child1.json) |
| Case 9773 | Target and first intermediate. | [target](../../../results/cases/multistep/9773_root.json), [intermediate](../../../results/cases/multistep/9773_child1.json) |
| Figures 3e,f and S3f | The case input table associates the displayed route, selected bond and precursor position with each step. Each node retains its recorded size condition. | [inputs and positions](data/two_step_case_inputs_and_positions.json), [main-text routes](../4_5_error_analysis/figures/figure_3_case_studies_combined.png), [supplementary route](../../supporting_information/s5_agent_cases/figures/figure_s3_case_studies_additional.png), [case parameters](models/README.md) |

<details>
<summary>Additional source records</summary>

- [`results/reproduction/single_multistep/source_map.json`](../../../results/reproduction/single_multistep/source_map.json)

</details>

[Complete file list, sources and SHA-256 digests](artifacts.json).

[Route-planning code](../../../src/agentic_retrosynthesis) and [case-specific scripts](../../../src/experiments/single_multistep_cases).
