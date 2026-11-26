# Six single-step retrosynthesis examples

[Figure 3c,d](../4_5_error_analysis/figures/figure_3_case_studies_combined.png) · [Figure S3b–e](../../supporting_information/s5_agent_cases/figures/figure_s3_case_studies_additional.png)

Figure 3c,d presents benzofuran acylation and direct arylation of a tetrahydroindolizine derivative. Figure S3b–e presents four additional single-step cases. For each product, the case records show the center predicted from its molecular graph and the rank of the precursor candidate obtained by completing the case synthons.

| Paper item or input | What it contains | Files |
|---|---|---|
| Single-step cases | Use the case IDs and atom indices to connect the illustrated transformations with their recorded positions. | [case inputs and positions](data/six_case_inputs_and_positions.json), [Figure 3c,d](../4_5_error_analysis/figures/figure_3_case_studies_combined.png), [Figure S3b–e](../../supporting_information/s5_agent_cases/figures/figure_s3_case_studies_additional.png) |
| Candidate evidence | Individual JSON records supply candidate predictions. The script index identifies the implementations associated with single-step records 02–05. | [case records](../../../results/cases/single_step), [candidate record](../../../results/cases/single_step/record_07_candidate_evidence.json), [script index for records 02–05](../../../results/reproduction/single_multistep/source_map.json), [case parameter records](models/README.md) |

[Complete file list, sources and SHA-256 digests](artifacts.json).

[Product prediction backend](../../../src/agentic_retrosynthesis/product_backend.py) and [case-specific scripts](../../../src/experiments/single_multistep_cases).
