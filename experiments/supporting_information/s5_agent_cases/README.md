# S5. Route settings, assessments and case records

[Figure S3](figures/figure_s3_case_studies_additional.png) · [Table S14](tables/table_s14.tex) · [Table S15](tables/table_s15.tex) · [Table S16](tables/table_s16.tex) · [Table S17](tables/table_s17.tex)

Tables S14–S17 describe the route-search settings, recorded whole-route assessments, structural errors and two-step examples. The pharmaceutical route search uses the supplied candidate pools and terminal inventories.

| Paper item or input | What it contains | Files |
|---|---|---|
| Table S14 | Search limits and terminal criteria govern the pharmaceutical case evaluation. | [search protocol](../../../results/cases/agent/protocol.json), [terminal inventory](../../../datasets/supplementary_data/agent_terminal_inventory.csv), [targets](../../../datasets/supplementary_data/agent_case_inputs.csv), [candidate pools](../../../results/cases/agent/candidate_pools.json) |
| Table S15 | Per-route scores, step records and the assessment model used in the case studies. | [assessment results](../../../datasets/supplementary_data/agent_route_results.csv), [assessment manifest](../../../results/cases/agent_assessment/manifest.json), [route steps](../../../datasets/supplementary_data/agent_route_steps.csv) |
| Table S16 | Recorded target and predicted structures for the errors illustrated in Figures 3a,b and S3a. | [structure table](../../../datasets/supplementary_data/figure8_case_structures.csv), [per-case records](../../../results/cases/incorrect_predictions) |
| Table S17 | Reaction-center and precursor-set positions refer to individual nodes in the three two-step examples. | [node positions](../../../datasets/supplementary_data/two_step_figure_evidence.json), [node records](../../../results/cases/multistep) |

Section S5.4 also describes the [six single-step examples](../../section_4_results/4_6_single_step_cases/) and [three two-step examples](../../section_4_results/4_7_multistep_cases/).

<details>
<summary>Additional source records</summary>

- [`results/cases/agent`](../../../results/cases/agent)
- [`results/cases/agent_assessment`](../../../results/cases/agent_assessment)
- [`results/cases/agent_generation`](../../../results/cases/agent_generation)
- [`results/cases/agent_generation/whole_routes/requests.json`](../../../results/cases/agent_generation/whole_routes/requests.json)

</details>

[Complete file list, sources and SHA-256 digests](artifacts.json). [Model files and selection details](models/README.md).

[Agent implementation](../../../src/agentic_retrosynthesis) and [route assessment](../../../CHRetroAgent/webapp/route_assessment.py).
