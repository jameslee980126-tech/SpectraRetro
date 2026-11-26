# CHRetroAgent pharmaceutical routes

[Figure 5](figures/figure_5_chretroagent_case_studies.png) · [SI §S8, CHRetroAgent interface](../../supporting_information/s8_agent_interface_routes/README.md)

Figure 5 shows routes for lapatinib (a), nemiralisib (b) and tafamidis (c). CHRetroAgent expands each intermediate using the case-specific list of precursor candidates. The product of the fragment frequencies determines the order of those candidates. A branch closes when it reaches the literature-defined starting-material inventory. The arrow labels give single-step candidate ranks. Table S15 reports the assessments of complete routes.

| Paper item or input | What it contains | Files |
|---|---|---|
| Figure S7 | Route visualization, assistant responses and supplier information. | [CHRetroAgent interface](../../supporting_information/s8_agent_interface_routes/figures/figure_s7_chretroagent_interface.png) |
| Figure 5, search | Route requests, terminal molecules and candidate pools determine which route branches can close. | [requests](../../../results/cases/agent_generation/whole_routes/requests.json), [terminal inventory](../../../datasets/supplementary_data/agent_terminal_inventory.csv), [candidate pools](../../../results/cases/agent/candidate_pools.json), [search protocol](../../../results/cases/agent/protocol.json) |
| Figure 5, routes and assessments | Per-route records give the step count, assessment scores and model responses. | [route steps](../../../datasets/supplementary_data/agent_route_steps.csv), [route-level scores](../../../datasets/supplementary_data/agent_route_results.csv), [assessment manifest](../../../results/cases/agent_assessment/manifest.json), [illustrated routes](figures/figure_5_chretroagent_case_studies.png) |

<details>
<summary>Additional source records</summary>

- [`datasets/supplementary_data/agent_case_inputs.csv`](../../../datasets/supplementary_data/agent_case_inputs.csv)
- [`results/cases/agent`](../../../results/cases/agent)
- [`results/cases/agent_assessment`](../../../results/cases/agent_assessment)
- [`results/cases/agent_generation`](../../../results/cases/agent_generation)

</details>

[Complete file list, sources and SHA-256 digests](artifacts.json). [Model files and selection details](models/README.md).

[CHRetroAgent interface](../../../CHRetroAgent/README.md), [route planner](../../../src/agentic_retrosynthesis) and [assessment implementation](../../../datasets/supplementary_data/route_assessment.py).
