# Case studies

| Display | Input and prediction records |
|---|---|
| Figures 3a–b and S3a, Table S16 · incorrect predictions | [Structures](../../datasets/supplementary_data/figure8_case_structures.csv), [prediction records](incorrect_predictions) |
| Figures 3c–d and S3b–e · six single-step examples | [Verified case identities and candidate positions](../../datasets/supplementary_data/six_successful_cases_verification.json), [linked original files](single_step) |
| Figures 3e–f and S3f, Table S17 · two-step examples | [Per-node evidence](../../datasets/supplementary_data/two_step_figure_evidence.json), [linked files](multistep) |
| Figure 5 and Tables S14–S15 · CHRetroAgent pharmaceutical routes | [Inputs](../../datasets/supplementary_data/agent_case_inputs.csv), [steps](../../datasets/supplementary_data/agent_route_steps.csv), [assessments](../../datasets/supplementary_data/agent_route_results.csv), [terminals](../../datasets/supplementary_data/agent_terminal_inventory.csv), [route resources](agent) |

Use the stated input conditions and candidate provenance when interpreting case ranks. Recorded LLM assessments are observations from the specified service configuration. A new request may return a different assessment.

For the three pharmaceutical targets in Figure 5, the [CHRetroAgent guide](../../CHRetroAgent/README.md) links each saved route to its assessment and shows how to export a molecular diagram. Shared generation banks and input mappings retain the complete input batch and its original ordering.
